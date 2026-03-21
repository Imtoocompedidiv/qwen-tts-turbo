"""TTSEngine — encapsulates model loading, caching, and generation."""

import math as _math
import os
import threading
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from faster_qwen3_tts import FasterQwen3TTS
from faster_qwen3_tts.sampling import sample_logits

# Monkey-patch to use low_cpu_mem_usage (10s faster model load)
import qwen_tts

_orig_from_pretrained = qwen_tts.Qwen3TTSModel.from_pretrained


@classmethod
def _fast_from_pretrained(cls, *args, **kwargs):
    kwargs.setdefault("low_cpu_mem_usage", True)
    return _orig_from_pretrained.__func__(cls, *args, **kwargs)


qwen_tts.Qwen3TTSModel.from_pretrained = _fast_from_pretrained


# ── Presets: voices x languages x tones ──────────────────────────────
TONE_PRESETS = [
    "", "Parle d'un ton chaleureux et professionnel",
    "Voix douce et rassurante", "Ton dynamique et enthousiaste",
    "Parle calmement avec empathie", "Ton sérieux et formel",
    "Voix joyeuse et souriante", "Parle avec autorité et confiance",
]


class TTSEngine:
    """Encapsulates model, megakernels, KV caches, and generation logic.

    All mutable state is instance attributes — no module globals.
    """

    def __init__(self):
        self.t0 = time.time()

        # ── Start cache loading from disk IMMEDIATELY (overlaps with everything) ──
        self.CACHE_PATH = "/workspace/prefill_cache.pt"
        self._cache_cpu = [None]

        def _load_cache_bg():
            if os.path.exists(self.CACHE_PATH):
                self._cache_cpu[0] = torch.load(
                    self.CACHE_PATH, map_location="cpu", weights_only=False
                )

        self._cache_thread = threading.Thread(target=_load_cache_bg, daemon=True)
        self._cache_thread.start()

        # ── GPU optimizations ───────────────────────────────────────────────
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

        # ── Config ──────────────────────────────────────────────────────────
        self.MODEL_SIZE = os.environ.get("MODEL_SIZE", "1.7B")
        self.CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1"))
        self.USE_CACHE = os.environ.get("USE_CACHE", "1") == "1"

        self.model_path = f"/workspace/models/Qwen3-TTS-12Hz-{self.MODEL_SIZE}-CustomVoice"
        if not os.path.exists(self.model_path):
            self.model_path = f"Qwen/Qwen3-TTS-12Hz-{self.MODEL_SIZE}-CustomVoice"

        print(f"Loading {self.MODEL_SIZE} from {self.model_path}...")

        self.model = FasterQwen3TTS.from_pretrained(self.model_path)
        self.inner = self.model.model.model

        # Reduce graph warmup to 1 iteration
        self.model.predictor_graph._orig_capture = self.model.predictor_graph.capture
        self.model.talker_graph._orig_capture = self.model.talker_graph.capture

        def _fast_pred(num_warmup=1):
            self.model.predictor_graph._orig_capture(num_warmup=1)

        def _fast_talk(prefill_len=10, num_warmup=1):
            self.model.talker_graph._orig_capture(
                prefill_len=prefill_len, num_warmup=1
            )

        self.model.predictor_graph.capture = _fast_pred
        self.model.talker_graph.capture = _fast_talk

        print("CUDA graph capture...")
        self.model._warmup(10)

        # ── GPU capability check ────────────────────────────────────────────
        self._gpu_cc = torch.cuda.get_device_capability()
        self.gpu_arch = self._gpu_cc[0] * 10 + self._gpu_cc[1]
        self.gpu_name = torch.cuda.get_device_name(0)

        if self.gpu_arch < 90 and os.environ.get("USE_MEGAKERNEL", "0") == "1":
            print(
                f"  GPU {self.gpu_name} (sm_{self.gpu_arch}) < sm_90: disabling megakernels"
            )
            os.environ["USE_MEGAKERNEL"] = "0"
            os.environ["USE_TALKER_MK"] = "0"

        # ── Megakernel predictor (optional, ~1.9x faster) ──────────────────
        self.mk_predictor = None
        self._setup_megakernel_predictor()

        # ── Talker megakernel (optional, ~3.25x faster) ─────────────────────
        self.mk_talker = None
        self._TALKER_MAX_SEQ = 2048
        self._setup_talker_megakernel()

        # ── Cached references ───────────────────────────────────────────────
        self.cached_suppress_mask = None
        self.cached_eos_id = None
        self.cached_talker = None
        self.cached_talker_codec_embed = None
        self.cached_talker_codec_head = None
        self.cached_predictor_embeds = None
        self.cached_num_code_groups = None
        self.cached_rope_deltas = None
        self.cached_suppress_list = None
        self.cached_tts_eos_embed = None
        self.device = None

        self.CACHE_VOICES = os.environ.get(
            "CACHE_VOICES", "Vivian,Serena,Dylan,Eric,Ryan,Aiden"
        ).split(",")
        self.CACHE_LANGUAGES = os.environ.get(
            "CACHE_LANGUAGES",
            "French,English,Chinese,Japanese,Korean,German,Russian,Portuguese,Spanish,Italian",
        ).split(",")
        self.CACHE_TONES = (
            os.environ.get("CACHE_TONES", ",".join(TONE_PRESETS)).split("|")
            if os.environ.get("CACHE_TONES")
            else TONE_PRESETS
        )

        self.prefill_cache = {}

        if self.USE_CACHE:
            self._build_cached_references()

        # ── TTH LRU cache ──────────────────────────────────────────────────
        self._tth_cache = OrderedDict()
        self._TTH_CACHE_MAX = 200
        self._tth_stream = torch.cuda.Stream()

        # ── Last cache key tracking ─────────────────────────────────────────
        self._last_cache_key = [None]

        # ── Async custom tone builder ────────────────────────────────────────
        self._tone_build_pending = set()
        self._tone_build_stream = torch.cuda.Stream()

        # ── Prime the pipeline ──────────────────────────────────────────────
        self.cached_ttfp_ms = -1
        if self.USE_CACHE and self.prefill_cache:
            self._prime_pipeline()

        # ── Clone model management ──────────────────────────────────────────
        self._clone_model = None
        self._clone_cache = {}  # clone_id -> ref_audio path
        self._clone_prefill = {}  # clone_id -> {kv, logits, past_hidden, ...}
        self._clone_refs = {}  # Cached clone model references
        self._clone_last_key = [None]
        self._clone_tth_cache = OrderedDict()
        self._clone_tth_stream = torch.cuda.Stream()

        # Pre-download Base model in background
        self._predownload_base()

        print(f"\nModel loaded + warmed up in {time.time()-self.t0:.1f}s")
        print(f"  Cached TTFP: {self.cached_ttfp_ms:.1f}ms")

    # ────────────────────────────────────────────────────────────────────────
    # Setup helpers (called from __init__)
    # ────────────────────────────────────────────────────────────────────────

    def _setup_megakernel_predictor(self):
        USE_MEGAKERNEL = os.environ.get("USE_MEGAKERNEL", "0") == "1"
        if not USE_MEGAKERNEL:
            return
        try:
            import sys as _sys

            _sys.path.insert(
                0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
            )
            from deploy.industrial.model_tts import CodePredictorKernel

            print("Building megakernel predictor...")
            t_mk = time.time()

            talker = self.inner.talker
            cp_model = talker.code_predictor.model
            cp_state = cp_model.state_dict()
            for name, param in talker.code_predictor.named_parameters():
                if not name.startswith("model."):
                    cp_state[name] = param.data
            embed_weight = talker.model.codec_embedding.weight.data

            mk_kernel = CodePredictorKernel(
                {"code_predictor": cp_state, "embed_weight": embed_weight},
                device="cuda",
            )

            proj_w = talker.code_predictor.small_to_mtp_projection.weight.data
            proj_b = getattr(
                talker.code_predictor.small_to_mtp_projection, "bias", None
            )
            if proj_b is not None:
                proj_b = proj_b.data

            proj_embed_w = F.linear(
                embed_weight.float(),
                proj_w.float(),
                proj_b.float() if proj_b is not None else None,
            ).to(torch.bfloat16)
            proj_codec = []
            for g in range(mk_kernel.num_groups):
                pe = F.linear(
                    mk_kernel.codec_embeddings[g].float(),
                    proj_w.float(),
                    proj_b.float() if proj_b is not None else None,
                ).to(torch.bfloat16)
                proj_codec.append(pe)

            class _MKPredictor:
                def __init__(self, mk, pw, pb, proj_ew, proj_ce):
                    self.mk = mk
                    self.pw = pw.bfloat16()
                    self.pb = pb.bfloat16() if pb is not None else None
                    self.proj_ew = proj_ew
                    self.proj_ce = proj_ce

                def _proj(self, x):
                    return F.linear(
                        x.float(),
                        self.pw.float(),
                        self.pb.float() if self.pb is not None else None,
                    ).bfloat16()

                @torch.no_grad()
                def run(self, past_hidden, token_id):
                    """Replace model.predictor_graph.run(). Returns [15] codebook tokens."""
                    self.mk.reset()
                    self.mk._step_with_embed(
                        self._proj(past_hidden.squeeze(0).squeeze(0))
                    )
                    tok_buf = torch.tensor(
                        [token_id], dtype=torch.long, device="cuda"
                    )
                    self.mk._step_with_embed(
                        F.embedding(tok_buf, self.proj_ew).squeeze(0)
                    )
                    out = []
                    for g in range(self.mk.num_groups):
                        logits = F.linear(
                            self.mk._norm_out.to(torch.bfloat16).unsqueeze(0),
                            self.mk.lm_heads[g],
                        ).squeeze(0)
                        t = logits.argmax(keepdim=True).long()
                        out.append(t.squeeze())
                        if g < self.mk.num_groups - 1:
                            self.mk._step_with_embed(
                                F.embedding(t, self.proj_ce[g]).squeeze(0)
                            )
                    return torch.stack(out)

            self.mk_predictor = _MKPredictor(
                mk_kernel, proj_w, proj_b, proj_embed_w, proj_codec
            )

            # Warmup with deadlock watchdog (30s timeout)
            dummy_h = torch.randn(1, 1, 2048, dtype=torch.bfloat16, device="cuda")
            _warmup_ok = [False]

            def _warmup_predictor():
                for _ in range(3):
                    self.mk_predictor.run(dummy_h, 0)
                torch.cuda.synchronize()
                _warmup_ok[0] = True

            _wt = threading.Thread(target=_warmup_predictor, daemon=True)
            _wt.start()
            _wt.join(timeout=30)
            if not _warmup_ok[0]:
                raise RuntimeError(
                    "Predictor megakernel warmup deadlocked (30s timeout)"
                )
            print(f"  Megakernel predictor ready ({time.time()-t_mk:.1f}s)")
        except Exception as e:
            print(
                f"  Megakernel FAILED: {e} — falling back to CUDA graph predictor"
            )
            self.mk_predictor = None

    def _setup_talker_megakernel(self):
        USE_TALKER_MK = os.environ.get("USE_TALKER_MK", "0") == "1"
        if not USE_TALKER_MK:
            return
        try:
            import sys as _sys

            t_tmk = time.time()

            _sys.path.insert(
                0,
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "..", "industrial"
                ),
            )
            from build_talker_megakernel import get_talker_extension

            print("Building talker megakernel...")
            get_talker_extension()

            _TALKER_LAYERS = 28
            _TALKER_HIDDEN = 2048
            _TALKER_INTER = 6144
            _TALKER_KV_HEADS = 8
            _TALKER_HEAD_DIM = 128
            _TALKER_Q_SIZE = 2048
            _TALKER_MAX_SEQ = self._TALKER_MAX_SEQ

            talker_obj = self.inner.talker
            _tlw = []
            for i in range(_TALKER_LAYERS):
                layer = talker_obj.model.layers[i]
                _tlw.extend(
                    [
                        layer.input_layernorm.weight.data.contiguous(),
                        layer.self_attn.q_proj.weight.data.contiguous(),
                        layer.self_attn.k_proj.weight.data.contiguous(),
                        layer.self_attn.v_proj.weight.data.contiguous(),
                        layer.self_attn.q_norm.weight.data.contiguous(),
                        layer.self_attn.k_norm.weight.data.contiguous(),
                        layer.self_attn.o_proj.weight.data.contiguous(),
                        layer.post_attention_layernorm.weight.data.contiguous(),
                        layer.mlp.gate_proj.weight.data.contiguous(),
                        layer.mlp.up_proj.weight.data.contiguous(),
                        layer.mlp.down_proj.weight.data.contiguous(),
                    ]
                )
            _tlw_packed = torch.empty(len(_tlw), dtype=torch.int64, device="cuda")
            for i, w in enumerate(_tlw):
                _tlw_packed[i] = w.data_ptr()
            _tfn = talker_obj.model.norm.weight.data.contiguous()

            # M-RoPE cos/sin tables
            _ROPE_THETA = 1000000.0
            _MROPE_T = 24
            _inv_freq_t = 1.0 / (
                _ROPE_THETA
                ** (
                    torch.arange(0, _MROPE_T * 2, 2, dtype=torch.float32)
                    / _TALKER_HEAD_DIM
                )
            )
            _positions = torch.arange(_TALKER_MAX_SEQ, dtype=torch.float32)
            _cos_t = torch.ones(
                _TALKER_MAX_SEQ, _TALKER_HEAD_DIM, dtype=torch.float32
            )
            _sin_t = torch.zeros(
                _TALKER_MAX_SEQ, _TALKER_HEAD_DIM, dtype=torch.float32
            )
            for p in range(_TALKER_MAX_SEQ):
                for i in range(_MROPE_T):
                    angle = _positions[p] * _inv_freq_t[i]
                    _cos_t[p, i] = torch.cos(angle)
                    _cos_t[p, i + _TALKER_HEAD_DIM // 2] = torch.cos(angle)
                    _sin_t[p, i] = torch.sin(angle)
                    _sin_t[p, i + _TALKER_HEAD_DIM // 2] = torch.sin(angle)
            _cos_t = _cos_t.to(torch.bfloat16).to("cuda").contiguous()
            _sin_t = _sin_t.to(torch.bfloat16).to("cuda").contiguous()

            # Store for access in _MKTalker
            engine_ref = self

            class _MKTalker:
                def __init__(self):
                    dev = "cuda"
                    f32 = dict(dtype=torch.float32, device=dev)
                    bf16 = dict(dtype=torch.bfloat16, device=dev)
                    self.k_cache = torch.zeros(
                        _TALKER_LAYERS,
                        _TALKER_KV_HEADS,
                        _TALKER_MAX_SEQ,
                        _TALKER_HEAD_DIM,
                        **bf16,
                    )
                    self.v_cache = torch.zeros_like(self.k_cache)
                    self.hidden_buf = torch.empty(_TALKER_HIDDEN, **bf16)
                    self.act_buf = torch.empty(_TALKER_HIDDEN, **f32)
                    self.res_buf = torch.empty(_TALKER_HIDDEN, **f32)
                    self.q_buf = torch.empty(_TALKER_Q_SIZE, **f32)
                    self.k_buf = torch.empty(
                        _TALKER_KV_HEADS * _TALKER_HEAD_DIM, **f32
                    )
                    self.v_buf = torch.empty(
                        _TALKER_KV_HEADS * _TALKER_HEAD_DIM, **f32
                    )
                    self.attn_buf = torch.empty(_TALKER_Q_SIZE, **f32)
                    self.mlp_buf = torch.empty(_TALKER_INTER, **f32)
                    self.norm_buf = torch.empty(_TALKER_HIDDEN, **f32)
                    self.bmax_vals = torch.empty(4096, **f32)
                    self.bmax_idxs = torch.empty(
                        4096, dtype=torch.int32, device=dev
                    )
                    self.out_token = torch.empty(1, dtype=torch.int32, device=dev)
                    self.dummy_embed = torch.zeros(3072, _TALKER_HIDDEN, **bf16)
                    self.dummy_lm = torch.zeros(3072, _TALKER_HIDDEN, **bf16)
                    self.attn_scale = 1.0 / _math.sqrt(_TALKER_HEAD_DIM)
                    self._layer_weights = _tlw  # prevent GC

                def step(self, embed_bf16, position):
                    self.hidden_buf.copy_(embed_bf16.view(-1))
                    torch.ops.qwen_megakernel_talker_C.decode(
                        self.out_token,
                        -1,
                        self.dummy_embed,
                        _tlw_packed,
                        _tfn,
                        self.dummy_lm,
                        _cos_t,
                        _sin_t,
                        self.k_cache,
                        self.v_cache,
                        self.hidden_buf,
                        self.act_buf,
                        self.res_buf,
                        self.q_buf,
                        self.k_buf,
                        self.v_buf,
                        self.attn_buf,
                        self.mlp_buf,
                        self.norm_buf,
                        self.bmax_vals,
                        self.bmax_idxs,
                        _TALKER_LAYERS,
                        position,
                        _TALKER_MAX_SEQ,
                        self.attn_scale,
                    )

                def restore_kv(self, tc_kv, prefill_len):
                    self.k_cache.zero_()
                    self.v_cache.zero_()
                    for layer_idx, (k, v) in enumerate(tc_kv):
                        self.k_cache[layer_idx, :, :prefill_len, :].copy_(
                            k.squeeze(0)[:, :prefill_len, :]
                        )
                        self.v_cache[layer_idx, :, :prefill_len, :].copy_(
                            v.squeeze(0)[:, :prefill_len, :]
                        )

            self.mk_talker = _MKTalker()
            dummy_ie = torch.randn(
                1, 1, _TALKER_HIDDEN, dtype=torch.bfloat16, device="cuda"
            )
            _talker_ok = [False]

            def _warmup_talker():
                for _ in range(3):
                    self.mk_talker.step(dummy_ie, 0)
                torch.cuda.synchronize()
                _talker_ok[0] = True

            _tt = threading.Thread(target=_warmup_talker, daemon=True)
            _tt.start()
            _tt.join(timeout=30)
            if not _talker_ok[0]:
                raise RuntimeError(
                    "Talker megakernel warmup deadlocked (30s timeout)"
                )
            print(f"  Talker megakernel ready ({time.time()-t_tmk:.1f}s)")
        except Exception as e:
            print(f"  Talker megakernel FAILED: {e} — falling back to CUDA graph")
            self.mk_talker = None

    def _build_cached_references(self):
        """Build cached model references and KV prefill cache."""
        self.cached_talker = self.inner.talker
        self.cached_talker_codec_embed = self.cached_talker.get_input_embeddings()
        self.cached_talker_codec_head = self.cached_talker.codec_head
        self.cached_predictor_embeds = (
            self.cached_talker.code_predictor.get_input_embeddings()
        )

        _, _, config, _, _, _, _ = self.model._prepare_generation_custom(
            text="Test.", language="French", speaker="Vivian", instruct=""
        )
        self.cached_eos_id = config.codec_eos_token_id
        self.cached_num_code_groups = config.num_code_groups
        self.cached_rope_deltas = getattr(self.cached_talker, "rope_deltas", None)
        vocab_size = config.vocab_size
        self.device = next(self.cached_talker.parameters()).device

        self.cached_suppress_mask = torch.zeros(
            vocab_size, dtype=torch.bool, device=self.device
        )
        suppress_start = max(0, vocab_size - 1024)
        for i in range(suppress_start, vocab_size):
            if i != self.cached_eos_id:
                self.cached_suppress_mask[i] = True

        self.cached_suppress_list = [self.cached_eos_id]

        with torch.inference_mode():
            self.cached_tts_eos_embed = self.cached_talker.text_projection(
                self.cached_talker.get_text_embeddings()(
                    torch.tensor(
                        [[self.inner.config.tts_eos_token_id]],
                        device=self.device,
                        dtype=torch.long,
                    )
                )
            )

        # ── Full KV cache: (voice, language, instruct) -> prefill state ──
        self._cache_thread.join()
        raw_cache = self._cache_cpu[0]

        if raw_cache is not None:
            print(f"Moving pre-loaded KV caches to GPU...")
            t_load = time.time()
            for key_str, state in raw_cache.items():
                key = tuple(key_str.split("|||"))
                self.prefill_cache[key] = {
                    "kv": tuple(
                        tuple(t.to(self.device) for t in layer)
                        for layer in state["kv"]
                    ),
                    "logits": state["logits"].to(self.device),
                    "past_hidden": state["past_hidden"].to(self.device),
                    "gen_step": state["gen_step"],
                    "tam": state["tam"].to(self.device),
                    "tpe": state["tpe"].to(self.device),
                    "prefill_len": state["prefill_len"],
                }
            print(
                f"  Loaded {len(self.prefill_cache)} combos in {time.time()-t_load:.1f}s"
            )
        else:
            n_combos = (
                len(self.CACHE_VOICES)
                * len(self.CACHE_LANGUAGES)
                * len(self.CACHE_TONES)
            )
            print(
                f"Building KV caches: {len(self.CACHE_VOICES)} voices x "
                f"{len(self.CACHE_LANGUAGES)} langs x {len(self.CACHE_TONES)} tones = {n_combos}..."
            )
            t_build = time.time()

            for voice in self.CACHE_VOICES:
                for lang in self.CACHE_LANGUAGES:
                    for instruct_str in self.CACHE_TONES:
                        key = (voice.strip(), lang.strip(), instruct_str.strip())
                        try:
                            self.prefill_cache[key] = self._build_one(*key)
                            state = self.prefill_cache[key]
                            state["kv"] = tuple(
                                tuple(t.to(self.device) for t in layer)
                                for layer in state["kv"]
                            )
                            state["logits"] = state["logits"].to(self.device)
                            state["past_hidden"] = state["past_hidden"].to(
                                self.device
                            )
                            state["tam"] = state["tam"].to(self.device)
                            state["tpe"] = state["tpe"].to(self.device)
                        except Exception as e:
                            print(f"  SKIP {key}: {e}")

            print(
                f"  Built {len(self.prefill_cache)} combos in {time.time()-t_build:.1f}s"
            )

            print(f"  Saving to {self.CACHE_PATH}...")
            save_data = {}
            for key, state in self.prefill_cache.items():
                key_str = "|||".join(key)
                save_data[key_str] = {
                    "kv": tuple(
                        tuple(t.cpu() for t in layer) for layer in state["kv"]
                    ),
                    "logits": state["logits"].cpu(),
                    "past_hidden": state["past_hidden"].cpu(),
                    "gen_step": state["gen_step"],
                    "tam": state["tam"].cpu(),
                    "tpe": state["tpe"].cpu(),
                    "prefill_len": state["prefill_len"],
                }
            torch.save(save_data, self.CACHE_PATH)
            print(
                f"  Saved ({os.path.getsize(self.CACHE_PATH)/1024/1024:.0f}MB)"
            )

        sample_kv = next(iter(self.prefill_cache.values()))["kv"]
        kv_bytes = sum(
            t.numel() * t.element_size() for l in sample_kv for t in l
        )
        print(
            f"  {len(self.prefill_cache)} combos cached, "
            f"{kv_bytes/1024/1024:.1f}MB each, "
            f"{kv_bytes*len(self.prefill_cache)/1024/1024:.0f}MB total"
        )

    def _build_one(self, voice_name, lang_name, instruct_str):
        m, talker, cfg, tie, tam, tth_dummy, tpe = (
            self.model._prepare_generation_custom(
                text="Test.",
                language=lang_name,
                speaker=voice_name,
                instruct=instruct_str if instruct_str else None,
            )
        )
        with torch.inference_mode():
            out = talker.forward(
                inputs_embeds=tie,
                attention_mask=tam,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
                trailing_text_hidden=tth_dummy,
                tts_pad_embed=tpe,
                generation_step=None,
                past_hidden=None,
                past_key_values=None,
            )
            return {
                "kv": tuple(
                    tuple(t.clone().cpu() for t in layer)
                    for layer in out.past_key_values
                ),
                "logits": out.logits[:, -1, :].clone().cpu(),
                "past_hidden": out.past_hidden.clone().cpu(),
                "gen_step": out.generation_step,
                "tam": tam.clone().cpu(),
                "tpe": tpe.clone().cpu(),
                "prefill_len": out.past_key_values[0][0].shape[2],
            }

    @torch.inference_mode()
    def _build_one_gpu(self, voice_name, lang_name, instruct_str):
        """Like _build_one but keeps tensors on GPU (no CPU roundtrip)."""
        m, talker, cfg, tie, tam, tth_dummy, tpe = (
            self.model._prepare_generation_custom(
                text="Test.",
                language=lang_name,
                speaker=voice_name,
                instruct=instruct_str if instruct_str else None,
            )
        )
        out = talker.forward(
            inputs_embeds=tie,
            attention_mask=tam,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            trailing_text_hidden=tth_dummy,
            tts_pad_embed=tpe,
            generation_step=None,
            past_hidden=None,
            past_key_values=None,
        )
        return {
            "kv": tuple(
                tuple(t.clone() for t in layer)
                for layer in out.past_key_values
            ),
            "logits": out.logits[:, -1, :].clone(),
            "past_hidden": out.past_hidden.clone(),
            "gen_step": out.generation_step,
            "tam": tam.clone(),
            "tpe": tpe.clone(),
            "prefill_len": out.past_key_values[0][0].shape[2],
        }

    def _queue_tone_build(self, voice, language, inst):
        """Build a custom tone's KV cache in background on a separate CUDA stream."""
        cache_key = (voice, language, inst)
        if cache_key in self._tone_build_pending:
            return
        self._tone_build_pending.add(cache_key)

        def _do():
            try:
                with torch.cuda.stream(self._tone_build_stream):
                    tc = self._build_one_gpu(voice, language, inst)
                self._tone_build_stream.synchronize()
                self.prefill_cache[cache_key] = tc
            except Exception as e:
                print(f"  Background tone build failed for {cache_key}: {e}")
            finally:
                self._tone_build_pending.discard(cache_key)

        threading.Thread(target=_do, daemon=True).start()

    def _prime_pipeline(self):
        """Prime the pipeline: codec generation + speech tokenizer decode."""
        # 1. Warm codec generation path
        for _ in self.generate_cached_codec("Test."):
            break

        # 2. Warm speech tokenizer decode (vocoder) — critical for PCM TTFP.
        # Cold vocoder adds ~10ms penalty. We warm with VARIED tokens (not
        # the same dummy repeated, which only warms the vocoder's internal
        # cache for that specific input — misleading in benchmarks).
        speech_tokenizer = self.inner.speech_tokenizer
        print("  Warming up speech tokenizer decode...")
        for i in range(5):
            varied_codes = torch.randint(0, 2048, (1, 16), device=self.device)
            speech_tokenizer.decode({"audio_codes": varied_codes.unsqueeze(0)})
        # Also warm with real codec tokens from a short generation
        for cb in self.generate_cached_codec("Test."):
            speech_tokenizer.decode({"audio_codes": cb.unsqueeze(0).unsqueeze(0)})
            break
        # And warm the full PCM streaming path once (end-to-end)
        for _ in self.generate_cached_streaming("Test."):
            break
        torch.cuda.synchronize()
        print("  Speech tokenizer decode warmed up")

        # 3. Measure honest TTFP (codec raw)
        _ttfp_text = "Bienvenue, comment puis-je vous aider aujourd'hui ?"
        self._tth_cache.pop(_ttfp_text, None)
        torch.cuda.synchronize()
        t_s = time.perf_counter()
        for cb in self.generate_cached_codec(_ttfp_text):
            _ = cb.cpu()
            self.cached_ttfp_ms = (time.perf_counter() - t_s) * 1000
            break

    def _predownload_base(self):
        """Pre-download Base model in background."""

        def _do():
            base_path = self.model_path.replace("CustomVoice", "Base")
            if not os.path.exists(base_path):
                from huggingface_hub import snapshot_download

                base_id = f"Qwen/Qwen3-TTS-12Hz-{self.MODEL_SIZE}-Base"
                snapshot_download(
                    base_id, local_dir=base_path, ignore_patterns=["*.md"]
                )
                print("Base model pre-downloaded for cloning")

        threading.Thread(target=_do, daemon=True).start()

    # ────────────────────────────────────────────────────────────────────────
    # TTH computation with LRU cache
    # ────────────────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def _compute_tth(self, text):
        """Compute trailing_text_hiddens with LRU cache."""
        cached = self._tth_cache.get(text)
        if cached is not None:
            self._tth_cache.move_to_end(text)
            return cached
        input_texts = [self.model.model._build_assistant_text(text)]
        input_ids = self.model.model._tokenize_texts(input_texts)[0]
        if input_ids.shape[1] < 9:
            text_tokens = input_ids
        else:
            text_tokens = input_ids[:, 4:-5]
        tth = self.cached_talker.text_projection(
            self.cached_talker.get_text_embeddings()(text_tokens)
        )
        result = torch.cat((tth, self.cached_tts_eos_embed), dim=1)
        if len(self._tth_cache) >= self._TTH_CACHE_MAX:
            self._tth_cache.popitem(last=False)
        self._tth_cache[text] = result
        return result

    # ────────────────────────────────────────────────────────────────────────
    # Generation functions
    # ────────────────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def generate_cached_codec(
        self, text, voice="Vivian", language="French", instruct="", chunk_size=1
    ):
        """Yield raw codec token tensors (no speech_tokenizer decode).

        Selects the right KV cache based on instruct/tone.
        Skips KV restore if same combo as last request.

        Honest TTFP: text encoding (TTH) is computed eagerly on a background
        CUDA stream, overlapping with KV restore and first token sampling,
        then synced BEFORE the first yield. The first frame is fully conditioned
        on real text + voice + language + instruct parameters.
        """
        inst = instruct or ""
        cache_key = (voice, language, inst)
        tc = self.prefill_cache.get(cache_key)
        if tc is None:
            import unicodedata as _ud

            def _strip(s, _n=_ud.normalize):
                return _n("NFD", s).encode("ascii", "ignore").decode()

            stripped = _strip(inst)
            for k, v in self.prefill_cache.items():
                if (
                    k[0] == voice
                    and k[1] == language
                    and _strip(k[2]) == stripped
                ):
                    tc = v
                    cache_key = k
                    break
            if tc is None and inst:
                # Check if a background build already completed
                tc = self.prefill_cache.get(cache_key)
                if tc is None:
                    # Queue background build for next request
                    self._queue_tone_build(voice, language, inst)
                    # Use closest available cache for THIS request (0 penalty)
            if tc is None:
                tc = self.prefill_cache.get((voice, language, ""))
                cache_key = (voice, language, "")
            if tc is None:
                tc = self.prefill_cache[("Vivian", "French", "")]
                cache_key = ("Vivian", "French", "")
        tc_kv = tc["kv"]
        tc_logits = tc["logits"]
        tc_past_hidden = tc["past_hidden"]
        tc_gen_step = tc["gen_step"]
        tc_tam = tc["tam"]
        tc_tpe = tc["tpe"]
        tc_prefill_len = tc["prefill_len"]

        # ── Eager TTH: start text encoding on background CUDA stream NOW,
        # so it overlaps with KV restore + first token sampling below.
        cached_tth = self._tth_cache.get(text)
        if cached_tth is not None:
            tth_new = cached_tth
            tth_event = None
        else:
            tth_event = torch.cuda.Event()
            with torch.cuda.stream(self._tth_stream):
                tth_new = self._compute_tth(text)
                tth_event.record()

        # ── KV restore (overlaps with TTH on background stream) ──
        if self._last_cache_key[0] != cache_key:
            if self.mk_talker is not None:
                self.mk_talker.restore_kv(tc_kv, tc_prefill_len)
            else:
                for layer_idx, (k, v) in enumerate(tc_kv):
                    self.model.talker_graph.static_cache.update(k, v, layer_idx)
            self._last_cache_key[0] = cache_key
        if self.mk_talker is None:
            self.model.talker_graph.set_generation_state(
                tc_tam, self.cached_rope_deltas
            )

        token = sample_logits(
            tc_logits,
            temperature=0.9,
            top_k=50,
            top_p=1.0,
            do_sample=True,
            suppress_mask=self.cached_suppress_mask,
            suppress_tokens=self.cached_suppress_list,
        )

        past_hidden = tc_past_hidden.clone()
        gen_step = tc_gen_step

        # ── Sync TTH before first yield — honest TTFP includes text encoding ──
        if tth_event is not None:
            tth_event.synchronize()
            tth_event = None

        for step_idx in range(2048):
            if token.item() == self.cached_eos_id:
                break

            last_id_hidden = self.cached_talker_codec_embed(token.unsqueeze(1))
            if self.mk_predictor is not None:
                codebook_token_ids = self.mk_predictor.run(
                    past_hidden, token.item()
                )
            else:
                pred_input = torch.cat((past_hidden, last_id_hidden), dim=1)
                codebook_token_ids = self.model.predictor_graph.run(pred_input)
            all_cb = torch.cat([token.view(1), codebook_token_ids])

            yield all_cb

            codec_hiddens = [last_id_hidden]
            for ci in range(self.cached_num_code_groups - 1):
                codec_hiddens.append(
                    self.cached_predictor_embeds[ci](
                        codebook_token_ids[ci].unsqueeze(0).unsqueeze(0)
                    )
                )
            inputs_embeds = torch.cat(codec_hiddens, dim=1).sum(1, keepdim=True)

            if gen_step < tth_new.shape[1]:
                inputs_embeds = inputs_embeds + tth_new[:, gen_step].unsqueeze(1)
            else:
                inputs_embeds = inputs_embeds + tc_tpe

            current_pos = tc_prefill_len + step_idx
            max_pos = (
                self._TALKER_MAX_SEQ - 1
                if self.mk_talker is not None
                else self.model.talker_graph.max_seq_len - 1
            )
            if current_pos >= max_pos:
                break

            if self.mk_talker is not None:
                self.mk_talker.step(inputs_embeds, current_pos)
                norm_bf16 = self.mk_talker.norm_buf.to(torch.bfloat16)
                logits = F.linear(
                    norm_bf16.unsqueeze(0),
                    self.cached_talker_codec_head.weight,
                ).unsqueeze(0)
                past_hidden = norm_bf16.unsqueeze(0).unsqueeze(0)
            else:
                hidden_states = self.model.talker_graph.run(
                    inputs_embeds, position=current_pos
                )
                logits = self.cached_talker_codec_head(
                    hidden_states[:, -1, :]
                ).unsqueeze(0)
                past_hidden = hidden_states[:, -1:, :].clone()

            token = sample_logits(
                logits.squeeze(0),
                temperature=0.9,
                top_k=50,
                top_p=1.0,
                do_sample=True,
                suppress_mask=self.cached_suppress_mask,
            )
            gen_step += 1

    @torch.inference_mode()
    def generate_cached_streaming(
        self,
        text,
        voice="Vivian",
        language="French",
        instruct="",
        chunk_size=1,
    ):
        """Streaming generation with cached prefill KV + speech decode."""
        speech_tokenizer = self.inner.speech_tokenizer
        all_codes = []
        prev_audio_len = 0
        samples_per_frame = None
        context_frames = 25
        chunk_buffer = []

        for codec_ids in self.generate_cached_codec(
            text, voice, language, instruct, chunk_size
        ):
            chunk_buffer.append(codec_ids.detach())

            if len(chunk_buffer) >= chunk_size:
                chunk_codes = torch.stack(chunk_buffer)
                all_codes.append(chunk_codes)
                all_flat = torch.cat(all_codes, dim=0)
                n_new = chunk_codes.shape[0]
                n_total = all_flat.shape[0]

                if samples_per_frame is None:
                    audio_list, sr = speech_tokenizer.decode(
                        {"audio_codes": all_flat.unsqueeze(0)}
                    )
                    audio = audio_list[0]
                    if hasattr(audio, "cpu"):
                        audio = audio.flatten().cpu().numpy()
                    new_audio = audio[prev_audio_len:]
                    prev_audio_len = len(audio)
                    if n_total >= max(context_frames, chunk_size):
                        samples_per_frame = len(audio) / n_total
                else:
                    ctx_start = max(0, n_total - n_new - context_frames)
                    window = all_flat[ctx_start:]
                    n_ctx = window.shape[0] - n_new
                    audio_list, sr = speech_tokenizer.decode(
                        {"audio_codes": window.unsqueeze(0)}
                    )
                    audio = audio_list[0]
                    if hasattr(audio, "cpu"):
                        audio = audio.flatten().cpu().numpy()
                    if n_ctx > 0:
                        ctx_samples = int(round(n_ctx * samples_per_frame))
                        new_audio = audio[ctx_samples:]
                    else:
                        new_audio = audio

                yield new_audio, sr
                chunk_buffer = []

        if chunk_buffer:
            chunk_codes = torch.stack(chunk_buffer)
            all_codes.append(chunk_codes)
            all_flat = torch.cat(all_codes, dim=0)
            audio_list, sr = speech_tokenizer.decode(
                {"audio_codes": all_flat.unsqueeze(0)}
            )
            audio = audio_list[0]
            if hasattr(audio, "cpu"):
                audio = audio.flatten().cpu().numpy()
            new_audio = audio[prev_audio_len:]
            yield new_audio, sr

    # ────────────────────────────────────────────────────────────────────────
    # Voice clone management
    # ────────────────────────────────────────────────────────────────────────

    def _get_clone_model(self):
        """Lazy-load the Base model for voice cloning (first call only)."""
        if self._clone_model is not None:
            return self._clone_model
        print("Loading Base model for voice cloning...")
        t0 = time.time()
        base_path = self.model_path.replace("CustomVoice", "Base")
        if not os.path.exists(base_path):
            from huggingface_hub import snapshot_download

            base_id = f"Qwen/Qwen3-TTS-12Hz-{self.MODEL_SIZE}-Base"
            snapshot_download(
                base_id, local_dir=base_path, ignore_patterns=["*.md"]
            )
        self._clone_model = FasterQwen3TTS.from_pretrained(base_path)
        self._clone_model._warmup(10)

        clone_inner = self._clone_model.model.model
        clone_talker = clone_inner.talker
        self._clone_refs["talker"] = clone_talker
        self._clone_refs["codec_embed"] = clone_talker.get_input_embeddings()
        self._clone_refs["codec_head"] = clone_talker.codec_head
        self._clone_refs["pred_embeds"] = (
            clone_talker.code_predictor.get_input_embeddings()
        )
        self._clone_refs["eos_id"] = self.cached_eos_id
        self._clone_refs["num_code_groups"] = self.cached_num_code_groups
        self._clone_refs["rope_deltas"] = getattr(
            clone_talker, "rope_deltas", None
        )
        self._clone_refs["suppress_mask"] = self.cached_suppress_mask
        with torch.inference_mode():
            self._clone_refs["tts_eos_embed"] = clone_talker.text_projection(
                clone_talker.get_text_embeddings()(
                    torch.tensor(
                        [[clone_inner.config.tts_eos_token_id]],
                        device=self.device,
                        dtype=torch.long,
                    )
                )
            )
        print(f"  Base model loaded in {time.time()-t0:.1f}s")
        return self._clone_model

    def build_clone_prefill(self, clone_id, ref_audio, ref_text, language="French"):
        """Build and cache the KV prefill for a cloned voice."""
        cm = self._get_clone_model()
        result = cm._prepare_generation(
            text="Test.",
            ref_audio=ref_audio,
            ref_text=ref_text,
            language=language,
        )
        m, talker, config, tie, tam, tth_dummy, tpe = result[:7]
        with torch.inference_mode():
            out = talker.forward(
                inputs_embeds=tie,
                attention_mask=tam,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
                trailing_text_hidden=tth_dummy,
                tts_pad_embed=tpe,
                generation_step=None,
                past_hidden=None,
                past_key_values=None,
            )
            self._clone_prefill[clone_id] = {
                "kv": tuple(
                    tuple(t.clone() for t in layer)
                    for layer in out.past_key_values
                ),
                "logits": out.logits[:, -1, :].clone(),
                "past_hidden": out.past_hidden.clone(),
                "gen_step": out.generation_step,
                "tam": tam.clone(),
                "tpe": tpe.clone(),
                "prefill_len": out.past_key_values[0][0].shape[2],
            }
        # Warmup the clone codec path to match CustomVoice performance
        for _ in range(5):
            for _ in self.generate_clone_cached_codec("Test.", clone_id, language):
                break
        return self._clone_prefill[clone_id]

    @torch.inference_mode()
    def _compute_clone_tth(self, text):
        """Compute trailing_text_hiddens using the Base model's text encoder."""
        cached = self._clone_tth_cache.get(text)
        if cached is not None:
            self._clone_tth_cache.move_to_end(text)
            return cached
        cm = self._get_clone_model()
        input_texts = [cm.model._build_assistant_text(text)]
        input_ids = cm.model._tokenize_texts(input_texts)[0]
        if input_ids.shape[1] < 9:
            text_tokens = input_ids
        else:
            text_tokens = input_ids[:, 4:-5]
        tth = self._clone_refs["talker"].text_projection(
            self._clone_refs["talker"].get_text_embeddings()(text_tokens)
        )
        result = torch.cat(
            (tth, self._clone_refs["tts_eos_embed"]), dim=1
        )
        if len(self._clone_tth_cache) >= self._TTH_CACHE_MAX:
            self._clone_tth_cache.popitem(last=False)
        self._clone_tth_cache[text] = result
        return result

    @torch.inference_mode()
    def generate_clone_cached_codec(self, text, clone_id, language="French"):
        """Fast clone codec generation using cached KV (same pattern as CustomVoice).

        Eager TTH: text encoding computed before first yield (honest TTFP).
        """
        tc = self._clone_prefill.get(clone_id)
        if tc is None:
            raise ValueError(
                f"Clone '{clone_id}' not found. Send ref_audio first."
            )

        cm = self._get_clone_model()

        # ── Eager TTH on background stream (overlaps with KV restore) ──
        cached_tth = self._clone_tth_cache.get(text)
        if cached_tth is not None:
            tth_new = cached_tth
            tth_event = None
        else:
            tth_event = torch.cuda.Event()
            with torch.cuda.stream(self._clone_tth_stream):
                tth_new = self._compute_clone_tth(text)
                tth_event.record()

        if self._clone_last_key[0] != clone_id:
            for layer_idx, (k, v) in enumerate(tc["kv"]):
                cm.talker_graph.static_cache.update(k, v, layer_idx)
            cm.talker_graph.set_generation_state(
                tc["tam"], self._clone_refs["rope_deltas"]
            )
            self._clone_last_key[0] = clone_id
        else:
            cm.talker_graph.set_generation_state(
                tc["tam"], self._clone_refs["rope_deltas"]
            )

        token = sample_logits(
            tc["logits"],
            temperature=0.9,
            top_k=50,
            top_p=1.0,
            do_sample=True,
            suppress_mask=self._clone_refs["suppress_mask"],
            suppress_tokens=self.cached_suppress_list,
        )

        past_hidden = tc["past_hidden"].clone()
        gen_step = tc["gen_step"]
        eos_id = self._clone_refs["eos_id"]
        codec_embed = self._clone_refs["codec_embed"]
        codec_head = self._clone_refs["codec_head"]
        pred_embeds = self._clone_refs["pred_embeds"]
        num_cg = self._clone_refs["num_code_groups"]

        # ── Sync TTH before first yield ──
        if tth_event is not None:
            tth_event.synchronize()
            tth_event = None

        for step_idx in range(2048):
            if token.item() == eos_id:
                break
            last_id_hidden = codec_embed(token.unsqueeze(1))
            pred_input = torch.cat((past_hidden, last_id_hidden), dim=1)
            codebook_token_ids = cm.predictor_graph.run(pred_input)
            all_cb = torch.cat([token.view(1), codebook_token_ids])

            yield all_cb

            codec_hiddens = [last_id_hidden]
            for ci in range(num_cg - 1):
                codec_hiddens.append(
                    pred_embeds[ci](
                        codebook_token_ids[ci].unsqueeze(0).unsqueeze(0)
                    )
                )
            inputs_embeds = torch.cat(codec_hiddens, dim=1).sum(1, keepdim=True)
            if gen_step < tth_new.shape[1]:
                inputs_embeds = (
                    inputs_embeds + tth_new[:, gen_step].unsqueeze(1)
                )
            else:
                inputs_embeds = inputs_embeds + tc["tpe"]

            current_pos = tc["prefill_len"] + step_idx
            if current_pos >= cm.talker_graph.max_seq_len - 1:
                break
            hidden_states = cm.talker_graph.run(
                inputs_embeds, position=current_pos
            )
            logits = codec_head(hidden_states[:, -1, :]).unsqueeze(0)
            token = sample_logits(
                logits.squeeze(0),
                temperature=0.9,
                top_k=50,
                top_p=1.0,
                do_sample=True,
                suppress_mask=self._clone_refs["suppress_mask"],
            )
            past_hidden = hidden_states[:, -1:, :].clone()
            gen_step += 1
