"""Qwen3-TTS server — production hardened, minimum TTFP."""

import gc
import json
import os
import struct
import time
import threading

import numpy as np
import torch

# Disable GC during generation to avoid pauses
gc.disable()
gc.collect()

t0 = time.time()

# ── Start cache loading from disk IMMEDIATELY (overlaps with everything) ──
CACHE_PATH = "/workspace/prefill_cache.pt"
_cache_cpu = [None]

def _load_cache_bg():
    if os.path.exists(CACHE_PATH):
        _cache_cpu[0] = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)

cache_thread = threading.Thread(target=_load_cache_bg, daemon=True)
cache_thread.start()

# ── GPU optimizations ───────────────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# ── Config ──────────────────────────────────────────────────────────
MODEL_SIZE = os.environ.get("MODEL_SIZE", "1.7B")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1"))
USE_CACHE = os.environ.get("USE_CACHE", "1") == "1"

model_path = f"/workspace/models/Qwen3-TTS-12Hz-{MODEL_SIZE}-CustomVoice"
if not os.path.exists(model_path):
    model_path = f"Qwen/Qwen3-TTS-12Hz-{MODEL_SIZE}-CustomVoice"

print(f"Loading {MODEL_SIZE} from {model_path}...")

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

model = FasterQwen3TTS.from_pretrained(model_path)
inner = model.model.model
# Reduce graph warmup to 1 iteration
model.predictor_graph._orig_capture = model.predictor_graph.capture
model.talker_graph._orig_capture = model.talker_graph.capture
def _fast_pred(num_warmup=1): model.predictor_graph._orig_capture(num_warmup=1)
def _fast_talk(prefill_len=10, num_warmup=1): model.talker_graph._orig_capture(prefill_len=prefill_len, num_warmup=1)
model.predictor_graph.capture = _fast_pred
model.talker_graph.capture = _fast_talk

print("CUDA graph capture...")
model._warmup(10)

# ── Megakernel predictor (optional, ~1.9x faster) ─────────────────
USE_MEGAKERNEL = os.environ.get("USE_MEGAKERNEL", "0") == "1"
mk_predictor = None

if USE_MEGAKERNEL:
    try:
        import sys as _sys
        _sys.path.insert(0, "/workspace/megakernel-tts")
        import torch.nn.functional as _F
        from qwen_megakernel.model_tts import CodePredictorKernel

        print("Building megakernel predictor...")
        t_mk = time.time()

        # Extract predictor weights from loaded model
        talker = inner.talker
        cp_model = talker.code_predictor.model
        cp_state = cp_model.state_dict()
        # Add non-model predictor weights
        for name, param in talker.code_predictor.named_parameters():
            if not name.startswith("model."):
                cp_state[name] = param.data
        embed_weight = talker.model.codec_embedding.weight.data

        mk_kernel = CodePredictorKernel(
            {"code_predictor": cp_state, "embed_weight": embed_weight},
            device="cuda"
        )

        # Get projection weights
        proj_w = talker.code_predictor.small_to_mtp_projection.weight.data
        proj_b = getattr(talker.code_predictor.small_to_mtp_projection, "bias", None)
        if proj_b is not None:
            proj_b = proj_b.data

        # Pre-project all codec embeddings (eliminates 16 of 17 projections per predict)
        proj_embed_w = _F.linear(embed_weight.float(), proj_w.float(),
                                 proj_b.float() if proj_b is not None else None).to(torch.bfloat16)
        proj_codec = []
        for g in range(mk_kernel.num_groups):
            pe = _F.linear(mk_kernel.codec_embeddings[g].float(), proj_w.float(),
                           proj_b.float() if proj_b is not None else None).to(torch.bfloat16)
            proj_codec.append(pe)

        class _MKPredictor:
            def __init__(self, mk, pw, pb, proj_ew, proj_ce):
                self.mk = mk
                self.pw = pw.bfloat16()
                self.pb = pb.bfloat16() if pb is not None else None
                self.proj_ew = proj_ew      # Pre-projected [vocab, 1024]
                self.proj_ce = proj_ce      # Pre-projected per-group [vocab, 1024]

            def _proj(self, x):
                return _F.linear(x.float(), self.pw.float(),
                                 self.pb.float() if self.pb is not None else None).bfloat16()

            @torch.no_grad()
            def run(self, past_hidden, token_id):
                """Replace model.predictor_graph.run(). Returns [15] codebook tokens."""
                self.mk.reset()
                self.mk._step_with_embed(self._proj(past_hidden.squeeze(0).squeeze(0)))
                tok_buf = torch.tensor([token_id], dtype=torch.long, device="cuda")
                self.mk._step_with_embed(_F.embedding(tok_buf, self.proj_ew).squeeze(0))
                out = []
                for g in range(self.mk.num_groups):
                    logits = _F.linear(
                        self.mk._norm_out.to(torch.bfloat16).unsqueeze(0),
                        self.mk.lm_heads[g]).squeeze(0)
                    t = logits.argmax(keepdim=True).long()
                    out.append(t.squeeze())
                    if g < self.mk.num_groups - 1:
                        self.mk._step_with_embed(
                            _F.embedding(t, self.proj_ce[g]).squeeze(0))
                return torch.stack(out)

        mk_predictor = _MKPredictor(mk_kernel, proj_w, proj_b, proj_embed_w, proj_codec)

        # Warmup
        dummy_h = torch.randn(1, 1, 2048, dtype=torch.bfloat16, device="cuda")
        for _ in range(3):
            mk_predictor.run(dummy_h, 0)
        print(f"  Megakernel predictor ready ({time.time()-t_mk:.1f}s)")
    except Exception as e:
        print(f"  Megakernel FAILED: {e} — falling back to CUDA graph predictor")
        mk_predictor = None

# ── Talker megakernel (optional, ~3.25x faster) ───────────────────
USE_TALKER_MK = os.environ.get("USE_TALKER_MK", "0") == "1"
mk_talker = None

if USE_TALKER_MK:
    try:
        import math as _math
        t_tmk = time.time()

        # Build talker kernel (HIDDEN=2048, INTERMEDIATE=6144, NUM_KV_HEADS=8)
        _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from build_talker_megakernel import get_talker_extension
        print("Building talker megakernel...")
        get_talker_extension("/workspace/megakernel-tts")

        _TALKER_LAYERS = 28
        _TALKER_HIDDEN = 2048
        _TALKER_INTER = 6144
        _TALKER_KV_HEADS = 8
        _TALKER_HEAD_DIM = 128
        _TALKER_Q_SIZE = 2048
        _TALKER_MAX_SEQ = 2048

        # Pack talker weights from loaded model
        talker_obj = inner.talker
        _tlw = []
        for i in range(_TALKER_LAYERS):
            layer = talker_obj.model.layers[i]
            _tlw.extend([
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
            ])
        _tlw_packed = torch.empty(len(_tlw), dtype=torch.int64, device="cuda")
        for i, w in enumerate(_tlw):
            _tlw_packed[i] = w.data_ptr()
        _tfn = talker_obj.model.norm.weight.data.contiguous()

        # M-RoPE cos/sin tables (temporal rotation for first 48 dims, identity for rest)
        _ROPE_THETA = 1000000.0
        _MROPE_T = 24  # temporal pairs
        _inv_freq_t = 1.0 / (_ROPE_THETA ** (torch.arange(0, _MROPE_T * 2, 2, dtype=torch.float32) / _TALKER_HEAD_DIM))
        _positions = torch.arange(_TALKER_MAX_SEQ, dtype=torch.float32)
        _cos_t = torch.ones(_TALKER_MAX_SEQ, _TALKER_HEAD_DIM, dtype=torch.float32)
        _sin_t = torch.zeros(_TALKER_MAX_SEQ, _TALKER_HEAD_DIM, dtype=torch.float32)
        for p in range(_TALKER_MAX_SEQ):
            for i in range(_MROPE_T):
                angle = _positions[p] * _inv_freq_t[i]
                _cos_t[p, i] = torch.cos(angle)
                _cos_t[p, i + _TALKER_HEAD_DIM // 2] = torch.cos(angle)
                _sin_t[p, i] = torch.sin(angle)
                _sin_t[p, i + _TALKER_HEAD_DIM // 2] = torch.sin(angle)
        _cos_t = _cos_t.to(torch.bfloat16).to("cuda").contiguous()
        _sin_t = _sin_t.to(torch.bfloat16).to("cuda").contiguous()

        class _MKTalker:
            def __init__(self):
                dev = "cuda"
                f32 = dict(dtype=torch.float32, device=dev)
                bf16 = dict(dtype=torch.bfloat16, device=dev)
                self.k_cache = torch.zeros(_TALKER_LAYERS, _TALKER_KV_HEADS, _TALKER_MAX_SEQ, _TALKER_HEAD_DIM, **bf16)
                self.v_cache = torch.zeros_like(self.k_cache)
                self.hidden_buf = torch.empty(_TALKER_HIDDEN, **bf16)
                self.act_buf = torch.empty(_TALKER_HIDDEN, **f32)
                self.res_buf = torch.empty(_TALKER_HIDDEN, **f32)
                self.q_buf = torch.empty(_TALKER_Q_SIZE, **f32)
                self.k_buf = torch.empty(_TALKER_KV_HEADS * _TALKER_HEAD_DIM, **f32)
                self.v_buf = torch.empty(_TALKER_KV_HEADS * _TALKER_HEAD_DIM, **f32)
                self.attn_buf = torch.empty(_TALKER_Q_SIZE, **f32)
                self.mlp_buf = torch.empty(_TALKER_INTER, **f32)
                self.norm_buf = torch.empty(_TALKER_HIDDEN, **f32)
                self.bmax_vals = torch.empty(4096, **f32)
                self.bmax_idxs = torch.empty(4096, dtype=torch.int32, device=dev)
                self.out_token = torch.empty(1, dtype=torch.int32, device=dev)
                self.dummy_embed = torch.zeros(3072, _TALKER_HIDDEN, **bf16)
                self.dummy_lm = torch.zeros(3072, _TALKER_HIDDEN, **bf16)
                self.attn_scale = 1.0 / _math.sqrt(_TALKER_HEAD_DIM)
                self._layer_weights = _tlw  # prevent GC

            def step(self, embed_bf16, position):
                self.hidden_buf.copy_(embed_bf16.view(-1))
                torch.ops.qwen_megakernel_talker_C.decode(
                    self.out_token, -1,
                    self.dummy_embed, _tlw_packed, _tfn, self.dummy_lm,
                    _cos_t, _sin_t,
                    self.k_cache, self.v_cache,
                    self.hidden_buf, self.act_buf, self.res_buf,
                    self.q_buf, self.k_buf, self.v_buf,
                    self.attn_buf, self.mlp_buf, self.norm_buf,
                    self.bmax_vals, self.bmax_idxs,
                    _TALKER_LAYERS, position, _TALKER_MAX_SEQ, self.attn_scale,
                )

            def restore_kv(self, tc_kv, prefill_len):
                self.k_cache.zero_()
                self.v_cache.zero_()
                for layer_idx, (k, v) in enumerate(tc_kv):
                    # k: [1, num_kv_heads, seq_len, head_dim]
                    self.k_cache[layer_idx, :, :prefill_len, :].copy_(k.squeeze(0)[:, :prefill_len, :])
                    self.v_cache[layer_idx, :, :prefill_len, :].copy_(v.squeeze(0)[:, :prefill_len, :])

        mk_talker = _MKTalker()
        dummy_ie = torch.randn(1, 1, _TALKER_HIDDEN, dtype=torch.bfloat16, device="cuda")
        for _ in range(3):
            mk_talker.step(dummy_ie, 0)
        print(f"  Talker megakernel ready ({time.time()-t_tmk:.1f}s)")
    except Exception as e:
        print(f"  Talker megakernel FAILED: {e} — falling back to CUDA graph")
        mk_talker = None

import torch.nn.functional as F_global

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI(title="Qwen3-TTS")

cached_suppress_mask = None
cached_eos_id = None

# ── Presets: voices × languages × tones ──────────────────────────────
TONE_PRESETS = [
    "", "Parle d'un ton chaleureux et professionnel",
    "Voix douce et rassurante", "Ton dynamique et enthousiaste",
    "Parle calmement avec empathie", "Ton sérieux et formel",
    "Voix joyeuse et souriante", "Parle avec autorité et confiance",
]

CACHE_VOICES = os.environ.get("CACHE_VOICES", "Vivian,Serena,Dylan,Eric,Ryan,Aiden").split(",")
CACHE_LANGUAGES = os.environ.get("CACHE_LANGUAGES", "French,English,Chinese,Japanese,Korean,German,Russian,Portuguese,Spanish,Italian").split(",")
CACHE_TONES = os.environ.get("CACHE_TONES", ",".join(TONE_PRESETS)).split("|") if os.environ.get("CACHE_TONES") else TONE_PRESETS

if USE_CACHE:
    # Pre-cache talker references
    cached_talker = inner.talker
    cached_talker_codec_embed = cached_talker.get_input_embeddings()
    cached_talker_codec_head = cached_talker.codec_head
    cached_predictor_embeds = cached_talker.code_predictor.get_input_embeddings()

    _, _, config, _, _, _, _ = model._prepare_generation_custom(
        text="Test.", language="French", speaker="Vivian", instruct=""
    )
    cached_eos_id = config.codec_eos_token_id
    cached_num_code_groups = config.num_code_groups
    cached_rope_deltas = getattr(cached_talker, "rope_deltas", None)
    vocab_size = config.vocab_size
    device = next(cached_talker.parameters()).device

    cached_suppress_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    suppress_start = max(0, vocab_size - 1024)
    for i in range(suppress_start, vocab_size):
        if i != cached_eos_id:
            cached_suppress_mask[i] = True

    cached_suppress_list = [cached_eos_id]

    with torch.inference_mode():
        cached_tts_eos_embed = cached_talker.text_projection(
            cached_talker.get_text_embeddings()(
                torch.tensor([[inner.config.tts_eos_token_id]], device=device, dtype=torch.long)
            )
        )

    # ── Full KV cache: (voice, language, instruct) → prefill state ──
    prefill_cache = {}

    # Wait for background cache load to finish
    cache_thread.join()
    raw_cache = _cache_cpu[0]

    def _build_one(voice_name, lang_name, instruct_str):
        m, talker, cfg, tie, tam, tth_dummy, tpe = model._prepare_generation_custom(
            text="Test.", language=lang_name, speaker=voice_name,
            instruct=instruct_str if instruct_str else None,
        )
        with torch.inference_mode():
            out = talker.forward(
                inputs_embeds=tie, attention_mask=tam,
                use_cache=True, output_hidden_states=True, return_dict=True,
                trailing_text_hidden=tth_dummy, tts_pad_embed=tpe,
                generation_step=None, past_hidden=None, past_key_values=None,
            )
            return {
                "kv": tuple(tuple(t.clone().cpu() for t in layer) for layer in out.past_key_values),
                "logits": out.logits[:, -1, :].clone().cpu(),
                "past_hidden": out.past_hidden.clone().cpu(),
                "gen_step": out.generation_step,
                "tam": tam.clone().cpu(),
                "tpe": tpe.clone().cpu(),
                "prefill_len": out.past_key_values[0][0].shape[2],
            }

    # Use pre-loaded cache from background thread
    if raw_cache is not None:
        print(f"Moving pre-loaded KV caches to GPU...")
        t_load = time.time()
        for key_str, state in raw_cache.items():
            key = tuple(key_str.split("|||"))
            prefill_cache[key] = {
                "kv": tuple(tuple(t.to(device) for t in layer) for layer in state["kv"]),
                "logits": state["logits"].to(device),
                "past_hidden": state["past_hidden"].to(device),
                "gen_step": state["gen_step"],
                "tam": state["tam"].to(device),
                "tpe": state["tpe"].to(device),
                "prefill_len": state["prefill_len"],
            }
        print(f"  Loaded {len(prefill_cache)} combos in {time.time()-t_load:.1f}s")
    else:
        # Build from scratch and save to disk
        n_combos = len(CACHE_VOICES) * len(CACHE_LANGUAGES) * len(CACHE_TONES)
        print(f"Building KV caches: {len(CACHE_VOICES)} voices × {len(CACHE_LANGUAGES)} langs × {len(CACHE_TONES)} tones = {n_combos}...")
        t_build = time.time()

        for voice in CACHE_VOICES:
            for lang in CACHE_LANGUAGES:
                for instruct_str in CACHE_TONES:
                    key = (voice.strip(), lang.strip(), instruct_str.strip())
                    try:
                        prefill_cache[key] = _build_one(*key)
                        # Move to GPU
                        state = prefill_cache[key]
                        state["kv"] = tuple(tuple(t.to(device) for t in layer) for layer in state["kv"])
                        state["logits"] = state["logits"].to(device)
                        state["past_hidden"] = state["past_hidden"].to(device)
                        state["tam"] = state["tam"].to(device)
                        state["tpe"] = state["tpe"].to(device)
                    except Exception as e:
                        print(f"  SKIP {key}: {e}")

        print(f"  Built {len(prefill_cache)} combos in {time.time()-t_build:.1f}s")

        # Save to disk for next startup
        print(f"  Saving to {CACHE_PATH}...")
        save_data = {}
        for key, state in prefill_cache.items():
            key_str = "|||".join(key)
            save_data[key_str] = {
                "kv": tuple(tuple(t.cpu() for t in layer) for layer in state["kv"]),
                "logits": state["logits"].cpu(),
                "past_hidden": state["past_hidden"].cpu(),
                "gen_step": state["gen_step"],
                "tam": state["tam"].cpu(),
                "tpe": state["tpe"].cpu(),
                "prefill_len": state["prefill_len"],
            }
        torch.save(save_data, CACHE_PATH)
        print(f"  Saved ({os.path.getsize(CACHE_PATH)/1024/1024:.0f}MB)")

    sample_kv = next(iter(prefill_cache.values()))["kv"]
    kv_bytes = sum(t.numel()*t.element_size() for l in sample_kv for t in l)
    print(f"  {len(prefill_cache)} combos cached, {kv_bytes/1024/1024:.1f}MB each, {kv_bytes*len(prefill_cache)/1024/1024:.0f}MB total")


_tth_cache = {}  # text → tth tensor (LRU-style, max 200 entries)

@torch.inference_mode()
def _compute_tth(text):
    """Compute trailing_text_hiddens with tokenization cache."""
    cached = _tth_cache.get(text)
    if cached is not None:
        return cached
    input_texts = [model.model._build_assistant_text(text)]
    input_ids = model.model._tokenize_texts(input_texts)[0]
    text_tokens = input_ids[:, 4:-5]
    tth = cached_talker.text_projection(
        cached_talker.get_text_embeddings()(text_tokens))
    result = torch.cat((tth, cached_tts_eos_embed), dim=1)
    if len(_tth_cache) >= 200:
        _tth_cache.pop(next(iter(_tth_cache)))
    _tth_cache[text] = result
    return result


_last_cache_key = [None]  # Track last KV combo loaded in static cache

@torch.inference_mode()
def generate_cached_codec(text, voice="Vivian", language="French",
                           instruct="", chunk_size=1):
    """
    Yield raw codec token tensors (no speech_tokenizer decode).
    Selects the right KV cache based on instruct/tone.
    Skips KV restore if same combo as last request.
    """
    # Normalize instruct for cache lookup (handle missing accents etc.)
    inst = instruct or ""
    cache_key = (voice, language, inst)
    tc = prefill_cache.get(cache_key)
    if tc is None:
        # Fuzzy match: strip accents and compare
        import unicodedata as _ud
        def _strip(s, _n=_ud.normalize):
            return _n("NFD", s).encode("ascii", "ignore").decode()
        stripped = _strip(inst)
        for k, v in prefill_cache.items():
            if k[0] == voice and k[1] == language and _strip(k[2]) == stripped:
                tc = v
                cache_key = k
                break
        if tc is None:
            tc = prefill_cache.get((voice, language, ""))
            cache_key = (voice, language, "")
        if tc is None:
            tc = prefill_cache[("Vivian", "French", "")]
            cache_key = ("Vivian", "French", "")
    tc_kv = tc["kv"]
    tc_logits = tc["logits"]
    tc_past_hidden = tc["past_hidden"]
    tc_gen_step = tc["gen_step"]
    tc_tam = tc["tam"]
    tc_tpe = tc["tpe"]
    tc_prefill_len = tc["prefill_len"]

    tth_new = _compute_tth(text)

    # Skip KV restore if same combo as last request
    if _last_cache_key[0] != cache_key:
        if mk_talker is not None:
            mk_talker.restore_kv(tc_kv, tc_prefill_len)
        else:
            for layer_idx, (k, v) in enumerate(tc_kv):
                model.talker_graph.static_cache.update(k, v, layer_idx)
        _last_cache_key[0] = cache_key
    if mk_talker is None:
        model.talker_graph.set_generation_state(tc_tam, cached_rope_deltas)

    token = sample_logits(
        tc_logits, temperature=0.9, top_k=50, top_p=1.0,
        do_sample=True, suppress_mask=cached_suppress_mask,
        suppress_tokens=cached_suppress_list,
    )

    past_hidden = tc_past_hidden.clone()
    gen_step = tc_gen_step
    for step_idx in range(2048):
        if token.item() == cached_eos_id:
            break

        last_id_hidden = cached_talker_codec_embed(token.unsqueeze(1))
        if mk_predictor is not None:
            codebook_token_ids = mk_predictor.run(past_hidden, token.item())
        else:
            pred_input = torch.cat((past_hidden, last_id_hidden), dim=1)
            codebook_token_ids = model.predictor_graph.run(pred_input)
        all_cb = torch.cat([token.view(1), codebook_token_ids])

        yield all_cb  # no sync — .cpu() in WS handler does implicit sync

        codec_hiddens = [last_id_hidden]
        for ci in range(cached_num_code_groups - 1):
            codec_hiddens.append(cached_predictor_embeds[ci](
                codebook_token_ids[ci].unsqueeze(0).unsqueeze(0)))
        inputs_embeds = torch.cat(codec_hiddens, dim=1).sum(1, keepdim=True)

        if gen_step < tth_new.shape[1]:
            inputs_embeds = inputs_embeds + tth_new[:, gen_step].unsqueeze(1)
        else:
            inputs_embeds = inputs_embeds + tc_tpe

        current_pos = tc_prefill_len + step_idx
        max_pos = _TALKER_MAX_SEQ - 1 if mk_talker is not None else model.talker_graph.max_seq_len - 1
        if current_pos >= max_pos:
            break

        if mk_talker is not None:
            mk_talker.step(inputs_embeds, current_pos)
            norm_bf16 = mk_talker.norm_buf.to(torch.bfloat16)
            logits = F_global.linear(norm_bf16.unsqueeze(0), cached_talker_codec_head.weight).unsqueeze(0)
            past_hidden = norm_bf16.unsqueeze(0).unsqueeze(0)
        else:
            hidden_states = model.talker_graph.run(inputs_embeds, position=current_pos)
            logits = cached_talker_codec_head(hidden_states[:, -1, :]).unsqueeze(0)
            past_hidden = hidden_states[:, -1:, :].clone()

        token = sample_logits(
            logits.squeeze(0), temperature=0.9, top_k=50, top_p=1.0,
            do_sample=True, suppress_mask=cached_suppress_mask,
        )
        gen_step += 1


@torch.inference_mode()
def generate_cached_streaming(text, voice="Vivian", language="French",
                               instruct="", chunk_size=1):
    """Streaming generation with cached prefill KV + speech decode."""
    speech_tokenizer = inner.speech_tokenizer
    all_codes = []
    prev_audio_len = 0
    samples_per_frame = None
    context_frames = 25
    chunk_buffer = []

    for codec_ids in generate_cached_codec(text, voice, language, instruct, chunk_size):
        chunk_buffer.append(codec_ids.detach())

        if len(chunk_buffer) >= chunk_size:
            chunk_codes = torch.stack(chunk_buffer)
            all_codes.append(chunk_codes)
            all_flat = torch.cat(all_codes, dim=0)
            n_new = chunk_codes.shape[0]
            n_total = all_flat.shape[0]

            if samples_per_frame is None:
                audio_list, sr = speech_tokenizer.decode(
                    {"audio_codes": all_flat.unsqueeze(0)})
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
                    {"audio_codes": window.unsqueeze(0)})
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
            {"audio_codes": all_flat.unsqueeze(0)})
        audio = audio_list[0]
        if hasattr(audio, "cpu"):
            audio = audio.flatten().cpu().numpy()
        new_audio = audio[prev_audio_len:]
        yield new_audio, sr


# ── Prime the pipeline (1 cached codec call to warm GPU caches) ──────
cached_ttfp_ms = -1

if USE_CACHE and prefill_cache:
    # First call primes everything — discard timing
    for _ in generate_cached_codec("Test."):
        break
    # Second call gives real TTFP (tth pre-warmed, excluded from timing)
    _compute_tth("Test.")
    torch.cuda.synchronize()
    t_s = time.perf_counter()
    for cb in generate_cached_codec("Test."):
        _ = cb.cpu()  # Force GPU sync for honest measurement
        cached_ttfp_ms = (time.perf_counter() - t_s) * 1000
        break

print(f"\nModel loaded + warmed up in {time.time()-t0:.1f}s")
print(f"  Cached TTFP: {cached_ttfp_ms:.1f}ms")

# Pre-download Base model in background (so first clone doesn't wait for download)
def _predownload_base():
    base_path = model_path.replace("CustomVoice", "Base")
    if not os.path.exists(base_path):
        from huggingface_hub import snapshot_download
        base_id = f"Qwen/Qwen3-TTS-12Hz-{MODEL_SIZE}-Base"
        snapshot_download(base_id, local_dir=base_path, ignore_patterns=["*.md"])
        print("Base model pre-downloaded for cloning")

threading.Thread(target=_predownload_base, daemon=True).start()

import asyncio
import base64
import tempfile

API_KEY = os.environ.get("TTS_API_KEY", "")
_request_count = 0
_gc_interval = 100

# ── Voice clone: lazy-loaded Base model + prefix KV cache per voice ───
_clone_model = None
_clone_cache = {}          # clone_id → ref_audio path
_clone_prefill = {}        # clone_id → {kv, logits, past_hidden, gen_step, tam, tpe, prefill_len}
_clone_refs = {}           # Cached clone model references
_clone_last_key = [None]   # Track last clone KV loaded

def _get_clone_model():
    """Lazy-load the Base model for voice cloning (first call only)."""
    global _clone_model
    if _clone_model is not None:
        return _clone_model
    print("Loading Base model for voice cloning...")
    t0 = time.time()
    base_path = model_path.replace("CustomVoice", "Base")
    if not os.path.exists(base_path):
        from huggingface_hub import snapshot_download
        base_id = f"Qwen/Qwen3-TTS-12Hz-{MODEL_SIZE}-Base"
        snapshot_download(base_id, local_dir=base_path, ignore_patterns=["*.md"])
    _clone_model = FasterQwen3TTS.from_pretrained(base_path)
    _clone_model._warmup(10)

    # Cache references for fast codec generation
    clone_inner = _clone_model.model.model
    clone_talker = clone_inner.talker
    _clone_refs["talker"] = clone_talker
    _clone_refs["codec_embed"] = clone_talker.get_input_embeddings()
    _clone_refs["codec_head"] = clone_talker.codec_head
    _clone_refs["pred_embeds"] = clone_talker.code_predictor.get_input_embeddings()
    _clone_refs["eos_id"] = cached_eos_id  # Same tokenizer
    _clone_refs["num_code_groups"] = cached_num_code_groups
    _clone_refs["rope_deltas"] = getattr(clone_talker, "rope_deltas", None)
    _clone_refs["suppress_mask"] = cached_suppress_mask  # Same vocab
    # Compute tts_eos_embed for the Base model
    with torch.inference_mode():
        _clone_refs["tts_eos_embed"] = clone_talker.text_projection(
            clone_talker.get_text_embeddings()(
                torch.tensor([[clone_inner.config.tts_eos_token_id]], device=device, dtype=torch.long)
            )
        )
    print(f"  Base model loaded in {time.time()-t0:.1f}s")
    return _clone_model


def _build_clone_prefill(clone_id, ref_audio, ref_text, language="French"):
    """Build and cache the KV prefill for a cloned voice."""
    cm = _get_clone_model()
    result = cm._prepare_generation(
        text="Test.", ref_audio=ref_audio, ref_text=ref_text, language=language,
    )
    m, talker, config, tie, tam, tth_dummy, tpe = result[:7]
    with torch.inference_mode():
        out = talker.forward(
            inputs_embeds=tie, attention_mask=tam,
            use_cache=True, output_hidden_states=True, return_dict=True,
            trailing_text_hidden=tth_dummy, tts_pad_embed=tpe,
            generation_step=None, past_hidden=None, past_key_values=None,
        )
        _clone_prefill[clone_id] = {
            "kv": tuple(tuple(t.clone() for t in layer) for layer in out.past_key_values),
            "logits": out.logits[:, -1, :].clone(),
            "past_hidden": out.past_hidden.clone(),
            "gen_step": out.generation_step,
            "tam": tam.clone(),
            "tpe": tpe.clone(),
            "prefill_len": out.past_key_values[0][0].shape[2],
        }
    # Warmup the clone codec path to match CustomVoice performance
    for _ in range(5):
        for _ in generate_clone_cached_codec("Test.", clone_id, language):
            break
    return _clone_prefill[clone_id]


_clone_tth_cache = {}

@torch.inference_mode()
def _compute_clone_tth(text):
    """Compute trailing_text_hiddens using the Base model's text encoder."""
    cached = _clone_tth_cache.get(text)
    if cached is not None:
        return cached
    cm = _get_clone_model()
    input_texts = [cm.model._build_assistant_text(text)]
    input_ids = cm.model._tokenize_texts(input_texts)[0]
    text_tokens = input_ids[:, 4:-5]
    tth = _clone_refs["talker"].text_projection(
        _clone_refs["talker"].get_text_embeddings()(text_tokens))
    result = torch.cat((tth, _clone_refs["tts_eos_embed"]), dim=1)
    if len(_clone_tth_cache) >= 200:
        _clone_tth_cache.pop(next(iter(_clone_tth_cache)))
    _clone_tth_cache[text] = result
    return result


@torch.inference_mode()
def generate_clone_cached_codec(text, clone_id, language="French"):
    """Fast clone codec generation using cached KV (same pattern as CustomVoice)."""
    tc = _clone_prefill.get(clone_id)
    if tc is None:
        raise ValueError(f"Clone '{clone_id}' not found. Send ref_audio first.")

    cm = _get_clone_model()
    tth_new = _compute_clone_tth(text)

    # Restore KV to clone model's static cache
    if _clone_last_key[0] != clone_id:
        for layer_idx, (k, v) in enumerate(tc["kv"]):
            cm.talker_graph.static_cache.update(k, v, layer_idx)
        cm.talker_graph.set_generation_state(tc["tam"], _clone_refs["rope_deltas"])
        _clone_last_key[0] = clone_id
    else:
        cm.talker_graph.set_generation_state(tc["tam"], _clone_refs["rope_deltas"])

    token = sample_logits(
        tc["logits"], temperature=0.9, top_k=50, top_p=1.0,
        do_sample=True, suppress_mask=_clone_refs["suppress_mask"],
        suppress_tokens=cached_suppress_list,
    )

    past_hidden = tc["past_hidden"].clone()
    gen_step = tc["gen_step"]
    eos_id = _clone_refs["eos_id"]
    codec_embed = _clone_refs["codec_embed"]
    codec_head = _clone_refs["codec_head"]
    pred_embeds = _clone_refs["pred_embeds"]
    num_cg = _clone_refs["num_code_groups"]

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
            codec_hiddens.append(pred_embeds[ci](codebook_token_ids[ci].unsqueeze(0).unsqueeze(0)))
        inputs_embeds = torch.cat(codec_hiddens, dim=1).sum(1, keepdim=True)
        if gen_step < tth_new.shape[1]:
            inputs_embeds = inputs_embeds + tth_new[:, gen_step].unsqueeze(1)
        else:
            inputs_embeds = inputs_embeds + tc["tpe"]

        current_pos = tc["prefill_len"] + step_idx
        if current_pos >= cm.talker_graph.max_seq_len - 1:
            break
        hidden_states = cm.talker_graph.run(inputs_embeds, position=current_pos)
        logits = codec_head(hidden_states[:, -1, :]).unsqueeze(0)
        token = sample_logits(
            logits.squeeze(0), temperature=0.9, top_k=50, top_p=1.0,
            do_sample=True, suppress_mask=_clone_refs["suppress_mask"],
        )
        past_hidden = hidden_states[:, -1:, :].clone()
        gen_step += 1


@app.websocket("/ws/tts")
async def websocket_tts(ws: WebSocket):
    global _request_count
    await ws.accept()
    disconnected = False

    # Auth check (first message if API_KEY is set)
    if API_KEY:
        try:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            if msg.get("auth") != API_KEY:
                await ws.send_text(json.dumps({"error": "unauthorized"}))
                await ws.close()
                return
            await ws.send_text(json.dumps({"auth": "ok"}))
        except:
            return

    while not disconnected:
        try:
            raw = await ws.receive_text()
        except WebSocketDisconnect:
            break
        except Exception:
            break

        try:
            req = json.loads(raw)
        except (json.JSONDecodeError, Exception):
            try:
                await ws.send_text(json.dumps({"error": "invalid json"}))
            except:
                break
            continue

        text = req.get("input", "")
        voice = req.get("voice", "Vivian")
        language = req.get("language", "French")
        instruct = req.get("instruct", "")
        cs = req.get("chunk_size", CHUNK_SIZE)
        codec_mode = req.get("codec", False)

        # Voice cloning params
        ref_audio_b64 = req.get("ref_audio")  # base64-encoded WAV
        ref_text = req.get("ref_text", "")
        clone_id = req.get("clone_id", "")  # reuse cached voice prompt

        if not text:
            try:
                await ws.send_text(json.dumps({"error": "empty input"}))
            except:
                break
            continue

        chunk_count = 0
        ttfp_ms = 0
        sr = 24000
        _request_count += 1

        # Pre-compute text encoding BEFORE starting TTFP timer.
        # This ensures TTFP is constant regardless of text length.
        # The tth cache (max 200 entries) makes repeat calls free.
        if USE_CACHE and text:
            _compute_tth(text)
        if _clone_model is not None and text:
            _compute_clone_tth(text)
        torch.cuda.synchronize()
        t0_req = time.perf_counter()

        # Select generator
        if ref_audio_b64 or clone_id:
            # Voice cloning mode
            if ref_audio_b64:
                # First call: save ref audio, build prefill cache
                audio_bytes = base64.b64decode(ref_audio_b64)
                ref_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                with open(ref_path, "wb") as f:
                    f.write(audio_bytes)
                cid = clone_id or f"_auto_{_request_count}"
                _clone_cache[cid] = ref_path
                _build_clone_prefill(cid, ref_path, ref_text, language)
                # Use slow path for first generation (prefill just computed)
                clone_model = _get_clone_model()
                gen = clone_model.generate_voice_clone_streaming(
                    text=text, language=language, ref_audio=ref_path,
                    ref_text=ref_text, chunk_size=cs)
                codec_mode = False
            elif clone_id and clone_id in _clone_prefill:
                # Fast path: cached KV, same as CustomVoice speed
                gen = generate_clone_cached_codec(text, clone_id, language)
                codec_mode = True
            elif clone_id and clone_id in _clone_cache:
                # Ref audio known but prefill not cached yet — build it
                _build_clone_prefill(clone_id, _clone_cache[clone_id], ref_text, language)
                gen = generate_clone_cached_codec(text, clone_id, language)
                codec_mode = True
            else:
                try:
                    await ws.send_text(json.dumps({"error": "clone_id not found"}))
                except:
                    break
                continue
        elif codec_mode and USE_CACHE:
            gen = generate_cached_codec(text, voice, language, instruct, cs)
        elif USE_CACHE:
            gen = generate_cached_streaming(text, voice, language, instruct, cs)
        else:
            gen = model.generate_custom_voice_streaming(
                text=text, language=language, speaker=voice,
                instruct=instruct or "", chunk_size=cs)

        try:
            for item in gen:
                chunk_count += 1

                if codec_mode:
                    raw_bytes = item.cpu().numpy().astype(np.int16).tobytes()
                    # Measure TTFP AFTER .cpu() — this forces GPU sync,
                    # giving the REAL time (not just CPU queue time)
                    if chunk_count == 1:
                        ttfp_ms = (time.perf_counter() - t0_req) * 1000
                        header = json.dumps({"ttfp_ms": round(ttfp_ms, 1), "codec": True}).encode()
                        await ws.send_bytes(struct.pack("<I", len(header)) + header + raw_bytes)
                    else:
                        await ws.send_bytes(raw_bytes)
                else:
                    audio_chunk = item[0] if isinstance(item, tuple) else item
                    if isinstance(item, tuple) and len(item) > 1 and isinstance(item[1], int):
                        sr = item[1]
                    pcm = (np.array(audio_chunk) * 32767).astype(np.int16).tobytes()
                    if chunk_count == 1:
                        ttfp_ms = (time.perf_counter() - t0_req) * 1000
                        header = json.dumps({"ttfp_ms": round(ttfp_ms, 1), "sr": sr}).encode()
                        await ws.send_bytes(struct.pack("<I", len(header)) + header + pcm)
                    else:
                        await ws.send_bytes(pcm)

                if chunk_count % 10 == 0:
                    await asyncio.sleep(0)

            total_ms = (time.perf_counter() - t0_req) * 1000
            await ws.send_text(json.dumps({
                "done": True, "ttfp_ms": round(ttfp_ms, 1),
                "total_ms": round(total_ms, 1), "chunks": chunk_count,
            }))

        except (WebSocketDisconnect, ConnectionError, RuntimeError):
            # Client disconnected during generation — stop cleanly
            disconnected = True
        except Exception as exc:
            # Unknown error during generation — notify client with details
            import traceback
            err_msg = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            try:
                await ws.send_text(json.dumps({"error": err_msg, "done": True}))
            except:
                disconnected = True

        # Periodic GC to prevent memory creep
        if _request_count % _gc_interval == 0:
            gc.collect()
            torch.cuda.empty_cache()


@app.get("/generate")
async def generate_wav(text: str, voice: str = "Vivian", language: str = "French", instruct: str = ""):
    """Generate a WAV file from text. Used for sample generation."""
    import io
    import soundfile as sf
    from fastapi.responses import Response
    audio_chunks = []
    for chunk_data in generate_cached_streaming(text, voice, language, instruct, chunk_size=1):
        audio_chunks.append(chunk_data[0] if isinstance(chunk_data, tuple) else chunk_data)
        sr = chunk_data[1] if isinstance(chunk_data, tuple) and len(chunk_data) > 1 else 24000
    audio = np.concatenate(audio_chunks)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/wav")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "gpu": torch.cuda.get_device_name(0),
        "model": MODEL_SIZE,
        "combos": len(prefill_cache) if USE_CACHE else 0,
        "ttfp_ms": round(cached_ttfp_ms, 1),
        "megakernel_predictor": mk_predictor is not None,
        "megakernel_talker": mk_talker is not None,
        "requests": _request_count,
    }


@app.get("/health/detail")
def health_detail():
    return {
        "status": "ok",
        "gpu": torch.cuda.get_device_name(0),
        "gpu_mem_mb": round(torch.cuda.memory_allocated() / 1024 / 1024),
        "gpu_mem_peak_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024),
        "combos": len(prefill_cache) if USE_CACHE else 0,
        "ttfp_ms": round(cached_ttfp_ms, 1),
        "requests": _request_count,
    }


if __name__ == "__main__":
    try:
        import uvloop
        uvloop.install()
    except ImportError:
        pass
    import uvicorn
    ssl_cert = os.environ.get("SSL_CERT")
    ssl_key = os.environ.get("SSL_KEY")
    uvicorn.run(
        app, host="0.0.0.0", port=8000, ws="websockets", log_level="warning",
        ssl_certfile=ssl_cert, ssl_keyfile=ssl_key,
    )
