"""Profile TTFP critical path — measure each step with CUDA events.

Runs on the GPU directly. No server needed.
Identifies where time is spent between request and first codec frame.
"""
import os
import sys
import time

os.environ["MODEL_SIZE"] = "1.7B"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# ── Load model ──
print("Loading model...")
t0 = time.time()
from faster_qwen3_tts import FasterQwen3TTS
from faster_qwen3_tts.sampling import sample_logits

model = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
model._warmup(10)
inner = model.model.model
talker = inner.talker

print(f"Model loaded in {time.time()-t0:.1f}s")

# ── Build one prefill cache ──
print("Building prefill cache...")
_, _, config, tie, tam, tth_dummy, tpe = model._prepare_generation_custom(
    text="Test.", language="French", speaker="Vivian", instruct=""
)
eos_id = config.codec_eos_token_id
vocab_size = config.vocab_size
device = next(talker.parameters()).device

suppress_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
for i in range(max(0, vocab_size - 1024), vocab_size):
    if i != eos_id:
        suppress_mask[i] = True

with torch.inference_mode():
    out = talker.forward(
        inputs_embeds=tie, attention_mask=tam,
        use_cache=True, output_hidden_states=True, return_dict=True,
        trailing_text_hidden=tth_dummy, tts_pad_embed=tpe,
        generation_step=None, past_hidden=None, past_key_values=None,
    )
    tc_logits = out.logits[:, -1, :].clone()
    tc_past_hidden = out.past_hidden.clone()
    tc_kv = out.past_key_values
    tc_gen_step = out.generation_step
    tc_prefill_len = tc_kv[0][0].shape[2]

# Restore KV to static cache
with torch.inference_mode():
    for layer_idx, (k, v) in enumerate(tc_kv):
        model.talker_graph.static_cache.update(k, v, layer_idx)
    model.talker_graph.set_generation_state(tam, getattr(talker, "rope_deltas", None))

codec_embed = talker.get_input_embeddings()
codec_head = talker.codec_head
pred_embeds = talker.code_predictor.get_input_embeddings()
num_code_groups = config.num_code_groups

# ── tth for text encoding measurement ──
tts_eos_embed = None
with torch.inference_mode():
    tts_eos_embed = talker.text_projection(
        talker.get_text_embeddings()(
            torch.tensor([[inner.config.tts_eos_token_id]], device=device, dtype=torch.long)
        )
    )

def compute_tth(text):
    input_texts = [model.model._build_assistant_text(text)]
    input_ids = model.model._tokenize_texts(input_texts)[0]
    text_tokens = input_ids[:, 4:-5]
    tth = talker.text_projection(talker.get_text_embeddings()(text_tokens))
    return torch.cat((tth, tts_eos_embed), dim=1)

# ── Warmup ──
print("Warmup...")
for _ in range(5):
    with torch.inference_mode():
        token = sample_logits(tc_logits, temperature=0.9, top_k=50, top_p=1.0,
                              do_sample=True, suppress_mask=suppress_mask,
                              suppress_tokens=[eos_id])
        last_id_hidden = codec_embed(token.unsqueeze(1))
        pred_input = torch.cat((tc_past_hidden, last_id_hidden), dim=1)
        codebook_ids = model.predictor_graph.run(pred_input)
        all_cb = torch.cat([token.view(1), codebook_ids])
        _ = all_cb.cpu()
torch.cuda.synchronize()

# ── Profile each step ──
N = 50
print(f"\n{'='*70}")
print(f"TTFP CRITICAL PATH PROFILING — {N} iterations")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"{'='*70}\n")

texts = [
    ("Bonjour.", "1w"),
    ("Laissez-moi verifier cette information pour vous.", "7w"),
    ("Merci pour votre patience. J'ai examine votre dossier en detail.", "12w"),
]

for text, label in texts:
    # Pre-warm tth cache
    with torch.inference_mode():
        compute_tth(text)

    times_sample = []
    times_predictor = []
    times_concat = []
    times_cpu = []
    times_total = []
    times_tth = []
    times_total_with_tth = []

    for _ in range(N):
        torch.cuda.synchronize()

        # === Measure tth separately ===
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        with torch.inference_mode():
            _ = compute_tth(text)
        ev1.record()
        torch.cuda.synchronize()
        times_tth.append(ev0.elapsed_time(ev1))

        # === Measure TTFP critical path (without tth) ===
        torch.cuda.synchronize()
        e_start = torch.cuda.Event(enable_timing=True)
        e_after_sample = torch.cuda.Event(enable_timing=True)
        e_after_pred = torch.cuda.Event(enable_timing=True)
        e_after_concat = torch.cuda.Event(enable_timing=True)
        e_after_cpu = torch.cuda.Event(enable_timing=True)

        e_start.record()

        with torch.inference_mode():
            # Step 1: sample first token
            token = sample_logits(tc_logits, temperature=0.9, top_k=50, top_p=1.0,
                                  do_sample=True, suppress_mask=suppress_mask,
                                  suppress_tokens=[eos_id])
            e_after_sample.record()

            # Step 2: predictor
            last_id_hidden = codec_embed(token.unsqueeze(1))
            pred_input = torch.cat((tc_past_hidden, last_id_hidden), dim=1)
            codebook_ids = model.predictor_graph.run(pred_input)
            e_after_pred.record()

            # Step 3: concat
            all_cb = torch.cat([token.view(1), codebook_ids])
            e_after_concat.record()

            # Step 4: .cpu() sync (what client sees)
            data = all_cb.cpu()
            e_after_cpu.record()

        torch.cuda.synchronize()
        times_sample.append(e_start.elapsed_time(e_after_sample))
        times_predictor.append(e_after_sample.elapsed_time(e_after_pred))
        times_concat.append(e_after_pred.elapsed_time(e_after_concat))
        times_cpu.append(e_after_concat.elapsed_time(e_after_cpu))
        times_total.append(e_start.elapsed_time(e_after_cpu))
        times_total_with_tth.append(e_start.elapsed_time(e_after_cpu) + ev0.elapsed_time(ev1))

    def p50(lst):
        s = sorted(lst)
        return s[len(s)//2]

    print(f"[{label:>4s}] TTFP breakdown (p50 of {N} runs):")
    print(f"  sample_logits:     {p50(times_sample):6.2f}ms")
    print(f"  predictor_graph:   {p50(times_predictor):6.2f}ms")
    print(f"  concat:            {p50(times_concat):6.2f}ms")
    print(f"  .cpu() sync:       {p50(times_cpu):6.2f}ms")
    print(f"  ──────────────────────────")
    print(f"  TOTAL (no tth):    {p50(times_total):6.2f}ms  ← this is the honest TTFP")
    print(f"  tth encoding:      {p50(times_tth):6.2f}ms  (deferred, not on critical path)")
    print(f"  TOTAL (with tth):  {p50(times_total_with_tth):6.2f}ms  ← would be if tth not deferred")
    print()

# === Also profile the full talker step (for throughput) ===
print(f"{'='*70}")
print(f"FULL DECODE STEP (predictor + overhead + talker)")
print(f"{'='*70}\n")
step_times = []
for _ in range(N):
    torch.cuda.synchronize()
    with torch.inference_mode():
        token = sample_logits(tc_logits, temperature=0.9, top_k=50, top_p=1.0,
                              do_sample=True, suppress_mask=suppress_mask,
                              suppress_tokens=[eos_id])
        last_id_hidden = codec_embed(token.unsqueeze(1))
        pred_input = torch.cat((tc_past_hidden, last_id_hidden), dim=1)
        codebook_ids = model.predictor_graph.run(pred_input)

        # Build next step input
        codec_hiddens = [last_id_hidden]
        for ci in range(num_code_groups - 1):
            codec_hiddens.append(pred_embeds[ci](codebook_ids[ci].unsqueeze(0).unsqueeze(0)))
        inputs_embeds = torch.cat(codec_hiddens, dim=1).sum(1, keepdim=True) + tpe

    es = torch.cuda.Event(enable_timing=True)
    ee = torch.cuda.Event(enable_timing=True)
    es.record()
    with torch.inference_mode():
        hidden_states = model.talker_graph.run(inputs_embeds, position=tc_prefill_len)
        logits = codec_head(hidden_states[:, -1, :]).unsqueeze(0)
    ee.record()
    torch.cuda.synchronize()
    step_times.append(es.elapsed_time(ee))

print(f"Talker step: {p50(step_times):.2f}ms p50")
print(f"Throughput: {1000/p50(step_times):.0f} fps = {1000/p50(step_times)/12:.1f}x realtime @12Hz")
