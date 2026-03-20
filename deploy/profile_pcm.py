"""Profile PCM decode path — find where the 24ms vocoder time goes.

Measures:
1. Speech tokenizer decode time vs input size (1, 2, 4, 8 frames)
2. First decode vs subsequent (warmup effect)
3. CUDA graph captured decode vs raw
4. Pipeline overlap potential
"""
import os, sys, time
os.environ["MODEL_SIZE"] = "1.7B"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

print("Loading model...")
t0 = time.time()
from faster_qwen3_tts import FasterQwen3TTS
model = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
model._warmup(10)
inner = model.model.model
speech_tokenizer = inner.speech_tokenizer
print(f"Model loaded in {time.time()-t0:.1f}s")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Generate some real codec tokens to use for decode profiling
print("\nGenerating codec tokens for profiling...")
all_tokens = []
for audio_data in model.generate_custom_voice_streaming(
    text="Bonjour, comment puis-je vous aider aujourd'hui ?",
    language="French", speaker="Vivian", instruct="", chunk_size=1
):
    if isinstance(audio_data, tuple):
        break  # streaming returns (audio, sr)
    all_tokens.append(audio_data.detach())
    if len(all_tokens) >= 20:
        break

# If streaming didn't give us codec tokens, generate with codec path
if not all_tokens:
    print("Using generate_custom_voice for tokens...")
    wavs, sr = model.generate_custom_voice(
        text="Bonjour.", language="French", speaker="Vivian", instruct="")
    # Create dummy codec tokens
    all_tokens = [torch.randint(0, 2048, (16,), device="cuda") for _ in range(20)]

print(f"Got {len(all_tokens)} codec frames")

# Stack into [N, 16] tensor
codec_frames = torch.stack(all_tokens[:20])  # [20, 16]
print(f"Codec frames shape: {codec_frames.shape}")

N = 30

def benchmark_decode(label, codes, warmup=5):
    """Benchmark speech_tokenizer.decode with CUDA events."""
    # Warmup
    for _ in range(warmup):
        audio_list, sr = speech_tokenizer.decode({"audio_codes": codes.unsqueeze(0)})
    torch.cuda.synchronize()

    times = []
    for _ in range(N):
        torch.cuda.synchronize()
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_start.record()
        audio_list, sr = speech_tokenizer.decode({"audio_codes": codes.unsqueeze(0)})
        ev_end.record()
        torch.cuda.synchronize()
        times.append(ev_start.elapsed_time(ev_end))

    times.sort()
    p50 = times[len(times)//2]
    print(f"  [{label:>20s}] p50={p50:.2f}ms  min={times[0]:.2f}ms  max={times[-1]:.2f}ms  "
          f"samples={codes.shape[0]} frames")
    return p50

print(f"\n{'='*70}")
print(f"SPEECH TOKENIZER DECODE PROFILING — {N} iterations each")
print(f"{'='*70}\n")

# 1. Decode time vs number of frames
print("--- Decode time vs frame count ---")
for n_frames in [1, 2, 3, 4, 8, 16]:
    codes = codec_frames[:n_frames]
    benchmark_decode(f"{n_frames} frames", codes)

# 2. First decode (cold) vs warmed
print("\n--- First decode (cold) vs warmed ---")
torch.cuda.synchronize()
# Clear any caches
torch.cuda.empty_cache()
ev_s = torch.cuda.Event(enable_timing=True)
ev_e = torch.cuda.Event(enable_timing=True)
codes_1 = codec_frames[:1]
ev_s.record()
audio_list, sr = speech_tokenizer.decode({"audio_codes": codes_1.unsqueeze(0)})
ev_e.record()
torch.cuda.synchronize()
print(f"  First (cold) decode: {ev_s.elapsed_time(ev_e):.2f}ms")
# Warmed
benchmark_decode("warmed 1 frame", codes_1)

# 3. Test if we can pipeline decode with generation
print("\n--- Pipeline potential: decode on background stream ---")
bg_stream = torch.cuda.Stream()
codes_1 = codec_frames[:1]

# Baseline: decode on default stream
torch.cuda.synchronize()
times_default = []
for _ in range(N):
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    audio_list, sr = speech_tokenizer.decode({"audio_codes": codes_1.unsqueeze(0)})
    e.record()
    torch.cuda.synchronize()
    times_default.append(s.elapsed_time(e))
times_default.sort()
print(f"  Default stream: p50={times_default[len(times_default)//2]:.2f}ms")

# On background stream
torch.cuda.synchronize()
times_bg = []
for _ in range(N):
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record(bg_stream)
    with torch.cuda.stream(bg_stream):
        audio_list, sr = speech_tokenizer.decode({"audio_codes": codes_1.unsqueeze(0)})
    e.record(bg_stream)
    bg_stream.synchronize()
    times_bg.append(s.elapsed_time(e))
times_bg.sort()
print(f"  Background stream: p50={times_bg[len(times_bg)//2]:.2f}ms")

# 4. Full PCM TTFP simulation
print(f"\n{'='*70}")
print("FULL PCM TTFP SIMULATION")
print(f"{'='*70}\n")

from faster_qwen3_tts.sampling import sample_logits

talker = inner.talker
_, _, config, tie, tam, tth_dummy, tpe = model._prepare_generation_custom(
    text="Test.", language="French", speaker="Vivian", instruct=""
)
with torch.inference_mode():
    out = talker.forward(
        inputs_embeds=tie, attention_mask=tam,
        use_cache=True, output_hidden_states=True, return_dict=True,
        trailing_text_hidden=tth_dummy, tts_pad_embed=tpe,
        generation_step=None, past_hidden=None, past_key_values=None,
    )
    tc_logits = out.logits[:, -1, :].clone()

suppress_mask = torch.zeros(config.vocab_size, dtype=torch.bool, device="cuda")
for i in range(max(0, config.vocab_size - 1024), config.vocab_size):
    if i != config.codec_eos_token_id:
        suppress_mask[i] = True

# Simulate: generate 1 codec frame + decode = PCM TTFP
print("--- Current PCM TTFP (sequential) ---")
ttfps = []
for _ in range(N):
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e_gen = torch.cuda.Event(enable_timing=True)
    e_decode = torch.cuda.Event(enable_timing=True)
    s.record()
    with torch.inference_mode():
        token = sample_logits(tc_logits, temperature=0.9, top_k=50, top_p=1.0,
                              do_sample=True, suppress_mask=suppress_mask)
        codec_embed = talker.get_input_embeddings()(token.unsqueeze(1))
        pred_input = torch.cat((out.past_hidden, codec_embed), dim=1)
        codebook_ids = model.predictor_graph.run(pred_input)
        all_cb = torch.cat([token.view(1), codebook_ids])
    e_gen.record()
    # Decode
    audio_list, sr = speech_tokenizer.decode({"audio_codes": all_cb.unsqueeze(0).unsqueeze(0)})
    a = audio_list[0]
    if hasattr(a, "cpu"): a = a.cpu()
    e_decode.record()
    torch.cuda.synchronize()
    gen_ms = s.elapsed_time(e_gen)
    total_ms = s.elapsed_time(e_decode)
    decode_ms = total_ms - gen_ms
    ttfps.append((gen_ms, decode_ms, total_ms))

gen_p50 = sorted([t[0] for t in ttfps])[N//2]
dec_p50 = sorted([t[1] for t in ttfps])[N//2]
tot_p50 = sorted([t[2] for t in ttfps])[N//2]
print(f"  Generation:  {gen_p50:.2f}ms")
print(f"  Decode:      {dec_p50:.2f}ms")
print(f"  TOTAL:       {tot_p50:.2f}ms")

# Pipelined: decode on bg stream, overlap with generation of next frame
print("\n--- Pipelined PCM TTFP (decode on bg stream) ---")
ttfps_pipe = []
for _ in range(N):
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e_total = torch.cuda.Event(enable_timing=True)
    s.record()
    with torch.inference_mode():
        token = sample_logits(tc_logits, temperature=0.9, top_k=50, top_p=1.0,
                              do_sample=True, suppress_mask=suppress_mask)
        codec_embed = talker.get_input_embeddings()(token.unsqueeze(1))
        pred_input = torch.cat((out.past_hidden, codec_embed), dim=1)
        codebook_ids = model.predictor_graph.run(pred_input)
        all_cb = torch.cat([token.view(1), codebook_ids])
    # Kick decode to bg stream
    decode_ev = torch.cuda.Event()
    with torch.cuda.stream(bg_stream):
        audio_list, sr = speech_tokenizer.decode({"audio_codes": all_cb.unsqueeze(0).unsqueeze(0)})
        decode_ev.record()
    # Meanwhile, main stream does nothing (in real pipeline, would generate frame 2)
    decode_ev.synchronize()
    a = audio_list[0]
    if hasattr(a, "cpu"): a = a.cpu()
    e_total.record()
    torch.cuda.synchronize()
    ttfps_pipe.append(s.elapsed_time(e_total))

pipe_p50 = sorted(ttfps_pipe)[N//2]
print(f"  Pipelined TOTAL: {pipe_p50:.2f}ms")
print(f"  Improvement: {tot_p50 - pipe_p50:.2f}ms ({(1 - pipe_p50/tot_p50)*100:.0f}%)")

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"  Codec raw TTFP:        ~{gen_p50:.1f}ms")
print(f"  PCM TTFP (current):    ~{tot_p50:.1f}ms  (gen {gen_p50:.1f} + decode {dec_p50:.1f})")
print(f"  PCM TTFP (pipelined):  ~{pipe_p50:.1f}ms")
print(f"  Vocoder 1-frame:       ~{dec_p50:.1f}ms")
print(f"  Vocoder amortized (4f): check above")
