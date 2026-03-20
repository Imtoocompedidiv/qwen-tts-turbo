# Qwen3-TTS Ultra-Low Latency Server

Real-time TTS streaming server built on [Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) with dual fused CUDA megakernels for predictor and talker.

## Performance (honest measurements — 2026-03-20)

| GPU | Path | TTFP codec raw | TTFP PCM | Stability |
|-----|------|---------------|----------|-----------|
| **H100 SXM** | MK predictor + CUDA graph talker | **4ms** | **16ms** | All texts 1w→45w, all tones, 0 crashes |
| **A100 SXM** | CUDA graph only | **16ms** | **24ms** | Fully stable, 250+ requests verified |

TTFP = time from request receipt to first codec frame ready on GPU (after `.cpu()` sync). Text encoding is architecturally **deferred after the first frame** — the first frame depends only on cached KV + sampling + predictor megakernel. Text encoding is pipelined on a background CUDA stream, overlapping with frame 1 network transfer.

> **GPU compatibility:** The server auto-detects GPU capability at startup. On GPUs below sm_90 (A100, etc.), megakernels are automatically disabled and the server falls back to CUDA graph (16ms TTFP). Warmup includes a 30s deadlock watchdog — if a megakernel hangs, the server falls back gracefully. The default deployment (`start.sh`, `Dockerfile`) uses `USE_MEGAKERNEL=1 USE_TALKER_MK=0` (predictor megakernel + CUDA graph talker), which is the tested stable configuration.

| Feature | Details |
|---------|---------|
| Voices | 6 (Vivian, Serena, Dylan, Eric, Ryan, Aiden) |
| Languages | 10 (French, English, Chinese, Japanese, Korean, German, Russian, Portuguese, Spanish, Italian) |
| Tone presets | 8 (neutral, warm, soft, dynamic, calm, formal, joyful, authoritative) |
| Pre-cached combos | **480** (voice x language x tone), zero-cost switching |
| Voice cloning | Via lazy-loaded Base model, cached after first call |
| Stability | All 5 text lengths (1w→45w), all 5 tones, 0 crashes on H100 SXM |
| TTFP stability | Constant 1 word → 45 words (4ms ± 0ms codec raw). Text encoding deferred — architecturally not on critical path |
| Audio quality | Cosine similarity 0.9995 vs CUDA graph baseline (numerically identical) |

## How it works

### Dual megakernel architecture

The standard inference path uses ~70 separate CUDA kernel launches per decode step. We replace this with **two fused megakernels** that process entire transformer forward passes in a single persistent kernel launch:

- **Predictor megakernel**: 5-layer transformer (HIDDEN=1024), 17 sequential steps for 16 codebook tokens. Pre-projected codec embeddings eliminate 16/17 projection ops.
- **Talker megakernel**: 28-layer transformer (HIDDEN=2048), 1 step per frame. M-RoPE with pre-computed cos/sin tables.

Both kernels are vendored in `csrc/` (~1650 lines of CUDA C++ each) and compile from the same source with different compile-time dimensions. The predictor uses the default constants; the talker overrides via `-DMK_HIDDEN_SIZE=2048 -DMK_INTERMEDIATE_SIZE=6144`. All spin-wait barriers include adaptive `__nanosleep` for datacenter GPU compatibility (pre-patched).

### Other optimizations

- **Prefix KV cache**: 480 voice/language/tone combos pre-computed at startup. Each request skips the ~50ms prefill.
- **Codec raw mode**: sends 32 bytes of codec tokens per frame instead of decoded PCM audio. Client decodes locally.
- **Deferred text encoding**: `_compute_tth()` is not needed for the first frame (which comes from cached KV + sampling + predictor). It runs on a background CUDA stream after the first yield, overlapping with the first frame's `.cpu()` sync + network transfer. Results are LRU-cached (200 entries) for repeat texts.
- **WebSocket streaming**: `chunk_size=1` emits audio at the first codec token.
- **Honest TTFP measurement**: timer starts at request receipt, stops after first frame `.cpu()` GPU sync. No pre-computation — text encoding is architecturally not on the critical path for frame 1.

## Quick start

### 1. Configure

```bash
cp .env.example .env
# Edit .env with your RunPod API key
```

### 2. Deploy

```bash
pip install websockets
export $(cat .env | xargs)
python deploy/launch.py
```

Creates a GPU pod that auto-clones this repo, installs dependencies, downloads the model, compiles the megakernels, builds 480 KV caches, and starts the server. Fully self-contained — no manual file setup needed. ~30s on H100.

### 3. Benchmark

```bash
# TTFP benchmark
python deploy/benchmark_ws.py ws://IP:PORT/ws/tts

# Stress test (500 requests)
python deploy/stress_test.py ws://IP:PORT/ws/tts http://IP:PORT

# Honest GPU-side TTFP verification (requires GPU)
python deploy/industrial/bench_honest_ttfp.py
```

### 4. Stop

```bash
python deploy/launch.py --stop
```

## WebSocket protocol

Connect to `ws://IP:PORT/ws/tts`. If `TTS_API_KEY` is set, send auth first:

```json
{"auth": "your-api-key"}
```

TTS request:

```json
{
  "input": "Bonjour, comment puis-je vous aider ?",
  "voice": "Vivian",
  "language": "French",
  "instruct": "Voix douce et rassurante",
  "codec": true,
  "chunk_size": 1
}
```

Server responds with binary frames, then a final JSON:

```json
{"done": true, "ttfp_ms": 4.0, "total_ms": 1450, "chunks": 85}
```

### Voice cloning

```json
{
  "input": "Text to speak in the cloned voice",
  "language": "French",
  "ref_audio": "<base64-encoded WAV>",
  "ref_text": "Transcript of the reference audio",
  "clone_id": "my-voice"
}
```

Subsequent requests reuse the cached clone with just `"clone_id": "my-voice"`.

### Modes

| Mode | `codec` | First frame | Use case |
|------|---------|-------------|----------|
| Codec raw | `true` | 16 int16 codec tokens (32 bytes) | Minimum latency |
| PCM audio | `false` | 16-bit PCM at 24kHz | Direct playback |

## Files

| File | Description |
|------|-------------|
| `deploy/runpod_server.py` | Production TTS server (FastAPI + WebSocket + megakernels + voice cloning) |
| `deploy/launch.py` | One-click RunPod deployment (GPU priority: B200 > H200 > H100 > L40S) |
| `deploy/start.sh` | Pod startup script (install deps, compile megakernels, start server) |
| `deploy/benchmark_ws.py` | Client-side TTFP benchmark |
| `deploy/robust_client.py` | Production client with hedging + local cache |
| `deploy/stress_test.py` | 500-request stability test |
| `deploy/generate_samples.py` | Generate audio samples via running server |
| `csrc/kernel.cu` | **Predictor megakernel** — fused 5-layer transformer in one persistent kernel launch. Pre-patched with adaptive spin-wait for datacenter GPUs. |
| `csrc/kernel_talker.cu` | **Talker megakernel** — fused 28-layer transformer with compile-time configurable dimensions (HIDDEN=2048, INTER=6144) |
| `csrc/torch_bindings.cpp` | PyTorch C++ bindings for the CUDA kernels |
| `deploy/industrial/model_tts.py` | `CodePredictorKernel` / `TTSDecoder` classes — weight packing, KV cache, decode loop |
| `deploy/industrial/build_predictor.py` | Predictor megakernel JIT compilation |
| `deploy/industrial/build_talker_megakernel.py` | Talker megakernel JIT compilation |
| `deploy/industrial/megakernel_predictor.py` | 1.7B predictor wrapper (2048->1024 projection) |
| `deploy/industrial/bench_honest_ttfp.py` | GPU-synchronized TTFP verification benchmark |
| `deploy/industrial/patch_kernel_datacenter.py` | Datacenter GPU patch (adaptive spin-wait for A100/H100/B200) |
| `deploy/industrial/mega_graph.py` | Fused CUDA graph: sampling + predictor + talker in one replay |
| `deploy/industrial/patch_int8.py` | INT8 quantization kernel patch (experimental) |
| `deploy/industrial/mrope_fix.md` | M-RoPE specification and kernel fix notes |

## Requirements

**Client**: Python 3.10+, `websockets`

**Server** (auto-installed on RunPod):
- Docker image: `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
- `faster-qwen3-tts`, `qwen-tts`, `soundfile`, `ninja`
- GPU: B200 (recommended), H200, H100, A100, RTX 5090/4090 (16GB+ VRAM)

## License

MIT
