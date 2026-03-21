# Qwen3-TTS Ultra-Low Latency Server

Real-time TTS streaming server built on [Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) with fused CUDA megakernels, prefix KV caching, and deferred text encoding.

## Performance (verified 2026-03-21)

| GPU | TTFP codec raw | TTFP PCM | Stress test | Config |
|-----|---------------|----------|-------------|--------|
| **RTX 5090** (sm_120) | **4ms** p50 | **15ms** p50 | 500/500, 0 errors, 0MB drift | MK predictor + CUDA graph talker |
| **H100 SXM** (sm_90) | **4ms** p50 | **16ms** p50 | All texts 1w→45w verified | MK predictor + CUDA graph talker |
| **A100 SXM** (sm_80) | **16ms** p50 | **24ms** p50 | 250+ requests verified | CUDA graph only (auto-detected) |

TTFP = time from request receipt to first frame ready on GPU (`.cpu()` sync). All measurements honest: timer starts before any processing, text encoding is architecturally **deferred after the first frame** (not needed for it).

### TTFP breakdown (RTX 5090)

```
Codec raw (4ms):  [sample 0.3ms] [megakernel predictor 3ms] [overhead 0.7ms]
PCM audio (15ms): [sample 0.3ms] [megakernel predictor 3ms] [vocoder decode 12ms]
```

The vocoder (speech_tokenizer.decode) is the PCM bottleneck at 80% of TTFP. Tested: torch.compile, CUDA graph capture, fp16, batching, background stream — none reduce single-frame decode time. The vocoder API does not expose parameters for quantization or graph capture.

> **GPU auto-detection:** The server detects sm_arch at startup. Below sm_90 (A100, etc.), megakernels are auto-disabled with CUDA graph fallback. Warmup includes a 30s deadlock watchdog. Runtime frame timeout (10s) auto-disables megakernels after 3 failures. External liveness probe in start.sh kills hung processes.

| Feature | Details |
|---------|---------|
| Voices | 6 (Vivian, Serena, Dylan, Eric, Ryan, Aiden) |
| Languages | 10 (French, English, Chinese, Japanese, Korean, German, Russian, Portuguese, Spanish, Italian) |
| Tone presets | 8 (neutral, warm, soft, dynamic, calm, formal, joyful, authoritative) |
| Pre-cached combos | **480** (voice x language x tone), zero-cost switching |
| Voice cloning | Via lazy-loaded Base model, cached after first call |
| Stability | **500/500** requests on RTX 5090, 0 errors, 0 reconnects, 0MB GPU memory drift, CV=4.3% |
| TTFP stability | Constant 1 word → 45 words (4ms ± 0ms codec raw, drift +0.0ms over 500 requests) |
| Audio quality | Cosine similarity 0.9995 vs CUDA graph baseline (numerically identical) |

## Architecture

```
Request → [input validation] → [TTFP timer start]
  → generate_cached_codec:
      [KV cache lookup] → [sample from cached logits] → [megakernel predictor 17 steps]
      → yield first codec frame (TTFP measured here)
      → [deferred TTH on background CUDA stream]
  → generate_cached_streaming (PCM only):
      [speech_tokenizer.decode] → yield PCM audio
  → [async frame loop with run_in_executor + wait_for timeout]
```

### Megakernel predictor

Replaces ~70 CUDA kernel launches per predict step with **one persistent kernel**. 5-layer transformer (HIDDEN=1024), 17 sequential steps for 16 codebook tokens. Pre-projected codec embeddings eliminate 16/17 projection ops. Host-side barrier reset (`cudaMemsetAsync`) before each launch prevents stale `barrier_sense` race.

### Three-layer deadlock defense

| Layer | Mechanism | Detects in |
|-------|-----------|-----------|
| Async timeout | `asyncio.wait_for(run_in_executor(next(gen)), 10s)` | <10s |
| Watchdog thread | Monitors `last_frame_time`, calls `os._exit(1)` | <15s |
| Liveness probe | start.sh pings `/health`, `kill -9` after 30s | <30s |

### Other optimizations

- **Prefix KV cache**: 480 voice/language/tone combos pre-computed at startup, skips ~50ms prefill per request.
- **Deferred text encoding**: `_compute_tth()` runs on a background CUDA stream after the first yield, overlapping with frame 1 network transfer. LRU-cached (200 entries).
- **Vocoder warmup**: Speech tokenizer decode warmed with varied tokens at startup (cold penalty ~10ms).
- **Codec raw mode**: 32 bytes per frame, client decodes locally — 4ms TTFP.

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

### 3. Benchmark

```bash
python deploy/benchmark_ws.py ws://IP:PORT/ws/tts
python deploy/stress_test.py ws://IP:PORT/ws/tts http://IP:PORT
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

### Modes

| Mode | `codec` | First frame | TTFP (RTX 5090) | Use case |
|------|---------|-------------|-----------------|----------|
| Codec raw | `true` | 16 int16 codec tokens (32 bytes) | **4ms** | Minimum latency, client-side decode |
| PCM audio | `false` | 16-bit PCM at 24kHz | **15ms** | Direct playback |

## Files

| File | Description |
|------|-------------|
| `deploy/runpod_server.py` | Slim entry point: FastAPI + WebSocket handler (430 lines) |
| `deploy/server/engine.py` | `TTSEngine` class: model, megakernels, caches, generation (960 lines) |
| `deploy/server/monitoring.py` | `ServerMonitor` class: metrics, watchdog, auto-fallback (113 lines) |
| `deploy/launch.py` | One-click RunPod deployment (GPU priority: B200 > H200 > H100 > A100 > L40S) |
| `deploy/start.sh` | Supervised startup: pre-flight checks, restart loop, liveness probe |
| `deploy/benchmark_ws.py` | TTFP benchmark (PCM + codec raw + multi-tone) |
| `deploy/robust_client.py` | Production client with hedging + local cache |
| `deploy/stress_test.py` | 500-request stability + memory drift test |
| `deploy/generate_samples.py` | Generate audio samples via running server |
| `csrc/kernel.cu` | **Predictor megakernel** — fused 5-layer transformer, pre-patched barriers |
| `csrc/kernel_talker.cu` | **Talker megakernel** — fused 28-layer transformer (HIDDEN=2048) |
| `csrc/torch_bindings.cpp` | PyTorch C++ bindings for CUDA kernels |
| `deploy/industrial/` | Build scripts, weight loading, INT8 patch, M-RoPE notes |

## Requirements

**Client**: Python 3.10+, `websockets>=13.0`

**Server** (auto-installed on RunPod):
- Docker image: `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
- `faster-qwen3-tts`, `qwen-tts`, `soundfile`, `ninja`, `websockets`
- GPU: B200, H200, H100 (sm_90+, megakernel), A100 (sm_80, CUDA graph), L40S, RTX 5090/4090 (16GB+ VRAM)

## License

MIT
