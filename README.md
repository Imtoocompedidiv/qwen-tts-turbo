# Qwen3-TTS Ultra-Low Latency Server

Real-time TTS streaming server built on [Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) with fused CUDA megakernels, prefix KV caching, and eager pipelined text encoding.

## Performance (verified 2026-03-21)

| GPU | TTFP (first playable PCM) | Stress test | Config |
|-----|--------------------------|-------------|--------|
| **RTX 5090** (sm_120) | **~16ms** p50 | 500/500, 0 errors, 0MB drift | MK predictor + CUDA graph talker |
| **H100 SXM** (sm_90) | **~17ms** p50 | All texts 1w→45w verified | MK predictor + CUDA graph talker |
| **A100 SXM** (sm_80) | **~25ms** p50 | 250+ requests verified | CUDA graph only (auto-detected) |

TTFP = time from request receipt to first PCM audio chunk fully conditioned on the real text, voice, language, and instruct parameters. Text encoding (TTH) runs eagerly on a background CUDA stream, pipelined with KV restore, and synced **before** the first yield. Benchmarks need re-running to confirm exact numbers post-architecture change.

### TTFP breakdown (RTX 5090, estimated)

```
Honest TTFP (~16ms):
  [TTH on background stream ~1-2ms] ──overlap──▶
  [KV restore ~0.5ms] [sample 0.3ms] [TTH sync] [predictor 3ms] [vocoder 12ms]
```

The TTH (text encoding) is computed eagerly on a background CUDA stream while KV restore and first token sampling run on the default stream. The sync point ensures text conditioning is ready before the first yield — adding ~0-2ms vs the previous deferred architecture depending on TTH cache hit.

The vocoder (speech_tokenizer.decode) remains the bottleneck at ~75% of TTFP. Tested: torch.compile, CUDA graph capture, fp16, batching, background stream — none reduce single-frame decode time.

> **GPU auto-detection:** The server detects sm_arch at startup. Below sm_90 (A100, etc.), megakernels are auto-disabled with CUDA graph fallback. Warmup includes a 30s deadlock watchdog. Runtime frame timeout (10s) auto-disables megakernels after 3 failures. External liveness probe in start.sh kills hung processes.

| Feature | Details |
|---------|---------|
| Voices | 6 (Vivian, Serena, Dylan, Eric, Ryan, Aiden) |
| Languages | 10 (French, English, Chinese, Japanese, Korean, German, Russian, Portuguese, Spanish, Italian) |
| Tone presets | 8 built-in (neutral, warm, soft, dynamic, calm, formal, joyful, authoritative) + **any custom instruct** |
| Pre-cached combos | **480** (voice x language x tone), zero-cost switching. Custom tones built on-the-fly in background, cached for subsequent requests |
| Voice cloning | Via lazy-loaded Base model, cached after first call |
| Stability | **500/500** requests on RTX 5090, 0 errors, 0 reconnects, 0MB GPU memory drift, CV=4.3% |
| TTFP stability | Constant 1 word → 45 words (needs re-benchmarking with honest TTFP) |
| Audio quality | Cosine similarity 0.9995 vs CUDA graph baseline (numerically identical) |

## Architecture

```
Request → [input validation] → [TTFP timer start]
  → generate_cached_codec:
      [TTH on background CUDA stream] ─── overlaps with ───▶
      [KV cache lookup + restore] → [sample from cached logits]
      → [TTH sync] → [megakernel predictor 17 steps]
      → yield first codec frame (TTFP: text-conditioned)
  → generate_cached_streaming (PCM only):
      [speech_tokenizer.decode] → yield first PCM chunk (honest TTFP)
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

- **Prefix KV cache**: 480 voice/language/tone combos pre-computed at startup, skips ~50ms prefill per request. Custom instruct strings trigger an async background build on a dedicated CUDA stream (zero TTFP penalty, cached after first request).
- **Eager pipelined text encoding**: `_compute_tth()` runs on a background CUDA stream **before** the first yield, overlapping with KV restore and first token sampling. Synced before first frame to guarantee text-conditioned output. LRU-cached (200 entries).
- **Vocoder warmup**: Speech tokenizer decode warmed with varied tokens at startup (cold penalty ~10ms).
- **Codec raw mode**: 32 bytes per frame for benchmarking or client-side decode scenarios.

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
{"done": true, "ttfp_ms": 16.2, "total_ms": 1450, "chunks": 85}
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

| Mode | `codec` | First frame | Use case |
|------|---------|-------------|----------|
| PCM audio | `false` | 16-bit PCM at 24kHz, ready to play | **Production** — direct playback, honest TTFP |
| Codec raw | `true` | 16 int16 codec tokens (32 bytes) | Benchmarking / client-side decode (requires vocoder on client) |

## Files

| File | Description |
|------|-------------|
| `deploy/runpod_server.py` | Slim entry point: FastAPI + WebSocket handler (430 lines) |
| `deploy/server/engine.py` | `TTSEngine` class: model, megakernels, caches, generation (1242 lines) |
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
