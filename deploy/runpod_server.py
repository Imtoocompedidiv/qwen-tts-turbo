"""Qwen3-TTS server — production hardened, minimum TTFP.

Slim entry point: creates TTSEngine + ServerMonitor, wires up FastAPI.
"""

import gc

# Disable GC during generation to avoid pauses
gc.disable()
gc.collect()

import asyncio
import base64
import concurrent.futures
import hmac
import json
import os
import struct
import tempfile
import time

import numpy as np
import torch

from server.engine import TTSEngine
from server.monitoring import ServerMonitor

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# ── Create engine and monitor ──────────────────────────────────────────
engine = TTSEngine()
monitor = ServerMonitor(engine)

app = FastAPI(title="Qwen3-TTS")

API_KEY = os.environ.get("TTS_API_KEY", "")
_gc_interval = 100

# Thread pool for async generation (run_in_executor)
_gen_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="gen"
)
_GEN_SENTINEL = object()  # signals StopIteration across executor boundary


def _safe_next(gen_iter):
    """next() wrapper that returns _GEN_SENTINEL instead of raising StopIteration.

    StopIteration doesn't propagate cleanly through run_in_executor,
    so we convert it to a sentinel value.
    """
    try:
        return next(gen_iter)
    except StopIteration:
        return _GEN_SENTINEL


@app.websocket("/ws/tts")
async def websocket_tts(ws: WebSocket):
    await ws.accept()
    disconnected = False

    # Auth check (first message if API_KEY is set)
    if API_KEY:
        try:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            if not hmac.compare_digest(msg.get("auth", ""), API_KEY):
                await ws.send_text(json.dumps({"error": "unauthorized"}))
                await ws.close()
                return
            await ws.send_text(json.dumps({"auth": "ok"}))
        except (WebSocketDisconnect, Exception):
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
        except (json.JSONDecodeError, ValueError):
            try:
                await ws.send_text(json.dumps({"error": "invalid json"}))
            except (WebSocketDisconnect, Exception):
                break
            continue

        text = req.get("input", "")
        voice = req.get("voice", "Vivian")
        language = req.get("language", "French")
        instruct = req.get("instruct", "")
        cs = max(1, min(16, int(req.get("chunk_size", engine.CHUNK_SIZE))))
        codec_mode = req.get("codec", False)

        # Voice cloning params
        ref_audio_b64 = req.get("ref_audio")  # base64-encoded WAV
        ref_text = req.get("ref_text", "")
        clone_id = req.get("clone_id", "")  # reuse cached voice prompt

        # Input validation
        MAX_TEXT_LEN = 10000
        MAX_AUDIO_B64 = 10 * 1024 * 1024  # 10MB
        if not text:
            try:
                await ws.send_text(json.dumps({"error": "empty input"}))
            except (WebSocketDisconnect, Exception):
                break
            continue
        if len(text) > MAX_TEXT_LEN:
            try:
                await ws.send_text(
                    json.dumps(
                        {
                            "error": f"text too long ({len(text)} > {MAX_TEXT_LEN})"
                        }
                    )
                )
            except (WebSocketDisconnect, Exception):
                break
            continue
        if ref_audio_b64 and len(ref_audio_b64) > MAX_AUDIO_B64:
            try:
                await ws.send_text(json.dumps({"error": "ref_audio too large"}))
            except (WebSocketDisconnect, Exception):
                break
            continue

        chunk_count = 0
        ttfp_ms = 0
        sr = 24000
        monitor.request_count += 1

        # TTFP timer — no cuda sync here; generate_cached_codec syncs
        # TTH before first yield, which is the honest measurement point.
        t0_req = time.perf_counter()

        # Select generator
        if ref_audio_b64 or clone_id:
            # Voice cloning mode
            if ref_audio_b64:
                audio_bytes = base64.b64decode(ref_audio_b64)
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                ref_path = tmp.name
                try:
                    tmp.write(audio_bytes)
                    tmp.close()
                except Exception:
                    tmp.close()
                    os.unlink(ref_path)
                    raise
                cid = clone_id or f"_auto_{monitor.request_count}"
                engine._clone_cache[cid] = ref_path
                engine.build_clone_prefill(cid, ref_path, ref_text, language)
                clone_model = engine._get_clone_model()
                gen = clone_model.generate_voice_clone_streaming(
                    text=text,
                    language=language,
                    ref_audio=ref_path,
                    ref_text=ref_text,
                    chunk_size=cs,
                )
                codec_mode = False
            elif clone_id and clone_id in engine._clone_prefill:
                gen = engine.generate_clone_cached_codec(
                    text, clone_id, language
                )
                codec_mode = True
            elif clone_id and clone_id in engine._clone_cache:
                engine.build_clone_prefill(
                    clone_id,
                    engine._clone_cache[clone_id],
                    ref_text,
                    language,
                )
                gen = engine.generate_clone_cached_codec(
                    text, clone_id, language
                )
                codec_mode = True
            else:
                try:
                    await ws.send_text(
                        json.dumps({"error": "clone_id not found"})
                    )
                except (WebSocketDisconnect, Exception):
                    break
                continue
        elif codec_mode and engine.USE_CACHE:
            gen = engine.generate_cached_codec(
                text, voice, language, instruct, cs
            )
        elif engine.USE_CACHE:
            gen = engine.generate_cached_streaming(
                text, voice, language, instruct, cs
            )
        else:
            gen = engine.model.generate_custom_voice_streaming(
                text=text,
                language=language,
                speaker=voice,
                instruct=instruct or "",
                chunk_size=cs,
            )

        try:
            monitor.generation_active[0] = True
            monitor.last_frame_time[0] = time.monotonic()
            req_deadline = time.monotonic() + monitor.GENERATION_TIMEOUT
            loop = asyncio.get_event_loop()

            gen_iter = iter(gen)

            # Couche 1: async frame loop with hard timeout on next(gen).
            # run_in_executor moves the blocking next() call to a thread.
            # wait_for enforces the deadline BEFORE Python blocks.
            # If CUDA deadlocks, wait_for raises TimeoutError while the
            # stuck thread is abandoned (watchdog couche 2 handles cleanup).
            while True:
                if time.monotonic() > req_deadline:
                    raise RuntimeError(
                        f"Request timeout ({monitor.GENERATION_TIMEOUT}s)"
                    )

                try:
                    item = await asyncio.wait_for(
                        loop.run_in_executor(
                            _gen_pool, _safe_next, gen_iter
                        ),
                        timeout=monitor.FRAME_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    monitor.mk_failure_count += 1
                    if (
                        engine.mk_predictor is not None
                        and monitor.mk_failure_count
                        >= monitor.MK_FAILURE_THRESHOLD
                    ):
                        monitor._auto_disable_megakernel()
                    raise RuntimeError(
                        f"Frame timeout ({monitor.FRAME_TIMEOUT}s), probable deadlock "
                        f"(mk_failures: {monitor.mk_failure_count})"
                    )

                if item is _GEN_SENTINEL:
                    break  # Generator exhausted

                monitor.last_frame_time[0] = time.monotonic()
                chunk_count += 1

                if codec_mode:
                    raw_bytes = (
                        item.cpu().numpy().astype(np.int16).tobytes()
                    )
                    if chunk_count == 1:
                        ttfp_ms = (time.perf_counter() - t0_req) * 1000
                        header = json.dumps(
                            {"ttfp_ms": round(ttfp_ms, 1), "codec": True}
                        ).encode()
                        await ws.send_bytes(
                            struct.pack("<I", len(header)) + header + raw_bytes
                        )
                    else:
                        await ws.send_bytes(raw_bytes)
                else:
                    audio_chunk = (
                        item[0] if isinstance(item, tuple) else item
                    )
                    if (
                        isinstance(item, tuple)
                        and len(item) > 1
                        and isinstance(item[1], int)
                    ):
                        sr = item[1]
                    pcm = (
                        (np.array(audio_chunk) * 32767)
                        .astype(np.int16)
                        .tobytes()
                    )
                    if chunk_count == 1:
                        ttfp_ms = (time.perf_counter() - t0_req) * 1000
                        header = json.dumps(
                            {"ttfp_ms": round(ttfp_ms, 1), "sr": sr}
                        ).encode()
                        await ws.send_bytes(
                            struct.pack("<I", len(header)) + header + pcm
                        )
                    else:
                        await ws.send_bytes(pcm)

                if chunk_count % 10 == 0:
                    await asyncio.sleep(0)

            monitor.generation_active[0] = False
            total_ms = (time.perf_counter() - t0_req) * 1000
            monitor.record_ttfp(ttfp_ms)
            await ws.send_text(
                json.dumps(
                    {
                        "done": True,
                        "ttfp_ms": round(ttfp_ms, 1),
                        "total_ms": round(total_ms, 1),
                        "chunks": chunk_count,
                    }
                )
            )

        except (WebSocketDisconnect, ConnectionError):
            disconnected = True
            monitor.generation_active[0] = False
        except RuntimeError as rte:
            monitor.generation_active[0] = False
            monitor.error_count += 1
            print(f"RUNTIME ERROR: {rte}")
            try:
                await ws.send_text(
                    json.dumps({"error": str(rte)[:200], "done": True})
                )
            except (WebSocketDisconnect, Exception):
                disconnected = True
        except Exception as exc:
            monitor.generation_active[0] = False
            monitor.error_count += 1
            import traceback

            err_msg = f"{type(exc).__name__}: {str(exc)[:200]}"
            traceback.print_exc()
            try:
                await ws.send_text(
                    json.dumps({"error": err_msg, "done": True})
                )
            except (WebSocketDisconnect, Exception):
                disconnected = True

        # Periodic GC to prevent memory creep
        if monitor.request_count % _gc_interval == 0:
            gc.collect()
            torch.cuda.empty_cache()


@app.get("/generate")
async def generate_wav(
    text: str,
    voice: str = "Vivian",
    language: str = "French",
    instruct: str = "",
):
    """Generate a WAV file from text. Used for sample generation."""
    import io
    import soundfile as sf
    from fastapi.responses import Response

    audio_chunks = []
    for chunk_data in engine.generate_cached_streaming(
        text, voice, language, instruct, chunk_size=1
    ):
        audio_chunks.append(
            chunk_data[0] if isinstance(chunk_data, tuple) else chunk_data
        )
        sr = (
            chunk_data[1]
            if isinstance(chunk_data, tuple) and len(chunk_data) > 1
            else 24000
        )
    audio = np.concatenate(audio_chunks)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/wav")


@app.get("/health")
def health():
    return {
        "status": monitor.health_status(),
        "gpu": torch.cuda.get_device_name(0),
        "model": engine.MODEL_SIZE,
        "combos": len(engine.prefill_cache) if engine.USE_CACHE else 0,
        "ttfp_ms": round(engine.cached_ttfp_ms, 1),
        "megakernel_predictor": engine.mk_predictor is not None,
        "megakernel_talker": engine.mk_talker is not None,
        "requests": monitor.request_count,
        "errors": monitor.error_count,
        "mk_failures": monitor.mk_failure_count,
    }


@app.get("/health/detail")
def health_detail():
    return {
        "status": monitor.health_status(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_arch": f"sm_{engine.gpu_arch}",
        "gpu_mem_mb": round(torch.cuda.memory_allocated() / 1024 / 1024),
        "gpu_mem_peak_mb": round(
            torch.cuda.max_memory_allocated() / 1024 / 1024
        ),
        "combos": len(engine.prefill_cache) if engine.USE_CACHE else 0,
        "ttfp_ms": round(engine.cached_ttfp_ms, 1),
        "ttfp_live": monitor.ttfp_percentiles(),
        "requests": monitor.request_count,
        "errors": monitor.error_count,
        "mk_failures": monitor.mk_failure_count,
        "degraded": monitor.server_degraded,
        "generation_active": monitor.generation_active[0],
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
        app,
        host="0.0.0.0",
        port=8000,
        ws="websockets",
        log_level="warning",
        ssl_certfile=ssl_cert,
        ssl_keyfile=ssl_key,
    )
