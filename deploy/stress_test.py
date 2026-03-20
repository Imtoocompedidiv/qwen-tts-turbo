"""
Production stress test — 500 requests, variance analysis, memory monitoring.
Usage: python stress_test.py ws://IP:PORT/ws/tts [http://IP:PORT]
"""
import asyncio
import json
import struct
import sys
import time
import statistics
import urllib.request

try:
    import websockets
except ImportError:
    print("pip install websockets")
    sys.exit(1)

N = 500
TEXTS = [
    "Bonjour.", "Oui.", "Merci.", "Un instant.",
    "Laissez-moi verifier.", "Merci pour votre patience.",
    "Je comprends et je suis la pour vous aider.",
    "Votre dossier a bien ete mis a jour avec les nouvelles informations.",
    "D'apres les informations, votre commande a ete expediee hier.",
    "Je vais transmettre votre demande au service concerne pour un traitement rapide.",
]
VOICES = ["Vivian", "Serena", "Dylan", "Eric", "Ryan", "Aiden"]
TONES = ["", "Voix douce et rassurante", "Ton dynamique et enthousiaste",
         "Parle calmement avec empathie", "Ton s\u00e9rieux et formel"]
LANGUAGES = ["French", "English", "Spanish", "German"]


async def connect_with_retry(url, max_retries=5):
    """Connect to WebSocket with auto-retry."""
    for attempt in range(max_retries):
        try:
            ws = await websockets.connect(url, max_size=10*1024*1024, ping_timeout=300)
            # Prime
            await ws.send(json.dumps({"input": "Test.", "voice": "Vivian",
                                       "language": "French", "codec": True}))
            while True:
                msg = await ws.recv()
                if isinstance(msg, str) and json.loads(msg).get("done"):
                    break
            return ws
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Connect retry {attempt+1}: {e}")
                await asyncio.sleep(2)
            else:
                raise


async def stress_test(ws_url, http_url=None):
    print(f"Connecting to {ws_url}...")
    ws = await connect_with_retry(ws_url)

    mem_start = None
    if http_url:
        try:
            r = urllib.request.urlopen(f"{http_url}/health/detail", timeout=5)
            d = json.loads(r.read())
            mem_start = d.get("gpu_mem_mb")
        except:
            pass

    print(f"\n{'='*70}")
    print(f"STRESS TEST: {N} requests, {len(VOICES)} voices, {len(LANGUAGES)} langs, {len(TONES)} tones")
    print(f"{'='*70}\n")

    all_srv = []
    all_cli = []
    errors = 0
    reconnects = 0
    batch = 50

    for i in range(N):
        text = TEXTS[i % len(TEXTS)]
        voice = VOICES[i % len(VOICES)]
        lang = LANGUAGES[i % len(LANGUAGES)]
        tone = TONES[i % len(TONES)]

        try:
            t0 = time.perf_counter()
            await ws.send(json.dumps({
                "input": text, "voice": voice, "language": lang,
                "instruct": tone, "codec": True, "chunk_size": 1,
            }))
            first = True
            srv = None
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=60)
                if isinstance(msg, bytes):
                    if first:
                        all_cli.append((time.perf_counter() - t0) * 1000)
                        first = False
                        hl = struct.unpack("<I", msg[:4])[0]
                        if hl < 200:
                            try:
                                srv = json.loads(msg[4:4+hl]).get("ttfp_ms")
                            except:
                                pass
                elif isinstance(msg, str):
                    if json.loads(msg).get("done"):
                        if srv is not None:
                            all_srv.append(srv)
                        break
        except (websockets.exceptions.ConnectionClosed, asyncio.TimeoutError) as e:
            errors += 1
            if errors <= 10:
                print(f"  ERR {i}: {type(e).__name__}")
            try:
                await ws.close()
            except:
                pass
            try:
                ws = await connect_with_retry(ws_url, max_retries=3)
                reconnects += 1
            except:
                print(f"  FATAL: Cannot reconnect")
                break
        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"  ERR {i}: {e}")

        if (i + 1) % batch == 0 and all_cli:
            rc = all_cli[-min(batch, len(all_cli)):]
            rs = all_srv[-min(batch, len(all_srv)):] if all_srv else [0]
            rc_s = sorted(rc)
            rs_s = sorted(rs)
            cv = statistics.stdev(rc) / statistics.mean(rc) * 100 if len(rc) > 1 else 0
            print(f"  [{i+1:3d}/{N}] srv={rs_s[len(rs_s)//2]:.1f}ms  "
                  f"cli p50={rc_s[len(rc_s)//2]:.0f}ms "
                  f"[{rc_s[0]:.0f}-{rc_s[-1]:.0f}] "
                  f"CV={cv:.0f}% err={errors} reconn={reconnects}")

    mem_end = None
    if http_url:
        try:
            r = urllib.request.urlopen(f"{http_url}/health/detail", timeout=5)
            d = json.loads(r.read())
            mem_end = d.get("gpu_mem_mb")
        except:
            pass

    if not all_cli:
        print("No successful requests.")
        return

    s = sorted(all_cli)
    s2 = sorted(all_srv) if all_srv else [0]
    cv_cli = statistics.stdev(all_cli) / statistics.mean(all_cli) * 100 if len(all_cli) > 1 else 0
    cv_srv = statistics.stdev(all_srv) / statistics.mean(all_srv) * 100 if len(all_srv) > 1 else 0

    print(f"\n{'='*70}")
    print(f"RESULTS: {len(all_cli)}/{N} ok, {errors} errors, {reconnects} reconnects")
    print(f"{'='*70}")
    print(f"  Client TTFP:")
    print(f"    min={s[0]:.0f}  p10={s[int(len(s)*0.1)]:.0f}  p25={s[len(s)//4]:.0f}  "
          f"p50={s[len(s)//2]:.0f}  p75={s[3*len(s)//4]:.0f}  "
          f"p90={s[int(len(s)*0.9)]:.0f}  p99={s[int(len(s)*0.99)]:.0f}  max={s[-1]:.0f}ms")
    print(f"    mean={statistics.mean(all_cli):.0f}ms  stdev={statistics.stdev(all_cli):.0f}ms  CV={cv_cli:.1f}%")
    if all_srv:
        print(f"  Server TTFP:")
        print(f"    min={s2[0]:.1f}  p50={s2[len(s2)//2]:.1f}  p99={s2[int(len(s2)*0.99)]:.1f}  max={s2[-1]:.1f}ms")
        print(f"    CV={cv_srv:.1f}%")

    if len(all_srv) >= 50:
        chunks = [all_srv[i:i+50] for i in range(0, len(all_srv), 50)]
        p50s = [sorted(c)[len(c)//2] for c in chunks if len(c) >= 10]
        if len(p50s) >= 2:
            drift = p50s[-1] - p50s[0]
            print(f"  Stability:")
            print(f"    p50 per batch: {' -> '.join(f'{p:.1f}' for p in p50s)}ms")
            print(f"    Drift: {drift:+.1f}ms ({'STABLE' if abs(drift) < 2 else 'DRIFT'})")

    if mem_start and mem_end:
        leak = mem_end - mem_start
        print(f"  Memory:")
        print(f"    Start={mem_start}MB  End={mem_end}MB  Delta={leak:+.0f}MB "
              f"({'OK' if abs(leak) < 100 else 'LEAK'})")

    try:
        await ws.close()
    except:
        pass


if __name__ == "__main__":
    ws_url = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8000/ws/tts"
    http_url = sys.argv[2] if len(sys.argv) > 2 else None
    # Auto-derive HTTP from WS URL
    if not http_url:
        http_url = ws_url.replace("ws://", "http://").replace("/ws/tts", "")
    asyncio.run(stress_test(ws_url, http_url))
