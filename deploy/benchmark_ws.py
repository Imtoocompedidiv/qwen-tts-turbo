"""
WebSocket TTS latency benchmark.
Measures TTFP over a persistent connection (no TLS/TCP handshake per request).

Usage:
  python benchmark_ws.py ws://IP:PORT/ws/tts
  python benchmark_ws.py wss://pod-id-8000.proxy.runpod.net/ws/tts
"""

import asyncio
import json
import struct
import sys
import time

try:
    import websockets
except ImportError:
    print("pip install websockets")
    sys.exit(1)


TEXTS = [
    ("Bonjour.", "1w"),
    ("Oui, bien sur.", "3w"),
    ("Laissez-moi verifier cette information pour vous.", "7w"),
    ("D'apres les informations dont je dispose, votre commande a ete expediee hier "
     "et devrait arriver dans deux jours.", "18w"),
    ("Merci pour votre patience. J'ai examine votre dossier en detail et je constate "
     "que le remboursement a bien ete initie le quinze mars dernier. Cependant, en "
     "raison d'un delai de traitement bancaire, le montant n'apparaitra sur votre "
     "releve que dans cinq a sept jours ouvres.", "45w"),
]

N_RUNS = 5


async def benchmark(url: str):
    print(f"Connecting to {url}...")
    async with websockets.connect(url, max_size=10 * 1024 * 1024) as ws:
        # Warmup: one throwaway request to stabilize the connection
        await ws.send(json.dumps({"input": "Test.", "voice": "Vivian", "language": "French"}))
        while True:
            msg = await ws.recv()
            if isinstance(msg, str):
                d = json.loads(msg)
                if d.get("done"):
                    break

        print("=" * 70)
        print(f"WebSocket TTS Benchmark — {N_RUNS} runs per text")
        print(f"Connection: {url}")
        print("=" * 70)

        for text, label in TEXTS:
            ttfps_client = []
            ttfps_server = []
            totals_client = []

            for run in range(N_RUNS):
                req = json.dumps({
                    "input": text,
                    "voice": "Vivian",
                    "language": "French",
                    "chunk_size": 1,
                })
                t0 = time.perf_counter()
                await ws.send(req)

                first_chunk = True
                srv_ttfp = None

                while True:
                    msg = await ws.recv()
                    if isinstance(msg, bytes):
                        if first_chunk:
                            ttfb = (time.perf_counter() - t0) * 1000
                            ttfps_client.append(ttfb)
                            first_chunk = False
                            # Parse server TTFP from header
                            if len(msg) >= 4:
                                hl = struct.unpack("<I", msg[:4])[0]
                                if hl < 200:
                                    try:
                                        hdr = json.loads(msg[4:4 + hl])
                                        srv_ttfp = hdr.get("ttfp_ms")
                                    except:
                                        pass
                    elif isinstance(msg, str):
                        d = json.loads(msg)
                        if d.get("done"):
                            total = (time.perf_counter() - t0) * 1000
                            totals_client.append(total)
                            if srv_ttfp is not None:
                                ttfps_server.append(srv_ttfp)
                            break

            ttfps_client.sort()
            totals_client.sort()
            p50_ttfp = ttfps_client[len(ttfps_client) // 2]
            p50_total = totals_client[len(totals_client) // 2]
            min_ttfp = ttfps_client[0]

            srv_str = ""
            if ttfps_server:
                ttfps_server.sort()
                s = ttfps_server[len(ttfps_server) // 2]
                oh = p50_ttfp - s
                srv_str = f"  srv={s:.0f}ms  overhead={oh:.0f}ms"

            print(
                f"  [{label:>4s}] TTFP p50={p50_ttfp:5.0f}ms  min={min_ttfp:5.0f}ms  "
                f"total={p50_total:5.0f}ms{srv_str}"
            )
            all_str = " ".join(f"{t:.0f}" for t in ttfps_client)
            print(f"         all=[{all_str}]")

        # === Codec-raw mode benchmark ===
        print()
        print("=" * 70)
        print(f"WebSocket CODEC-RAW Benchmark (no speech decode) — {N_RUNS} runs")
        print("=" * 70)

        for text, label in TEXTS:
            ttfps_client = []
            ttfps_server = []

            for run in range(N_RUNS):
                req = json.dumps({
                    "input": text,
                    "voice": "Vivian",
                    "language": "French",
                    "chunk_size": 1,
                    "codec": True,
                })
                t0 = time.perf_counter()
                await ws.send(req)

                first_chunk = True
                srv_ttfp = None

                while True:
                    msg = await ws.recv()
                    if isinstance(msg, bytes):
                        if first_chunk:
                            ttfb = (time.perf_counter() - t0) * 1000
                            ttfps_client.append(ttfb)
                            first_chunk = False
                            if len(msg) >= 4:
                                hl = struct.unpack("<I", msg[:4])[0]
                                if hl < 200:
                                    try:
                                        hdr = json.loads(msg[4:4 + hl])
                                        srv_ttfp = hdr.get("ttfp_ms")
                                    except:
                                        pass
                    elif isinstance(msg, str):
                        d = json.loads(msg)
                        if d.get("done"):
                            if srv_ttfp is not None:
                                ttfps_server.append(srv_ttfp)
                            break

            ttfps_client.sort()
            p50_ttfp = ttfps_client[len(ttfps_client) // 2]
            min_ttfp = ttfps_client[0]

            srv_str = ""
            if ttfps_server:
                ttfps_server.sort()
                s = ttfps_server[len(ttfps_server) // 2]
                srv_str = f"  srv={s:.0f}ms  overhead={p50_ttfp - s:.0f}ms"

            print(
                f"  [{label:>4s}] TTFP p50={p50_ttfp:5.0f}ms  min={min_ttfp:5.0f}ms{srv_str}"
            )

        # === Multi-tone benchmark ===
        print()
        print("=" * 70)
        print(f"WebSocket MULTI-TONE Benchmark (codec raw) — {N_RUNS} runs")
        print("=" * 70)

        tones = [
            ("", "neutre"),
            ("Parle d'un ton chaleureux et professionnel", "chaleureux"),
            ("Voix douce et rassurante", "doux"),
            ("Ton dynamique et enthousiaste", "dynamique"),
            ("Parle calmement avec empathie", "empathique"),
        ]
        test_text = "Bonjour, comment puis-je vous aider ?"

        for instruct, label in tones:
            ttfps_client = []
            ttfps_server = []

            for run in range(N_RUNS):
                req = json.dumps({
                    "input": test_text,
                    "voice": "Vivian",
                    "language": "French",
                    "instruct": instruct,
                    "chunk_size": 1,
                    "codec": True,
                })
                t0 = time.perf_counter()
                await ws.send(req)

                first_chunk = True
                srv_ttfp = None

                while True:
                    msg = await ws.recv()
                    if isinstance(msg, bytes):
                        if first_chunk:
                            ttfb = (time.perf_counter() - t0) * 1000
                            ttfps_client.append(ttfb)
                            first_chunk = False
                            if len(msg) >= 4:
                                hl = struct.unpack("<I", msg[:4])[0]
                                if hl < 200:
                                    try:
                                        hdr = json.loads(msg[4:4 + hl])
                                        srv_ttfp = hdr.get("ttfp_ms")
                                    except:
                                        pass
                    elif isinstance(msg, str):
                        d = json.loads(msg)
                        if d.get("done"):
                            if srv_ttfp is not None:
                                ttfps_server.append(srv_ttfp)
                            break

            ttfps_client.sort()
            p50 = ttfps_client[len(ttfps_client) // 2]
            srv_str = ""
            if ttfps_server:
                ttfps_server.sort()
                srv_str = f"  srv={ttfps_server[len(ttfps_server)//2]:.0f}ms"

            print(f"  [{label:>12s}] TTFP p50={p50:5.0f}ms  min={ttfps_client[0]:5.0f}ms{srv_str}")

        print()
        print("=" * 70)
        print("REFERENCE POINTS")
        print("=" * 70)
        print("  Previous best (PCM mode):         ~75ms TTFP")
        print("  Google Cloud TTS Neural2 (batch): ~336ms")
        print("  DashScope Flash (SSE):            ~1500ms")


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8000/ws/tts"
    asyncio.run(benchmark(url))
