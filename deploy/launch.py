"""
One-click TTS deployment.

Usage:
  python launch.py          # Create pod, wait for ready, print WS URL
  python launch.py --stop   # Stop and terminate the pod
  python launch.py --status # Check status

The network volume (qwen3-tts-vol) has everything pre-installed:
  - Model weights
  - 480 pre-computed KV caches (6 voices × 10 langs × 8 tones)
  - Server code + startup script
  - pip requirements
"""

import json
import os
import subprocess
import sys
import time

API_KEY = os.environ.get("RUNPOD_API_KEY", "")
VOLUME_ID = os.environ.get("RUNPOD_VOLUME_ID", "")
VOLUMES = {os.environ.get("RUNPOD_DC", "EU-NL-1"): VOLUME_ID} if VOLUME_ID else {}
IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
# Ordered by cost-effectiveness for this TTS workload:
# - All sm_90+ GPUs have megakernel support and similar TTFP (~16-17ms)
#   so we prefer cheapest first (RTX 5090 ~ $0.50/h vs B200 ~ $5/h)
# - sm_80 (A100) falls back to CUDA graph (~25ms), still usable
# - sm_89 (L40S, RTX 4090) also CUDA graph fallback
GPU_CHAIN = [
    # sm_90+ with megakernel (cheapest first, ~same performance)
    "NVIDIA GeForce RTX 5090",
    "NVIDIA H100 PCIe",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H200",
    "NVIDIA B200",
    # sm_80+ fallback (CUDA graph, no megakernel)
    "NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA L40S",
]
SSH_KEY = os.environ.get("RUNPOD_SSH_KEY", "")


def api(method, path, data=None):
    cmd = ["curl", "-s", "-X", method,
           "-H", f"Authorization: Bearer {API_KEY}",
           "-H", "Content-Type: application/json"]
    if data:
        cmd += ["-d", json.dumps(data)]
    cmd.append(f"https://rest.runpod.io/v1/{path}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return json.loads(r.stdout) if r.stdout.strip() else {}


def graphql(query):
    cmd = ["curl", "-s", "-H", "Content-Type: application/json",
           "-H", f"Authorization: Bearer {API_KEY}",
           "-d", json.dumps({"query": query}),
           "https://api.runpod.io/graphql"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return json.loads(r.stdout) if r.stdout.strip() else {}


def find_pod():
    d = graphql('{ myself { pods { id name desiredStatus costPerHr machine { gpuDisplayName } runtime { ports { ip publicPort privatePort } } } } }')
    for p in d.get("data", {}).get("myself", {}).get("pods", []):
        if "qwen3" in p.get("name", "").lower():
            return p
    return None


def get_urls(pod):
    ssh = tcp = None
    for p in (pod.get("runtime") or {}).get("ports") or []:
        if p.get("privatePort") == 22 and p.get("ip"):
            ssh = f"{p['ip']}:{p['publicPort']}"
        if p.get("privatePort") == 8000 and p.get("ip"):
            tcp = f"{p['ip']}:{p['publicPort']}"
    return ssh, tcp


def create_pod():
    if not API_KEY:
        print("ERROR: Set RUNPOD_API_KEY environment variable")
        sys.exit(1)
    balance = graphql('{ myself { clientBalance } }')["data"]["myself"]["clientBalance"]
    print(f"Balance: ${balance:.2f}")

    REPO_URL = "https://github.com/Imtoocompedidiv/qwen-tts-turbo.git"
    # Self-contained startup: clone repo then run start.sh
    STARTUP_CMD = [
        "bash", "-c",
        f"git clone --depth 1 {REPO_URL} /workspace/qwen-tts-turbo && "
        f"bash /workspace/qwen-tts-turbo/deploy/start.sh",
    ]

    def try_create(dc, vol_id):
        for gpu in GPU_CHAIN:
            pod_config = {
                "name": "qwen3-tts",
                "imageName": IMAGE,
                "gpuTypeIds": [gpu],
                "containerDiskInGb": 50,
                "ports": ["8000/tcp", "22/tcp"],
                "volumeMountPath": "/workspace",
                "dockerStartCmd": STARTUP_CMD,
                "env": {"PUBLIC_KEY": SSH_KEY},
            }
            if vol_id:
                pod_config["networkVolumeId"] = vol_id
            if dc:
                pod_config["dataCenterIds"] = [dc]
            result = api("POST", "pods", pod_config)
            if "id" in result:
                gpu_name = result.get("machine", {}).get("gpuTypeId", gpu)
                cost = result.get("costPerHr", "?")
                print(f"Pod {result['id']} created: {gpu_name} in {dc} @ ${cost}/hr")
                return result["id"]
            elif result:
                # Log API errors (schema mismatch, out of stock, etc.)
                err = result[0].get("error", result) if isinstance(result, list) else result
                print(f"    {gpu}: {err}")
        return None

    # Try each DC (with or without volume)
    if VOLUMES:
        for dc, vol_id in VOLUMES.items():
            print(f"Trying {dc}...")
            pod_id = try_create(dc, vol_id)
            if pod_id:
                return pod_id
            print(f"  No GPU in {dc}")
    else:
        # No volume configured — deploy without volume (fully self-contained)
        print("No volume configured. Deploying self-contained pod...")
        pod_id = try_create("", None)
        if pod_id:
            return pod_id

    # Retry loop
    print("No GPU available. Retrying every 30s...")
    for attempt in range(20):
        time.sleep(30)
        if VOLUMES:
            for dc, vol_id in VOLUMES.items():
                pod_id = try_create(dc, vol_id)
                if pod_id:
                    return pod_id
        else:
            pod_id = try_create("", None)
            if pod_id:
                return pod_id
        print(f"  Retry {attempt+1}/20...")

    print("ERROR: No GPU available after 10 minutes")
    sys.exit(1)


def wait_ready(pod_id):
    print("Waiting for server...")
    for i in range(180):  # 180 × 5s = 15 minutes (cold start without volume)
        pod = graphql(f'{{ pod(input: {{podId: "{pod_id}"}}) {{ desiredStatus runtime {{ ports {{ ip publicPort privatePort }} }} }} }}')
        pod_data = pod.get("data", {}).get("pod", {})
        _, tcp = get_urls(pod_data)

        if tcp:
            try:
                r = subprocess.run(
                    ["curl", "-s", "--max-time", "3", f"http://{tcp}/health"],
                    capture_output=True, text=True, timeout=5
                )
                if '"ok"' in r.stdout:
                    health = json.loads(r.stdout)
                    print(f"\nREADY in {i*5}s")
                    print(f"  GPU: {health.get('gpu')}")
                    print(f"  Combos cached: {health.get('combos')}")
                    print(f"  TTFP: {health.get('ttfp_ms')}ms")
                    print(f"\n  WebSocket: ws://{tcp}/ws/tts")
                    print(f"  HTTP:      http://{tcp}/v1/audio/speech")
                    print(f"  Health:    http://{tcp}/health")
                    return tcp
            except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
                pass

        if i % 6 == 0 and i > 0:
            print(f"  [{i*5}s] still starting...")
        time.sleep(5)

    print("ERROR: Server did not start in 15 minutes")
    sys.exit(1)


def stop():
    pod = find_pod()
    if pod:
        api("DELETE", f"pods/{pod['id']}")
        print(f"Pod {pod['id']} terminated")
    else:
        print("No qwen3-tts pod found")
    # Show balance
    d = graphql('{ myself { clientBalance } }')
    print(f"Balance: ${d['data']['myself']['clientBalance']:.2f}")


def status():
    pod = find_pod()
    if not pod:
        print("No qwen3-tts pod running")
        d = graphql('{ myself { clientBalance networkVolumes { id name dataCenterId } } }')
        me = d["data"]["myself"]
        print(f"Balance: ${me['clientBalance']:.2f}")
        print(f"Volume: {me['networkVolumes']}")
        return

    ssh, tcp = get_urls(pod)
    print(f"Pod: {pod['id']} ({pod.get('desiredStatus')})")
    print(f"GPU: {pod.get('machine', {}).get('gpuDisplayName')}")
    print(f"Cost: ${pod.get('costPerHr')}/hr")
    if tcp:
        try:
            r = subprocess.run(
                ["curl", "-s", "--max-time", "3", f"http://{tcp}/health"],
                capture_output=True, text=True, timeout=5
            )
            if r.stdout.strip():
                h = json.loads(r.stdout)
                print(f"Status: READY ({h.get('combos')} combos, {h.get('ttfp_ms')}ms)")
                print(f"WS: ws://{tcp}/ws/tts")
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
            print("Status: starting...")


if __name__ == "__main__":
    if "--stop" in sys.argv:
        stop()
    elif "--status" in sys.argv:
        status()
    else:
        pod = find_pod()
        if pod:
            print(f"Pod already exists: {pod['id']}")
            _, tcp = get_urls(pod)
            if tcp:
                print(f"WS: ws://{tcp}/ws/tts")
            sys.exit(0)
        pod_id = create_pod()
        wait_ready(pod_id)
