#!/bin/bash
set -e

# ── Self-contained startup with supervised restart + liveness probe ──
# Clones repo, installs deps, starts server.
# Three layers of failure recovery:
#   1. Server-internal: async timeout + auto-fallback to CUDA graph
#   2. Server-internal: watchdog thread → os._exit(1) on stuck generation
#   3. This script: liveness probe → kill -9 on unresponsive process
# On any crash, waits 5s and restarts (up to MAX_RESTARTS).

REPO_DIR="/workspace/qwen-tts-turbo"
REPO_URL="https://github.com/Imtoocompedidiv/qwen-tts-turbo.git"
MAX_RESTARTS=10
RESTART_DELAY=5
LIVENESS_INTERVAL=5        # seconds between health checks
LIVENESS_FAIL_THRESHOLD=6  # consecutive failures before kill (= 30s)
LIVENESS_START_DELAY=120   # wait for model load before checking

# Clone repo if not present
if [ ! -f "$REPO_DIR/deploy/runpod_server.py" ]; then
    echo "Cloning qwen-tts-turbo..."
    git clone --depth 1 "$REPO_URL" "$REPO_DIR"
fi

# Install dependencies
python3 -c "import faster_qwen3_tts" 2>/dev/null || pip install -q faster-qwen3-tts qwen-tts
pip install -q ninja soundfile websockets 2>/dev/null

# Pre-flight checks
echo "Pre-flight checks..."
python3 -c "
import torch
assert torch.cuda.is_available(), 'No CUDA GPU'
cc = torch.cuda.get_device_capability()
name = torch.cuda.get_device_name(0)
mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
print(f'  GPU: {name} (sm_{cc[0]}{cc[1]}, {mem:.0f}GB)')
assert mem >= 16, f'Need 16GB+ VRAM, got {mem:.0f}GB'
" || { echo "FATAL: Pre-flight failed"; exit 1; }

export MODEL_SIZE=1.7B CHUNK_SIZE=1 USE_CACHE=1 USE_MEGAKERNEL=1 USE_TALKER_MK=0
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

# ── Couche 3: external liveness probe ─────────────────────────────────
# Runs in background. If /health doesn't respond for 30s straight,
# kill -9 the server. The restart loop below picks up and relaunches.
_liveness_monitor() {
    local server_pid=$1
    sleep "$LIVENESS_START_DELAY"
    local fails=0
    while kill -0 "$server_pid" 2>/dev/null; do
        if curl -sf --max-time 5 http://localhost:8000/health > /dev/null 2>&1; then
            fails=0
        else
            fails=$((fails + 1))
            if [ $fails -ge $LIVENESS_FAIL_THRESHOLD ]; then
                echo "LIVENESS: /health failed ${fails}x (${LIVENESS_INTERVAL}s intervals). Killing PID $server_pid."
                kill -9 "$server_pid" 2>/dev/null
                return
            fi
        fi
        sleep "$LIVENESS_INTERVAL"
    done
}

# ── Supervised restart loop ───────────────────────────────────────────
restarts=0
while [ $restarts -lt $MAX_RESTARTS ]; do
    echo "Starting server (attempt $((restarts + 1))/$MAX_RESTARTS)..."
    python3 -u "$REPO_DIR/deploy/runpod_server.py" &
    server_pid=$!

    # Start liveness monitor in background
    _liveness_monitor "$server_pid" &
    monitor_pid=$!

    # Wait for server to exit (crash, deadlock watchdog, or liveness kill)
    wait "$server_pid" 2>/dev/null
    exit_code=$?

    # Stop liveness monitor
    kill "$monitor_pid" 2>/dev/null
    wait "$monitor_pid" 2>/dev/null

    # Clean exit (code 0) = intentional stop
    if [ $exit_code -eq 0 ]; then
        echo "Server exited cleanly."
        break
    fi

    restarts=$((restarts + 1))
    echo "Server exited with code $exit_code. Restarting in ${RESTART_DELAY}s... ($restarts/$MAX_RESTARTS)"

    # Clear CUDA state between restarts
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep $RESTART_DELAY
done

if [ $restarts -ge $MAX_RESTARTS ]; then
    echo "FATAL: Server crashed $MAX_RESTARTS times. Giving up."
    exit 1
fi
