#!/bin/bash
set -e

# ── Self-contained startup with supervised restart + liveness probe ──

REPO_DIR="/workspace/qwen-tts-turbo"
REPO_URL="https://github.com/Imtoocompedidiv/qwen-tts-turbo.git"
MAX_RESTARTS=10
RESTART_DELAY=5
LIVENESS_INTERVAL=5
LIVENESS_FAIL_THRESHOLD=6
LIVENESS_START_DELAY=120

# Clone or update repo
if [ ! -f "$REPO_DIR/deploy/runpod_server.py" ]; then
    echo "Cloning qwen-tts-turbo..."
    rm -rf "$REPO_DIR"
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
props = torch.cuda.get_device_properties(0)
mem = (getattr(props, 'total_memory', 0) or getattr(props, 'total_mem', 0)) / 1024**3
print(f'  GPU: {name} (sm_{cc[0]}{cc[1]}, {mem:.0f}GB)')
assert mem >= 16, f'Need 16GB+ VRAM, got {mem:.0f}GB'
" || { echo "FATAL: Pre-flight failed"; exit 1; }

export MODEL_SIZE=1.7B CHUNK_SIZE=1 USE_CACHE=1 USE_MEGAKERNEL=1 USE_TALKER_MK=1
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

# ── Couche 3: external liveness probe ─────────────────────────────────
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

    _liveness_monitor "$server_pid" &
    monitor_pid=$!

    wait "$server_pid" 2>/dev/null
    exit_code=$?

    kill "$monitor_pid" 2>/dev/null
    wait "$monitor_pid" 2>/dev/null

    if [ $exit_code -eq 0 ]; then
        echo "Server exited cleanly."
        break
    fi

    restarts=$((restarts + 1))
    echo "Server exited with code $exit_code. Restarting in ${RESTART_DELAY}s... ($restarts/$MAX_RESTARTS)"

    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep $RESTART_DELAY
done

if [ $restarts -ge $MAX_RESTARTS ]; then
    echo "FATAL: Server crashed $MAX_RESTARTS times. Giving up."
    exit 1
fi
