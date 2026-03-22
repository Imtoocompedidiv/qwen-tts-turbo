#!/bin/bash
set -e

# ── Fast self-contained startup with supervised restart + liveness probe ──
# Cold start target: ~2 min (model download || kernel compile in parallel)

REPO_DIR="/workspace/qwen-tts-turbo"
REPO_URL="https://github.com/Imtoocompedidiv/qwen-tts-turbo.git"
MAX_RESTARTS=10
RESTART_DELAY=5
LIVENESS_INTERVAL=5
LIVENESS_FAIL_THRESHOLD=6
LIVENESS_START_DELAY=120
MODEL_DIR="/workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice"

# Clone or update repo
if [ ! -f "$REPO_DIR/deploy/runpod_server.py" ]; then
    echo "Cloning qwen-tts-turbo..."
    rm -rf "$REPO_DIR"
    git clone --depth 1 "$REPO_URL" "$REPO_DIR"
else
    echo "Updating qwen-tts-turbo..."
    cd "$REPO_DIR" && git fetch --depth 1 origin master && git reset --hard origin/master && cd /
fi

# Install dependencies (must complete before parallel steps)
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

export MODEL_SIZE=1.7B CHUNK_SIZE=1 USE_CACHE=1 USE_MEGAKERNEL=1 USE_TALKER_MK=0
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

# ── Parallel: model download + kernel compile (~40s saved) ────────────
# Both depend on pip but not on each other. Non-fatal fallbacks.
t0=$(date +%s)

if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/model.safetensors" ]; then
    echo "  [model] already at $MODEL_DIR"
else
    echo "  [model] downloading in background..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice',
                  local_dir='$MODEL_DIR', ignore_patterns=['*.md'])
print('  [model] done')
" &
    pid_model=$!
fi

echo "  [kernels] compiling in background..."
python3 -c "
import sys; sys.path.insert(0, '$REPO_DIR')
from deploy.industrial.build_predictor import get_predictor_extension
get_predictor_extension()
print('  [kernels] done')
" &
pid_kernels=$!

# Wait for both (|| true: don't exit on failure, server handles it)
[ -n "${pid_model:-}" ] && wait $pid_model || true
wait $pid_kernels || true

echo "  Setup done in $(($(date +%s) - t0))s"

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
