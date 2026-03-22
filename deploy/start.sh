#!/bin/bash
set -e

# ── Fastest possible cold start ──
# Single Python process does everything: dep check, pre-flight, parallel
# downloads, then execs the server. Avoids 4x Python startup overhead (~8s).

REPO_DIR="/workspace/qwen-tts-turbo"
REPO_URL="https://github.com/Imtoocompedidiv/qwen-tts-turbo.git"
MAX_RESTARTS=10
RESTART_DELAY=5
LIVENESS_INTERVAL=5
LIVENESS_FAIL_THRESHOLD=6
LIVENESS_START_DELAY=90

# Clone or update repo
if [ ! -f "$REPO_DIR/deploy/runpod_server.py" ]; then
    git clone --depth 1 "$REPO_URL" "$REPO_DIR"
else
    cd "$REPO_DIR" && git fetch --depth 1 origin master && git reset --hard origin/master && cd /
fi

# Install deps only if missing (single check, no separate Python process)
python3 -c "import faster_qwen3_tts, soundfile, websockets, ninja" 2>/dev/null || \
    pip install -q faster-qwen3-tts qwen-tts soundfile ninja websockets

# Fast-start: fewer KV combos on cold start (12 vs 480)
if [ ! -f /workspace/prefill_cache.pt ]; then
    export CACHE_VOICES="Vivian,Dylan"
    export CACHE_LANGUAGES="French,English,Chinese"
    export CACHE_TONES="|Parle d'un ton chaleureux et professionnel"
fi
export MODEL_SIZE=1.7B CHUNK_SIZE=1 USE_CACHE=1 USE_MEGAKERNEL=1 USE_TALKER_MK=0
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

# Pre-flight + parallel setup in ONE Python process (saves ~8s of interpreter startups)
python3 -u -c "
import torch, os, sys, threading, time
t0 = time.time()

# Pre-flight
assert torch.cuda.is_available(), 'No CUDA GPU'
cc = torch.cuda.get_device_capability()
name = torch.cuda.get_device_name(0)
props = torch.cuda.get_device_properties(0)
mem = (getattr(props, 'total_memory', 0) or getattr(props, 'total_mem', 0)) / 1024**3
print(f'GPU: {name} (sm_{cc[0]}{cc[1]}, {mem:.0f}GB)')
assert mem >= 16, f'Need 16GB+ VRAM, got {mem:.0f}GB'

# Parallel: model download + kernel compile
model_dir = '/workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice'
def download_model():
    if os.path.isfile(os.path.join(model_dir, 'model.safetensors')):
        print('  [model] cached')
        return
    print('  [model] downloading...')
    from huggingface_hub import snapshot_download
    snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', local_dir=model_dir, ignore_patterns=['*.md'])
    print('  [model] done')

def compile_kernel():
    print('  [kernel] compiling...')
    sys.path.insert(0, '$REPO_DIR')
    from deploy.industrial.build_predictor import get_predictor_extension
    get_predictor_extension()
    print('  [kernel] done')

t1 = threading.Thread(target=download_model, daemon=True)
t2 = threading.Thread(target=compile_kernel, daemon=True)
t1.start(); t2.start()
t1.join(); t2.join()
print(f'Setup: {time.time()-t0:.1f}s')
" || { echo "FATAL: Setup failed"; exit 1; }

# ── Liveness probe ────────────────────────────────────────────────────
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
                echo "LIVENESS: /health failed ${fails}x. Killing PID $server_pid."
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
