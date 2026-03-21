#!/bin/bash
set -e

# ── Fast self-contained startup with supervised restart + liveness probe ──
# Target: cold start in ~3-4 minutes (was 15-20 minutes).
# Parallelizes: pip install || model download || CUDA kernel pre-compile.

REPO_DIR="/workspace/qwen-tts-turbo"
REPO_URL="https://github.com/Imtoocompedidiv/qwen-tts-turbo.git"
MAX_RESTARTS=10
RESTART_DELAY=5
LIVENESS_INTERVAL=5
LIVENESS_FAIL_THRESHOLD=6
LIVENESS_START_DELAY=120

# Clone repo if not present
if [ ! -f "$REPO_DIR/deploy/runpod_server.py" ]; then
    echo "Cloning qwen-tts-turbo..."
    git clone --depth 1 "$REPO_URL" "$REPO_DIR"
fi

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

# ── Parallel setup: pip + model download + kernel compile ──────────────
echo "Parallel setup starting..."
t0=$(date +%s)

# 1) pip install (skip PyTorch, already in image)
_pip_install() {
    python3 -c "import faster_qwen3_tts" 2>/dev/null && echo "  [pip] already installed" && return 0
    echo "  [pip] installing..."
    pip install --no-deps -q faster-qwen3-tts qwen-tts 2>/dev/null
    # Install only missing lightweight deps (skip torch, numpy, etc already in image)
    pip install -q transformers accelerate safetensors tokenizers soundfile ninja websockets 2>/dev/null
    echo "  [pip] done"
}

# 2) Download model weights
_download_model() {
    local model_dir="/workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    if [ -d "$model_dir" ] && [ -f "$model_dir/model.safetensors" ]; then
        echo "  [model] already downloaded"
        return 0
    fi
    echo "  [model] downloading..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice',
                  local_dir='/workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice',
                  ignore_patterns=['*.md'])
print('  [model] done')
"
}

# 3) Pre-compile CUDA kernels (JIT cache persists in /root/.cache/torch_extensions)
_compile_kernels() {
    if python3 -c "import torch; torch.ops.load_library" 2>/dev/null && \
       [ -d "/root/.cache/torch_extensions" ] && \
       find /root/.cache/torch_extensions -name "qwen_megakernel_C*" -newer "$REPO_DIR/csrc/kernel.cu" 2>/dev/null | grep -q .; then
        echo "  [kernels] already compiled"
        return 0
    fi
    echo "  [kernels] compiling predictor megakernel..."
    python3 -c "
import sys
sys.path.insert(0, '$REPO_DIR')
from deploy.industrial.build_predictor import get_predictor_extension
get_predictor_extension()
print('  [kernels] done')
"
}

# Run all three in parallel
_pip_install &
pid_pip=$!
_download_model &
pid_model=$!

# Wait for pip before compiling kernels (needs faster_qwen3_tts imported by engine)
wait $pid_pip
_compile_kernels &
pid_kernels=$!

# Wait for all
wait $pid_model
wait $pid_kernels

t1=$(date +%s)
echo "Setup complete in $((t1 - t0))s"

export MODEL_SIZE=1.7B CHUNK_SIZE=1 USE_CACHE=1 USE_MEGAKERNEL=1 USE_TALKER_MK=0
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
