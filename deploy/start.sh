#!/bin/bash
set -e

# ── Self-contained startup with supervised restart ──
# Clones repo, installs deps, starts server.
# On crash/deadlock, waits 5s and restarts automatically.

REPO_DIR="/workspace/qwen-tts-turbo"
REPO_URL="https://github.com/Imtoocompedidiv/qwen-tts-turbo.git"
MAX_RESTARTS=10
RESTART_DELAY=5

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

# Megakernel CUDA source is vendored in csrc/ — no external clone needed.
# Kernels compile on first import via torch.utils.cpp_extension.load().
# Server auto-detects GPU and disables megakernels on sm_80 and below.

export MODEL_SIZE=1.7B CHUNK_SIZE=1 USE_CACHE=1 USE_MEGAKERNEL=1 USE_TALKER_MK=0
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

# Supervised restart loop
restarts=0
while [ $restarts -lt $MAX_RESTARTS ]; do
    echo "Starting server (attempt $((restarts + 1))/$MAX_RESTARTS)..."
    python3 -u "$REPO_DIR/deploy/runpod_server.py" && break
    exit_code=$?
    restarts=$((restarts + 1))
    echo "Server exited with code $exit_code. Restarting in ${RESTART_DELAY}s... ($restarts/$MAX_RESTARTS)"
    sleep $RESTART_DELAY
done

if [ $restarts -ge $MAX_RESTARTS ]; then
    echo "FATAL: Server crashed $MAX_RESTARTS times. Giving up."
    exit 1
fi
