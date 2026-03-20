#!/bin/bash
set -e

# ── Self-contained startup for RunPod ──
# This script can run from anywhere. It clones the repo if needed
# and uses paths relative to the repo root.

REPO_DIR="/workspace/qwen-tts-turbo"
REPO_URL="https://github.com/Imtoocompedidiv/qwen-tts-turbo.git"

# Clone repo if not present
if [ ! -f "$REPO_DIR/deploy/runpod_server.py" ]; then
    echo "Cloning qwen-tts-turbo..."
    git clone --depth 1 "$REPO_URL" "$REPO_DIR" 2>/dev/null || true
fi

# Install dependencies
python3 -c "import faster_qwen3_tts" 2>/dev/null || pip install -q --no-deps faster-qwen3-tts qwen-tts
pip install -q ninja 2>/dev/null

# Clone + patch megakernel for datacenter GPUs
if [ ! -d /workspace/megakernel-tts ]; then
    git clone --depth 1 https://github.com/jayanth-kumar-morem/qwen-megakernel-tts /workspace/megakernel-tts
    python3 "$REPO_DIR/deploy/industrial/patch_kernel_datacenter.py" /workspace/megakernel-tts
    rm -rf ~/.cache/torch_extensions
fi

export MODEL_SIZE=1.7B CHUNK_SIZE=1 USE_CACHE=1 USE_MEGAKERNEL=1 USE_TALKER_MK=1
exec python3 "$REPO_DIR/deploy/runpod_server.py"
