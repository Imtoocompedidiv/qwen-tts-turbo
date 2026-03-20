#!/bin/bash
set -e

# ── Self-contained startup ──
# Everything is in this repo: CUDA kernels in csrc/, server in deploy/.
# No external repo cloning needed.

REPO_DIR="/workspace/qwen-tts-turbo"
REPO_URL="https://github.com/Imtoocompedidiv/qwen-tts-turbo.git"

# Clone repo if not present
if [ ! -f "$REPO_DIR/deploy/runpod_server.py" ]; then
    echo "Cloning qwen-tts-turbo..."
    git clone --depth 1 "$REPO_URL" "$REPO_DIR"
fi

# Install dependencies
python3 -c "import faster_qwen3_tts" 2>/dev/null || pip install -q faster-qwen3-tts qwen-tts
pip install -q ninja soundfile websockets 2>/dev/null

# Megakernel CUDA source is vendored in csrc/ — no external clone needed.
# Kernels compile on first import via torch.utils.cpp_extension.load().

export MODEL_SIZE=1.7B CHUNK_SIZE=1 USE_CACHE=1 USE_MEGAKERNEL=1 USE_TALKER_MK=0
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
exec python3 "$REPO_DIR/deploy/runpod_server.py"
