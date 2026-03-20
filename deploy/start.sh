#!/bin/bash
# Deps are in the torch 2.8 image. Only faster-qwen3-tts needs install.
python3 -c "import faster_qwen3_tts" 2>/dev/null || pip install -q --no-deps faster-qwen3-tts qwen-tts
pip install -q ninja 2>/dev/null  # For megakernel compilation

# Clone + patch megakernel (if not already present)
if [ ! -d /workspace/megakernel-tts ]; then
    git clone --depth 1 https://github.com/jayanth-kumar-morem/qwen-megakernel-tts /workspace/megakernel-tts
    python3 /workspace/patch_kernel_datacenter.py /workspace/megakernel-tts
    rm -rf ~/.cache/torch_extensions
fi

export MODEL_SIZE=1.7B CHUNK_SIZE=1 USE_CACHE=1 USE_MEGAKERNEL=1 USE_TALKER_MK=1
exec python3 /workspace/tts_server.py
