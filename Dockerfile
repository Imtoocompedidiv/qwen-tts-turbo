FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir faster-qwen3-tts qwen-tts soundfile ninja websockets

# Copy project
COPY . /app/

# Download model at build time (cached in image)
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', \
    local_dir='/app/model', ignore_patterns=['*.md'])"

# Pre-compile megakernel (cached in image, ~10s)
# The kernel compiles on first import via torch.utils.cpp_extension.load()

ENV MODEL_SIZE=1.7B \
    CHUNK_SIZE=1 \
    USE_CACHE=1 \
    USE_MEGAKERNEL=1 \
    USE_TALKER_MK=0 \
    PYTHONPATH=/app

EXPOSE 8000

CMD ["python3", "deploy/runpod_server.py"]
