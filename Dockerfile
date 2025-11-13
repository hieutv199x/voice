# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Ho_Chi_Minh

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install additional API dependencies
RUN pip3 install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    python-multipart

# Copy application files
COPY . .

# Create sample directory if it doesn't exist
RUN mkdir -p sample

# Download models from Hugging Face during build
# Use HF_HUB_ENABLE_HF_TRANSFER for faster downloads
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN pip3 install --no-cache-dir hf-transfer && \
    python3 -c "from huggingface_hub import snapshot_download; \
    print('Downloading VieNeu-TTS model...'); \
    snapshot_download(repo_id='pnnbao-ump/VieNeu-TTS', cache_dir='/root/.cache/huggingface'); \
    print('Downloading NeuCodec model...'); \
    snapshot_download(repo_id='neuphonic/neucodec', cache_dir='/root/.cache/huggingface'); \
    print('✅ Models downloaded successfully')" && \
    pip3 uninstall -y hf-transfer && \
    rm -rf /tmp/* /var/tmp/*

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the API
CMD ["python3", "api.py"]
