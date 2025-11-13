# VieNeu-TTS Docker Deployment

This guide explains how to build and run the VieNeu-TTS API using Docker with GPU support.

## Prerequisites

### For GPU Support
1. **NVIDIA GPU** with CUDA support
2. **NVIDIA Driver** installed on host machine
3. **NVIDIA Container Toolkit** installed:
   ```bash
   # Add NVIDIA package repositories
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   # Install nvidia-container-toolkit
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   
   # Restart Docker
   sudo systemctl restart docker
   ```

4. Verify GPU is accessible:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

### For CPU Only
- Docker and Docker Compose installed

## Quick Start

### Option 1: GPU Mode (Recommended)

Build and run with GPU support:

```bash
# Build the Docker image
docker-compose build

# Run the container with GPU
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### Option 2: CPU Mode

Run without GPU:

```bash
# Build and run with CPU only
docker-compose -f docker-compose.cpu.yml up -d

# Check logs
docker-compose -f docker-compose.cpu.yml logs -f

# Stop
docker-compose -f docker-compose.cpu.yml down
```

## Manual Docker Commands

### Build Image
```bash
docker build -t vieneu-tts-api:latest .
```

### Run with GPU
```bash
docker run -d \
  --name vieneu-tts-api \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/sample:/app/sample:ro \
  -e CUDA_VISIBLE_DEVICES=0 \
  vieneu-tts-api:latest
```

### Run with CPU
```bash
docker run -d \
  --name vieneu-tts-api \
  -p 8000:8000 \
  -v $(pwd)/sample:/app/sample:ro \
  -e CUDA_VISIBLE_DEVICES="" \
  vieneu-tts-api:latest
```

### Run with specific GPU
```bash
# Use GPU 1 only
docker run -d \
  --name vieneu-tts-api \
  --gpus '"device=1"' \
  -p 8000:8000 \
  -v $(pwd)/sample:/app/sample:ro \
  vieneu-tts-api:latest
```

## Configuration

### Environment Variables

- `CUDA_VISIBLE_DEVICES`: Specify which GPU(s) to use
  - `0` - Use first GPU
  - `1` - Use second GPU
  - `0,1` - Use first and second GPU
  - `""` - Disable GPU (CPU mode)

- `PYTORCH_CUDA_ALLOC_CONF`: CUDA memory allocation configuration
  - `max_split_size_mb:512` - Helps with memory fragmentation

### Port Mapping

Default port is `8000`. To change:

```bash
# Map to different port (e.g., 9000)
docker run -d -p 9000:8000 --gpus all vieneu-tts-api:latest
```

Or edit `docker-compose.yml`:
```yaml
ports:
  - "9000:8000"  # Host:Container
```

## Accessing the API

Once running, the API is available at:

- **API Endpoint**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### Test the API

```bash
# List available voices
curl http://localhost:8000/voices

# Generate speech
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Xin chào, đây là giọng nói được tổng hợp từ VieNeu-TTS",
    "voice_id": "nam_1"
  }' \
  --output output.wav
```

## Monitoring

### View Logs
```bash
# Docker Compose
docker-compose logs -f

# Docker
docker logs -f vieneu-tts-api
```

### Check GPU Usage Inside Container
```bash
# Enter container
docker exec -it vieneu-tts-api bash

# Check GPU
nvidia-smi

# Check PyTorch CUDA availability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### Container Stats
```bash
docker stats vieneu-tts-api
```

## Troubleshooting

### GPU Not Detected

1. Verify NVIDIA drivers on host:
   ```bash
   nvidia-smi
   ```

2. Check Docker can access GPU:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

3. Check container logs:
   ```bash
   docker logs vieneu-tts-api
   ```

### Out of Memory

Reduce batch size or use CPU mode:
```bash
docker-compose -f docker-compose.cpu.yml up -d
```

### Model Download Issues

The first run will download models from Hugging Face. This may take time. Check logs:
```bash
docker-compose logs -f
```

## Production Deployment

### Using Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 300s;
    }
}
```

### Resource Limits

Add to `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 8G
    reservations:
      cpus: '2'
      memory: 4G
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## Cleanup

```bash
# Stop and remove containers
docker-compose down

# Remove images
docker rmi vieneu-tts-api:latest

# Remove volumes (careful - deletes cached models)
docker volume rm vieneu-tts_huggingface-cache
```

## Performance Tips

1. **Use GPU** for faster inference (5-10x speedup)
2. **Persistent volume** for HuggingFace cache to avoid re-downloading models
3. **Multiple GPUs**: Set `count: all` in docker-compose.yml for multi-GPU support
4. **Memory**: Ensure at least 4GB RAM + 4GB VRAM for smooth operation
