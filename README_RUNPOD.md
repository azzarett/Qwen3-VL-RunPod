# RunPod Deployment for Qwen2-VL

This folder contains the necessary files to deploy Qwen2-VL on RunPod serverless or pods.

## Files

- `Dockerfile`: Defines the environment (PyTorch, CUDA, dependencies).
- `requirements.txt`: Python dependencies.
- `handler.py`: The serverless handler script.

## Setup

1. **Build the Docker image:**

```bash
docker build -t <your-username>/qwen2-vl-runpod:latest .
```

2. **Push to Docker Hub (or other registry):**

```bash
docker push <your-username>/qwen2-vl-runpod:latest
```

3. **Deploy on RunPod:**

- Create a new Template on RunPod.
- Image Name: `<your-username>/qwen2-vl-runpod:latest`
- Container Disk: 20GB (Model is ~15GB for 7B, larger for others)
- Volume Disk: 50GB (for caching model weights)
- Volume Mount Path: `/runpod-volume`
- Environment Variables:
    - `MODEL_ID`: `Qwen/Qwen2-VL-7B-Instruct` (default) or other Qwen2-VL models.
    - `HF_TOKEN`: (Optional) If using gated models.

## Usage

The handler accepts input in a format similar to OpenAI Chat Completions or a simplified format.

### Input Formats

You can provide images via:
1. **URL**: `http://` or `https://` links.
2. **Base64**: `data:image/jpeg;base64,...` strings.

**Simplified Input:**
```json
{
  "input": {
    "prompt": "Describe this image.",
    "images": ["https://example.com/image.jpg"],
    "max_new_tokens": 200
  }
}
```

**Base64 Image Example:**
```json
{
  "input": {
    "prompt": "Read the text in this image.",
    "images": ["data:image/jpeg;base64,/9j/4AAQSkZJRg..."],
    "max_new_tokens": 200
  }
}
```

**Messages Input:**
```json
{
  "input": {
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image", "image": "https://example.com/image.jpg"},
          {"type": "text", "text": "Describe this image."}
        ]
      }
    ],
    "max_new_tokens": 200
  }
}
```
