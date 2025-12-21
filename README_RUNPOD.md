# RunPod Deployment for Qwen3-VL / Qwen2-VL

This folder contains the necessary files to deploy Qwen3-VL (or Qwen2-VL) on RunPod serverless or pods.

## Files

- `Dockerfile`: Defines the environment (PyTorch, CUDA, dependencies).
- `requirements.txt`: Python dependencies.
- `handler.py`: The serverless handler script.
- `runpod_demo.sh`: Script to run the Web Demo on a RunPod Pod.

## Option 1: RunPod Serverless

1. **Build the Docker image:**

```bash
docker build -t <your-username>/qwen-vl-runpod:latest .
```

2. **Push to Docker Hub (or other registry):**

```bash
docker push <your-username>/qwen-vl-runpod:latest
```

3. **Deploy on RunPod:**

- Create a new Template on RunPod.
- Image Name: `<your-username>/qwen-vl-runpod:latest`
- Container Disk: 50GB (Model is ~16GB for 8B, larger for others)
- Volume Disk: 100GB (for caching model weights)
- Volume Mount Path: `/runpod-volume`
- Environment Variables:
    - `MODEL_ID`: `Qwen/Qwen3-VL-8B-Instruct` (default). You can change this to `Qwen/Qwen3-VL-2B-Instruct` for faster loading or other models.
    - `HF_TOKEN`: (Optional) If using gated models.

### GPU/CUDA compatibility

This image uses stable PyTorch CUDA 12.1 to maximize compatibility across common RunPod GPUs (T4/V100/A10/30xx/40xx/L4). If you see a runtime error like:

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

it typically means the PyTorch build inside the image doesn’t include kernels for the GPU assigned to your job. Rebuild locally (or in CI) and redeploy:

```bash
# Rebuild and push
docker build -t <your-username>/qwen-vl-runpod:latest .
docker push <your-username>/qwen-vl-runpod:latest
```

At startup, the server logs will print the GPU name, compute capability, and the Torch/CUDA versions to help diagnose mismatches.

## Option 2: RunPod Pod (Interactive Web Demo)

If you want to run the Gradio Web Demo on a RunPod GPU instance:

1. **Start a Pod** using a PyTorch template (e.g., RunPod PyTorch 2.2.0).
2. **Clone this repository** inside the Pod.
3. **Run the demo script:**

```bash
bash runpod_demo.sh
```

This script will:
- Install necessary dependencies.
- Start the Web Demo with `Qwen/Qwen3-VL-8B-Instruct` (approx. 16GB VRAM required).
- Expose the demo on port 7860.

**Note:** Ensure you have exposed port 7860 in your Pod configuration (TCP Port) or use the RunPod Proxy link.

## Usage (Serverless)

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
