FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    libgl1 libglib2.0-0 poppler-utils tesseract-ocr git wget build-essential cmake ffmpeg && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
# Install stable PyTorch with CUDA 12.1 for broad GPU compatibility (T4/V100/A10/30xx/40xx/L4)
# Pin matched torchvision to avoid ABI mismatches
ENV PIP_PREFER_BINARY=1
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.1+cu121 torchvision==0.19.1+cu121 && \
    pip install -r requirements.txt --no-cache-dir

# Install flash-attn (optional but recommended for speed)
# RUN pip install flash-attn --no-build-isolation

# PRE-DOWNLOAD THE MODEL - This makes the image larger but initialization faster
ARG MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
ENV MODEL_ID=${MODEL_ID} \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_CACHE=/root/.cache/huggingface/hub \
    TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers \
    HF_HUB_DISABLE_PROGRESS_BARS=1
RUN mkdir -p /root/.cache/huggingface
RUN python -c "import os; from transformers import AutoModelForImageTextToText, AutoProcessor; \
    model_id = os.environ.get('MODEL_ID'); \
    print(f'Downloading {model_id}...'); \
    AutoModelForImageTextToText.from_pretrained(model_id, trust_remote_code=True); \
    AutoProcessor.from_pretrained(model_id, trust_remote_code=True); \
    print('Download complete.')"

# Copy the rest of the application
COPY . .

# Create cache directory (already used during build)
RUN mkdir -p /root/.cache/huggingface

# Set the entrypoint
CMD [ "python", "-u", "handler.py" ]
