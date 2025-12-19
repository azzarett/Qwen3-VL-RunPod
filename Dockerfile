FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 poppler-utils tesseract-ocr git wget build-essential python3-dev cmake ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
# Use PyTorch nightly cu124 (includes newer arch support; required for RTX 5090 sm_120)
ENV PIP_PREFER_BINARY=1
RUN pip install --upgrade pip && \
    pip uninstall -y torch torchvision torchaudio || true && \
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124 --no-cache-dir && \
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
