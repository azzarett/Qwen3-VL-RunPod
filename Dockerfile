FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 poppler-utils tesseract-ocr git wget build-essential python3-dev cmake ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Install flash-attn (optional but recommended for speed)
# RUN pip install flash-attn --no-build-isolation

# PRE-DOWNLOAD THE MODEL - This makes the image larger but initialization faster
ARG MODEL_ID=Qwen/Qwen2-VL-7B-Instruct
ENV MODEL_ID=${MODEL_ID}
RUN python -c "from transformers import Qwen2VLForConditionalGeneration, AutoProcessor; \
    Qwen2VLForConditionalGeneration.from_pretrained('${MODEL_ID}'); \
    AutoProcessor.from_pretrained('${MODEL_ID}')"

# Copy the rest of the application
COPY . .

# Create volume directory structure
RUN mkdir -p /runpod-volume/cache

# Set environment variables for cache
ENV HF_HOME="/runpod-volume/cache"
ENV HF_HUB_CACHE="/runpod-volume/cache/hub"
ENV TRANSFORMERS_CACHE="/runpod-volume/cache/transformers"

# Set the entrypoint
CMD [ "python", "-u", "handler.py" ]
