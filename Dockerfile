FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 poppler-utils tesseract-ocr git wget build-essential python3-dev cmake ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Install flash-attn (optional but recommended)
# RUN pip install flash-attn --no-build-isolation

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
