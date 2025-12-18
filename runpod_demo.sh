#!/bin/bash

# Script to run Qwen3-VL Web Demo on RunPod (Pod Mode)

# 1. Install dependencies
echo "Installing dependencies..."
pip install -r requirements_web_demo.txt
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation

# 2. Run the Web Demo
# We use the 8B model by default as it is a good balance between performance and speed.
# The 235B model takes too long to download and requires massive GPU resources.
MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"

echo "Starting Web Demo with model: $MODEL_ID"
echo "This may take a few minutes to download the model on the first run."

python web_demo_mm.py \
    -c $MODEL_ID \
    --server-name 0.0.0.0 \
    --server-port 7860 \
    --flash-attn2
