import os
import torch
import runpod
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

# Environment variables
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")

# Require CUDA (5090 GPUs); fail fast if unavailable
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available but GPU is required (RTX 5090 nodes).")

DEVICE = "cuda"
TORCH_DTYPE = torch.float16  # fp16 for maximum compatibility; 5090 supports bf16 but fp16 is safer with older kernels

def _print_cuda_env():
    try:
        dev_idx = torch.cuda.current_device() if torch.cuda.is_available() else None
        name = torch.cuda.get_device_name(dev_idx) if dev_idx is not None else "CPU"
        cap = torch.cuda.get_device_capability(dev_idx) if dev_idx is not None else (0, 0)
        print(
            f"CUDA available={torch.cuda.is_available()} | device={name} | capability=sm_{cap[0]}{cap[1]} | "
            f"torch={torch.__version__} | torch.cuda={getattr(torch.version, 'cuda', 'n/a')}"
        )
    except Exception as e:
        print(f"[warn] Failed to query CUDA env: {e}")

print(f"Loading model: {MODEL_ID} on {DEVICE} with dtype {TORCH_DTYPE}")
_print_cuda_env()

def load_model(attn_impl: str):
    return AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        attn_implementation=attn_impl,
        device_map="auto",
        trust_remote_code=True,
    )

try:
    model = load_model("sdpa")
except RuntimeError as e:
    print(f"Failed to load with sdpa on CUDA: {e}. Retrying with eager.")
    model = load_model("eager")

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

def handler(job):
    job_input = job["input"]
    
    # Extract prompt and images/videos
    # Expected format: 
    # {
    #   "messages": [
    #     {
    #       "role": "user",
    #       "content": [
    #         {"type": "image", "image": "https://..."} or {"type": "image", "image": "file:///..."},
    #         {"type": "text", "text": "Describe this image."}
    #       ]
    #     }
    #   ]
    # }
    # OR simplified:
    # {
    #   "prompt": "Describe this image",
    #   "images": ["http://..."],
    #   "videos": ["http://..."]
    # }
    
    messages = job_input.get("messages")
    
    if not messages:
        # Handle simplified input
        prompt = job_input.get("prompt", "Describe this image.")
        images = job_input.get("images", [])
        videos = job_input.get("videos", [])
        
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        for vid in videos:
            content.append({"type": "video", "video": vid})
        content.append({"type": "text", "text": prompt})
        
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(DEVICE)

    # Inference parameters
    max_new_tokens = job_input.get("max_new_tokens", 128)
    
    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]

runpod.serverless.start({"handler": handler})
