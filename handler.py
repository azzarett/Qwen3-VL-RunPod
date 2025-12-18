import os
import torch
import runpod
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Environment variables
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model: {MODEL_ID} on {DEVICE}")

# Load model and processor
# Note: flash_attention_2 is recommended for performance but requires flash-attn package
try:
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
except Exception as e:
    print(f"Failed to load with flash_attention_2, falling back to default: {e}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

processor = AutoProcessor.from_pretrained(MODEL_ID)

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
