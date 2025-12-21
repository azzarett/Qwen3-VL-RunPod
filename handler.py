import os
import torch
import runpod
import requests
import base64
from io import BytesIO
from PIL import UnidentifiedImageError
from pdf2image import convert_from_bytes
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

def expand_pdf_inputs(messages):
    """
    Detects PDF inputs (URL or base64) in messages and converts them to PIL images.
    Returns a new list of messages with PDFs expanded into images.
    """
    new_messages = []
    for msg in messages:
        new_content = []
        if isinstance(msg.get("content"), list):
            for item in msg["content"]:
                if item.get("type") == "image":
                    img_str = item.get("image", "")
                    pdf_data = None
                    
                    # Check for PDF URL
                    if isinstance(img_str, str) and (img_str.lower().endswith(".pdf") or ".pdf?" in img_str.lower()):
                        try:
                            print(f"Downloading PDF from: {img_str}")
                            response = requests.get(img_str)
                            response.raise_for_status()
                            pdf_data = response.content
                        except Exception as e:
                            print(f"Failed to download PDF: {e}")
                            # Keep original item to let downstream fail or handle
                            new_content.append(item)
                            continue

                    # Check for base64 PDF
                    elif isinstance(img_str, str) and img_str.startswith("data:application/pdf;base64,"):
                        try:
                            print("Decoding base64 PDF")
                            base64_data = img_str.split("base64,")[1]
                            pdf_data = base64.b64decode(base64_data)
                        except Exception as e:
                            print(f"Failed to decode base64 PDF: {e}")
                            new_content.append(item)
                            continue
                    
                    if pdf_data:
                        try:
                            print("Converting PDF to images...")
                            images = convert_from_bytes(pdf_data)
                            print(f"Converted PDF to {len(images)} images.")
                            for page_img in images:
                                new_content.append({"type": "image", "image": page_img})
                        except Exception as e:
                            print(f"Error converting PDF with pdf2image: {e}")
                            new_content.append(item)
                    else:
                        new_content.append(item)
                else:
                    new_content.append(item)
            
            # Create new message with updated content
            new_msg = msg.copy()
            new_msg["content"] = new_content
            new_messages.append(new_msg)
        else:
            new_messages.append(msg)
    return new_messages

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

    # Expand PDFs if any
    messages = expand_pdf_inputs(messages)

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    try:
        image_inputs, video_inputs = process_vision_info(messages)
    except UnidentifiedImageError as e:
        print(f"Error identifying image: {e}")
        # Log details about images to help debugging
        for msg in messages:
            if isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if item.get("type") == "image":
                        img = item.get("image", "")
                        if isinstance(img, str):
                            if img.startswith("data:image"):
                                print(f"Image (base64) length: {len(img)}")
                                print(f"Image (base64) prefix: {img[:50]}...")
                            else:
                                print(f"Image URL: {img}")
        return {"error": f"Failed to process image. The image data might be corrupted or in an unsupported format. Error: {str(e)}"}
    
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
