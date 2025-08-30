# code/eval_lora.py
import os, argparse, inspect
from dataclasses import dataclass
from typing import Tuple, Dict

import torch
from safetensors.torch import load_file
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import AutoTokenizer
from peft import LoraConfig as PeftLoraConfig, get_peft_model

# ---------------------------
# Configuration container with sensible defaults
# ---------------------------
@dataclass
class Args:
    model_name_or_path: str = "runwayml/stable-diffusion-v1-5"  # Base SD model to load
    tokenizer_dir: str = "./artifacts/tokenizer"                # Tokenizer directory (supports added special tokens)
    lora_path: str = "./lora_out/pytorch_lora_weights.safetensors"  # LoRA weights (combined TE+UNet, namespaced)
    prompt: str = "a busy market, in <sks> style"               # Positive prompt (can reference learned tokens)
    negative_prompt: str = ""                                   
    num_images: int = 3                                         # How many images to render
    guidance: float = 7.5                                       # Classifier-free guidance scale
    steps: int = 30                                             # Sampler steps
    height: int = 512                                           # Output height
    width: int = 512                                            # Output width
    output_dir: str = "./samples"                               # Save outputs
    seed: int = 1234                                            # Base seed (incremented per image)
    float16: bool = True                                        # Use FP16 if CUDA is available
    local_files_only: bool = False                              # Hugging Face cache only (no network) if True

# ---------------------------
# CLI parsing that preserves dataclass defaults and types
# ---------------------------
def parse_args() -> Args:
    p = argparse.ArgumentParser()
    for f, field in Args.__dataclass_fields__.items():
        t = field.type
        if t is bool:
            p.add_argument(f"--{f}", action="store_true")
        else:
            p.add_argument(f"--{f}")
    ns = p.parse_args()
    d = Args().__dict__.copy()
    for k, v in vars(ns).items():
        if v is not None:
            if isinstance(d[k], int): v = int(v)     # keep int types
            if isinstance(d[k], float): v = float(v) # keep float types
            d[k] = v
    return Args(**d)

# ---------------------------
# Attach PEFT LoRA adapters to both the text encoder (TE) and UNet
# Also wraps UNet forward to ignore unexpected kwargs from schedulers/pipelines
# ---------------------------
def wrap_with_lora_te_and_unet(pipe: StableDiffusionPipeline, rank: int = 8):
    # TE LoRA
    te_cfg = PeftLoraConfig(
        r=rank, lora_alpha=rank, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        bias="none", task_type="FEATURE_EXTRACTION",
    )
    pipe.text_encoder = get_peft_model(pipe.text_encoder, te_cfg)

    # UNet LoRA
    unet_cfg = PeftLoraConfig(
        r=rank, lora_alpha=rank, lora_dropout=0.0,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        bias="none", task_type="FEATURE_EXTRACTION",
    )
    pipe.unet = get_peft_model(pipe.unet, unet_cfg)

    # Robust UNet forward: pass only expected args
    # Some scheduler/pipeline versions pass extra kwargs; filter to the base signature.
    base_unet = getattr(pipe.unet, "model", None) or getattr(pipe.unet, "base_model", None) or pipe.unet
    base_sig = inspect.signature(base_unet.forward)
    valid = set(base_sig.parameters.keys())

    def safe_forward(*args, **kwargs):
        clean = {k: v for k, v in kwargs.items() if k in valid}
        return base_unet(*args, **clean)

    pipe.unet.forward = safe_forward  

# ---------------------------
# Load safetensors LoRA weights and split them into TE/UNet state dicts
# Expected keys: "text_encoder.*" and "unet.*"
# ---------------------------
def load_split_lora_weights(lora_path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    sd = load_file(lora_path, device="cpu")
    te_state, unet_state = {}, {}
    for k, v in sd.items():
        if k.startswith("text_encoder."):
            te_state[k[len("text_encoder."):]] = v
        elif k.startswith("unet."):
            unet_state[k[len("unet."):]] = v
    return te_state, unet_state

# ---------------------------
# Encode prompts using the base CLIPTextModel underneath PEFT
# Avoids input_embeds shape/path issues when TE is wrapped by PEFT
# ---------------------------
@torch.no_grad()
def encode_prompts_with_peft_te(pipe: StableDiffusionPipeline,
                                tokenizer: AutoTokenizer,
                                prompt: str,
                                negative_prompt: str,
                                device: str,
                                dtype: torch.dtype):
    # Tokenize positive and negative prompts to fixed-length sequences
    pos = tokenizer([prompt], padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt")
    neg = tokenizer([negative_prompt] if negative_prompt else [""],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt")

    # unwrap PEFT TE to the base CLIPTextModel (avoids inputs_embeds mismatch)
    peft_te = pipe.text_encoder
    base_te = getattr(peft_te, "get_base_model", lambda: peft_te)()
    if hasattr(base_te, "base_model"):
        base_te = base_te.base_model
    base_te = base_te.to(device=device, dtype=dtype)

    # Forward pass to get embeddings (last hidden state)
    prompt_embeds = base_te(input_ids=pos.input_ids.to(device))[0]
    negative_embeds = base_te(input_ids=neg.input_ids.to(device))[0]
    return prompt_embeds, negative_embeds

# ---------------------------
# Main entry: builds pipeline, applies LoRA, pre-encodes prompts, and generates images
# ---------------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Mixed precision on CUDA if available; else fall back to FP32/CPU
    dtype = torch.float16 if args.float16 and torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f">> Loading base pipeline from: {args.model_name_or_path}")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
        safety_checker=None,   # Disable safety checker for speed/compat; ensure prompts are safe
        feature_extractor=None,
    )

    # Switch to DPM++ 2M Karras 
    try:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True
        )
        print(">> Sampler: DPM++ 2M Karras")
    except Exception as e:
        print(">> Could not set DPM++ 2M Karras; using default scheduler.", e)

    # Use a tokenizer that may include added special tokens (e.g., <sks>)
    print(f">> Using tokenizer at: {args.tokenizer_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=False)
    pipe.tokenizer = tokenizer
    new_vocab = len(tokenizer)
    old_vocab = pipe.text_encoder.get_input_embeddings().num_embeddings
    if new_vocab != old_vocab:
        print(f"Resizing text encoder embeddings: {old_vocab} -> {new_vocab}")
        pipe.text_encoder.resize_token_embeddings(new_vocab)

    # Attach LoRA adapters (same target modules as training)
    wrap_with_lora_te_and_unet(pipe, rank=8)

    # Load LoRA weights and apply to the PEFT-wrapped modules
    print(f">> Loading LoRA from: {args.lora_path}")
    te_state, unet_state = load_split_lora_weights(args.lora_path)
    missing_te, unexpected_te = pipe.text_encoder.load_state_dict(te_state, strict=False)
    missing_unet, unexpected_unet = pipe.unet.load_state_dict(unet_state, strict=False)
    print(f"Loaded TE LoRA (missing={len(missing_te)}, unexpected={len(unexpected_te)})")
    print(f"Loaded UNet LoRA (missing={len(missing_unet)}, unexpected={len(unexpected_unet)})")

    # Final device move and light memory opts
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    try: pipe.enable_attention_slicing()
    except Exception: pass
    try: pipe.enable_vae_slicing()
    except Exception: pass

    # Pre-encode prompts with base TE (keeps PEFT quirks out of __call__)
    prompt_embeds, negative_prompt_embeds = encode_prompts_with_peft_te(
        pipe, tokenizer, args.prompt, args.negative_prompt, device, dtype
    )

    # Generate deterministically per image with incremented seeds
    print(f">> Rendering {args.num_images} image(s) to: {args.output_dir}")
    for i in range(args.num_images):
        g = torch.Generator(device=device).manual_seed(args.seed + i)
        out = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            height=args.height,
            width=args.width,
            generator=g,
        )
        image = out.images[0]
        path = os.path.join(args.output_dir, f"sks_{i:02d}.png")
        image.save(path)
        print("Saved:", path)

# Standard Python entrypoint guard
if __name__ == "__main__":
    main()
