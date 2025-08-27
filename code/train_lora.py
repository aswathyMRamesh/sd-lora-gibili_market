# code/train_lora.py
import os, glob, random, argparse
from dataclasses import dataclass
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from accelerate import Accelerator
from safetensors.torch import save_file

from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import AutoTokenizer, CLIPTextModel

from peft import LoraConfig as PeftLoraConfig, get_peft_model
from peft.utils import TaskType


# --------------------------
# Args
# --------------------------
@dataclass
class Args:
    model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
    data_dir: str = "./data/data/512"
    output_dir: str = "./lora_out"
    tokenizer_dir: str = "./artifacts/tokenizer"
    instance_token: str = "<sks>"
    resolution: int = 512
    max_steps: int = 800
    batch_size: int = 4
    lr: float = 1e-4
    lora_rank: int = 8
    seed: int = 42
    num_workers: int = 4
    log_every: int = 50
    float16: bool = True
    local_files_only: bool = True  # offline-safe


def parse_args():
    p = argparse.ArgumentParser()
    for f in Args.__dataclass_fields__.keys():
        t = Args.__dataclass_fields__[f].type
        if t is bool:
            p.add_argument(f"--{f}", action="store_true")
        else:
            p.add_argument(f"--{f}")
    ns = p.parse_args()
    d = Args().__dict__
    for k, v in vars(ns).items():
        if v is not None:
            if isinstance(d[k], int): v = int(v)
            if isinstance(d[k], float): v = float(v)
            d[k] = v
        elif isinstance(d[k], bool):
            pass
    return Args(**d)


# --------------------------
# Dataset
# --------------------------
class ImageDataset(Dataset):
    def __init__(self, folder, size, instance_token, tokenizer):
        self.paths = sorted(sum([glob.glob(os.path.join(folder, ext))
                                 for ext in ("*.png","*.jpg","*.jpeg","*.webp")], []))
        if not self.paths:
            raise ValueError(f"No images found in {folder}")
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        # single fixed prompt for simple style tuning
        prompt = f"a busy market, in {instance_token} style"
        self.input_ids = tokenizer(
            prompt, padding="max_length", truncation=True,
            max_length=tokenizer.model_max_length, return_tensors="pt"
        ).input_ids[0]

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return {"pixel_values": self.transform(img), "input_ids": self.input_ids}


# --------------------------
# Utils
# --------------------------
def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# NEW: add LoRA to UNet using PEFT (works across diffusers versions)
def add_lora_to_unet_with_peft(unet: UNet2DConditionModel, rank: int):
    """
    Wrap the UNet with a PEFT LoRA adapter targeting cross/self-attn projections.
    Returns (wrapped_unet, list_of_trainable_params).
    """
    cfg = PeftLoraConfig(
        r=rank,
        lora_alpha=rank,
        lora_dropout=0.0,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],  # diffusers UNet attn proj names
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    unet = get_peft_model(unet, cfg)
    for n, p in unet.named_parameters():
        p.requires_grad_(("lora_" in n))
    trainable = [p for p in unet.parameters() if p.requires_grad]
    return unet, trainable


# --------------------------
# Train
# --------------------------
def main():
    args = parse_args()
    print(f">> Using base model at: {args.model_name_or_path}")
    print(f">> Using tokenizer at: {args.tokenizer_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    accelerator = Accelerator(gradient_accumulation_steps=1,
                              mixed_precision=("fp16" if args.float16 else "no"))
    set_seed(args.seed)

    dtype = torch.float16 if args.float16 else torch.float32
    torch.backends.cudnn.benchmark = True
    # --- Load tokenizer and TEXT ENCODER (OFFLINE) ---
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_name_or_path, subfolder="text_encoder",
        local_files_only=args.local_files_only
    )

    # >>> resize TE embeddings to match tokenizer (adds <sks>)
    new_vocab = len(tokenizer)
    old_vocab = text_encoder.get_input_embeddings().num_embeddings
    if new_vocab != old_vocab:
        print(f"Resizing text encoder embeddings: {old_vocab} -> {new_vocab}")
        text_encoder.resize_token_embeddings(new_vocab)

    # Ensure <sks> exists
    tok_id = tokenizer.convert_tokens_to_ids(args.instance_token)
    if tok_id is None or tok_id < 0 or tok_id >= new_vocab:
        raise ValueError(
            f"Instance token {args.instance_token!r} not found in tokenizer. "
            f"Make sure you saved tokenizer with the new token added."
        )

    # --- Load VAE / UNet / Scheduler (OFFLINE) ---
    vae = AutoencoderKL.from_pretrained(
        args.model_name_or_path, subfolder="vae",
        local_files_only=args.local_files_only
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.model_name_or_path, subfolder="unet",
        local_files_only=args.local_files_only
    )

    # >>> Ensure UNet is NOT PEFT-wrapped (we use Diffusers attention processors)
    try:
        # If someone wrapped UNet with PEFT earlier, unwrap it.
        if hasattr(unet, "get_base_model"):
            unet = unet.get_base_model()
        elif hasattr(unet, "base_model"):
            unet = unet.base_model
    except Exception:
        pass
    
    noise_sched = DDPMScheduler.from_pretrained(
        args.model_name_or_path, subfolder="scheduler",
        local_files_only=args.local_files_only
    )

    # cast to dtype
    vae.to(dtype); unet.to(dtype); text_encoder.to(dtype)

    # --- Freeze base weights (we only train LoRA adapters) ---
    for m in (vae, unet, text_encoder):
        m.requires_grad_(False)

    # --- Add LoRA to TEXT ENCODER (PEFT) ---
    te_lora_cfg = PeftLoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    text_encoder = get_peft_model(text_encoder, te_lora_cfg)
    for n, p in text_encoder.named_parameters():
        p.requires_grad_(("lora_" in n))

    # --- Add LoRA to UNet (PEFT) — THIS is the part you were missing ---
    unet, unet_lora_params = add_lora_to_unet_with_peft(unet, args.lora_rank)

    if accelerator.is_main_process:
        print(f"Trainable params - Text Encoder: {count_params(text_encoder)}")
        print(f"Trainable params - UNet (LoRA only): {sum(p.numel() for p in unet_lora_params)}")

    # --- Data ---
    dataset = ImageDataset(args.data_dir, args.resolution, args.instance_token, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # --- Optimizer (TE-LoRA + UNet-LoRA) ---
    te_params = [p for p in text_encoder.parameters() if p.requires_grad]
    params = te_params + list(unet_lora_params)
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    # Prepare (DDP-safe)
    text_encoder, unet, vae, optimizer, dataloader = accelerator.prepare(
        text_encoder, unet, vae, optimizer, dataloader
    )
    unet_call = accelerator.unwrap_model(unet)
    # If it's a PEFT wrapper, drop to the base UNet (LoRA modules are still attached)   
    unet_call = getattr(unet_call, "get_base_model", lambda: unet_call)()
    unet_call = getattr(unet_call, "base_model", unet_call)
    # --- Train ---
    text_encoder.train(); unet.train(); vae.eval()
    global_step = 0

    while global_step < args.max_steps:
        for batch in dataloader:
            with accelerator.accumulate(unet):
                # unwrap the TE to avoid PEFT forward kwargs mismatch on older stacks
                input_ids = batch["input_ids"].to(accelerator.device)
                te_unwrapped = accelerator.unwrap_model(text_encoder)
                if hasattr(te_unwrapped, "get_base_model"):
                    base_te = te_unwrapped.get_base_model()
                elif hasattr(te_unwrapped, "base_model"):
                    base_te = te_unwrapped.base_model
                else:
                    base_te = te_unwrapped
                text_hidden_states = base_te(input_ids=input_ids)[0]  # [B, T, C]

                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=dtype)
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_sched.config.num_train_timesteps, (latents.shape[0],),
                    device=latents.device
                ).long()
                noisy_latents = noise_sched.add_noise(latents, noise, timesteps)

                out = unet_call(noisy_latents, timesteps, text_hidden_states)
                model_pred = out.sample if hasattr(out, "sample") else out[0]
                loss = torch.nn.functional.mse_loss(model_pred, noise)

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                if accelerator.is_local_main_process and global_step % args.log_every == 0:
                    print(f"Step {global_step}/{args.max_steps}  Loss: {loss.item():.4f}")
                if global_step >= args.max_steps:
                    break

    # --- Save LoRA weights (UNet + TE) ---
    if accelerator.is_main_process:
        out_file = os.path.join(args.output_dir, "pytorch_lora_weights.safetensors")
        unet_state = {f"unet.{k}": v.detach().cpu()
                      for k, v in accelerator.unwrap_model(unet).state_dict().items()
                      if "lora_" in k}
        te_state = {f"text_encoder.{k}": v.detach().cpu()
                    for k, v in accelerator.unwrap_model(text_encoder).state_dict().items()
                    if "lora_" in k}
        merged = {**unet_state, **te_state}
        save_file(merged, out_file)
        print(f"Training complete. LoRA weights saved to {out_file}")


if __name__ == "__main__":
    main()
