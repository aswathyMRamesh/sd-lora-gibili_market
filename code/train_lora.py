# code/train_lora.py
import os, glob, random, argparse, math
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
# Args (default training settings, all customizable by CLI args)
# --------------------------
@dataclass
class Args:
    model_name_or_path: str = "runwayml/stable-diffusion-v1-5"  # base Stable Diffusion model
    data_dir: str = "./data/512"  # path to training images
    output_dir: str = "./lora_out"  # folder where LoRA weights will be saved
    tokenizer_dir: str = "./artifacts/tokenizer"  # tokenizer with new token
    instance_token: str = "<sks>"  # the special style token
    resolution: int = 512
    max_steps: int = 1200
    batch_size: int = 1
    lr: float = 5e-5
    lora_rank: int = 8  # LoRA rank (controls trainable params)
    seed: int = 42
    num_workers: int = 4
    log_every: int = 50
    float16: bool = True  # use fp16 training if available
    local_files_only: bool = False  # set True for offline training

    # Extra knobs for training stability
    grad_checkpointing: bool = True
    xformers: bool = True
    warmup_ratio: float = 0.05   # % of steps used for LR warmup
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

def parse_args():
    # Turn dataclass fields into command-line args
    p = argparse.ArgumentParser()
    for f in Args.__dataclass_fields__.keys():
        t = Args.__dataclass_fields__[f].type
        if t is bool:
            p.add_argument(f"--{f}", action="store_true")
        else:
            p.add_argument(f"--{f}")
    ns = p.parse_args()
    d = Args().__dict__.copy()
    # Cast parsed args to correct types
    for k, v in vars(ns).items():
        if v is not None:
            if isinstance(d[k], int): v = int(v)
            if isinstance(d[k], float): v = float(v)
            d[k] = v
    return Args(**d)

# --------------------------
# Dataset (loads images + fixed text prompt with instance token)
# --------------------------
class ImageDataset(Dataset):
    def __init__(self, folder, size, instance_token, tokenizer):
        # collect all image paths
        self.paths = sorted(sum([glob.glob(os.path.join(folder, ext))
                                 for ext in ("*.png","*.jpg","*.jpeg","*.webp")], []))
        if not self.paths:
            raise ValueError(f"No images found in {folder}")
        # image preprocessing / augmentation
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(p=0.5),  # randomly flip half of images
            transforms.ColorJitter(brightness=0.1, contrast=0.1),  # light color aug
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        # fixed prompt with the new style token
        prompt = f"a busy market, in {instance_token} style"
        self.input_ids = tokenizer(
            prompt, padding="max_length", truncation=True,
            max_length=tokenizer.model_max_length, return_tensors="pt"
        ).input_ids[0]

    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        # load and preprocess image
        img = Image.open(self.paths[idx]).convert("RGB")
        return {"pixel_values": self.transform(img), "input_ids": self.input_ids}

# --------------------------
# Utils
# --------------------------
def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def count_params(model):
    # counts trainable params only
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def add_lora_to_unet_with_peft(unet: UNet2DConditionModel, rank: int):
    # attach LoRA adapters to UNet attention layers
    cfg = PeftLoraConfig(
        r=rank, lora_alpha=rank, lora_dropout=0.0,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],  # UNet attention projection layers
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    unet = get_peft_model(unet, cfg)
    # enable training only on LoRA params
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

    # setup
    os.makedirs(args.output_dir, exist_ok=True)
    accelerator = Accelerator(gradient_accumulation_steps=1,
                              mixed_precision=("fp16" if args.float16 else "no"))
    set_seed(args.seed)

    dtype = torch.float16 if args.float16 else torch.float32

    # --- Load tokenizer & text encoder ---
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_name_or_path, subfolder="text_encoder",
        local_files_only=args.local_files_only
    )

    # resize text encoder embeddings if tokenizer size changed
    new_vocab = len(tokenizer)
    old_vocab = text_encoder.get_input_embeddings().num_embeddings
    if new_vocab != old_vocab:
        print(f"Resizing text encoder embeddings: {old_vocab} -> {new_vocab}")
        text_encoder.resize_token_embeddings(new_vocab)

    # sanity check that token exists
    tok_id = tokenizer.convert_tokens_to_ids(args.instance_token)
    if tok_id is None or tok_id < 0 or tok_id >= new_vocab:
        raise ValueError(
            f"Instance token {args.instance_token!r} not found in tokenizer. "
            f"Make sure you saved tokenizer with the new token added."
        )

    # --- Load VAE, UNet, Scheduler ---
    vae = AutoencoderKL.from_pretrained(
        args.model_name_or_path, subfolder="vae",
        local_files_only=args.local_files_only
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.model_name_or_path, subfolder="unet",
        local_files_only=args.local_files_only
    )
    noise_sched = DDPMScheduler.from_pretrained(
        args.model_name_or_path, subfolder="scheduler",
        local_files_only=args.local_files_only
    )

    # unwrap UNet if wrapped
    try:
        if hasattr(unet, "get_base_model"): unet = unet.get_base_model()
        elif hasattr(unet, "base_model"):   unet = unet.base_model
    except Exception:
        pass

    # move to correct dtype
    vae.to(dtype); unet.to(dtype); text_encoder.to(dtype)

    # enable xFormers for memory-efficient attention
    if args.xformers:
        try:
            unet.enable_xformers_memory_efficient_attention()
            print(">> xFormers memory efficient attention: ON")
        except Exception:
            print(">> xFormers not available; continuing without it.")

    # gradient checkpointing for saving VRAM
    if args.grad_checkpointing:
        try:
            unet.enable_gradient_checkpointing()
            print(">> UNet gradient checkpointing: ON")
        except Exception:
            print(">> Could not enable UNet gradient checkpointing.")
        try:
            text_encoder.gradient_checkpointing_enable()
            print(">> Text encoder gradient checkpointing: ON")
        except Exception:
            print(">> Could not enable TE gradient checkpointing.")

    # freeze all base model weights
    for m in (vae, unet, text_encoder):
        m.requires_grad_(False)

    # add LoRA to text encoder
    te_lora_cfg = PeftLoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank, lora_dropout=0.0,
        target_modules=["q_proj","k_proj","v_proj","out_proj"],
        bias="none", task_type=TaskType.FEATURE_EXTRACTION,
    )
    text_encoder = get_peft_model(text_encoder, te_lora_cfg)
    for n, p in text_encoder.named_parameters():
        p.requires_grad_(("lora_" in n))

    # add LoRA to UNet
    unet, unet_lora_params = add_lora_to_unet_with_peft(unet, args.lora_rank)

    # print number of trainable params
    if accelerator.is_main_process:
        print(f"Trainable params - Text Encoder: {count_params(text_encoder)}")
        print(f"Trainable params - UNet (LoRA only): {sum(p.numel() for p in unet_lora_params)}")

    # dataset + dataloader
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"Data folder not found: {args.data_dir}")
    dataset = ImageDataset(args.data_dir, args.resolution, args.instance_token, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=True
    )

    # optimizer (AdamW + cosine LR with warmup)
    te_params = [p for p in text_encoder.parameters() if p.requires_grad]
    params = te_params + list(unet_lora_params)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # prepare models for accelerate
    text_encoder, unet, vae, optimizer, dataloader = accelerator.prepare(
        text_encoder, unet, vae, optimizer, dataloader
    )

    # unwrap models (needed for calls)
    te_unwrapped = accelerator.unwrap_model(text_encoder)
    if hasattr(te_unwrapped, "get_base_model"):
        base_te = te_unwrapped.get_base_model()
    elif hasattr(te_unwrapped, "base_model"):
        base_te = te_unwrapped.base_model
    else:
        base_te = te_unwrapped

    unet_call = accelerator.unwrap_model(unet)
    if hasattr(unet_call, "get_base_model"):
        unet_call = unet_call.get_base_model()
    elif hasattr(unet_call, "base_model"):
        unet_call = unet_call.base_model

    # cosine LR scheduler with warmup
    warmup_steps = max(1, int(args.max_steps * args.warmup_ratio))
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, args.max_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # start training loop
    text_encoder.train(); unet.train(); vae.eval()
    global_step = 0

    while global_step < args.max_steps:
        for batch in dataloader:
            with accelerator.accumulate(unet):
                # get text embeddings
                input_ids = batch["input_ids"].to(accelerator.device)
                text_hidden_states = base_te(input_ids=input_ids)[0]

                # encode images into latent space
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=dtype)
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

                # add noise to latents
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_sched.config.num_train_timesteps, (latents.shape[0],),
                    device=latents.device
                ).long()
                noisy_latents = noise_sched.add_noise(latents, noise, timesteps)

                # predict noise with UNet
                out = unet_call(noisy_latents, timesteps, text_hidden_states)
                model_pred = out.sample if hasattr(out, "sample") else out[0]
                loss = torch.nn.functional.mse_loss(model_pred, noise)

                # backprop + optimizer step
                accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                # logging
                if accelerator.is_local_main_process and global_step % args.log_every == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    print(f"Step {global_step}/{args.max_steps}  Loss: {loss.item():.4f}  LR: {lr_now:.2e}")
                if global_step >= args.max_steps:
                    break

    # save LoRA weights for UNet + text encoder
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
