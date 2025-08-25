# code/train_lora.py
import os, glob, random
from dataclasses import dataclass
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from accelerate import Accelerator

from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import AutoTokenizer, CLIPTextModel
from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0

# PEFT for text encoder
from peft import LoraConfig as PeftLoraConfig, get_peft_model
from peft.utils import TaskType


# -----------------------------
# Args
# -----------------------------
@dataclass
class Args:
    model_name: str = "runwayml/stable-diffusion-v1-5"  # accept license & login on HF
    data_dir: str = "/content/drive/MyDrive/data/512"
    output_dir: str = "lora_out"
    instance_token: str = "<sks>"                      # style token in prompt
    resolution: int = 512
    max_steps: int = 800
    batch_size: int = 4
    lr: float = 1e-4
    lora_rank_unet: int = 8
    lora_rank_text: int = 8
    seed: int = 42
    num_workers: int = 2
    log_every: int = 50
    hf_token: str | None = None


# -----------------------------
# Dataset
# -----------------------------
class ImageDataset(Dataset):
    def __init__(self, folder, size, instance_token, tokenizer):
        self.paths = sorted(sum([glob.glob(os.path.join(folder, e))
                                 for e in ("*.png","*.jpg","*.jpeg","*.webp")], []))
        if not self.paths:
            raise ValueError(f"No images found in {folder}")
        self.tf = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        prompt = f"a busy market, in {instance_token} style"
        self.input_ids = tokenizer(
            prompt, truncation=True, padding="max_length",
            max_length=tokenizer.model_max_length, return_tensors="pt"
        ).input_ids[0]

    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return {"pixel_values": self.tf(img), "input_ids": self.input_ids}


# -----------------------------
# Helpers
# -----------------------------
def set_seed(s):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def add_unet_lora(unet, rank: int = 8):
    """
    Attach LoRA layers to the UNet in a version-safe way.
    """
    attn_procs = {}
    lora_cls = LoRAAttnProcessor2_0 if "LoRAAttnProcessor2_0" in globals() else LoRAAttnProcessor

    for name in unet.attn_processors.keys():
        attn_procs[name] = lora_cls()
    unet.set_attn_processor(attn_procs)

    # Set LoRA ranks
    for name, module in unet.attn_processors.items():
        if hasattr(module, "set_lora_layer"):
            module.set_lora_layer(rank)

    # Freeze base params, unfreeze LoRA
    for p in unet.parameters():
        p.requires_grad_(False)
    for n, p in unet.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)

    print(f"LoRA adapters with rank={rank} added to UNet successfully!")


# -----------------------------
# Main
# -----------------------------
def main():
    a = Args()
    os.makedirs(a.output_dir, exist_ok=True)
    accelerator = Accelerator(gradient_accumulation_steps=1)
    set_seed(a.seed)

    token = a.hf_token

    # Load base SD1.5 components
    tokenizer    = AutoTokenizer.from_pretrained(a.model_name, subfolder="tokenizer", use_fast=False, token=token)
    text_encoder = CLIPTextModel.from_pretrained(a.model_name, subfolder="text_encoder", token=token)
    vae          = AutoencoderKL.from_pretrained(a.model_name, subfolder="vae", token=token)
    unet         = UNet2DConditionModel.from_pretrained(a.model_name, subfolder="unet", token=token)
    noise_sched  = DDPMScheduler.from_pretrained(a.model_name, subfolder="scheduler", token=token)

    # Freeze VAE
    for p in vae.parameters():
        p.requires_grad_(False)

    # LoRA for text encoder
    te_cfg = PeftLoraConfig(
        r=a.lora_rank_text,
        lora_alpha=a.lora_rank_text,
        lora_dropout=0.0,
        target_modules=["q_proj","k_proj","v_proj","out_proj"],
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )
    text_encoder = get_peft_model(text_encoder, te_cfg)
    for n, p in text_encoder.named_parameters():
        p.requires_grad_(("lora_" in n))

    # LoRA for UNet
    add_unet_lora(unet, a.lora_rank_unet)

    # Dataset and loader
    ds = ImageDataset(a.data_dir, a.resolution, a.instance_token, tokenizer)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, drop_last=True,
                    num_workers=a.num_workers, pin_memory=True)

    # Optimizer
    params = [p for p in text_encoder.parameters() if p.requires_grad] + \
             [p for p in unet.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr)

    # Accelerator prepare
    text_encoder, unet, vae, opt, dl = accelerator.prepare(text_encoder, unet, vae, opt, dl)

    # Training
    unet.train()
    text_encoder.train()
    global_step = 0
    while global_step < a.max_steps:
        for batch in dl:
            with accelerator.accumulate(unet):
                # ---- Text encoding ----
                input_ids = batch["input_ids"].to(accelerator.device)
                text_outputs = text_encoder(input_ids=input_ids)
                encoder_hidden_states = text_outputs.last_hidden_state  # [B, T, C]

                # ---- Image encoding ----
                imgs = batch["pixel_values"].to(accelerator.device)
                latents = vae.encode(imgs).latent_dist.sample() * 0.18215

                # ---- Add noise ----
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_sched.config.num_train_timesteps, (latents.size(0),),
                    device=latents.device, dtype=torch.long
                )
                noisy_latents = noise_sched.add_noise(latents, noise, timesteps)

                # ---- UNet forward ----
                pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states
                ).sample

                # ---- Loss ----
                loss = torch.nn.functional.mse_loss(pred, noise)
                accelerator.backward(loss)
                opt.step()
                opt.zero_grad(set_to_none=True)

                global_step += 1
                if accelerator.is_local_main_process and global_step % a.log_every == 0:
                    print(f"step {global_step}/{a.max_steps}  loss={loss.item():.4f}")
                if global_step >= a.max_steps:
                    break

    # Save adapters
    if accelerator.is_main_process:
        unet_lora_dir = os.path.join(a.output_dir, "unet_lora")
        te_lora_dir   = os.path.join(a.output_dir, "text_encoder_lora")
        os.makedirs(unet_lora_dir, exist_ok=True)
        os.makedirs(te_lora_dir, exist_ok=True)

        unet.save_attn_procs(unet_lora_dir)
        text_encoder.save_pretrained(te_lora_dir)

        print("Saved UNet LoRA to:", unet_lora_dir)
        print("Saved Text Encoder LoRA to:", te_lora_dir)
        print("Done.")


if __name__ == "__main__":
    main()
