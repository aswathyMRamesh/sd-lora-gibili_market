# sd-lora-gibili_market
Fine-tune Stable Diffusion 1.5 with LoRA to create images in the Gibili_Market style.
1 · DATASET
original/ : full-resolution PNG/JPG reference images

512/ : 512 × 512 crops ready to train

dataset is added in the branch

2 · TASK
Add a new style token <sks> to the tokenizer.

Finetune both the UNet and the text encoder with LoRA so that the prompt: "a busy market, in <sks> style" produces Ghibli-like images.

Submit runnable code (with clear instructions on how to run the scripts) and the resulting LoRA weights.

3 · IMPLEMENTATION REQUIREMENTS
Libraries: diffusers, peft, PyTorch 2+, safetensors

Training script: code/train_lora.py (or notebook)
• saves exactly one file → lora_out/pytorch_lora_weights.safetensors

Evaluation script: code/eval_lora.py
• loads base SD 1.5 and your adapter
• renders at least three images for the prompt above (baseline optional)

Suggested hyper-parameters (feel free to change):
MODEL_NAME = runwayml/stable-diffusion-v1-5
INSTANCE_TOKEN = <sks>
RESOLUTION = 512 
LORA_RANK = 8 
LR = 1e-4 
MAX_STEPS = 800
