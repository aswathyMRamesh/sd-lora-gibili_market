Ghibli Market · LoRA Style (SD 1.5)

This project fine-tunes Stable Diffusion 1.5 with LoRA to render images in <sks> style.

1. Setup
Create environment
    micromamba create -y -n sd-lora python=3.10
    micromamba activate sd-lora
    pip install -r requirements.txt

2. Add the <sks> token
(Run once)
    python code/add_token.py \
  --model_name_or_path runwayml/stable-diffusion-v1-5 \
  --token "<sks>" \
  --output_dir ./artifacts/tokenizer

3. Training
Online mode (auto-download from Hugging Face Hub)
    python code/train_lora.py \
  --model_name_or_path "runwayml/stable-diffusion-v1-5" \
  --tokenizer_dir "./artifacts/tokenizer" \
  --data_dir "./data/data/512" \
  --output_dir "./lora_out" \
  --batch_size 1 --max_steps 800 --log_every 10 --num_workers 0
  
Offline mode (local model)
    python code/train_lora.py \
  --model_name_or_path "./cache/sd15/sd15" \
  --tokenizer_dir "./artifacts/tokenizer" \
  --data_dir "./data/data/512" \
  --output_dir "./lora_out" \
  --batch_size 1 --max_steps 800 --log_every 10 --num_workers 0

Output:
    lora_out/pytorch_lora_weights.safetensors

4. Evaluation / Sampling
    python code/eval_lora.py \
  --model_name_or_path "runwayml/stable-diffusion-v1-5" \
  --tokenizer_dir "./artifacts/tokenizer" \
  --lora_path "./lora_out/pytorch_lora_weights.safetensors" \
  --output_dir "./samples" \
  --num_images 3 --steps 30 --guidance 7.5 --seed 1234

Images will be saved under:
    samples/sks_00.png
    samples/sks_01.png
    samples/sks_02.png

5. Notes
	•	For offline use, download SD 1.5 to ./cache/sd15/sd15 and include ./artifacts/tokenizer.
	•	For reproducibility, keep the same seed, steps, guidance, and resolution.
	•	Both text encoder and UNet LoRA weights are saved into one .safetensors file.
