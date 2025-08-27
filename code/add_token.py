import os
from transformers import AutoTokenizer

MODEL_NAME = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "./artifacts/tokenizer"
NEW_TOKEN  = "<sks>"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer", use_fast=False)
    if NEW_TOKEN in tok.get_vocab():
        print(f"Token {NEW_TOKEN} already exists.")
    else:
        tok.add_tokens([NEW_TOKEN])
        print(f"Added token: {NEW_TOKEN}")
    tok.save_pretrained(OUTPUT_DIR)
    print(f"Saved tokenizer with {len(tok)} tokens -> {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
