import os
from transformers import AutoTokenizer

# Define the base model and where to save the updated tokenizer
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "./artifacts/tokenizer"
NEW_TOKEN  = "<sks>"   # this is the new special token we want to add

def main():
    # Create the output folder if it doesn’t already exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load the tokenizer from the pretrained Stable Diffusion model
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer", use_fast=False)

    # Check if the new token is already in the vocabulary
    if NEW_TOKEN in tok.get_vocab():
        print(f"Token {NEW_TOKEN} already exists.")
    else:
        # If not, add it to the tokenizer’s vocabulary
        tok.add_tokens([NEW_TOKEN])
        print(f"Added token: {NEW_TOKEN}")

    # Save the updated tokenizer to the output directory
    tok.save_pretrained(OUTPUT_DIR)
    print(f"Saved tokenizer with {len(tok)} tokens -> {OUTPUT_DIR}")

# Run the main function when this file is executed directly
if __name__ == "__main__":
    main()
