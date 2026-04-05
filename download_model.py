"""
Download LLM weights from HuggingFace for offline use on Lanta.
Run this on the login node (which has internet) before submitting training jobs.

Usage:
    # Download Qwen3-8B (default)
    python download_model.py --save_dir /project/zz991000-zdeva/zz991010/MeetingTranscription/model/qwen3-8b

    # Download a different model
    python download_model.py --model Qwen/Qwen3-4B --save_dir ./model/qwen3-4b

    # With HuggingFace token (for gated models)
    python download_model.py --hf_token YOUR_TOKEN --save_dir ./model/qwen3-8b
"""

import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer


def download_model(model_name, save_dir, hf_token=None):
    print(f"Downloading model: {model_name}")
    print(f"Saving to: {save_dir}")

    print("\n[1/2] Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(save_dir)
    print(f"  Tokenizer saved to {save_dir}")

    print("\n[2/2] Downloading model weights (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    model.save_pretrained(save_dir)
    print(f"  Model saved to {save_dir}")

    print(f"\nDone! Use '{save_dir}' as model_name_or_path in YAML configs.")


def main():
    parser = argparse.ArgumentParser(description="Download LLM weights for offline training on Lanta")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="HuggingFace model name (default: Qwen/Qwen3-8B)")
    parser.add_argument("--save_dir", required=True, help="Local directory to save model weights")
    parser.add_argument("--hf_token", default=None, help="HuggingFace token (for gated models)")
    args = parser.parse_args()

    download_model(args.model, args.save_dir, args.hf_token)


if __name__ == "__main__":
    main()
