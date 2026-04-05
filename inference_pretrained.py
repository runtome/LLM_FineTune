"""
ASR Correction Inference Script (Pretrained Model).

Uses the base pretrained Qwen3-8B model with sophisticated prompting
and few-shot examples to correct ASR errors, hallucinations, and
mangled English words in Thai transcriptions.
"""

import argparse
import os
import re

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "คุณคือผู้ช่วยแก้ไขข้อความจากระบบ ASR (Automatic Speech Recognition) ภาษาไทย\n"
    "กฎการแก้ไข:\n"
    "1. ลบวลีที่ซ้ำกัน 3 ครั้งขึ้นไปติดต่อกัน (hallucination จาก ASR)\n"
    "2. ลบเนื้อหาที่ไม่เกี่ยวข้องหรือไม่สมเหตุสมผลที่ ASR สร้างขึ้นมาเอง\n"
    "3. แก้คำไทยที่ผิดจากเสียงคล้ายกัน โดยดูจากบริบท เช่น อิสลาก→อิสระ, แว่นน้ำ→ว่ายน้ำ\n"
    "4. แก้คำภาษาอังกฤษที่สะกดผิดหรือถูกตัดคำ เช่น Outdoo→Outdoor, Indoo→Indoor, Wave→Microwave (ดูจากบริบท)\n"
    "5. ห้ามเพิ่มหรือลบเนื้อหา ห้ามเรียบเรียงใหม่ แก้เฉพาะข้อผิดพลาดเท่านั้น\n"
    "6. คงคำอุทานไว้ เช่น ครับ ค่ะ อืม เออ อ่า\n"
    "7. ตอบเฉพาะข้อความที่แก้ไขแล้ว ไม่ต้องอธิบาย"
)

FEW_SHOT_EXAMPLES = [
    {
        "user": "กีฬาไว้น้ำ แล้วก็ชอบความเป็นอิสลาก",
        "assistant": "กีฬาว่ายน้ำ แล้วก็ชอบความเป็นอิสระ",
    },
    {
        "user": "ชอบเล่น Outdoo กับ Indoo ครับ แล้วก็ชอบไปเที่ยว",
        "assistant": "ชอบเล่น Outdoor กับ Indoor ครับ แล้วก็ชอบไปเที่ยว",
    },
    {
        "user": "ใช้ Wave อุ่นอาหารทุกวันเลยครับ",
        "assistant": "ใช้ Microwave อุ่นอาหารทุกวันเลยครับ",
    },
]


def load_model(model_name: str, cache_dir: str | None = None):
    """Load model and tokenizer from HF cache."""
    print(f"Loading model: {model_name}")
    if cache_dir:
        print(f"Cache dir: {cache_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model loaded: {model.config._name_or_path}")
    return model, tokenizer


def build_messages(text: str, enable_thinking: bool = True, use_few_shot: bool = True) -> list:
    """Build chat messages with system prompt and optional few-shot examples."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if use_few_shot:
        for example in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": example["user"]})
            messages.append({"role": "assistant", "content": example["assistant"]})

    user_content = text if enable_thinking else text + "/no_think"
    messages.append({"role": "user", "content": user_content})

    return messages


def postprocess(original: str, corrected: str) -> str:
    """Post-process the corrected text to catch remaining issues."""
    if not corrected.strip():
        return original

    # Strip any "corrected:" or similar prefix the model might prepend
    corrected = re.sub(r"^(?:corrected|แก้ไข|ข้อความที่แก้ไข)\s*[:：]\s*", "", corrected, flags=re.IGNORECASE)

    # Character n-gram repetition detection for Thai (no spaces):
    # Detect 10+ char sequences repeated 3+ consecutive times, collapse to one
    corrected = re.sub(r"(.{10,}?)\1{2,}", r"\1", corrected)

    # Length ratio guard: if corrected is >2.5x original length, return original
    if len(corrected) > 2.5 * len(original) and len(original) > 0:
        return original

    return corrected.strip()


@torch.inference_mode()
def inference(
    text: str,
    model,
    tokenizer,
    max_new_tokens: int = 512,
    enable_thinking: bool = False,
    use_few_shot: bool = True,
) -> str:
    """Run inference on a single ASR text and return corrected text."""
    messages = build_messages(text, enable_thinking=enable_thinking, use_few_shot=use_few_shot)

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        repetition_penalty=1.1,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

    if enable_thinking:
        # Parse output: strip thinking tokens (151668 = </think>)
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True)
    else:
        content = tokenizer.decode(output_ids, skip_special_tokens=True)

    return postprocess(text, content.strip())


def main():
    parser = argparse.ArgumentParser(description="ASR Correction Inference (Pretrained Model)")
    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen3-8B",
        help="HF model name to load from cache (default: Qwen/Qwen3-8B). "
             "Examples: Qwen/Qwen3-8B, Qwen/Qwen2.5-7B",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="HF cache directory (default: uses HF_HOME/TRANSFORMERS_CACHE env var)",
    )
    parser.add_argument(
        "--input_csv",
        default=os.path.join(os.path.dirname(__file__), "datasets", "test", "asr_ouput_test.csv"),
        help="Input CSV with columns: path, sentence",
    )
    parser.add_argument(
        "--output_csv",
        default=os.path.join(os.path.dirname(__file__), "datasets", "test", "test_pretrained_inference.csv"),
        help="Output CSV path",
    )
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens for generation")
    parser.add_argument("--think", action="store_true", help="Enable thinking mode (slower but may improve quality)")
    parser.add_argument("--no_few_shot", action="store_true", help="Skip few-shot examples")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_name, cache_dir=args.cache_dir)

    enable_thinking = args.think
    use_few_shot = not args.no_few_shot
    print(f"Thinking mode: {'enabled' if enable_thinking else 'disabled'}")
    print(f"Few-shot examples: {'enabled' if use_few_shot else 'disabled'}")

    # Read input
    df = pd.read_csv(args.input_csv)
    print(f"Input rows: {len(df)}")

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Correcting ASR"):
        path = row["path"]
        sentence = str(row["sentence"]) if pd.notna(row["sentence"]) else ""

        if sentence.strip():
            corrected = inference(
                sentence,
                model,
                tokenizer,
                max_new_tokens=args.max_new_tokens,
                enable_thinking=enable_thinking,
                use_few_shot=use_few_shot,
            )
        else:
            corrected = ""

        print(f"\n[{path}]")
        print(f"  IN:  {sentence}")
        print(f"  OUT: {corrected}")

        results.append({"path": path, "sentence": sentence, "corrected_sentence": corrected})

    # Save output
    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved {len(out_df)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
