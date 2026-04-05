"""
ASR Correction Inference Script.

Loads a fine-tuned model and corrects raw ASR transcriptions.
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


def load_model(model_path: str):
    """Load model and tokenizer."""
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model loaded: {model.config._name_or_path}")
    return model, tokenizer


def postprocess(original: str, corrected: str) -> str:
    """Guard against LLM making things worse."""
    if not corrected.strip():
        return original

    # Strip any prefix the model might prepend
    corrected = re.sub(r"^(?:corrected|แก้ไข|ข้อความที่แก้ไข)\s*[:：]\s*", "", corrected, flags=re.IGNORECASE)

    # Collapse long repeated sequences (hallucination from LLM)
    corrected = re.sub(r"(.{10,}?)\1{2,}", r"\1", corrected)

    # Length guard: if corrected is >1.5x or <0.3x original length, keep original
    if len(original) > 0:
        ratio = len(corrected) / len(original)
        if ratio > 1.5 or ratio < 0.3:
            return original

    return corrected.strip()


def inference(text: str, model, tokenizer, max_new_tokens: int = 512) -> str:
    """Run inference on a single ASR text and return corrected text."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text + "/no_think"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.001,
        repetition_penalty=1.1,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

    # Parse output: strip thinking tokens (151668 = </think>)
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return postprocess(text, content.strip())


def main():
    parser = argparse.ArgumentParser(description="ASR Correction Inference")
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to fine-tuned merged model",
    )
    parser.add_argument(
        "--input_csv",
        default=os.path.join(os.path.dirname(__file__), "datasets", "test", "asr_ouput_test.csv"),
        help="Input CSV with columns: path, sentence",
    )
    parser.add_argument(
        "--output_csv",
        default=os.path.join(os.path.dirname(__file__), "datasets", "test", "test_LLM_inferance.csv"),
        help="Output CSV path",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (currently sequential)")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens for generation")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path)

    # Read input
    df = pd.read_csv(args.input_csv)
    print(f"Input rows: {len(df)}")

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Correcting ASR"):
        path = row["path"]
        sentence = str(row["sentence"]) if pd.notna(row["sentence"]) else ""

        if sentence.strip():
            corrected = inference(sentence, model, tokenizer, args.max_new_tokens)
        else:
            corrected = ""

        results.append({"path": path, "sentence": sentence, "corrected_sentence": corrected})

    # Save output
    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved {len(out_df)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
