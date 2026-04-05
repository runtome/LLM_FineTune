"""
Batch summarization inference for Thai text using a merged fine-tuned model.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "คุณเป็นผู้ช่วยสรุปข้อความภาษาไทย\n"
    "สรุปให้กระชับ ชัดเจน และยึดตามข้อมูลในต้นฉบับเท่านั้น\n"
    "ห้ามเพิ่มข้อเท็จจริงใหม่ และตอบเป็นข้อความสรุปอย่างเดียว"
)


def load_input(path: str) -> pd.DataFrame:
    """Load CSV or JSONL input."""
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported input format: {suffix}")


def load_model(model_path: str):
    """Load tokenizer and merged model."""
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    return model, tokenizer


def postprocess_summary(text: str) -> str:
    """Remove common prefixes occasionally added by the model."""
    text = text.strip()
    text = re.sub(r"^(?:สรุป|summary)\s*[:：-]\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


@torch.inference_mode()
def summarize_text(
    body: str,
    model,
    tokenizer,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Generate one summary."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": body + "\n/no_think"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=1.1,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    return postprocess_summary(output_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch inference for Thai summarization")
    parser.add_argument("--model_path", required=True, help="Path to merged fine-tuned model")
    parser.add_argument(
        "--input_path",
        default=os.path.join(os.path.dirname(__file__), "dataset", "thaisum_hf", "validation.jsonl"),
        help="Input file (.csv or .jsonl) with a text column",
    )
    parser.add_argument(
        "--output_path",
        default=os.path.join(os.path.dirname(__file__), "dataset", "thaisum_hf", "validation_predictions.csv"),
        help="Path to write model predictions",
    )
    parser.add_argument("--text_column", default="body", help="Column containing source text")
    parser.add_argument("--reference_column", default="summary", help="Optional reference-summary column")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.2, help="Generation temperature")
    args = parser.parse_args()

    input_df = load_input(args.input_path)
    if args.text_column not in input_df.columns:
        raise ValueError(f"Missing text column: {args.text_column}")

    model, tokenizer = load_model(args.model_path)

    rows = []
    for _, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Generating summaries"):
        body = str(row[args.text_column] or "").strip()
        generated_summary = summarize_text(
            body=body,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        ) if body else ""

        result_row = {
            args.text_column: body,
            "generated_summary": generated_summary,
        }
        if args.reference_column in input_df.columns:
            result_row[args.reference_column] = row.get(args.reference_column, "")
        rows.append(result_row)

    output_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    output_df.to_csv(args.output_path, index=False)
    print(f"Saved predictions to: {args.output_path}")


if __name__ == "__main__":
    main()
