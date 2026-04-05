"""
Download and prepare the ThaiSum dataset for Qwen3 summarization SFT.

This script:
1. downloads `pythainlp/thaisum` from Hugging Face
2. keeps only the `body` and `summary` columns
3. removes empty rows
4. saves the cleaned dataset locally under `dataset/thaisum_hf`
5. exports JSONL snapshots for each split
6. converts each split to ShareGPT JSONL for LLaMA Factory
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from datasets import Dataset, DatasetDict, load_dataset


SYSTEM_PROMPT = (
    "คุณเป็นผู้ช่วยสรุปข่าวภาษาไทย\n"
    "สรุปเฉพาะจากข้อมูลในบทความต้นฉบับเท่านั้น\n"
    "เขียนสรุปให้กระชับ ชัดเจน และไม่เพิ่มข้อเท็จจริงที่ไม่มีในต้นฉบับ"
)


def is_valid_record(example: dict) -> bool:
    """Keep only rows with non-empty body and summary."""
    body = str(example.get("body") or "").strip()
    summary = str(example.get("summary") or "").strip()
    return bool(body and summary)


def normalize_dataset(dataset: DatasetDict) -> DatasetDict:
    """Keep only training columns and drop invalid records."""
    keep_columns = {"body", "summary"}
    first_split = next(iter(dataset.values()))
    drop_columns = [col for col in first_split.column_names if col not in keep_columns]

    if drop_columns:
        dataset = dataset.remove_columns(drop_columns)

    dataset = dataset.filter(is_valid_record, desc="Filtering empty body/summary rows")
    return dataset


def export_split_jsonl(split_dataset: Dataset, output_path: Path) -> None:
    """Export a dataset split to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_dataset.to_json(str(output_path), force_ascii=False)


def to_sharegpt_rows(rows: Iterable[dict]) -> Iterable[dict]:
    """Convert records into LLaMA Factory ShareGPT message format."""
    for row in rows:
        body = str(row["body"]).strip()
        summary = str(row["summary"]).strip()
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": body},
                {"role": "assistant", "content": summary},
            ]
        }


def export_sharegpt(split_dataset: Dataset, output_path: Path) -> None:
    """Write ShareGPT JSONL output for a split."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for row in to_sharegpt_rows(split_dataset):
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare ThaiSum summarization dataset")
    parser.add_argument("--dataset_name", default="pythainlp/thaisum", help="Hugging Face dataset id")
    parser.add_argument(
        "--raw_output_dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "thaisum_hf"),
        help="Directory to store the cleaned local HF dataset and JSONL exports",
    )
    parser.add_argument(
        "--sharegpt_output_dir",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "dataset",
            "format_dataset",
            "thai_news_summary",
        ),
        help="Directory to store ShareGPT JSONL files for LLaMA Factory",
    )
    args = parser.parse_args()

    raw_output_dir = Path(args.raw_output_dir)
    sharegpt_output_dir = Path(args.sharegpt_output_dir)

    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name)
    dataset = normalize_dataset(dataset)

    raw_output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(raw_output_dir))
    print(f"Saved Hugging Face dataset to: {raw_output_dir}")

    for split_name, split_dataset in dataset.items():
        jsonl_path = raw_output_dir / f"{split_name}.jsonl"
        export_split_jsonl(split_dataset, jsonl_path)
        print(f"Saved split JSONL: {jsonl_path} ({len(split_dataset)} rows)")

        sharegpt_path = sharegpt_output_dir / f"thai_news_summary_{split_name}.json"
        export_sharegpt(split_dataset, sharegpt_path)
        print(f"Saved ShareGPT JSONL: {sharegpt_path}")


if __name__ == "__main__":
    main()
