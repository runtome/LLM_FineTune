"""
Prepare ASR correction dataset for LLaMA Factory SFT training.

Reads paired CSVs (asr_output.csv + train.csv/val.csv), matches rows by
their path column, and outputs ShareGPT-format JSON files.

Both CSVs share the same paths (same session/topic/mic/chunk filenames).
The asr_output.csv contains raw Whisper ASR output (potentially with errors),
while train.csv/val.csv contains the ground truth transcription.
"""

import csv
import os
import json
import argparse


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


def load_csv(csv_path: str) -> dict[str, str]:
    """Load CSV and return {path: sentence} dict, filtering junk rows."""
    rows = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = row["path"]
            sentence = row.get("sentence", "")
            # Filter out junk rows (paths starting with '._')
            if path.startswith("._"):
                continue
            # Skip empty sentences
            if not sentence or not sentence.strip():
                continue
            rows[path] = sentence
    return rows


def build_sharegpt(asr_text: str, corrected_text: str) -> dict:
    """Build a single ShareGPT-format entry."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": asr_text.strip()},
            {"role": "assistant", "content": corrected_text.strip()},
        ]
    }


def process_split(asr_csv: str, gt_csv: str, output_path: str, split_name: str) -> int:
    """Process one data split (train or val)."""
    print(f"\n{'='*60}")
    print(f"Processing {split_name} split")
    print(f"  ASR CSV:    {asr_csv}")
    print(f"  GT CSV:     {gt_csv}")
    print(f"{'='*60}")

    asr_rows = load_csv(asr_csv)
    gt_rows = load_csv(gt_csv)

    print(f"  ASR rows (after filtering): {len(asr_rows)}")
    print(f"  GT rows (after filtering):  {len(gt_rows)}")

    # Match by path
    matched_keys = sorted(set(asr_rows.keys()) & set(gt_rows.keys()))
    unmatched_asr = len(asr_rows) - len(matched_keys)
    unmatched_gt = len(gt_rows) - len(matched_keys)

    print(f"  Matched rows: {len(matched_keys)}")
    if unmatched_asr > 0:
        print(f"  Unmatched ASR rows: {unmatched_asr}")
    if unmatched_gt > 0:
        print(f"  Unmatched GT rows: {unmatched_gt}")

    # Build ShareGPT entries
    entries = []
    for path in matched_keys:
        entries.append(build_sharegpt(asr_rows[path], gt_rows[path]))

    # Write output (JSONL format)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"  Output: {output_path} ({len(entries)} entries)")

    # Print sample
    if entries:
        print(f"\n  Sample entry:")
        print(f"    Path:      {matched_keys[0]}")
        try:
            user_text = entries[0]["messages"][1]["content"]
            asst_text = entries[0]["messages"][2]["content"]
            print(f"    User:      {user_text[:80]}...")
            print(f"    Assistant: {asst_text[:80]}...")
        except UnicodeEncodeError:
            print("    (Thai text sample omitted due to console encoding)")

    return len(entries)


def main():
    parser = argparse.ArgumentParser(description="Prepare ASR correction dataset for SFT")
    parser.add_argument(
        "--data_dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets"),
        help="Root directory containing train/val/test subdirs",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "dataset", "format_dataset", "asr_correction",
        ),
        help="Output directory for JSON files",
    )
    args = parser.parse_args()

    total = 0

    # Train split
    total += process_split(
        asr_csv=os.path.join(args.data_dir, "train", "asr_output.csv"),
        gt_csv=os.path.join(args.data_dir, "train", "train.csv"),
        output_path=os.path.join(args.output_dir, "asr_correction_train.json"),
        split_name="train",
    )

    # Validation split
    total += process_split(
        asr_csv=os.path.join(args.data_dir, "val", "asr_output.csv"),
        gt_csv=os.path.join(args.data_dir, "val", "val.csv"),
        output_path=os.path.join(args.output_dir, "asr_correction_val.json"),
        split_name="val",
    )

    print(f"\n{'='*60}")
    print(f"Total entries: {total}")
    print(f"Output dir: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
