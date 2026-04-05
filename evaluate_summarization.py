"""
Evaluate generated summaries against references with ROUGE metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_predictions(path: str) -> pd.DataFrame:
    """Load CSV or JSONL prediction file."""
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported evaluation file format: {suffix}")


def compute_rouge(df: pd.DataFrame, prediction_column: str, reference_column: str) -> dict[str, float]:
    """Return mean ROUGE F1 scores."""
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise ImportError(
            "Missing dependency `rouge_score`. Install it with `pip install rouge-score` "
            "or run `bash setup.sh` in this repository."
        ) from exc

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    valid_rows = 0
    for _, row in df.iterrows():
        prediction = str(row.get(prediction_column) or "").strip()
        reference = str(row.get(reference_column) or "").strip()
        if not prediction or not reference:
            continue

        scores = scorer.score(reference, prediction)
        for metric in totals:
            totals[metric] += scores[metric].fmeasure
        valid_rows += 1

    if valid_rows == 0:
        raise ValueError("No valid prediction/reference pairs were found.")

    return {metric: round(total / valid_rows, 6) for metric, total in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Thai summarization predictions")
    parser.add_argument(
        "--prediction_path",
        default="dataset/thaisum_hf/validation_predictions.csv",
        help="CSV or JSONL file containing generated summaries and references",
    )
    parser.add_argument("--prediction_column", default="generated_summary", help="Prediction column name")
    parser.add_argument("--reference_column", default="summary", help="Reference column name")
    parser.add_argument("--output_path", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    prediction_df = load_predictions(args.prediction_path)
    metrics = compute_rouge(prediction_df, args.prediction_column, args.reference_column)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved metrics to: {output_path}")


if __name__ == "__main__":
    main()
