"""
Evaluation script for the Automated Insurance Claim Reviewer.

Compares the generated output (dataset/output.csv) against the sample
expected outputs (dataset/sample_claims.csv) and reports accuracy metrics.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys


VERDICT_FIELDS = [
    "evidence_standard_met",
    "claim_status",
    "valid_image",
    "severity",
]


def load_csv(path: str) -> list[dict[str, str]]:
    """Load a CSV file into a list of dicts."""
    if not os.path.isfile(path):
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def evaluate(output_path: str, expected_path: str) -> None:
    """Compare output against expected results and print metrics."""
    output_rows = load_csv(output_path)
    expected_rows = load_csv(expected_path)

    # Index output by (user_id, image_paths)
    output_index: dict[tuple[str, str], dict[str, str]] = {}
    for row in output_rows:
        key = (row["user_id"].strip(), row["image_paths"].strip())
        output_index[key] = row

    total = 0
    matched = 0
    field_matches: dict[str, int] = {f: 0 for f in VERDICT_FIELDS}
    missing = 0

    for exp in expected_rows:
        key = (exp["user_id"].strip(), exp["image_paths"].strip())
        total += 1

        if key not in output_index:
            missing += 1
            continue

        out = output_index[key]
        all_match = True
        for field in VERDICT_FIELDS:
            if field in exp and field in out:
                if exp[field].strip().lower() == out[field].strip().lower():
                    field_matches[field] += 1
                else:
                    all_match = False

        if all_match:
            matched += 1

    print("=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Total expected claims : {total}")
    print(f"  Found in output       : {total - missing}")
    print(f"  Missing from output   : {missing}")
    print(f"  Exact match (all fields): {matched}/{total} ({100 * matched / max(total, 1):.1f}%)")
    print()
    print("  Per-field accuracy:")
    for field in VERDICT_FIELDS:
        acc = 100 * field_matches[field] / max(total - missing, 1)
        print(f"    {field:30s} : {field_matches[field]}/{total - missing} ({acc:.1f}%)")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate claim review output against expected results.")
    parser.add_argument(
        "--output",
        default=os.path.join("..", "..", "dataset", "output.csv"),
        help="Path to the generated output CSV.",
    )
    parser.add_argument(
        "--expected",
        default=os.path.join("..", "..", "dataset", "sample_claims.csv"),
        help="Path to the expected results CSV.",
    )
    args = parser.parse_args()
    evaluate(args.output, args.expected)


if __name__ == "__main__":
    main()
