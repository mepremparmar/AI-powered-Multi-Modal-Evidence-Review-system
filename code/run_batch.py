"""
Batch claims processor using Gemini 2.5 Flash.

Reads claims, joins user history and evidence requirements, calls the
multimodal review API in groups (batches), and writes outputs to the output CSV.

This minimizes API calls and successfully works within the daily 20 request quota.
Supports loading and rotating through multiple API keys on 429 rate limit.
Supports resuming from a previous run via --resume.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from google.genai.errors import APIError

from models import ClaimInput, ClaimObject, EvidenceRequirements, UserHistory
from claim_reviewer import review_claims_batch


# ── Env Loader helper ─────────────────────────────────────────────────────────


def load_env_keys(path: str = ".env") -> list[str]:
    """Load API keys from .env file (GEMINI_API_KEYS or GEMINI_API_KEY)."""
    keys = []
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k in ("GEMINI_API_KEYS", "GEMINI_API_KEY"):
                        parts = [p.strip() for p in v.split(",")]
                        keys.extend([p for p in parts if p])
    return keys


# ── CSV Helper Functions ──────────────────────────────────────────────────────


def load_user_history(path: str) -> dict[str, dict[str, Any]]:
    """Load user history metadata indexed by user_id."""
    history = {}
    if not os.path.isfile(path):
        print(f"Warning: user_history.csv not found at {path}", file=sys.stderr)
        return history

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_id = row["user_id"].strip()
            # Split history flags
            flags_str = row.get("history_flags", "none")
            flags = []
            if flags_str and flags_str.lower() != "none":
                flags = [fl.strip() for fl in flags_str.split(";")]

            history[user_id] = {
                "past_claim_count": int(row.get("past_claim_count", 0)),
                "accepted_count": int(row.get("accept_claim", 0)),
                "rejected_count": int(row.get("rejected_claim", 0)),
                "manual_count": int(row.get("manual_review_claim", 0)),
                "history_flags": flags,
                "history_summary": row.get("history_summary", "").strip(),
            }
    return history


def load_evidence_requirements(path: str) -> list[dict[str, Any]]:
    """Load all evidence requirements from the CSV file."""
    requirements = []
    if not os.path.isfile(path):
        print(f"Warning: evidence_requirements.csv not found at {path}", file=sys.stderr)
        return requirements

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            requirements.append(
                {
                    "requirement_id": row["requirement_id"].strip(),
                    "claim_object": row["claim_object"].strip().lower(),
                    "applies_to": row["applies_to"].strip(),
                    "minimum_image_evidence": row["minimum_image_evidence"].strip(),
                }
            )
    return requirements


def build_evidence_notes(requirements: list[dict[str, Any]], claim_object: str) -> str:
    """Format requirements matching the claim object into a notes string."""
    matched = []
    for req in requirements:
        obj = req["claim_object"]
        if obj == "all" or obj == claim_object:
            matched.append(f"- {req['requirement_id']}: {req['minimum_image_evidence']}")
    return "Minimum evidence standards:\n" + "\n".join(matched)


def resolve_image_paths(paths_str: str, dataset_dir: str) -> list[str]:
    """Split and resolve image paths to exist on disk."""
    raw_paths = [p.strip() for p in paths_str.split(";")]
    resolved = []
    for p in raw_paths:
        if not p:
            continue
        # Try direct path
        if os.path.isfile(p):
            resolved.append(p)
        else:
            # Try path relative to the dataset folder
            rel_path = os.path.join(dataset_dir, p)
            if os.path.isfile(rel_path):
                resolved.append(rel_path)
            else:
                # Fallback to current dir check or the original string
                resolved.append(p)
    return resolved


def _extract_retry_delay(error: APIError) -> float | None:
    """Try to extract the recommended retry delay from a 429 error message."""
    msg = str(error)
    # Match patterns like "Please retry in 27.576351371s" or "retryDelay: 27s"
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", msg, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


# ── Resume helpers ────────────────────────────────────────────────────────────


def load_existing_results(output_path: str) -> dict[tuple[str, str], dict[str, str]]:
    """
    Load previous output CSV results, keyed by (user_id, image_paths).
    Returns a dict mapping each claim key to its output row dict.
    """
    results: dict[tuple[str, str], dict[str, str]] = {}
    if not os.path.isfile(output_path):
        return results

    with open(output_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["user_id"].strip(), row["image_paths"].strip())
            results[key] = row
    return results


def is_successful_result(row: dict[str, str]) -> bool:
    """Check if a previously-written output row was a successful API result."""
    return row.get("claim_status_justification", "").strip() != "Review failed due to API errors."


# ── Batch Runner ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch claim verification script powered by Gemini 2.5 Flash."
    )
    parser.add_argument(
        "--claims-csv",
        default="../dataset/claims.csv",
        help="Path to the claims input CSV file.",
    )
    parser.add_argument(
        "--history-csv",
        default="../dataset/user_history.csv",
        help="Path to the user history CSV file.",
    )
    parser.add_argument(
        "--requirements-csv",
        default="../dataset/evidence_requirements.csv",
        help="Path to the evidence requirements CSV.",
    )
    parser.add_argument(
        "--output-csv",
        default="../dataset/output.csv",
        help="Path to save output results CSV.",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model to use.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Gemini API Key (optional, splits on commas, falls back to .env GEMINI_API_KEYS).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of claims processed (useful for testing).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of claims to group in a single Gemini API request (default: 5).",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=8,
        help="Number of retry attempts on API errors.",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=2,
        help="Initial backoff delay in seconds.",
    )
    parser.add_argument(
        "--rate-delay",
        type=int,
        default=5,
        help="Seconds to wait between batch API calls.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume mode: skip claims that were previously successful, only retry failures.",
    )

    args = parser.parse_args()

    # ── Verify dataset directory ───────────────────────────────────────────
    dataset_dir = os.path.dirname(args.claims_csv) or "dataset"

    # ── Load API Keys ──────────────────────────────────────────────────────
    api_keys = []
    if args.api_key:
        api_keys = [k.strip() for k in args.api_key.split(",") if k.strip()]

    if not api_keys:
        api_keys = load_env_keys()

    if not api_keys:
        env_key = os.environ.get("GEMINI_API_KEY")
        if env_key:
            api_keys = [env_key]

    if not api_keys:
        print(
            "[ERROR] No Gemini API keys found. "
            "Please provide them via --api-key, .env, or GEMINI_API_KEY environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[KEYS] Loaded {len(api_keys)} API key(s) for rotation.", file=sys.stderr)

    # ── Load metadata ──────────────────────────────────────────────────────
    print("[INFO] Loading metadata...", file=sys.stderr)
    histories = load_user_history(args.history_csv)
    requirements = load_evidence_requirements(args.requirements_csv)
    print(f"   Loaded history records: {len(histories)}", file=sys.stderr)
    print(f"   Loaded requirement definitions: {len(requirements)}", file=sys.stderr)

    # ── Read claims ────────────────────────────────────────────────────────
    if not os.path.isfile(args.claims_csv):
        print(f"[ERROR] Claims file not found: {args.claims_csv}", file=sys.stderr)
        sys.exit(1)

    claims_to_process = []
    with open(args.claims_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            claims_to_process.append(row)

    if args.limit:
        claims_to_process = claims_to_process[: args.limit]

    total_claims = len(claims_to_process)

    # ── Define output headers ──────────────────────────────────────────────
    output_headers = [
        "user_id",
        "image_paths",
        "user_claim",
        "claim_object",
        "evidence_standard_met",
        "evidence_standard_met_reason",
        "risk_flags",
        "issue_type",
        "object_part",
        "claim_status",
        "claim_status_justification",
        "supporting_image_ids",
        "valid_image",
        "severity",
    ]

    # Prepare output file (resume vs clean start)
    output_file_path = Path(args.output_csv)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    output_rows_by_key = {}
    if args.resume and os.path.isfile(output_file_path):
        prev_results = load_existing_results(args.output_csv)
        for key, row in prev_results.items():
            if is_successful_result(row):
                output_rows_by_key[key] = [row.get(h, "") for h in output_headers]
        print(f"[RESUME] Loaded {len(output_rows_by_key)} successful results.", file=sys.stderr)
    else:
        # Clear/initialize the file with headers
        with open(output_file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(output_headers)

    # Filter out claims that have already been processed successfully
    claims_to_call = []
    for claim_row in claims_to_process:
        user_id = claim_row["user_id"].strip()
        raw_image_paths = claim_row["image_paths"].strip()
        claim_key = (user_id, raw_image_paths)
        if claim_key not in output_rows_by_key:
            claims_to_call.append(claim_row)

    total_claims_to_call = len(claims_to_call)

    # Group remaining claims into batches
    batch_size = args.batch_size
    batches = [claims_to_call[i:i + batch_size] for i in range(0, len(claims_to_call), batch_size)]
    total_batches = len(batches)

    print(
        f"[START] Starting batch processing of {total_claims_to_call} pending claims in {total_batches} batches (batch size: {batch_size})...",
        file=sys.stderr,
    )

    success_count = len(output_rows_by_key)
    fail_count = 0
    current_key_idx = 0

    for b_idx, batch in enumerate(batches, start=1):
        print(
            f"\n[BATCH] Batch [{b_idx}/{total_batches}] Processing {len(batch)} claims...",
            file=sys.stderr,
        )

        # 1. Resolve and construct claim inputs for this batch
        batch_inputs = []
        for claim_row in batch:
            user_id = claim_row["user_id"].strip()
            raw_image_paths = claim_row["image_paths"].strip()
            user_claim_text = claim_row["user_claim"].strip()
            claim_obj_val = claim_row["claim_object"].strip().lower()

            resolved_images = resolve_image_paths(raw_image_paths, dataset_dir)

            user_hist_data = histories.get(
                user_id,
                {
                    "past_claim_count": 0,
                    "accepted_count": 0,
                    "rejected_count": 0,
                    "manual_count": 0,
                    "history_flags": [],
                    "history_summary": "No claim history.",
                },
            )

            history_obj = UserHistory(
                past_claim_count=user_hist_data["past_claim_count"],
                accepted_count=user_hist_data["accepted_count"],
                rejected_count=user_hist_data["rejected_count"],
                manual_count=user_hist_data["manual_count"],
                history_flags=user_hist_data["history_flags"],
                history_summary=user_hist_data["history_summary"],
            )

            notes = build_evidence_notes(requirements, claim_obj_val)
            req_obj = EvidenceRequirements(
                min_resolution="640x480",
                required_visibility="claimed_part_clearly_visible",
                notes=notes,
            )

            claim_input = ClaimInput(
                claim_object=ClaimObject(claim_obj_val),
                user_claim=user_claim_text,
                image_paths=resolved_images,
                user_history=history_obj,
                evidence_requirements=req_obj,
            )
            batch_inputs.append(claim_input)

        # 2. Call batch API with retries and key rotation
        verdicts = None
        delay = args.retry_delay
        keys_exhausted_this_batch = set()

        for attempt in range(1, args.retry_attempts + 1):
            current_key = api_keys[current_key_idx]
            try:
                verdicts = review_claims_batch(
                    batch_inputs,
                    model=args.model,
                    api_key=current_key,
                )
                if len(verdicts) != len(batch):
                    raise ValueError(f"Expected {len(batch)} results, got {len(verdicts)}")
                break
            except APIError as e:
                is_rate_limit = False
                if hasattr(e, "code") and e.code == 429:
                    is_rate_limit = True
                elif "quota" in str(e).lower() or "limit" in str(e).lower() or "429" in str(e):
                    is_rate_limit = True

                if is_rate_limit:
                    keys_exhausted_this_batch.add(current_key_idx)
                    next_key_idx = (current_key_idx + 1) % len(api_keys)

                    if len(keys_exhausted_this_batch) < len(api_keys):
                        print(
                            f"   [WARNING] Rate limit (429) hit for key index {current_key_idx}. "
                            f"Rotating to key index {next_key_idx}...",
                            file=sys.stderr,
                        )
                        current_key_idx = next_key_idx
                        time.sleep(1)
                        continue
                    else:
                        server_delay = _extract_retry_delay(e)
                        if server_delay and server_delay > delay:
                            wait_time = server_delay + 5
                        else:
                            wait_time = delay

                        print(
                            f"   [WARNING] All API keys rate-limited. Waiting {wait_time:.0f}s before retrying...",
                            file=sys.stderr,
                        )
                        time.sleep(wait_time)
                        delay = min(delay * 2, 120)
                        keys_exhausted_this_batch.clear()
                        current_key_idx = next_key_idx
                else:
                    print(
                        f"   [WARNING] API Error on attempt {attempt}/{args.retry_attempts}: {e}. Retrying in {delay}s...",
                        file=sys.stderr,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 120)
            except Exception as e:
                print(
                    f"   [WARNING] Unexpected error on attempt {attempt}/{args.retry_attempts}: {e}. Retrying in {delay}s...",
                    file=sys.stderr,
                )
                time.sleep(delay)
                delay = min(delay * 2, 120)

        # 3. Store results incrementally in memory
        for c_idx, claim_row in enumerate(batch):
            user_id = claim_row["user_id"].strip()
            raw_image_paths = claim_row["image_paths"].strip()
            user_claim_text = claim_row["user_claim"].strip()
            claim_obj_val = claim_row["claim_object"].strip().lower()
            claim_key = (user_id, raw_image_paths)

            if verdicts is None:
                print(
                    f"   [ERROR] Failed to process claim for {user_id} due to API failures.",
                    file=sys.stderr,
                )
                fail_count += 1
                out_row = [
                    user_id,
                    raw_image_paths,
                    user_claim_text,
                    claim_obj_val,
                    "false",
                    "Review failed due to API errors.",
                    "manual_review_required",
                    "unknown",
                    "unknown",
                    "not_enough_information",
                    "Review failed due to API errors.",
                    "none",
                    "false",
                    "unknown",
                ]
            else:
                verdict = verdicts[c_idx]
                print(f"   [SUCCESS] Claim for {user_id} ({claim_obj_val}) reviewed successfully.")
                success_count += 1
                out_row = [
                    user_id,
                    raw_image_paths,
                    user_claim_text,
                    claim_obj_val,
                    verdict.evidence_standard_met,
                    verdict.evidence_standard_met_reason,
                    verdict.risk_flags,
                    verdict.issue_type,
                    verdict.object_part,
                    verdict.claim_status,
                    verdict.claim_status_justification,
                    verdict.supporting_image_ids,
                    verdict.valid_image,
                    verdict.severity,
                ]
            output_rows_by_key[claim_key] = out_row

        # Overwrite file to update progress and preserve order
        try:
            with open(output_file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(output_headers)
                for original_row in claims_to_process:
                    orig_key = (original_row["user_id"].strip(), original_row["image_paths"].strip())
                    if orig_key in output_rows_by_key:
                        writer.writerow(output_rows_by_key[orig_key])
                    else:
                        writer.writerow([
                            original_row["user_id"].strip(),
                            original_row["image_paths"].strip(),
                            original_row["user_claim"].strip(),
                            original_row["claim_object"].strip().lower(),
                            "false",
                            "Review failed due to API errors.",
                            "manual_review_required",
                            "unknown",
                            "unknown",
                            "not_enough_information",
                            "Review failed due to API errors.",
                            "none",
                            "false",
                            "unknown",
                        ])
        except PermissionError:
            print(
                f"   [WARNING] File {output_file_path} is locked by another program. "
                "Progress is saved in memory but could not write to disk. Please close any programs using this file.",
                file=sys.stderr,
            )

        # 4. Pacing wait between batches
        if b_idx < total_batches and verdicts is not None:
            print(
                f"   [PACING] Waiting {args.rate_delay}s before next batch...",
                file=sys.stderr,
            )
            time.sleep(args.rate_delay)

    print("\n[FINISH] Batch processing complete!", file=sys.stderr)
    print(f"   Success count: {success_count}", file=sys.stderr)
    print(f"   Failure count: {fail_count}", file=sys.stderr)
    print(f"   Output saved to: {args.output_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
