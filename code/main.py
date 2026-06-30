"""
CLI entry point for the Automated Insurance Claim Reviewer.

Usage
-----
    # Review a car claim with two images:
    python main.py \
        --claim-object car \
        --user-claim "My front bumper got dented in a parking lot collision." \
        --images img_1.jpg img_2.jpg

    # Full options with user history:
    python main.py \
        --claim-object laptop \
        --user-claim "The screen cracked when it fell off my desk." \
        --images screen_photo.png \
        --past-claims 3 \
        --accepted 2 \
        --rejected 1 \
        --history-flags user_history_risk \
        --history-summary "User has filed 3 claims in 6 months."

    # Load claim from a JSON file:
    python main.py --from-json claim_input.json

Environment
-----------
    Set GEMINI_API_KEY in your environment or pass --api-key.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from models import ClaimInput, ClaimObject, EvidenceRequirements, UserHistory
from claim_reviewer import review_claim, review_claim_raw


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="claim-reviewer",
        description="Automated Insurance Claim Reviewer — powered by Claude Vision.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Input modes ────────────────────────────────────────────────────────
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--from-json",
        metavar="FILE",
        help="Load the full ClaimInput from a JSON file.",
    )
    mode.add_argument(
        "--claim-object",
        choices=["car", "laptop", "package"],
        help="Type of object being claimed.",
    )

    # ── Claim details (used with --claim-object) ───────────────────────────
    p.add_argument(
        "--user-claim",
        default="",
        help="Chat transcript or claim description.",
    )
    p.add_argument(
        "--images",
        nargs="+",
        default=[],
        metavar="PATH",
        help="One or more image file paths.",
    )

    # ── User history (optional) ────────────────────────────────────────────
    p.add_argument("--past-claims", type=int, default=0)
    p.add_argument("--accepted", type=int, default=0)
    p.add_argument("--rejected", type=int, default=0)
    p.add_argument("--manual", type=int, default=0)
    p.add_argument(
        "--history-flags",
        nargs="*",
        default=[],
        help="History risk flags (e.g. user_history_risk manual_review_required).",
    )
    p.add_argument("--history-summary", default="")

    # ── Evidence requirements (optional) ───────────────────────────────────
    p.add_argument("--min-resolution", default="640x480")
    p.add_argument(
        "--required-visibility", default="claimed_part_clearly_visible"
    )

    # ── Model / API options ────────────────────────────────────────────────
    p.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model name (default: gemini-2.5-flash).",
    )
    p.add_argument("--api-key", default=None, help="Google Gemini API key.")
    p.add_argument("--max-tokens", type=int, default=1024)

    # ── Output options ─────────────────────────────────────────────────────
    p.add_argument(
        "--raw",
        action="store_true",
        help="Return raw dict instead of validated Pydantic model.",
    )
    p.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Pretty-print the JSON output (default: True).",
    )
    p.add_argument(
        "--output",
        metavar="FILE",
        help="Write JSON output to a file instead of stdout.",
    )

    return p


def _load_from_json(path: str) -> ClaimInput:
    """Load and validate a ClaimInput from a JSON file."""
    file_path = Path(path)
    if not file_path.is_file():
        print(f"Error: File not found: {path}", file=sys.stderr)
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return ClaimInput(**data)


def _build_from_args(args: argparse.Namespace) -> ClaimInput:
    """Build a ClaimInput from CLI arguments."""
    if not args.user_claim:
        print("Error: --user-claim is required with --claim-object.", file=sys.stderr)
        sys.exit(1)
    if not args.images:
        print("Error: --images is required with --claim-object.", file=sys.stderr)
        sys.exit(1)

    return ClaimInput(
        claim_object=ClaimObject(args.claim_object),
        user_claim=args.user_claim,
        image_paths=args.images,
        user_history=UserHistory(
            past_claim_count=args.past_claims,
            accepted_count=args.accepted,
            rejected_count=args.rejected,
            manual_count=args.manual,
            history_flags=args.history_flags,
            history_summary=args.history_summary,
        ),
        evidence_requirements=EvidenceRequirements(
            min_resolution=args.min_resolution,
            required_visibility=args.required_visibility,
        ),
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # ── Build claim input ──────────────────────────────────────────────────
    if args.from_json:
        claim = _load_from_json(args.from_json)
    else:
        claim = _build_from_args(args)

    # ── Run the review ─────────────────────────────────────────────────────
    print("🔍 Reviewing claim...", file=sys.stderr)
    print(f"   Object : {claim.claim_object.value}", file=sys.stderr)
    print(f"   Images : {len(claim.image_paths)}", file=sys.stderr)

    try:
        if args.raw:
            result = review_claim_raw(
                claim,
                model=args.model,
                max_tokens=args.max_tokens,
                api_key=args.api_key,
            )
        else:
            result_model = review_claim(
                claim,
                model=args.model,
                max_tokens=args.max_tokens,
                api_key=args.api_key,
            )
            result = result_model.model_dump()
    except FileNotFoundError as e:
        print(f"\n❌ Image error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Review failed: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Output ─────────────────────────────────────────────────────────────
    indent = 2 if args.pretty else None
    json_output = json.dumps(result, indent=indent, ensure_ascii=False)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json_output, encoding="utf-8")
        print(f"\n✅ Result written to {args.output}", file=sys.stderr)
    else:
        print(json_output)

    # ── Summary to stderr ──────────────────────────────────────────────────
    status = result.get("claim_status", "unknown")
    severity = result.get("severity", "unknown")
    flags = result.get("risk_flags", "none")

    emoji = {"supported": "✅", "contradicted": "❌", "not_enough_information": "⚠️"}
    print(
        f"\n{emoji.get(status, '❓')} Status: {status} | "
        f"Severity: {severity} | Flags: {flags}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
