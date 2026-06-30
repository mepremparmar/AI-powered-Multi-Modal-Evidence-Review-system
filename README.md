# Automated Insurance Claim Reviewer

A standalone Python tool that verifies insurance damage claims using submitted images and claim context. Powered by **Gemini 2.5 Flash**, it follows a rigorous 10-step analysis process to produce structured JSON verdicts.

---

## Approach Overview

### Problem

The dataset contains **44 insurance claims**, each with associated images. A naive implementation sends one Gemini API request per claim, but the **Gemini 2.5 Flash free tier limits usage to 20 requests per day (RPD)**. Processing all 44 claims individually would exceed the quota on the first run.

### Solution: Batched Multi-Claim Requests

Instead of sending one claim per request, this tool **groups multiple claims into a single API request** and asks Gemini to return a JSON array of results — one entry per claim.

**How it works:**

1. All 44 claims are read from `dataset/claims.csv` and joined with `dataset/user_history.csv` and `dataset/evidence_requirements.csv`.
2. Claims are grouped into batches (default: **5 claims per batch**).
3. For each batch, all claim texts and associated images are combined into a **single multimodal prompt**.
4. Gemini receives the prompt and returns a **structured JSON array** with one verdict object per claim.
5. The script parses the response and maps each result back to its corresponding claim by index.

**Impact on API usage:**

| Strategy | Total API Requests |
|---|---|
| One claim per request | 44 requests (exceeds 20 RPD) |
| 5 claims per batch | ~9 requests |
| 8 claims per batch | ~6 requests |
| 10 claims per batch | ~5 requests |

This keeps the entire dataset well within the free-tier quota in a single run.

### Additional Reliability Features

- **Key Rotation** — Supply multiple API keys via `.env`. On a 429 rate-limit error, the script automatically rotates to the next key and retries.
- **Resume Mode** — If a run is interrupted (quota hit, network error, etc.), re-run with `--resume` to skip already-successful claims and retry only the failures.
- **Exponential Backoff** — Retries with increasing delays on transient errors.
- **Pacing Delay** — Configurable wait between batch requests to avoid hitting per-minute rate limits.

---

## Setup Instructions

### Prerequisites

- **Python 3.10+**
- A **Google Gemini API key** with access to `gemini-2.5-flash` ([get one here](https://aistudio.google.com/apikey))

### 1. Clone / download the project

```bash
cd Hackerrank_Orchestrate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `google-genai` — Google Gemini SDK
- `Pillow` — Image loading
- `pydantic` — Data validation models

### 3. Configure your API key

Create a `.env` file in the project root:

```env
# Single key
GEMINI_API_KEY=AIzaSy...

# Or multiple comma-separated keys for automatic rotation on rate limits
GEMINI_API_KEYS=AIzaSy_key1,AIzaSy_key2,AIzaSy_key3
```

Alternatively, set it as an environment variable:

```bash
# Windows PowerShell
$env:GEMINI_API_KEY = "AIzaSy..."

# Linux / macOS
export GEMINI_API_KEY="AIzaSy..."
```

### 4. Prepare the dataset

Ensure the `dataset/` folder contains:

| File | Description |
|---|---|
| `claims.csv` | Input claims with `user_id`, `image_paths`, `user_claim`, `claim_object` |
| `user_history.csv` | Past claim history per user |
| `evidence_requirements.csv` | Minimum evidence standards per claim type |
| `images/` | Folder containing all referenced claim images |

---

## Usage

### Batch Processing (Full Dataset)

Process all claims in the dataset:

```bash
python run_batch.py
```

This reads `dataset/claims.csv`, batches 5 claims per request, and writes results to `dataset/output.csv`.

### Customize batch size

```bash
# 8 claims per request (recommended for fewer API calls)
python run_batch.py --batch-size 8

# 10 claims per request (maximum recommended)
python run_batch.py --batch-size 10
```

### Resume an interrupted run

If the script was stopped mid-way (quota hit, crash, etc.), resume without re-processing successful claims:

```bash
python run_batch.py --resume
```

### All CLI options for `run_batch.py`

| Flag | Default | Description |
|---|---|---|
| `--claims-csv` | `dataset/claims.csv` | Path to the input claims CSV |
| `--history-csv` | `dataset/user_history.csv` | Path to user history CSV |
| `--requirements-csv` | `dataset/evidence_requirements.csv` | Path to evidence requirements CSV |
| `--output-csv` | `dataset/output.csv` | Path to save output results |
| `--model` | `gemini-2.5-flash` | Gemini model name |
| `--api-key` | *(from .env)* | API key(s), comma-separated |
| `--batch-size` | `5` | Claims per API request |
| `--limit` | *(all)* | Process only first N claims (for testing) |
| `--retry-attempts` | `8` | Max retry attempts on API errors |
| `--retry-delay` | `2` | Initial backoff delay in seconds |
| `--rate-delay` | `5` | Seconds to wait between batch API calls |
| `--resume` | `false` | Skip previously successful claims |

### Single Claim Review (CLI)

Review a single claim directly:

```bash
python main.py \
    --claim-object car \
    --user-claim "My front bumper got dented in a parking lot." \
    --images photos/img_1.jpg photos/img_2.jpg
```

Or from a JSON file:

```bash
python main.py --from-json examples/car_claim.json
```

### Programmatic Usage

```python
from models import ClaimInput, ClaimObject, UserHistory, EvidenceRequirements
from claim_reviewer import review_claim

claim = ClaimInput(
    claim_object=ClaimObject.CAR,
    user_claim="My front bumper got dented in a parking lot collision.",
    image_paths=["photos/img_1.jpg", "photos/img_2.jpg"],
    user_history=UserHistory(
        past_claim_count=2,
        accepted_count=1,
        rejected_count=0,
        manual_count=1,
        history_flags=[],
        history_summary="Normal claim history.",
    ),
    evidence_requirements=EvidenceRequirements(
        min_resolution="640x480",
        required_visibility="claimed_part_clearly_visible",
    ),
)

result = review_claim(claim)
print(result.model_dump_json(indent=2))
```

---

## Output Format

Every review returns a JSON object with exactly these fields:

| Field | Type | Description |
|---|---|---|
| `evidence_standard_met` | `"true"` / `"false"` | Whether images meet the minimum evidence bar |
| `evidence_standard_met_reason` | string | One-sentence explanation |
| `risk_flags` | `"none"` or `"flag1;flag2"` | Detected risk flags (semicolon-separated) |
| `issue_type` | string | Damage type: `dent`, `scratch`, `crack`, etc. |
| `object_part` | string | Claimed part: `front_bumper`, `screen`, `box`, etc. |
| `claim_status` | string | `supported`, `contradicted`, or `not_enough_information` |
| `claim_status_justification` | string | 1-2 sentence justification referencing image IDs |
| `supporting_image_ids` | string | Image IDs supporting the decision, or `"none"` |
| `valid_image` | `"true"` / `"false"` | Whether images are usable for automated review |
| `severity` | string | `none`, `low`, `medium`, `high`, or `unknown` |

### Example output

```json
{
  "evidence_standard_met": "true",
  "evidence_standard_met_reason": "img_1 clearly shows the full front bumper with visible damage.",
  "risk_flags": "none",
  "issue_type": "dent",
  "object_part": "front_bumper",
  "claim_status": "supported",
  "claim_status_justification": "img_1 shows a deep dent with paint chipping on the front bumper, consistent with the claimed parking lot collision damage.",
  "supporting_image_ids": "img_1",
  "valid_image": "true",
  "severity": "medium"
}
```

---

## Risk Flags Reference

| Flag | Meaning |
|---|---|
| `blurry_image` | Image is too blurry to evaluate |
| `cropped_or_obstructed` | Claimed part is cropped out or blocked |
| `low_light_or_glare` | Poor lighting or glare prevents evaluation |
| `wrong_angle` | Angle doesn't show the claimed part |
| `wrong_object` | Image shows a different object type |
| `wrong_object_part` | Image shows a different part than claimed |
| `damage_not_visible` | No damage visible on the claimed part |
| `claim_mismatch` | Mismatch between claim text and visual evidence |
| `possible_manipulation` | Signs of image tampering |
| `non_original_image` | Stock photo, screenshot, or non-original image |
| `text_instruction_present` | Prompt injection detected in claim or image |
| `user_history_risk` | User history flags indicate elevated risk |
| `manual_review_required` | Claim should be escalated to manual review |

---

## Project Structure

```
├── README.md                         # You are here
├── code/                             # Build your solution here
│   ├── main.py                       # Suggested terminal entry point
│   ├── claim_reviewer.py             # Core review engine (single + batch)
│   ├── models.py                     # Pydantic data models & enums
│   ├── run_batch.py                  # Batch processor with resume & key rotation
│   ├── requirements.txt              # Python dependencies
│   ├── .env                          # API key(s) configuration
│   └── evaluation/
│       └── main.py                   # Suggested evaluation entry point
└── dataset/
    ├── sample_claims.csv             # Inputs + expected outputs for development
    ├── claims.csv                    # Inputs only; run your system on these rows
    ├── user_history.csv              # Historical claim counts and risk context
    ├── evidence_requirements.csv     # Minimum image evidence requirements
    ├── output.csv                    # Generated review results
    └── images/
        ├── sample/                   # Images referenced by sample_claims.csv
        └── test/                     # Images referenced by claims.csv
```

---

## Features

- **Gemini 2.5 Flash Vision** — Multimodal analysis via the `google-genai` SDK
- **Batched processing** — Groups 5-10 claims per API request to stay within free-tier limits
- **Multi-object support** — Cars, laptops, and packages
- **Multi-image analysis** — Evaluates each image independently
- **Prompt injection detection** — Flags manipulation attempts in claims or images
- **Multi-language support** — Handles Hindi, Spanish, mixed-language claims
- **User history awareness** — Risk-flags high-frequency claimants
- **Key rotation** — Automatic rotation across multiple API keys on 429 errors
- **Resume support** — Skip already-evaluated claims to save daily quota
- **Structured JSON output** — Machine-readable verdicts with 10 fields
