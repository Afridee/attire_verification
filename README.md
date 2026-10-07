# BR Attire Verification

Offline eval CLI that checks full-body BR attire photos:

1. MediaPipe Pose framing gate (visible nose + ankles, min person pixel height) + person-centric upper / lower / feet / chest crops
2. Marqo FashionSigLIP zero-shot scoring per crop, plus image-to-image match of the upper crop against official black-polo reference photos
3. Unified dress-code rule engine → score `N/100` (25 points each for chest, feet, upper, lower)

The CLI is for local eval. `POST /verify` is the same pipeline, exposed so a Nest backend can call it.

Internals: [docs/ATTIRE_VERIFICATION_GUIDE.md](docs/ATTIRE_VERIFICATION_GUIDE.md).

## Setup

Requires Python >= 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
cd /Volumes/T7-Dev/attire_verification
uv sync
```

First run downloads Marqo FashionSigLIP weights (~150MB+).

Pinned to `mediapipe==0.10.21` (classic Pose solutions API) because newer MediaPipe Tasks builds can abort on macOS Metal.

## Commands

### Single image

```bash
uv run python -m attire_verification verify \
  --image "/path/to/photo.jpg"

uv run python -m attire_verification verify \
  --image photo.jpg --role BR --pretty

uv run python -m attire_verification verify \
  --image photo.jpg --debug-crops ./debug/
```

Flags:

| Flag | Description |
|------|-------------|
| `--image PATH` | Required input image |
| `--pretty` | Pretty-print JSON |
| `--debug-crops DIR` | Save `upper.jpg` / `lower.jpg` / `feet.jpg` / `chest.jpg` |
| `--min-confidence F` | Default `0.40` |
| `--min-margin F` | Default `0.08` (top1 − top2; skipped when both are allowed items) |
| `--polo-match-threshold F` | Default `0.85` (cosine vs official polo refs) |
| `--polo-refs DIR` | Override bundled official-polo upper crops |
| `--role ROLE` | Staff role (default `BR`). `BR` and `BR_SUP` require a visible ID badge |

JSON includes `score`, `failReasons`, `regions`, `regionPoints` (each region `0` or `25`), `role` (also set on framing rejects), and `poloMatch` when scoring completes:

```json
"poloMatch": {
  "score": 0.9012,
  "threshold": 0.85,
  "matched": true,
  "refs": 2
}
```

### Batch folder eval

```bash
uv run python -m attire_verification batch \
  --dir "/Users/afridee/Downloads/BR Attire Verification_Sample Photos" \
  --output results.jsonl
```

Recursively finds `*.jpg` / `*.jpeg` / `*.png`. Ground truth is inferred from parent folder name:

- `Right Attire` → expected `100/100`
- `Wrong Attire` → expected `<100`

Each JSONL line is a verify result plus `expected` and `match`. Framing rejects (`incomplete_body_in_frame`) count as `rejected`, not scored.

Shared flags: `--min-confidence`, `--min-margin`, `--polo-match-threshold`, `--polo-refs`, and `--role` (one role for every image).

Stdout prints `total`, `scored`, `correct`, `accuracy`, `falsePass`, `falseFail`, `rejected`, and `output`.

### Client PDF report

Scores photo folders, writes HTML in a temp directory, converts it to PDF with Chrome, and keeps only the PDF.

```bash
uv run python scripts/generate_client_report.py
```

Defaults: `/Users/afridee/Downloads/Photos` and `/Users/afridee/Downloads/BR Attire Verification_Sample Photos`, role `FC` (ID badge not required), output `reports/attire_verification_report.pdf`.

```bash
uv run python scripts/generate_client_report.py \
  --dir "/Users/afridee/Downloads/Photos" \
  --dir "/Users/afridee/Downloads/BR Attire Verification_Sample Photos" \
  --role FC \
  --output reports/attire_verification_report.pdf
```

Requires Google Chrome for the HTML → PDF step. `--dir` can be repeated.

## HTTP API

Same check as `verify`, for Nest (or any other caller).

```bash
uv run python -m uvicorn attire_verification.api:app --host 0.0.0.0 --port 8000
```

```bash
curl -s -X POST http://localhost:8000/verify \
  -H "X-API-Key: $SERVICE_API_KEY" \
  -F "image=@/path/to/photo.jpg" \
  -F "role=BR"
```

| Call | Behavior |
|------|----------|
| `POST /verify` | Multipart `image` (JPEG or PNG, max 25MB) and `role` (default `BR`) |
| `GET /health` | Liveness, no auth |
| `X-API-Key` | Required only when `SERVICE_API_KEY` is set. Nest should send the same value |

The JSON body matches the CLI: `score`, `failReasons`, `regions`, `regionPoints`, `role`, `poloMatch`. Models load once at startup and inference runs one photo at a time.

Docker (CPU image, weights baked in at build):

```bash
DOCKER_BUILDKIT=1 docker build -t attire-verification .
docker run -p 8000:8000 -e SERVICE_API_KEY=... attire-verification
```

`GET /docs` is the OpenAPI page.

## Exit codes (`verify`)

| Code | Meaning |
|------|---------|
| `0` | `100/100` (all four regions passed) |
| `1` | Score below 100, or framing rejected (`0/100` with `incomplete_body_in_frame`) |

## Score

Each region is worth **25 points**. A region scores 25 only when it clearly meets dress code; violations, low confidence, and missing crops score 0.

| Region | Pass (25) | Zero (0) |
|--------|-----------|----------|
| **upper** | Tucked-in solid-colour formal shirt (white, light blue, mint, beige, navy, light gray), or official polo (text or ref match ≥ 0.85) | Casual or non-formal shirt, low confidence |
| **lower** | Black, navy, grey, or light grey trousers | Casual trousers, low confidence |
| **feet** | Closed shoes, loafers, or single-color sober sneakers | Sandals; bright/neon shoes when confidence ≥ `0.70` and margin ≥ `0.08` (weaker neon guesses are ignored); low confidence |
| **chest** | Visible ID badge (`BR` / `BR_SUP`); auto-pass for other roles | Missing badge (BR / BR_SUP), low confidence |

Example: chest, feet, and lower pass but upper fails → `"score": "75/100"`.

Framing failures (not full body) return `"score": "0/100"` with `failReasons: ["incomplete_body_in_frame"]` — retake, not an attire fail.

Official polo is the **black polo with purple sleeve trim**. The upper crop is scored against that visual label, then compared to bundled reference crops (`src/attire_verification/refs/official_polo/`). Cosine ≥ `0.85` promotes the shirt to official polo even if the text labels are split.

## Sample photos

```
/Users/afridee/Downloads/BR Attire Verification_Sample Photos/Right Attire/
/Users/afridee/Downloads/BR Attire Verification_Sample Photos/Wrong Attire/
```

Example:

```bash
uv run python -m attire_verification verify \
  --image "/Users/afridee/Downloads/BR Attire Verification_Sample Photos/Wrong Attire/DHKN-1.jpeg" \
  --pretty
```

## Tests

No GPU / MediaPipe. Covers the rule engine (`tests/test_rules.py`), role parsing and ID-badge rules (`tests/test_roles.py`), and framing / crop-prep helpers (`tests/test_pose_framing.py`).

```bash
uv run pytest
```
