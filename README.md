# BR Attire Verification

Offline eval CLI that checks full-body BR attire photos:

1. MediaPipe Pose framing gate (visible nose + ankles, min person pixel height) + person-centric upper / lower / feet / chest crops  
2. Marqo FashionSigLIP zero-shot scoring per crop, plus image-to-image match of the upper crop against official black-polo reference photos  
3. Unified dress-code rule engine → score `N/100` (25 points each for chest, feet, upper, lower)

Not a production API — local eval only.

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

### Batch folder eval

```bash
uv run python -m attire_verification batch \
  --dir "/Users/afridee/Downloads/BR Attire Verification_Sample Photos" \
  --output results.jsonl
```

Recursively finds `*.jpg` / `*.jpeg` / `*.png`. Ground truth is inferred from parent folder name:

- `Right Attire` → expected `100/100`
- `Wrong Attire` → expected `<100`

Writes one JSON object per line to `--output` and prints an accuracy summary to stdout.

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

Requires Google Chrome for the HTML → PDF step.

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
| **feet** | Closed shoes, loafers, or single-color sober sneakers | Sandals, bright/neon shoes, low confidence |
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

Rule-engine unit tests (no GPU / MediaPipe):

```bash
uv run pytest
```
