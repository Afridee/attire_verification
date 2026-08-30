# BR Attire Verification

Offline eval CLI that checks full-body BR attire photos:

1. MediaPipe Pose framing gate (visible nose + ankles, min person pixel height) + person-centric upper / lower / feet / chest crops  
2. Marqo FashionSigLIP zero-shot scoring per crop  
3. Unified dress-code rule engine → `PASSED` / `FAILED` / `UNCERTAIN` / `REJECTED`

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
  --image photo.jpg --pretty

uv run python -m attire_verification verify \
  --image photo.jpg --debug-crops ./debug/
```

Flags:

| Flag | Description |
|------|-------------|
| `--image PATH` | Required input image |
| `--pretty` | Pretty-print JSON |
| `--debug-crops DIR` | Save `upper.jpg` / `lower.jpg` / `feet.jpg` / `chest.jpg` |
| `--min-confidence F` | Default `0.55` |
| `--min-margin F` | Default `0.08` (top1 − top2) |

### Batch folder eval

```bash
uv run python -m attire_verification batch \
  --dir "/Users/afridee/Downloads/BR Attire Verification_Sample Photos" \
  --output results.jsonl
```

Recursively finds `*.jpg` / `*.jpeg` / `*.png`. Ground truth is inferred from parent folder name:

- `Right Attire` → expected `PASSED`
- `Wrong Attire` → expected `FAILED`

Writes one JSON object per line to `--output` and prints an accuracy summary to stdout.

## Exit codes (`verify`)

| Code | Meaning |
|------|---------|
| `0` | `PASSED` |
| `1` | `FAILED`, `UNCERTAIN`, or `REJECTED` |

## Status values

| Status | Meaning |
|--------|---------|
| `PASSED` | Meets unified dress code |
| `FAILED` | Clear violation (sandals, casual shirt, wrong trousers, etc.) |
| `UNCERTAIN` | Low confidence / low margin / borderline footwear (sneakers) |
| `REJECTED` | Bad framing (not full body) — retake, not an attire fail |

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
