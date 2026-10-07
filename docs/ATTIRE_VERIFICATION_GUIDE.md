# Attire Verification — Developer Guide

This document explains how the offline BR attire verification pipeline works: the framing gate, region crops, zero-shot scoring, official-polo reference matching, dress-code rules, and the tools/models involved.

## Pipeline overview

Verification runs in three stages:

```
Photo → PoseCropper → FashionSigLIP (+ polo ref match) → apply_rules → Score
              ↓
         0/100 (stops early if framing fails)
```

| Stage | Component | Output |
|-------|-----------|--------|
| 1 | `PoseCropper` | Region crops or `0/100` (`incomplete_body_in_frame`) |
| 2 | `FashionSigLIPScorer` | Label probabilities per region + polo similarity score |
| 3 | `apply_rules()` | `score` `N/100` (25 per region) |

Stage 2 now has **two scoring paths for the upper body**:

1. **Text labels** — zero-shot classification against fixed prompts (all regions)
2. **Reference images** — cosine similarity of the upper crop vs bundled official-polo photos

The reference match can **override** the text label and promote the shirt to official polo when similarity ≥ threshold (default `0.85`).

Run the full pipeline via CLI:

```bash
uv run python -m attire_verification verify \
  --image "/path/to/photo.jpg" \
  --role BR \
  --pretty \
  --debug-crops ./debug/
```

The `--debug-crops` flag saves `upper.jpg`, `lower.jpg`, `feet.jpg`, and `chest.jpg` for inspection.

The same pipeline is available over HTTP for the Nest backend: `POST /verify` with a multipart `image` and `role`. See the HTTP API section in the README.

### `verify` flags

| Flag | Default | Description |
|------|---------|-------------|
| `--image PATH` | — | Required input image |
| `--pretty` | off | Pretty-print JSON |
| `--debug-crops DIR` | — | Save region crop JPEGs |
| `--min-confidence F` | `0.40` | Minimum top-1 text score |
| `--min-margin F` | `0.08` | Minimum top1 − top2 margin (skipped when both labels are allowed) |
| `--polo-match-threshold F` | `0.85` | Min cosine similarity vs official polo refs |
| `--polo-refs DIR` | bundled | Override official-polo reference folder |
| `--role ROLE` | `BR` | Staff role. `BR` and `BR_SUP` require a visible ID badge |

`verify` exits `0` when the score is `100/100`, and `1` otherwise (partial score or framing reject).

### Batch eval

```bash
uv run python -m attire_verification batch \
  --dir "/path/to/photos" \
  --output results.jsonl
```

Recursively finds `*.jpg`, `*.jpeg`, and `*.png`. Ground truth comes from a parent folder name:

- `Right Attire` → expected `100/100`
- `Wrong Attire` → expected `<100`

Each JSONL line is a verify result plus `expected` and `match`. Framing rejects are counted as rejected, not scored. Stdout prints `total`, `scored`, `correct`, `accuracy`, `falsePass`, `falseFail`, `rejected`, and `output`.

Shared flags: `--min-confidence`, `--min-margin`, `--polo-match-threshold`, `--polo-refs`, and `--role` (one role for every image).

### Client PDF report

```bash
uv run python scripts/generate_client_report.py \
  --dir "/path/to/photos" \
  --role FC \
  --output reports/attire_verification_report.pdf
```

Scores each photo with `run_verify`, writes HTML in a temp directory, prints it to PDF with Google Chrome, and keeps only the PDF. `--dir` can be repeated. Defaults are `/Users/afridee/Downloads/Photos` and `/Users/afridee/Downloads/BR Attire Verification_Sample Photos`, role `FC` (ID badge not required), and `reports/attire_verification_report.pdf`.

---

## Score outcomes

Each of **chest**, **feet**, **upper**, and **lower** is worth **25 points**. The JSON `score` is `"N/100"`. A region earns 25 only when it clearly meets dress code.

| Score | Meaning | Typical cause |
|-------|---------|----------------|
| **0/100** (framing) | Photo unsuitable for verification | Bad framing, no pose, unreadable image (`incomplete_body_in_frame`) |
| **0–75/100** | One or more regions did not pass | Sandals, casual/non-formal shirt, casual trousers, bright/neon shoes, missing ID badge, low confidence |
| **100/100** | All four regions passed | Tucked-in formal shirt or official polo + black/navy/grey trousers + closed shoes, loafers, or sober sneakers (+ badge for BR / BR_SUP) |

Example: chest, feet, and lower pass but upper fails → `"score": "75/100"`.

JSON also includes `regionPoints` (`upper` / `lower` / `feet` / `chest`, each `0` or `25`) and `role` (the canonical role string). `role` is set on framing rejects as well.

JSON output may also include a `poloMatch` block when scoring completes:

```json
"poloMatch": {
  "score": 0.9012,
  "threshold": 0.85,
  "matched": true,
  "refs": 2
}
```

---

## Stage 1: Pose cropper (`pose_cropper.py`)

### Purpose

MediaPipe Pose detects body landmarks, checks that a **full body** is visible, crops the person, then extracts four clothing regions for downstream scoring.

### Key constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `VISIBILITY_THRESHOLD` | 0.5 | Nose and ankles must be this visible |
| `MIN_PERSON_HEIGHT_PX` | 250 | Minimum nose-to-ankle height in pixels |
| `PERSON_LANDMARK_VIS` | 0.3 | Threshold for building the person crop box |

### Framing gate (`evaluate_framing`)

Full-body check before any dress-code logic runs:

```python
def evaluate_framing(lms, width: int, height: int) -> tuple[bool, dict]:
    """Full-body gate: visible nose + ankles, and enough person pixels.

    Does not require the person to fill 55% of the *image*. Extra background
    around a complete figure is fine.
    """
```

**Passes when:**

- Nose visibility ≥ 0.5
- Both ankles visibility ≥ 0.5
- Person height (nose → lowest ankle) ≥ 250 px

**Fails when:**

- Head cut off (nose not visible)
- Feet cut off (either ankle not visible)
- Person too small in the image (< 250 px tall)

Extra background is fine — the person does **not** need to fill most of the frame.

### `PoseCropper.process()` flow

1. Load image with OpenCV (`cv2.imread`)
2. Run MediaPipe Pose on RGB image
3. Call `evaluate_framing()` — reject if framing fails
4. Build a person crop box with `person_crop_box()` (head/foot padding)
5. Re-run pose on the cropped person (or remap landmarks if re-detection fails)
6. Compute region boxes with `_region_boxes()`
7. Return `CropResult` with PIL crops and boxes mapped to original image coordinates

Early exits all return `score: "0/100"` with `failReasons: ["incomplete_body_in_frame"]`:

- Unreadable image
- No pose detected
- Framing check failed
- Could not build person crop box

### Region crops

| Region | Landmarks used | Purpose |
|--------|----------------|---------|
| **upper** | Shoulders + hips | Shirt / top (+ polo ref match) |
| **lower** | Hips + ankles | Trousers |
| **feet** | Ankles → bottom of person crop | Footwear |
| **chest** | Shoulders → 40% down torso | ID badge (scored separately) |

On success, `CropResult` contains:

- `crops`: PIL images keyed by region name
- `boxes`: `CropBox` coordinates in the original image
- `pose`: `PoseInfo` with framing details and crop metadata

---

## Stage 2: FashionSigLIP scoring (`siglip_scorer.py`)

### Model

| Detail | Value |
|--------|--------|
| Model | `Marqo/marqo-fashionSigLIP` |
| Loaded via | `open-clip-torch` |
| Hub path | `hf-hub:Marqo/marqo-fashionSigLIP` |
| Runtime | PyTorch (CUDA if available, else CPU) |
| First run | Downloads ~150MB+ weights from Hugging Face Hub |

Optional: set `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` in the environment or a repo `.env` for faster Hub downloads.

### How text scoring works

1. Preprocess each crop with `prepare_crop` (see below)
2. Encode the crop image → embedding
3. Compare to pre-encoded text labels for that region
4. Softmax over similarities → probabilities
5. Return `RegionScore` with `topLabel`, `topScore`, `secondLabel`, `secondScore`

`prepare_crop` runs before the model's square resize. Tall crops are handled first:

| Condition | Result |
|-----------|--------|
| Aspect (long ÷ short) ≥ `2.0` | Letterbox to 224×224 on gray `(128, 128, 128)` |
| Region is `upper` (and not tall) | Resize to `190×280` |
| Any other region with long side > `280` | Downscale so the long side is 280, keeping aspect |
| Otherwise | Leave the crop as-is |

A very tall upper crop is letterboxed, not forced to `190×280`.

### Official polo reference matching

Because FashionSigLIP has no built-in notion of "official" vs "non-official", the pipeline adds **image-to-image matching** for the upper crop:

```python
def match_official_polo(self, crop_pil: Image.Image) -> float:
    """Max cosine similarity of an upper crop vs official-polo reference embeddings."""
```

- Reference images live in `src/attire_verification/refs/official_polo/` (e.g. `ctg3_upper.jpg`, `ctg5_upper.jpg`)
- Override with `--polo-refs DIR`
- Each reference is encoded once at startup; the upper crop is compared via max cosine similarity
- Same shirt across photos typically scores ~0.88–0.92; other garments ~0.68–0.78 on the sample set
- An empty reference folder scores `0.0` (no promotion)

### Label sets (`labels.py`)

Polo labels use **visual descriptions** (not "official company polo"):

**Upper (8 labels):**

- white formal button-down shirt tucked in ← pass
- light blue formal button-down shirt tucked in ← pass
- light mint green formal button-down shirt tucked in ← pass
- beige formal button-down shirt tucked in ← pass
- navy formal button-down shirt tucked in ← pass
- light gray formal button-down shirt tucked in ← pass
- **black polo shirt with purple sleeve trim** ← official polo (visual) ← pass
- **casual or non-formal shirt** ← dump-bin fail (t-shirts, unofficial polos, untucked shirts, etc.)

**Lower (5 labels):**

- black trousers
- dark navy trousers
- grey trousers
- light grey trousers
- **casual trousers** ← jeans, shorts, chinos, cargo, or other non-formal bottoms

**Feet (5 labels):**

- formal closed shoes
- leather loafers
- single-color sober sneakers
- sandals or slides
- brightly colored or neon shoes

**Chest (2 labels):**

- blue lanyard with ID badge visible
- no ID badge visible

---

## Stage 3: Rule engine (`rules.py`)

Plain Python logic — no ML. Consumes `RegionScores` from FashionSigLIP and optional `polo_match_score` from reference matching.

### Official polo override

Before fail/pass checks, the rule engine may promote the shirt type:

```python
polo_matched = (
    polo_match_score is not None and polo_match_score >= polo_match_threshold
)
if polo_matched:
    shirt = ShirtType.OFFICIAL_POLO
```

When `polo_matched` is true:

- The shirt is treated as **official polo** even if text labels say `casual or non-formal shirt`
- **Upper-region confidence/margin gates are skipped** (text scores for similar polo prompts are often split)
- Lower and feet regions still require confidence ≥ 0.40. Margin ≥ 0.08 is required unless the second-best label is also allowed (black vs navy, loafers vs sneakers).

### Per-region scoring

Each region is judged independently (no whole-photo short-circuit):

```
Pass (25)  → region meets dress code with confident scores
Zero (0)   → clear violation, low confidence/margin, missing crop, or unknown label
Total      → upper + lower + feet + chest  (max 100)
```

### Thresholds

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `MIN_CONFIDENCE` | 0.40 | Top label score must be ≥ this |
| `MIN_MARGIN` | 0.08 | Gap between top1 and top2 must be ≥ this (skipped when both labels are allowed) |
| `BRIGHT_MIN_CONFIDENCE` | 0.70 | Bright/neon shoe dump-bin counts as a colour violation only at or above this **and** with margin ≥ `MIN_MARGIN` |
| `POLO_MATCH_THRESHOLD` | 0.85 | Cosine similarity vs ref crops to promote official polo |

Margin = `topScore - secondScore`.

### Roles

Staff role strings match the mobile `RoleInfo` constants. Default is `BR`.

| Role | ID badge required |
|------|-------------------|
| `BR`, `BR_SUP` | Yes — chest crop must score `blue lanyard with ID badge visible` to earn 25 |
| All other roles (`OM`, `FC`, `DXO`, …) | No — chest auto-passes (25) even without a badge |

### Allowed vs forbidden

| Region | Pass (25) | Zero (0) |
|--------|-----------|----------|
| **Shirt** | Solid-colour formal tucked in (white, light blue, mint, beige, navy, light gray), official polo (text or ref match) | Casual or non-formal shirt, low confidence |
| **Trousers** | Black, dark navy, grey, light grey | Casual trousers, low confidence |
| **Feet** | Closed shoes, loafers, single-color sober sneakers | Sandals/slides, bright/neon shoes (only if ≥ 0.70 and margin ≥ 0.08), low confidence |
| **Chest** (BR / BR_SUP only) | Blue lanyard with ID badge visible | No ID badge visible, missing crop, low badge confidence |

### Fail reasons

| Reason | Trigger |
|--------|---------|
| `casual_shirt` | Top label is `casual or non-formal shirt` (and polo refs did not match) |
| `bright_color` | Bright/neon shoes with score ≥ 0.70 and margin ≥ 0.08 |
| `open_footwear` | Sandals or slides |
| `wrong_trousers` | Casual trousers (jeans, shorts, chinos, cargo, or other non-formal bottoms) |
| `missing_id_badge` | BR / BR_SUP and chest top label is no ID badge |
| `low_confidence` | Score or margin below threshold, weak bright/neon guess, or missing region (including missing chest when badge is required) |
| `incomplete_body_in_frame` | Pose/framing failed (from `PoseCropper` only) |

### Core rule logic

```python
# Each region independently earns 0 or 25
points = RegionPoints()

# Shirt: resolve from text label, then optionally override via ref match
shirt = SHIRT_LABEL_MAP.get(upper.topLabel)
if polo_match_score >= polo_match_threshold:
    shirt = ShirtType.OFFICIAL_POLO
if shirt in PASS_SHIRTS and (polo_matched or confident(upper)):
    points.upper = 25
elif shirt in FAIL_SHIRTS:
    fail_reasons.append("casual_shirt")
else:
    fail_reasons.append("low_confidence")

# Feet / trousers follow the same pass-or-zero pattern
# Sober sneakers pass; bright/neon shoes fail (`bright_color`) only when
# score ≥ 0.70 and margin ≥ 0.08. A weaker neon guess is dropped so the
# report shows the next-best label
# Casual trousers fail as `wrong_trousers`
# Chest: BR / BR_SUP need a visible badge; other roles auto-pass chest

score = f"{points.upper + points.lower + points.feet + points.chest}/100"
```

---

## Full pipeline flow (`cli.py`)

```python
def run_verify(image, ...):
    pose_out = cropper.process(image)
    if isinstance(pose_out, VerifyResult):
        pose_out.role = canonical_role
        return pose_out  # framing rejected: 0/100 (role still set)

    upper_crop = pose_out.crops["upper"]
    polo_score = scorer.match_official_polo(upper_crop)

    regions = RegionScores(
        upper=scorer.score_region(upper_crop, "upper"),
        lower=scorer.score_region(pose_out.crops["lower"], "lower"),
        feet=scorer.score_region(pose_out.crops["feet"], "feet"),
        chest=scorer.score_region(pose_out.crops["chest"], "chest"),
    )

    result = apply_rules(
        regions,
        polo_match_score=polo_score,
        polo_match_threshold=polo_match_threshold,
        role=role,
        ...
    )
    result.poloMatch = PoloMatch(score=..., threshold=..., matched=..., refs=...)
    return result
```

---

## Unit tests (`tests/test_rules.py`)

These tests **skip** `PoseCropper` and MediaPipe entirely. They feed mock classifier scores directly into `apply_rules()`, optionally with `polo_match_score`.

### Helper

```python
def _score(top: str, top_s: float, second: str, second_s: float) -> RegionScore:
    return RegionScore(
        topLabel=top,
        topScore=top_s,
        secondLabel=second,
        secondScore=second_s,
        scores={top: top_s, second: second_s},
    )
```

### Test cases

| Test | Score | Why |
|------|-------|-----|
| `test_formal_wear_scores_100` | **100/100** | Tucked-in white formal shirt + black trousers + loafers + badge |
| `test_extra_solid_formal_shirts_score_100` | **100/100** | Mint, beige, navy, or light-gray formal shirt |
| `test_grey_trousers_score_100` | **100/100** | Grey or light grey trousers |
| `test_official_polo_visual_label_scores_100` | **100/100** | Black polo with purple trim label wins text scoring |
| `test_sandals_and_casual_shirt_score_50` | **50/100** | Casual/non-formal shirt + sandals → `casual_shirt` + `open_footwear` |
| `test_casual_trousers_score_75` | **75/100** | Casual trousers → `wrong_trousers` (lower 0) |
| `test_non_formal_shirt_score_75` | **75/100** | Casual/non-formal shirt, no ref match → `casual_shirt` (upper 0) |
| `test_low_confidence_upper_scores_75` | **75/100** | Upper score 0.35 < 0.40 |
| `test_low_margin_upper_scores_75` | **75/100** | Pass label vs fail label, margin 0.06 < 0.08 |
| `test_sober_sneakers_score_100` | **100/100** | Single-color sober sneakers pass |
| `test_polo_match_promotes_non_formal_shirt` | **100/100** | Casual/non-formal text label, but `polo_match_score=0.90` promotes to official |
| `test_polo_match_below_threshold_still_zero_upper` | **75/100** | Casual/non-formal shirt + ref score 0.70 → `casual_shirt` |
| `test_polo_match_skips_low_upper_confidence` | **100/100** | Low upper text scores ignored when ref match succeeds |
| `test_polo_match_sober_sneakers_score_100` | **100/100** | Polo ref match + sober sneakers both pass |
| `test_bright_shoes_score_75` | **75/100** | Bright/neon shoes 0.73 vs 0.14 → `bright_color` (feet 0) |
| `test_near_miss_black_trousers_score_100` | **100/100** | Black trousers 0.52 (≥ 0.40) with navy second |
| `test_pass_footwear_split_skips_margin` | **100/100** | Sneakers vs closed shoes — both allowed, margin skipped |
| `test_br_missing_id_badge_scores_75` | **75/100** | Default BR role, chest says no badge |
| `test_br_sup_missing_id_badge_scores_75` | **75/100** | `BR_SUP` also requires a badge |
| `test_br_missing_chest_crop_scores_75` | **75/100** | BR with no chest crop → `low_confidence` |
| `test_br_low_badge_confidence_scores_75` | **75/100** | Badge label at 0.35 → `low_confidence` |
| `test_om_scores_100_without_id_badge` | **100/100** | OM does not require a badge (chest auto-pass) |
| `test_om_scores_100_without_chest_crop` | **100/100** | OM auto-passes chest even with no crop |
| `test_br_missing_badge_and_sandals_score_50` | **50/100** | Missing badge + sandals → `missing_id_badge` + `open_footwear` |
| `test_unknown_role_raises` | raises | Unknown role string → `ValueError` |

Run rule tests only (no GPU / MediaPipe):

```bash
uv run pytest tests/test_rules.py -v
```

`tests/test_pose_framing.py` covers the framing helpers, `prepare_crop` (letterbox, upper `190×280`, downscale), and polo-ref file listing. `tests/test_roles.py` covers role parsing, aliases, and which roles require an ID badge.

---

## Tools and models

### ML models (2)

#### 1. MediaPipe Pose

- **Package:** `mediapipe==0.10.21` (pinned)
- **API:** Classic `mp.solutions.pose.Pose` (not MediaPipe Tasks)
- **Settings:** `static_image_mode=True`, `model_complexity=1`, `min_detection_confidence=0.5`
- **Runtime:** CPU via TensorFlow Lite + XNNPACK
- **Job:** Detect 33 body landmarks → framing gate → region crops

Pinned to 0.10.21 because newer MediaPipe Tasks builds can abort on macOS Metal.

#### 2. Marqo FashionSigLIP

- **Model ID:** `Marqo/marqo-fashionSigLIP`
- **Loaded via:** `open-clip-torch`
- **Job:** Zero-shot image–text matching + image–image cosine similarity for polo refs

### Supporting libraries

| Library | Role |
|---------|------|
| **OpenCV** (`opencv-python-headless`) | Read images, BGR→RGB conversion |
| **Pillow** | Crop images, resize/letterbox before FashionSigLIP |
| **NumPy** | Array handling for pose and images |
| **PyTorch** | Runs FashionSigLIP inference |
| **open-clip-torch** | Loads and runs the SigLIP model |
| **Pydantic** | Result schemas (`VerifyResult`, `RegionScore`, `RegionPoints`, `PoloMatch`) |
| **Typer** | CLI entry point |
| **pytest** | Unit tests |
| **uv** | Dependency and environment management |

`transformers` is listed in `pyproject.toml` but is not imported in the active pipeline code.

### What is not an ML model

- **`apply_rules()`** — deterministic Python dress-code logic (with polo ref override)
- **`evaluate_framing()`** — landmark visibility and pixel-height checks
- **`person_crop_box()` / `_region_boxes()`** — geometry from pose landmarks
- **Reference polo images** — static JPEGs in `refs/official_polo/`, encoded at startup

---

## Quick reference cheat sheet

### Pose cropper (→ 0/100)

| Check | Threshold |
|-------|-----------|
| Nose visible | ≥ 0.5 |
| Both ankles visible | ≥ 0.5 |
| Person height | ≥ 250 px |

### Dress code rules

| Outcome | When |
|---------|------|
| **0 for a region** | Casual/non-formal shirt (no polo ref match), bright/neon shoes (≥ 0.70 and margin ≥ 0.08), sandals, casual trousers, missing ID badge (BR / BR_SUP), low confidence (< 0.40), low margin vs a forbidden label (< 0.08), missing crop |
| **25 for a region** | Tucked-in formal shirt or official polo (text or ref match ≥ 0.85); black/navy/grey trousers; loafers/closed shoes/sober sneakers; BR / BR_SUP visible ID badge (other roles auto-pass chest) |
| **100/100** | All four regions earned 25 |
| **0/100** (framing) | Pose/framing failed (only from `PoseCropper`) |

### Official polo decision

```
upper crop
    ├── text label: "black polo shirt with purple sleeve trim" → official (if confident)
    ├── text label: "casual or non-formal shirt" + ref score ≥ 0.85 → promoted to official
    └── text label: "casual or non-formal shirt" + ref score < 0.85 → upper 0 (`casual_shirt`)
```

---

## File map

| File | Responsibility |
|------|----------------|
| `src/attire_verification/cli.py` | `verify` and `batch` CLI, plus `run_verify` orchestration |
| `src/attire_verification/pose_cropper.py` | MediaPipe framing gate and region crops |
| `src/attire_verification/siglip_scorer.py` | FashionSigLIP scoring, `prepare_crop`, polo ref matching |
| `src/attire_verification/rules.py` | Dress-code rule engine (with polo override + role badge rules) |
| `src/attire_verification/roles.py` | Staff roles and which ones require an ID badge |
| `src/attire_verification/labels.py` | Fixed text labels per region |
| `src/attire_verification/models.py` | Pydantic schemas (`VerifyResult`, `RegionPoints`, `PoloMatch`, etc.) |
| `src/attire_verification/refs/official_polo/` | Bundled official-polo upper reference JPEGs |
| `scripts/generate_client_report.py` | Score photo folders and write a client PDF via Chrome |
| `tests/test_rules.py` | Rule engine unit tests (mock scores + polo match + roles) |
| `tests/test_roles.py` | Role parsing and ID-badge requirement tests |
| `tests/test_pose_framing.py` | Framing, `prepare_crop`, and polo-ref listing tests (no MediaPipe) |
