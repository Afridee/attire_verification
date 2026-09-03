# Attire Verification — Developer Guide

This document explains how the offline BR attire verification pipeline works: the framing gate, region crops, zero-shot scoring, official-polo reference matching, dress-code rules, and the tools/models involved.

## Pipeline overview

Verification runs in three stages:

```
Photo → PoseCropper → FashionSigLIP (+ polo ref match) → apply_rules → Verdict
              ↓
         REJECTED (stops early if framing fails)
```

| Stage | Component | Output |
|-------|-----------|--------|
| 1 | `PoseCropper` | Region crops or `REJECTED` |
| 2 | `FashionSigLIPScorer` | Label probabilities per region + polo similarity score |
| 3 | `apply_rules()` | `PASSED` / `FAILED` / `UNCERTAIN` |

Stage 2 now has **two scoring paths for the upper body**:

1. **Text labels** — zero-shot classification against fixed prompts (all regions)
2. **Reference images** — cosine similarity of the upper crop vs bundled official-polo photos

The reference match can **override** the text label and promote the shirt to official polo when similarity ≥ threshold (default `0.85`).

Run the full pipeline via CLI:

```bash
uv run python -m attire_verification verify \
  --image "/path/to/photo.jpg" \
  --pretty \
  --debug-crops ./debug/
```

The `--debug-crops` flag saves `upper.jpg`, `lower.jpg`, `feet.jpg`, and `chest.jpg` for inspection.

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--image PATH` | — | Required input image |
| `--pretty` | off | Pretty-print JSON |
| `--debug-crops DIR` | — | Save region crop JPEGs |
| `--min-confidence F` | `0.55` | Minimum top-1 text score |
| `--min-margin F` | `0.08` | Minimum top1 − top2 margin |
| `--polo-match-threshold F` | `0.85` | Min cosine similarity vs official polo refs |
| `--polo-refs DIR` | bundled | Override official-polo reference folder |

---

## Status outcomes

Four possible final statuses:

| Status | Meaning | Typical cause |
|--------|---------|----------------|
| **REJECTED** | Photo unsuitable for verification | Bad framing, no pose, unreadable image |
| **FAILED** | Clear dress-code violation | Sandals, casual shirt, wrong polo, chinos, etc. |
| **UNCERTAIN** | Model not confident enough, or borderline item | Low scores, sneakers |
| **PASSED** | Compliant attire with confident scores | Formal shirt or official polo + black/navy trousers + closed shoes |

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

Early exits all return `Status.REJECTED` with `failReasons: ["incomplete_body_in_frame"]`:

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

1. Preprocess each crop (`prepare_crop`) — resize or letterbox tall strips before the model's 224×224 input
2. Encode the crop image → embedding
3. Compare to pre-encoded text labels for that region
4. Softmax over similarities → probabilities
5. Return `RegionScore` with `topLabel`, `topScore`, `secondLabel`, `secondScore`

### Official polo reference matching (new)

Because FashionSigLIP has no built-in notion of "official" vs "non-official", the pipeline adds **image-to-image matching** for the upper crop:

```python
def match_official_polo(self, crop_pil: Image.Image) -> float:
    """Max cosine similarity of an upper crop vs official-polo reference embeddings."""
```

- Reference images live in `src/attire_verification/refs/official_polo/` (e.g. `ctg3_upper.jpg`, `ctg5_upper.jpg`)
- Override with `--polo-refs DIR`
- Each reference is encoded once at startup; the upper crop is compared via max cosine similarity
- Same shirt across photos typically scores ~0.88–0.92; other garments ~0.68–0.78 on the sample set

### Label sets (`labels.py`)

Polo labels use **visual descriptions** (not "official company polo"):

**Upper (7 labels):**

- white formal button-down shirt
- light blue formal button-down shirt
- **black polo shirt with purple sleeve trim** ← official polo (visual)
- casual t-shirt
- striped t-shirt
- plaid or checkered shirt
- **colored polo shirt** ← non-official polo (visual)

**Lower (3 labels):**

- black formal trousers
- dark navy trousers
- beige or tan chinos

**Feet (4 labels):**

- black formal closed shoes
- black leather loafers
- white sneakers
- sandals or slides

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

- The shirt is treated as **official polo** even if text labels say `colored polo shirt`
- **Upper-region confidence/margin gates are skipped** (text scores for similar polo prompts are often split)
- Lower and feet regions still require confidence ≥ 0.55 and margin ≥ 0.08

### Decision priority

```
1. Auto FAIL   → clear violations (takes precedence)
2. UNCERTAIN   → sneakers, low confidence, low margin
3. Auto PASS   → all regions acceptable and confident
4. UNCERTAIN   → unknown / unmatched labels (fallback)
```

### Thresholds

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `MIN_CONFIDENCE` | 0.55 | Top label score must be ≥ this |
| `MIN_MARGIN` | 0.08 | Gap between top1 and top2 must be ≥ this |
| `POLO_MATCH_THRESHOLD` | 0.85 | Cosine similarity vs ref crops to promote official polo |

Margin = `topScore - secondScore`.

### Allowed vs forbidden

| Region | Pass | Fail | Uncertain |
|--------|------|------|-----------|
| **Shirt** | White formal, light blue formal, official polo (text or ref match) | Casual tee, striped, plaid, colored polo (no ref match) | — |
| **Trousers** | Black formal, dark navy | Beige/tan chinos | — |
| **Feet** | Closed shoes, loafers | Sandals/slides | Sneakers |

### Fail reasons

| Reason | Trigger |
|--------|---------|
| `casual_shirt` | Casual tee, striped, or plaid shirt |
| `non_official_polo` | Colored polo with no ref match (score < threshold) |
| `open_footwear` | Sandals or slides |
| `wrong_trousers` | Beige or tan chinos |
| `borderline_footwear` | White sneakers (never auto-pass in v1) |
| `low_confidence` | Score or margin below threshold, or missing region |
| `incomplete_body_in_frame` | Pose/framing failed (from `PoseCropper` only) |

### Core rule logic

```python
# Resolve shirt from text label, then optionally override via ref match
shirt = SHIRT_LABEL_MAP.get(upper.topLabel)
if polo_match_score >= polo_match_threshold:
    shirt = ShirtType.OFFICIAL_POLO

# Auto FAIL checks (clear violations take precedence)
if shirt in FAIL_SHIRTS:
    fail_reasons.append("casual_shirt" or "non_official_polo")

if footwear == FootwearType.SANDALS:
    fail_reasons.append("open_footwear")

if trouser == TrouserType.BEIGE_CHINOS:
    fail_reasons.append("wrong_trousers")

# Sneakers are borderline — never auto-pass
if footwear == FootwearType.SNEAKERS:
    return UNCERTAIN, ["borderline_footwear"]

# Confidence / margin gates — skip upper when polo ref matched
scored_regions = [lower, feet] if polo_matched else [upper, lower, feet]
for region in scored_regions:
    if topScore < 0.55 or margin < 0.08:
        return UNCERTAIN, ["low_confidence"]

# Auto PASS
if shirt in PASS_SHIRTS and trouser in PASS_TROUSERS and footwear in PASS_FEET:
    return PASSED
```

---

## Full pipeline flow (`cli.py`)

```python
def run_verify(image, ...):
    pose_out = cropper.process(image)
    if isinstance(pose_out, VerifyResult):
        return pose_out  # REJECTED

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

| Test | Status | Why |
|------|--------|-----|
| `test_formal_wear_passes` | **PASSED** | White formal shirt + black trousers + loafers, all confident |
| `test_official_polo_visual_label_passes` | **PASSED** | Black polo with purple trim label wins text scoring |
| `test_sandals_and_striped_shirt_fail` | **FAILED** | Striped shirt + sandals → `casual_shirt` + `open_footwear` |
| `test_beige_chinos_fail` | **FAILED** | Beige chinos → `wrong_trousers` |
| `test_non_official_polo_fail` | **FAILED** | Colored polo, no ref match → `non_official_polo` |
| `test_low_confidence_uncertain` | **UNCERTAIN** | Upper score 0.40 < 0.55 |
| `test_low_margin_uncertain` | **UNCERTAIN** | Top score 0.50 < 0.55 (margin also too low at 0.05) |
| `test_sneakers_uncertain` | **UNCERTAIN** | Sneakers → `borderline_footwear` |
| `test_polo_match_promotes_colored_polo` | **PASSED** | Colored polo text label, but `polo_match_score=0.90` promotes to official |
| `test_polo_match_below_threshold_still_fails` | **FAILED** | Colored polo + ref score 0.70 → `non_official_polo` |
| `test_polo_match_skips_low_upper_confidence` | **PASSED** | Low upper text scores ignored when ref match succeeds |
| `test_polo_match_sneakers_still_uncertain` | **UNCERTAIN** | Ref match does not override sneaker borderline rule |

Run rule tests only (no GPU / MediaPipe):

```bash
uv run pytest tests/test_rules.py -v
```

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
| **Pydantic** | Result schemas (`VerifyResult`, `RegionScore`, `PoloMatch`, `Status`) |
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

### Pose cropper (→ REJECTED)

| Check | Threshold |
|-------|-----------|
| Nose visible | ≥ 0.5 |
| Both ankles visible | ≥ 0.5 |
| Person height | ≥ 250 px |

### Dress code rules

| Status | When |
|--------|------|
| **FAILED** | Striped/plaid/casual tee, colored polo (no ref match), sandals, beige chinos |
| **UNCERTAIN** | Sneakers, low confidence (< 0.55), low margin (< 0.08), missing regions |
| **PASSED** | Formal shirt or official polo (text or ref match ≥ 0.85) + black/navy trousers + loafers/closed shoes + confident scores |
| **REJECTED** | Pose/framing failed (only from `PoseCropper`) |

### Official polo decision

```
upper crop
    ├── text label: "black polo shirt with purple sleeve trim" → official (if confident)
    ├── text label: "colored polo shirt" + ref score ≥ 0.85 → promoted to official
    └── text label: "colored polo shirt" + ref score < 0.85 → FAILED (non_official_polo)
```

---

## File map

| File | Responsibility |
|------|----------------|
| `src/attire_verification/cli.py` | CLI and full pipeline orchestration |
| `src/attire_verification/pose_cropper.py` | MediaPipe framing gate and region crops |
| `src/attire_verification/siglip_scorer.py` | FashionSigLIP scoring + polo ref matching |
| `src/attire_verification/rules.py` | Dress-code rule engine (with polo override) |
| `src/attire_verification/labels.py` | Fixed text labels per region |
| `src/attire_verification/models.py` | Pydantic schemas (`Status`, `VerifyResult`, `PoloMatch`, etc.) |
| `src/attire_verification/refs/official_polo/` | Bundled official-polo upper reference JPEGs |
| `tests/test_rules.py` | Rule engine unit tests (mock scores + polo match) |
| `tests/test_pose_framing.py` | Framing helper unit tests (no MediaPipe) |
