"""Typer CLI for attire verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from attire_verification.models import RegionScores, Status, VerifyResult
from attire_verification.pose_cropper import CropResult, PoseCropper
from attire_verification.rules import MIN_CONFIDENCE, MIN_MARGIN, apply_rules
from attire_verification.siglip_scorer import FashionSigLIPScorer

app = typer.Typer(
    name="attire_verification",
    help="BR attire verification eval CLI (offline).",
    no_args_is_help=True,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

_pose_cropper: PoseCropper | None = None
_scorer: FashionSigLIPScorer | None = None


def _get_pose_cropper() -> PoseCropper:
    global _pose_cropper
    if _pose_cropper is None:
        _pose_cropper = PoseCropper()
    return _pose_cropper


def _get_scorer() -> FashionSigLIPScorer:
    global _scorer
    if _scorer is None:
        _scorer = FashionSigLIPScorer()
    return _scorer


def _save_debug_crops(crops: dict, debug_dir: Path) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    for name, img in crops.items():
        img.save(debug_dir / f"{name}.jpg", quality=92)


def run_verify(
    image: Path,
    *,
    min_confidence: float = MIN_CONFIDENCE,
    min_margin: float = MIN_MARGIN,
    debug_crops: Path | None = None,
) -> VerifyResult:
    """Run the full verify pipeline on a single image."""
    cropper = _get_pose_cropper()
    pose_out = cropper.process(image)

    if isinstance(pose_out, VerifyResult):
        return pose_out

    assert isinstance(pose_out, CropResult)
    if debug_crops is not None:
        _save_debug_crops(pose_out.crops, debug_crops)

    scorer = _get_scorer()
    regions = RegionScores(
        upper=scorer.score_region(pose_out.crops["upper"], "upper"),
        lower=scorer.score_region(pose_out.crops["lower"], "lower"),
        feet=scorer.score_region(pose_out.crops["feet"], "feet"),
        chest=scorer.score_region(pose_out.crops["chest"], "chest"),
    )

    result = apply_rules(
        regions,
        min_confidence=min_confidence,
        min_margin=min_margin,
        image_path=str(image),
    )
    result.pose = pose_out.pose
    return result


def _print_result(result: VerifyResult, pretty: bool) -> None:
    data = result.model_dump(mode="json", exclude_none=False)
    # Drop batch-only fields when unused
    if data.get("expected") is None:
        data.pop("expected", None)
    if data.get("match") is None:
        data.pop("match", None)
    if pretty:
        typer.echo(json.dumps(data, indent=2))
    else:
        typer.echo(json.dumps(data))


def _infer_expected(path: Path) -> str | None:
    """Infer ground truth from parent folder name."""
    for parent in path.parents:
        name = parent.name.strip().lower()
        if name == "right attire":
            return Status.PASSED.value
        if name == "wrong attire":
            return Status.FAILED.value
    return None


@app.command("verify")
def verify_cmd(
    image: Path = typer.Option(..., "--image", exists=True, readable=True, help="Input image path"),
    pretty: bool = typer.Option(False, "--pretty", help="Pretty-print JSON"),
    debug_crops: Optional[Path] = typer.Option(
        None, "--debug-crops", help="Directory to save region crop JPEGs"
    ),
    min_confidence: float = typer.Option(
        MIN_CONFIDENCE, "--min-confidence", help="Minimum top-1 confidence"
    ),
    min_margin: float = typer.Option(
        MIN_MARGIN, "--min-margin", help="Minimum top1-top2 score margin"
    ),
) -> None:
    """Verify attire in a single full-body photo. Exit 0 if PASSED, else 1."""
    result = run_verify(
        image,
        min_confidence=min_confidence,
        min_margin=min_margin,
        debug_crops=debug_crops,
    )
    _print_result(result, pretty)
    raise typer.Exit(code=0 if result.status == Status.PASSED else 1)


@app.command("batch")
def batch_cmd(
    dir: Path = typer.Option(..., "--dir", exists=True, file_okay=False, help="Folder of images"),
    output: Path = typer.Option(..., "--output", help="JSONL output path"),
    min_confidence: float = typer.Option(MIN_CONFIDENCE, "--min-confidence"),
    min_margin: float = typer.Option(MIN_MARGIN, "--min-margin"),
) -> None:
    """Batch-evaluate images under a directory; write JSONL + accuracy summary."""
    images = sorted(
        p for p in dir.rglob("*") if p.is_file() and p.suffix in IMAGE_EXTENSIONS
    )
    if not images:
        typer.echo(f"No images found under {dir}", err=True)
        raise typer.Exit(code=1)

    output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    correct = 0
    false_pass = 0
    false_fail = 0
    rejected = 0
    uncertain = 0
    scored = 0  # has expected label and not REJECTED

    with output.open("w", encoding="utf-8") as f:
        for img_path in images:
            result = run_verify(
                img_path,
                min_confidence=min_confidence,
                min_margin=min_margin,
            )
            expected = _infer_expected(img_path)
            result.expected = expected

            match: bool | None = None
            if expected is not None:
                if result.status == Status.REJECTED:
                    rejected += 1
                    match = False
                elif result.status == Status.UNCERTAIN:
                    uncertain += 1
                    # UNCERTAIN is not a correct prediction of PASSED/FAILED
                    match = False
                    scored += 1
                else:
                    scored += 1
                    match = result.status.value == expected
                    if match:
                        correct += 1
                    elif result.status == Status.PASSED and expected == Status.FAILED.value:
                        false_pass += 1
                    elif result.status == Status.FAILED and expected == Status.PASSED.value:
                        false_fail += 1
            result.match = match
            total += 1

            data = result.model_dump(mode="json")
            f.write(json.dumps(data) + "\n")

    accuracy = (correct / scored) if scored else 0.0
    typer.echo(
        json.dumps(
            {
                "total": total,
                "scored": scored,
                "correct": correct,
                "accuracy": round(accuracy, 4),
                "falsePass": false_pass,
                "falseFail": false_fail,
                "rejected": rejected,
                "uncertain": uncertain,
                "output": str(output),
            },
            indent=2,
        )
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
