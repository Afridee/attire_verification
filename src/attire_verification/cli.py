"""Typer CLI for attire verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from attire_verification.models import (
    PERFECT_SCORE,
    PoloMatch,
    RegionScores,
    VerifyResult,
)
from attire_verification.pose_cropper import CropResult, PoseCropper
from attire_verification.roles import Role, parse_role
from attire_verification.rules import MIN_CONFIDENCE, MIN_MARGIN, POLO_MATCH_THRESHOLD, apply_rules
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


def _get_scorer(polo_refs: Path | None = None) -> FashionSigLIPScorer:
    global _scorer
    if _scorer is None:
        _scorer = FashionSigLIPScorer(polo_refs=polo_refs)
    return _scorer


def _save_debug_crops(crops: dict, debug_dir: Path) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    for name, img in crops.items():
        img.save(debug_dir / f"{name}.jpg", quality=92)


def _parse_role_option(
    _ctx: typer.Context,
    _param: typer.CallbackParam,
    value: str,
) -> str:
    try:
        return parse_role(value).value
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def run_verify(
    image: Path,
    *,
    min_confidence: float = MIN_CONFIDENCE,
    min_margin: float = MIN_MARGIN,
    polo_match_threshold: float = POLO_MATCH_THRESHOLD,
    polo_refs: Path | None = None,
    debug_crops: Path | None = None,
    role: str = Role.BR.value,
) -> VerifyResult:
    """Run the full verify pipeline on a single image."""
    canonical_role = parse_role(role).value
    cropper = _get_pose_cropper()
    pose_out = cropper.process(image)

    if isinstance(pose_out, VerifyResult):
        pose_out.role = canonical_role
        return pose_out

    assert isinstance(pose_out, CropResult)
    if debug_crops is not None:
        _save_debug_crops(pose_out.crops, debug_crops)

    scorer = _get_scorer(polo_refs)
    upper_crop = pose_out.crops["upper"]
    polo_score = scorer.match_official_polo(upper_crop)
    polo_match = PoloMatch(
        score=round(polo_score, 4),
        threshold=polo_match_threshold,
        matched=polo_score >= polo_match_threshold,
        refs=scorer.polo_ref_count,
    )
    regions = RegionScores(
        upper=scorer.score_region(upper_crop, "upper"),
        lower=scorer.score_region(pose_out.crops["lower"], "lower"),
        feet=scorer.score_region(pose_out.crops["feet"], "feet"),
        chest=scorer.score_region(pose_out.crops["chest"], "chest"),
    )

    result = apply_rules(
        regions,
        min_confidence=min_confidence,
        min_margin=min_margin,
        polo_match_score=polo_score,
        polo_match_threshold=polo_match_threshold,
        image_path=str(image),
        role=canonical_role,
    )
    result.pose = pose_out.pose
    result.poloMatch = polo_match
    return result


def _print_result(result: VerifyResult, pretty: bool) -> None:
    data = result.public_dict()
    if pretty:
        typer.echo(json.dumps(data, indent=2))
    else:
        typer.echo(json.dumps(data))


EXPECTED_WRONG = "<100"


def _infer_expected(path: Path) -> str | None:
    """Infer ground truth from parent folder name."""
    for parent in path.parents:
        name = parent.name.strip().lower()
        if name == "right attire":
            return PERFECT_SCORE
        if name == "wrong attire":
            return EXPECTED_WRONG
    return None


def _is_rejected(result: VerifyResult) -> bool:
    return "incomplete_body_in_frame" in result.failReasons


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
    polo_match_threshold: float = typer.Option(
        POLO_MATCH_THRESHOLD,
        "--polo-match-threshold",
        help="Min cosine similarity vs official polo reference crops",
    ),
    polo_refs: Optional[Path] = typer.Option(
        None,
        "--polo-refs",
        exists=True,
        file_okay=False,
        help="Folder of official-polo upper-body reference JPEGs",
    ),
    role: str = typer.Option(
        Role.BR.value,
        "--role",
        help="Staff role (BR and BR_SUP require a visible ID badge)",
        callback=_parse_role_option,
    ),
) -> None:
    """Verify attire in a single full-body photo. Exit 0 if 100/100, else 1."""
    result = run_verify(
        image,
        min_confidence=min_confidence,
        min_margin=min_margin,
        polo_match_threshold=polo_match_threshold,
        polo_refs=polo_refs,
        debug_crops=debug_crops,
        role=role,
    )
    _print_result(result, pretty)
    raise typer.Exit(code=0 if result.score == PERFECT_SCORE else 1)


@app.command("batch")
def batch_cmd(
    dir: Path = typer.Option(..., "--dir", exists=True, file_okay=False, help="Folder of images"),
    output: Path = typer.Option(..., "--output", help="JSONL output path"),
    min_confidence: float = typer.Option(MIN_CONFIDENCE, "--min-confidence"),
    min_margin: float = typer.Option(MIN_MARGIN, "--min-margin"),
    polo_match_threshold: float = typer.Option(
        POLO_MATCH_THRESHOLD, "--polo-match-threshold"
    ),
    polo_refs: Optional[Path] = typer.Option(
        None, "--polo-refs", exists=True, file_okay=False
    ),
    role: str = typer.Option(
        Role.BR.value,
        "--role",
        help="Staff role applied to every image (BR and BR_SUP require a visible ID badge)",
        callback=_parse_role_option,
    ),
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
    scored = 0  # has expected label and not framing-rejected

    with output.open("w", encoding="utf-8") as f:
        for img_path in images:
            result = run_verify(
                img_path,
                min_confidence=min_confidence,
                min_margin=min_margin,
                polo_match_threshold=polo_match_threshold,
                polo_refs=polo_refs,
                role=role,
            )
            expected = _infer_expected(img_path)
            result.expected = expected

            match: bool | None = None
            if expected is not None:
                if _is_rejected(result):
                    rejected += 1
                    match = False
                else:
                    scored += 1
                    perfect = result.score == PERFECT_SCORE
                    expect_perfect = expected == PERFECT_SCORE
                    match = perfect == expect_perfect
                    if match:
                        correct += 1
                    elif perfect and not expect_perfect:
                        false_pass += 1
                    elif not perfect and expect_perfect:
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
                "output": str(output),
            },
            indent=2,
        )
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
