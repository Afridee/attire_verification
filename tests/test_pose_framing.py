"""Unit tests for person-centric framing and crop prep helpers (no MediaPipe)."""

import pytest
from PIL import Image

from attire_verification.pose_cropper import (
    MIN_PERSON_HEIGHT_PX,
    evaluate_framing,
    person_crop_box,
)
from attire_verification.siglip_scorer import (
    LETTERBOX_SIZE,
    MAX_CROP_SIDE,
    UPPER_SIZE,
    iter_polo_ref_images,
    prepare_crop,
)


class _Lm:
    def __init__(self, x: float, y: float, visibility: float = 1.0) -> None:
        self.x = x
        self.y = y
        self.visibility = visibility


def _full_body(*, span: float = 0.48, vis: float = 0.9) -> list[_Lm]:
    """33 dummy landmarks for a centered full-body figure at the given span."""
    nose_y = 0.20
    ankle_y = nose_y + span
    coords = [(0.5, 0.5)] * 33
    coords[0] = (0.50, nose_y)
    coords[11] = (0.62, 0.32)
    coords[12] = (0.38, 0.32)
    coords[23] = (0.58, 0.52)
    coords[24] = (0.42, 0.52)
    coords[27] = (0.58, ankle_y)
    coords[28] = (0.42, ankle_y)
    return [_Lm(x, y, vis) for x, y in coords]


def test_wide_shot_full_body_passes_framing():
    """48% of a tall frame used to REJECT; it is still a complete figure."""
    width, height = 3120, 4160
    lms = _full_body(span=0.48)
    ok, details = evaluate_framing(lms, width, height)
    assert ok is True
    assert details["verticalSpan"] == 0.48
    assert details["personHeightPx"] >= MIN_PERSON_HEIGHT_PX
    assert "fullBodySpan" not in details


def test_tiny_person_rejected():
    width, height = 1000, 1000
    lms = _full_body(span=0.10)
    ok, details = evaluate_framing(lms, width, height)
    assert ok is False
    assert details.get("personTooSmall") is True


def test_missing_ankles_rejected():
    lms = _full_body()
    lms[27].visibility = 0.1
    lms[28].visibility = 0.1
    ok, details = evaluate_framing(lms, 3120, 4160)
    assert ok is False
    assert details.get("anklesVisible") is False


def test_person_crop_box_clamped_and_padded():
    lms = _full_body(span=0.50)
    box = person_crop_box(lms, 1000, 2000)
    assert box is not None
    assert box.x >= 0
    assert box.y >= 0
    assert box.x + box.w <= 1000
    assert box.y + box.h <= 2000
    assert box.h > 900


def test_prepare_crop_letterboxes_tall_strips():
    tall = Image.new("RGB", (100, 400), (10, 10, 10))
    out = prepare_crop(tall)
    assert out.size == (LETTERBOX_SIZE, LETTERBOX_SIZE)


def test_prepare_crop_downscales_large_non_upper():
    wide = Image.new("RGB", (900, 700), (200, 200, 200))
    out = prepare_crop(wide, "chest")
    assert max(out.size) <= MAX_CROP_SIDE
    assert out.size[0] / out.size[1] == pytest.approx(900 / 700, rel=0.02)


def test_prepare_crop_canonical_upper_size():
    big = Image.new("RGB", (569, 877), (220, 220, 220))
    assert prepare_crop(big, "upper").size == UPPER_SIZE


def test_iter_polo_ref_images_skips_non_images(tmp_path):
    (tmp_path / "ctg3_upper.jpg").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("skip")
    (tmp_path / "nested").mkdir()
    paths = iter_polo_ref_images(tmp_path)
    assert [p.name for p in paths] == ["ctg3_upper.jpg"]
    assert iter_polo_ref_images(tmp_path / "missing") == []
