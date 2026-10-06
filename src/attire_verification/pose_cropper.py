"""MediaPipe Pose framing gate and region crop boxes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

from attire_verification.models import (
    ZERO_SCORE,
    CropBox,
    PoseInfo,
    RegionPoints,
    VerifyResult,
)

# MediaPipe Pose landmark indices
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

VISIBILITY_THRESHOLD = 0.5
# Person must be large enough in pixels for clothing crops — not a % of the frame.
# Wide environmental shots with a visible full body should pass.
MIN_PERSON_HEIGHT_PX = 250
PERSON_LANDMARK_VIS = 0.3
PERSON_BOX_PAD = 0.12
PERSON_HEADROOM = 0.12  # extra above extreme landmarks (hair above nose)
PERSON_FOOTROOM = 0.08  # extra below extreme landmarks (soles below ankles)


@dataclass
class CropResult:
    crops: dict[str, Image.Image]
    boxes: dict[str, CropBox]
    pose: PoseInfo


def _visibility(lm) -> float:
    return float(getattr(lm, "visibility", 1.0) or 0.0)


def _pad_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    pad: float,
    width: int,
    height: int,
) -> CropBox:
    bw = x2 - x1
    bh = y2 - y1
    px = bw * pad
    py = bh * pad
    left = int(max(0, x1 - px))
    top = int(max(0, y1 - py))
    right = int(min(width, x2 + px))
    bottom = int(min(height, y2 + py))
    return CropBox(x=left, y=top, w=max(1, right - left), h=max(1, bottom - top))


def _crop_pil(image: Image.Image, box: CropBox) -> Image.Image:
    return image.crop((box.x, box.y, box.x + box.w, box.y + box.h))


def _offset_box(box: CropBox, ox: int, oy: int) -> CropBox:
    return CropBox(x=box.x + ox, y=box.y + oy, w=box.w, h=box.h)


def person_crop_box(lms, width: int, height: int) -> CropBox | None:
    """Tight box around visible pose landmarks, with head/foot/side padding."""
    xs: list[float] = []
    ys: list[float] = []
    for lm in lms:
        if _visibility(lm) >= PERSON_LANDMARK_VIS:
            xs.append(lm.x * width)
            ys.append(lm.y * height)
    if len(xs) < 4:
        return None
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    box_h = max(1.0, y2 - y1)
    y1 -= box_h * PERSON_HEADROOM
    y2 += box_h * PERSON_FOOTROOM
    return _pad_box(x1, y1, x2, y2, PERSON_BOX_PAD, width, height)


def evaluate_framing(lms, width: int, height: int) -> tuple[bool, dict]:
    """Full-body gate: visible nose + ankles, and enough person pixels.

    Does not require the person to fill 55% of the *image*. Extra background
    around a complete figure is fine.
    """
    details: dict = {}
    ok = True

    nose = lms[NOSE]
    left_ankle = lms[LEFT_ANKLE]
    right_ankle = lms[RIGHT_ANKLE]
    nose_vis = _visibility(nose)
    la_vis = _visibility(left_ankle)
    ra_vis = _visibility(right_ankle)

    if nose_vis < VISIBILITY_THRESHOLD:
        ok = False
        details["noseVisible"] = False
    if la_vis < VISIBILITY_THRESHOLD or ra_vis < VISIBILITY_THRESHOLD:
        ok = False
        details["anklesVisible"] = False

    nose_y = nose.y * height
    ankle_y = max(left_ankle.y, right_ankle.y) * height
    person_h = ankle_y - nose_y
    vertical_span = person_h / height if height > 0 else 0.0
    details["verticalSpan"] = round(vertical_span, 3)
    details["personHeightPx"] = int(round(person_h))

    if person_h < MIN_PERSON_HEIGHT_PX:
        ok = False
        details["personTooSmall"] = True

    return ok, details


def _region_boxes(
    *,
    width: int,
    height: int,
    ls_x: float,
    ls_y: float,
    rs_x: float,
    rs_y: float,
    lh_x: float,
    lh_y: float,
    rh_x: float,
    rh_y: float,
    la_x: float,
    la_y: float,
    ra_x: float,
    ra_y: float,
) -> dict[str, CropBox]:
    upper_x1 = min(ls_x, rs_x, lh_x, rh_x)
    upper_x2 = max(ls_x, rs_x, lh_x, rh_x)
    upper_y1 = min(ls_y, rs_y)
    upper_y2 = max(lh_y, rh_y)
    upper_box = _pad_box(upper_x1, upper_y1, upper_x2, upper_y2, 0.10, width, height)

    lower_x1 = min(lh_x, rh_x, la_x, ra_x)
    lower_x2 = max(lh_x, rh_x, la_x, ra_x)
    lower_y1 = min(lh_y, rh_y)
    lower_y2 = max(la_y, ra_y)
    lower_box = _pad_box(lower_x1, lower_y1, lower_x2, lower_y2, 0.05, width, height)

    # Feet: ankles down to the person-normalized frame bottom (not the original
    # photo bottom — that pulls in floor clutter on wide shots).
    feet_x1 = min(la_x, ra_x)
    feet_x2 = max(la_x, ra_x)
    feet_y1 = min(la_y, ra_y)
    feet_y2 = float(height)
    ankle_w = max(1.0, feet_x2 - feet_x1)
    feet_x1 -= ankle_w * 0.5
    feet_x2 += ankle_w * 0.5
    feet_box = _pad_box(feet_x1, feet_y1, feet_x2, feet_y2, 0.10, width, height)

    torso_top = min(ls_y, rs_y)
    torso_bottom = max(lh_y, rh_y)
    chest_bottom = torso_top + 0.40 * max(1.0, torso_bottom - torso_top)
    chest_x1 = min(ls_x, rs_x)
    chest_x2 = max(ls_x, rs_x)
    chest_box = _pad_box(chest_x1, torso_top, chest_x2, chest_bottom, 0.05, width, height)

    return {
        "upper": upper_box,
        "lower": lower_box,
        "feet": feet_box,
        "chest": chest_box,
    }


class PoseCropper:
    """Detect pose landmarks and crop upper / lower / feet / chest regions."""

    def __init__(self) -> None:
        # Classic solutions API — more reliable on macOS than Tasks+Metal.
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        )

    def close(self) -> None:
        self._pose.close()

    def _detect(self, rgb: np.ndarray):
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return None
        return result.pose_landmarks.landmark

    def process(self, image_path: str | Path) -> CropResult | VerifyResult:
        path = Path(image_path)
        bgr = cv2.imread(str(path))
        if bgr is None:
            return VerifyResult(
                score=ZERO_SCORE,
                failReasons=["incomplete_body_in_frame"],
                imagePath=str(path),
                regions=None,
                regionPoints=RegionPoints(),
                pose=PoseInfo(
                    fullBodyOk=False,
                    landmarksDetected=0,
                    details={"error": "unreadable_image"},
                ),
            )

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = rgb.shape[:2]
        lms = self._detect(rgb)

        if lms is None:
            return VerifyResult(
                score=ZERO_SCORE,
                failReasons=["incomplete_body_in_frame"],
                imagePath=str(path),
                regions=None,
                regionPoints=RegionPoints(),
                pose=PoseInfo(fullBodyOk=False, landmarksDetected=0),
            )

        framing_ok, framing_details = evaluate_framing(lms, orig_w, orig_h)
        if not framing_ok:
            return VerifyResult(
                score=ZERO_SCORE,
                failReasons=["incomplete_body_in_frame"],
                imagePath=str(path),
                regions=None,
                regionPoints=RegionPoints(),
                pose=PoseInfo(
                    fullBodyOk=False,
                    landmarksDetected=len(lms),
                    details=framing_details,
                ),
            )

        pil_orig = Image.fromarray(rgb)
        pbox = person_crop_box(lms, orig_w, orig_h)
        if pbox is None:
            return VerifyResult(
                score=ZERO_SCORE,
                failReasons=["incomplete_body_in_frame"],
                imagePath=str(path),
                regions=None,
                regionPoints=RegionPoints(),
                pose=PoseInfo(
                    fullBodyOk=False,
                    landmarksDetected=len(lms),
                    details={**framing_details, "error": "no_person_box"},
                ),
            )

        person_pil = _crop_pil(pil_orig, pbox)
        work_rgb = np.array(person_pil.convert("RGB"))
        work_h, work_w = work_rgb.shape[:2]
        work_lms = self._detect(work_rgb)
        ox, oy = pbox.x, pbox.y

        if work_lms is not None:
            lms = work_lms
            width, height = work_w, work_h
            work_pil = person_pil
            framing_details["personCrop"] = pbox.as_list()
            framing_details["refinedPose"] = True
        else:
            # Fall back to first-pass landmarks, remapped into the person crop.
            width, height = work_w, work_h
            work_pil = person_pil
            framing_details["personCrop"] = pbox.as_list()
            framing_details["refinedPose"] = False

            class _Shifted:
                __slots__ = ("x", "y", "visibility")

                def __init__(self, lm) -> None:
                    self.x = ((lm.x * orig_w) - ox) / max(1, width)
                    self.y = ((lm.y * orig_h) - oy) / max(1, height)
                    self.visibility = _visibility(lm)

            lms = [_Shifted(lm) for lm in lms]

        n = len(lms)

        def px(idx: int) -> tuple[float, float]:
            lm = lms[idx]
            return lm.x * width, lm.y * height

        ls_x, ls_y = px(LEFT_SHOULDER)
        rs_x, rs_y = px(RIGHT_SHOULDER)
        lh_x, lh_y = px(LEFT_HIP)
        rh_x, rh_y = px(RIGHT_HIP)
        la_x, la_y = px(LEFT_ANKLE)
        ra_x, ra_y = px(RIGHT_ANKLE)

        local_boxes = _region_boxes(
            width=width,
            height=height,
            ls_x=ls_x,
            ls_y=ls_y,
            rs_x=rs_x,
            rs_y=rs_y,
            lh_x=lh_x,
            lh_y=lh_y,
            rh_x=rh_x,
            rh_y=rh_y,
            la_x=la_x,
            la_y=la_y,
            ra_x=ra_x,
            ra_y=ra_y,
        )
        # Report boxes in the original image so overlays still line up.
        boxes = {name: _offset_box(box, ox, oy) for name, box in local_boxes.items()}
        crops = {name: _crop_pil(work_pil, local_boxes[name]) for name in local_boxes}

        pose = PoseInfo(
            fullBodyOk=True,
            landmarksDetected=n,
            cropBoxes={k: v.as_list() for k, v in boxes.items()},
            details=framing_details,
        )
        return CropResult(crops=crops, boxes=boxes, pose=pose)
