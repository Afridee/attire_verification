"""Pydantic schemas for attire verification results."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Status(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    UNCERTAIN = "UNCERTAIN"
    REJECTED = "REJECTED"


class RegionScore(BaseModel):
    topLabel: str
    topScore: float
    secondLabel: str
    secondScore: float
    scores: dict[str, float]


class CropBox(BaseModel):
    """Pixel crop box as [x, y, w, h]."""

    x: int
    y: int
    w: int
    h: int

    def as_list(self) -> list[int]:
        return [self.x, self.y, self.w, self.h]


class PoseInfo(BaseModel):
    fullBodyOk: bool
    landmarksDetected: int = 0
    cropBoxes: dict[str, list[int]] | None = None
    details: dict[str, Any] | None = None


class RegionScores(BaseModel):
    upper: RegionScore | None = None
    lower: RegionScore | None = None
    feet: RegionScore | None = None
    chest: RegionScore | None = None


class PoloMatch(BaseModel):
    """Image-to-image similarity vs bundled official-polo reference crops."""

    score: float
    threshold: float
    matched: bool
    refs: int = 0


class VerifyResult(BaseModel):
    status: Status
    failReasons: list[str] = Field(default_factory=list)
    imagePath: str | None = None
    regions: RegionScores | None = None
    pose: PoseInfo | None = None
    poloMatch: PoloMatch | None = None
    expected: str | None = None  # batch mode only
    match: bool | None = None  # batch mode only
