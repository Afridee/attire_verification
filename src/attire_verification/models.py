"""Pydantic schemas for attire verification results."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

REGION_POINTS = 25
MAX_SCORE = 100
PERFECT_SCORE = f"{MAX_SCORE}/{MAX_SCORE}"
ZERO_SCORE = f"0/{MAX_SCORE}"


def format_score(points: int) -> str:
    """Format earned points as ``N/100``."""
    return f"{points}/{MAX_SCORE}"


class RegionPoints(BaseModel):
    """Points earned per region (0 or 25)."""

    upper: int = 0
    lower: int = 0
    feet: int = 0
    chest: int = 0


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
    score: str
    failReasons: list[str] = Field(default_factory=list)
    imagePath: str | None = None
    regions: RegionScores | None = None
    regionPoints: RegionPoints | None = None
    pose: PoseInfo | None = None
    poloMatch: PoloMatch | None = None
    role: str | None = None
    expected: str | None = None  # batch mode only
    match: bool | None = None  # batch mode only

    def public_dict(self) -> dict[str, Any]:
        """JSON payload shared by the CLI and the HTTP API.

        Drops batch-only and unused optional fields so a single-image response
        does not include nulls for ``expected``, ``match``, and the like.
        """
        data = self.model_dump(mode="json", exclude_none=False)
        for key in ("expected", "match", "poloMatch", "role", "regionPoints"):
            if data.get(key) is None:
                data.pop(key, None)
        return data
