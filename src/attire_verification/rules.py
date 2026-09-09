"""Unified dress-code rule engine."""

from __future__ import annotations

from enum import Enum

from attire_verification.labels import (
    COLORED_POLO_LABEL,
    ID_BADGE_VISIBLE_LABEL,
    NO_ID_BADGE_LABEL,
    OFFICIAL_POLO_LABEL,
)
from attire_verification.models import (
    REGION_POINTS,
    RegionPoints,
    RegionScore,
    RegionScores,
    VerifyResult,
    format_score,
)
from attire_verification.roles import Role, parse_role, requires_id_badge

MIN_CONFIDENCE = 0.55
MIN_MARGIN = 0.08
# Cosine vs official-polo reference crops. Same shirt in different photos
# scored ~0.88–0.92; other garments ~0.68–0.78 on the sample set.
POLO_MATCH_THRESHOLD = 0.85


class ShirtType(str, Enum):
    WHITE_FORMAL = "WHITE_FORMAL"
    LIGHT_BLUE_FORMAL = "LIGHT_BLUE_FORMAL"
    OFFICIAL_POLO = "OFFICIAL_POLO"
    CASUAL_TEE = "CASUAL_TEE"
    STRIPED = "STRIPED"
    PLAID = "PLAID"
    NON_OFFICIAL_POLO = "NON_OFFICIAL_POLO"


class TrouserType(str, Enum):
    BLACK_TROUSERS = "BLACK_TROUSERS"
    NAVY_TROUSERS = "NAVY_TROUSERS"
    BEIGE_CHINOS = "BEIGE_CHINOS"


class FootwearType(str, Enum):
    FORMAL_CLOSED = "FORMAL_CLOSED"
    LOAFERS = "LOAFERS"
    SNEAKERS = "SNEAKERS"
    SANDALS = "SANDALS"


SHIRT_LABEL_MAP: dict[str, ShirtType] = {
    "white formal button-down shirt": ShirtType.WHITE_FORMAL,
    "light blue formal button-down shirt": ShirtType.LIGHT_BLUE_FORMAL,
    OFFICIAL_POLO_LABEL: ShirtType.OFFICIAL_POLO,
    "casual t-shirt": ShirtType.CASUAL_TEE,
    "striped t-shirt": ShirtType.STRIPED,
    "plaid or checkered shirt": ShirtType.PLAID,
    COLORED_POLO_LABEL: ShirtType.NON_OFFICIAL_POLO,
}

TROUSER_LABEL_MAP: dict[str, TrouserType] = {
    "black formal trousers": TrouserType.BLACK_TROUSERS,
    "dark navy trousers": TrouserType.NAVY_TROUSERS,
    "beige or tan chinos": TrouserType.BEIGE_CHINOS,
}

FOOTWEAR_LABEL_MAP: dict[str, FootwearType] = {
    "black formal closed shoes": FootwearType.FORMAL_CLOSED,
    "black leather loafers": FootwearType.LOAFERS,
    "white sneakers": FootwearType.SNEAKERS,
    "sandals or slides": FootwearType.SANDALS,
}

FAIL_SHIRTS = {
    ShirtType.CASUAL_TEE,
    ShirtType.STRIPED,
    ShirtType.PLAID,
    ShirtType.NON_OFFICIAL_POLO,
}

PASS_SHIRTS = {
    ShirtType.WHITE_FORMAL,
    ShirtType.LIGHT_BLUE_FORMAL,
    ShirtType.OFFICIAL_POLO,
}

PASS_TROUSERS = {TrouserType.BLACK_TROUSERS, TrouserType.NAVY_TROUSERS}
PASS_FEET = {FootwearType.FORMAL_CLOSED, FootwearType.LOAFERS}


def _margin(score: RegionScore) -> float:
    return score.topScore - score.secondScore


def _confident(
    score: RegionScore,
    min_confidence: float,
    min_margin: float,
) -> bool:
    return score.topScore >= min_confidence and _margin(score) >= min_margin


def _unique(reasons: list[str]) -> list[str]:
    seen: set[str] = set()
    return [r for r in reasons if not (r in seen or seen.add(r))]


def apply_rules(
    regions: RegionScores,
    *,
    min_confidence: float = MIN_CONFIDENCE,
    min_margin: float = MIN_MARGIN,
    polo_match_score: float | None = None,
    polo_match_threshold: float = POLO_MATCH_THRESHOLD,
    image_path: str | None = None,
    role: str | Role = Role.BR,
) -> VerifyResult:
    """Score each region independently (25 points each, 100 total).

    A region earns 25 only when it clearly meets dress code. Violations,
    borderline items, low confidence, and missing crops score 0.
    BR and BR_SUP must show a visible ID badge on the chest crop; other
    roles auto-pass chest.
    """
    canonical_role = parse_role(role)
    need_badge = requires_id_badge(canonical_role)
    fail_reasons: list[str] = []
    points = RegionPoints()

    polo_matched = (
        polo_match_score is not None and polo_match_score >= polo_match_threshold
    )

    upper = regions.upper
    if upper is None:
        fail_reasons.append("low_confidence")
    else:
        shirt = SHIRT_LABEL_MAP.get(upper.topLabel)
        if polo_matched:
            shirt = ShirtType.OFFICIAL_POLO
        if shirt in FAIL_SHIRTS:
            if shirt == ShirtType.NON_OFFICIAL_POLO:
                fail_reasons.append("non_official_polo")
            else:
                fail_reasons.append("casual_shirt")
        elif shirt in PASS_SHIRTS and (polo_matched or _confident(upper, min_confidence, min_margin)):
            points.upper = REGION_POINTS
        else:
            fail_reasons.append("low_confidence")

    feet = regions.feet
    if feet is None:
        fail_reasons.append("low_confidence")
    else:
        footwear = FOOTWEAR_LABEL_MAP.get(feet.topLabel)
        if footwear == FootwearType.SANDALS:
            fail_reasons.append("open_footwear")
        elif footwear == FootwearType.SNEAKERS:
            fail_reasons.append("borderline_footwear")
        elif footwear in PASS_FEET and _confident(feet, min_confidence, min_margin):
            points.feet = REGION_POINTS
        else:
            fail_reasons.append("low_confidence")

    lower = regions.lower
    if lower is None:
        fail_reasons.append("low_confidence")
    else:
        trouser = TROUSER_LABEL_MAP.get(lower.topLabel)
        if trouser == TrouserType.BEIGE_CHINOS:
            fail_reasons.append("wrong_trousers")
        elif trouser in PASS_TROUSERS and _confident(lower, min_confidence, min_margin):
            points.lower = REGION_POINTS
        else:
            fail_reasons.append("low_confidence")

    chest = regions.chest
    if not need_badge:
        points.chest = REGION_POINTS
    elif chest is None:
        fail_reasons.append("low_confidence")
    elif chest.topLabel == NO_ID_BADGE_LABEL:
        fail_reasons.append("missing_id_badge")
    elif chest.topLabel == ID_BADGE_VISIBLE_LABEL and _confident(
        chest, min_confidence, min_margin
    ):
        points.chest = REGION_POINTS
    else:
        fail_reasons.append("low_confidence")

    total = points.upper + points.lower + points.feet + points.chest
    return VerifyResult(
        score=format_score(total),
        failReasons=_unique(fail_reasons),
        imagePath=image_path,
        regions=regions,
        regionPoints=points,
        role=canonical_role.value,
    )
