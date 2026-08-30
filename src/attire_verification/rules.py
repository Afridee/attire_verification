"""Unified dress-code rule engine."""

from __future__ import annotations

from enum import Enum

from attire_verification.models import RegionScore, RegionScores, Status, VerifyResult

MIN_CONFIDENCE = 0.55
MIN_MARGIN = 0.08


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
    "official company polo shirt": ShirtType.OFFICIAL_POLO,
    "casual t-shirt": ShirtType.CASUAL_TEE,
    "striped t-shirt": ShirtType.STRIPED,
    "plaid or checkered shirt": ShirtType.PLAID,
    "non-official colored polo shirt": ShirtType.NON_OFFICIAL_POLO,
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


def apply_rules(
    regions: RegionScores,
    *,
    min_confidence: float = MIN_CONFIDENCE,
    min_margin: float = MIN_MARGIN,
    image_path: str | None = None,
) -> VerifyResult:
    """Apply unified dress-code rules to region scores."""
    if regions.upper is None or regions.lower is None or regions.feet is None:
        return VerifyResult(
            status=Status.UNCERTAIN,
            failReasons=["low_confidence"],
            imagePath=image_path,
            regions=regions,
        )

    upper = regions.upper
    lower = regions.lower
    feet = regions.feet

    shirt = SHIRT_LABEL_MAP.get(upper.topLabel)
    trouser = TROUSER_LABEL_MAP.get(lower.topLabel)
    footwear = FOOTWEAR_LABEL_MAP.get(feet.topLabel)

    fail_reasons: list[str] = []

    # Auto FAIL checks (clear violations take precedence)
    if shirt in FAIL_SHIRTS:
        if shirt == ShirtType.NON_OFFICIAL_POLO:
            fail_reasons.append("non_official_polo")
        else:
            fail_reasons.append("casual_shirt")

    if footwear == FootwearType.SANDALS:
        fail_reasons.append("open_footwear")

    if trouser == TrouserType.BEIGE_CHINOS:
        fail_reasons.append("wrong_trousers")

    if fail_reasons:
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique = [r for r in fail_reasons if not (r in seen or seen.add(r))]
        return VerifyResult(
            status=Status.FAILED,
            failReasons=unique,
            imagePath=image_path,
            regions=regions,
        )

    # Sneakers are borderline — never auto-pass in v1
    if footwear == FootwearType.SNEAKERS:
        return VerifyResult(
            status=Status.UNCERTAIN,
            failReasons=["borderline_footwear"],
            imagePath=image_path,
            regions=regions,
        )

    # Confidence / margin gates
    scored_regions = [upper, lower, feet]
    for region in scored_regions:
        if region.topScore < min_confidence:
            return VerifyResult(
                status=Status.UNCERTAIN,
                failReasons=["low_confidence"],
                imagePath=image_path,
                regions=regions,
            )
        if _margin(region) < min_margin:
            return VerifyResult(
                status=Status.UNCERTAIN,
                failReasons=["low_confidence"],
                imagePath=image_path,
                regions=regions,
            )

    # Auto PASS if all regions are acceptable
    if (
        shirt in PASS_SHIRTS
        and trouser in PASS_TROUSERS
        and footwear in PASS_FEET
    ):
        return VerifyResult(
            status=Status.PASSED,
            failReasons=[],
            imagePath=image_path,
            regions=regions,
        )

    # Unknown / unmatched labels
    return VerifyResult(
        status=Status.UNCERTAIN,
        failReasons=["low_confidence"],
        imagePath=image_path,
        regions=regions,
    )
