"""Unified dress-code rule engine."""

from __future__ import annotations

from enum import Enum

from attire_verification.labels import (
    BRIGHT_SHOE_LABEL,
    CASUAL_TROUSERS_LABEL,
    ID_BADGE_VISIBLE_LABEL,
    NO_ID_BADGE_LABEL,
    NON_FORMAL_SHIRT_LABEL,
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

MIN_CONFIDENCE = 0.50
MIN_MARGIN = 0.08
# Weak "bright/neon" shoe guesses are dump-bin labels. Only treat them as a
# proven colour violation when the model is actually sure.
BRIGHT_MIN_CONFIDENCE = 0.70
# Cosine vs official-polo reference crops. Same shirt in different photos
# scored ~0.88–0.92; other garments ~0.68–0.78 on the sample set.
POLO_MATCH_THRESHOLD = 0.85


class ShirtType(str, Enum):
    WHITE_FORMAL = "WHITE_FORMAL"
    LIGHT_BLUE_FORMAL = "LIGHT_BLUE_FORMAL"
    OFFICIAL_POLO = "OFFICIAL_POLO"
    NON_FORMAL = "NON_FORMAL"


class TrouserType(str, Enum):
    BLACK_TROUSERS = "BLACK_TROUSERS"
    NAVY_TROUSERS = "NAVY_TROUSERS"
    CASUAL_TROUSERS = "CASUAL_TROUSERS"


class FootwearType(str, Enum):
    FORMAL_CLOSED = "FORMAL_CLOSED"
    LOAFERS = "LOAFERS"
    SNEAKERS = "SNEAKERS"
    SANDALS = "SANDALS"
    BRIGHT_SHOES = "BRIGHT_SHOES"


SHIRT_LABEL_MAP: dict[str, ShirtType] = {
    "white formal button-down shirt tucked in": ShirtType.WHITE_FORMAL,
    "light blue formal button-down shirt tucked in": ShirtType.LIGHT_BLUE_FORMAL,
    OFFICIAL_POLO_LABEL: ShirtType.OFFICIAL_POLO,
    NON_FORMAL_SHIRT_LABEL: ShirtType.NON_FORMAL,
}

TROUSER_LABEL_MAP: dict[str, TrouserType] = {
    "black formal trousers": TrouserType.BLACK_TROUSERS,
    "dark navy trousers": TrouserType.NAVY_TROUSERS,
    CASUAL_TROUSERS_LABEL: TrouserType.CASUAL_TROUSERS,
}

FOOTWEAR_LABEL_MAP: dict[str, FootwearType] = {
    "formal closed shoes": FootwearType.FORMAL_CLOSED,
    "leather loafers": FootwearType.LOAFERS,
    "single-color sober sneakers": FootwearType.SNEAKERS,
    "sandals or slides": FootwearType.SANDALS,
    BRIGHT_SHOE_LABEL: FootwearType.BRIGHT_SHOES,
}

FAIL_SHIRTS = {ShirtType.NON_FORMAL}

PASS_SHIRTS = {
    ShirtType.WHITE_FORMAL,
    ShirtType.LIGHT_BLUE_FORMAL,
    ShirtType.OFFICIAL_POLO,
}

PASS_TROUSERS = {TrouserType.BLACK_TROUSERS, TrouserType.NAVY_TROUSERS}
FAIL_TROUSERS = {TrouserType.CASUAL_TROUSERS}
PASS_FEET = {FootwearType.FORMAL_CLOSED, FootwearType.LOAFERS, FootwearType.SNEAKERS}

_BRIGHT_LABELS = {BRIGHT_SHOE_LABEL}


def _margin(score: RegionScore) -> float:
    return score.topScore - score.secondScore


def _confident(
    score: RegionScore,
    min_confidence: float,
    min_margin: float,
) -> bool:
    return score.topScore >= min_confidence and _margin(score) >= min_margin


def _bright_confirmed(score: RegionScore, min_margin: float) -> bool:
    return score.topScore >= BRIGHT_MIN_CONFIDENCE and _margin(score) >= min_margin


def _demote_unconfirmed_bright(score: RegionScore, min_margin: float) -> RegionScore:
    """If top is a weak bright/neon guess, surface the next-best label instead.

    Keeps original ``scores`` so the discarded neon probability is still visible.
    """
    if score.topLabel not in _BRIGHT_LABELS or _bright_confirmed(score, min_margin):
        return score
    others = [(label, value) for label, value in score.scores.items() if label not in _BRIGHT_LABELS]
    if not others:
        return score
    ranked = sorted(others, key=lambda item: item[1], reverse=True)
    top_label, top_score = ranked[0]
    second_label, second_score = ranked[1] if len(ranked) > 1 else ("", 0.0)
    return RegionScore(
        topLabel=top_label,
        topScore=top_score,
        secondLabel=second_label,
        secondScore=second_score,
        scores=score.scores,
    )


def _passes_allowed(
    score: RegionScore,
    kind: object | None,
    pass_set: set,
    label_map: dict,
    min_confidence: float,
    min_margin: float,
) -> bool:
    """Pass when the top label is allowed and the model is sure enough.

    Margin is skipped when second place is also allowed (e.g. loafers vs
    formal shoes, black vs navy trousers) — that split is not a dress-code doubt.
    """
    if kind not in pass_set or score.topScore < min_confidence:
        return False
    second_kind = label_map.get(score.secondLabel)
    if second_kind in pass_set:
        return True
    return _margin(score) >= min_margin


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
    low confidence, and missing crops score 0.
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

    resolved = RegionScores(
        upper=regions.upper,
        lower=regions.lower,
        feet=_demote_unconfirmed_bright(regions.feet, min_margin) if regions.feet else None,
        chest=regions.chest,
    )

    upper = resolved.upper
    if upper is None:
        fail_reasons.append("low_confidence")
    else:
        shirt = SHIRT_LABEL_MAP.get(upper.topLabel)
        if polo_matched:
            shirt = ShirtType.OFFICIAL_POLO
        if shirt in FAIL_SHIRTS:
            fail_reasons.append("casual_shirt")
        elif shirt in PASS_SHIRTS and (
            polo_matched
            or _passes_allowed(
                upper, shirt, PASS_SHIRTS, SHIRT_LABEL_MAP, min_confidence, min_margin
            )
        ):
            points.upper = REGION_POINTS
        else:
            fail_reasons.append("low_confidence")

    feet = resolved.feet
    if feet is None:
        fail_reasons.append("low_confidence")
    else:
        footwear = FOOTWEAR_LABEL_MAP.get(feet.topLabel)
        if footwear == FootwearType.SANDALS:
            fail_reasons.append("open_footwear")
        elif footwear == FootwearType.BRIGHT_SHOES:
            fail_reasons.append("bright_color")
        elif _passes_allowed(
            feet, footwear, PASS_FEET, FOOTWEAR_LABEL_MAP, min_confidence, min_margin
        ):
            points.feet = REGION_POINTS
        else:
            fail_reasons.append("low_confidence")

    lower = resolved.lower
    if lower is None:
        fail_reasons.append("low_confidence")
    else:
        trouser = TROUSER_LABEL_MAP.get(lower.topLabel)
        if trouser in FAIL_TROUSERS:
            fail_reasons.append("wrong_trousers")
        elif _passes_allowed(
            lower, trouser, PASS_TROUSERS, TROUSER_LABEL_MAP, min_confidence, min_margin
        ):
            points.lower = REGION_POINTS
        else:
            fail_reasons.append("low_confidence")

    chest = resolved.chest
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
        regions=resolved,
        regionPoints=points,
        role=canonical_role.value,
    )
