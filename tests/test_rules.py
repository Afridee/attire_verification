"""Unit tests for the dress-code rule engine (no GPU / MediaPipe)."""

import pytest

from attire_verification.labels import (
    BEIGE_FORMAL_SHIRT_LABEL,
    CASUAL_TROUSERS_LABEL,
    ID_BADGE_VISIBLE_LABEL,
    LIGHT_BLUE_FORMAL_SHIRT_LABEL,
    LIGHT_GRAY_FORMAL_SHIRT_LABEL,
    MINT_GREEN_FORMAL_SHIRT_LABEL,
    NAVY_FORMAL_SHIRT_LABEL,
    NO_ID_BADGE_LABEL,
    NON_FORMAL_SHIRT_LABEL,
    OFFICIAL_POLO_LABEL,
    WHITE_FORMAL_SHIRT_LABEL,
)
from attire_verification.models import RegionScore, RegionScores
from attire_verification.roles import Role
from attire_verification.rules import apply_rules


def _score(top: str, top_s: float, second: str, second_s: float) -> RegionScore:
    return RegionScore(
        topLabel=top,
        topScore=top_s,
        secondLabel=second,
        secondScore=second_s,
        scores={top: top_s, second: second_s},
    )


def _badge(*, visible: bool = True, top_s: float = 0.80, second_s: float = 0.12) -> RegionScore:
    if visible:
        return _score(ID_BADGE_VISIBLE_LABEL, top_s, NO_ID_BADGE_LABEL, second_s)
    return _score(NO_ID_BADGE_LABEL, top_s, ID_BADGE_VISIBLE_LABEL, second_s)


def _formal(
    *,
    chest: RegionScore | None = None,
    include_badge: bool = True,
) -> RegionScores:
    return RegionScores(
        upper=_score(
            WHITE_FORMAL_SHIRT_LABEL, 0.82, LIGHT_BLUE_FORMAL_SHIRT_LABEL, 0.10
        ),
        lower=_score("black formal trousers", 0.79, "dark navy trousers", 0.12),
        feet=_score("leather loafers", 0.76, "formal closed shoes", 0.14),
        chest=_badge() if include_badge and chest is None else chest,
    )


def test_sandals_and_casual_shirt_score_50():
    regions = RegionScores(
        upper=_score(NON_FORMAL_SHIRT_LABEL, 0.81, WHITE_FORMAL_SHIRT_LABEL, 0.12),
        lower=_score("black formal trousers", 0.74, "dark navy trousers", 0.15),
        feet=_score("sandals or slides", 0.88, "single-color sober sneakers", 0.08),
        chest=_badge(),
    )
    result = apply_rules(regions)
    assert result.score == "50/100"
    assert result.regionPoints is not None
    assert result.regionPoints.upper == 0
    assert result.regionPoints.lower == 25
    assert result.regionPoints.feet == 0
    assert result.regionPoints.chest == 25
    assert "open_footwear" in result.failReasons
    assert "casual_shirt" in result.failReasons
    assert result.role == Role.BR.value


def test_formal_wear_scores_100():
    result = apply_rules(_formal())
    assert result.score == "100/100"
    assert result.failReasons == []
    assert result.role == Role.BR.value
    assert result.regionPoints is not None
    assert result.regionPoints.upper == 25
    assert result.regionPoints.lower == 25
    assert result.regionPoints.feet == 25
    assert result.regionPoints.chest == 25


@pytest.mark.parametrize(
    "shirt_label",
    [
        MINT_GREEN_FORMAL_SHIRT_LABEL,
        BEIGE_FORMAL_SHIRT_LABEL,
        NAVY_FORMAL_SHIRT_LABEL,
        LIGHT_GRAY_FORMAL_SHIRT_LABEL,
    ],
)
def test_extra_solid_formal_shirts_score_100(shirt_label: str):
    regions = RegionScores(
        upper=_score(shirt_label, 0.82, WHITE_FORMAL_SHIRT_LABEL, 0.10),
        lower=_score("black formal trousers", 0.79, "dark navy trousers", 0.12),
        feet=_score("leather loafers", 0.76, "formal closed shoes", 0.14),
        chest=_badge(),
    )
    result = apply_rules(regions)
    assert result.score == "100/100"
    assert result.failReasons == []
    assert result.regionPoints is not None
    assert result.regionPoints.upper == 25


def test_official_polo_visual_label_scores_100():
    regions = RegionScores(
        upper=_score(
            OFFICIAL_POLO_LABEL, 0.82, NON_FORMAL_SHIRT_LABEL, 0.10
        ),
        lower=_score("dark navy trousers", 0.79, "black formal trousers", 0.12),
        feet=_score("leather loafers", 0.76, "formal closed shoes", 0.14),
        chest=_badge(),
    )
    result = apply_rules(regions)
    assert result.score == "100/100"
    assert result.failReasons == []


def test_low_confidence_upper_scores_75():
    regions = RegionScores(
        upper=_score(
            WHITE_FORMAL_SHIRT_LABEL, 0.35, LIGHT_BLUE_FORMAL_SHIRT_LABEL, 0.25
        ),
        lower=_score("black formal trousers", 0.79, "dark navy trousers", 0.12),
        feet=_score("leather loafers", 0.76, "formal closed shoes", 0.14),
        chest=_badge(),
    )
    result = apply_rules(regions)
    assert result.score == "75/100"
    assert result.regionPoints is not None
    assert result.regionPoints.upper == 0
    assert result.failReasons == ["low_confidence"]


def test_sober_sneakers_score_100():
    regions = RegionScores(
        upper=_score(
            WHITE_FORMAL_SHIRT_LABEL, 0.82, LIGHT_BLUE_FORMAL_SHIRT_LABEL, 0.10
        ),
        lower=_score("black formal trousers", 0.79, "dark navy trousers", 0.12),
        feet=_score("single-color sober sneakers", 0.70, "formal closed shoes", 0.15),
        chest=_badge(),
    )
    result = apply_rules(regions)
    assert result.score == "100/100"
    assert result.regionPoints is not None
    assert result.regionPoints.feet == 25
    assert result.failReasons == []


def test_casual_trousers_score_75():
    regions = RegionScores(
        upper=_score(
            "black polo shirt with purple sleeve trim", 0.80, WHITE_FORMAL_SHIRT_LABEL, 0.10
        ),
        lower=_score(CASUAL_TROUSERS_LABEL, 0.72, "black formal trousers", 0.15),
        feet=_score("formal closed shoes", 0.75, "leather loafers", 0.12),
        chest=_badge(),
    )
    result = apply_rules(regions)
    assert result.score == "75/100"
    assert result.regionPoints is not None
    assert result.regionPoints.lower == 0
    assert "wrong_trousers" in result.failReasons


def test_non_formal_shirt_score_75():
    regions = RegionScores(
        upper=_score(
            NON_FORMAL_SHIRT_LABEL, 0.78, OFFICIAL_POLO_LABEL, 0.12
        ),
        lower=_score("black formal trousers", 0.79, "dark navy trousers", 0.12),
        feet=_score("leather loafers", 0.76, "formal closed shoes", 0.14),
        chest=_badge(),
    )
    result = apply_rules(regions)
    assert result.score == "75/100"
    assert result.regionPoints is not None
    assert result.regionPoints.upper == 0
    assert "casual_shirt" in result.failReasons


def test_low_margin_upper_scores_75():
    regions = RegionScores(
        upper=_score(
            WHITE_FORMAL_SHIRT_LABEL, 0.52, NON_FORMAL_SHIRT_LABEL, 0.46
        ),
        lower=_score("black formal trousers", 0.79, "dark navy trousers", 0.12),
        feet=_score("leather loafers", 0.76, "formal closed shoes", 0.14),
        chest=_badge(),
    )
    result = apply_rules(regions)
    assert result.score == "75/100"
    assert result.failReasons == ["low_confidence"]


def _polo_regions(*, upper_top: str = NON_FORMAL_SHIRT_LABEL) -> RegionScores:
    return RegionScores(
        upper=_score(upper_top, 0.40, OFFICIAL_POLO_LABEL, 0.35),
        lower=_score("black formal trousers", 0.79, "dark navy trousers", 0.12),
        feet=_score("leather loafers", 0.76, "formal closed shoes", 0.14),
        chest=_badge(),
    )


def test_polo_match_promotes_non_formal_shirt():
    result = apply_rules(_polo_regions(), polo_match_score=0.90)
    assert result.score == "100/100"
    assert result.failReasons == []


def test_polo_match_below_threshold_still_zero_upper():
    result = apply_rules(_polo_regions(), polo_match_score=0.70)
    assert result.score == "75/100"
    assert "casual_shirt" in result.failReasons


def test_polo_match_skips_low_upper_confidence():
    result = apply_rules(_polo_regions(), polo_match_score=0.88)
    assert result.score == "100/100"


def test_polo_match_sober_sneakers_score_100():
    regions = RegionScores(
        upper=_score(NON_FORMAL_SHIRT_LABEL, 0.78, OFFICIAL_POLO_LABEL, 0.12),
        lower=_score("black formal trousers", 0.79, "dark navy trousers", 0.12),
        feet=_score("single-color sober sneakers", 0.70, "formal closed shoes", 0.15),
        chest=_badge(),
    )
    result = apply_rules(regions, polo_match_score=0.91)
    assert result.score == "100/100"
    assert result.failReasons == []


def test_br_missing_id_badge_scores_75():
    result = apply_rules(_formal(chest=_badge(visible=False)), role=Role.BR)
    assert result.score == "75/100"
    assert result.regionPoints is not None
    assert result.regionPoints.chest == 0
    assert result.failReasons == ["missing_id_badge"]


def test_br_sup_missing_id_badge_scores_75():
    result = apply_rules(_formal(chest=_badge(visible=False)), role="BR_SUP")
    assert result.score == "75/100"
    assert result.failReasons == ["missing_id_badge"]
    assert result.role == Role.BR_SUP.value


def test_br_missing_chest_crop_scores_75():
    result = apply_rules(_formal(include_badge=False), role=Role.BR)
    assert result.score == "75/100"
    assert result.failReasons == ["low_confidence"]


def test_br_low_badge_confidence_scores_75():
    result = apply_rules(
        _formal(chest=_badge(visible=True, top_s=0.35, second_s=0.25)),
        role=Role.BR,
    )
    assert result.score == "75/100"
    assert result.failReasons == ["low_confidence"]


def test_om_scores_100_without_id_badge():
    result = apply_rules(_formal(chest=_badge(visible=False)), role=Role.OM)
    assert result.score == "100/100"
    assert result.failReasons == []
    assert result.role == Role.OM.value
    assert result.regionPoints is not None
    assert result.regionPoints.chest == 25


def test_om_scores_100_without_chest_crop():
    result = apply_rules(_formal(include_badge=False), role="om")
    assert result.score == "100/100"
    assert result.failReasons == []


def test_br_missing_badge_and_sandals_score_50():
    regions = RegionScores(
        upper=_score(
            WHITE_FORMAL_SHIRT_LABEL, 0.82, LIGHT_BLUE_FORMAL_SHIRT_LABEL, 0.10
        ),
        lower=_score("black formal trousers", 0.79, "dark navy trousers", 0.12),
        feet=_score("sandals or slides", 0.88, "single-color sober sneakers", 0.08),
        chest=_badge(visible=False),
    )
    result = apply_rules(regions, role=Role.BR)
    assert result.score == "50/100"
    assert result.failReasons == ["open_footwear", "missing_id_badge"]


def test_bright_shoes_score_75():
    regions = RegionScores(
        upper=_score(
            WHITE_FORMAL_SHIRT_LABEL, 0.82, LIGHT_BLUE_FORMAL_SHIRT_LABEL, 0.10
        ),
        lower=_score("black formal trousers", 0.79, "dark navy trousers", 0.12),
        feet=_score("brightly colored or neon shoes", 0.73, "single-color sober sneakers", 0.14),
        chest=_badge(),
    )
    result = apply_rules(regions)
    assert result.score == "75/100"
    assert result.regionPoints is not None
    assert result.regionPoints.feet == 0
    assert result.failReasons == ["bright_color"]


def test_near_miss_black_trousers_score_100():
    regions = RegionScores(
        upper=_score(
            WHITE_FORMAL_SHIRT_LABEL, 0.82, LIGHT_BLUE_FORMAL_SHIRT_LABEL, 0.10
        ),
        lower=_score("black formal trousers", 0.52, "dark navy trousers", 0.30),
        feet=_score("leather loafers", 0.76, "formal closed shoes", 0.14),
        chest=_badge(),
    )
    result = apply_rules(regions)
    assert result.score == "100/100"
    assert result.failReasons == []


def test_pass_footwear_split_skips_margin():
    regions = RegionScores(
        upper=_score(
            WHITE_FORMAL_SHIRT_LABEL, 0.82, LIGHT_BLUE_FORMAL_SHIRT_LABEL, 0.10
        ),
        lower=_score("black formal trousers", 0.79, "dark navy trousers", 0.12),
        feet=_score("single-color sober sneakers", 0.55, "formal closed shoes", 0.48),
        chest=_badge(),
    )
    result = apply_rules(regions)
    assert result.score == "100/100"
    assert result.failReasons == []


def test_unknown_role_raises():
    with pytest.raises(ValueError, match="unknown role"):
        apply_rules(_formal(), role="NOT_A_ROLE")
