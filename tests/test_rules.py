"""Unit tests for the dress-code rule engine (no GPU / MediaPipe)."""

from attire_verification.models import RegionScore, RegionScores, Status
from attire_verification.rules import apply_rules


def _score(top: str, top_s: float, second: str, second_s: float) -> RegionScore:
    return RegionScore(
        topLabel=top,
        topScore=top_s,
        secondLabel=second,
        secondScore=second_s,
        scores={top: top_s, second: second_s},
    )


def test_sandals_and_striped_shirt_fail():
    regions = RegionScores(
        upper=_score("striped t-shirt", 0.81, "casual t-shirt", 0.12),
        lower=_score("black formal trousers", 0.74, "dark navy trousers", 0.15),
        feet=_score("sandals or slides", 0.88, "white sneakers", 0.08),
    )
    result = apply_rules(regions)
    assert result.status == Status.FAILED
    assert "open_footwear" in result.failReasons
    assert "casual_shirt" in result.failReasons


def test_formal_wear_passes():
    regions = RegionScores(
        upper=_score(
            "white formal button-down shirt", 0.82, "light blue formal button-down shirt", 0.10
        ),
        lower=_score("black formal trousers", 0.79, "dark navy trousers", 0.12),
        feet=_score("black leather loafers", 0.76, "black formal closed shoes", 0.14),
    )
    result = apply_rules(regions)
    assert result.status == Status.PASSED
    assert result.failReasons == []


def test_official_polo_visual_label_passes():
    regions = RegionScores(
        upper=_score(
            "black polo shirt with purple sleeve trim", 0.82, "colored polo shirt", 0.10
        ),
        lower=_score("dark navy trousers", 0.79, "black formal trousers", 0.12),
        feet=_score("black leather loafers", 0.76, "black formal closed shoes", 0.14),
    )
    result = apply_rules(regions)
    assert result.status == Status.PASSED
    assert result.failReasons == []


def test_low_confidence_uncertain():
    regions = RegionScores(
        upper=_score(
            "white formal button-down shirt", 0.40, "light blue formal button-down shirt", 0.30
        ),
        lower=_score("black formal trousers", 0.79, "dark navy trousers", 0.12),
        feet=_score("black leather loafers", 0.76, "black formal closed shoes", 0.14),
    )
    result = apply_rules(regions)
    assert result.status == Status.UNCERTAIN
    assert result.failReasons == ["low_confidence"]


def test_sneakers_uncertain():
    regions = RegionScores(
        upper=_score(
            "white formal button-down shirt", 0.82, "light blue formal button-down shirt", 0.10
        ),
        lower=_score("black formal trousers", 0.79, "dark navy trousers", 0.12),
        feet=_score("white sneakers", 0.70, "black formal closed shoes", 0.15),
    )
    result = apply_rules(regions)
    assert result.status == Status.UNCERTAIN
    assert result.failReasons == ["borderline_footwear"]


def test_beige_chinos_fail():
    regions = RegionScores(
        upper=_score(
            "black polo shirt with purple sleeve trim", 0.80, "white formal button-down shirt", 0.10
        ),
        lower=_score("beige or tan chinos", 0.72, "black formal trousers", 0.15),
        feet=_score("black formal closed shoes", 0.75, "black leather loafers", 0.12),
    )
    result = apply_rules(regions)
    assert result.status == Status.FAILED
    assert "wrong_trousers" in result.failReasons


def test_non_official_polo_fail():
    regions = RegionScores(
        upper=_score(
            "colored polo shirt", 0.78, "black polo shirt with purple sleeve trim", 0.12
        ),
        lower=_score("black formal trousers", 0.79, "dark navy trousers", 0.12),
        feet=_score("black leather loafers", 0.76, "black formal closed shoes", 0.14),
    )
    result = apply_rules(regions)
    assert result.status == Status.FAILED
    assert "non_official_polo" in result.failReasons


def test_low_margin_uncertain():
    regions = RegionScores(
        upper=_score(
            "white formal button-down shirt", 0.50, "light blue formal button-down shirt", 0.45
        ),
        lower=_score("black formal trousers", 0.79, "dark navy trousers", 0.12),
        feet=_score("black leather loafers", 0.76, "black formal closed shoes", 0.14),
    )
    result = apply_rules(regions)
    assert result.status == Status.UNCERTAIN
    assert result.failReasons == ["low_confidence"]


def _polo_regions(*, upper_top: str = "colored polo shirt") -> RegionScores:
    return RegionScores(
        upper=_score(upper_top, 0.40, "black polo shirt with purple sleeve trim", 0.35),
        lower=_score("black formal trousers", 0.79, "dark navy trousers", 0.12),
        feet=_score("black leather loafers", 0.76, "black formal closed shoes", 0.14),
    )


def test_polo_match_promotes_colored_polo():
    result = apply_rules(_polo_regions(), polo_match_score=0.90)
    assert result.status == Status.PASSED
    assert result.failReasons == []


def test_polo_match_below_threshold_still_fails():
    result = apply_rules(_polo_regions(), polo_match_score=0.70)
    assert result.status == Status.FAILED
    assert "non_official_polo" in result.failReasons


def test_polo_match_skips_low_upper_confidence():
    result = apply_rules(_polo_regions(), polo_match_score=0.88)
    assert result.status == Status.PASSED


def test_polo_match_sneakers_still_uncertain():
    regions = RegionScores(
        upper=_score("colored polo shirt", 0.78, "black polo shirt with purple sleeve trim", 0.12),
        lower=_score("black formal trousers", 0.79, "dark navy trousers", 0.12),
        feet=_score("white sneakers", 0.70, "black formal closed shoes", 0.15),
    )
    result = apply_rules(regions, polo_match_score=0.91)
    assert result.status == Status.UNCERTAIN
    assert result.failReasons == ["borderline_footwear"]
