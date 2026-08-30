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
            "official company polo shirt", 0.80, "white formal button-down shirt", 0.10
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
            "non-official colored polo shirt", 0.78, "official company polo shirt", 0.12
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
