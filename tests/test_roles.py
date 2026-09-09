"""Unit tests for role parsing and ID-badge requirements."""

import pytest

from attire_verification.roles import Role, parse_role, requires_id_badge


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("BR", Role.BR),
        ("br", Role.BR),
        (Role.BR, Role.BR),
        ("BR_SUP", Role.BR_SUP),
        ("br_sup", Role.BR_SUP),
        ("brsup", Role.BR_SUP),
        ("Synergy", Role.SYNERGY),
        ("synergy", Role.SYNERGY),
        ("all access", Role.ALL_ACCESS),
        ("ALL_ACCESS", Role.ALL_ACCESS),
        ("Test Role", Role.TEST_ROLE),
        ("AKT-DX", Role.AKT_DX),
        ("akt tm", Role.AKT_TM),
        ("OM", Role.OM),
    ],
)
def test_parse_role_aliases(raw: str | Role, expected: Role):
    assert parse_role(raw) is expected


def test_parse_role_rejects_unknown():
    with pytest.raises(ValueError, match="unknown role"):
        parse_role("INTERN")


def test_parse_role_rejects_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        parse_role("   ")


@pytest.mark.parametrize(
    ("role", "needed"),
    [
        (Role.BR, True),
        (Role.BR_SUP, True),
        ("br", True),
        (Role.OM, False),
        (Role.SUPER_ADMIN, False),
        (Role.FC, False),
        (Role.DXO, False),
        (Role.ALL_ACCESS, False),
        (Role.AKT_TM, False),
    ],
)
def test_requires_id_badge(role: str | Role, needed: bool):
    assert requires_id_badge(role) is needed
