"""Staff roles and role-specific dress-code requirements.

Values match the mobile `RoleInfo` constants so the same strings can be
passed through from the app.
"""

from __future__ import annotations

from enum import Enum


class Role(str, Enum):
    SUPER_ADMIN = "SUPER_ADMIN"
    OM = "OM"
    BR = "BR"
    BR_SUP = "BR_SUP"
    FC = "FC"
    SYNERGY = "Synergy"
    DXO = "DXO"
    ADXO = "ADXO"
    DDXO = "DDXO"
    DTMO = "DTMO"
    TMS = "TMS"
    TMO = "TMO"
    TMR = "TMR"
    ALL_ACCESS = "all access"
    TEST_ROLE = "Test Role"
    AKT_DX = "AKT-DX"
    AKT_TM = "AKT TM"


# Branch roles must wear a visible ID badge / lanyard.
ROLES_REQUIRING_ID_BADGE: frozenset[Role] = frozenset({Role.BR, Role.BR_SUP})

_VALUE_BY_LOWER = {role.value.lower(): role for role in Role}
_NAME_BY_LOWER = {role.name.lower(): role for role in Role}

_ALIASES: dict[str, Role] = {
    "brsup": Role.BR_SUP,
    "superadmin": Role.SUPER_ADMIN,
    "allaccess": Role.ALL_ACCESS,
    "all_access": Role.ALL_ACCESS,
    "testrole": Role.TEST_ROLE,
    "test_role": Role.TEST_ROLE,
    "aktdx": Role.AKT_DX,
    "akt_dx": Role.AKT_DX,
    "akttm": Role.AKT_TM,
    "akt_tm": Role.AKT_TM,
}


def parse_role(value: str | Role) -> Role:
    """Resolve a role string (canonical value, enum name, or alias) to `Role`."""
    if isinstance(value, Role):
        return value
    key = value.strip()
    if not key:
        raise ValueError("role must not be empty")

    lower = key.lower()
    if lower in _VALUE_BY_LOWER:
        return _VALUE_BY_LOWER[lower]
    if lower in _NAME_BY_LOWER:
        return _NAME_BY_LOWER[lower]

    compact = lower.replace("-", "_").replace(" ", "_")
    if compact in _ALIASES:
        return _ALIASES[compact]
    nosep = compact.replace("_", "")
    if nosep in _ALIASES:
        return _ALIASES[nosep]

    known = ", ".join(role.value for role in Role)
    raise ValueError(f"unknown role {value!r}; expected one of: {known}")


def requires_id_badge(role: str | Role) -> bool:
    """True when the role must show an ID badge (BR and BR_SUP)."""
    return parse_role(role) in ROLES_REQUIRING_ID_BADGE
