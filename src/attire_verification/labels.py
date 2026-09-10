"""Fixed label sets for FashionSigLIP zero-shot scoring."""

# Visual descriptions — FashionSigLIP has no notion of "official" vs "non-official".
OFFICIAL_POLO_LABEL = "black polo shirt with purple sleeve trim"
NON_FORMAL_SHIRT_LABEL = "casual or non-formal shirt"
BRIGHT_SHOE_LABEL = "brightly colored or neon shoes"
CASUAL_TROUSERS_LABEL = "casual trousers"

UPPER_LABELS = [
    "white formal button-down shirt tucked in",
    "light blue formal button-down shirt tucked in",
    OFFICIAL_POLO_LABEL,
    NON_FORMAL_SHIRT_LABEL,
]

LOWER_LABELS = [
    "black formal trousers",
    "dark navy trousers",
    CASUAL_TROUSERS_LABEL,
]

FEET_LABELS = [
    "formal closed shoes",
    "leather loafers",
    "single-color sober sneakers",
    "sandals or slides",
    BRIGHT_SHOE_LABEL,
]

ID_BADGE_VISIBLE_LABEL = "blue lanyard with ID badge visible"
NO_ID_BADGE_LABEL = "no ID badge visible"

CHEST_LABELS = [
    ID_BADGE_VISIBLE_LABEL,
    NO_ID_BADGE_LABEL,
]

REGION_LABELS: dict[str, list[str]] = {
    "upper": UPPER_LABELS,
    "lower": LOWER_LABELS,
    "feet": FEET_LABELS,
    "chest": CHEST_LABELS,
}
