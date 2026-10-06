"""Fixed label sets for FashionSigLIP zero-shot scoring."""

# Visual descriptions — FashionSigLIP has no notion of "official" vs "non-official".
OFFICIAL_POLO_LABEL = "black polo shirt with purple sleeve trim"
NON_FORMAL_SHIRT_LABEL = "casual or non-formal shirt"
BRIGHT_SHOE_LABEL = "brightly colored or neon shoes"
CASUAL_TROUSERS_LABEL = "casual trousers"
GREY_TROUSERS_LABEL = "grey trousers"
LIGHT_GREY_TROUSERS_LABEL = "light grey trousers"

WHITE_FORMAL_SHIRT_LABEL = "white formal button-down shirt tucked in"
LIGHT_BLUE_FORMAL_SHIRT_LABEL = "light blue formal button-down shirt tucked in"
MINT_GREEN_FORMAL_SHIRT_LABEL = "light mint green formal button-down shirt tucked in"
BEIGE_FORMAL_SHIRT_LABEL = "beige formal button-down shirt tucked in"
NAVY_FORMAL_SHIRT_LABEL = "navy formal button-down shirt tucked in"
LIGHT_GRAY_FORMAL_SHIRT_LABEL = "light gray formal button-down shirt tucked in"

FORMAL_SHIRT_LABELS = [
    WHITE_FORMAL_SHIRT_LABEL,
    LIGHT_BLUE_FORMAL_SHIRT_LABEL,
    MINT_GREEN_FORMAL_SHIRT_LABEL,
    BEIGE_FORMAL_SHIRT_LABEL,
    NAVY_FORMAL_SHIRT_LABEL,
    LIGHT_GRAY_FORMAL_SHIRT_LABEL,
]

UPPER_LABELS = [
    *FORMAL_SHIRT_LABELS,
    OFFICIAL_POLO_LABEL,
    NON_FORMAL_SHIRT_LABEL,
]

LOWER_LABELS = [
    "black trousers",
    "dark navy trousers",
    GREY_TROUSERS_LABEL,
    LIGHT_GREY_TROUSERS_LABEL,
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
