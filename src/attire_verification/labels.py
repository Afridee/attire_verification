"""Fixed label sets for FashionSigLIP zero-shot scoring."""

# Visual descriptions — FashionSigLIP has no notion of "official" vs "non-official".
OFFICIAL_POLO_LABEL = "black polo shirt with purple sleeve trim"
COLORED_POLO_LABEL = "colored polo shirt"

UPPER_LABELS = [
    "white formal button-down shirt",
    "light blue formal button-down shirt",
    OFFICIAL_POLO_LABEL,
    "casual t-shirt",
    "striped t-shirt",
    "plaid or checkered shirt",
    COLORED_POLO_LABEL,
]

LOWER_LABELS = [
    "black formal trousers",
    "dark navy trousers",
    "beige or tan chinos",
]

FEET_LABELS = [
    "black formal closed shoes",
    "black leather loafers",
    "white sneakers",
    "sandals or slides",
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
