"""Fixed label sets for FashionSigLIP zero-shot scoring."""

UPPER_LABELS = [
    "white formal button-down shirt",
    "light blue formal button-down shirt",
    "official company polo shirt",
    "casual t-shirt",
    "striped t-shirt",
    "plaid or checkered shirt",
    "non-official colored polo shirt",
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

CHEST_LABELS = [
    "blue lanyard with ID badge visible",
    "no ID badge visible",
]

REGION_LABELS: dict[str, list[str]] = {
    "upper": UPPER_LABELS,
    "lower": LOWER_LABELS,
    "feet": FEET_LABELS,
    "chest": CHEST_LABELS,
}
