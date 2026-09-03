"""Marqo FashionSigLIP zero-shot region scorer."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from attire_verification.labels import REGION_LABELS
from attire_verification.models import RegionScore

MODEL_ID = "hf-hub:Marqo/marqo-fashionSigLIP"
DEFAULT_POLO_REFS_DIR = Path(__file__).resolve().parent / "refs" / "official_polo"
REF_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

# FashionSigLIP stretches every crop to 224×224. Tall trouser/feet strips then
# look like a smear; huge phone originals also over-emphasize badge/texture.
MAX_CROP_SIDE = 280
TALL_ASPECT = 2.0
LETTERBOX_SIZE = 224
LETTERBOX_FILL = (128, 128, 128)
# Tight full-body phone crops land near this upper size; large originals
# need the same aspect or the stretched 224² view looks like a polo.
UPPER_SIZE = (190, 280)

_TOKEN_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def _load_hf_token() -> None:
    """Load HF_TOKEN from a local .env if the process is not already authenticated."""
    if any(os.environ.get(key) for key in _TOKEN_KEYS):
        return
    repo_root = Path(__file__).resolve().parents[2]
    for env_path in (Path.cwd() / ".env", repo_root / ".env"):
        if not env_path.is_file():
            continue
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() in _TOKEN_KEYS:
                os.environ["HF_TOKEN"] = value.strip().strip('"').strip("'")
                return


def prepare_crop(crop_pil: Image.Image, region: str | None = None) -> Image.Image:
    """Normalize a region crop before the model's square resize."""
    im = crop_pil.convert("RGB")
    width, height = im.size
    aspect = max(width, height) / max(1, min(width, height))
    if aspect >= TALL_ASPECT:
        return ImageOps.pad(
            im, (LETTERBOX_SIZE, LETTERBOX_SIZE), color=LETTERBOX_FILL, centering=(0.5, 0.5)
        )
    if region == "upper":
        return im.resize(UPPER_SIZE, Image.Resampling.LANCZOS)
    if max(width, height) > MAX_CROP_SIDE:
        scale = MAX_CROP_SIDE / max(width, height)
        im = im.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.LANCZOS,
        )
    return im


def iter_polo_ref_images(ref_dir: Path) -> list[Path]:
    """Sorted image paths in an official-polo reference folder."""
    if not ref_dir.is_dir():
        return []
    return sorted(
        p for p in ref_dir.iterdir() if p.is_file() and p.suffix.lower() in REF_IMAGE_SUFFIXES
    )


class FashionSigLIPScorer:
    """Lazy-loaded FashionSigLIP scorer with cached text embeddings."""

    def __init__(self, polo_refs: Path | None = None) -> None:
        import open_clip

        _load_hf_token()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(MODEL_ID)
        self.tokenizer = open_clip.get_tokenizer(MODEL_ID)
        self.model = self.model.to(self.device).eval()

        self._text_embeds: dict[str, torch.Tensor] = {}
        self._labels: dict[str, list[str]] = {}
        for region, labels in REGION_LABELS.items():
            self._cache_labels(region, labels)

        self._polo_ref_paths = iter_polo_ref_images(polo_refs or DEFAULT_POLO_REFS_DIR)
        self._polo_ref_embeds = self._encode_polo_refs(self._polo_ref_paths)

    def _cache_labels(self, region: str, labels: list[str]) -> None:
        tokens = self.tokenizer(labels).to(self.device)
        with torch.no_grad():
            embeds = self.model.encode_text(tokens, normalize=True)
        self._text_embeds[region] = embeds
        self._labels[region] = labels

    def _encode_image(self, crop_pil: Image.Image, region: str | None = None) -> torch.Tensor:
        image = self.preprocess(prepare_crop(crop_pil, region)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model.encode_image(image, normalize=True).squeeze(0)

    def _encode_polo_refs(self, paths: list[Path]) -> torch.Tensor | None:
        if not paths:
            return None
        embeds = []
        for path in paths:
            with Image.open(path) as im:
                embeds.append(self._encode_image(im.convert("RGB"), "upper"))
        return torch.stack(embeds)

    @property
    def polo_ref_count(self) -> int:
        return len(self._polo_ref_paths)

    def match_official_polo(self, crop_pil: Image.Image) -> float:
        """Max cosine similarity of an upper crop vs official-polo reference embeddings."""
        if self._polo_ref_embeds is None:
            return 0.0
        feat = self._encode_image(crop_pil, "upper")
        return float((feat @ self._polo_ref_embeds.T).max().item())

    def score_crop(self, crop_pil: Image.Image, labels: list[str]) -> RegionScore:
        """Score a crop against an arbitrary label list (encodes text each call)."""
        tokens = self.tokenizer(labels).to(self.device)
        image = self.preprocess(prepare_crop(crop_pil)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image, normalize=True)
            text_features = self.model.encode_text(tokens, normalize=True)
            sims = (100.0 * image_features @ text_features.T).squeeze(0)
            probs = F.softmax(sims, dim=-1)

        scores = {label: float(probs[i].item()) for i, label in enumerate(labels)}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_label, top_score = ranked[0]
        second_label, second_score = ranked[1] if len(ranked) > 1 else ("", 0.0)
        return RegionScore(
            topLabel=top_label,
            topScore=top_score,
            secondLabel=second_label,
            secondScore=second_score,
            scores=scores,
        )

    def score_region(self, crop_pil: Image.Image, region: str) -> RegionScore:
        """Score a crop using cached embeddings for a named region."""
        labels = self._labels[region]
        text_features = self._text_embeds[region]
        image_features = self._encode_image(crop_pil, region).unsqueeze(0)
        sims = (100.0 * image_features @ text_features.T).squeeze(0)
        probs = F.softmax(sims, dim=-1)

        scores = {label: float(probs[i].item()) for i, label in enumerate(labels)}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_label, top_score = ranked[0]
        second_label, second_score = ranked[1] if len(ranked) > 1 else ("", 0.0)
        return RegionScore(
            topLabel=top_label,
            topScore=top_score,
            secondLabel=second_label,
            secondScore=second_score,
            scores=scores,
        )
