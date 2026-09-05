"""Create reproducible synthetic tamper examples for baseline experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


@dataclass(frozen=True)
class SyntheticTamper:
    """Generated image metadata and the manipulation label applied."""

    output_path: Path
    manipulation: str
    source_path: Path


def _box(width: int, height: int, rng: random.Random) -> tuple[int, int, int, int]:
    left = rng.randrange(max(1, width // 5))
    top = rng.randrange(max(1, height // 5))
    box_width = max(8, width // 4)
    box_height = max(8, height // 5)
    return left, top, min(width, left + box_width), min(height, top + box_height)


def generate_tamper(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    manipulation: str,
    seed: int = 7,
) -> SyntheticTamper:
    """Generate one labeled tamper without modifying the source image.

    This is a training/data-augmentation utility only. It must not be used as
    evidence that the generated manipulation represents real-world fraud.
    """
    source = Path(source_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    image = Image.open(source).convert("RGB")
    width, height = image.size
    left, top, right, bottom = _box(width, height, rng)

    if manipulation == "photo_replacement":
        patch = image.crop((left, top, right, bottom)).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image.paste(patch, (left, top))
    elif manipulation == "text_patch":
        draw = ImageDraw.Draw(image)
        draw.rectangle((left, top, right, bottom), fill=(235, 235, 235))
        draw.text((left + 4, top + 4), "ALTERED", fill=(20, 20, 20))
    elif manipulation == "stamp_overlay":
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.ellipse((left, top, right, bottom), outline=(170, 20, 20, 180), width=4)
        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    else:
        raise ValueError("manipulation must be photo_replacement, text_patch, or stamp_overlay")

    output = destination / f"{source.stem}_{manipulation}_{seed}.png"
    image.filter(ImageFilter.SHARPEN).save(output)
    return SyntheticTamper(output, manipulation, source)