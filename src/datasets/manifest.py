"""Build deterministic, leakage-aware image manifests for model training."""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


@dataclass(frozen=True)
class ManifestRecord:
    """Serializable metadata for one validated image sample."""

    path: str
    dataset: str
    split: str
    label: str
    group_id: str
    sha256: str
    width: int
    height: int
    mode: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_group_id(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    return relative.parent.as_posix() or path.stem


def build_manifest(
    root: str | Path,
    *,
    dataset: str,
    split: str,
    label: str,
    group_id: str | None = None,
    paths: Iterable[str | Path] | None = None,
) -> list[ManifestRecord]:
    """Validate images and return sorted records with stable hashes.

    ``group_id`` should identify a document, person, or video. Supplying it is
    recommended when a directory contains samples from multiple identities.
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root_path}")
    if not dataset.strip() or not split.strip() or not label.strip():
        raise ValueError("dataset, split, and label are required")

    candidates = (
        (Path(path) for path in paths)
        if paths is not None
        else root_path.rglob("*")
    )
    records: list[ManifestRecord] = []
    image_candidates = (path for path in candidates if path.is_file())
    for candidate in sorted(image_candidates, key=lambda item: str(item)):
        if candidate.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            with Image.open(candidate) as image:
                image.verify()
            with Image.open(candidate) as image:
                width, height = image.size
                mode = image.mode
        except (OSError, UnidentifiedImageError):
            continue
        records.append(
            ManifestRecord(
                path=str(candidate.resolve()),
                dataset=dataset.strip(),
                split=split.strip(),
                label=label.strip(),
                group_id=group_id or _default_group_id(candidate, root_path),
                sha256=_sha256(candidate),
                width=width,
                height=height,
                mode=mode,
            )
        )
    return records


def write_manifest(records: Iterable[ManifestRecord], output_path: str | Path) -> Path:
    """Write manifest records as a stable CSV file."""
    rows = list(records)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(ManifestRecord.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    return path