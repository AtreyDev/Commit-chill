"""FG-NET manifest adapter for subject-disjoint face experiments."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from src.datasets.manifest import ManifestRecord, build_manifest


def build_fgnet_manifest(root: str | Path, *, split: str) -> list[ManifestRecord]:
    """Build FG-NET records grouped by the immediate subject directory."""
    records = build_manifest(root, dataset="fgnet", split=split, label="face")
    return [replace(record, group_id=Path(record.path).parent.name) for record in records]