"""MIDV-2020 manifest adapter.

The dataset is not bundled. Directory names are retained as group identifiers
so video frames or document instances can be split together.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from src.datasets.manifest import ManifestRecord, build_manifest


def build_midv2020_manifest(root: str | Path, *, split: str) -> list[ManifestRecord]:
    """Build records for MIDV images using the nearest parent as group ID."""
    records = build_manifest(root, dataset="midv2020", split=split, label="document")
    return [replace(record, group_id=Path(record.path).parent.name) for record in records]