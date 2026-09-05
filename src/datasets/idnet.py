"""IDNet manifest adapter for authentic/tampered document samples."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from src.datasets.manifest import ManifestRecord, build_manifest


def _label(path: str) -> str:
    parts = {part.lower() for part in Path(path).parts}
    if parts.intersection({"tampered", "fake", "forged", "manipulated"}):
        return "tampered"
    if parts.intersection({"authentic", "real", "genuine", "original"}):
        return "authentic"
    return "unknown"


def build_idnet_manifest(root: str | Path, *, split: str) -> list[ManifestRecord]:
    """Build IDNet records; unknown folder conventions stay explicitly unknown."""
    records = build_manifest(root, dataset="idnet", split=split, label="unknown")
    return [
        replace(record, label=_label(record.path), group_id=Path(record.path).stem)
        for record in records
    ]