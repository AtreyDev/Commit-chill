"""Cross-case perceptual-hash lookup for possible duplicate submissions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def hamming_distance(left: str, right: str) -> int:
    """Return bit distance between two hexadecimal perceptual hashes."""
    if len(left) != len(right):
        raise ValueError("Perceptual hashes must have equal length")
    return (int(left, 16) ^ int(right, 16)).bit_count()


def find_similar_documents(
    audit_path: str | Path, perceptual_hash: str, *, max_distance: int = 6
) -> list[dict[str, Any]]:
    """Find prior audit records with near-identical perceptual hashes."""
    path = Path(audit_path)
    if not path.exists():
        return []
    matches = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        previous = record.get("document", {}).get("perceptual_hash")
        if previous and hamming_distance(previous, perceptual_hash) <= max_distance:
            matches.append({
                "document_name": record.get("document_name", ""),
                "distance": hamming_distance(previous, perceptual_hash),
                "record_hash": record.get("record_hash", ""),
            })
    return matches