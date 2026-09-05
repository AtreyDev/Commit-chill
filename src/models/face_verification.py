"""Optional, consent-gated face embedding comparison contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FaceVerificationResult:
    """Explainable comparison result, not an identity determination."""

    consent_required: bool
    compared: bool
    similarity: float | None
    matched: bool | None
    age_gap_warning: bool
    disclaimer: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_embeddings(
    reference: list[float] | np.ndarray,
    query: list[float] | np.ndarray,
    *,
    consent_given: bool = False,
    threshold: float = 0.5,
    reference_age: int | None = None,
    query_age: int | None = None,
) -> FaceVerificationResult:
    """Compare two embeddings only after explicit consent is provided."""
    disclaimer = (
        "Face similarity is an assistive signal, not definitive identity proof. "
        "Do not use it as the sole basis for an identity or eligibility decision."
    )
    if not consent_given:
        return FaceVerificationResult(True, False, None, None, False, disclaimer)

    reference_vector = np.asarray(reference, dtype=np.float32).reshape(-1)
    query_vector = np.asarray(query, dtype=np.float32).reshape(-1)
    if reference_vector.size == 0 or reference_vector.shape != query_vector.shape:
        raise ValueError("Face embeddings must be non-empty and have the same dimensions")
    reference_norm = np.linalg.norm(reference_vector)
    query_norm = np.linalg.norm(query_vector)
    if reference_norm == 0 or query_norm == 0:
        raise ValueError("Face embeddings must not be zero vectors")

    similarity = float(np.dot(reference_vector, query_vector) / (reference_norm * query_norm))
    age_gap_warning = (
        reference_age is not None
        and query_age is not None
        and abs(reference_age - query_age) > 15
    )
    return FaceVerificationResult(
        False,
        True,
        round(similarity, 4),
        similarity >= threshold,
        age_gap_warning,
        disclaimer,
    )