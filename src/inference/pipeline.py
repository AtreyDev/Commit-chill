"""Unified, serializable inference report construction."""

from __future__ import annotations

from typing import Any

from src.models.fusion import fuse_risk_signals


def build_inference_report(
    *,
    document_type: str,
    ocr: dict[str, Any] | None = None,
    mrz: dict[str, Any] | None = None,
    tamper: dict[str, Any] | None = None,
    copy_move_score: float = 0.0,
    image_quality_anomaly: float | None = None,
    face_verification: dict[str, Any] | None = None,
    model_versions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build the stable report contract used by UI and future API layers."""
    ocr_result = ocr or {}
    mrz_result = mrz or {}
    tamper_result = tamper or {}
    face_result = face_verification or {}
    mrz_failure = None if not mrz_result else float(not mrz_result.get("valid", False))
    ocr_inconsistency = None
    if ocr_result:
        ocr_inconsistency = round(1.0 - float(ocr_result.get("avg_confidence", 0.0)), 4)
    face_failure = None
    if face_result.get("compared"):
        face_failure = float(not face_result.get("matched", False))
    signals = {
        "tamper_probability": tamper_result.get("score"),
        "copy_move_score": copy_move_score,
        "mrz_validation_failure": mrz_failure,
        "ocr_inconsistency": ocr_inconsistency,
        "image_quality_anomaly": image_quality_anomaly,
        "face_verification_failure": face_failure,
    }
    return {
        "document_type": document_type,
        "ocr": ocr_result,
        "mrz": mrz_result,
        "tamper": tamper_result,
        "copy_move": {"score": round(max(0.0, min(1.0, copy_move_score)), 4)},
        "face_verification": face_result,
        "risk": fuse_risk_signals(signals),
        "model_versions": model_versions or {},
    }