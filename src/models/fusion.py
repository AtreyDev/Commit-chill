"""Explainable weighted fusion of independent document signals."""

from __future__ import annotations

from typing import Any

DEFAULT_WEIGHTS = {
    "tamper_probability": 0.25,
    "copy_move_score": 0.20,
    "mrz_validation_failure": 0.20,
    "ocr_inconsistency": 0.15,
    "image_quality_anomaly": 0.10,
    "face_verification_failure": 0.10,
}


def fuse_risk_signals(
    signals: dict[str, float], *, weights: dict[str, float] | None = None
) -> dict[str, Any]:
    """Return bounded risk with contributions and missing-signal details.

    Missing optional signals are excluded and the remaining weights are
    renormalized, so an unrequested face comparison does not add risk.
    """
    configured = dict(weights or DEFAULT_WEIGHTS)
    if any(value < 0 for value in configured.values()) or not configured:
        raise ValueError("weights must be non-negative and non-empty")
    if sum(configured.values()) <= 0:
        raise ValueError("weights must have a positive total")

    available = {
        name: min(1.0, max(0.0, float(signals[name])))
        for name in configured
        if name in signals and signals[name] is not None
    }
    available_weight = sum(configured[name] for name in available)
    if not available:
        return {
            "score": 0.0,
            "verdict": "Unknown",
            "signals": {},
            "contributions": {},
            "weights": configured,
            "missing_signals": sorted(configured),
        }

    contributions = {
        name: round(value * configured[name] / available_weight, 4)
        for name, value in available.items()
    }
    score = round(min(1.0, sum(contributions.values())), 4)
    verdict = "Authentic" if score < 0.10 else "Suspicious" if score < 0.55 else "Forged"
    return {
        "score": score,
        "verdict": verdict,
        "signals": {name: round(value, 4) for name, value in available.items()},
        "contributions": contributions,
        "weights": configured,
        "missing_signals": sorted(set(configured) - set(available)),
    }