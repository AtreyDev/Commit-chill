"""Tamper-detection contracts with an offline baseline implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class TamperPrediction:
    """Stable output contract for heuristic or trained tamper models."""

    verdict: str
    score: float
    regions: list[dict[str, int]]
    signals: dict[str, float]
    model: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class TamperDetector(Protocol):
    """Interface implemented by heuristic and checkpoint-backed detectors."""

    def predict(self, image_path: str | Path) -> TamperPrediction:
        ...


def _verdict(score: float) -> str:
    return "Authentic" if score < 0.10 else "Suspicious" if score < 0.55 else "Forged"


class HeuristicTamperDetector:
    """Combine existing ELA and copy-move evidence without requiring weights."""

    model_name = "ela+copy_move_heuristic"

    def predict(self, image_path: str | Path) -> TamperPrediction:
        from src.analysis.ela import ela_score, generate_ela
        from src.copy_move.detector import detect_copy_move

        path = Path(image_path)
        ela_value = float(ela_score(generate_ela(path)))
        copy_move = detect_copy_move(path)
        copy_move_value = float(copy_move.get("score", 0.0))
        score = min(1.0, (min(1.0, ela_value * 10.0) + copy_move_value) / 2.0)
        return TamperPrediction(
            verdict=_verdict(score),
            score=round(score, 4),
            regions=[],
            signals={
                "compression_anomaly": round(min(1.0, ela_value * 10.0), 4),
                "copy_move": round(copy_move_value, 4),
            },
            model=self.model_name,
        )