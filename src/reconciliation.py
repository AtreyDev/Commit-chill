"""Read-vs-Trust reconciliation and experimental Indic-script heuristics."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def indic_script_stats(text: str) -> dict[str, Any]:
    """Return script and spacing statistics for an experimental SDGI signal."""
    indic_scripts = ("DEVANAGARI", "BENGALI", "TAMIL")
    indic = [
        char
        for char in text
        if any(script in unicodedata.name(char, "") for script in indic_scripts)
    ]
    combining = sum(unicodedata.combining(char) for char in indic)
    repeated_spaces = len(re.findall(r" {2,}", text))
    words = [word for word in re.split(r"\s+", text.strip()) if word]
    script_ratio = len(indic) / max(1, len(text))
    spacing_ratio = repeated_spaces / max(1, len(words))
    return {
        "indic_characters": len(indic),
        "script_ratio": round(script_ratio, 4),
        "combining_marks": combining,
        "repeated_space_ratio": round(spacing_ratio, 4),
    }


def detect_sdgi_anomaly(text: str) -> dict[str, Any]:
    """Flag unusual Indic spacing/glyph statistics; experimental, not proof of fraud."""
    stats = indic_script_stats(text)
    reasons: list[str] = []
    if stats["indic_characters"] >= 8 and stats["repeated_space_ratio"] > 0.20:
        reasons.append("abnormal spacing between Indic-script tokens")
    if stats["indic_characters"] >= 8 and stats["combining_marks"] == 0:
        reasons.append("Indic text has no combining marks; review glyph rendering")
    return {
        "detected": bool(reasons),
        "experimental": True,
        "reasons": reasons,
        "stats": stats,
    }


def measure_indic_regions(
    image_path: str | Path, ocr_words: list[dict[str, Any]]
) -> dict[str, Any]:
    """Measure stroke and baseline consistency inside Indic OCR boxes.

    This is a measurable forensic feature, not a trained classifier. It is
    useful for experiments and must be calibrated against labeled data before
    being used as a decision threshold.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    indic_scripts = ("DEVANAGARI", "BENGALI", "TAMIL")
    stroke_widths: list[float] = []
    baselines: list[float] = []
    diacritic_positions: list[float] = []
    regions = 0
    for word in ocr_words:
        text = str(word.get("text", ""))
        if not any(
            any(script in unicodedata.name(char, "") for script in indic_scripts)
            for char in text
        ):
            continue
        points = word.get("bbox", [])
        if len(points) < 4:
            continue
        xs = [int(point[0]) for point in points]
        ys = [int(point[1]) for point in points]
        left, right = max(0, min(xs)), min(image.shape[1], max(xs))
        top, bottom = max(0, min(ys)), min(image.shape[0], max(ys))
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            continue
        _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
        values = distance[distance > 0]
        if values.size:
            stroke_widths.append(float(np.median(values) * 2))
        baselines.append(float(bottom))
        ink_rows = np.where(binary.any(axis=1))[0]
        if ink_rows.size:
            diacritic_positions.append(float(ink_rows[0] / max(1, crop.shape[0])))
        regions += 1
    return {
        "regions": regions,
        "stroke_width_variance": round(float(np.var(stroke_widths)), 4) if stroke_widths else 0.0,
        "baseline_variance": round(float(np.var(baselines)), 4) if baselines else 0.0,
        "diacritic_alignment_variance": (
            round(float(np.var(diacritic_positions)), 4)
            if diacritic_positions
            else 0.0
        ),
        "measured": bool(regions),
        "calibration_required": True,
    }


def reconcile_read_trust(
    *,
    ocr_confidence: float,
    ela_score: float,
    copy_move_score: float,
    wavelet_score: float = 0.0,
    sdgi_anomaly: bool = False,
) -> dict[str, Any]:
    """Compare readability confidence with independent visual trust evidence."""
    read_score = max(0.0, min(1.0, ocr_confidence))
    trust_penalty = ela_score * 8 + copy_move_score * 0.7 + wavelet_score * 0.3
    trust_score = max(0.0, min(1.0, 1.0 - trust_penalty))
    disagreement = read_score >= 0.75 and trust_score < 0.45
    if sdgi_anomaly:
        trust_score = max(0.0, trust_score - 0.15)
        disagreement = disagreement or read_score >= 0.60
    return {
        "read_score": round(read_score, 4),
        "trust_score": round(trust_score, 4),
        "disagreement": disagreement,
        "message": (
            "Text is readable, but visual evidence does not support it."
            if disagreement
            else "Read and trust signals are broadly aligned."
        ),
    }
