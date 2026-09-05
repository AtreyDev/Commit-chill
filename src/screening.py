"""Offline document screening orchestration and transparent risk scoring."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from PIL import Image


def extract_metadata(image_path: str | Path) -> dict[str, Any]:
    """Return safe, JSON-serialisable metadata and editing indicators."""
    path = Path(image_path)
    with Image.open(path) as image:
        exif = image.getexif()
        tags = {str(key): str(value) for key, value in exif.items()}
        software = tags.get("305", "")
        anomalies: list[str] = []
        if software:
            lowered = software.lower()
            editing_tools = ("photoshop", "gimp", "paint", "canva", "affinity")
            if any(tool in lowered for tool in editing_tools):
                anomalies.append(f"editing software metadata: {software}")
        if image.format not in {"JPEG", "PNG", "TIFF", "BMP"}:
            anomalies.append(f"unexpected image format: {image.format}")
        return {
            "format": image.format,
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "has_exif": bool(tags),
            "exif": tags,
            "anomalies": anomalies,
        }


def extract_identity_fields(text: str) -> dict[str, str]:
    """Extract common Indian identity-document fields from OCR text.

    This is deliberately conservative: missing fields are omitted rather than
    guessed, so downstream discrepancies remain explainable to an operator.
    """
    compact = re.sub(r"\s+", " ", text.upper()).strip()
    fields: dict[str, str] = {}
    aadhaar = re.search(r"(?<!\d)(\d{4}\s?\d{4}\s?\d{4})(?!\d)", compact)
    pan = re.search(r"\b([A-Z]{5}\d{4}[A-Z])\b", compact)
    dob = re.search(r"\b(\d{2}[/-]\d{2}[/-]\d{4})\b", compact)
    dates = re.findall(r"\b(\d{2}[/-]\d{2}[/-]\d{4})\b", compact)
    name = re.search(r"(?:NAME|नाम)\s*[:\-]?\s*([A-Z][A-Z .]{2,})", compact)
    if aadhaar:
        fields["aadhaar"] = re.sub(r"\D", "", aadhaar.group(1))
    if pan:
        fields["pan"] = pan.group(1)
    if dob:
        fields["date_of_birth"] = dob.group(1).replace("/", "-")
    if name:
        fields["name"] = name.group(1).strip()
    if len(dates) >= 2:
        fields["issue_date"] = dates[0].replace("/", "-")
        fields["expiry_date"] = dates[1].replace("/", "-")
    return fields


def compare_identity_fields(texts: list[str]) -> list[dict[str, Any]]:
    """Find fields present with conflicting values across OCR text blocks."""
    observed: dict[str, set[str]] = {}
    for text in texts:
        for field, value in extract_identity_fields(text).items():
            observed.setdefault(field, set()).add(value)
    discrepancies = []
    for field, values in observed.items():
        if len(values) > 1:
            discrepancies.append({"field": field, "values": sorted(values)})
    return discrepancies


def calculate_risk(
    *,
    ela_score_value: float,
    copy_move_score: float,
    ocr_confidence: float = 1.0,
    metadata_anomalies: int = 0,
    field_discrepancies: int = 0,
    sdgi_anomalies: int = 0,
) -> dict[str, Any]:
    """Combine explainable signals into a score in [0, 1]."""
    components = {
        "visual_tamper": min(1.0, max(0.0, ela_score_value * 10.0)),
        "copy_move": min(1.0, max(0.0, copy_move_score)),
        "ocr_uncertainty": min(1.0, max(0.0, 1.0 - ocr_confidence)),
        "metadata": min(1.0, metadata_anomalies / 3.0),
        "field_discrepancy": min(1.0, field_discrepancies / 2.0),
        "indic_script_anomaly": min(1.0, sdgi_anomalies / 1.0),
    }
    weights = {
        "visual_tamper": 0.30,
        "copy_move": 0.30,
        "ocr_uncertainty": 0.15,
        "metadata": 0.10,
        "field_discrepancy": 0.15,
        "indic_script_anomaly": 0.05,
    }
    score = min(1.0, sum(components[name] * weights[name] for name in components))
    confidence_bucket = (
        "High-confidence Authentic"
        if score < 0.10
        else "Borderline - route to human review"
        if score < 0.55
        else "High-confidence Forged"
    )
    verdict = "Authentic" if score < 0.10 else "Suspicious" if score < 0.55 else "Forged"
    return {
        "score": round(score, 4),
        "verdict": verdict,
        "components": {name: round(value, 4) for name, value in components.items()},
        "weights": weights,
        "confidence_bucket": confidence_bucket,
        "calibration": "thresholds_v1; validate on a held-out representative set",
    }


def screen_document(
    image_path: str | Path,
    *,
    ocr_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the available offline signals and return a screening report."""
    from src.analysis.ela import ela_score, generate_ela
    from src.copy_move.detector import detect_copy_move
    from src.logic import run_logic_checks
    from src.reconciliation import (
        detect_sdgi_anomaly,
        measure_indic_regions,
        reconcile_read_trust,
    )

    path = Path(image_path)
    ela_image = generate_ela(path)
    copy_move = detect_copy_move(path)
    metadata = extract_metadata(path)
    ocr_confidence = float(ocr_result.get("avg_confidence", 1.0)) if ocr_result else 1.0
    fields = extract_identity_fields(ocr_result.get("full_text", "")) if ocr_result else {}
    logic_checks = run_logic_checks(fields)
    sdgi = detect_sdgi_anomaly(ocr_result.get("full_text", "") if ocr_result else "")
    if ocr_result and ocr_result.get("words"):
        sdgi["region_metrics"] = measure_indic_regions(path, ocr_result["words"])
    read_trust = reconcile_read_trust(
        ocr_confidence=ocr_confidence,
        ela_score=ela_score(ela_image),
        copy_move_score=float(copy_move["score"]),
        sdgi_anomaly=sdgi["detected"],
    )
    evidence_regions = []
    mask = copy_move.get("mask")
    if mask is not None:
        import cv2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        evidence_regions = [
            {"x": x, "y": y, "width": width, "height": height}
            for contour in contours
            if (x_y_w_h := cv2.boundingRect(contour)) and x_y_w_h[2] * x_y_w_h[3] >= 25
            for x, y, width, height in [x_y_w_h]
        ]
    risk = calculate_risk(
        ela_score_value=ela_score(ela_image),
        copy_move_score=float(copy_move["score"]),
        ocr_confidence=ocr_confidence,
        metadata_anomalies=len(metadata["anomalies"]),
        field_discrepancies=sum(not check["passed"] for check in logic_checks),
        sdgi_anomalies=int(sdgi["detected"]),
    )
    return {
        "risk": risk,
        "ela_score": round(ela_score(ela_image), 4),
        "copy_move": {
            key: value
            for key, value in copy_move.items()
            if key not in {"mask", "heatmap"}
        },
        "metadata": metadata,
        "fields": fields,
        "ocr": ocr_result,
        "logic_checks": logic_checks,
        "sdgi": sdgi,
        "read_trust": read_trust,
        "evidence_regions": evidence_regions,
    }
