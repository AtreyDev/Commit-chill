"""Plain-language explanations for document screening decisions."""

from __future__ import annotations

from typing import Any


def build_rationale(report: dict[str, Any]) -> str:
    """Turn screening evidence into a concise, non-diagnostic explanation."""
    risk = report.get("risk", {})
    components = risk.get("components", {})
    verdict = risk.get("verdict", "Unknown")
    reasons: list[str] = []

    if components.get("visual_tamper", 0) >= 0.4:
        reasons.append("the image shows elevated compression or visual-tamper signals")
    if components.get("copy_move", 0) >= 0.4:
        reasons.append("copy-move analysis found suspicious repeated regions")
    if components.get("field_discrepancy", 0) > 0:
        reasons.append("identity fields contain conflicting values")
    if components.get("ocr_uncertainty", 0) >= 0.5:
        reasons.append("OCR confidence is low, so text evidence is uncertain")
    if components.get("metadata", 0) > 0:
        reasons.append("metadata contains editing or format anomalies")

    mrz = report.get("mrz", {})
    if mrz.get("format") not in {None, "", "unknown"}:
        if mrz.get("valid"):
            reasons.append("the MRZ format and check digits pass")
        else:
            reasons.append("the MRZ was found but one or more check digits failed")

    read_trust = report.get("read_trust", {})
    if read_trust.get("message"):
        reasons.append(str(read_trust["message"]).splitlines()[0])

    if not reasons:
        reasons.append("no strong anomaly was found in the available evidence")
    return f"{verdict}: " + "; ".join(reasons) + "."