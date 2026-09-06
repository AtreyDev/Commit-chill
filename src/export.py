"""JSON and PDF evidence report exports."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> str:
    return str(value)


def build_evidence_report(report: dict[str, Any], *, document_path: str | Path) -> dict[str, Any]:
    """Add a document hash and UTC timestamp to a serialisable report."""
    path = Path(document_path)
    document_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "document": {"name": path.name, "sha256": document_hash},
        "report": report,
    }


def export_json(
    report: dict[str, Any], output_path: str | Path, *, document_path: str | Path
) -> Path:
    """Write the evidence report as formatted JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    evidence = build_evidence_report(report, document_path=document_path)
    path.write_text(json.dumps(evidence, indent=2, default=_json_default), encoding="utf-8")
    return path


def export_pdf(
    report: dict[str, Any], output_path: str | Path, *, document_path: str | Path
) -> Path:
    """Write a lightweight text evidence report as a PDF."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    evidence = build_evidence_report(report, document_path=document_path)
    risk = report.get("risk", {})
    lines = [
        "VeriLogic Evidence Report",
        f"Document: {evidence['document']['name']}",
        f"SHA-256: {evidence['document']['sha256']}",
        f"Generated: {evidence['generated_at']}",
        "",
        f"Verdict: {risk.get('verdict', 'Unknown')}",
        f"Risk score: {risk.get('score', 0):.1%}",
        "",
        "Checks and evidence:",
    ]
    for check in report.get("logic_checks", []):
        marker = "PASS" if check.get("passed") else "FAIL"
        lines.append(f"[{marker}] {check.get('name')}: {check.get('detail', '')}")
    lines.extend(
        f"[READ/TRUST] {item}"
        for item in report.get("read_trust", {}).get("message", "").splitlines()
    )
    lines.extend(
        f"[SDGI EXPERIMENTAL] {item}"
        for item in report.get("sdgi", {}).get("reasons", [])
    )
    with PdfPages(path) as pdf:
        figure = plt.figure(figsize=(8.5, 11))
        figure.text(0.08, 0.94, "\n".join(lines), va="top", family="monospace", fontsize=10)
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)
    return path
