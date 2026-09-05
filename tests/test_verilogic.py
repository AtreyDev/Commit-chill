"""Focused tests for VeriLogic's deterministic report layer."""

from __future__ import annotations

import json
from datetime import date

from PIL import Image

from src.export import build_evidence_report, export_json, export_pdf
from src.logic import compare_field_sets, run_logic_checks, validate_aadhaar
from src.reconciliation import detect_sdgi_anomaly, measure_indic_regions, reconcile_read_trust


def test_aadhaar_verhoeff_examples():
    assert validate_aadhaar("2363  ಶ") is False
    assert validate_aadhaar("2363 1076 2114") is True


def test_logic_engine_checks_dates_and_pan():
    checks = run_logic_checks(
        {
            "pan": "ABCAF1234F",
            "date_of_birth": "01-01-2000",
            "issue_date": "01-01-2020",
            "expiry_date": "01-01-2030",
        },
        today=date(2026, 9, 5),
    )
    assert all(check["passed"] for check in checks)


def test_cross_document_comparison_allows_name_ocr_variance():
    assert compare_field_sets([{"name": "RAVI KUMAR"}, {"name": "RAVI  KUMAR"}]) == []
    assert compare_field_sets([{"pan": "ABCDE1234F"}, {"pan": "ABCDE9999F"}])[0]["field"] == "pan"


def test_read_trust_flags_readable_but_unsupported_text():
    result = reconcile_read_trust(ocr_confidence=0.95, ela_score=0.12, copy_move_score=0.2)
    assert result["disagreement"] is True


def test_sdgi_is_explicitly_experimental():
    result = detect_sdgi_anomaly("नाम  कुमार  सिंह")
    assert result["experimental"] is True
    assert "stats" in result


def test_sdgi_region_metrics_measure_ocr_box(tmp_path):
    image_path = tmp_path / "indic.png"
    Image.new("L", (40, 20), "white").save(image_path)
    metrics = measure_indic_regions(
        image_path,
        [{"text": "नाम", "bbox": [[2, 2], [20, 2], [20, 15], [2, 15]]}],
    )
    assert metrics["measured"] is True
    assert metrics["regions"] == 1
    assert metrics["calibration_required"] is True
    assert "diacritic_alignment_variance" in metrics


def test_exports_include_hash_and_timestamp(tmp_path):
    document = tmp_path / "doc.png"
    Image.new("RGB", (8, 8), "white").save(document)
    report = {"risk": {"score": 0.2, "verdict": "Suspicious"}, "logic_checks": []}
    evidence = build_evidence_report(report, document_path=document)
    assert len(evidence["document"]["sha256"]) == 64
    json_path = export_json(report, tmp_path / "report.json", document_path=document)
    assert json.loads(json_path.read_text(encoding="utf-8"))["schema_version"] == "1.0"
    assert export_pdf(report, tmp_path / "report.pdf", document_path=document).exists()
