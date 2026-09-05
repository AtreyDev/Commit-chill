"""Tests for the offline screening and audit services."""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.audit import append_audit_record, verify_audit_chain
from src.rationale import build_rationale
from src.screening import (
    calculate_risk,
    compare_identity_fields,
    extract_identity_fields,
    extract_metadata,
)


def test_identity_fields_and_discrepancies():
    first = "Name TEST USER PAN ABCDE1234F DOB 01/02/1990 Aadhaar 1234 5678 9012"
    second = "PAN ABCDE9999F DOB 01/02/1990"
    assert extract_identity_fields(first)["pan"] == "ABCDE1234F"
    discrepancies = compare_identity_fields([first, second])
    assert discrepancies == [{"field": "pan", "values": ["ABCDE1234F", "ABCDE9999F"]}]


def test_risk_is_bounded_and_explainable():
    result = calculate_risk(
        ela_score_value=0.2,
        copy_move_score=0.8,
        ocr_confidence=0.4,
        metadata_anomalies=2,
        field_discrepancies=1,
    )
    assert 0 <= result["score"] <= 1
    assert set(result["components"]) == {
        "visual_tamper", "copy_move", "ocr_uncertainty", "metadata",
        "field_discrepancy", "indic_script_anomaly",
    }
    assert result["verdict"] == "Forged"


def test_metadata_is_json_serialisable(tmp_path):
    image_path = tmp_path / "document.png"
    Image.fromarray(np.zeros((10, 20, 3), dtype=np.uint8)).save(image_path)
    metadata = extract_metadata(image_path)
    assert metadata["width"] == 20
    assert metadata["height"] == 10
    assert metadata["anomalies"] == []


def test_audit_chain_detects_tampering(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    append_audit_record({"risk": {"score": 0.2}}, document_name="one.png", audit_path=audit_path)
    append_audit_record({"risk": {"score": 0.7}}, document_name="two.png", audit_path=audit_path)
    assert verify_audit_chain(audit_path)["valid"]
    audit_path.write_text(
        audit_path.read_text(encoding="utf-8").replace("two.png", "changed.png"),
        encoding="utf-8",
    )
    assert not verify_audit_chain(audit_path)["valid"]


def test_rationale_explains_evidence_and_mrz_status():
    rationale = build_rationale({
        "risk": {"verdict": "Suspicious", "components": {"copy_move": 0.8}},
        "mrz": {"format": "TD3", "valid": False},
    })
    assert "copy-move" in rationale
    assert "check digits failed" in rationale
