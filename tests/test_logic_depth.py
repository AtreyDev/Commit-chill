from datetime import date

from src.logic import run_logic_checks, validate_pan
from src.screening import calculate_risk


def test_pan_holder_type_and_bilingual_name_checks():
    assert validate_pan("ABCAF1234F") is True
    assert validate_pan("ABCX1234F") is False
    checks = run_logic_checks(
        {"pan": "ABCAF1234F", "name": "RAVI KUMAR", "native_name": "RAVI KUMAR"},
        today=date(2026, 9, 5),
    )
    assert any(check["name"] == "Bilingual name consistency" for check in checks)


def test_risk_exposes_calibrated_bucket():
    result = calculate_risk(ela_score_value=0.01, copy_move_score=0.01)

    assert result["confidence_bucket"] == "High-confidence Authentic"
    assert "held-out" in result["calibration"]