from src.models.fusion import fuse_risk_signals


def test_fusion_reports_contributions_and_renormalizes_missing_optional_signal():
    result = fuse_risk_signals(
        {
            "tamper_probability": 0.8,
            "copy_move_score": 0.4,
            "mrz_validation_failure": 0.0,
        }
    )

    assert result["verdict"] == "Suspicious"
    assert result["missing_signals"] == [
        "face_verification_failure",
        "image_quality_anomaly",
        "ocr_inconsistency",
    ]
    assert round(sum(result["contributions"].values()), 4) == result["score"]


def test_fusion_clamps_untrusted_signal_values():
    result = fuse_risk_signals({"tamper_probability": 4.0})

    assert result["score"] == 1.0
    assert result["signals"]["tamper_probability"] == 1.0
    assert result["verdict"] == "Forged"