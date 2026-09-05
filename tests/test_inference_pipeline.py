from src.inference.pipeline import build_inference_report


def test_inference_report_contains_all_pipeline_sections():
    report = build_inference_report(
        document_type="passport",
        ocr={"full_text": "text", "avg_confidence": 0.9},
        mrz={"valid": False, "fields": {}},
        tamper={"verdict": "Suspicious", "score": 0.6},
        copy_move_score=0.2,
        model_versions={"tamper": "heuristic-v1"},
    )

    assert set(report) == {
        "document_type",
        "ocr",
        "mrz",
        "tamper",
        "copy_move",
        "face_verification",
        "risk",
        "model_versions",
    }
    assert report["risk"]["signals"]["mrz_validation_failure"] == 1.0
    assert "face_verification_failure" in report["risk"]["missing_signals"]


def test_unrequested_face_verification_does_not_add_risk():
    report = build_inference_report(document_type="id-card")

    assert report["face_verification"] == {}
    assert "face_verification_failure" in report["risk"]["missing_signals"]