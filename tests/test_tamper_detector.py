from PIL import Image

from src.models.tamper_detector import HeuristicTamperDetector, TamperPrediction


def test_tamper_prediction_contract_is_serializable():
    prediction = TamperPrediction("Authentic", 0.02, [], {"copy_move": 0.0}, "test")

    assert prediction.as_dict() == {
        "verdict": "Authentic",
        "score": 0.02,
        "regions": [],
        "signals": {"copy_move": 0.0},
        "model": "test",
    }


def test_heuristic_detector_returns_bounded_prediction(tmp_path):
    image_path = tmp_path / "document.png"
    Image.new("RGB", (64, 64), color="white").save(image_path)

    prediction = HeuristicTamperDetector().predict(image_path)

    assert 0 <= prediction.score <= 1
    assert prediction.verdict in {"Authentic", "Suspicious", "Forged"}
    assert set(prediction.signals) == {"compression_anomaly", "copy_move"}