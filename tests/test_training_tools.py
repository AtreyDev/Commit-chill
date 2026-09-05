import pytest

from src.training.prepare_manifests import prepare


def test_prepare_manifests_reports_missing_dataset(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset root does not exist"):
        prepare("idnet", tmp_path / "missing", tmp_path / "idnet.csv", "train")