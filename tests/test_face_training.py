from pathlib import Path

import pytest

from src.training.train_face_age import parse_fgnet_name, split_fgnet


def test_parse_fgnet_filename():
    assert parse_fgnet_name(Path("001A43b.JPG")) == ("001", 43)


def test_split_fgnet_requires_dataset(tmp_path):
    with pytest.raises(FileNotFoundError):
        split_fgnet(tmp_path / "missing")