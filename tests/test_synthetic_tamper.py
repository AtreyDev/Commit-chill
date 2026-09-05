from PIL import Image

from src.datasets.synthetic_tamper import generate_tamper


def test_synthetic_tamper_generator_preserves_source_and_records_label(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (80, 60), color="white").save(source)

    result = generate_tamper(source, tmp_path / "generated", manipulation="stamp_overlay")

    assert result.output_path.exists()
    assert result.manipulation == "stamp_overlay"
    assert source.read_bytes() != result.output_path.read_bytes()