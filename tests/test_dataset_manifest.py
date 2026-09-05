from PIL import Image

from src.datasets.manifest import build_manifest, write_manifest


def test_manifest_records_verified_images_and_groups_by_directory(tmp_path):
    image_dir = tmp_path / "person-01"
    image_dir.mkdir()
    Image.new("RGB", (12, 8), color="white").save(image_dir / "sample.png")
    (image_dir / "broken.jpg").write_bytes(b"not an image")

    records = build_manifest(tmp_path, dataset="fgnet", split="train", label="face")

    assert len(records) == 1
    assert records[0].group_id == "person-01"
    assert records[0].width == 12
    assert len(records[0].sha256) == 64


def test_manifest_can_override_group_and_write_csv(tmp_path):
    Image.new("L", (4, 4)).save(tmp_path / "frame-01.png")

    records = build_manifest(
        tmp_path,
        dataset="midv2020",
        split="validation",
        label="passport",
        group_id="video-07",
    )
    output = write_manifest(records, tmp_path / "manifests" / "midv2020.csv")

    assert output.exists()
    assert "video-07" in output.read_text(encoding="utf-8")