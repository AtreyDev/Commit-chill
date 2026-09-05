from PIL import Image

from src.audit import append_audit_record, perceptual_hash
from src.audit_similarity import find_similar_documents, hamming_distance


def test_perceptual_hash_finds_near_duplicate_submission(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (32, 32), "white").save(first)
    Image.new("RGB", (32, 32), "white").save(second)
    audit_path = tmp_path / "audit.jsonl"
    append_audit_record({}, document_name="first.png", document_path=first, audit_path=audit_path)

    matches = find_similar_documents(audit_path, perceptual_hash(second))

    assert hamming_distance(perceptual_hash(first), perceptual_hash(second)) == 0
    assert matches[0]["document_name"] == "first.png"