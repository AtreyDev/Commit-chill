import pytest

from src.models.face_verification import compare_embeddings


def test_face_comparison_requires_consent():
    result = compare_embeddings([1.0, 0.0], [1.0, 0.0])

    assert result.consent_required is True
    assert result.compared is False
    assert result.similarity is None


def test_face_comparison_returns_similarity_and_age_warning():
    result = compare_embeddings(
        [1.0, 0.0], [0.8, 0.6], consent_given=True, threshold=0.7, reference_age=20, query_age=45
    )

    assert result.compared is True
    assert result.matched is True
    assert result.age_gap_warning is True
    assert 0 <= result.similarity <= 1


def test_face_comparison_rejects_mismatched_embeddings():
    with pytest.raises(ValueError, match="same dimensions"):
        compare_embeddings([1.0], [1.0, 0.0], consent_given=True)