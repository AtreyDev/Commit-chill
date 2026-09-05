from src.analysis.mrz import check_digit, parse_mrz


def test_known_td3_mrz_validates_and_extracts_identity_fields():
    text = (
        "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\n"
        "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
    )

    result = parse_mrz(text)

    assert result["format"] == "TD3"
    assert result["valid"] is True
    assert result["fields"]["document_number"] == "L898902C3"
    assert result["fields"]["surname"] == "ERIKSSON"
    assert result["fields"]["given_names"] == "ANNA MARIA"


def test_mrz_rejects_bad_check_digit_and_handles_missing_text():
    assert check_digit("L898902C3", "6") is True
    invalid = parse_mrz("P<UTOERIKSSON<<ANNA\nL898902C36UTO7408122F1204159ZE184226B<<<<<11")
    empty = parse_mrz("ordinary OCR text")

    assert invalid["valid"] is False
    assert empty["format"] == "unknown"
    assert empty["fields"] == {}