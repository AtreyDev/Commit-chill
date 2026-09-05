"""ICAO 9303 MRZ normalization, parsing, and check-digit validation."""

from __future__ import annotations

import re
from typing import Any

_WEIGHTS = (7, 3, 1)


def _value(character: str) -> int:
    if character == "<":
        return 0
    if character.isdigit():
        return int(character)
    return ord(character) - ord("A") + 10


def check_digit(value: str, expected: str) -> bool:
    """Validate one ICAO MRZ check digit."""
    total = sum(
        _value(char) * _WEIGHTS[index % 3] for index, char in enumerate(value)
    )
    return expected.isdigit() and total % 10 == int(expected)


def _clean_lines(text: str) -> list[str]:
    return [
        re.sub(r"[^A-Z0-9<]", "", line.upper())
        for line in text.splitlines()
        if len(re.sub(r"[^A-Z0-9<]", "", line.upper())) >= 30
    ]


def _name_parts(value: str) -> tuple[str, str]:
    surname, _, given_names = value.partition("<<")
    return surname.replace("<", " ").strip(), given_names.replace("<", " ").strip()


def _result(
    layout: str,
    lines: list[str],
    checks: dict[str, bool],
    fields: dict[str, str],
) -> dict[str, Any]:
    return {
        "format": layout,
        "lines": lines,
        "fields": fields,
        "checks": checks,
        "valid": bool(checks) and all(checks.values()),
        "disclaimer": (
            "MRZ parsing is an extraction and checksum aid, not proof of document authenticity."
        ),
    }


def _parse_td3(lines: list[str]) -> dict[str, Any]:
    first, second = (line.ljust(44, "<")[:44] for line in lines[:2])
    document_number = second[0:9]
    surname, given_names = _name_parts(first[5:44])
    checks = {
        "document_number": check_digit(second[0:9], second[9]),
        "date_of_birth": check_digit(second[13:19], second[19]),
        "expiry_date": check_digit(second[21:27], second[27]),
        "optional_data": check_digit(second[28:42], second[42]),
        "composite": check_digit(second[0:10] + second[13:20] + second[21:43], second[43]),
    }
    return _result(
        "TD3",
        [first, second],
        checks,
        {
            "document_code": first[0:2].replace("<", ""),
            "issuing_country": first[2:5].replace("<", ""),
            "document_number": document_number.replace("<", ""),
            "nationality": second[10:13].replace("<", ""),
            "date_of_birth": second[13:19],
            "sex": second[20].replace("<", ""),
            "expiry_date": second[21:27],
            "surname": surname,
            "given_names": given_names,
        },
    )


def _parse_td2(lines: list[str]) -> dict[str, Any]:
    first, second = (line.ljust(36, "<")[:36] for line in lines[:2])
    surname, given_names = _name_parts(first[5:36])
    checks = {
        "document_number": check_digit(second[0:9], second[9]),
        "date_of_birth": check_digit(second[13:19], second[19]),
        "expiry_date": check_digit(second[21:27], second[27]),
        "composite": check_digit(second[0:10] + second[13:20] + second[21:35], second[35]),
    }
    return _result(
        "TD2",
        [first, second],
        checks,
        {
            "document_code": first[0:2].replace("<", ""),
            "issuing_country": first[2:5].replace("<", ""),
            "document_number": second[0:9].replace("<", ""),
            "nationality": second[10:13].replace("<", ""),
            "date_of_birth": second[13:19],
            "sex": second[20].replace("<", ""),
            "expiry_date": second[21:27],
            "surname": surname,
            "given_names": given_names,
        },
    )


def _parse_td1(lines: list[str]) -> dict[str, Any]:
    first, second, third = (line.ljust(30, "<")[:30] for line in lines[:3])
    checks = {
        "document_number": check_digit(first[5:14], first[14]),
        "date_of_birth": check_digit(second[0:6], second[6]),
        "expiry_date": check_digit(second[8:14], second[14]),
        "optional_data": check_digit(second[18:29], second[29]),
        "composite": check_digit(first[5:15] + second[0:7] + second[8:30], third[29]),
    }
    return _result(
        "TD1",
        [first, second, third],
        checks,
        {
            "document_code": first[0:2].replace("<", ""),
            "issuing_country": first[2:5].replace("<", ""),
            "document_number": first[5:14].replace("<", ""),
            "nationality": second[15:18].replace("<", ""),
            "date_of_birth": second[0:6],
            "sex": second[7].replace("<", ""),
            "expiry_date": second[8:14],
        },
    )


def parse_mrz(text: str) -> dict[str, Any]:
    """Parse OCR text containing a TD1, TD2, or TD3 MRZ."""
    lines = _clean_lines(text)
    if len(lines) >= 3 and len(lines[0]) >= 30 and len(lines[1]) >= 30 and len(lines[2]) >= 30:
        return _parse_td1(lines)
    if len(lines) >= 2 and lines[0].startswith(("P<", "V<")):
        return _parse_td3(lines)
    if len(lines) >= 2 and len(lines[0]) >= 36 and lines[1].startswith(("I<", "A<", "C<")):
        return _parse_td2(lines)
    return _result("unknown", lines, {}, {})