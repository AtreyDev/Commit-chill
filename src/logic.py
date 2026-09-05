"""Deterministic identity-document checks used by the unified report."""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from difflib import SequenceMatcher
from typing import Any


def normalize_name(value: str) -> str:
    """Normalize case, punctuation, and whitespace without transliterating text."""
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[\w]+", value, flags=re.UNICODE))


def validate_aadhaar(value: str) -> bool:
    """Validate a 12-digit Aadhaar number with the Verhoeff checksum."""
    digits = re.sub(r"\D", "", value)
    if len(digits) != 12 or len(set(digits)) == 1:
        return False
    multiplication = (
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
        (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
        (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
        (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
        (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
        (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
        (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
        (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
        (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
    )
    permutation = (
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        (0, 5, 7, 8, 9, 4, 2, 1, 3, 6),
        (0, 8, 1, 6, 5, 7, 2, 9, 3, 4),
        (0, 3, 5, 7, 6, 2, 8, 9, 4, 1),
        (0, 6, 3, 2, 8, 7, 1, 9, 5, 4),
        (0, 1, 7, 4, 2, 9, 5, 8, 3, 6),
        (0, 4, 8, 6, 1, 3, 5, 9, 7, 2),
        (0, 2, 9, 5, 8, 7, 6, 3, 1, 4),
    )
    checksum = 0
    for position, digit in enumerate(reversed(digits)):
        checksum = multiplication[checksum][permutation[position % 8][int(digit)]]
    return checksum == 0


PAN_HOLDER_TYPES = frozenset("ABCFGPTLH")


def validate_pan(value: str) -> bool:
    """Validate PAN structure and its fourth-character holder type."""
    normalized = value.strip().upper()
    return bool(
        re.fullmatch(r"[A-Z]{5}\d{4}[A-Z]", normalized)
        and normalized[3] in PAN_HOLDER_TYPES
    )


def _parse_date(value: str) -> date | None:
    match = re.fullmatch(r"(\d{2})[-/](\d{2})[-/](\d{4})", value.strip())
    if not match:
        return None
    try:
        return date(int(match.group(3)), int(match.group(2)), int(match.group(1)))
    except ValueError:
        return None


def run_logic_checks(fields: dict[str, str], *, today: date | None = None) -> list[dict[str, Any]]:
    """Run deterministic checks and return explainable pass/fail records."""
    today = today or date.today()
    checks: list[dict[str, Any]] = []
    if "aadhaar" in fields:
        valid = validate_aadhaar(fields["aadhaar"])
        checks.append({"name": "Aadhaar checksum", "passed": valid, "detail": "Verhoeff checksum"})
    if "pan" in fields:
        valid = validate_pan(fields["pan"])
        checks.append({
            "name": "PAN format and holder type",
            "passed": valid,
            "detail": "AAAAA9999A; fourth character must be a valid holder type",
        })
    if "date_of_birth" in fields:
        dob = _parse_date(fields["date_of_birth"])
        valid = dob is not None and dob <= today
        checks.append({"name": "Date of birth", "passed": valid, "detail": "Valid past date"})
        if dob:
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            checks.append({"name": "DOB to age", "passed": 0 <= age <= 120, "detail": f"Age {age}"})
    issue = _parse_date(fields["issue_date"]) if "issue_date" in fields else None
    expiry = _parse_date(fields["expiry_date"]) if "expiry_date" in fields else None
    if issue or expiry:
        valid = bool(issue and expiry and issue <= expiry)
        checks.append({
            "name": "Issue to expiry",
            "passed": valid,
            "detail": "Issue date precedes expiry",
        })
    if expiry:
        checks.append({
            "name": "Expiry status",
            "passed": expiry >= today,
            "detail": expiry.isoformat(),
        })
    if fields.get("name") and fields.get("native_name"):
        similarity = SequenceMatcher(
            None, normalize_name(fields["name"]), normalize_name(fields["native_name"])
        ).ratio()
        checks.append({
            "name": "Bilingual name consistency",
            "passed": similarity >= 0.70,
            "detail": f"Normalized similarity {similarity:.2f}",
        })
    return checks


def compare_field_sets(field_sets: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Compare shared fields across documents, allowing small name OCR variance."""
    discrepancies: list[dict[str, Any]] = []
    for field in sorted({key for fields in field_sets for key in fields}):
        values = [fields[field] for fields in field_sets if field in fields]
        if len(values) < 2:
            continue
        normalized = [normalize_name(value) if field == "name" else value for value in values]
        similarity = SequenceMatcher(None, normalized[0], normalized[1]).ratio()
        if len(set(normalized)) > 1 and not (field == "name" and similarity >= 0.85):
            discrepancies.append({
                "field": field,
                "values": values,
                "similarity": round(similarity, 3),
            })
    return discrepancies
