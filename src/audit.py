"""Append-only, hash-chained audit records for screening decisions."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def append_audit_record(
    report: dict[str, Any],
    *,
    document_name: str,
    document_path: str | Path | None = None,
    audit_path: str | Path = ".verilogic/audit.jsonl",
) -> dict[str, Any]:
    """Append a report with a previous-hash pointer and return the record."""
    path = Path(audit_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous_hash = ""
    if path.exists():
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            previous_hash = json.loads(lines[-1])["record_hash"]
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "document_name": document_name,
        "previous_hash": previous_hash,
        "report": report,
    }
    if document_path:
        record["document"] = {"perceptual_hash": perceptual_hash(document_path)}
    record["record_hash"] = hashlib.sha256(_canonical(record)).hexdigest()
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    return record


def perceptual_hash(image_path: str | Path) -> str:
    """Return a compact average perceptual hash for cross-case lookup."""
    import numpy as np

    image = Image.open(image_path).convert("L").resize((32, 32))
    pixels = np.asarray(image, dtype=np.float32)
    threshold = float(pixels.mean())
    bits = "".join("1" if pixel >= threshold else "0" for pixel in pixels.flat)
    return f"{int(bits, 2):0{len(bits) // 4}x}"


def verify_audit_chain(audit_path: str | Path) -> dict[str, Any]:
    """Verify ordering and hashes for every record in an audit log."""
    path = Path(audit_path)
    if not path.exists():
        return {"valid": True, "records": 0, "error": None}
    previous_hash = ""
    records = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        expected = record.pop("record_hash", None)
        if record.get("previous_hash", "") != previous_hash:
            return {
                "valid": False,
                "records": records,
                "error": f"broken link at line {line_number}",
            }
        actual = hashlib.sha256(_canonical(record)).hexdigest()
        if expected != actual:
            return {
                "valid": False,
                "records": records,
                "error": f"invalid hash at line {line_number}",
            }
        previous_hash = expected
        records += 1
    return {"valid": True, "records": records, "error": None}


def main() -> None:
    """Verify an audit file from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Verify a VeriLogic audit chain")
    parser.add_argument("command", nargs="?", choices=("verify",), default="verify")
    parser.add_argument("audit_path", type=Path)
    args = parser.parse_args()
    result = verify_audit_chain(args.audit_path)
    if result["valid"]:
        print(f"{result['records']} records verified, chain intact")
        return
    print(f"BROKEN after {result['records']} records: {result['error']}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
