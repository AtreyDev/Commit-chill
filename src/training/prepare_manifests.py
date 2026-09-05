"""Prepare dataset manifests from locally downloaded datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.datasets.fgnet import build_fgnet_manifest
from src.datasets.idnet import build_idnet_manifest
from src.datasets.manifest import write_manifest
from src.datasets.midv2020 import build_midv2020_manifest


def prepare(dataset: str, root: Path, output: Path, split: str) -> Path:
    """Build one dataset manifest and fail clearly when its root is absent."""
    builders = {
        "midv2020": build_midv2020_manifest,
        "idnet": build_idnet_manifest,
        "fgnet": build_fgnet_manifest,
    }
    if dataset not in builders:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    records = builders[dataset](root, split=split)
    if not records:
        raise ValueError(f"No valid images found under {root}")
    return write_manifest(records, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("midv2020", "idnet", "fgnet"))
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    output = prepare(args.dataset, args.root, args.output, args.split)
    print(f"Wrote manifest: {output}")


if __name__ == "__main__":
    main()