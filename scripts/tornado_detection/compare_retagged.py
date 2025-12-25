#!/usr/bin/env python3

"""
Compare original TorNet files against retagged outputs and report differences.

By default, compares .nc files under:
  - raw root: tornet_raw
  - retagged root: tornet_raw/retagged_shift

Outputs counts of matching files, differing files, missing retagged files, and
extra files present only in the retagged tree.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple


def _iter_nc(root: Path) -> Iterable[Path]:
    return (p for p in root.rglob("*.nc") if p.is_file())


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compare_trees(raw_root: Path, retagged_root: Path) -> Dict[str, object]:
    raw_root = raw_root.resolve()
    retagged_root = retagged_root.resolve()

    raw_files: Set[Path] = {p.relative_to(raw_root) for p in _iter_nc(raw_root)}
    retagged_files: Set[Path] = {p.relative_to(retagged_root) for p in _iter_nc(retagged_root)}

    missing = sorted(raw_files - retagged_files)
    extra = sorted(retagged_files - raw_files)

    common = raw_files & retagged_files
    same = []
    different = []

    for rel in sorted(common):
        raw_path = raw_root / rel
        retagged_path = retagged_root / rel
        if _sha256(raw_path) == _sha256(retagged_path):
            same.append(rel)
        else:
            different.append(rel)

    return {
        "raw_root": str(raw_root),
        "retagged_root": str(retagged_root),
        "counts": {
            "raw_files": len(raw_files),
            "retagged_files": len(retagged_files),
            "common": len(common),
            "same": len(same),
            "different": len(different),
            "missing_in_retagged": len(missing),
            "extra_in_retagged": len(extra),
        },
        "missing": [str(p) for p in missing],
        "extra": [str(p) for p in extra],
        "different": [str(p) for p in different],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare raw vs retagged TorNet NetCDF files.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("tornet_raw"),
        help="Path to original data root (default: tornet_raw).",
    )
    parser.add_argument(
        "--retagged-root",
        type=Path,
        default=Path("tornet_raw") / "retagged_shift",
        help="Path to retagged data root (default: tornet_raw/retagged_shift).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print lists of differing/missing/extra files.",
    )
    args = parser.parse_args()

    result = compare_trees(args.raw_root, args.retagged_root)
    counts = result["counts"]  # type: ignore[assignment]

    print("Comparison summary:")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    if args.list:
        for key in ("missing", "extra", "different"):
            items = result[key]  # type: ignore[index]
            if items:
                print(f"\n{key} ({len(items)}):")
                for p in items:
                    print(f"  {p}")


if __name__ == "__main__":
    main()
