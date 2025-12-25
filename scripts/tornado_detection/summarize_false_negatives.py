"""
Summarize EF numbers and categories for false negatives listed in false_cases.json.

Example:
    python summarize_false_negatives.py tornado_baseline251221231516-None-None/false_cases.json \
        --base /Users/you/path/to/PredictTornet

If the JSON contains absolute paths that don't exist locally, the script will look for
the first occurrence of "tornet_raw" in the path and rebuild it under --base.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Tuple

import xarray as xr


def _resolve_path(raw_path: str, base: Path | None) -> Path | None:
    """
    Resolve file path, optionally rewriting under a new base if missing.
    """
    p = Path(raw_path)
    if p.exists():
        return p
    if base:
        parts = p.parts
        if "tornet_raw" in parts:
            idx = parts.index("tornet_raw")
            candidate = base / Path(*parts[idx:])
            if candidate.exists():
                return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize EF number and category for false negatives from false_cases.json."
    )
    parser.add_argument("json_path", type=Path, help="Path to false_cases.json")
    parser.add_argument(
        "--base",
        type=Path,
        default=None,
        help="Base path to rebuild file paths if JSON paths do not exist locally.",
    )
    args = parser.parse_args()

    data = json.loads(Path(args.json_path).read_text())
    false_negs = data.get("false_negatives", [])
    if not false_negs:
        print("No false_negatives entries found.")
        return

    ef_counts: Counter[int] = Counter()
    cat_counts: Counter[str] = Counter()
    combo_counts: Counter[Tuple[str, int]] = Counter()
    missing: Dict[str, str] = {}

    for entry in false_negs:
        raw_path = entry.get("file", "")
        resolved = _resolve_path(raw_path, args.base)
        if resolved is None:
            missing[raw_path] = "not found locally"
            continue

        try:
            with xr.open_dataset(resolved) as ds:
                ef = int(ds.attrs.get("ef_number", -1))
                cat = str(ds.attrs.get("category", "UNK"))
        except Exception as exc:
            missing[raw_path] = f"read error: {exc}"
            continue

        ef_counts[ef] += 1
        cat_counts[cat] += 1
        combo_counts[(cat, ef)] += 1

    total = sum(ef_counts.values())
    print(f"False negatives summarized: {total}")
    if missing:
        print(f"Missing/failed files: {len(missing)}")
        for k, v in list(missing.items())[:5]:
            print(f"  {k} -> {v}")

    print("\nCounts by EF number:")
    for ef, cnt in sorted(ef_counts.items()):
        print(f"  EF{ef}: {cnt}")

    print("\nCounts by category:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: x[0]):
        print(f"  {cat}: {cnt}")

    print("\nCounts by category + EF:")
    for (cat, ef), cnt in sorted(combo_counts.items(), key=lambda x: (x[0][0], x[0][1])):
        print(f"  {cat}, EF{ef}: {cnt}")


if __name__ == "__main__":
    main()
