"""
Summarize EF numbers and categories for false negatives/positives listed in false_cases.json.

Example:
    python summarize_false_negatives.py tornado_baseline251221231516-None-None/false_cases.json \
        --base /Users/you/path/to/PredictTornet

If the JSON contains absolute paths that don't exist locally, the script will look for
the first occurrence of "tornet_raw" in the path and rebuild it under --base.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
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


def _summarize(cases, base: Path | None, label: str):
    ef_counts: Counter[int] = Counter()
    cat_counts: Counter[str] = Counter()
    combo_counts: Counter[Tuple[str, int]] = Counter()
    missing: Dict[str, str] = {}
    probs = []

    for entry in cases:
        raw_path = entry.get("file", "")
        resolved = _resolve_path(raw_path, base)
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

        prob = entry.get("prob")
        if prob is not None:
            try:
                probs.append(float(prob))
            except Exception:
                pass

        ef_counts[ef] += 1
        cat_counts[cat] += 1
        combo_counts[(cat, ef)] += 1

    total = sum(ef_counts.values())
    print(f"\n== {label} summary ==")
    print(f"Total {label.lower()}: {total}")
    if probs:
        avg = sum(probs) / len(probs)
        print(f"Average predicted probability: {avg:.4f} (n={len(probs)})")

    if missing:
        print(f"Missing/failed files: {len(missing)} (showing up to 5)")
        for k, v in list(missing.items())[:5]:
            print(f"  {k} -> {v}")

    if ef_counts:
        print("\nCounts by EF number:")
        for ef, cnt in sorted(ef_counts.items()):
            print(f"  EF{ef}: {cnt}")

    if cat_counts:
        print("\nCounts by category:")
        for cat, cnt in sorted(cat_counts.items(), key=lambda x: x[0]):
            print(f"  {cat}: {cnt}")

    if combo_counts:
        print("\nCounts by category + EF:")
        for (cat, ef), cnt in sorted(combo_counts.items(), key=lambda x: (x[0][0], x[0][1])):
            print(f"  {cat}, EF{ef}: {cnt}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize EF number and category for false negatives from false_cases.json."
    )
    parser.add_argument(
        "json_path",
        type=Path,
        nargs="?",
        default=Path("/Users/jackgao/Documents/TornadoSight/PredictTornet/PredictTornet/tornado_baseline251221231516-None-None/false_cases.json"),
        help="Path to false_cases.json (default: %(default)s)",
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Base path to rebuild file paths if JSON paths do not exist locally (default: project root).",
    )
    args = parser.parse_args()

    if not args.json_path.exists():
        raise SystemExit(f"JSON file not found: {args.json_path}. Please provide a valid path.")

    data = json.loads(args.json_path.read_text())
    false_negs = data.get("false_negatives", [])
    false_pos = data.get("false_positives", [])
    if not false_negs and not false_pos:
        print("No false_negatives or false_positives entries found.")
        return

    if false_negs:
        _summarize(false_negs, args.base, label="False Negatives")
    if false_pos:
        _summarize(false_pos, args.base, label="False Positives")


if __name__ == "__main__":
    main()
