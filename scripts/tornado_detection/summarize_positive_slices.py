#!/usr/bin/env python3

"""
Summarize how many positives occur in each time-slice category from a
positive_cases_by_slice.csv produced by catalog_positive_slices.py.

The CSV is expected to have columns:
  file, positive_slices
where positive_slices is a space-separated list of slice indices (e.g., "0 1 3").

Outputs counts per slice index to stdout and optionally writes a CSV.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def parse_positive_slices(path: Path) -> Counter:
    counts = Counter()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if "positive_slices" not in reader.fieldnames:
            raise SystemExit(f"CSV at {path} missing 'positive_slices' column.")
        for row in reader:
            slices = row["positive_slices"].strip()
            if not slices:
                continue
            for s in slices.split():
                counts[s] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Count how many positives occur in each slice index.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("positive_cases_by_slice.csv"),
        help="Path to positive_cases_by_slice.csv (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write counts as CSV (columns: slice,count).",
    )
    args = parser.parse_args()

    counts = parse_positive_slices(args.input)
    if not counts:
        raise SystemExit("No positive slices found.")

    print("Slice counts:")
    for slice_id, count in sorted(counts.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
        print(f"  {slice_id}: {count}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["slice", "count"])
            for slice_id, count in sorted(counts.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
                writer.writerow([slice_id, count])


if __name__ == "__main__":
    main()
