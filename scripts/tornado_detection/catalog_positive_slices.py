#!/usr/bin/env python3

"""
Scan a TorNet dataset and record which time slices contain a positive label.

Outputs a CSV with two columns:
  file             Relative path to the NetCDF sample
  positive_slices  Space-separated list of slice indices where frame_labels == 1
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path
from typing import Iterable, List

import xarray as xr

LOG = logging.getLogger(__name__)


def _iter_nc_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.nc")):
        if path.is_file():
            yield path


def _positive_slices(path: Path) -> List[int]:
    with xr.open_dataset(path) as ds:
        if "frame_labels" not in ds:
            raise RuntimeError(f"frame_labels not found in {path}")
        labels = ds["frame_labels"].values
        if labels.ndim != 1:
            raise RuntimeError(f"Unexpected frame_labels shape {labels.shape} in {path}")
        return [i for i, v in enumerate(labels) if int(v) == 1]


def main():
    parser = argparse.ArgumentParser(
        description="Build a CSV of files and the time slices where frame_labels == 1.",
    )
    parser.add_argument(
        "--root",
        default=None,
        type=Path,
        help="Path to tornet_raw root (default: $TORNET_ROOT, then platform-specific fallback).",
    )
    parser.add_argument(
        "--output",
        default="positive_cases_by_slice.csv",
        type=Path,
        help="Where to write the CSV (default: %(default)s)."
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Include files with no positive slices in the CSV.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    if args.root is not None:
        root = args.root.expanduser().resolve()
    elif os.environ.get("TORNET_ROOT"):
        root = Path(os.environ["TORNET_ROOT"]).expanduser().resolve()
    else:
        default_windows = Path(r"C:\Users\bygao\Downloads\tornet_raw")
        root = default_windows if os.name == "nt" else Path("tornet_raw")
        root = root.expanduser().resolve()

    if not root.exists():
        raise SystemExit(f"Data root {root} does not exist.")

    records = []
    for nc_path in _iter_nc_files(root):
        try:
            slices = _positive_slices(nc_path)
        except Exception as exc:
            LOG.warning("Skipping %s: %s", nc_path, exc)
            continue

        if not slices and not args.include_empty:
            continue

        rel_path = nc_path.relative_to(root)
        slice_field = " ".join(str(i) for i in slices)
        records.append((str(rel_path), slice_field))

    if not records:
        raise SystemExit("No records found; rerun with --include-empty or check data root.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "positive_slices"])
        writer.writerows(records)

    LOG.info("Wrote %d rows to %s", len(records), args.output)


if __name__ == "__main__":
    main()
