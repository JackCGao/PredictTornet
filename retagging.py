"""
Retag NetCDF samples by removing existing tornadic frames and shifting the label
one step earlier in time.

For each file:
- Drop all time slices where `frame_labels` is 1 (tornadic).
- Identify the first tornadic slice in the original file; retag the immediately
  preceding non-tornadic slice as tornadic.

Outputs are written to `OUTPUT_ROOT` while preserving filenames.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import xarray as xr

TARGET_VAR = "frame_labels"
OUTPUT_ROOT = Path("tornet_raw") / "retagged_shift"
INPUT_ROOT = Path("tornet_raw")


def _first_tornadic_index(labels: np.ndarray) -> int | None:
    pos = np.where(labels == 1)[0]
    return int(pos[0]) if pos.size else None


def retag_dataset(ds: xr.Dataset) -> Tuple[xr.Dataset | None, str]:
    """
    Retag a single dataset according to the rules described in the module docstring.

    Returns (new_dataset, reason). If new_dataset is None, the reason explains why
    the file was skipped.
    """
    if TARGET_VAR not in ds:
        return None, f"{TARGET_VAR} not found"

    labels = ds[TARGET_VAR].values
    if labels.ndim != 1:
        return None, f"{TARGET_VAR} is not 1D"

    first_tornadic = _first_tornadic_index(labels)
    if first_tornadic is None:
        # Keep file as-is
        kept = ds.copy(deep=True)
        return kept, "no tornadic frames; kept original"
    if first_tornadic == 0:
        return None, "tornado starts at first frame; cannot shift"

    keep_mask = labels == 0
    keep_indices = np.nonzero(keep_mask)[0]
    if keep_indices.size == 0:
        return None, "no non-tornadic frames to keep"

    target_original_idx = first_tornadic - 1
    if not keep_mask[target_original_idx]:
        return None, "preceding frame already tornadic; cannot retag"

    try:
        target_subset_idx = int(np.where(keep_indices == target_original_idx)[0][0])
    except IndexError:
        return None, "failed to locate target frame after filtering"

    # Create new labels aligned to the kept frames
    new_labels = np.zeros(keep_indices.shape, dtype=labels.dtype)
    new_labels[target_subset_idx] = 1

    subset = ds.isel(time=keep_indices)
    subset[TARGET_VAR][:] = new_labels

    return subset, "retagged"


def process_file(src: Path, dst_root: Path, base_root: Path | None = None) -> None:
    try:
        with xr.open_dataset(src) as ds:
            retagged, reason = retag_dataset(ds)
            if retagged is None:
                print(f"Skipping {src}: {reason}")
                return

            rel_path: Path
            if base_root:
                try:
                    rel_path = src.relative_to(base_root)
                except ValueError:
                    rel_path = Path(src.name)
            else:
                rel_path = Path(src.name)

            dst_path = dst_root / rel_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            retagged.to_netcdf(dst_path)
            print(f"Wrote retagged file → {dst_path}")
    except Exception as exc:
        print(f"Failed to process {src}: {exc}")


def discover_nc_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix == ".nc" else []
    nc_files: list[Path] = []
    for root, _, files in os.walk(path):
        for fname in files:
            if fname.endswith(".nc"):
                nc_files.append(Path(root) / fname)
    return sorted(nc_files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retag TorNet samples by shifting tornado onset backward.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=[INPUT_ROOT],
        help="NetCDF files or directories containing .nc files. "
        "Defaults to the tornet_raw directory.",
    )
    parser.add_argument(
        "--output-root",
        default=str(OUTPUT_ROOT),
        help=f"Directory to store retagged files (default: {OUTPUT_ROOT}).",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    for p in args.paths:
        base_root = Path(p).resolve()
        files = discover_nc_files(base_root)
        if not files:
            print(f"No .nc files found under {p}")
            continue
        for f in files:
            process_file(f, output_root, base_root=base_root)


if __name__ == "__main__":
    main()
