"""
Quick inspection utility for a single TorNet NetCDF file.

Example:
    python analyze_nc_file.py /path/to/sample.nc --frames 1 --variables DBZ VEL
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    import xarray as xr
except ImportError:  # pragma: no cover - runtime dependency
    xr = None  # type: ignore

from tornet.data.constants import ALL_VARIABLES


def _require_xarray() -> None:
    if xr is None:
        raise RuntimeError(
            "xarray is required to run this script. Install with `pip install xarray netcdf4`."
        )


def _spacing_from_coord(values: np.ndarray) -> Tuple[float | None, float | None]:
    """Return median positive step and range (min, max) for a coordinate array."""
    arr = np.asarray(values).astype(np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        return None, None
    sorted_vals = np.sort(finite)
    deltas = np.diff(sorted_vals)
    deltas = deltas[deltas > 0]
    step = float(np.median(deltas)) if deltas.size else None
    span = float(finite.min()), float(finite.max())
    return step, span


def _first_present(ds, candidates: Iterable[str]) -> str | None:
    for name in candidates:
        if name in ds.variables or name in ds.coords:
            return name
    return None


def _coord_summary(ds: xr.Dataset, names: List[str], fallback_bins: int | None) -> Dict:
    coord_name = _first_present(ds, names)
    if coord_name:
        step, span = _spacing_from_coord(ds[coord_name].values)
    elif "range_limits" in ds.variables and fallback_bins:
        limits = ds["range_limits"].values.reshape(-1)
        if limits.size >= 2:
            span = (float(limits[0]), float(limits[1]))
            step = (span[1] - span[0]) / float(fallback_bins)
        else:
            step = span = None
    else:
        step = span = None

    return {
        "coord": coord_name,
        "step": step,
        "span": span,
    }


def _variable_stats(arr: np.ndarray) -> Dict[str, float | int]:
    flat = np.asarray(arr).ravel()
    finite = flat[np.isfinite(flat)]
    return {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "nan_pct": float(100.0 * (np.isnan(flat).sum() / flat.size)),
        "min": float(finite.min()) if finite.size else np.nan,
        "max": float(finite.max()) if finite.size else np.nan,
        "mean": float(finite.mean()) if finite.size else np.nan,
        "std": float(finite.std()) if finite.size else np.nan,
    }


def summarize_file(path: Path, variables: List[str], frames: int | None) -> Dict:
    _require_xarray()
    with xr.open_dataset(path) as ds:
        # pick a representative data variable to describe the grid
        primary_var = None
        for v in variables:
            if v in ds.data_vars:
                primary_var = v
                break
        if primary_var is None:
            data_vars = list(ds.data_vars)
            if not data_vars:
                raise RuntimeError("No data variables found in dataset.")
            primary_var = data_vars[0]

        data = ds[primary_var]
        if frames:
            data = data.isel(time=slice(-frames, None)) if "time" in data.dims else data

        grid_shape = {
            "variable": primary_var,
            "dims": list(data.dims),
            "shape": list(data.shape),
        }

        az_bins = None
        rng_bins = None
        if "azimuth" in data.dims:
            az_bins = data.sizes["azimuth"]
        if "range" in data.dims:
            rng_bins = data.sizes["range"]

        az_summary = _coord_summary(ds, ["azimuth", "azimuth_angles"], az_bins)
        rng_summary = _coord_summary(ds, ["range"], rng_bins)

        var_summaries = {}
        for v in variables:
            if v not in ds.data_vars:
                continue
            var = ds[v]
            if frames and "time" in var.dims:
                var = var.isel(time=slice(-frames, None))
            stats = _variable_stats(var.values)
            var_summaries[v] = stats

        label_summary = None
        if "frame_labels" in ds.variables:
            labels = ds["frame_labels"].values
            label_summary = {
                "total_frames": int(labels.shape[0]),
                "positive": int(np.sum(labels == 1)),
                "negative": int(np.sum(labels == 0)),
            }

    return {
        "file": str(path),
        "grid": grid_shape,
        "resolution": {
            "azimuth": az_summary,
            "range": rng_summary,
        },
        "variables": var_summaries,
        "labels": label_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a TorNet NetCDF sample for grid resolution and basic stats."
    )
    parser.add_argument("file", type=Path, help="Path to a .nc file")
    parser.add_argument(
        "--variables",
        nargs="*",
        default=ALL_VARIABLES,
        help="Variables to summarize (default: all TorNet variables).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Only include the last N time steps when computing stats.",
    )
    args = parser.parse_args()

    stats = summarize_file(args.file, args.variables, args.frames)

    print(f"File: {stats['file']}")
    print(f"Grid variable: {stats['grid']['variable']}")
    print(f"Grid dims: {stats['grid']['dims']} -> {stats['grid']['shape']}")
    print("Resolution:")
    for axis, info in stats["resolution"].items():
        coord = info["coord"] or "n/a"
        step = f"{info['step']:.3f}" if info["step"] is not None else "unknown"
        span = (
            f"[{info['span'][0]:.3f}, {info['span'][1]:.3f}]" if info["span"] else "unknown"
        )
        print(f"  {axis}: coord={coord}, step={step}, span={span}")

    if stats["labels"]:
        lbl = stats["labels"]
        print(
            f"Labels: total={lbl['total_frames']} positives={lbl['positive']} negatives={lbl['negative']}"
        )

    print("Variable stats:")
    for name, info in stats["variables"].items():
        print(
            f"  {name}: shape={info['shape']} dtype={info['dtype']} "
            f"min={info['min']:.3f} max={info['max']:.3f} "
            f"mean={info['mean']:.3f} std={info['std']:.3f} "
            f"nan%={info['nan_pct']:.2f}"
        )


if __name__ == "__main__":
    main()
