"""
Quick visualization of all radar variables for a single TorNet NetCDF sample.

Example:
    python visualize_samples.py /path/to/sample.nc --frame -1 --variables DBZ VEL
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")  # Safe for headless by default; set MPLBACKEND to override.
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from tornet.data.constants import ALL_VARIABLES

def _select_frame(arr: np.ndarray, frame_idx: int) -> np.ndarray:
    """Return a single time slice, handling negative indexing and clamping."""
    if arr.ndim == 0:
        raise ValueError("Array has no dimensions; cannot select frame.")
    total = arr.shape[0]
    idx = frame_idx if frame_idx >= 0 else total + frame_idx
    idx = max(0, min(total - 1, idx))
    return arr[idx]


def _split_tilts(frame: np.ndarray, tilt_index: int | None) -> List[np.ndarray]:
    """Split a frame into per-tilt 2D arrays (azimuth x range)."""
    if frame.ndim == 2:
        tilts = [frame]
    elif frame.ndim == 3:
        tilts = [frame[:, :, i] for i in range(frame.shape[-1])]
    else:
        tilts = [np.squeeze(frame)]

    if tilt_index is not None:
        if 0 <= tilt_index < len(tilts):
            return [tilts[tilt_index]]
        raise IndexError(f"Requested tilt_index {tilt_index} but only {len(tilts)} tilts available.")
    return tilts


def _load_variables(ds: xr.Dataset, variables: Sequence[str], frame_idx: int, tilt_index: int | None):
    data = []
    max_tilts = 0
    for var in variables:
        if var not in ds:
            continue
        arr = ds[var].values
        frame = _select_frame(arr, frame_idx)
        tilts = _split_tilts(frame, tilt_index)
        data.append((var, tilts))
        max_tilts = max(max_tilts, len(tilts))
    if not data:
        raise RuntimeError("None of the requested variables were found in the dataset.")
    return data, max_tilts


def plot_file(path: Path, variables: Sequence[str], frame_idx: int, tilt_index: int | None, save_path: Path | None):
    with xr.open_dataset(path) as ds:
        var_data, max_tilts = _load_variables(ds, variables, frame_idx, tilt_index)

        fig, axes = plt.subplots(
            len(var_data),
            max_tilts,
            figsize=(4 * max_tilts, 3 * len(var_data)),
            squeeze=False,
        )

        for row, (var, tilts) in enumerate(var_data):
            for col in range(max_tilts):
                ax = axes[row][col]
                if col >= len(tilts):
                    ax.axis("off")
                    continue
                im = ax.imshow(tilts[col], aspect="auto", origin="lower")
                ax.set_title(f"{var} tilt {col}")
                ax.set_xlabel("Range bin")
                if col == 0:
                    ax.set_ylabel("Azimuth bin")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(f"{path.name} | frame {frame_idx} | tilts {tilt_index if tilt_index is not None else 'all'}")
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
        plt.close(fig)


def _safe_relpath(path: Path, root: Path | None) -> Path:
    if root is None:
        return Path(path.name)
    try:
        return path.resolve().relative_to(root.resolve())
    except Exception:
        return Path(path.name)


def plot_false_cases(
    false_cases_json: Path,
    out_dir: Path,
    variables: Sequence[str],
    frame_idx: int,
    tilt_index: int | None,
    data_root: Path | None,
):
    with false_cases_json.open("r") as f:
        data = json.load(f)
    for key in ["false_positives", "false_negatives"]:
        cases = data.get(key, [])
        if not cases:
            continue
        for entry in cases:
            file_path = Path(entry["file"])
            rel = _safe_relpath(file_path, data_root)
            save_path = out_dir / key / rel.with_suffix(".png")
            try:
                plot_file(file_path, variables, frame_idx, tilt_index, save_path)
            except Exception as exc:  # noqa: BLE001
                print(f"Failed to plot {file_path}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize TorNet variables for a single NetCDF sample or cataloged false cases.",
    )
    parser.add_argument("file", type=Path, nargs="?", help="Path to a .nc file")
    parser.add_argument(
        "--false-cases",
        type=Path,
        help="Path to false_cases.json (from test_tornado_torch.py) to batch-plot false positives/negatives.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("false_case_plots"),
        help="Output directory for batch plotting false cases (default: false_case_plots).",
    )
    parser.add_argument(
        "--variables",
        nargs="*",
        default=ALL_VARIABLES,
        help="Variables to plot (default: all TorNet variables).",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=-1,
        help="Time frame index to plot (default: -1 for last frame).",
    )
    parser.add_argument(
        "--tilt-index",
        type=int,
        default=None,
        help="Plot only a specific tilt (0-based). Omit to plot all tilts.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Path to save the figure. If omitted, displays the plot.",
    )
    args = parser.parse_args()

    if args.false_cases:
        data_root = Path(os.environ["TORNET_ROOT"]).resolve() if "TORNET_ROOT" in os.environ else None
        plot_false_cases(
            false_cases_json=args.false_cases,
            out_dir=args.output_dir,
            variables=args.variables,
            frame_idx=args.frame,
            tilt_index=args.tilt_index,
            data_root=data_root,
        )
    elif args.file:
        plot_file(args.file, args.variables, args.frame, args.tilt_index, args.save)
    else:
        parser.error("Either provide a file to plot or --false-cases for batch plotting.")

if __name__ == "__main__":
    main()
