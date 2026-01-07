from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")  # Safe for headless by default; set MPLBACKEND to override.
import matplotlib.pyplot as plt
import numpy as np

from tornet.data.constants import ALL_VARIABLES
from tornet.data.loader import read_file
from tornet.display.display import plot_radar


def _grid_for_channels(n_channels: int) -> tuple[int, int]:
    """Pick a rows/cols grid similar to the notebook (2x3 for 6 vars)."""

    if n_channels <= 3:
        return (1, n_channels)
    if n_channels <= 6:
        return (2, 3)
    cols = int(np.ceil(np.sqrt(n_channels)))
    rows = int(np.ceil(n_channels / cols))
    return rows, cols


def _load_sample_for_radar(path: Path, variables: Sequence[str]) -> Dict[str, np.ndarray]:
    """Use the existing loader helper so the structure matches plot_radar expectations."""

    data = read_file(str(path), variables=variables, tilt_last=True)
    return data


def plot_file(path: Path, variables: Sequence[str], frame_idx: int, tilt_index: int | None, save_path: Path | None):
    data = _load_sample_for_radar(path, variables)
    # Clamp indices to available frames/tilts to avoid crashes.
    time_len = data[variables[0]].shape[0]
    time_idx = frame_idx if frame_idx >= 0 else time_len + frame_idx
    time_idx = max(0, min(time_len - 1, time_idx))

    n_tilts = data[variables[0]].shape[-1] if data[variables[0]].ndim >= 3 else 1
    chosen_tilt = tilt_index if tilt_index is not None else 0
    chosen_tilt = max(0, min(n_tilts - 1, chosen_tilt))
    sweep_idx = [chosen_tilt] * len(variables)

    n_rows, n_cols = _grid_for_channels(len(variables))

    fig = plt.figure(figsize=(12, 6), edgecolor="k")
    plot_radar(
        data,
        channels=list(variables),
        fig=fig,
        time_idx=time_idx,
        sweep_idx=sweep_idx,
        include_cbar=True,
        n_rows=n_rows,
        n_cols=n_cols,
    )
    fig.suptitle(f"{path.name} | frame {time_idx}", y=0.98)
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
