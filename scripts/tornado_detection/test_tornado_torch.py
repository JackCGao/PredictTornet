from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:  # pragma: no cover - optional dependency
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryStatScores,
)
from torchmetrics.functional.classification import (
    binary_precision_recall_curve,
    binary_roc,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tornet.data.constants import ALL_VARIABLES
from tornet.data.loader import get_dataloader, read_file
import tornet.data.preprocess as pp
from tornet.models.torch.cnn_baseline import TornadoClassifier, TornadoLikelihood

logging.basicConfig(level=logging.INFO)

DATA_ROOT = os.environ["TORNET_ROOT"]
DATA_ROOT_PATH = Path(DATA_ROOT)
logging.info("TORNET_ROOT=%s", DATA_ROOT)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate TorNet Torch model.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Path to a Lightning checkpoint (.ckpt) produced by train_tornado_torch.py. "
            "If omitted, the script will try to use latest/checkpoints/last.ckpt under EXP_DIR "
            "(default: current directory)."
        ),
    )
    parser.add_argument(
        "--dataloader",
        default="torch",
        choices=["torch", "torch-tfds", "keras", "tensorflow", "tensorflow-tfds"],
        help="Which data loader backend to use.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Evaluation batch size."
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use (cpu, cuda, cuda:0, etc). 'auto' selects CUDA if available.",
    )
    parser.add_argument(
        "--tilt-last",
        action="store_true",
        help="Keep tilt dimension last (matches keras defaults).",
    )
    parser.add_argument(
        "--include-range-folded",
        action="store_true",
        help="Force inclusion of range_folded_mask feature if your checkpoint expects it.",
    )
    optuna_group = parser.add_mutually_exclusive_group()
    optuna_group.add_argument(
        "--use-best-optuna",
        action="store_true",
        default=True,
        help=(
            "Load the best trial params from the Optuna study and apply them for evaluation "
            "(default: enabled)."
        ),
    )
    optuna_group.add_argument(
        "--no-best-optuna",
        dest="use_best_optuna",
        action="store_false",
        help="Disable loading the best Optuna trial params.",
    )
    parser.add_argument(
        "--optuna-storage",
        default="sqlite:///tornet_optuna.db",
        help="Optuna storage URL (default: sqlite:///tornet_optuna.db).",
    )
    parser.add_argument(
        "--optuna-study",
        default="*",
        help=(
            "Optuna study name(s) to consider (comma-separated). "
            "Use '*' to search all studies in the storage (default: '*')."
        ),
    )
    grad_cam_group = parser.add_mutually_exclusive_group()
    grad_cam_group.add_argument(
        "--grad-cam",
        action="store_true",
        default=True,
        help="Generate Grad-CAM plots for a subset of samples (default: enabled).",
    )
    grad_cam_group.add_argument(
        "--no-grad-cam",
        dest="grad_cam",
        action="store_false",
        help="Disable Grad-CAM plots.",
    )
    parser.add_argument(
        "--grad-cam-output",
        type=Path,
        default=None,
        help="Directory to save Grad-CAM plots (default: eval output directory).",
    )
    parser.add_argument(
        "--grad-cam-samples",
        type=int,
        default=2,
        help="Maximum number of samples to render Grad-CAM plots for.",
    )
    parser.add_argument(
        "--grad-cam-tilt-index",
        type=int,
        default=0,
        help="Tilt index to plot when rendering Grad-CAM (default: 0).",
    )
    parser.add_argument(
        "--non-retagged-root",
        type=Path,
        default=DATA_ROOT_PATH.parent / "tornet_raw",
        help=(
            "Root directory for non-retagged Tornet files to plot the actual labeled "
            "signature for the success-case sample."
        ),
    )
    return parser


def _select_device(device_opt: str) -> torch.device:
    if device_opt == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_opt)


def _collate_with_labels(batch):
    """Collate samples into a dict batch; defined at module level for Windows pickling."""

    sample = batch[0]
    has_weights = len(sample) == 3
    feats = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    weights = [x[2] for x in batch] if has_weights else None

    feats = default_collate(feats)
    feats["label"] = default_collate(labels)
    if has_weights:
        feats["sample_weights"] = default_collate(weights)
    return feats


def _wrap_loader(loader: DataLoader) -> DataLoader:
    """Convert (features, label[, weight]) -> dict batches for compatibility."""

    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        persistent_workers=(
            getattr(loader, "persistent_workers", False) and loader.num_workers > 0
        ),
        collate_fn=_collate_with_labels,
    )


def _infer_input_shapes(
    batch: Dict[str, torch.Tensor], variables: Iterable[str]
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    feature_shape = None
    for var in variables:
        if var in batch:
            feature_shape = tuple(batch[var].shape[1:])
            break
    if feature_shape is None:
        raise RuntimeError("Unable to infer feature shape from batch.")
    coord_shape = tuple(batch["coordinates"].shape[1:])
    return feature_shape, coord_shape


def _move_to_device(batch: Dict[str, torch.Tensor], device: torch.device):
    for key, value in batch.items():
        if torch.is_tensor(value):
            batch[key] = value.to(device)
    return batch


def _build_metrics(device: torch.device) -> MetricCollection:
    return MetricCollection(
        {
            "AUC": BinaryAUROC(),
            "AUCPR": BinaryAveragePrecision(),
            "AUCPD": BinaryAveragePrecision(),
            "BinaryAccuracy": BinaryAccuracy(),
            "Precision": BinaryPrecision(),
            "Recall": BinaryRecall(),
            "F1": BinaryF1Score(),
        }
    ).to(device)

def _compute_csi(tp: float, fp: float, fn: float) -> float | None:
    denom = tp + fp + fn
    if denom == 0:
        return None
    return float(tp / denom)

def _load_training_config(checkpoint_path: Path) -> Dict[str, Any]:
    """Load the training config saved alongside a checkpoint (params.json)."""

    cfg_path = checkpoint_path.resolve().parent.parent / "params.json"
    if not cfg_path.exists():
        return {}
    try:
        data = json.load(open(cfg_path, "r"))
    except Exception as exc:  # noqa: BLE001
        logging.warning("Could not read training config at %s: %s", cfg_path, exc)
        return {}

    if isinstance(data, dict):
        config = data.get("config", data)
        return config if isinstance(config, dict) else {}
    return {}


def _load_best_optuna_params(storage: str, study_name: str) -> Dict[str, Any]:
    """Load best trial parameters from one or more Optuna studies."""

    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("optuna is required to load best trial params. Install with `pip install optuna`.") from exc

    study_filter = [s.strip() for s in study_name.split(",") if s.strip()]
    use_all = not study_filter or any(s in {"*", "any", "all"} for s in study_filter)
    use_first = any(s in {"first"} for s in study_filter)
    if use_all:
        study_filter = []

    summaries = optuna.study.get_all_study_summaries(storage=storage)
    if not summaries:
        raise SystemExit(f"No Optuna studies found in storage {storage}.")
    if use_first:
        summaries = [
            min(
                summaries,
                key=lambda s: getattr(s, "study_id", getattr(s, "_study_id", 0)),
            )
        ]
        study_filter = []

    candidates: List[Tuple[float, str, Any]] = []
    for summary in summaries:
        if study_filter and summary.study_name not in study_filter:
            continue
        study = optuna.load_study(study_name=summary.study_name, storage=storage)
        if len(study.directions) > 1:
            if not study.best_trials:
                continue
            best = max(study.best_trials, key=lambda t: t.values[0])
            best_value = best.values[0]
            direction = study.directions[0]
        else:
            best = study.best_trial
            best_value = best.value
            direction = study.direction
        if best_value is None or not math.isfinite(float(best_value)):
            continue
        score = float(best_value)
        if direction == optuna.study.StudyDirection.MINIMIZE:
            score = -score
        candidates.append((score, summary.study_name, best))

    if not candidates:
        raise SystemExit("No usable Optuna trials found for evaluation.")

    _, study_name, best = max(candidates, key=lambda item: item[0])
    best_value = best.values[0] if getattr(best, "values", None) else best.value
    logging.info(
        "Loaded best Optuna trial %s from study '%s' (value=%s)",
        best.number,
        study_name,
        best_value,
    )
    return dict(best.params)


def _normalize_optuna_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Optuna parameter names into training config keys when needed."""

    normalized = dict(params)
    conv_keys = [k for k in params if k.startswith("convs_block")]
    if conv_keys:
        convs_per_block = []
        for idx in range(1, 5):
            key = f"convs_block{idx}"
            if key in params:
                convs_per_block.append(int(params[key]))
        if convs_per_block:
            normalized["convs_per_block"] = convs_per_block
        for key in conv_keys:
            normalized.pop(key, None)
    return normalized


def _values_match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, (list, tuple)) or isinstance(actual, (list, tuple)):
        expected_list = list(expected) if isinstance(expected, (list, tuple)) else [expected]
        actual_list = list(actual) if isinstance(actual, (list, tuple)) else [actual]
        if len(expected_list) != len(actual_list):
            return False
        return all(_values_match(e, a) for e, a in zip(expected_list, actual_list))
    if isinstance(expected, (int, float)) or isinstance(actual, (int, float)):
        try:
            return math.isclose(float(expected), float(actual), rel_tol=1e-6, abs_tol=1e-8)
        except (TypeError, ValueError):
            return False
    return expected == actual


def _config_matches_params(config: Dict[str, Any], params: Dict[str, Any]) -> bool:
    for key, value in params.items():
        if key not in config:
            return False
        if not _values_match(config[key], value):
            return False
    return True


def _log_config_parity(config: Dict[str, Any], params: Dict[str, Any]) -> None:
    mismatches = []
    for key, value in params.items():
        if key not in config:
            mismatches.append((key, value, "<missing>"))
            continue
        if not _values_match(config[key], value):
            mismatches.append((key, value, config[key]))
    if mismatches:
        logging.warning("Optuna params do not fully match training config:")
        for key, expected, actual in mismatches:
            logging.warning("  %s: optuna=%s config=%s", key, expected, actual)
    else:
        logging.info("Optuna params match training config.")


def _find_checkpoint_in_expdir(exp_dir: Path) -> Path | None:
    ckpt_dir = exp_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ckpts:
        return None
    non_last = [p for p in ckpts if p.name != "last.ckpt"]
    return non_last[0] if non_last else ckpts[0]


def _find_optuna_checkpoint(best_params: Dict[str, Any], exp_root: Path) -> Path | None:
    """Find the checkpoint directory whose params.json matches the best Optuna params."""

    candidates: List[Path] = []
    if exp_root.exists():
        candidates.extend([p for p in exp_root.iterdir() if p.is_dir()])

    for exp_dir in candidates:
        params_path = exp_dir / "params.json"
        if not params_path.exists():
            continue
        try:
            payload = json.loads(params_path.read_text())
        except Exception:  # noqa: BLE001
            continue
        config = payload.get("config", payload) if isinstance(payload, dict) else {}
        if not isinstance(config, dict):
            continue
        if not _config_matches_params(config, best_params):
            continue
        checkpoint = _find_checkpoint_in_expdir(exp_dir)
        if checkpoint:
            return checkpoint
    return None

def _drop_missing_files(loader: DataLoader, loader_name: str) -> None:
    """
    Remove entries from the dataloader's file list if the files are missing on disk.

    Some catalogs may contain stale paths; we quietly skip them so evaluation can
    continue with the remaining files.
    """

    dataset = getattr(loader, "dataset", None)
    file_list = getattr(dataset, "file_list", None)
    if not file_list:
        return

    missing = [f for f in file_list if not Path(f).exists()]
    if not missing:
        return

    dataset.file_list = [f for f in file_list if Path(f).exists()]
    logging.warning(
        "%s dataloader: %d files listed in catalog were missing; skipping them. "
        "First missing example: %s",
        loader_name,
        len(missing),
        missing[0],
    )

    if not dataset.file_list:
        raise RuntimeError(
            f"All files referenced by the {loader_name} catalog are missing. "
            "Cannot continue."
        )

def _find_default_checkpoint() -> Path | None:
    """Try to find a checkpoint from the latest experiment directory."""

    exp_dir = Path(os.environ.get("EXP_DIR", ".")).resolve()
    ckpt_dir = exp_dir / "latest" / "checkpoints"
    if not ckpt_dir.exists():
        return None

    for name in ("last.ckpt", "best.ckpt"):
        candidate = ckpt_dir / name
        if candidate.exists():
            return candidate

    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0] if ckpts else None


class _GradCAM:
    """Lightweight Grad-CAM helper that hooks into a target module."""

    def __init__(self, target_module: torch.nn.Module):
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.handles = [
            target_module.register_forward_hook(self._forward_hook),
            target_module.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output

    def _backward_hook(self, _module, _grad_inputs, grad_outputs):
        self.gradients = grad_outputs[0]

    def clear(self):
        self.activations = None
        self.gradients = None

    def remove(self):
        for h in self.handles:
            h.remove()

    def build_cam(self) -> torch.Tensor:
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)
        cam_min = cam.amin(dim=(1, 2), keepdim=True)
        cam_max = cam.amax(dim=(1, 2), keepdim=True)
        return (cam - cam_min) / (cam_max - cam_min + 1e-6)


def _safe_filename(file_list: List[str] | None, idx: int) -> str:
    if file_list and 0 <= idx < len(file_list):
        return Path(file_list[idx]).with_suffix("").name
    return f"sample_{idx}"

def _map_to_non_retagged(
    sample_path: Path,
    non_retagged_root: Path,
    retagged_root: Path | None,
) -> Path | None:
    if retagged_root is not None:
        try:
            rel_path = sample_path.relative_to(retagged_root)
        except ValueError:
            rel_path = None
        if rel_path is not None:
            candidate = non_retagged_root / rel_path
            if candidate.exists():
                return candidate

    matches = sorted(non_retagged_root.rglob(sample_path.name))
    if matches:
        if len(matches) > 1:
            logging.warning(
                "Multiple non-retagged matches for %s; using %s",
                sample_path.name,
                matches[0],
            )
        return matches[0]
    return None

def _find_tornadic_time_index(sample_path: Path) -> int | None:
    try:
        with xr.open_dataset(sample_path) as ds:
            if "frame_labels" not in ds:
                return None
            labels = ds["frame_labels"].values
    except Exception as exc:  # noqa: BLE001
        logging.warning("Failed to read frame_labels from %s: %s", sample_path, exc)
        return None
    positive = np.where(labels == 1)[0]
    if positive.size == 0:
        return None
    return int(positive[-1])

def _load_plot_data(
    sample_path: Path,
    variables: List[str],
    time_idx: int,
    tilt_last: bool = True,
) -> Dict[str, np.ndarray]:
    data: Dict[str, np.ndarray] = {}
    with xr.open_dataset(sample_path) as ds:
        for v in variables:
            if v not in ds:
                continue
            arr = ds[v].isel(time=time_idx).values
            arr = arr[None, ...]
            if not tilt_last and arr.ndim == 4:
                arr = np.transpose(arr, (0, 3, 1, 2))
            data[v] = arr
        if "time" in ds:
            data["time"] = np.array([np.int64(ds["time"].values[time_idx])])
        if "azimuth_limits" in ds:
            data["az_lower"] = np.array(ds["azimuth_limits"].values[0:1])
            data["az_upper"] = np.array(ds["azimuth_limits"].values[1:])
        if "range_limits" in ds:
            data["rng_lower"] = np.array(ds["range_limits"].values[0:1])
            data["rng_upper"] = np.array(ds["range_limits"].values[1:])
        _update_latlon_metadata(ds, data)
        data["event_id"] = np.array([int(ds.attrs.get("event_id", -1))], dtype=np.int64)
        data["ef_number"] = np.array([int(ds.attrs.get("ef_number", -1))], dtype=np.int64)
    return data

def _load_plot_metadata(sample_path: Path) -> Dict[str, np.ndarray]:
    data: Dict[str, np.ndarray] = {}
    try:
        with xr.open_dataset(sample_path) as ds:
            if "time" in ds:
                data["time"] = np.array([np.int64(ds["time"].values[-1])])
            if "azimuth_limits" in ds:
                data["az_lower"] = np.array(ds["azimuth_limits"].values[0:1])
                data["az_upper"] = np.array(ds["azimuth_limits"].values[1:])
            if "range_limits" in ds:
                data["rng_lower"] = np.array(ds["range_limits"].values[0:1])
                data["rng_upper"] = np.array(ds["range_limits"].values[1:])
            _update_latlon_metadata(ds, data)
            data["event_id"] = np.array([int(ds.attrs.get("event_id", -1))], dtype=np.int64)
            data["ef_number"] = np.array([int(ds.attrs.get("ef_number", -1))], dtype=np.int64)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Failed to load plot metadata from %s: %s", sample_path, exc)
    return data


def _update_latlon_metadata(ds: xr.Dataset, data: Dict[str, np.ndarray]) -> None:
    keys = [
        ("radar_lat", "radar_lon"),
        ("station_lat", "station_lon"),
        ("site_lat", "site_lon"),
        ("site_latitude", "site_longitude"),
        ("latitude", "longitude"),
        ("lat", "lon"),
    ]
    for lat_key, lon_key in keys:
        if lat_key in ds.attrs and lon_key in ds.attrs:
            data["radar_lat"] = np.array([float(ds.attrs[lat_key])])
            data["radar_lon"] = np.array([float(ds.attrs[lon_key])])
            return
        if lat_key in ds and lon_key in ds:
            data["radar_lat"] = np.array([float(ds[lat_key].values)])
            data["radar_lon"] = np.array([float(ds[lon_key].values)])
            return


def _latlon_grid(
    plot_meta: Dict[str, np.ndarray],
    target_h: int,
    target_w: int,
) -> Tuple[np.ndarray, np.ndarray] | None:
    if "radar_lat" not in plot_meta or "radar_lon" not in plot_meta:
        return None
    if "az_lower" not in plot_meta or "az_upper" not in plot_meta:
        return None
    if "rng_lower" not in plot_meta or "rng_upper" not in plot_meta:
        return None

    lat0 = float(np.asarray(plot_meta["radar_lat"])[0])
    lon0 = float(np.asarray(plot_meta["radar_lon"])[0])
    az_lower = float(np.asarray(plot_meta["az_lower"])[0])
    az_upper = float(np.asarray(plot_meta["az_upper"])[0])
    rng_lower = float(np.asarray(plot_meta["rng_lower"])[0])
    rng_upper = float(np.asarray(plot_meta["rng_upper"])[0])
    if az_upper <= az_lower:
        az_lower, az_upper = 0.0, 360.0
    if rng_upper <= rng_lower:
        rng_lower, rng_upper = 0.0, 1.0

    az = np.deg2rad(np.linspace(az_lower, az_upper, target_h))
    rng = np.linspace(rng_lower, rng_upper, target_w)
    rng_grid, az_grid = np.meshgrid(rng, az)

    earth_radius = 6371000.0
    lat0_rad = np.deg2rad(lat0)
    lon0_rad = np.deg2rad(lon0)
    ang_dist = rng_grid / earth_radius

    sin_lat0 = np.sin(lat0_rad)
    cos_lat0 = np.cos(lat0_rad)
    sin_ad = np.sin(ang_dist)
    cos_ad = np.cos(ang_dist)

    lat_rad = np.arcsin(sin_lat0 * cos_ad + cos_lat0 * sin_ad * np.cos(az_grid))
    lon_rad = lon0_rad + np.arctan2(
        np.sin(az_grid) * sin_ad * cos_lat0,
        cos_ad - sin_lat0 * np.sin(lat_rad),
    )

    return np.rad2deg(lon_rad), np.rad2deg(lat_rad)


def _contour_levels(data: np.ndarray, n_levels: int = 4) -> np.ndarray | None:
    finite = np.asarray(data)[np.isfinite(data)]
    if finite.size == 0:
        return None
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if vmax <= vmin:
        return None
    return np.linspace(vmin, vmax, n_levels + 2)[1:-1]

def _build_model_input(
    sample_path: Path,
    variables: List[str],
    time_idx: int,
    tilt_last: bool,
    include_range_folded: bool,
    include_az: bool,
) -> Tuple[Dict[str, torch.Tensor], int, int, Dict[str, np.ndarray]]:
    data: Dict[str, torch.Tensor] = {}
    plot_meta: Dict[str, np.ndarray] = {}
    with xr.open_dataset(sample_path) as ds:
        if "time" in ds:
            plot_meta["time"] = np.array([np.int64(ds["time"].values[time_idx])])
        if "azimuth_limits" in ds:
            plot_meta["az_lower"] = np.array(ds["azimuth_limits"].values[0:1])
            plot_meta["az_upper"] = np.array(ds["azimuth_limits"].values[1:])
        if "range_limits" in ds:
            plot_meta["rng_lower"] = np.array(ds["range_limits"].values[0:1])
            plot_meta["rng_upper"] = np.array(ds["range_limits"].values[1:])
        _update_latlon_metadata(ds, plot_meta)
        plot_meta["event_id"] = np.array([int(ds.attrs.get("event_id", -1))], dtype=np.int64)
        plot_meta["ef_number"] = np.array([int(ds.attrs.get("ef_number", -1))], dtype=np.int64)

        target_h = None
        target_w = None
        for v in variables:
            if v not in ds:
                continue
            arr = ds[v].isel(time=time_idx).values
            if not tilt_last and arr.ndim == 3:
                arr = np.transpose(arr, (2, 0, 1))
            if target_h is None or target_w is None:
                if tilt_last:
                    target_h, target_w = arr.shape[0], arr.shape[1]
                else:
                    target_h, target_w = arr.shape[1], arr.shape[2]
            data[v] = torch.as_tensor(arr)

        if include_range_folded and "range_folded_mask" in ds:
            arr = ds["range_folded_mask"].isel(time=time_idx).values
            if not tilt_last and arr.ndim == 3:
                arr = np.transpose(arr, (2, 0, 1))
            data["range_folded_mask"] = torch.as_tensor(arr)

        data["az_lower"] = torch.as_tensor(plot_meta.get("az_lower", np.array([0.0])))
        data["az_upper"] = torch.as_tensor(plot_meta.get("az_upper", np.array([360.0])))
        data["rng_lower"] = torch.as_tensor(plot_meta.get("rng_lower", np.array([0.0])))
        data["rng_upper"] = torch.as_tensor(plot_meta.get("rng_upper", np.array([1.0])))

    if target_h is None or target_w is None:
        raise RuntimeError(f"No variables available to build model input for {sample_path}")

    data = pp.add_coordinates(
        data, include_az=include_az, tilt_last=tilt_last, backend=torch
    )
    for key, value in data.items():
        if torch.is_tensor(value):
            data[key] = value.unsqueeze(0)
    return data, target_h, target_w, plot_meta

def _grid_for_channels(n_channels: int) -> Tuple[int, int]:
    if n_channels <= 3:
        return (1, n_channels)
    if n_channels <= 6:
        return (2, 3)
    cols = int(np.ceil(np.sqrt(n_channels)))
    rows = int(np.ceil(n_channels / cols))
    return rows, cols


def _prepare_plot_sample(
    batch: Dict[str, torch.Tensor], sample_idx: int, variables: Iterable[str], tilt_last: bool
) -> Dict[str, np.ndarray]:
    """Format a single sample for plot_radar (tilt-last with time dim)."""

    data: Dict[str, np.ndarray] = {}
    required_keys = {
        "event_id",
        "ef_number",
        "time",
        "az_lower",
        "az_upper",
        "rng_lower",
        "rng_upper",
    }
    for key, value in batch.items():
        if not torch.is_tensor(value):
            continue
        arr = value[sample_idx].detach().cpu().numpy()
        data[key] = arr

    for key, arr in list(data.items()):
        if key in variables or key == "range_folded_mask":
            if not tilt_last:
                if arr.ndim == 3:
                    arr = np.transpose(arr, (1, 2, 0))
                elif arr.ndim == 4:
                    arr = np.transpose(arr, (0, 2, 3, 1))
            if arr.ndim == 3:
                arr = arr[None, ...]
            data[key] = arr
        elif np.isscalar(arr):
            data[key] = np.array([arr])
        else:
            data[key] = arr

    missing = required_keys.difference(data.keys())
    for key in missing:
        if key in {"az_lower", "rng_lower"}:
            data[key] = np.array([0.0], dtype=np.float64)
        elif key == "az_upper":
            data[key] = np.array([360.0], dtype=np.float64)
        elif key == "rng_upper":
            data[key] = np.array([1.0], dtype=np.float64)
        else:
            data[key] = np.array([-1], dtype=np.int64)
    return data


def _generate_grad_cam_plots(
    classifier: TornadoClassifier,
    loader: DataLoader,
    variables: List[str],
    tilt_last: bool,
    out_dir: Path,
    max_samples: int,
    file_list: List[str] | None,
    tilt_index: int,
    device: torch.device,
    non_retagged_root: Path,
    retagged_root: Path | None,
):
    if plt is None:  # pragma: no cover - optional dependency
        logging.warning("matplotlib unavailable; skipping likelihood plots.")
        return
    try:
        from tornet.display.display import get_cmap
    except Exception as exc:  # pragma: no cover - optional dependency
        logging.warning("tornet.display.display unavailable; skipping likelihood plots: %s", exc)
        return

    saved = 0
    sample_index = 0
    out_dir.mkdir(parents=True, exist_ok=True)

    for batch in loader:
        if saved >= max_samples:
            break
        batch_for_plot = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        batch = _move_to_device(batch, device)
        batch.pop("label", None)
        batch.pop("sample_weights", None)

        with torch.no_grad():
            logits = classifier.model(batch)
            probs = torch.sigmoid(logits).detach().cpu()

        batch_size = probs.shape[0]
        for i in range(batch_size):
            if saved >= max_samples:
                break
            plot_data = _prepare_plot_sample(batch_for_plot, i, variables, tilt_last)
            if file_list is not None:
                file_idx = sample_index + i
                if 0 <= file_idx < len(file_list):
                    source_path = Path(file_list[file_idx])
                    meta_path = _map_to_non_retagged(
                        source_path, non_retagged_root, retagged_root
                    ) or source_path
                    plot_data.update(_load_plot_metadata(meta_path))
            if not variables:
                logging.warning("No variables available for Grad-CAM plotting.")
                continue

            var_shape = plot_data[variables[0]].shape if variables else None
            if not var_shape or len(var_shape) < 3:
                logging.warning("Unexpected variable shape for likelihood plot: %s", var_shape)
                continue

            target_h = var_shape[1]
            target_w = var_shape[2]
            prob_tensor = probs[i].unsqueeze(0)
            prob_resized = F.interpolate(
                prob_tensor,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            plot_data["cnn_output"] = prob_resized.numpy()[None, ..., None]
            file_tag = _safe_filename(file_list, sample_index + i)

            az_lower = float(np.asarray(plot_data.get("az_lower", [0.0]))[0])
            az_upper = float(np.asarray(plot_data.get("az_upper", [360.0]))[0])
            rng_lower = float(np.asarray(plot_data.get("rng_lower", [0.0]))[0]) / 1e3
            rng_upper = float(np.asarray(plot_data.get("rng_upper", [1.0]))[0]) / 1e3
            if az_upper <= az_lower:
                az_lower, az_upper = 0.0, 360.0
            if rng_upper <= rng_lower:
                rng_lower, rng_upper = 0.0, 1.0

            cmap, norm = get_cmap("cnn_output")
            fig, ax = plt.subplots(figsize=(8, 6))
            lonlat = _latlon_grid(plot_data, target_h, target_w)
            if lonlat is None:
                img = ax.imshow(
                    plot_data["cnn_output"][0, ..., 0],
                    origin="lower",
                    aspect="auto",
                    extent=[rng_lower, rng_upper, az_lower, az_upper],
                    cmap=cmap,
                    norm=norm,
                )
                ax.set_xlabel("Range (km)")
                ax.set_ylabel("Azimuth (deg)")
            else:
                lon_grid, lat_grid = lonlat
                lon_min = float(np.nanmin(lon_grid))
                lon_max = float(np.nanmax(lon_grid))
                lat_min = float(np.nanmin(lat_grid))
                lat_max = float(np.nanmax(lat_grid))
                img = ax.pcolormesh(
                    lon_grid,
                    lat_grid,
                    plot_data["cnn_output"][0, ..., 0],
                    shading="nearest",
                    cmap=cmap,
                    norm=norm,
                )
                levels = _contour_levels(plot_data["cnn_output"][0, ..., 0])
                if levels is not None:
                    ax.contour(
                        lon_grid,
                        lat_grid,
                        plot_data["cnn_output"][0, ..., 0],
                        levels=levels,
                        colors="black",
                        linewidths=0.6,
                    )
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlim(lon_min, lon_max)
                ax.set_ylim(lat_min, lat_max)
            ax.set_title(f"{file_tag} | Tornado Likelihood")
            fig.colorbar(img, ax=ax, shrink=0.8, label="Likelihood")
            fig.tight_layout()
            out_path = out_dir / f"{file_tag}_likelihood.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            saved += 1
        sample_index += batch_size

    logging.info("Saved %d likelihood map sample(s) to %s", saved, out_dir)


def _plot_metric_curves(
    metrics: Dict[str, float],
    probs: torch.Tensor,
    labels: torch.Tensor,
    out_dir: Path,
    model_label: str = "TorSight",
):
    if plt is None:  # pragma: no cover - optional dependency
        logging.warning("matplotlib unavailable; skipping metric plots.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        fpr, tpr, _ = binary_roc(probs, labels)
    except Exception as exc:  # pragma: no cover - safety net
        logging.warning("Could not compute ROC curve: %s", exc)
        fpr, tpr = None, None

    if fpr is not None and tpr is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr.cpu(), tpr.cpu(), color="#1f77b4", label=model_label)
        ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)
        tvs_auc = 0.6308
        ax.scatter(
            [tvs_auc],
            [tvs_auc],
            marker="v",
            color="black",
            label="TVS",
            zorder=3,
        )
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        title = "ROC Curve"
        ax.set_title(title)
        ax.legend(loc="lower right")
        fig.tight_layout()
        out_path = out_dir / "roc_auc.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logging.info("Saved ROC plot to %s", out_path)

    try:
        precision, recall, _ = binary_precision_recall_curve(probs, labels)
    except Exception as exc:  # pragma: no cover - safety net
        logging.warning("Could not compute precision-recall curve: %s", exc)
        precision, recall = None, None

    if precision is not None and recall is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(recall.cpu(), precision.cpu(), color="#2ca02c", label=model_label)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        title = "Precision-Recall Curve"
        ax.set_title(title)
        ax.legend(loc="lower left")
        fig.tight_layout()
        out_path = out_dir / "pr_aucpd.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logging.info("Saved AUCPD plot to %s", out_path)

        precision_t = precision.cpu().numpy()
        recall_t = recall.cpu().numpy()

        tvs_csi = 0.2002
        tvs_sr = 2.0 / (1.0 / tvs_csi + 1.0)
        tvs_tpr = tvs_sr

        sr_vals = np.linspace(0.01, 1.0, 100)
        pod_vals = np.linspace(0.01, 1.0, 100)
        sr_grid, pod_grid = np.meshgrid(sr_vals, pod_vals)
        denom = (1.0 / sr_grid) + (1.0 / pod_grid) - 1.0
        csi_grid = np.where(denom > 0, 1.0 / denom, np.nan)

        fig, ax = plt.subplots(figsize=(6, 4))
        levels = np.linspace(0.0, 1.0, 11)
        contour = ax.contourf(sr_grid, pod_grid, csi_grid, levels=levels, cmap="Blues")
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label("Critical Success Index (CSI)")
        ax.plot(precision_t, recall_t, color="#d62728", label=model_label)
        ax.scatter([tvs_sr], [tvs_tpr], marker="v", color="black", label="TVS", zorder=3)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Success Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Performance Diagram")
        ax.legend(loc="upper right")
        fig.tight_layout()
        out_path = out_dir / "csi_performance.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logging.info("Saved CSI plot to %s", out_path)

def _plot_success_case(
    sample_path: Path | None,
    variables: List[str],
    out_dir: Path,
    label: str,
    non_retagged_root: Path,
    retagged_root: Path | None = None,
    file_tag: str | None = None,
):
    if plt is None:  # pragma: no cover - optional dependency
        logging.warning("matplotlib unavailable; skipping success-case plot.")
        return
    try:
        from tornet.display.display import plot_radar
    except Exception as exc:  # pragma: no cover - optional dependency
        logging.warning("tornet.display.display unavailable; skipping success-case plot: %s", exc)
        return

    if not variables:
        logging.warning("No variables available to plot for success-case.")
        return
    if sample_path is None:
        logging.warning("No file path available for success-case plot.")
        return

    def _plot_from_data(
        plot_data: Dict[str, np.ndarray],
        filename_suffix: str,
        title_suffix: str,
        frame_idx: int,
    ) -> None:
        n_tilts = plot_data[variables[0]].shape[-1] if plot_data[variables[0]].ndim >= 3 else 1
        chosen_tilt = max(0, min(n_tilts - 1, 0))
        sweep_idx = [chosen_tilt] * len(variables)
        n_rows, n_cols = _grid_for_channels(len(variables))

        fig = plt.figure(figsize=(12, 6), edgecolor="k")
        plot_radar(
            plot_data,
            channels=list(variables),
            fig=fig,
            time_idx=0,
            sweep_idx=sweep_idx,
            include_cbar=True,
            n_rows=n_rows,
            n_cols=n_cols,
        )
        fig.suptitle(f"{label}{title_suffix} | frame {frame_idx}", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        out_dir.mkdir(parents=True, exist_ok=True)
        tag = file_tag or label
        out_path = out_dir / f"success_case_{tag}{filename_suffix}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logging.info("Saved success-case plot to %s", out_path)

    if not non_retagged_root.exists():
        logging.warning("Non-retagged root does not exist: %s", non_retagged_root)
        return

    non_retagged_path = _map_to_non_retagged(sample_path, non_retagged_root, retagged_root)
    if non_retagged_path is None:
        logging.warning("No non-retagged match found for %s", sample_path)
        non_retagged_path = None

    torn_time_idx = None
    if non_retagged_path is not None:
        torn_time_idx = _find_tornadic_time_index(non_retagged_path)
        if torn_time_idx is None:
            logging.warning("No tornadic frame_labels found in %s", non_retagged_path)

    if non_retagged_path is not None and torn_time_idx is not None:
        pre_time_idx = max(0, torn_time_idx - 1)
        try:
            pre_data = _load_plot_data(
                non_retagged_path,
                variables,
                pre_time_idx,
                tilt_last=True,
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning("Failed to load pre-tornadic slice from %s: %s", non_retagged_path, exc)
            pre_data = {}
        if pre_data:
            _plot_from_data(pre_data, "", " (pre-tornadic)", pre_time_idx)
        else:
            logging.warning("No variables available for pre-tornadic plot from %s", non_retagged_path)
    else:
        try:
            fallback_data = read_file(str(sample_path), variables=variables, tilt_last=True)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Failed to load success-case sample %s: %s", sample_path, exc)
            return
        time_len = fallback_data[variables[0]].shape[0]
        time_idx = max(0, time_len - 1)
        _plot_from_data(fallback_data, "", "", time_idx)

    if non_retagged_path is None or torn_time_idx is None:
        return

    try:
        plot_data = _load_plot_data(
            non_retagged_path,
            variables,
            torn_time_idx,
            tilt_last=True,
        )
    except Exception as exc:  # noqa: BLE001
        logging.warning("Failed to load tornadic slice from %s: %s", non_retagged_path, exc)
        return

    if not plot_data:
        logging.warning("No variables available to plot for %s", non_retagged_path)
        return

    _plot_from_data(plot_data, "_actual", " (actual)", torn_time_idx)


def _plot_success_likelihood(
    classifier: TornadoClassifier,
    sample_path: Path | None,
    variables: List[str],
    out_dir: Path,
    label: str,
    non_retagged_root: Path,
    retagged_root: Path | None,
    include_range_folded: bool,
    tilt_last: bool,
    device: torch.device,
):
    if plt is None:  # pragma: no cover - optional dependency
        logging.warning("matplotlib unavailable; skipping likelihood plot.")
        return
    try:
        from tornet.display.display import plot_radar, get_cmap
    except Exception as exc:  # pragma: no cover - optional dependency
        logging.warning("tornet.display.display unavailable; skipping likelihood plot: %s", exc)
        return

    if sample_path is None:
        logging.warning("No file path available for likelihood plot.")
        return

    if not non_retagged_root.exists():
        logging.warning("Non-retagged root does not exist: %s", non_retagged_root)
        return

    non_retagged_path = _map_to_non_retagged(sample_path, non_retagged_root, retagged_root)
    if non_retagged_path is None:
        logging.warning("No non-retagged match found for %s", sample_path)
        return

    torn_time_idx = _find_tornadic_time_index(non_retagged_path)
    if torn_time_idx is None:
        logging.warning("No tornadic frame_labels found in %s", non_retagged_path)
        return

    pre_time_idx = max(0, torn_time_idx - 1)
    include_range_folded = getattr(
        classifier.model, "include_range_folded", include_range_folded
    )
    coord_shape = getattr(classifier.model, "c_shape", None)
    if coord_shape:
        coord_channels = coord_shape[-1] if tilt_last else coord_shape[0]
        include_az = coord_channels >= 3
    else:
        include_az = True
    try:
        model_input, target_h, target_w, plot_meta = _build_model_input(
            non_retagged_path,
            variables,
            pre_time_idx,
            tilt_last=tilt_last,
            include_range_folded=include_range_folded,
            include_az=include_az,
        )
    except Exception as exc:  # noqa: BLE001
        logging.warning("Failed to build model input for likelihood plot: %s", exc)
        return

    model_input = _move_to_device(model_input, device)
    with torch.no_grad():
        logits = classifier.model(model_input)
        probs = torch.sigmoid(logits)
        prob_resized = F.interpolate(
            probs,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )[0, 0].detach().cpu().numpy()

    plot_data: Dict[str, np.ndarray] = dict(plot_meta)
    plot_data["cnn_output"] = prob_resized[None, ..., None]

    lonlat = _latlon_grid(plot_data, target_h, target_w)
    if lonlat is None:
        fig = plt.figure(figsize=(12, 6), edgecolor="k")
        plot_radar(
            plot_data,
            channels=["cnn_output"],
            fig=fig,
            time_idx=0,
            sweep_idx=[0],
            include_cbar=True,
            n_rows=1,
            n_cols=1,
        )
        fig.suptitle(f"{label} | Tornado Likelihood", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        lon_grid, lat_grid = lonlat
        lon_min = float(np.nanmin(lon_grid))
        lon_max = float(np.nanmax(lon_grid))
        lat_min = float(np.nanmin(lat_grid))
        lat_max = float(np.nanmax(lat_grid))
        lon_min = float(np.nanmin(lon_grid))
        lon_max = float(np.nanmax(lon_grid))
        lat_min = float(np.nanmin(lat_grid))
        lat_max = float(np.nanmax(lat_grid))
        fig, ax = plt.subplots(figsize=(8, 6))
        cmap, norm = get_cmap("cnn_output")
        mesh = ax.pcolormesh(
            lon_grid,
            lat_grid,
            prob_resized,
            shading="nearest",
            cmap=cmap,
            norm=norm,
        )
        levels = _contour_levels(prob_resized)
        if levels is not None:
            ax.contour(
                lon_grid,
                lat_grid,
                prob_resized,
                levels=levels,
                colors="black",
                linewidths=0.6,
            )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_title(f"{label} | Tornado Likelihood")
        fig.colorbar(mesh, ax=ax, shrink=0.8, label="Likelihood")
        fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"success_case_{label}_likelihood.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logging.info("Saved likelihood plot to %s", out_path)

    if not variables:
        return

    ablation_maps: Dict[str, np.ndarray] = {}
    raw_maps: Dict[str, np.ndarray] = {}
    base_prob = probs

    def _extract_raw_map(var_name: str) -> np.ndarray | None:
        if var_name not in model_input:
            return None
        arr = model_input[var_name]
        if not torch.is_tensor(arr):
            return None
        arr = arr.detach().cpu()
        if arr.dim() == 4:
            if tilt_last:
                return arr[0, :, :, 0].numpy()
            return arr[0, 0, :, :].numpy()
        if arr.dim() == 3:
            if tilt_last:
                return arr[:, :, 0].numpy()
            return arr[0, :, :].numpy()
        return None

    def _ablation_prob(var_name: str) -> np.ndarray | None:
        if var_name not in model_input:
            return None
        ablated = {
            key: (val.clone() if torch.is_tensor(val) else val)
            for key, val in model_input.items()
        }
        ablated[var_name] = torch.zeros_like(ablated[var_name])
        with torch.no_grad():
            ablated_logits = classifier.model(ablated)
            ablated_probs = torch.sigmoid(ablated_logits)
            delta = base_prob - ablated_probs
            delta_resized = F.interpolate(
                delta,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )[0, 0].detach().cpu().numpy()
        return delta_resized

    for var in variables:
        delta_map = _ablation_prob(var)
        if delta_map is not None:
            ablation_maps[var] = delta_map
        raw_map = _extract_raw_map(var)
        if raw_map is not None:
            raw_maps[var] = raw_map

    if not ablation_maps:
        return

    cmap_names = [
        "Reds",
        "Blues",
        "Greens",
        "Oranges",
        "Purples",
        "Greys",
        "YlGn",
        "PuRd",
    ]
    lonlat = _latlon_grid(plot_data, target_h, target_w)
    if lonlat is None:
        az_lower = float(np.asarray(plot_data.get("az_lower", [0.0]))[0])
        az_upper = float(np.asarray(plot_data.get("az_upper", [360.0]))[0])
        rng_lower = float(np.asarray(plot_data.get("rng_lower", [0.0]))[0]) / 1e3
        rng_upper = float(np.asarray(plot_data.get("rng_upper", [1.0]))[0]) / 1e3
        if az_upper <= az_lower:
            az_lower, az_upper = 0.0, 360.0
        if rng_upper <= rng_lower:
            rng_lower, rng_upper = 0.0, 1.0
        extent = [rng_lower, rng_upper, az_lower, az_upper]
    else:
        lon_grid, lat_grid = lonlat

    n_panels = 1 + 2 * len(ablation_maps)
    n_rows, n_cols = _grid_for_channels(n_panels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    base_img = plot_data["cnn_output"][0, ..., 0]
    if lonlat is None:
        axes[0].imshow(
            base_img,
            origin="lower",
            aspect="auto",
            extent=extent,
        )
        axes[0].set_xlabel("Range (km)")
        axes[0].set_ylabel("Azimuth (deg)")
    else:
        axes[0].pcolormesh(
            lon_grid,
            lat_grid,
            base_img,
            shading="nearest",
        )
        axes[0].set_xlabel("Longitude")
        axes[0].set_ylabel("Latitude")
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].set_xlim(lon_min, lon_max)
        axes[0].set_ylim(lat_min, lat_max)
        levels = _contour_levels(base_img)
        if levels is not None:
            axes[0].contour(
                lon_grid,
                lat_grid,
                base_img,
                levels=levels,
                colors="black",
                linewidths=0.6,
            )
    axes[0].set_title("Baseline", color="#1f77b4")

    panel_idx = 1
    for var, delta_map in ablation_maps.items():
        raw_map = raw_maps.get(var)
        cmap_var, norm_var = get_cmap(var)
        if raw_map is not None:
            if lonlat is None:
                axes[panel_idx].imshow(
                    raw_map,
                    origin="lower",
                    aspect="auto",
                    extent=extent,
                    cmap=cmap_var,
                    norm=norm_var,
                )
                axes[panel_idx].set_xlabel("Range (km)")
                axes[panel_idx].set_ylabel("Azimuth (deg)")
            else:
                axes[panel_idx].pcolormesh(
                    lon_grid,
                    lat_grid,
                    raw_map,
                    shading="nearest",
                    cmap=cmap_var,
                    norm=norm_var,
                )
                axes[panel_idx].set_xlabel("Longitude")
                axes[panel_idx].set_ylabel("Latitude")
                axes[panel_idx].set_aspect("equal", adjustable="box")
                axes[panel_idx].set_xlim(lon_min, lon_max)
                axes[panel_idx].set_ylim(lat_min, lat_max)
            axes[panel_idx].set_title(f"{var} raw", color="#333333")
            panel_idx += 1

        cmap_name = cmap_names[(panel_idx - 2) % len(cmap_names)]
        cmap = plt.get_cmap(cmap_name)
        if lonlat is None:
            img = axes[panel_idx].imshow(
                delta_map,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap=cmap,
            )
            axes[panel_idx].set_xlabel("Range (km)")
            axes[panel_idx].set_ylabel("Azimuth (deg)")
        else:
            img = axes[panel_idx].pcolormesh(
                lon_grid,
                lat_grid,
                delta_map,
                shading="nearest",
                cmap=cmap,
            )
            max_abs = float(np.nanmax(np.abs(delta_map))) if np.isfinite(delta_map).any() else 0.0
            if max_abs > 0:
                axes[panel_idx].contour(
                    lon_grid,
                    lat_grid,
                    delta_map,
                    levels=np.linspace(-max_abs, max_abs, 5),
                    colors="black",
                    linewidths=0.6,
                )
            axes[panel_idx].set_xlabel("Longitude")
            axes[panel_idx].set_ylabel("Latitude")
            axes[panel_idx].set_aspect("equal", adjustable="box")
            axes[panel_idx].set_xlim(lon_min, lon_max)
            axes[panel_idx].set_ylim(lat_min, lat_max)
        axes[panel_idx].set_title(f"{var} delta", color=cmap(0.7))
        fig.colorbar(img, ax=axes[panel_idx], shrink=0.8, label="Delta prob")
        panel_idx += 1

    for idx in range(panel_idx, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(f"{label} | Variable Ablation Deltas", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    ablation_out = out_dir / f"success_case_{label}_likelihood_ablation.png"
    fig.savefig(ablation_out, dpi=150)
    plt.close(fig)
    logging.info("Saved likelihood ablation plot to %s", ablation_out)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    device = _select_device(args.device)
    logging.info("Evaluating on device: %s", device)
    if "tfds" in args.dataloader and "TFDS_DATA_DIR" in os.environ:
        logging.info("Using TFDS dataset at %s", os.environ["TFDS_DATA_DIR"])

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else _find_default_checkpoint()
    if args.use_best_optuna:
        best_params = _normalize_optuna_params(
            _load_best_optuna_params(args.optuna_storage, args.optuna_study)
        )
        exp_root = Path(os.environ.get("EXP_DIR", ".")).resolve()
        optuna_checkpoint = _find_optuna_checkpoint(best_params, exp_root)
        if optuna_checkpoint is None:
            raise RuntimeError(
                "Could not locate a checkpoint for the best Optuna trial. "
                "Ensure experiment directories exist under EXP_DIR or pass --checkpoint."
            )
        if args.checkpoint:
            logging.warning(
                "Overriding --checkpoint with best Optuna checkpoint: %s",
                optuna_checkpoint,
            )
        checkpoint_path = optuna_checkpoint

    if not checkpoint_path or not checkpoint_path.exists():
        raise RuntimeError(
            "No checkpoint provided and no default checkpoint found. "
            "Specify one with --checkpoint."
        )
    train_cfg = _load_training_config(checkpoint_path)
    if args.use_best_optuna:
        train_cfg.update(best_params)
        _log_config_parity(train_cfg, best_params)

    input_variables = train_cfg.get("input_variables", ALL_VARIABLES)
    dataloader_kwargs_cfg = train_cfg.get("dataloader_kwargs", {})
    select_keys = train_cfg.get(
        "dataloader_keys",
        input_variables + ["range_folded_mask", "coordinates"],
    )
    if select_keys is not None:
        select_keys = list(select_keys)
        if "ef_number" not in select_keys:
            select_keys.append("ef_number")
    dataloader_kwargs = {
        "tilt_last": dataloader_kwargs_cfg.get("tilt_last", args.tilt_last),
        "select_keys": select_keys,
    }
    weights = None
    batch_size = int(train_cfg.get("batch_size", args.batch_size))

    ds = get_dataloader(
        args.dataloader,
        DATA_ROOT,
        years=list(range(2013, 2023)),
        data_type="test",
        batch_size=batch_size,
        weights=weights,
        **dataloader_kwargs,
    )
    _drop_missing_files(ds, "test")
    ds = _wrap_loader(ds)
    dataset = getattr(ds, "dataset", None)
    file_list: List[str] | None = getattr(dataset, "file_list", None)
    can_catalog = file_list is not None and "tfds" not in args.dataloader
    if not can_catalog:
        logging.info("Per-sample cataloging disabled (no file_list available for this dataloader).")

    sample_batch = next(iter(ds))
    input_shape, coord_shape = _infer_input_shapes(sample_batch, input_variables)
    include_range_folded = args.include_range_folded or "range_folded_mask" in sample_batch

    start_filters = int(train_cfg.get("start_filters", 48))
    kernel_size = int(train_cfg.get("kernel_size", 3))
    n_blocks = int(train_cfg.get("n_blocks", 4))
    convs_per_block = train_cfg.get("convs_per_block")
    drop_rate = float(train_cfg.get("drop_rate", 0.1))

    likelihood = TornadoLikelihood(
        shape=input_shape,
        c_shape=coord_shape,
        input_variables=input_variables,
        include_range_folded=include_range_folded,
        start_filters=start_filters,
        kernel_size=kernel_size,
        n_blocks=n_blocks,
        convs_per_block=convs_per_block,
        drop_rate=drop_rate,
    )
    classifier = TornadoClassifier(
        model=likelihood,
        metrics=_build_metrics(device),
    )
    logging.info("Loading checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    classifier.load_state_dict(checkpoint["state_dict"])
    classifier.to(device)
    classifier.eval()

    metric_collection = _build_metrics(device)
    stat_scores = BinaryStatScores(threshold=0.5).to(device)
    total_loss = 0.0
    total_count = 0
    false_catalog = {"false_positives": [], "false_negatives": []} if can_catalog else None
    sample_index = 0
    all_probs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    success_path: Path | None = None
    success_prob = -1.0
    success_ef = -1.0
    success_label = "success"
    weak_success_path: Path | None = None
    weak_success_prob = float("inf")
    weak_success_ef = float("inf")
    weak_success_label = "weak_success"
    weak_success_file_tag = None

    with torch.no_grad():
        for batch in ds:
            batch = _move_to_device(batch, device)
            labels = torch.squeeze(batch.pop("label")).long()
            logits = classifier.model(batch)
            logits = F.max_pool2d(logits, kernel_size=logits.size()[2:])
            logits = torch.cat((-logits, logits), dim=1)
            logits = torch.squeeze(logits)
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)
                labels = labels.unsqueeze(0)
            loss = classifier.loss(logits, labels)
            total_loss += loss.item() * labels.shape[0]
            total_count += labels.shape[0]
            probs = torch.sigmoid(logits[:, 1])
            preds = (probs >= 0.5).long()
            metric_collection.update(probs, labels)
            stat_scores.update(probs, labels)
            all_probs.append(probs.detach().cpu())
            all_labels.append(labels.detach().cpu())
            batch_size = labels.shape[0]
            if file_list and "ef_number" in batch:
                ef_values = batch.get("ef_number")
                if ef_values is not None:
                    ef_flat = torch.as_tensor(ef_values).detach().cpu().view(-1)
                    matches = ((preds == 1) & (labels == 1)).nonzero(as_tuple=False)
                    if matches.numel() > 0:
                        match_idx = matches.squeeze(1)
                        match_idx_cpu = match_idx.detach().cpu()
                        match_probs = probs[match_idx].detach().cpu()
                        match_ef = ef_flat[match_idx_cpu]
                        valid_mask = match_ef >= 0
                        if valid_mask.any():
                            match_probs = match_probs[valid_mask]
                            match_ef = match_ef[valid_mask]
                            match_idx_cpu = match_idx_cpu[valid_mask]
                        # Pick highest EF, break ties with probability
                        best_local = None
                        worst_local = None
                        for ef_val, prob_val, idx_val in zip(
                            match_ef.tolist(), match_probs.tolist(), match_idx_cpu.tolist()
                        ):
                            if best_local is None:
                                best_local = (ef_val, prob_val, idx_val)
                                worst_local = (ef_val, prob_val, idx_val)
                                continue
                            if ef_val > best_local[0] or (math.isclose(ef_val, best_local[0]) and prob_val > best_local[1]):
                                best_local = (ef_val, prob_val, idx_val)
                            if ef_val < worst_local[0] or (math.isclose(ef_val, worst_local[0]) and prob_val < worst_local[1]):
                                worst_local = (ef_val, prob_val, idx_val)
                        if best_local:
                            ef_val, prob_val, idx_val = best_local
                            global_idx = sample_index + int(idx_val)
                            if ef_val > success_ef or (math.isclose(ef_val, success_ef) and prob_val > success_prob):
                                success_ef = ef_val
                                success_prob = prob_val
                                success_path = Path(file_list[global_idx])
                                success_label = f"{_safe_filename(file_list, global_idx)}_EF{int(ef_val)}"
                        if worst_local:
                            ef_val, prob_val, idx_val = worst_local
                            global_idx = sample_index + int(idx_val)
                            if ef_val < weak_success_ef or (math.isclose(ef_val, weak_success_ef) and prob_val < weak_success_prob):
                                weak_success_ef = ef_val
                                weak_success_prob = prob_val
                                weak_success_path = Path(file_list[global_idx])
                                weak_success_label = f"{_safe_filename(file_list, global_idx)}_EF{int(ef_val)}"
                                weak_success_file_tag = f"{weak_success_label}_weak"
            if false_catalog:
                for i in range(batch_size):
                    idx = sample_index + i
                    if idx >= len(file_list):  # type: ignore[arg-type]
                        continue
                    path = file_list[idx]  # type: ignore[index]
                    label = int(labels[i].cpu())
                    pred = int(preds[i].cpu())
                    prob = float(probs[i].cpu())
                    if pred == 1 and label == 0:
                        false_catalog["false_positives"].append(
                            {"file": path, "prob": prob, "label": label, "pred": pred}
                        )
                    elif pred == 0 and label == 1:
                        false_catalog["false_negatives"].append(
                            {"file": path, "prob": prob, "label": label, "pred": pred}
                        )
            sample_index += batch_size

    metrics = metric_collection.compute()
    tp, fp, tn, fn, _ = (s.item() for s in stat_scores.compute())
    csi = _compute_csi(tp, fp, fn)
    avg_loss = total_loss / max(total_count, 1)
    metrics = {"Loss": avg_loss, **{k: float(v.cpu()) for k, v in metrics.items()}}
    if csi is not None:
        metrics["CSI"] = csi
    logging.info("Evaluation metrics: %s", metrics)
    eval_out_dir = checkpoint_path.resolve().parent.parent
    if all_probs and all_labels:
        probs_all = torch.cat(all_probs)
        labels_all = torch.cat(all_labels)
        _plot_metric_curves(metrics, probs_all, labels_all, eval_out_dir)
    if success_path is not None:
        _plot_success_case(
            success_path,
            input_variables,
            eval_out_dir,
            success_label,
            non_retagged_root=args.non_retagged_root,
            retagged_root=DATA_ROOT_PATH,
        )
        _plot_success_likelihood(
            classifier=classifier,
            sample_path=success_path,
            variables=input_variables,
            out_dir=eval_out_dir,
            label=success_label,
            non_retagged_root=args.non_retagged_root,
            retagged_root=DATA_ROOT_PATH,
            include_range_folded=include_range_folded,
            tilt_last=dataloader_kwargs["tilt_last"],
            device=device,
        )
    if weak_success_path is not None:
        _plot_success_case(
            weak_success_path,
            input_variables,
            eval_out_dir,
            weak_success_label,
            non_retagged_root=args.non_retagged_root,
            retagged_root=DATA_ROOT_PATH,
            file_tag=weak_success_file_tag,
        )
    if false_catalog:
        out_path = eval_out_dir / "false_cases.json"
        with open(out_path, "w") as f:
            json.dump(false_catalog, f, indent=2)
        logging.info(
            "Cataloged %d false positives and %d false negatives to %s",
            len(false_catalog["false_positives"]),
            len(false_catalog["false_negatives"]),
            out_path,
        )
    if args.grad_cam:
        grad_cam_out_dir = args.grad_cam_output or eval_out_dir
        _generate_grad_cam_plots(
            classifier=classifier,
            loader=ds,
            variables=input_variables,
            tilt_last=dataloader_kwargs["tilt_last"],
            out_dir=grad_cam_out_dir,
            max_samples=args.grad_cam_samples,
            file_list=file_list,
            tilt_index=args.grad_cam_tilt_index,
            device=device,
            non_retagged_root=args.non_retagged_root,
            retagged_root=DATA_ROOT_PATH,
        )


if __name__ == "__main__":
    main()
