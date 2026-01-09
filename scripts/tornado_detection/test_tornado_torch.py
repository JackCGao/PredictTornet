from __future__ import annotations

import argparse
import json
import logging
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

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tornet.data.constants import ALL_VARIABLES
from tornet.data.loader import get_dataloader
from tornet.models.torch.cnn_baseline import TornadoClassifier, TornadoLikelihood

logging.basicConfig(level=logging.INFO)

DATA_ROOT = os.environ["TORNET_ROOT"]
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
    parser.add_argument(
        "--use-best-optuna",
        action="store_true",
        help="Load the best trial params from the Optuna study and apply them for evaluation.",
    )
    parser.add_argument(
        "--optuna-storage",
        default="sqlite:///tornet_optuna.db",
        help="Optuna storage URL (default: sqlite:///tornet_optuna.db).",
    )
    parser.add_argument(
        "--optuna-study",
        default="tornet_optuna",
        help="Optuna study name (default: tornet_optuna).",
    )
    parser.add_argument(
        "--grad-cam",
        action="store_true",
        help="If set, generate Grad-CAM plots for a subset of samples.",
    )
    parser.add_argument(
        "--grad-cam-output",
        type=Path,
        default=Path("grad_cam"),
        help="Directory to save Grad-CAM plots (default: grad_cam).",
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
    """Load best trial parameters from an Optuna study."""

    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("optuna is required to load best trial params. Install with `pip install optuna`.") from exc

    study = optuna.load_study(study_name=study_name, storage=storage)
    best = study.best_trial
    logging.info("Loaded best Optuna trial %s (value=%s)", best.number, best.value)
    return dict(best.params)

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


def _prepare_plot_sample(
    batch: Dict[str, torch.Tensor], sample_idx: int, variables: Iterable[str], tilt_last: bool
) -> Dict[str, np.ndarray]:
    """Format a single sample for plot_radar (tilt-last with time dim)."""

    data: Dict[str, np.ndarray] = {}
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
):
    if plt is None:  # pragma: no cover - optional dependency
        logging.warning("matplotlib unavailable; skipping Grad-CAM plots.")
        return
    try:
        from tornet.display.display import plot_radar
    except Exception as exc:  # pragma: no cover - optional dependency
        logging.warning("tornet.display.display unavailable; skipping Grad-CAM plots: %s", exc)
        return

    grad_cam = _GradCAM(classifier.model.head[-1])
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

        grad_cam.clear()
        classifier.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = classifier.model(batch)
            pooled = F.max_pool2d(logits, kernel_size=logits.size()[2:])
            score = torch.squeeze(pooled)
            if score.ndim == 0:
                score = score.unsqueeze(0)
            score.sum().backward()
            cams = grad_cam.build_cam().detach().cpu()

        batch_size = cams.shape[0]
        for i in range(batch_size):
            if saved >= max_samples:
                break
            plot_data = _prepare_plot_sample(batch_for_plot, i, variables, tilt_last)
            cam_np = cams[i].numpy()
            plot_data["cnn_output"] = cam_np[None, ..., None]
            file_tag = _safe_filename(file_list, sample_index + i)

            tilt_for_plot = 0
            var_shape = plot_data[variables[0]].shape if variables else None
            if var_shape and len(var_shape) == 4:
                tilt_for_plot = min(max(tilt_index, 0), max(var_shape[-1] - 1, 0))

            for var in variables:
                fig = plt.figure(figsize=(10, 4), edgecolor="k")
                plot_radar(
                    plot_data,
                    channels=[var, "cnn_output"],
                    fig=fig,
                    time_idx=0,
                    sweep_idx=[tilt_for_plot, 0],
                    include_cbar=True,
                    n_rows=1,
                    n_cols=2,
                )
                fig.suptitle(f"{file_tag} | {var}", y=0.98)
                fig.tight_layout(rect=[0, 0, 1, 0.96])
                out_path = out_dir / f"{file_tag}_{var}_gradcam.png"
                fig.savefig(out_path, dpi=150)
                plt.close(fig)
            saved += 1
        sample_index += batch_size

    grad_cam.remove()
    logging.info("Saved %d Grad-CAM sample(s) to %s", saved, out_dir)


def _plot_metric_summary(metrics: Dict[str, float], out_dir: Path, model_label: str = "TorSight"):
    if plt is None:  # pragma: no cover - optional dependency
        logging.warning("matplotlib unavailable; skipping metric plot.")
        return
    targets = [("CSI", "CSI"), ("AUC", "AUC"), ("AUCPR", "AUCPD")]
    entries = [(label, float(metrics[key])) for key, label in targets if key in metrics]
    if not entries:
        logging.warning("No metrics available for plotting; skipping metric plot.")
        return

    labels, values = zip(*entries)
    x = np.arange(len(labels))
    width = 0.6

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(x, values, width, color="#1f77b4")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("TorSight Evaluation Metrics")
    ax.legend([bars[0]], [model_label], title="Model")

    for bar, val in zip(bars, values):
        ax.annotate(f"{val:.3f}", xy=(bar.get_x() + bar.get_width() / 2, val), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out_path = out_dir / "torsight_metrics.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logging.info("Saved metric plot to %s", out_path)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    device = _select_device(args.device)
    logging.info("Evaluating on device: %s", device)
    if "tfds" in args.dataloader and "TFDS_DATA_DIR" in os.environ:
        logging.info("Using TFDS dataset at %s", os.environ["TFDS_DATA_DIR"])

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else _find_default_checkpoint()
    if not checkpoint_path or not checkpoint_path.exists():
        raise RuntimeError(
            "No checkpoint provided and no default checkpoint found. "
            "Specify one with --checkpoint."
        )
    train_cfg = _load_training_config(checkpoint_path)
    if args.use_best_optuna:
        best_params = _load_best_optuna_params(args.optuna_storage, args.optuna_study)
        train_cfg.update(best_params)

    input_variables = train_cfg.get("input_variables", ALL_VARIABLES)
    dataloader_kwargs_cfg = train_cfg.get("dataloader_kwargs", {})
    dataloader_kwargs = {
        "tilt_last": dataloader_kwargs_cfg.get("tilt_last", args.tilt_last),
        "select_keys": train_cfg.get(
            "dataloader_keys",
            input_variables + ["range_folded_mask", "coordinates"],
        ),
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

    likelihood = TornadoLikelihood(
        shape=input_shape,
        c_shape=coord_shape,
        input_variables=input_variables,
        include_range_folded=include_range_folded,
        start_filters=start_filters,
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
            metric_collection.update(probs, labels)
            stat_scores.update(probs, labels)
            if false_catalog:
                batch_size = labels.shape[0]
                preds = (probs >= 0.5).long()
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
    _plot_metric_summary(metrics, eval_out_dir)
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
        _generate_grad_cam_plots(
            classifier=classifier,
            loader=ds,
            variables=input_variables,
            tilt_last=dataloader_kwargs["tilt_last"],
            out_dir=args.grad_cam_output,
            max_samples=args.grad_cam_samples,
            file_list=file_list,
            tilt_index=args.grad_cam_tilt_index,
            device=device,
        )


if __name__ == "__main__":
    main()
