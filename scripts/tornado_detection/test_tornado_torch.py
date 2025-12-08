"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

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

    ds = get_dataloader(
        args.dataloader,
        DATA_ROOT,
        years=list(range(2013, 2023)),
        data_type="test",
        batch_size=args.batch_size,
        weights=weights,
        **dataloader_kwargs,
    )
    _drop_missing_files(ds, "test")
    ds = _wrap_loader(ds)

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

    metrics = metric_collection.compute()
    tp, fp, tn, fn, _ = (s.item() for s in stat_scores.compute())
    csi = _compute_csi(tp, fp, fn)
    avg_loss = total_loss / max(total_count, 1)
    metrics = {"Loss": avg_loss, **{k: float(v.cpu()) for k, v in metrics.items()}}
    if csi is not None:
        metrics["CSI"] = csi
    logging.info("Evaluation metrics: %s", metrics)


if __name__ == "__main__":
    main()
