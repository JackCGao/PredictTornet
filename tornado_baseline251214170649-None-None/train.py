from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, Tuple

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
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
)

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tornet.data.constants import ALL_VARIABLES
from tornet.data.loader import get_dataloader
from tornet.models.torch.cnn_baseline import TornadoClassifier, TornadoLikelihood
from tornet.utils.general import make_callback_dirs, make_exp_dir

logging.basicConfig(level=logging.INFO)

EXP_DIR = os.environ.get("EXP_DIR", ".")
TMP_DIR = Path(EXP_DIR).resolve() / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)
(TMP_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(TMP_DIR))
os.environ.setdefault("TMP", str(TMP_DIR))
os.environ.setdefault("TEMP", str(TMP_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(TMP_DIR / "matplotlib"))
if "TORNET_ROOT" not in os.environ:
    # Default to the local dataset path if the caller did not set TORNET_ROOT.
    os.environ["TORNET_ROOT"] = "/home/bgao/PredictTornet/tornet_raw/retagged_shift"
DATA_ROOT = os.environ["TORNET_ROOT"]
logging.info("TORNET_ROOT=%s", DATA_ROOT)
DEFAULT_PARAMS_PATH = Path(__file__).resolve().parent / "config" / "params.json"

DEFAULT_CONFIG = {
    "epochs": 10,
    "input_variables": ALL_VARIABLES,
    "train_years": list(range(2013, 2021)),
    "val_years": list(range(2021, 2023)),
    "batch_size": 128,
    "model": "vgg",
    "start_filters": 48,
    "learning_rate": 1e-4,
    "decay_steps": 1386,
    "decay_rate": 0.958,
    "l2_reg": 1e-5,
    "wN": 1.0,
    "w0": 1.0,
    "w1": 1.0,
    "w2": 2.0,
    "wW": 0.5,
    "label_smooth": 0.0,
    "head": "maxpool",
    "exp_name": "tornet_baseline_torch",
    "exp_dir": EXP_DIR,
    "dataloader": "torch",
    "dataloader_kwargs": {"tilt_last": False},
    "accelerator": "auto",
    "devices": 1,
    "precision": "32-true",
    "log_every_n_steps": 25,
}


def _clone_dict(d: Dict | None) -> Dict:
    return deepcopy(d) if d else {}


def _to_float32(data):
    if isinstance(data, torch.Tensor) and data.dtype == torch.float64:
        return data.float()
    if isinstance(data, dict):
        return {k: _to_float32(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(_to_float32(v) for v in data)
    return data


def _lightning_collate(batch):
    """Attach labels (and optional weights) to feature dicts for Lightning."""
    first = batch[0]
    has_weights = len(first) == 3
    features = [sample[0] for sample in batch]
    labels = [sample[1] for sample in batch]
    weights = [sample[2] for sample in batch] if has_weights else None

    features = default_collate(features)
    features["label"] = default_collate(labels)
    if has_weights and weights is not None:
        features["sample_weights"] = default_collate(weights)
    return _to_float32(features)


def _wrap_loader_for_lightning(loader: DataLoader) -> DataLoader:
    """
    LightningModule expects dictionaries; make_torch_loader returns tuples. Wrap the
    loader with a collate_fn that reattaches labels (and optional weights) to the dict.
    """

    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        num_workers=loader.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=loader.pin_memory,
        persistent_workers=(
            getattr(loader, "persistent_workers", False) and loader.num_workers > 0
        ),
        collate_fn=_lightning_collate,
    )


def _infer_input_shapes(
    batch: Dict[str, torch.Tensor], input_variables: Iterable[str]
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    feat_shape = None
    for var in input_variables:
        if var in batch:
            feat_shape = tuple(batch[var].shape[1:])
            break
    if feat_shape is None:
        raise RuntimeError("Unable to infer feature shape from batch.")
    coord_shape = tuple(batch["coordinates"].shape[1:])
    return feat_shape, coord_shape


def _build_metrics() -> MetricCollection:
    return MetricCollection(
        {
            "AUC": BinaryAUROC(),
            "AUCPR": BinaryAveragePrecision(),
            "BinaryAccuracy": BinaryAccuracy(),
            "Precision": BinaryPrecision(),
            "Recall": BinaryRecall(),
            "F1": BinaryF1Score(),
        }
    )


def _compute_csi(precision: float | None, recall: float | None) -> float | None:
    """Compute Critical Success Index from precision and recall."""

    if precision is None or recall is None:
        return None
    denom = precision + recall - (precision * recall)
    if denom <= 0:
        return None
    return (precision * recall) / denom


def _prepare_dataloader_kwargs(config: Dict) -> Dict:
    dataloader_kwargs = _clone_dict(config.get("dataloader_kwargs"))
    dataloader_kwargs.setdefault("tilt_last", False)
    selected_keys = config.get(
        "dataloader_keys",
        config.get("input_variables", ALL_VARIABLES)
        + ["range_folded_mask", "coordinates"],
    )
    dataloader_kwargs.setdefault("select_keys", selected_keys)
    return dataloader_kwargs


def _drop_missing_files(loader: DataLoader, loader_name: str) -> None:
    """
    Remove entries from the dataloader's file list if the files are missing on disk.

    Some catalogs may contain stale paths; we quietly skip them so training can
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


def _load_params_json(path: Path) -> Dict:
    """Load a params.json file that may wrap the config in a top-level 'config' key."""

    if not path.exists():
        logging.warning("params.json not found at %s; using defaults.", path)
        return {}
    try:
        data = json.load(open(path, "r"))
    except Exception as exc:  # noqa: BLE001
        logging.warning("Failed to read params.json at %s: %s; using defaults.", path, exc)
        return {}

    if isinstance(data, dict):
        cfg = data.get("config", data)
        return cfg if isinstance(cfg, dict) else {}
    return {}


def main(config: Dict, checkpoint: str | None = None):
    config = deepcopy(config)
    epochs = config.get("epochs")
    batch_size = config.get("batch_size")
    start_filters = config.get("start_filters")
    learning_rate = config.get("learning_rate")
    decay_steps = config.get("decay_steps")
    decay_rate = config.get("decay_rate")
    l2_reg = config.get("l2_reg")
    label_smooth = config.get("label_smooth")
    input_variables = config.get("input_variables")
    exp_name = config.get("exp_name")
    exp_dir = config.get("exp_dir")
    train_years = config.get("train_years")
    val_years = config.get("val_years")
    dataloader_name = config.get("dataloader")
    dataloader_kwargs = _prepare_dataloader_kwargs(config)

    logging.info("Using %s dataloader", dataloader_name)
    if "tfds" in dataloader_name and "TFDS_DATA_DIR" in os.environ:
        logging.info("Using TFDS dataset at %s", os.environ["TFDS_DATA_DIR"])

    weights = {
        "wN": config.get("wN"),
        "w0": config.get("w0"),
        "w1": config.get("w1"),
        "w2": config.get("w2"),
        "wW": config.get("wW"),
    }

    ds_train_raw = get_dataloader(
        dataloader_name,
        DATA_ROOT,
        train_years,
        "train",
        batch_size,
        weights,
        **dataloader_kwargs,
    )
    _drop_missing_files(ds_train_raw, "train")
    ds_val_raw = get_dataloader(
        dataloader_name,
        DATA_ROOT,
        val_years,
        "train",
        batch_size,
        weights,
        **dataloader_kwargs,
    )
    _drop_missing_files(ds_val_raw, "validation")
    ds_train = _wrap_loader_for_lightning(ds_train_raw)
    ds_val = _wrap_loader_for_lightning(ds_val_raw)

    sample_batch = next(iter(ds_train))
    input_shape, coord_shape = _infer_input_shapes(sample_batch, input_variables)
    include_range_folded = "range_folded_mask" in sample_batch

    model = TornadoLikelihood(
        shape=input_shape,
        c_shape=coord_shape,
        input_variables=input_variables,
        start_filters=start_filters,
        include_range_folded=include_range_folded,
    )
    metrics = _build_metrics()
    classifier = TornadoClassifier(
        model=model,
        lr=learning_rate,
        lr_decay_rate=decay_rate,
        lr_decay_steps=decay_steps,
        label_smoothing=label_smooth,
        weight_decay=l2_reg,
        metrics=metrics,
    )

    expdir = make_exp_dir(exp_dir=exp_dir, prefix=exp_name)
    logging.info("expdir=%s", expdir)
    with open(os.path.join(expdir, "data.json"), "w") as f:
        json.dump(
            {
                "data_root": DATA_ROOT,
                "train_data": list(train_years),
                "val_data": list(val_years),
            },
            f,
        )
    with open(os.path.join(expdir, "params.json"), "w") as f:
        json.dump({"config": config}, f)
    shutil.copy(__file__, os.path.join(expdir, "train.py"))

    tboard_dir, checkpoints_dir = make_callback_dirs(expdir)
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoints_dir,
            filename="tornadoDetector_{epoch:03d}",
            monitor="val_loss",
            save_last=True,
            save_top_k=1,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    tb_logger = TensorBoardLogger(save_dir=tboard_dir, name="torch")
    csv_logger = CSVLogger(save_dir=expdir, name="logs")

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator=config.get("accelerator", "auto"),
        devices=config.get("devices", "auto"),
        precision=config.get("precision", "32-true"),
        log_every_n_steps=config.get("log_every_n_steps", 25),
        default_root_dir=expdir,
        logger=[tb_logger, csv_logger],
        callbacks=callbacks,
    )
    trainer.fit(
        classifier,
        train_dataloaders=ds_train,
        val_dataloaders=ds_val,
        ckpt_path=checkpoint,
    )

    val_results = trainer.validate(ckpt_path="best", dataloaders=ds_val)
    best_metrics = val_results[0] if val_results else {}
    best_auc = float(best_metrics.get("val_AUC", 0.5))
    best_aucpr = float(best_metrics.get("val_AUCPR", 0.0))
    best_precision = best_metrics.get("val_Precision")
    best_recall = best_metrics.get("val_Recall")
    best_precision = float(best_precision) if best_precision is not None else None
    best_recall = float(best_recall) if best_recall is not None else None
    best_csi = _compute_csi(best_precision, best_recall)
    logging.info(
        "Best validation metrics: AUC=%f AUCPR=%f CSI=%s",
        best_auc,
        best_aucpr,
        f"{best_csi:.4f}" if best_csi is not None else "n/a",
    )
    return {"AUC": best_auc, "AUCPR": best_aucpr, "CSI": best_csi}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TorNet Torch model.")
    parser.add_argument(
        "--config",
        help="Path to a JSON config file to override defaults.",
    )
    parser.add_argument(
        "--checkpoint",
        help="Optional checkpoint to resume training from.",
    )
    args = parser.parse_args()

    cfg = deepcopy(DEFAULT_CONFIG)
    if args.config:
        cfg.update(json.load(open(args.config, "r")))
    elif args.checkpoint:
        params_path = Path(args.checkpoint).resolve().parent.parent / "params.json"
        cfg.update(_load_params_json(params_path))
    else:
        cfg.update(_load_params_json(DEFAULT_PARAMS_PATH))
    main(cfg, checkpoint=args.checkpoint)
