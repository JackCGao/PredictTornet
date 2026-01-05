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
import shutil
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
import numpy as np  # used only for optuna logging

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
    raise RuntimeError("Please set TORNET_ROOT to the dataset root before training.")
DATA_ROOT = os.environ["TORNET_ROOT"]
logging.info("TORNET_ROOT=%s", DATA_ROOT)

# Optional plotting support; falls back gracefully if matplotlib is unavailable.
try:  # pragma: no cover - optional dependency
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

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
    "kernel_size": 3,
    "n_blocks": 4,
    "convs_per_block": [2, 2, 3, 3],
    "drop_rate": 0.1,
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


def _load_default_config() -> Dict:
    """
    Load default config, preferring the JSON file if it exists so users can tweak
    params without passing a path each time.
    """

    config = deepcopy(DEFAULT_CONFIG)
    json_path = Path(__file__).with_name("config") / "params.json"
    if json_path.exists():
        try:
            with open(json_path, "r") as f:
                config.update(json.load(f))
                logging.info("Loaded default config from %s", json_path)
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("Failed to load %s: %s; falling back to DEFAULT_CONFIG", json_path, exc)
    return config


def _suggest_config(trial, base_config: Dict) -> Dict:
    """
    Suggest hyperparameters for Optuna search based on keys present in the config.
    Only parameters found in params.json/defaults are sampled.
    """
    cfg = deepcopy(base_config)
    if "epochs" in cfg:
        cfg["epochs"] = trial.suggest_int("epochs", 5, 15)
    if "batch_size" in cfg:
        # Keep choices stable across study runs to avoid Optuna "dynamic value space" errors.
        cfg["batch_size"] = trial.suggest_categorical("batch_size", [64, 96, 128, 192, 256])
    if "start_filters" in cfg:
        # Single choice; study name updated when adjusting space to avoid mismatch.
        cfg["start_filters"] = trial.suggest_categorical("start_filters", [32])
    if "learning_rate" in cfg:
        cfg["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    if "decay_steps" in cfg:
        cfg["decay_steps"] = trial.suggest_int("decay_steps", 500, 2500)
    if "decay_rate" in cfg:
        cfg["decay_rate"] = trial.suggest_float("decay_rate", 0.90, 0.999)
    if "l2_reg" in cfg:
        cfg["l2_reg"] = trial.suggest_float("l2_reg", 1e-6, 1e-3, log=True)
    if "wN" in cfg:
        cfg["wN"] = trial.suggest_float("wN", 0.3, 2.5)
    if "w0" in cfg:
        cfg["w0"] = trial.suggest_float("w0", 0.8, 3.0)
    if "w1" in cfg:
        cfg["w1"] = trial.suggest_float("w1", 0.8, 3.0)
    if "w2" in cfg:
        cfg["w2"] = trial.suggest_float("w2", 1.0, 4.0)
    if "wW" in cfg:
        cfg["wW"] = trial.suggest_float("wW", 0.3, 2.0)
    if "label_smooth" in cfg:
        cfg["label_smooth"] = trial.suggest_float("label_smooth", 0.0, 0.1)
    if "head" in cfg:
        cfg["head"] = trial.suggest_categorical("head", ["maxpool", "avgpool"])
    if "loss" in cfg:
        cfg["loss"] = trial.suggest_categorical("loss", ["cce", "bce"])
    if "kernel_size" in cfg:
        cfg["kernel_size"] = trial.suggest_categorical("kernel_size", [3, 5])
    if "n_blocks" in cfg:
        cfg["n_blocks"] = trial.suggest_int("n_blocks", 4, 4)
    if "convs_per_block" in cfg:
        cfg["convs_per_block"] = [
            trial.suggest_int("convs_block1", 1, 3),
            trial.suggest_int("convs_block2", 1, 3),
            trial.suggest_int("convs_block3", 2, 4),
            trial.suggest_int("convs_block4", 2, 4),
        ]
    if "drop_rate" in cfg:
        cfg["drop_rate"] = trial.suggest_float("drop_rate", 0.1, 0.4)
    return cfg


def _clone_dict(d: Dict | None) -> Dict:
    return deepcopy(d) if d else {}


def _get_metric(metrics: Dict, keys: List[str]):
    """Return the first matching key from metrics, or None."""

    for key in keys:
        if key in metrics:
            return metrics[key]
    return None


@dataclass
class PlotMetricsCallback(L.Callback):
    """
    Collect train/val loss and AUC per epoch and emit a summary plot at fit end.
    """

    out_path: Path
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_auc: List[float] = field(default_factory=list)
    val_auc: List[float] = field(default_factory=list)

    def on_train_epoch_end(self, trainer: L.Trainer, *_):
        metrics = trainer.callback_metrics
        loss = _get_metric(metrics, ["train_loss", "train_loss_epoch"])
        auc = _get_metric(metrics, ["train_AUC", "train_AUC_epoch"])
        self.train_loss.append(float(loss) if loss is not None else float("nan"))
        self.train_auc.append(float(auc) if auc is not None else float("nan"))

    def on_validation_epoch_end(self, trainer: L.Trainer, *_):
        metrics = trainer.callback_metrics
        loss = _get_metric(metrics, ["val_loss", "val_loss_epoch"])
        auc = _get_metric(metrics, ["val_AUC", "val_AUC_epoch"])
        self.val_loss.append(float(loss) if loss is not None else float("nan"))
        self.val_auc.append(float(auc) if auc is not None else float("nan"))

    def on_fit_end(self, trainer: L.Trainer, *_):
        if plt is None:
            logging.warning("matplotlib not available; skipping metric plot.")
            return
        epochs = list(range(1, max(len(self.train_loss), len(self.val_loss)) + 1))
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        ax[0].plot(epochs[: len(self.train_loss)], self.train_loss, label="train_loss")
        ax[0].plot(epochs[: len(self.val_loss)], self.val_loss, label="val_loss")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].legend()
        ax[0].set_title("Loss")

        ax[1].plot(epochs[: len(self.train_auc)], self.train_auc, label="train_AUC")
        ax[1].plot(epochs[: len(self.val_auc)], self.val_auc, label="val_AUC")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("AUC")
        ax[1].legend()
        ax[1].set_title("AUC")

        fig.tight_layout()
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.out_path)
        plt.close(fig)
        logging.info("Saved training curves to %s", self.out_path)


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
    # Keep loader lightweight by default to reduce host memory pressure.
    dataloader_kwargs.setdefault("workers", 21)
    dataloader_kwargs.setdefault("pin_memory", False)
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


def main(config: Dict):
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
        kernel_size=config.get("kernel_size", 3),
        n_blocks=config.get("n_blocks", 4),
        convs_per_block=config.get("convs_per_block"),
        drop_rate=config.get("drop_rate", 0.1),
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
        loss_type=config.get("loss", "cce"),
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
    plot_path = Path(expdir) / "training_curves.png"
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
        PlotMetricsCallback(plot_path),
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
    trainer.fit(classifier, train_dataloaders=ds_train, val_dataloaders=ds_val)

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


def run_optuna(
    base_config: Dict,
    n_trials: int = 1,
    save_path: Path | None = None,
    objective_metric: str = "multi",
    study_name: str = "tornet_optuna_startfilters_32",
    storage: str | None = "sqlite:///tornet_optuna.db",
):
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("optuna is required for tuning. Install with `pip install optuna`.") from exc

    metric_key = objective_metric.upper()
    valid_metrics = {"AUC", "CSI", "AUCPR", "MULTI"}
    if metric_key not in valid_metrics:
        raise SystemExit(f"Invalid objective '{objective_metric}'. Choose from {sorted(valid_metrics)}.")

    def objective_single(trial):
        cfg = _suggest_config(trial, base_config)
        cfg["exp_name"] = f"{base_config.get('exp_name', 'tune')}_trial{trial.number}"
        result = main(cfg)
        auc = float(result.get("AUC", 0.0) or 0.0)
        aucpr = float(result.get("AUCPR", 0.0) or 0.0)
        csi = float(result.get("CSI", 0.0) or 0.0)
        trial.set_user_attr("AUCPR", aucpr)
        trial.set_user_attr("CSI", csi)
        target = {"AUC": auc, "AUCPR": aucpr, "CSI": csi}[metric_key]
        return target

    def objective_multi(trial):
        cfg = _suggest_config(trial, base_config)
        cfg["exp_name"] = f"{base_config.get('exp_name', 'tune')}_trial{trial.number}"
        result = main(cfg)
        auc = float(result.get("AUC", 0.0) or 0.0)
        aucpr = float(result.get("AUCPR", 0.0) or 0.0)
        csi = float(result.get("CSI", 0.0) or 0.0)
        trial.set_user_attr("AUC", auc)
        trial.set_user_attr("AUCPR", aucpr)
        trial.set_user_attr("CSI", csi)
        return auc, aucpr, csi

    if metric_key == "MULTI":
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            directions=["maximize", "maximize", "maximize"],
        )
        study.optimize(objective_multi, n_trials=n_trials)
        # pick a representative best trial (highest AUC among Pareto front)
        best = max(study.best_trials, key=lambda t: t.values[0])
        best_value = best.values
    else:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction="maximize",
        )
        study.optimize(objective_single, n_trials=n_trials)
        best = study.best_trial
        best_value = best.value

    logging.info("Best trial: %s values=%s", best.number, best_value)
    logging.info("User attrs: %s", best.user_attrs)
    logging.info("Params: %s", best.params)
    print("\nOptuna best values:", best_value)
    print("Best params:")
    for k, v in sorted(best.params.items()):
        print(f"  {k}: {v}")
    print("Additional metrics:", best.user_attrs)

    # Persist best hyperparameters to params.json (or provided path)
    if save_path is None:
        save_path = Path(__file__).with_name("config") / "params.json"
    best_config = deepcopy(base_config)
    best_config.update(best.params)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(best_config, f, indent=4)
    logging.info("Wrote best params to %s", save_path)
    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TorNet baseline (Torch) with optional Optuna tuning.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to JSON config override (e.g., params.json). Defaults to embedded + config/params.json if present.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        default=True,
        help="Run Optuna hyperparameter search instead of a single training run (default: on).",
    )
    parser.add_argument(
        "--no-tune",
        action="store_false",
        dest="tune",
        help="Disable Optuna tuning and run a single training run.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of Optuna trials to run when --tune is set (default: 1).",
    )
    parser.add_argument(
        "--save-params",
        type=Path,
        default=None,
        help="Where to save best hyperparameters when tuning (default: overwrite config/params.json).",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="multi",
        choices=["auc", "csi", "aucpr", "multi"],
        help="Metric to optimize when tuning (default: multi for AUC, AUCPR, CSI together).",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="tornet_optuna_startfilters_32",
        help="Optuna study name for persistent tuning (default: tornet_optuna_startfilters_32).",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///tornet_optuna.db",
        help="Optuna storage URL for persistent tuning (default: sqlite:///tornet_optuna.db).",
    )
    args = parser.parse_args()

    cfg = _load_default_config()
    if args.config:
        try:
            cfg.update(json.loads(Path(args.config).read_text()))
        except Exception as exc:
            raise SystemExit(f"Failed to load config {args.config}: {exc}") from exc

    if args.tune:
        run_optuna(
            cfg,
            n_trials=args.trials,
            save_path=args.save_params,
            objective_metric=args.objective,
            study_name=args.study_name,
            storage=args.storage,
        )
    else:
        main(cfg)
