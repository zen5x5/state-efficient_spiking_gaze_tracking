"""Train or smoke-test the retained static fixed-event-window pursuit ANN."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch

from ..augmentation import (
    RECOMBINATION_MODE_ALL_PHYSICAL_SOURCE_EPOCH_RANDOM,
    RECOMBINATION_MODE_EPOCH_RANDOM,
    RECOMBINATION_MODE_FIXED_INNER_CONTIGUOUS,
    RECOMBINATION_MODE_SOURCE_WISE_EPOCH_RANDOM,
    RECOMBINATION_MODES,
    SOURCE_WISE_START_DIFFERENCES_EVENTS,
)
from .dataloader import make_dataloader
from .evaluation import evaluate
from .metrics import squared_pixel_loss
from .model import (
    CHANNELS,
    COMPACT_CHANNELS,
    StaticPursuitANNCompact90K,
    StaticPursuitANNHiddenReadout,
    trainable_parameter_count,
)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


ANN_ARCHITECTURE = "eight_layer_8_16_24_32_40_40_40_40_hidden60_12_maxpool_96x128"
COMPACT90K_ANN_ARCHITECTURE = (
    "eight_layer_8_16_24_32_36_36_36_36_hidden48_12_maxpool_96x128"
)
ANN_ARCHITECTURES = (ANN_ARCHITECTURE, COMPACT90K_ANN_ARCHITECTURE)


def make_model(architecture: str) -> torch.nn.Module:
    if architecture == ANN_ARCHITECTURE:
        return StaticPursuitANNHiddenReadout()
    if architecture == COMPACT90K_ANN_ARCHITECTURE:
        return StaticPursuitANNCompact90K()
    raise ValueError(f"unknown ANN architecture: {architecture}")


def make_scheduler(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=9,
        threshold=0.0,
        threshold_mode="rel",
        cooldown=0,
        min_lr=1e-5,
        eps=0.0,
    )


def train_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    model.train()
    loss_sum = 0.0
    samples = 0
    for batch in loader:
        event_frame = batch["event_frame"].to(device)
        target = batch["target_norm"].to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(event_frame)
        per_sample = squared_pixel_loss(prediction, target)
        loss = per_sample.mean()
        loss.backward()
        optimizer.step()
        loss_sum += float(per_sample.detach().sum())
        samples += len(event_frame)
    return {"samples": samples, "mean_loss": loss_sum / samples}


def experiment_configuration(
    cache_dir: Path,
    seed: int,
    architecture: str,
    epochs: int,
    local_recombination: bool = False,
    recombination_max_label_distance_px: float = 32.0,
    local_recombination_mode: str = RECOMBINATION_MODE_EPOCH_RANDOM,
    recombination_sequential_cache_dir: Path | None = None,
) -> dict:
    if architecture not in ANN_ARCHITECTURES:
        raise ValueError(f"unknown ANN architecture: {architecture}")
    input_height, input_width = 96, 128
    if architecture == ANN_ARCHITECTURE:
        channels, hidden_features, second_hidden_features = CHANNELS, 60, 12
    elif architecture == COMPACT90K_ANN_ARCHITECTURE:
        channels, hidden_features, second_hidden_features = (
            COMPACT_CHANNELS,
            48,
            12,
        )
    else:
        raise ValueError(f"unknown ANN architecture: {architecture}")
    cache_manifest = json.loads(
        (cache_dir / "manifest.json").read_text(encoding="utf-8")
    )
    events_per_sample = int(cache_manifest["events_per_sample"])
    flattened_features = (
        channels[-1] * math.ceil(input_height / 32) * math.ceil(input_width / 32)
    )
    model = {
        "architecture": architecture.rsplit("_", 1)[0],
        "blocks": "Conv2d(bias=False,stride=1)+BatchNorm2d+ReLU; "
                  "MaxPool2d(2,2) after Conv1-Conv5",
        "batch_norm_layers": list(range(1, 9)),
        "channels": list(channels),
        "strides": [1] * 8,
        "pooling": "2x2 stride-2 MaxPool after Conv1/2/3/4/5",
        "kernel_size": 3,
        "readout": (
            f"Flatten({flattened_features})-"
            f"Linear{hidden_features}(bias=True)-ReLU-"
            f"Linear{second_hidden_features}(bias=True)-ReLU-Linear2(bias=False)"
        ),
    }
    expected_input_shape = [2, input_height, input_width]
    actual_input_shape = list(cache_manifest["input_shape"])
    if actual_input_shape != expected_input_shape:
        raise ValueError(
            f"architecture {architecture} expects input {expected_input_shape}, "
            f"but cache provides {actual_input_shape}"
        )
    source_wise = (
        local_recombination
        and local_recombination_mode
        == RECOMBINATION_MODE_SOURCE_WISE_EPOCH_RANDOM
    )
    all_physical_source_wise = (
        local_recombination
        and local_recombination_mode
        == RECOMBINATION_MODE_ALL_PHYSICAL_SOURCE_EPOCH_RANDOM
    )
    return {
        "cache_dir": str(cache_dir.resolve()),
        "input": {
            "events_per_sample": events_per_sample,
            "temporal_bins": 1,
            "channels": 2,
            "height": input_height,
            "width": input_width,
        },
        "model": model,
        "target": {
            "x": "(col - 960) / 960",
            "y": "(row - 540) / 540",
            "output_activation": "none",
        },
        "training": {
            "seed": seed,
            "batch_size": 16,
            "drop_last": True,
            "epochs": epochs,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "loss": "pixel_scaled_mse",
            "checkpoint_metric": "validation_mean_angular_error_deg",
            "learning_rate_schedule": {
                "metric": "validation_mean_angular_error_deg",
                "consecutive_epochs_without_improvement": 10,
                "factor": 0.5,
                "minimum_learning_rate": 1e-5,
            },
            "early_stopping": False,
            "augmentation": {
                "train_original_samples_retained": True,
                "local_nonadjacent_200_plus_200": local_recombination,
                "mode": local_recombination_mode if local_recombination else None,
                "source_windows": (
                    "every physical train sample, including stride-200 samples; "
                    "partners are in the same group and within the configured label "
                    "distance limit, prefer an "
                    "absolute start-index difference greater than 400 events, "
                    "and use exactly 400 only when no preferred partner exists"
                    if all_physical_source_wise
                    else
                    "each eligible complete non-stride, non-backfilled train source "
                    "chooses one partner in the same group whose absolute start-index "
                    "difference is 800, 1200, or 1600 events"
                    if source_wise
                    else "two complete non-stride, non-backfilled train windows in the "
                    "same group whose start indices differ by 800 events"
                ),
                "allowed_absolute_start_differences_events": (
                    None
                    if all_physical_source_wise
                    else
                    list(SOURCE_WISE_START_DIFFERENCES_EVENTS)
                    if source_wise
                    else [800]
                ),
                "preferred_absolute_start_difference_events": (
                    ">400" if all_physical_source_wise else None
                ),
                "adjacent_fallback_start_difference_events": (
                    400 if all_physical_source_wise else None
                ),
                "overlapping_start_difference_rejected_below_events": (
                    400 if all_physical_source_wise else None
                ),
                "intervening_window_split": (
                    "unrestricted and never read; may be train, validation or test"
                ),
                "events_from_each_source": 200,
                "sampling": (
                    "one partner per eligible physical source, selected from the "
                    "preferred pool or the adjacent fallback pool; partner and two "
                    "without-replacement 200-event subsets are deterministic by "
                    "seed/epoch/source and refreshed each epoch"
                    if all_physical_source_wise
                    else
                    "one partner per eligible source, selected from its candidate pool; "
                    "partner and two without-replacement 200-event subsets are "
                    "deterministic by seed/epoch/source and refreshed each epoch"
                    if source_wise
                    else "without replacement from each 400-event count frame; "
                    "deterministic by seed/epoch/pair and refreshed each epoch"
                    if local_recombination_mode == RECOMBINATION_MODE_EPOCH_RANDOM
                    else "fixed earlier-window last 200 ordered events plus "
                    "later-window first 200 ordered events"
                ),
                "sequential_source_cache": (
                    str(recombination_sequential_cache_dir.resolve())
                    if recombination_sequential_cache_dir is not None
                    else None
                ),
                "maximum_source_label_distance_px": (
                    recombination_max_label_distance_px
                ),
                "synthetic_target": "arithmetic mean of the two source row/col labels",
                "validation_and_test_augmented": False,
            },
        },
    }


def train(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if not args.local_recombination and args.recombination_sequential_cache_dir:
        raise ValueError("sequential recombination cache requires local recombination")
    if (
        args.local_recombination
        and args.local_recombination_mode
        == RECOMBINATION_MODE_FIXED_INNER_CONTIGUOUS
        and args.recombination_sequential_cache_dir is None
    ):
        raise ValueError("fixed contiguous mode requires a sequential cache")
    if (
        args.local_recombination_mode
        != RECOMBINATION_MODE_FIXED_INNER_CONTIGUOUS
        and args.recombination_sequential_cache_dir is not None
    ):
        raise ValueError("only fixed contiguous mode may receive a sequential cache")
    seed_everything(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    configuration = experiment_configuration(
        args.cache_dir,
        args.seed,
        args.architecture,
        args.epochs,
        args.local_recombination,
        args.recombination_max_label_distance_px,
        args.local_recombination_mode,
        args.recombination_sequential_cache_dir,
    )
    write_json(args.output_dir / "configuration.json", configuration)

    train_dataset, train_loader = make_dataloader(
        args.cache_dir,
        "train",
        16,
        args.seed,
        drop_last=True,
        local_recombination=args.local_recombination,
        recombination_max_label_distance_px=args.recombination_max_label_distance_px,
        local_recombination_mode=args.local_recombination_mode,
        recombination_sequential_cache_dir=args.recombination_sequential_cache_dir,
    )
    validation_dataset, validation_loader = make_dataloader(
        args.cache_dir, "validation", 16, args.seed
    )
    model = make_model(args.architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = make_scheduler(optimizer)
    history = []
    best_angle = float("inf")
    best_epoch = 0
    checkpoint_path = args.output_dir / "best_checkpoint.pt"

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        learning_rate = optimizer.param_groups[0]["lr"]
        training_metrics = train_epoch(model, train_loader, optimizer, device)
        validation_metrics = evaluate(model, validation_loader, device)
        angle = validation_metrics["mean_angular_error_deg"]
        if angle < best_angle:
            best_angle = angle
            best_epoch = epoch
            torch.save({"epoch": epoch, "model": model.state_dict()}, checkpoint_path)
        scheduler.step(angle)
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "next_learning_rate": optimizer.param_groups[0]["lr"],
            "best_epoch": best_epoch,
            "best_validation_mean_angular_error_deg": best_angle,
            "train": training_metrics,
            "validation": validation_metrics,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        write_json(args.output_dir / "training_history.json", history)
        print(
            f"epoch={epoch:03d} lr={learning_rate:.6g} "
            f"train_loss={training_metrics['mean_loss']:.6f} "
            f"val_angle={angle:.4f} best={best_angle:.4f}@{best_epoch}",
            flush=True,
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    source_train_dataset, full_train_loader = make_dataloader(
        args.cache_dir, "train", 16, args.seed
    )
    test_dataset, test_loader = make_dataloader(
        args.cache_dir, "test", 16, args.seed
    )
    summary = {
        "device": str(device),
        "trainable_parameters": trainable_parameter_count(model),
        "train_samples": len(train_dataset),
        "source_train_samples": len(source_train_dataset),
        "synthetic_train_samples_per_epoch": getattr(
            train_dataset, "synthetic_sample_count", 0
        ),
        "recombination_candidate_edges": getattr(
            train_dataset, "candidate_edge_count", 0
        ),
        "local_recombination_mode": getattr(train_dataset, "mode", None),
        "physical_stride_train_samples": getattr(
            train_dataset, "physical_stride_sample_count", 0
        ),
        "eligible_stride_recombination_sources": getattr(
            train_dataset, "eligible_stride_source_count", 0
        ),
        "preferred_recombination_sources": getattr(
            train_dataset, "preferred_source_count", 0
        ),
        "adjacent_fallback_recombination_sources": getattr(
            train_dataset, "adjacent_fallback_source_count", 0
        ),
        "ineligible_recombination_sources": getattr(
            train_dataset, "ineligible_source_count", 0
        ),
        "selectable_recombination_partners": getattr(
            train_dataset, "selectable_partner_count", 0
        ),
        "selectable_stride_recombination_partners": getattr(
            train_dataset, "selectable_partner_stride_count", 0
        ),
        "validation_samples": len(validation_dataset),
        "test_samples": len(test_dataset),
        "best_epoch": best_epoch,
        "best_train": evaluate(model, full_train_loader, device),
        "best_validation": evaluate(model, validation_loader, device),
        "test": evaluate(model, test_loader, device),
        "final_learning_rate": optimizer.param_groups[0]["lr"],
        "total_seconds": time.perf_counter() - started,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def smoke(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    device = torch.device(args.device)
    if (
        args.local_recombination
        and args.local_recombination_mode
        == RECOMBINATION_MODE_FIXED_INNER_CONTIGUOUS
        and args.recombination_sequential_cache_dir is None
    ):
        raise ValueError("fixed contiguous mode requires a sequential cache")
    dataset, loader = make_dataloader(
        args.cache_dir,
        "train",
        args.batch_size,
        args.seed,
        drop_last=True,
        local_recombination=args.local_recombination,
        recombination_max_label_distance_px=args.recombination_max_label_distance_px,
        local_recombination_mode=args.local_recombination_mode,
        recombination_sequential_cache_dir=args.recombination_sequential_cache_dir,
    )
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(1)
    model = make_model(args.architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    iterator = iter(loader)
    losses = []
    for _ in range(args.steps):
        batch = next(iterator)
        event_frame = batch["event_frame"].to(device)
        target = batch["target_norm"].to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(event_frame)
        loss = squared_pixel_loss(prediction, target).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    with tempfile.TemporaryDirectory(prefix="ebv-eye-static-ann-smoke-", dir="/tmp") as tmp:
        checkpoint_path = Path(tmp) / "model.pt"
        torch.save(model.state_dict(), checkpoint_path)
        restored = make_model(args.architecture).to(device)
        restored.load_state_dict(torch.load(checkpoint_path, map_location=device))
        validation = evaluate(restored, loader, device, max_batches=1)

    report = {
        "device": str(device),
        "train_samples": len(dataset),
        "batch_size": args.batch_size,
        "optimizer_steps": args.steps,
        "losses": losses,
        "trainable_parameters": trainable_parameter_count(model),
        "one_batch_evaluation": validation,
        "checkpoint_roundtrip": True,
        "temporary_artifacts_removed": True,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--cache-dir", type=Path, required=True)
    smoke_parser.add_argument("--device", default="cuda:0")
    smoke_parser.add_argument("--seed", type=int, default=2021)
    smoke_parser.add_argument("--batch-size", type=int, default=4)
    smoke_parser.add_argument("--steps", type=int, default=3)
    smoke_parser.add_argument("--local-recombination", action="store_true")
    smoke_parser.add_argument(
        "--local-recombination-mode",
        choices=RECOMBINATION_MODES,
        default=RECOMBINATION_MODE_EPOCH_RANDOM,
    )
    smoke_parser.add_argument("--recombination-sequential-cache-dir", type=Path)
    smoke_parser.add_argument(
        "--recombination-max-label-distance-px", type=float, default=32.0
    )
    smoke_parser.add_argument(
        "--architecture", choices=ANN_ARCHITECTURES, default=ANN_ARCHITECTURES[0]
    )
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--cache-dir", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--device", default="cuda:0")
    train_parser.add_argument("--seed", type=int, default=2021)
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--local-recombination", action="store_true")
    train_parser.add_argument(
        "--local-recombination-mode",
        choices=RECOMBINATION_MODES,
        default=RECOMBINATION_MODE_EPOCH_RANDOM,
    )
    train_parser.add_argument("--recombination-sequential-cache-dir", type=Path)
    train_parser.add_argument(
        "--recombination-max-label-distance-px", type=float, default=32.0
    )
    train_parser.add_argument(
        "--architecture", choices=ANN_ARCHITECTURES, default=ANN_ARCHITECTURES[0]
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "smoke":
        smoke(args)
    else:
        train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
