"""Train the alpha-0.75 T=2 SNN paired with the formal saccade ANN."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time
from collections.abc import Sequence

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch

from ..ann.augmentation import AllPhysicalSameTargetRecombinationDataset
from ..ann.metrics import squared_pixel_loss
from .dataloader import (
    LOSS_WEIGHTING_ORDINARY,
    LOSS_WEIGHTINGS,
    StaticSaccadeDataset,
    make_dataloader,
    make_loader,
    repeat_full_event_frame,
)
from .evaluation import evaluate, evaluate_grid_count
from .model import (
    GRID_SPIKE_COUNT_ARCHITECTURE,
    GridCountSpikingSaccadeSNN,
    LearnableThresholdSpikingLayer,
    SNN_ARCHITECTURE,
    SpikingSaccadeSNN,
    folded_batch_norm_bias,
    grid_count_membrane_state_count,
    learnable_threshold_count,
    membrane_state_count,
    trainable_parameter_count,
)


TIMESTEPS = 2
ALPHA = 0.75
EXPECTED_PARAMETERS = 96_792
EXPECTED_THRESHOLDS = 312
EXPECTED_MEMBRANE_STATES = 45_000
OUTPUT_ENCODING_CONTINUOUS = "continuous_xy"
OUTPUT_ENCODING_GRID_SPIKE_COUNT = "grid_spike_count_xy"
OUTPUT_ENCODINGS = (
    OUTPUT_ENCODING_CONTINUOUS,
    OUTPUT_ENCODING_GRID_SPIKE_COUNT,
)
GRID_EXPECTED_PARAMETERS = 96_794
GRID_EXPECTED_THRESHOLDS = 314
GRID_EXPECTED_MEMBRANE_STATES = 45_002


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch_save(value: object, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def persistent_model_state_dict(
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in model.state_dict().items()
        if not key.endswith(".v_mem")
    }


def load_persistent_model_state_dict(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None:
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = [
        key for key in incompatible.missing_keys if not key.endswith(".v_mem")
    ]
    if incompatible.unexpected_keys or missing:
        raise RuntimeError(
            f"checkpoint mismatch: missing={missing}, "
            f"unexpected={list(incompatible.unexpected_keys)}"
        )


def capture_training_state(
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    train_loader: torch.utils.data.DataLoader,
    history: list[dict],
    best_epoch: int,
    best_angle: float,
    best_model: dict[str, torch.Tensor],
) -> dict:
    if train_loader.generator is None:
        raise RuntimeError("training loader has no explicit random generator")
    return {
        "epoch": epoch,
        "model": persistent_model_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "train_loader_generator_state": train_loader.generator.get_state(),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_states": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "history": copy.deepcopy(history),
        "best_epoch": best_epoch,
        "best_validation_mean_angular_error_deg": best_angle,
        "best_model": best_model,
    }


def restore_training_state(
    state: dict,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    train_loader: torch.utils.data.DataLoader,
) -> None:
    load_persistent_model_state_dict(model, state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if train_loader.generator is None:
        raise RuntimeError("training loader has no explicit random generator")
    train_loader.generator.set_state(
        state["train_loader_generator_state"].cpu()
    )
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_random_state"].cpu())
    if torch.cuda.is_available():
        saved_cuda = state["cuda_random_states"]
        if len(saved_cuda) != torch.cuda.device_count():
            raise ValueError("resume requires the same number of visible CUDA devices")
        torch.cuda.set_rng_state_all([value.cpu() for value in saved_cuda])


def make_model(
    backend: str = "exodus",
    output_encoding: str = OUTPUT_ENCODING_CONTINUOUS,
    stateless_channels: dict[str, list[int]] | None = None,
) -> SpikingSaccadeSNN | GridCountSpikingSaccadeSNN:
    if output_encoding == OUTPUT_ENCODING_CONTINUOUS:
        model = SpikingSaccadeSNN(
            alpha=ALPHA,
            threshold_init=1.0,
            min_v_mem=-1.0,
            backend=backend,
            stateless_channels=stateless_channels,
        )
        expected_parameters = EXPECTED_PARAMETERS
        expected_thresholds = EXPECTED_THRESHOLDS
        expected_states = EXPECTED_MEMBRANE_STATES
    elif output_encoding == OUTPUT_ENCODING_GRID_SPIKE_COUNT:
        if stateless_channels:
            raise ValueError("stateless channel masks are only supported by the continuous head")
        model = GridCountSpikingSaccadeSNN(
            alpha=ALPHA,
            threshold_init=1.0,
            min_v_mem=-1.0,
            backend=backend,
        )
        expected_parameters = GRID_EXPECTED_PARAMETERS
        expected_thresholds = GRID_EXPECTED_THRESHOLDS
        expected_states = GRID_EXPECTED_MEMBRANE_STATES
    else:
        raise ValueError(f"unknown output encoding: {output_encoding!r}")
    if trainable_parameter_count(model) != expected_parameters:
        raise RuntimeError("unexpected SNN parameter count")
    if learnable_threshold_count(model) != expected_thresholds:
        raise RuntimeError("unexpected learnable threshold count")
    actual_states = (
        grid_count_membrane_state_count()["total"]
        if output_encoding == OUTPUT_ENCODING_GRID_SPIKE_COUNT
        else membrane_state_count()["total"]
    )
    if actual_states != expected_states:
        raise RuntimeError("unexpected membrane state count")
    return model


def load_stateless_channels(path: Path | None) -> dict[str, list[int]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("stateless mask JSON must be an object keyed by L1..L8")
    parsed: dict[str, list[int]] = {}
    for name, indices in payload.items():
        if not isinstance(name, str) or not isinstance(indices, list):
            raise ValueError("stateless mask values must be channel-index lists")
        parsed[name] = [int(index) for index in indices]
    return parsed


def cross_bin_state_counts(
    stateless_channels: dict[str, list[int]],
) -> tuple[int, int]:
    layer_widths = {
        "L1": 8,
        "L2": 16,
        "L3": 24,
        "L4": 32,
        "L5": 40,
        "L6": 40,
        "L7": 40,
        "L8": 40,
    }
    states = membrane_state_count()
    removed = 0
    for name, indices in stateless_channels.items():
        if name not in layer_widths:
            raise ValueError(f"unknown stateless layer name: {name}")
        if len(indices) != len(set(indices)):
            raise ValueError(f"duplicate stateless channel in {name}: {indices}")
        if any(index < 0 or index >= layer_widths[name] for index in indices):
            raise ValueError(f"invalid stateless channel in {name}: {indices}")
        removed += len(indices) * states[name] // layer_widths[name]
    return states["total"] - removed, removed


def train_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    output_encoding: str = OUTPUT_ENCODING_CONTINUOUS,
) -> dict:
    model.train()
    unweighted_loss_sum = 0.0
    weighted_loss_sum = 0.0
    sample_weight_sum = 0.0
    samples = 0
    batches = 0
    for batch in loader:
        event_bins = repeat_full_event_frame(
            batch["event_frame"].to(device), TIMESTEPS
        )
        target = batch["target_norm"].to(device)
        sample_weight = batch["sample_weight"].to(device)
        optimizer.zero_grad(set_to_none=True)
        if output_encoding == OUTPUT_ENCODING_GRID_SPIKE_COUNT:
            prediction = model(event_bins, clamp_to_grid=False)
        else:
            prediction = model(event_bins)
        per_sample = squared_pixel_loss(prediction, target)
        objective = (per_sample * sample_weight).mean()
        objective.backward()
        optimizer.step()
        unweighted_loss_sum += float(per_sample.detach().sum())
        weighted_loss_sum += float((per_sample.detach() * sample_weight).sum())
        sample_weight_sum += float(sample_weight.sum())
        samples += len(target)
        batches += 1
    return {
        "samples": samples,
        "batches": batches,
        "mean_unweighted_loss": unweighted_loss_sum / samples,
        "mean_weighted_objective": weighted_loss_sum / samples,
        "mean_sample_weight_seen": sample_weight_sum / samples,
    }


def threshold_statistics(model: torch.nn.Module) -> dict:
    statistics = {}
    for name, module in model.named_modules():
        if not isinstance(module, LearnableThresholdSpikingLayer):
            continue
        values = module.threshold.detach().cpu()
        statistics[name] = {
            "channels": values.numel(),
            "minimum": float(values.min()),
            "mean": float(values.mean()),
            "maximum": float(values.max()),
        }
    return statistics


def folded_bias_statistics(model: SpikingSaccadeSNN) -> dict:
    statistics = {}
    for name, values in folded_batch_norm_bias(model).items():
        values = values.detach().cpu()
        statistics[name] = {
            "channels": values.numel(),
            "minimum": float(values.min()),
            "mean": float(values.mean()),
            "maximum": float(values.max()),
            "robust_upper_bound": 1.0 - ALPHA,
            "constraint_enabled": False,
            "channels_above_bound": int((values > 1.0 - ALPHA + 1e-6).sum()),
        }
    return statistics


def experiment_configuration(
    cache_dir: Path,
    seed: int,
    epochs: int,
    loss_weighting: str,
    physical_train_dataset: StaticSaccadeDataset,
    train_dataset,
    backend: str,
    output_encoding: str = OUTPUT_ENCODING_CONTINUOUS,
    stateless_channels: dict[str, list[int]] | None = None,
    save_training_state_epoch: int = 0,
) -> dict:
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    weights = physical_train_dataset.sample_weights
    grid_output = output_encoding == OUTPUT_ENCODING_GRID_SPIKE_COUNT
    expected_parameters = GRID_EXPECTED_PARAMETERS if grid_output else EXPECTED_PARAMETERS
    expected_thresholds = GRID_EXPECTED_THRESHOLDS if grid_output else EXPECTED_THRESHOLDS
    expected_states = (
        GRID_EXPECTED_MEMBRANE_STATES if grid_output else EXPECTED_MEMBRANE_STATES
    )
    stateless_channels = stateless_channels or {}
    if grid_output and stateless_channels:
        raise ValueError("grid spike-count output does not support stateless masks")
    if grid_output:
        retained_states, removed_states = expected_states, 0
    else:
        retained_states, removed_states = cross_bin_state_counts(stateless_channels)
    return {
        "cache_dir": str(cache_dir.resolve()),
        "cache_format": manifest["format"],
        "cache_sha256": {
            name: sha256_file(cache_dir / name)
            for name in ("manifest.json", "train.npz", "validation.npz", "test.npz")
        },
        "input": {
            "events_per_cached_window": int(manifest["events_per_sample"]),
            "shape": list(manifest["input_shape"]),
            "temporal_steps": TIMESTEPS,
            "temporal_organization": "repeat_identical_complete_400_event_count_frame",
            "events_represented_per_step": 400,
            "logical_replayed_event_count": 800,
            "train_samples": len(train_dataset),
            "physical_train_samples": len(physical_train_dataset),
            "validation_samples": int(manifest["cached_counts"]["validation"]),
            "test_samples": int(manifest["cached_counts"]["test"]),
        },
        "model": {
            "architecture": (
                GRID_SPIKE_COUNT_ARCHITECTURE if grid_output else SNN_ARCHITECTURE
            ),
            "parameters": expected_parameters,
            "learnable_thresholds": expected_thresholds,
            "membrane_states_per_sample": expected_states,
            "channels": [2, 8, 16, 24, 32, 40, 40, 40, 40],
            "spiking_hidden_head": (
                "480-60-LIF-12-LIF-2-MultiSpike"
                if grid_output
                else "480-60-LIF-12-LIF-2"
            ),
            "output_encoding": output_encoding,
            "alpha": ALPHA,
            "alpha_trainable": False,
            "threshold": "positive per-channel exp(log_threshold), trainable",
            "spike_function": "MultiSpike",
            "surrogate_gradient": "PeriodicExponential",
            "reset": "MembraneSubtract",
            "minimum_membrane_potential": -1.0,
            "backend": backend,
            "state_reset": "before every independent sample batch forward",
            "stateless_channels": stateless_channels,
            "retained_cross_bin_membrane_states": retained_states,
            "removed_cross_bin_membrane_states": removed_states,
            "bias_constraint_layers": 0,
        },
        "target": {
            "x": "(col - 960) / 960",
            "y": "(row - 540) / 540",
            "output_activation": "MultiSpike" if grid_output else "none",
            "grid_count_decoding": (
                {
                    "axis_order": ["column", "row"],
                    "valid_count_levels": list(range(11)),
                    "column_px": "150 + 162 * count_x",
                    "row_px": "150 + 78 * count_y",
                    "training_decode": "raw integer count without clipping",
                    "inference_decode": "clip integer count to [0, 10]",
                }
                if grid_output
                else None
            ),
        },
        "training": {
            "seed": seed,
            "epochs": epochs,
            "batch_size": 16,
            "drop_last": True,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "loss": "pixel_scaled_squared_distance",
            "loss_weighting": loss_weighting,
            "weight_formula": (
                "1"
                if loss_weighting == LOSS_WEIGHTING_ORDINARY
                else "n_target^(-0.5) normalized to sample mean 1 over physical train cache"
            ),
            "sample_weight_min": float(weights.min()),
            "sample_weight_max": float(weights.max()),
            "sample_weight_mean": float(weights.mean()),
            "target_train_counts": physical_train_dataset.target_counts,
            "checkpoint_metric": "validation sample-mean angular error deg",
            "scheduler": {
                "name": "ReduceLROnPlateau",
                "factor": 0.5,
                "patience": 9,
                "threshold": 0.0,
                "min_lr": 1e-5,
            },
            "early_stopping": False,
            "full_training_state_snapshot_epoch": save_training_state_epoch,
        },
        "augmentation": train_dataset.audit_summary(),
    }


def train(args: argparse.Namespace) -> dict:
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if not 0 <= args.save_training_state_epoch <= args.epochs:
        raise ValueError("save-training-state-epoch must be zero or within training")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if (
        args.output_encoding == OUTPUT_ENCODING_GRID_SPIKE_COUNT
        and args.loss_weighting != LOSS_WEIGHTING_ORDINARY
    ):
        raise ValueError("grid spike-count output is restricted to ordinary loss")
    stateless_channels = load_stateless_channels(args.stateless_mask_json)
    resume_path = args.resume_training_state
    resumed_state = None
    source_configuration = None
    if resume_path is not None:
        if args.output_encoding != OUTPUT_ENCODING_CONTINUOUS:
            raise ValueError("resume with a state mask is restricted to continuous output")
        source_configuration = json.loads(
            (resume_path.parent / "configuration.json").read_text(encoding="utf-8")
        )
        checks = (
            (int(source_configuration["training"]["seed"]), args.seed),
            (source_configuration["training"]["loss_weighting"], args.loss_weighting),
            (source_configuration["model"]["output_encoding"], args.output_encoding),
            (float(source_configuration["model"]["alpha"]), ALPHA),
            (int(source_configuration["input"]["temporal_steps"]), TIMESTEPS),
            (Path(source_configuration["cache_dir"]).resolve(), args.cache_dir.resolve()),
        )
        if any(old != new for old, new in checks):
            raise ValueError("resume configuration differs from the source baseline")
        if source_configuration["model"].get("stateless_channels"):
            raise ValueError("the L1/L2 branch must start from an all-state snapshot")
        resumed_state = torch.load(
            resume_path, map_location="cpu", weights_only=False
        )
        if not 0 < int(resumed_state["epoch"]) < args.epochs:
            raise ValueError("resume epoch must be below the total training budget")
    seed_everything(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    physical_train_dataset = StaticSaccadeDataset(
        args.cache_dir, "train", args.loss_weighting
    )
    train_dataset = AllPhysicalSameTargetRecombinationDataset(
        physical_train_dataset, seed=args.seed
    )
    train_loader = make_loader(
        train_dataset, 16, args.seed, drop_last=True, shuffle=True
    )
    _, validation_loader = make_dataloader(
        args.cache_dir, "validation", 16, args.seed, shuffle=False
    )
    configuration = experiment_configuration(
        args.cache_dir,
        args.seed,
        args.epochs,
        args.loss_weighting,
        physical_train_dataset,
        train_dataset,
        args.backend,
        args.output_encoding,
        stateless_channels,
        args.save_training_state_epoch,
    )
    configuration["training"]["resume_training_state"] = (
        str(resume_path.resolve()) if resume_path else None
    )
    configuration["training"]["branch_start_epoch"] = (
        int(resumed_state["epoch"]) if resumed_state else 0
    )
    write_json(args.output_dir / "configuration.json", configuration)

    model = make_model(
        args.backend, args.output_encoding, stateless_channels
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = make_scheduler(optimizer)
    checkpoint_path = args.output_dir / "best_checkpoint.pt"
    history = []
    best_angle = float("inf")
    best_epoch = 0
    start_epoch = 0

    if resumed_state is not None:
        restore_training_state(
            resumed_state, model, optimizer, scheduler, train_loader
        )
        start_epoch = int(resumed_state["epoch"])
        history = copy.deepcopy(resumed_state["history"])
        if stateless_channels:
            initial_metrics = evaluate(
                model, validation_loader, device, timesteps=TIMESTEPS
            )
            best_epoch = start_epoch
            best_angle = initial_metrics["mean_angular_error_deg"]
            initial_best_model = persistent_model_state_dict(model)
            configuration["training"]["branch_initial_validation"] = initial_metrics
        else:
            best_epoch = int(resumed_state["best_epoch"])
            best_angle = float(
                resumed_state["best_validation_mean_angular_error_deg"]
            )
            initial_best_model = resumed_state["best_model"]
        atomic_torch_save(
            {"epoch": best_epoch, "model": initial_best_model}, checkpoint_path
        )
        write_json(args.output_dir / "configuration.json", configuration)
        write_json(args.output_dir / "training_history.json", history)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_started = time.perf_counter()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_dataset.set_epoch(epoch)
        training_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.output_encoding,
        )
        if args.output_encoding == OUTPUT_ENCODING_GRID_SPIKE_COUNT:
            validation_metrics = evaluate_grid_count(
                model, validation_loader, device, timesteps=TIMESTEPS
            )
        else:
            validation_metrics = evaluate(
                model, validation_loader, device, timesteps=TIMESTEPS
            )
        angle = validation_metrics["mean_angular_error_deg"]
        if angle < best_angle:
            best_angle = angle
            best_epoch = epoch
            atomic_torch_save(
                {"epoch": epoch, "model": persistent_model_state_dict(model)},
                checkpoint_path,
            )
        scheduler.step(angle)
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "next_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "best_epoch": best_epoch,
            "best_validation_mean_angular_error_deg": best_angle,
            "train": training_metrics,
            "validation": validation_metrics,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        write_json(args.output_dir / "training_history.json", history)
        if epoch == args.save_training_state_epoch:
            best_checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            atomic_torch_save(
                capture_training_state(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_loader=train_loader,
                    history=history,
                    best_epoch=best_epoch,
                    best_angle=best_angle,
                    best_model=best_checkpoint["model"],
                ),
                args.output_dir / f"training_state_epoch{epoch}.pt",
            )
        print(
            f"T={TIMESTEPS} alpha={ALPHA} epoch={epoch:03d} "
            f"lr={learning_rate:.6g} "
            f"train_objective={training_metrics['mean_weighted_objective']:.6f} "
            f"val_angle={angle:.4f} best={best_angle:.4f}@{best_epoch}",
            flush=True,
        )

    best_checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    full_state = capture_training_state(
        epoch=args.epochs,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        history=history,
        best_epoch=best_epoch,
        best_angle=best_angle,
        best_model=best_checkpoint["model"],
    )
    atomic_torch_save(
        full_state, args.output_dir / f"epoch_{args.epochs}_training_state.pt"
    )

    load_persistent_model_state_dict(model, best_checkpoint["model"])
    split_metrics = {}
    for split in ("train", "validation", "test"):
        _, loader = make_dataloader(
            args.cache_dir, split, 16, args.seed, shuffle=False
        )
        evaluator = (
            evaluate_grid_count
            if args.output_encoding == OUTPUT_ENCODING_GRID_SPIKE_COUNT
            else evaluate
        )
        split_metrics[split] = evaluator(
            model,
            loader,
            device,
            timesteps=TIMESTEPS,
            prediction_path=args.output_dir / f"predictions_{split}.npz",
        )
    membrane_states = (
        GRID_EXPECTED_MEMBRANE_STATES
        if args.output_encoding == OUTPUT_ENCODING_GRID_SPIKE_COUNT
        else EXPECTED_MEMBRANE_STATES
    )
    summary = {
        "status": "complete",
        "device": str(device),
        "seed": args.seed,
        "loss_weighting": args.loss_weighting,
        "output_encoding": args.output_encoding,
        "timesteps": TIMESTEPS,
        "alpha": ALPHA,
        "bias_constraint_layers": 0,
        "same_target_recombination_all_sources": True,
        "physical_train_samples": len(physical_train_dataset),
        "synthetic_train_samples_per_epoch": train_dataset.synthetic_sample_count,
        "trainable_parameters": trainable_parameter_count(model),
        "learnable_thresholds": learnable_threshold_count(model),
        "membrane_states_per_sample": membrane_states,
        "retained_cross_bin_membrane_states": configuration["model"][
            "retained_cross_bin_membrane_states"
        ],
        "removed_cross_bin_membrane_states": configuration["model"][
            "removed_cross_bin_membrane_states"
        ],
        "stateless_channels": stateless_channels,
        "branch_start_epoch": start_epoch,
        "optimizer_updates": sum(
            int(record["train"]["batches"]) for record in history
        ),
        "branch_optimizer_updates": sum(
            int(record["train"]["batches"])
            for record in history
            if int(record["epoch"]) > start_epoch
        ),
        "best_epoch": best_epoch,
        "best_validation_mean_angular_error_deg": best_angle,
        "metrics": split_metrics,
        "threshold_statistics": threshold_statistics(model),
        "folded_bias_statistics": folded_bias_statistics(model),
        "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
        "total_seconds": time.perf_counter() - started,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def smoke(args: argparse.Namespace) -> dict:
    if (
        args.output_encoding == OUTPUT_ENCODING_GRID_SPIKE_COUNT
        and args.loss_weighting != LOSS_WEIGHTING_ORDINARY
    ):
        raise ValueError("grid spike-count output is restricted to ordinary loss")
    seed_everything(args.seed)
    device = torch.device(args.device)
    physical_dataset = StaticSaccadeDataset(
        args.cache_dir, "train", args.loss_weighting
    )
    dataset = AllPhysicalSameTargetRecombinationDataset(
        physical_dataset, seed=args.seed
    )
    dataset.set_epoch(1)
    loader = make_loader(
        dataset,
        args.batch_size,
        args.seed,
        drop_last=True,
        shuffle=True,
    )
    model = make_model(args.backend, args.output_encoding).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    iterator = iter(loader)
    losses = []
    for _ in range(args.steps):
        batch = next(iterator)
        event_bins = repeat_full_event_frame(
            batch["event_frame"].to(device), TIMESTEPS
        )
        target = batch["target_norm"].to(device)
        weight = batch["sample_weight"].to(device)
        optimizer.zero_grad(set_to_none=True)
        if args.output_encoding == OUTPUT_ENCODING_GRID_SPIKE_COUNT:
            prediction = model(event_bins, clamp_to_grid=False)
        else:
            prediction = model(event_bins)
        loss = (squared_pixel_loss(prediction, target) * weight).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    with tempfile.TemporaryDirectory(prefix="ebv-saccade-snn-smoke-", dir="/tmp") as tmp:
        checkpoint = Path(tmp) / "model.pt"
        torch.save(persistent_model_state_dict(model), checkpoint)
        restored = make_model(args.backend, args.output_encoding).to(device)
        load_persistent_model_state_dict(
            restored,
            torch.load(checkpoint, map_location=device, weights_only=True),
        )
        evaluator = (
            evaluate_grid_count
            if args.output_encoding == OUTPUT_ENCODING_GRID_SPIKE_COUNT
            else evaluate
        )
        one_batch = evaluator(
            restored,
            loader,
            device,
            timesteps=TIMESTEPS,
            max_batches=1,
        )
    report = {
        "device": str(device),
        "backend": args.backend,
        "loss_weighting": args.loss_weighting,
        "output_encoding": args.output_encoding,
        "timesteps": TIMESTEPS,
        "alpha": ALPHA,
        "train_samples": len(dataset),
        "physical_train_samples": len(physical_dataset),
        "augmentation": dataset.audit_summary(),
        "optimizer_steps": args.steps,
        "losses": losses,
        "parameters": trainable_parameter_count(model),
        "one_batch_evaluation": one_batch,
        "temporary_artifacts_removed": True,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--cache-dir", type=Path, required=True)
    smoke_parser.add_argument("--device", default="cpu")
    smoke_parser.add_argument("--backend", choices=("exodus", "sinabs"), default="sinabs")
    smoke_parser.add_argument("--seed", type=int, default=2021)
    smoke_parser.add_argument("--batch-size", type=int, default=2)
    smoke_parser.add_argument("--steps", type=int, default=1)
    smoke_parser.add_argument(
        "--loss-weighting", choices=LOSS_WEIGHTINGS, default=LOSS_WEIGHTING_ORDINARY
    )
    smoke_parser.add_argument(
        "--output-encoding", choices=OUTPUT_ENCODINGS, default=OUTPUT_ENCODING_CONTINUOUS
    )
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--cache-dir", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--device", default="cuda:0")
    train_parser.add_argument("--backend", choices=("exodus",), default="exodus")
    train_parser.add_argument("--seed", type=int, default=2021)
    train_parser.add_argument("--epochs", type=int, default=200)
    train_parser.add_argument(
        "--loss-weighting", choices=LOSS_WEIGHTINGS, default=LOSS_WEIGHTING_ORDINARY
    )
    train_parser.add_argument(
        "--output-encoding", choices=OUTPUT_ENCODINGS, default=OUTPUT_ENCODING_CONTINUOUS
    )
    train_parser.add_argument("--stateless-mask-json", type=Path)
    train_parser.add_argument("--save-training-state-epoch", type=int, default=0)
    train_parser.add_argument("--resume-training-state", type=Path)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if args.command == "smoke":
        smoke(args)
    else:
        train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
