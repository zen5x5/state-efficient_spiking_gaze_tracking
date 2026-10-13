"""Train the paired pursuit SNN with explicit temporal input organization."""

from __future__ import annotations

import argparse
import copy
import json
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
from ..ann.metrics import squared_pixel_loss
from .dataloader import make_snn_dataloader, prepare_event_bins
from .evaluation import evaluate
from .model import (
    LearnableThresholdSpikingLayer,
    SNN_ARCHITECTURE,
    SpikingPursuitSNN,
    folded_batch_norm_bias,
    learnable_threshold_count,
    membrane_state_count,
    project_folded_batch_norm_bias_,
    trainable_parameter_count,
)


def write_json(path: Path, value: object) -> None:
    serialized = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    # Readers must see either the old complete JSON or the new complete JSON.
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            temporary.write(serialized)
            temporary.close()
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)


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


def make_model(
    alpha: float = 0.5,
    stateless_channels: dict[str, list[int]] | None = None,
) -> SpikingPursuitSNN:
    return SpikingPursuitSNN(
        alpha=alpha,
        threshold_init=1.0,
        min_v_mem=-1.0,
        backend="exodus",
        stateless_channels=stateless_channels,
    )


def load_stateless_channels(path: Path | None) -> dict[str, list[int]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("stateless mask JSON must be an object keyed by L1..L8")
    parsed = {}
    for name, indices in payload.items():
        if not isinstance(name, str) or not isinstance(indices, list):
            raise ValueError("stateless mask values must be channel-index lists")
        parsed[name] = [int(index) for index in indices]
    return parsed


def persistent_model_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return checkpoint state without batch-shaped transient membrane values."""
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
    unexpected = list(incompatible.unexpected_keys)
    missing_non_membrane = [
        key for key in incompatible.missing_keys if not key.endswith(".v_mem")
    ]
    if unexpected or missing_non_membrane:
        raise RuntimeError(
            "checkpoint mismatch: "
            f"missing_non_membrane={missing_non_membrane}, unexpected={unexpected}"
        )


def train_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    timesteps: int,
    temporal_organization: str,
    bias_constraint_layers: int,
) -> dict:
    model.train()
    loss_sum = 0.0
    samples = 0
    for batch in loader:
        target = batch["target_norm"].to(device)
        event_bins = prepare_event_bins(
            batch, timesteps, temporal_organization
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(event_bins)
        per_sample = squared_pixel_loss(prediction, target)
        loss = per_sample.mean()
        loss.backward()
        optimizer.step()
        project_folded_batch_norm_bias_(model, bias_constraint_layers)
        loss_sum += float(per_sample.detach().sum())
        samples += len(target)
    return {"samples": samples, "mean_loss": loss_sum / samples}


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


def folded_bias_statistics(
    model: SpikingPursuitSNN,
    constrained_through_layer: int,
) -> dict:
    upper_bound = 1.0 - model.alpha
    statistics = {}
    for name, values in folded_batch_norm_bias(model).items():
        values = values.detach().cpu()
        layer_index = int(name[1:])
        constrained = 0 < layer_index <= constrained_through_layer
        statistics[name] = {
            "channels": values.numel(),
            "minimum": float(values.min()),
            "mean": float(values.mean()),
            "maximum": float(values.max()),
            "robust_upper_bound": upper_bound,
            "constraint_enabled": constrained,
            "channels_above_bound": int((values > upper_bound + 1e-6).sum()),
        }
    return statistics


def experiment_configuration(
    cache_dir: Path,
    seed: int,
    timesteps: int,
    temporal_organization: str,
    alpha: float,
    bias_constraint_layers: int,
    epochs: int,
    stateless_channels: dict[str, list[int]] | None = None,
    save_training_state_epoch: int = 0,
    local_recombination: bool = False,
    recombination_max_label_distance_px: float = 32.0,
    local_recombination_mode: str = RECOMBINATION_MODE_EPOCH_RANDOM,
    recombination_sequential_cache_dir: Path | None = None,
) -> dict:
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if list(manifest["input_shape"]) != [2, 96, 128]:
        raise ValueError(f"expected cache input [2, 96, 128], got {manifest['input_shape']}")
    if int(manifest["events_per_sample"]) != 400:
        raise ValueError(
            f"expected 400 events per cached sample, got {manifest['events_per_sample']}"
        )
    if temporal_organization == "repeat_full_400":
        if int(manifest.get("bins_per_sample", 1)) != 1:
            raise ValueError("repeat_full_400 requires the static one-bin cache")
        input_configuration = {
            "events_per_cached_window": 400,
            "cached_bins_per_sample": 1,
            "temporal_steps": timesteps,
            "temporal_organization": "repeat_identical_complete_400_event_count_frame",
            "events_represented_per_step": 400,
            "logical_replayed_event_count": 400 * timesteps,
            "channels": 2,
            "height": 96,
            "width": 128,
        }
    elif temporal_organization == "sequential_equal_events":
        if int(manifest.get("cached_bins_per_sample", 0)) != 4:
            raise ValueError("sequential_equal_events requires the four-bin cache")
        if timesteps not in (2, 4):
            raise ValueError("sequential_equal_events supports only T=2 or T=4")
        input_configuration = {
            "events_per_cached_window": 400,
            "cached_bins_per_sample": 4,
            "temporal_steps": timesteps,
            "temporal_organization": "ordered_equal_event_bins",
            "events_represented_per_step": 400 // timesteps,
            "logical_replayed_event_count": 400,
            "channels": 2,
            "height": 96,
            "width": 128,
        }
    else:
        raise ValueError(f"unknown temporal organization: {temporal_organization}")
    neuron_name = "IAF" if alpha == 1.0 else "LIF"
    stateless_channels = stateless_channels or {}
    states = membrane_state_count()
    layer_widths = dict(zip(
        (f"L{index}" for index in range(1, 9)),
        (8, 16, 24, 32, 40, 40, 40, 40),
    ))
    saved_states = sum(
        len(indices) * states[name] // layer_widths[name]
        for name, indices in stateless_channels.items()
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
        "input": input_configuration,
        "model": {
            "architecture": SNN_ARCHITECTURE,
            "conv_channels": [2, 8, 16, 24, 32, 40, 40, 40, 40],
            "conv_blocks": f"Conv2d(bias=False,stride=1)+BatchNorm2d+MaxPool(if L1-L5)+{neuron_name}",
            "spiking_hidden_head": f"Flatten480-Linear60(bias=True)-{neuron_name}-Linear12(bias=True)-{neuron_name}",
            "readout": "mean H2 spikes over time then Linear12-to-2(bias=False)",
            "alpha": alpha,
            "alpha_trainable": False,
            "effective_threshold": "positive per-output-channel exp(log_threshold)",
            "threshold_initial_value": 1.0,
            "threshold_trainable": True,
            "spike_function": "MultiSpike",
            "surrogate_gradient": "PeriodicExponential",
            "reset": "MembraneSubtract",
            "minimum_membrane_potential": -1.0,
            "backend": "Exodus",
            "state_reset": "before every independent sample batch forward",
            "stateless_channels": stateless_channels,
            "retained_cross_bin_membrane_states": states["total"] - saved_states,
            "removed_cross_bin_membrane_states": saved_states,
            "folded_bn_bias_constraint": {
                "enabled": bias_constraint_layers > 0,
                "layers": (
                    f"L1-L{bias_constraint_layers}"
                    if bias_constraint_layers > 0
                    else "none"
                ),
                "normalized_upper_bound": 1.0 - alpha,
                "projection": "after every optimizer update",
            },
        },
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
            "full_training_state_snapshot_epoch": save_training_state_epoch,
            "augmentation": {
                "train_original_samples_retained": True,
                "local_nonadjacent_200_plus_200": local_recombination,
                "mode": local_recombination_mode if local_recombination else None,
                "source_windows": (
                    "every physical train sample, including stride-200 samples; "
                    "partners are in the same group and within the configured label "
                    "distance limit, prefer an absolute start-index difference "
                    "greater than 400 events, and use exactly 400 only when no "
                    "preferred partner exists"
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
    generator = train_loader.generator
    if generator is None:
        raise RuntimeError("training loader has no explicit random generator")
    return {
        "epoch": epoch,
        "model": persistent_model_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "train_loader_generator_state": generator.get_state(),
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
    """Restore the optimizer, sampling sequence and RNG at a branch boundary."""
    load_persistent_model_state_dict(model, state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    train_loader.generator.set_state(state["train_loader_generator_state"].cpu())
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_random_state"].cpu())
    if torch.cuda.is_available():
        saved_cuda = state["cuda_random_states"]
        if len(saved_cuda) != torch.cuda.device_count():
            raise ValueError("resume requires the same number of visible CUDA devices")
        torch.cuda.set_rng_state_all([value.cpu() for value in saved_cuda])


def train(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if not 0 <= args.save_training_state_epoch <= args.epochs:
        raise ValueError("save-training-state-epoch must be zero or within training")
    seed_everything(args.seed)
    device = torch.device(args.device)
    local_recombination = bool(getattr(args, "local_recombination", False))
    recombination_max_label_distance_px = float(
        getattr(args, "recombination_max_label_distance_px", 32.0)
    )
    local_recombination_mode = getattr(
        args, "local_recombination_mode", RECOMBINATION_MODE_EPOCH_RANDOM
    )
    recombination_sequential_cache_dir = getattr(
        args, "recombination_sequential_cache_dir", None
    )
    if local_recombination and args.temporal_organization != "repeat_full_400":
        raise ValueError("local recombination requires repeat_full_400 input")
    if not local_recombination and recombination_sequential_cache_dir is not None:
        raise ValueError("sequential recombination cache requires local recombination")
    if (
        local_recombination
        and local_recombination_mode
        == RECOMBINATION_MODE_FIXED_INNER_CONTIGUOUS
        and recombination_sequential_cache_dir is None
    ):
        raise ValueError("fixed contiguous mode requires a sequential cache")
    if (
        local_recombination_mode
        != RECOMBINATION_MODE_FIXED_INNER_CONTIGUOUS
        and recombination_sequential_cache_dir is not None
    ):
        raise ValueError("only fixed contiguous mode may receive a sequential cache")
    stateless_channels = load_stateless_channels(args.stateless_mask_json)
    resume_path = getattr(args, "resume_training_state", None)
    resumed_state = None
    if resume_path is not None:
        source_config = json.loads(
            (resume_path.parent / "configuration.json").read_text()
        )
        checks = (
            (source_config["training"]["seed"], args.seed),
            (source_config["model"]["alpha"], args.alpha),
            (source_config["input"]["temporal_steps"], args.timesteps),
            (source_config["model"]["folded_bn_bias_constraint"]["enabled"], args.bias_constraint_layers > 0),
            (Path(source_config["cache_dir"]).resolve(), args.cache_dir.resolve()),
            (
                source_config["training"].get("augmentation", {}).get(
                    "local_nonadjacent_200_plus_200", False
                ),
                local_recombination,
            ),
            (
                float(
                    source_config["training"].get("augmentation", {}).get(
                        "maximum_source_label_distance_px", 32.0
                    )
                ),
                recombination_max_label_distance_px,
            ),
            (
                source_config["training"].get("augmentation", {}).get(
                    "mode", RECOMBINATION_MODE_EPOCH_RANDOM
                ),
                local_recombination_mode,
            ),
            (
                source_config["training"].get("augmentation", {}).get(
                    "sequential_source_cache"
                ),
                (
                    str(recombination_sequential_cache_dir.resolve())
                    if recombination_sequential_cache_dir is not None
                    else None
                ),
            ),
        )
        if any(old != new for old, new in checks):
            raise ValueError("resume configuration differs from the source baseline")
        if source_config["model"].get("stateless_channels"):
            raise ValueError("channel branches must start from the all-state snapshot")
        resumed_state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if not 0 < resumed_state["epoch"] < args.epochs:
            raise ValueError("resume epoch must be below the total training budget")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    configuration = experiment_configuration(
        args.cache_dir,
        args.seed,
        args.timesteps,
        args.temporal_organization,
        args.alpha,
        args.bias_constraint_layers,
        args.epochs,
        stateless_channels,
        args.save_training_state_epoch,
        local_recombination,
        recombination_max_label_distance_px,
        local_recombination_mode,
        recombination_sequential_cache_dir,
    )
    configuration["training"]["resume_training_state"] = (
        str(resume_path.resolve()) if resume_path else None
    )
    configuration["training"]["branch_start_epoch"] = (
        resumed_state["epoch"] if resumed_state else 0
    )
    write_json(args.output_dir / "configuration.json", configuration)

    train_dataset, train_loader = make_snn_dataloader(
        args.cache_dir, "train", 16, args.seed, args.temporal_organization,
        drop_last=True,
        local_recombination=local_recombination,
        recombination_max_label_distance_px=recombination_max_label_distance_px,
        local_recombination_mode=local_recombination_mode,
        recombination_sequential_cache_dir=recombination_sequential_cache_dir,
    )
    validation_dataset, validation_loader = make_snn_dataloader(
        args.cache_dir, "validation", 16, args.seed, args.temporal_organization
    )
    model = make_model(args.alpha, stateless_channels).to(device)
    project_folded_batch_norm_bias_(model, args.bias_constraint_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = make_scheduler(optimizer)
    history = []
    best_angle = float("inf")
    best_epoch = 0
    checkpoint_path = args.output_dir / "best_checkpoint.pt"
    start_epoch = 0
    if resumed_state is not None:
        restore_training_state(resumed_state, model, optimizer, scheduler, train_loader)
        start_epoch = int(resumed_state["epoch"])
        history = copy.deepcopy(resumed_state["history"])
        if stateless_channels:
            # The old all-state best is not a checkpoint of the new architecture.
            initial_metrics = evaluate(
                model, validation_loader, device, args.timesteps,
                args.temporal_organization,
            )
            best_epoch = start_epoch
            best_angle = initial_metrics["mean_angular_error_deg"]
            initial_best_model = persistent_model_state_dict(model)
            configuration["training"]["branch_initial_validation"] = initial_metrics
        else:
            best_epoch = int(resumed_state["best_epoch"])
            best_angle = float(resumed_state["best_validation_mean_angular_error_deg"])
            initial_best_model = resumed_state["best_model"]
        torch.save({"epoch": best_epoch, "model": initial_best_model}, checkpoint_path)
        write_json(args.output_dir / "configuration.json", configuration)
        write_json(args.output_dir / "training_history.json", history)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_started = time.perf_counter()
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        learning_rate = optimizer.param_groups[0]["lr"]
        training_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.timesteps,
            args.temporal_organization,
            args.bias_constraint_layers,
        )
        validation_metrics = evaluate(
            model, validation_loader, device, args.timesteps,
            args.temporal_organization
        )
        angle = validation_metrics["mean_angular_error_deg"]
        if angle < best_angle:
            best_angle = angle
            best_epoch = epoch
            torch.save(
                {"epoch": epoch, "model": persistent_model_state_dict(model)},
                checkpoint_path,
            )
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
        if epoch == args.save_training_state_epoch:
            best_checkpoint = torch.load(checkpoint_path, map_location="cpu")
            torch.save(
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
            f"T={args.timesteps} epoch={epoch:03d} lr={learning_rate:.6g} "
            f"train_loss={training_metrics['mean_loss']:.6f} "
            f"val_angle={angle:.4f} best={best_angle:.4f}@{best_epoch}",
            flush=True,
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_persistent_model_state_dict(model, checkpoint["model"])
    source_train_dataset, full_train_loader = make_snn_dataloader(
        args.cache_dir, "train", 16, args.seed, args.temporal_organization
    )
    test_dataset, test_loader = make_snn_dataloader(
        args.cache_dir, "test", 16, args.seed, args.temporal_organization
    )
    summary = {
        "device": str(device),
        "timesteps": args.timesteps,
        "alpha": args.alpha,
        "bias_constraint_layers": args.bias_constraint_layers,
        "temporal_organization": configuration["input"]["temporal_organization"],
        "trainable_parameters": trainable_parameter_count(model),
        "learnable_thresholds": learnable_threshold_count(model),
        "membrane_states_per_sample": membrane_state_count()["total"],
        "retained_cross_bin_membrane_states": configuration["model"]["retained_cross_bin_membrane_states"],
        "removed_cross_bin_membrane_states": configuration["model"]["removed_cross_bin_membrane_states"],
        "stateless_channels": stateless_channels,
        "branch_start_epoch": start_epoch,
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
        "best_train": evaluate(
            model, full_train_loader, device, args.timesteps,
            args.temporal_organization
        ),
        "best_validation": evaluate(
            model, validation_loader, device, args.timesteps,
            args.temporal_organization
        ),
        "test": evaluate(
            model, test_loader, device, args.timesteps,
            args.temporal_organization
        ),
        "threshold_statistics": threshold_statistics(model),
        "folded_bias_statistics": folded_bias_statistics(
            model, args.bias_constraint_layers
        ),
        "final_learning_rate": optimizer.param_groups[0]["lr"],
        "total_seconds": time.perf_counter() - started,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def smoke(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    device = torch.device(args.device)
    local_recombination = bool(getattr(args, "local_recombination", False))
    recombination_max_label_distance_px = float(
        getattr(args, "recombination_max_label_distance_px", 32.0)
    )
    local_recombination_mode = getattr(
        args, "local_recombination_mode", RECOMBINATION_MODE_EPOCH_RANDOM
    )
    recombination_sequential_cache_dir = getattr(
        args, "recombination_sequential_cache_dir", None
    )
    if (
        local_recombination
        and local_recombination_mode
        == RECOMBINATION_MODE_FIXED_INNER_CONTIGUOUS
        and recombination_sequential_cache_dir is None
    ):
        raise ValueError("fixed contiguous mode requires a sequential cache")
    dataset, loader = make_snn_dataloader(
        args.cache_dir,
        "train",
        args.batch_size,
        args.seed,
        args.temporal_organization,
        drop_last=True,
        local_recombination=local_recombination,
        recombination_max_label_distance_px=recombination_max_label_distance_px,
        local_recombination_mode=local_recombination_mode,
        recombination_sequential_cache_dir=recombination_sequential_cache_dir,
    )
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(1)
    model = make_model(args.alpha).to(device)
    project_folded_batch_norm_bias_(model, args.bias_constraint_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    iterator = iter(loader)
    losses = []
    for _ in range(args.steps):
        batch = next(iterator)
        target = batch["target_norm"].to(device)
        event_bins = prepare_event_bins(
            batch, args.timesteps, args.temporal_organization
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(event_bins)
        loss = squared_pixel_loss(prediction, target).mean()
        loss.backward()
        optimizer.step()
        project_folded_batch_norm_bias_(model, args.bias_constraint_layers)
        losses.append(float(loss.detach()))

    with tempfile.TemporaryDirectory(prefix="ebv-eye-snn-smoke-", dir="/tmp") as tmp:
        checkpoint_path = Path(tmp) / "model.pt"
        torch.save(persistent_model_state_dict(model), checkpoint_path)
        restored = make_model(args.alpha).to(device)
        load_persistent_model_state_dict(
            restored, torch.load(checkpoint_path, map_location=device)
        )
        validation = evaluate(
            restored, loader, device, args.timesteps,
            args.temporal_organization, max_batches=1
        )

    report = {
        "device": str(device),
        "timesteps": args.timesteps,
        "temporal_organization": args.temporal_organization,
        "alpha": args.alpha,
        "bias_constraint_layers": args.bias_constraint_layers,
        "train_samples": len(dataset),
        "batch_size": args.batch_size,
        "optimizer_steps": args.steps,
        "losses": losses,
        "trainable_parameters": trainable_parameter_count(model),
        "one_batch_evaluation": validation,
        "folded_bias_statistics": folded_bias_statistics(
            model, args.bias_constraint_layers
        ),
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
    smoke_parser.add_argument("--timesteps", type=int, choices=(1, 2, 3, 4), required=True)
    smoke_parser.add_argument(
        "--temporal-organization",
        choices=("repeat_full_400", "sequential_equal_events"),
        default="repeat_full_400",
    )
    smoke_parser.add_argument("--device", default="cuda:0")
    smoke_parser.add_argument("--seed", type=int, default=2021)
    smoke_parser.add_argument("--alpha", type=float, choices=(0.0, 0.25, 0.5, 0.75, 1.0), default=0.5)
    smoke_parser.add_argument(
        "--bias-constraint-layers", type=int, choices=range(0, 9), default=0
    )
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
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--cache-dir", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--timesteps", type=int, choices=(1, 2, 3, 4), required=True)
    train_parser.add_argument(
        "--temporal-organization",
        choices=("repeat_full_400", "sequential_equal_events"),
        default="repeat_full_400",
    )
    train_parser.add_argument("--device", default="cuda:0")
    train_parser.add_argument("--seed", type=int, default=2021)
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--alpha", type=float, choices=(0.0, 0.25, 0.5, 0.75, 1.0), required=True)
    train_parser.add_argument(
        "--bias-constraint-layers", type=int, choices=range(0, 9), required=True
    )
    train_parser.add_argument("--stateless-mask-json", type=Path)
    train_parser.add_argument("--save-training-state-epoch", type=int, default=0)
    train_parser.add_argument("--resume-training-state", type=Path)
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
