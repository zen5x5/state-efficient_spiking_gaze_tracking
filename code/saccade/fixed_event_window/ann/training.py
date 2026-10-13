"""Train the retained eight-layer ANN on the formal static saccade cache."""

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

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch

from .dataloader import (
    LOSS_WEIGHTING_ORDINARY,
    LOSS_WEIGHTING_TARGET_SQRT,
    LOSS_WEIGHTINGS,
    StaticSaccadeDataset,
    make_dataloader,
    make_loader,
)
from .augmentation import (
    AllPhysicalSameTargetRecombinationDataset,
    NonstrideSameTargetRecombinationDataset,
)
from .evaluation import evaluate
from .metrics import squared_pixel_loss
from .model import StaticSaccadeANN, trainable_parameter_count


ARCHITECTURE = "eight_layer_8_16_24_32_40_40_40_40_hidden60_12_maxpool_96x128"


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


def make_training_dataset(args: argparse.Namespace, physical_dataset):
    if args.same_target_recombination_nonstride_sources:
        return NonstrideSameTargetRecombinationDataset(
            physical_dataset,
            seed=args.seed,
        )
    if args.same_target_recombination_all_sources:
        return AllPhysicalSameTargetRecombinationDataset(
            physical_dataset,
            seed=args.seed,
        )
    return physical_dataset


def recombination_enabled(args: argparse.Namespace) -> bool:
    return bool(
        args.same_target_recombination_nonstride_sources
        or args.same_target_recombination_all_sources
    )


def train_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    model.train()
    unweighted_loss_sum = 0.0
    weighted_loss_sum = 0.0
    sample_weight_sum = 0.0
    samples = 0
    batches = 0
    for batch in loader:
        event_frame = batch["event_frame"].to(device)
        target = batch["target_norm"].to(device)
        sample_weight = batch["sample_weight"].to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(event_frame)
        per_sample = squared_pixel_loss(prediction, target)
        objective = (per_sample * sample_weight).mean()
        objective.backward()
        optimizer.step()
        unweighted_loss_sum += float(per_sample.detach().sum())
        weighted_loss_sum += float((per_sample.detach() * sample_weight).sum())
        sample_weight_sum += float(sample_weight.sum())
        samples += len(event_frame)
        batches += 1
    return {
        "samples": samples,
        "batches": batches,
        "mean_unweighted_loss": unweighted_loss_sum / samples,
        "mean_weighted_objective": weighted_loss_sum / samples,
        "mean_sample_weight_seen": sample_weight_sum / samples,
    }


def experiment_configuration(
    cache_dir: Path,
    seed: int,
    epochs: int,
    loss_weighting: str,
    physical_train_dataset,
    train_dataset,
    recombination_enabled: bool,
) -> dict:
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    weights = physical_train_dataset.sample_weights
    return {
        "cache_dir": str(cache_dir.resolve()),
        "cache_format": manifest["format"],
        "cache_sha256": {
            name: sha256_file(cache_dir / name)
            for name in ("manifest.json", "train.npz", "validation.npz", "test.npz")
        },
        "input": {
            "events_per_sample": int(manifest["events_per_sample"]),
            "shape": list(manifest["input_shape"]),
            "train_samples": len(train_dataset),
            "physical_train_samples": len(physical_train_dataset),
            "validation_samples": int(manifest["cached_counts"]["validation"]),
            "test_samples": int(manifest["cached_counts"]["test"]),
        },
        "model": {
            "architecture": ARCHITECTURE,
            "parameters": 96_480,
            "blocks": "Conv2d(bias=False)+BatchNorm2d+ReLU, MaxPool after L1-L5",
            "channels": [2, 8, 16, 24, 32, 40, 40, 40, 40],
            "readout": "480-60-12-2",
        },
        "target": {
            "x": "(col - 960) / 960",
            "y": "(row - 540) / 540",
            "output_activation": "none",
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
        },
        "augmentation": (
            train_dataset.audit_summary()
            if recombination_enabled
            else {"enabled": False}
        ),
    }


def train(args: argparse.Namespace) -> dict:
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    resume_path = args.resume_training_state
    resumed_state = None
    if resume_path is not None:
        source_configuration = json.loads(
            (resume_path.parent / "configuration.json").read_text(encoding="utf-8")
        )
        checks = (
            (int(source_configuration["training"]["seed"]), args.seed),
            (source_configuration["training"]["loss_weighting"], args.loss_weighting),
            (Path(source_configuration["cache_dir"]).resolve(), args.cache_dir.resolve()),
            (bool(source_configuration["augmentation"]["enabled"]), recombination_enabled(args)),
            (
                source_configuration["augmentation"].get("source_pool_rule"),
                "all physical train samples" if args.same_target_recombination_all_sources
                else "physical train augmented == False"
                if args.same_target_recombination_nonstride_sources else None,
            ),
            (source_configuration["model"]["architecture"], ARCHITECTURE),
        )
        if any(old != new for old, new in checks):
            raise ValueError("resume configuration differs from the source ANN run")
        resumed_state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if not 0 < int(resumed_state["epoch"]) < args.epochs:
            raise ValueError("resume epoch must be below the total training budget")
    seed_everything(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    physical_train_dataset = StaticSaccadeDataset(
        args.cache_dir,
        "train",
        args.loss_weighting,
    )
    train_dataset = make_training_dataset(args, physical_train_dataset)
    train_loader = make_loader(
        train_dataset,
        16,
        args.seed,
        drop_last=True,
        shuffle=True,
    )
    _, validation_loader = make_dataloader(
        args.cache_dir,
        "validation",
        16,
        args.seed,
        shuffle=False,
    )
    configuration = experiment_configuration(
        args.cache_dir,
        args.seed,
        args.epochs,
        args.loss_weighting,
        physical_train_dataset,
        train_dataset,
        recombination_enabled(args),
    )
    if resumed_state is not None and (
        source_configuration["cache_sha256"] != configuration["cache_sha256"]
    ):
        raise ValueError("resume cache content differs from the source ANN run")
    configuration["training"]["resume_training_state"] = (
        str(resume_path.resolve()) if resume_path else None
    )
    configuration["training"]["branch_start_epoch"] = (
        int(resumed_state["epoch"]) if resumed_state else 0
    )
    write_json(args.output_dir / "configuration.json", configuration)

    model = StaticSaccadeANN().to(device)
    if trainable_parameter_count(model) != 96_480:
        raise RuntimeError("unexpected model parameter count")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = make_scheduler(optimizer)
    checkpoint_path = args.output_dir / "best_checkpoint.pt"
    history = []
    best_angle = float("inf")
    best_epoch = 0
    start_epoch = 0

    if resumed_state is not None:
        model.load_state_dict(resumed_state["model"])
        optimizer.load_state_dict(resumed_state["optimizer"])
        scheduler.load_state_dict(resumed_state["scheduler"])
        train_loader.generator.set_state(resumed_state["train_loader_generator_state"].cpu())
        random.setstate(resumed_state["python_rng_state"])
        np.random.set_state(resumed_state["numpy_rng_state"])
        torch.set_rng_state(resumed_state["torch_rng_state"].cpu())
        saved_cuda = resumed_state["cuda_rng_state_all"]
        if torch.cuda.is_available():
            if saved_cuda is None or len(saved_cuda) != torch.cuda.device_count():
                raise ValueError("resume requires the same number of visible CUDA devices")
            torch.cuda.set_rng_state_all([value.cpu() for value in saved_cuda])
        start_epoch = int(resumed_state["epoch"])
        history = copy.deepcopy(resumed_state["history"])
        best_epoch = int(resumed_state["best_epoch"])
        best_angle = float(history[-1]["best_validation_mean_angular_error_deg"])
        atomic_torch_save(
            {"epoch": best_epoch, "model": resumed_state["best_model"]}, checkpoint_path
        )
        write_json(args.output_dir / "training_history.json", history)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_started = time.perf_counter()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        training_metrics = train_epoch(model, train_loader, optimizer, device)
        validation_metrics = evaluate(model, validation_loader, device)
        angle = validation_metrics["mean_angular_error_deg"]
        if angle < best_angle:
            best_angle = angle
            best_epoch = epoch
            atomic_torch_save(
                {"epoch": epoch, "model": model.state_dict()}, checkpoint_path
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
        print(
            f"epoch={epoch:03d} lr={learning_rate:.6g} "
            f"train_objective={training_metrics['mean_weighted_objective']:.6f} "
            f"val_angle={angle:.4f} best={best_angle:.4f}@{best_epoch}",
            flush=True,
        )

    best_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    full_state = {
        "epoch": args.epochs,
        "model": model.state_dict(),
        "best_epoch": best_epoch,
        "best_model": best_checkpoint["model"],
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "history": history,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "train_loader_generator_state": train_loader.generator.get_state(),
    }
    atomic_torch_save(
        full_state, args.output_dir / f"epoch_{args.epochs}_training_state.pt"
    )

    model.load_state_dict(best_checkpoint["model"])
    split_metrics = {}
    for split in ("train", "validation", "test"):
        _, loader = make_dataloader(
            args.cache_dir, split, 16, args.seed, shuffle=False
        )
        split_metrics[split] = evaluate(
            model,
            loader,
            device,
            prediction_path=args.output_dir / f"predictions_{split}.npz",
        )
    summary = {
        "status": "complete",
        "device": str(device),
        "seed": args.seed,
        "loss_weighting": args.loss_weighting,
        "same_target_recombination_nonstride_sources": (
            args.same_target_recombination_nonstride_sources
        ),
        "same_target_recombination_all_sources": (
            args.same_target_recombination_all_sources
        ),
        "physical_train_samples": len(physical_train_dataset),
        "synthetic_train_samples_per_epoch": (
            train_dataset.synthetic_sample_count
            if recombination_enabled(args)
            else 0
        ),
        "trainable_parameters": trainable_parameter_count(model),
        "optimizer_updates": args.epochs * (len(train_dataset) // 16),
        "branch_start_epoch": start_epoch,
        "best_epoch": best_epoch,
        "best_validation_mean_angular_error_deg": best_angle,
        "metrics": split_metrics,
        "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
        "total_seconds": time.perf_counter() - started,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def smoke(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    device = torch.device(args.device)
    physical_dataset = StaticSaccadeDataset(
        args.cache_dir,
        "train",
        args.loss_weighting,
    )
    dataset = make_training_dataset(args, physical_dataset)
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(1)
    loader = make_loader(
        dataset,
        args.batch_size,
        args.seed,
        drop_last=True,
        shuffle=True,
    )
    model = StaticSaccadeANN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    iterator = iter(loader)
    losses = []
    for _ in range(args.steps):
        batch = next(iterator)
        frame = batch["event_frame"].to(device)
        target = batch["target_norm"].to(device)
        weight = batch["sample_weight"].to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(frame)
        loss = (squared_pixel_loss(prediction, target) * weight).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    with tempfile.TemporaryDirectory(prefix="ebv-saccade-ann-smoke-", dir="/tmp") as tmp:
        checkpoint = Path(tmp) / "model.pt"
        torch.save(model.state_dict(), checkpoint)
        restored = StaticSaccadeANN().to(device)
        restored.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )
        one_batch = evaluate(restored, loader, device, max_batches=1)
    report = {
        "device": str(device),
        "loss_weighting": args.loss_weighting,
        "train_samples": len(dataset),
        "physical_train_samples": len(physical_dataset),
        "sample_weight_mean": float(physical_dataset.sample_weights.mean()),
        "sample_weight_min": float(physical_dataset.sample_weights.min()),
        "sample_weight_max": float(physical_dataset.sample_weights.max()),
        "augmentation": (
            dataset.audit_summary()
            if recombination_enabled(args)
            else {"enabled": False}
        ),
        "optimizer_steps": args.steps,
        "losses": losses,
        "parameters": trainable_parameter_count(model),
        "one_batch_evaluation": one_batch,
        "temporary_artifacts_removed": True,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--cache-dir", type=Path, required=True)
    smoke_parser.add_argument("--device", default="cpu")
    smoke_parser.add_argument("--seed", type=int, default=2021)
    smoke_parser.add_argument("--batch-size", type=int, default=4)
    smoke_parser.add_argument("--steps", type=int, default=2)
    smoke_parser.add_argument(
        "--loss-weighting", choices=LOSS_WEIGHTINGS, default=LOSS_WEIGHTING_ORDINARY
    )
    smoke_recombination = smoke_parser.add_mutually_exclusive_group()
    smoke_recombination.add_argument(
        "--same-target-recombination-nonstride-sources", action="store_true"
    )
    smoke_recombination.add_argument(
        "--same-target-recombination-all-sources", action="store_true"
    )
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--cache-dir", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--device", default="cuda:0")
    train_parser.add_argument("--seed", type=int, default=2021)
    train_parser.add_argument("--epochs", type=int, default=200)
    train_parser.add_argument("--resume-training-state", type=Path)
    train_parser.add_argument(
        "--loss-weighting", choices=LOSS_WEIGHTINGS, default=LOSS_WEIGHTING_ORDINARY
    )
    train_recombination = train_parser.add_mutually_exclusive_group()
    train_recombination.add_argument(
        "--same-target-recombination-nonstride-sources", action="store_true"
    )
    train_recombination.add_argument(
        "--same-target-recombination-all-sources", action="store_true"
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
