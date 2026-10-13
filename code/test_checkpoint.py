"""Evaluate a saved ANN/SNN checkpoint on the original validation or test split."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import torch


def evaluate_checkpoint(task: str, family: str, result_dir: Path, cache_dir: Path,
                        device: str, split: str = "test", backend: str = "exodus",
                        max_batches: int | None = None) -> dict:
    config = json.loads((result_dir / "configuration.json").read_text())
    prefix = f"code.{task}.fixed_event_window.{family}"
    training = importlib.import_module(prefix + ".training")
    evaluation = importlib.import_module(prefix + ".evaluation")
    seed = int(config["training"]["seed"])
    if family == "ann":
        loader_module = importlib.import_module(prefix + ".dataloader")
        if task == "pursuit":
            model = training.make_model(config["model"]["architecture"])
        else:
            model = importlib.import_module(prefix + ".model").StaticSaccadeANN()
        _, loader = loader_module.make_dataloader(cache_dir, split, 16, seed)
    else:
        module = importlib.import_module(prefix + ".model")
        model_class = module.SpikingPursuitSNN if task == "pursuit" else module.SpikingSaccadeSNN
        model = model_class(alpha=float(config["model"]["alpha"]), backend=backend,
                            stateless_channels=config["model"].get("stateless_channels", {}))
        loader_module = importlib.import_module(f"code.{task}.fixed_event_window.ann.dataloader")
        _, loader = loader_module.make_dataloader(cache_dir, split, 16, seed)
    checkpoint = torch.load(result_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    if family == "snn":
        training.load_persistent_model_state_dict(model, checkpoint["model"])
    else:
        model.load_state_dict(checkpoint["model"])
    model.to(torch.device(device))
    options = {"max_batches": max_batches}
    if family == "snn":
        options["timesteps"] = int(config["input"]["temporal_steps"])
    metrics = evaluation.evaluate(model, loader, torch.device(device), **options)
    return {"task": task, "model": family, "split": split, "checkpoint_epoch": checkpoint["epoch"],
            "backend": backend if family == "snn" else "torch", "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("pursuit", "saccade"), required=True)
    parser.add_argument("--model", choices=("ann", "snn"), required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=("exodus", "sinabs"), default="exodus")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_checkpoint(args.task, args.model, args.result_dir, args.cache_dir,
                                 args.device, args.split, args.backend)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in result if key != "metrics"} |
                     {"samples": result["metrics"]["samples"],
                      "mean_pixel_error": result["metrics"]["mean_pixel_error"],
                      "mean_angular_error_deg": result["metrics"]["mean_angular_error_deg"]}, indent=2))


if __name__ == "__main__":
    main()
