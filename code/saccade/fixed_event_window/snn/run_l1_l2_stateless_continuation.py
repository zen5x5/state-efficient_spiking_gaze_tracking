"""Run the matched saccade L1/L2 cross-bin stateless continuation study.

For each seed, this workflow deterministically reproduces the first 100 epochs
of the retained 200-epoch all-state ordinary-loss reference, verifies the
history prefix, then resumes the exact epoch-100 optimizer/scheduler/RNG state
with all L1 and L2 channels made cross-bin stateless until epoch 200.

Jobs are split over two GPUs.  Existing output directories are never silently
overwritten or retried.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from .training import write_json


SEEDS = (2021, 2022, 2023)
STATELESS_MASK = {"L1": list(range(8)), "L2": list(range(16))}
TOTAL_STATES = 45_000
REMOVED_STATES = 36_864
RETAINED_STATES = TOTAL_STATES - REMOVED_STATES


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def histories_match(left: list[dict], right: list[dict]) -> bool:
    if len(left) != len(right):
        return False
    return all(
        {key: value for key, value in a.items() if key != "epoch_seconds"}
        == {key: value for key, value in b.items() if key != "epoch_seconds"}
        for a, b in zip(left, right)
    )


def environment(gpu: int) -> dict[str, str]:
    value = copy.deepcopy(os.environ)
    value["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return value


def training_command(
    *,
    cache_dir: Path,
    output_dir: Path,
    seed: int,
    epochs: int,
    save_epoch: int = 0,
    mask_path: Path | None = None,
    resume_path: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "code.saccade.fixed_event_window.snn.training",
        "train",
        "--cache-dir",
        str(cache_dir),
        "--output-dir",
        str(output_dir),
        "--device",
        "cuda:0",
        "--backend",
        "exodus",
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--loss-weighting",
        "ordinary",
        "--output-encoding",
        "continuous_xy",
    ]
    if save_epoch:
        command += ["--save-training-state-epoch", str(save_epoch)]
    if mask_path is not None:
        command += ["--stateless-mask-json", str(mask_path)]
    if resume_path is not None:
        command += ["--resume-training-state", str(resume_path)]
    return command


def run_logged(command: list[str], gpu: int, log_path: Path, name: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"START {name} gpu={gpu}", flush=True)
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment(gpu),
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            if " epoch=" in line and any(
                marker in line for marker in ("epoch=050", "epoch=100", "epoch=150", "epoch=200")
            ):
                print(f"{name}: {line.strip()}", flush=True)
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"{name} failed ({return_code}); see {log_path}")
    print(f"DONE {name}", flush=True)


def audit_baseline(
    baseline_dir: Path,
    reference_dir: Path,
    seed: int,
) -> dict:
    history = read_json(baseline_dir / "training_history.json")
    reference = read_json(reference_dir / "training_history.json")
    if len(history) != 100 or len(reference) != 200:
        raise AssertionError(f"seed{seed}: expected 100/200 history entries")
    if not histories_match(history, reference[:100]):
        raise AssertionError(f"seed{seed}: reproduced first 100 epochs differ")
    state_path = baseline_dir / "training_state_epoch100.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if int(state["epoch"]) != 100 or state["history"] != history:
        raise AssertionError(f"seed{seed}: invalid epoch-100 training state")
    required = {
        "model",
        "optimizer",
        "scheduler",
        "train_loader_generator_state",
        "python_random_state",
        "numpy_random_state",
        "torch_random_state",
        "cuda_random_states",
        "best_model",
    }
    missing = sorted(required - set(state))
    if missing:
        raise AssertionError(f"seed{seed}: snapshot missing {missing}")
    return {
        "seed": seed,
        "exact_reference_prefix": True,
        "history_entries": 100,
        "snapshot_epoch": 100,
        "best_epoch_through_100": int(state["best_epoch"]),
        "best_validation_angle_through_100": float(
            state["best_validation_mean_angular_error_deg"]
        ),
    }


def validate_continuation(output_dir: Path, seed: int) -> dict:
    history = read_json(output_dir / "training_history.json")
    configuration = read_json(output_dir / "configuration.json")
    summary = read_json(output_dir / "summary.json")
    if len(history) != 200 or history[-1]["epoch"] != 200:
        raise AssertionError(f"seed{seed}: continuation did not reach epoch 200")
    if configuration["model"]["stateless_channels"] != STATELESS_MASK:
        raise AssertionError(f"seed{seed}: stateless mask mismatch")
    if configuration["training"]["branch_start_epoch"] != 100:
        raise AssertionError(f"seed{seed}: branch did not start at epoch 100")
    if configuration["model"]["retained_cross_bin_membrane_states"] != RETAINED_STATES:
        raise AssertionError(f"seed{seed}: retained state count mismatch")
    if summary["stateless_channels"] != STATELESS_MASK:
        raise AssertionError(f"seed{seed}: summary stateless mask mismatch")
    if summary["retained_cross_bin_membrane_states"] != RETAINED_STATES:
        raise AssertionError(f"seed{seed}: summary retained state count mismatch")
    if summary["removed_cross_bin_membrane_states"] != REMOVED_STATES:
        raise AssertionError(f"seed{seed}: summary removed state count mismatch")
    if summary["branch_start_epoch"] != 100:
        raise AssertionError(f"seed{seed}: summary branch start mismatch")
    required = (
        "best_checkpoint.pt",
        "epoch_200_training_state.pt",
        "predictions_train.npz",
        "predictions_validation.npz",
        "predictions_test.npz",
    )
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise AssertionError(f"seed{seed}: missing outputs {missing}")
    return {
        "seed": seed,
        "best_epoch": int(summary["best_epoch"]),
        "train_angle_deg": float(summary["metrics"]["train"]["mean_angular_error_deg"]),
        "validation_angle_deg": float(
            summary["metrics"]["validation"]["mean_angular_error_deg"]
        ),
        "test_angle_deg": float(summary["metrics"]["test"]["mean_angular_error_deg"]),
        "retained_cross_bin_membrane_states": RETAINED_STATES,
        "removed_cross_bin_membrane_states": REMOVED_STATES,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)
    mask_path = args.output_root / "masks" / "l1_l2_all.json"
    mask_path.parent.mkdir(parents=True)
    write_json(mask_path, STATELESS_MASK)
    write_json(
        args.output_root / "experiment_manifest.json",
        {
            "seeds": list(SEEDS),
            "loss_weighting": "ordinary",
            "all_state_epochs": [1, 100],
            "l1_l2_stateless_epochs": [101, 200],
            "stateless_channels": STATELESS_MASK,
            "total_cross_bin_membrane_states": TOTAL_STATES,
            "removed_cross_bin_membrane_states": REMOVED_STATES,
            "retained_cross_bin_membrane_states": RETAINED_STATES,
            "removed_fraction": REMOVED_STATES / TOTAL_STATES,
        },
    )

    baseline_root = args.output_root / "baseline_epoch100"
    continuation_root = (
        args.output_root / "l1_l2_all" / "continue_epoch100_to200"
    )
    schedules = {0: (2021, 2023), 1: (2022,)}

    def worker(gpu: int) -> list[dict]:
        records = []
        for seed in schedules[gpu]:
            baseline_dir = baseline_root / f"seed{seed}"
            run_logged(
                training_command(
                    cache_dir=args.cache_dir,
                    output_dir=baseline_dir,
                    seed=seed,
                    epochs=100,
                    save_epoch=100,
                ),
                gpu,
                args.output_root / "logs" / f"baseline_epoch100_seed{seed}.log",
                f"baseline_epoch100_seed{seed}",
            )
            audit = audit_baseline(
                baseline_dir, args.reference_root / f"seed{seed}", seed
            )
            write_json(baseline_dir / "reference_prefix_audit.json", audit)

            continuation_dir = continuation_root / f"seed{seed}"
            run_logged(
                training_command(
                    cache_dir=args.cache_dir,
                    output_dir=continuation_dir,
                    seed=seed,
                    epochs=200,
                    mask_path=mask_path,
                    resume_path=baseline_dir / "training_state_epoch100.pt",
                ),
                gpu,
                args.output_root / "logs" / f"l1_l2_all_continue_seed{seed}.log",
                f"l1_l2_all_continue_seed{seed}",
            )
            records.append({"prefix": audit, "continuation": validate_continuation(
                continuation_dir, seed
            )})
        return records

    started = time.time()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker, gpu) for gpu in sorted(schedules)]
        records = [record for future in futures for record in future.result()]
    records.sort(key=lambda item: item["prefix"]["seed"])
    write_json(
        args.output_root / "study_summary.json",
        {
            "status": "complete",
            "elapsed_seconds": time.time() - started,
            "runs": records,
        },
    )
    print("saccade L1/L2 stateless continuation complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
