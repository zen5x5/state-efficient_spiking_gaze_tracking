"""Run the current-protocol L1+L2 cross-bin stateless controls.

The workflow waits for three all-state epoch-100 snapshots, verifies that
their histories and resumed epoch 101 exactly reproduce the retained
epoch-200 references, then freezes each epoch-100 model and conditionally
selects four L2 channels on validation with L1 fully stateless.  The selected
and deterministic-random L2-four masks resume the same epoch-100 state to
epoch 200.  Three matched from-scratch controls train L1-only, L1 plus random
L2-four, and L1+L2-all for 200 epochs.

No failed or partial run is retried automatically.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

import torch

from .training import write_json


SEEDS = (2021, 2022, 2023)
LAYER_STATE_ELEMENTS = {"L1": 3072, "L2": 768}
TOTAL_STATES = 45_000


def stateless_variants(
    selected_l2: dict[int, list[int]],
) -> dict[str, dict[int, dict[str, list[int]]]]:
    variants: dict[str, dict[int, dict[str, list[int]]]] = {
        "l1_all": {},
        "l1_all_l2_random4": {},
        "l1_all_l2_selected4": {},
        "l1_l2_all": {},
    }
    for seed in SEEDS:
        variants["l1_all"][seed] = {"L1": list(range(8))}
        variants["l1_all_l2_random4"][seed] = {
            "L1": list(range(8)),
            "L2": sorted(random.Random(seed).sample(range(16), 4)),
        }
        variants["l1_all_l2_selected4"][seed] = {
            "L1": list(range(8)),
            "L2": sorted(selected_l2[seed]),
        }
        variants["l1_l2_all"][seed] = {
            "L1": list(range(8)),
            "L2": list(range(16)),
        }
    return variants


def removed_states(mask: dict[str, list[int]]) -> int:
    return sum(
        len(indices) * LAYER_STATE_ELEMENTS[layer]
        for layer, indices in mask.items()
    )


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def histories_match(left: list[dict], right: list[dict]) -> bool:
    """Compare numerical training records while excluding wall-clock time."""
    if len(left) != len(right):
        return False
    return all(
        {key: value for key, value in a.items() if key != "epoch_seconds"}
        == {key: value for key, value in b.items() if key != "epoch_seconds"}
        for a, b in zip(left, right)
    )


def baseline_paths_ready(baseline_root: Path) -> bool:
    required = (
        "configuration.json",
        "training_history.json",
        "best_checkpoint.pt",
        "training_state_epoch100.pt",
        "summary.json",
    )
    return all(
        (baseline_root / f"seed{seed}" / name).is_file()
        for seed in SEEDS
        for name in required
    )


def wait_for_baselines(baseline_root: Path, poll_seconds: int) -> None:
    while not baseline_paths_ready(baseline_root):
        progress = []
        for seed in SEEDS:
            history_path = baseline_root / f"seed{seed}" / "training_history.json"
            epochs = len(read_json(history_path)) if history_path.is_file() else 0
            progress.append(f"seed{seed}:{epochs}/100")
        print("waiting for baselines: " + ", ".join(progress), flush=True)
        time.sleep(poll_seconds)


def audit_epoch100_snapshots(
    baseline_root: Path,
    reference_root: Path,
) -> dict:
    audit = {"seeds": {}, "exact_first_100": True}
    required_snapshot_keys = {
        "epoch",
        "model",
        "optimizer",
        "scheduler",
        "train_loader_generator_state",
        "python_random_state",
        "numpy_random_state",
        "torch_random_state",
        "cuda_random_states",
        "history",
        "best_epoch",
        "best_validation_mean_angular_error_deg",
        "best_model",
    }
    for seed in SEEDS:
        baseline_dir = baseline_root / f"seed{seed}"
        new_history = read_json(baseline_dir / "training_history.json")
        reference_history = read_json(
            reference_root / f"seed{seed}" / "training_history.json"
        )
        if len(new_history) != 100 or len(reference_history) != 200:
            raise AssertionError(
                f"seed{seed}: expected 100/200 history entries, got "
                f"{len(new_history)}/{len(reference_history)}"
            )
        if not histories_match(new_history, reference_history[:100]):
            raise AssertionError(f"seed{seed}: first 100 epochs differ")
        configuration = read_json(baseline_dir / "configuration.json")
        if configuration["training"]["epochs"] != 100:
            raise AssertionError(f"seed{seed}: baseline epochs is not 100")
        if configuration["training"]["full_training_state_snapshot_epoch"] != 100:
            raise AssertionError(f"seed{seed}: snapshot epoch is not configured as 100")
        snapshot = torch.load(
            baseline_dir / "training_state_epoch100.pt",
            map_location="cpu",
            weights_only=False,
        )
        missing = sorted(required_snapshot_keys - set(snapshot))
        if missing:
            raise AssertionError(f"seed{seed}: snapshot missing keys {missing}")
        if int(snapshot["epoch"]) != 100:
            raise AssertionError(f"seed{seed}: snapshot epoch is not 100")
        if snapshot["history"] != new_history:
            raise AssertionError(f"seed{seed}: snapshot history differs from disk")
        audit["seeds"][str(seed)] = {
            "history_entries": 100,
            "exact_reference_prefix": True,
            "snapshot_epoch": 100,
            "snapshot_keys": sorted(snapshot),
            "best_epoch_through_100": int(snapshot["best_epoch"]),
            "best_validation_angle_through_100": float(
                snapshot["best_validation_mean_angular_error_deg"]
            ),
        }
        del snapshot
    return audit


def training_command(
    *,
    cache_dir: Path,
    output_dir: Path,
    seed: int,
    epochs: int,
    mask_path: Path | None,
    resume_path: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "code.pursuit.fixed_event_window.snn.training",
        "train",
        "--cache-dir",
        str(cache_dir),
        "--output-dir",
        str(output_dir),
        "--timesteps",
        "2",
        "--temporal-organization",
        "repeat_full_400",
        "--device",
        "cuda:0",
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--alpha",
        "0.75",
        "--bias-constraint-layers",
        "0",
        "--local-recombination",
        "--local-recombination-mode",
        "all_physical_source_epoch_random",
        "--recombination-max-label-distance-px",
        "32",
    ]
    if mask_path is not None:
        command += ["--stateless-mask-json", str(mask_path)]
    if resume_path is not None:
        command += ["--resume-training-state", str(resume_path)]
    return command


def selection_command(
    *,
    baseline_dir: Path,
    cache_dir: Path,
    output_path: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "code.pursuit.fixed_event_window.snn.select_l2_stateless_channels",
        "--baseline-dir",
        str(baseline_dir),
        "--cache-dir",
        str(cache_dir),
        "--output-path",
        str(output_path),
        "--device",
        "cuda:0",
    ]


def subprocess_environment(gpu: int) -> dict[str, str]:
    environment = copy.deepcopy(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return environment


def run_logged(command: list[str], gpu: int, log_path: Path, task_name: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"START {task_name} gpu={gpu}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=subprocess_environment(gpu),
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()
            if " epoch=" in line and (
                "epoch=100" in line or "epoch=200" in line
            ):
                print(f"{task_name}: {line.strip()}", flush=True)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"{task_name} failed with exit code {return_code}; see {log_path}"
        )
    print(f"DONE {task_name}", flush=True)


def audit_resumed_epoch101(
    *,
    cache_dir: Path,
    baseline_root: Path,
    reference_root: Path,
    seed: int,
    gpu: int,
) -> dict:
    with tempfile.TemporaryDirectory(
        prefix=f"ebv-pursuit-resume-seed{seed}-", dir="/tmp"
    ) as temporary:
        output_dir = Path(temporary) / "epoch101"
        command = training_command(
            cache_dir=cache_dir,
            output_dir=output_dir,
            seed=seed,
            epochs=101,
            mask_path=None,
            resume_path=(
                baseline_root / f"seed{seed}" / "training_state_epoch100.pt"
            ),
        )
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=subprocess_environment(gpu),
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stdout)
        actual = read_json(output_dir / "training_history.json")[100]
        reference = read_json(
            reference_root / f"seed{seed}" / "training_history.json"
        )[100]
        if not histories_match([actual], [reference]):
            raise AssertionError(f"seed{seed}: resumed epoch101 differs from reference")
        return {
            "seed": seed,
            "exact_epoch101": True,
            "train_loss": actual["train"]["mean_loss"],
            "validation_angle": actual["validation"]["mean_angular_error_deg"],
        }


def validate_completed_run(
    output_dir: Path,
    reference_root: Path,
    seed: int,
    expected_branch_start: int,
    expected_mask: dict[str, list[int]],
    expected_removed_states: int,
) -> dict:
    history = read_json(output_dir / "training_history.json")
    configuration = read_json(output_dir / "configuration.json")
    summary = read_json(output_dir / "summary.json")
    if len(history) != 200 or history[-1]["epoch"] != 200:
        raise AssertionError(f"{output_dir}: incomplete 200-epoch history")
    if expected_branch_start == 100:
        reference = read_json(
            reference_root / f"seed{seed}" / "training_history.json"
        )
        if not histories_match(history[:100], reference[:100]):
            raise AssertionError(f"{output_dir}: inherited prefix differs")
    if configuration["model"]["stateless_channels"] != expected_mask:
        raise AssertionError(f"{output_dir}: wrong stateless mask")
    if int(summary["branch_start_epoch"]) != expected_branch_start:
        raise AssertionError(f"{output_dir}: wrong branch start")
    if int(summary["removed_cross_bin_membrane_states"]) != expected_removed_states:
        raise AssertionError(f"{output_dir}: wrong removed state count")
    if (
        int(summary["retained_cross_bin_membrane_states"])
        != TOTAL_STATES - expected_removed_states
    ):
        raise AssertionError(f"{output_dir}: wrong retained state count")
    return {
        "seed": seed,
        "best_epoch": int(summary["best_epoch"]),
        "branch_start_epoch": expected_branch_start,
        "train": summary["best_train"],
        "validation": summary["best_validation"],
        "test": summary["test"],
        "stateless_channels": expected_mask,
        "removed_cross_bin_membrane_states": expected_removed_states,
        "retained_cross_bin_membrane_states": TOTAL_STATES - expected_removed_states,
        "total_seconds": float(summary["total_seconds"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    wait_for_baselines(args.baseline_root, args.poll_seconds)
    args.output_root.mkdir(parents=True, exist_ok=False)
    audit = audit_epoch100_snapshots(args.baseline_root, args.reference_root)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first_two = [
            executor.submit(
                audit_resumed_epoch101,
                cache_dir=args.cache_dir,
                baseline_root=args.baseline_root,
                reference_root=args.reference_root,
                seed=seed,
                gpu=gpu,
            )
            for seed, gpu in ((2021, 0), (2022, 1))
        ]
        resumed = [future.result() for future in first_two]
    resumed.append(
        audit_resumed_epoch101(
            cache_dir=args.cache_dir,
            baseline_root=args.baseline_root,
            reference_root=args.reference_root,
            seed=2023,
            gpu=0,
        )
    )
    audit["resumed_epoch101"] = resumed
    audit["exact_resumed_epoch101"] = all(
        row["exact_epoch101"] for row in resumed
    )
    write_json(args.output_root / "epoch100_resume_audit.json", audit)
    print("epoch100 histories and resumed epoch101 are exact", flush=True)

    selection_paths = {
        seed: args.output_root / "selection" / f"seed{seed}.json"
        for seed in SEEDS
    }

    def select(seed: int, gpu: int) -> None:
        run_logged(
            selection_command(
                baseline_dir=args.baseline_root / f"seed{seed}",
                cache_dir=args.cache_dir,
                output_path=selection_paths[seed],
            ),
            gpu,
            args.output_root / "logs" / f"select_l2_four_seed{seed}.log",
            f"select_l2_four_seed{seed}",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(select, seed, gpu)
            for seed, gpu in ((2021, 0), (2022, 1))
        ]
        for future in futures:
            future.result()
    select(2023, 0)

    selections = {seed: read_json(selection_paths[seed]) for seed in SEEDS}
    selected_l2 = {
        seed: selections[seed]["selected_mask"]["L2"] for seed in SEEDS
    }
    for seed, channels in selected_l2.items():
        if len(channels) != 4 or len(set(channels)) != 4:
            raise AssertionError(f"seed{seed}: invalid selected L2 mask {channels}")

    variants = stateless_variants(selected_l2)
    mask_paths: dict[tuple[str, int], Path] = {}
    for variant, seed_masks in variants.items():
        for seed, mask in seed_masks.items():
            mask_path = args.output_root / "masks" / variant / f"seed{seed}.json"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(mask_path, mask)
            mask_paths[(variant, seed)] = mask_path

    write_json(
        args.output_root / "experiment_manifest.json",
        {
            "seeds": list(SEEDS),
            "total_cross_bin_membrane_states": TOTAL_STATES,
            "selection": {
                "split": "validation",
                "test_used": False,
                "method": "conditional greedy L2 selection with frozen epoch100 model and L1 fully stateless",
                "candidate_evaluations_per_seed": 58,
            },
            "variants": {
                variant: {
                    str(seed): {
                        "stateless_channels": mask,
                        "removed_cross_bin_membrane_states": removed_states(mask),
                        "removed_fraction": removed_states(mask) / TOTAL_STATES,
                    }
                    for seed, mask in seed_masks.items()
                }
                for variant, seed_masks in variants.items()
            },
            "controls_by_variant": {
                "l1_all": ["scratch_epoch200"],
                "l1_all_l2_random4": [
                    "continue_epoch100_to200",
                    "scratch_epoch200",
                ],
                "l1_all_l2_selected4": ["continue_epoch100_to200"],
                "l1_l2_all": ["scratch_epoch200"],
            },
        },
    )

    def task(variant: str, branch: str, seed: int, resume: bool):
        return (
            f"{variant}_{branch}_seed{seed}",
            variant,
            branch,
            seed,
            resume,
        )

    scratch_variants = ("l1_all", "l1_all_l2_random4", "l1_l2_all")
    task_lists = {
        0: (
            [
                task(variant, "continue_epoch100_to200", seed, True)
                for seed in (2021, 2023)
                for variant in (
                    "l1_all_l2_selected4",
                    "l1_all_l2_random4",
                )
            ]
            + [
                task(name, "scratch_epoch200", 2021, False)
                for name in scratch_variants
            ]
            + [task("l1_all", "scratch_epoch200", 2023, False)]
        ),
        1: (
            [
                task(variant, "continue_epoch100_to200", 2022, True)
                for variant in (
                    "l1_all_l2_selected4",
                    "l1_all_l2_random4",
                )
            ]
            + [
                task(name, "scratch_epoch200", 2022, False)
                for name in scratch_variants
            ]
            + [
                task(name, "scratch_epoch200", 2023, False)
                for name in scratch_variants
                if name != "l1_all"
            ]
        ),
    }
    scheduled_names = [task_row[0] for tasks in task_lists.values() for task_row in tasks]
    if len(scheduled_names) != 15 or len(set(scheduled_names)) != 15:
        raise AssertionError(
            f"expected 15 unique stateless training runs, got {scheduled_names}"
        )

    def worker(gpu: int) -> None:
        for name, variant, branch, seed, resume in task_lists[gpu]:
            output_dir = args.output_root / variant / branch / f"seed{seed}"
            if output_dir.exists():
                raise FileExistsError(output_dir)
            command = training_command(
                cache_dir=args.cache_dir,
                output_dir=output_dir,
                seed=seed,
                epochs=200,
                mask_path=mask_paths[(variant, seed)],
                resume_path=(
                    args.baseline_root
                    / f"seed{seed}"
                    / "training_state_epoch100.pt"
                    if resume
                    else None
                ),
            )
            run_logged(
                command,
                gpu,
                args.output_root / "logs" / f"{name}.log",
                name,
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker, gpu) for gpu in (0, 1)]
        for future in futures:
            future.result()

    summary: dict[str, object] = {
        "baseline_root": str(args.baseline_root.resolve()),
        "reference_root": str(args.reference_root.resolve()),
        "total_cross_bin_membrane_states": TOTAL_STATES,
        "variants": {},
    }
    for variant, seed_masks in variants.items():
        if variant == "l1_all_l2_selected4":
            branch_specs = [("continue_epoch100_to200", 100)]
        elif variant == "l1_all_l2_random4":
            branch_specs = [
                ("continue_epoch100_to200", 100),
                ("scratch_epoch200", 0),
            ]
        else:
            branch_specs = [("scratch_epoch200", 0)]
        variant_summary = {branch: [] for branch, _ in branch_specs}
        for branch, branch_start in branch_specs:
            for seed in SEEDS:
                mask = seed_masks[seed]
                variant_summary[branch].append(
                    validate_completed_run(
                        args.output_root / variant / branch / f"seed{seed}",
                        args.reference_root,
                        seed,
                        branch_start,
                        mask,
                        removed_states(mask),
                    )
                )
        summary["variants"][variant] = variant_summary
    write_json(args.output_root / "workflow_summary.json", summary)
    print("workflow complete", flush=True)


if __name__ == "__main__":
    main()
