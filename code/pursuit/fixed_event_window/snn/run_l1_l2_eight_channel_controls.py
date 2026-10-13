"""Run the current-protocol L1-all plus L2-eight stateless controls.

For each of seeds 2021, 2022 and 2023, this workflow:

1. freezes the exact epoch-100 all-state model and greedily selects eight L2
   channels on validation with all eight L1 channels stateless;
2. resumes that selected mask from epoch 100 through epoch 200;
3. draws a deterministic random eight-channel L2 mask and trains it from
   scratch for 200 epochs.

The workflow creates six training runs.  Test data never participates in
channel selection.  Failed or partial runs are not retried automatically.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import sys

from .run_l1_l2_stateless_controls import (
    LAYER_STATE_ELEMENTS,
    SEEDS,
    TOTAL_STATES,
    audit_epoch100_snapshots,
    run_logged,
    training_command,
    validate_completed_run,
)
from .training import write_json


L1_ALL = list(range(8))
L2_WIDTH = 16
L2_STATELESS_COUNT = 8


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def removed_states(mask: dict[str, list[int]]) -> int:
    return sum(
        len(indices) * LAYER_STATE_ELEMENTS[layer]
        for layer, indices in mask.items()
    )


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
        "--num-l2-channels",
        str(L2_STATELESS_COUNT),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=False)
    audit = audit_epoch100_snapshots(args.baseline_root, args.reference_root)
    write_json(args.output_root / "epoch100_prefix_audit.json", audit)
    print("epoch100 histories exactly match the retained references", flush=True)

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
            args.output_root / "logs" / f"select_l2_eight_seed{seed}.log",
            f"select_l2_eight_seed{seed}",
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
    variants: dict[str, dict[int, dict[str, list[int]]]] = {
        "l1_all_l2_selected8": {},
        "l1_all_l2_random8": {},
    }
    for seed in SEEDS:
        selected = selections[seed]["selected_mask"]["L2"]
        if len(selected) != L2_STATELESS_COUNT or len(set(selected)) != L2_STATELESS_COUNT:
            raise AssertionError(f"seed{seed}: invalid selected L2 mask {selected}")
        variants["l1_all_l2_selected8"][seed] = {
            "L1": L1_ALL,
            "L2": sorted(selected),
        }
        variants["l1_all_l2_random8"][seed] = {
            "L1": L1_ALL,
            "L2": sorted(
                random.Random(seed).sample(range(L2_WIDTH), L2_STATELESS_COUNT)
            ),
        }

    mask_paths: dict[tuple[str, int], Path] = {}
    for variant, seed_masks in variants.items():
        for seed, mask in seed_masks.items():
            mask_path = args.output_root / "masks" / variant / f"seed{seed}.json"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(mask_path, mask)
            mask_paths[(variant, seed)] = mask_path

    expected_removed_states = removed_states(
        {"L1": L1_ALL, "L2": list(range(L2_STATELESS_COUNT))}
    )
    if expected_removed_states != 30_720:
        raise AssertionError(
            f"unexpected L1+L2-eight state count {expected_removed_states}"
        )
    write_json(
        args.output_root / "experiment_manifest.json",
        {
            "seeds": list(SEEDS),
            "total_cross_bin_membrane_states": TOTAL_STATES,
            "selection": {
                "split": "validation",
                "test_used": False,
                "method": (
                    "conditional greedy selection of eight L2 channels with "
                    "L1 fully stateless"
                ),
                "candidate_evaluations_per_seed": sum(
                    L2_WIDTH - index for index in range(L2_STATELESS_COUNT)
                ),
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
                "l1_all_l2_selected8": ["continue_epoch100_to200"],
                "l1_all_l2_random8": ["scratch_epoch200"],
            },
        },
    )

    tasks = {
        0: [
            ("l1_all_l2_selected8", "continue_epoch100_to200", 2021, True),
            ("l1_all_l2_random8", "scratch_epoch200", 2021, False),
            ("l1_all_l2_random8", "scratch_epoch200", 2023, False),
        ],
        1: [
            ("l1_all_l2_selected8", "continue_epoch100_to200", 2022, True),
            ("l1_all_l2_random8", "scratch_epoch200", 2022, False),
            ("l1_all_l2_selected8", "continue_epoch100_to200", 2023, True),
        ],
    }
    scheduled = [row for gpu_tasks in tasks.values() for row in gpu_tasks]
    if len(scheduled) != 6 or len(set(scheduled)) != 6:
        raise AssertionError(f"expected six unique training runs, got {scheduled}")

    def worker(gpu: int) -> None:
        for variant, branch, seed, resume in tasks[gpu]:
            name = f"{variant}_{branch}_seed{seed}"
            output_dir = args.output_root / variant / branch / f"seed{seed}"
            if output_dir.exists():
                raise FileExistsError(output_dir)
            run_logged(
                training_command(
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
                ),
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
        branch = (
            "continue_epoch100_to200"
            if variant == "l1_all_l2_selected8"
            else "scratch_epoch200"
        )
        branch_start = 100 if variant == "l1_all_l2_selected8" else 0
        rows = []
        for seed in SEEDS:
            mask = seed_masks[seed]
            rows.append(
                validate_completed_run(
                    args.output_root / variant / branch / f"seed{seed}",
                    args.reference_root,
                    seed,
                    branch_start,
                    mask,
                    removed_states(mask),
                )
            )
        summary["variants"][variant] = {branch: rows}
    write_json(args.output_root / "workflow_summary.json", summary)
    print("L1+L2-eight workflow complete", flush=True)


if __name__ == "__main__":
    main()
