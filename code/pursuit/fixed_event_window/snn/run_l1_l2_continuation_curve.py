"""Complete the L1/L2 epoch-100 to epoch-200 continuation curve.

This workflow runs three masks for seeds 2021, 2022 and 2023:

- L1 fully stateless and no L2 stateless channels;
- L1 fully stateless plus the twelve-channel conditional-greedy L2 mask;
- L1 and L2 fully stateless.

Every branch resumes the exact all-state epoch-100 training snapshot and ends
at epoch 200.  Existing four- and eight-channel continuation runs are not
modified.  Failed or partial runs are not retried automatically.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from .run_l1_l2_stateless_controls import (
    LAYER_STATE_ELEMENTS,
    SEEDS,
    TOTAL_STATES,
    audit_epoch100_snapshots,
    read_json,
    run_logged,
    training_command,
    validate_completed_run,
)
from .training import write_json


L1_ALL = list(range(8))
L2_ALL = list(range(16))
L2_SELECTED_COUNT = 12
BRANCH = "continue_epoch100_to200"


def removed_states(mask: dict[str, list[int]]) -> int:
    return sum(
        len(indices) * LAYER_STATE_ELEMENTS[layer]
        for layer, indices in mask.items()
    )


def load_selected_twelve(selection_root: Path, seed: int) -> list[int]:
    selection = read_json(selection_root / f"seed{seed}.json")
    if selection["checkpoint_epoch"] != 100:
        raise AssertionError(f"seed{seed}: selection did not use epoch 100")
    if selection["selection_split"] != "validation" or selection["test_used"]:
        raise AssertionError(f"seed{seed}: invalid selection split")
    rounds = selection["rounds"]
    if len(rounds) < L2_SELECTED_COUNT:
        raise AssertionError(f"seed{seed}: selection stops before twelve channels")
    selected = sorted(rounds[L2_SELECTED_COUNT - 1]["selected"])
    if len(selected) != L2_SELECTED_COUNT or len(set(selected)) != L2_SELECTED_COUNT:
        raise AssertionError(f"seed{seed}: invalid twelve-channel mask {selected}")
    return selected


def build_variants(selection_root: Path) -> dict[str, dict[int, dict[str, list[int]]]]:
    variants: dict[str, dict[int, dict[str, list[int]]]] = {
        "l1_all": {},
        "l1_all_l2_selected12": {},
        "l1_l2_all": {},
    }
    for seed in SEEDS:
        variants["l1_all"][seed] = {"L1": L1_ALL}
        variants["l1_all_l2_selected12"][seed] = {
            "L1": L1_ALL,
            "L2": load_selected_twelve(selection_root, seed),
        }
        variants["l1_l2_all"][seed] = {"L1": L1_ALL, "L2": L2_ALL}
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    variants = build_variants(args.selection_root)
    audit = audit_epoch100_snapshots(args.baseline_root, args.reference_root)
    write_json(args.output_root / "epoch100_prefix_audit.json", audit)

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
            "branch": BRANCH,
            "branch_start_epoch": 100,
            "total_epochs": 200,
            "total_cross_bin_membrane_states": TOTAL_STATES,
            "selection_split": "validation",
            "test_used_for_selection": False,
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
        },
    )

    # GPU 0 receives five jobs because it has been faster in the latest runs.
    tasks = {
        0: [
            ("l1_all", 2021),
            ("l1_all_l2_selected12", 2021),
            ("l1_l2_all", 2021),
            ("l1_all_l2_selected12", 2023),
            ("l1_l2_all", 2023),
        ],
        1: [
            ("l1_all", 2022),
            ("l1_all_l2_selected12", 2022),
            ("l1_l2_all", 2022),
            ("l1_all", 2023),
        ],
    }
    scheduled = [(variant, seed) for rows in tasks.values() for variant, seed in rows]
    expected = [
        (variant, seed)
        for variant in variants
        for seed in SEEDS
    ]
    if sorted(scheduled) != sorted(expected):
        raise AssertionError(f"invalid task schedule: {scheduled}")

    def worker(gpu: int) -> None:
        for variant, seed in tasks[gpu]:
            output_dir = args.output_root / variant / BRANCH / f"seed{seed}"
            if output_dir.exists():
                raise FileExistsError(output_dir)
            name = f"{variant}_{BRANCH}_seed{seed}"
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
        "reference_root": str(args.reference_root.resolve()),
        "total_cross_bin_membrane_states": TOTAL_STATES,
        "variants": {},
    }
    for variant, seed_masks in variants.items():
        rows = []
        for seed in SEEDS:
            mask = seed_masks[seed]
            rows.append(
                validate_completed_run(
                    args.output_root / variant / BRANCH / f"seed{seed}",
                    args.reference_root,
                    seed,
                    100,
                    mask,
                    removed_states(mask),
                )
            )
        summary["variants"][variant] = {BRANCH: rows}
    write_json(args.output_root / "continuation_curve_summary.json", summary)
    print("L1/L2 continuation curve complete", flush=True)


if __name__ == "__main__":
    main()
