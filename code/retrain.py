"""Run the documented ANN or two-stage SNN reproduction contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from code.paths import CACHE_ROOT, PROJECT_ROOT


def commands(task: str, model: str, user: int, seed: int, cache_root: Path,
             output_root: Path, device: str) -> list[list[str]]:
    cache = cache_root / task / "fixed_event_window_400_stride200_96x128" / f"user{user}/eye0"
    directory = output_root / task / model / f"user{user}/eye0/seed{seed}"
    module = f"code.{task}.fixed_event_window.{model}.training"
    base = [sys.executable, "-m", module, "train", "--cache-dir", str(cache),
            "--device", device, "--seed", str(seed)]
    if task == "pursuit":
        base += ["--local-recombination", "--local-recombination-mode", "all_physical_source_epoch_random"]
    elif model == "ann":
        base += ["--same-target-recombination-all-sources", "--loss-weighting", "ordinary"]
    else:
        base += ["--loss-weighting", "ordinary", "--backend", "exodus"]
    if model == "ann":
        return [base + ["--epochs", "200", "--output-dir", str(directory)]]
    if task == "pursuit":
        base += ["--timesteps", "2", "--temporal-organization", "repeat_full_400",
                 "--alpha", "0.75", "--bias-constraint-layers", "0"]
    common = directory / "full_state_epoch100"
    final = directory / "l1_l2_stateless_epoch200"
    mask = PROJECT_ROOT / "code/pursuit/fixed_event_window/snn/l1_l2_stateless_mask.json"
    return [base + ["--epochs", "100", "--save-training-state-epoch", "100", "--output-dir", str(common)],
            base + ["--epochs", "200", "--stateless-mask-json", str(mask),
                    "--resume-training-state", str(common / "training_state_epoch100.pt"), "--output-dir", str(final)]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("pursuit", "saccade"), required=True)
    parser.add_argument("--model", choices=("ann", "snn"), required=True)
    parser.add_argument("--user", type=int, default=15)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2021])
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan = [c for seed in args.seeds for c in commands(args.task, args.model, args.user, seed,
                                                     args.cache_root, args.output_root, args.device)]
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    for command in plan:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
