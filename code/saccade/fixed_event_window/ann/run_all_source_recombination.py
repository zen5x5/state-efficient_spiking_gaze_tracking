"""Immediately launch one manual batch of all-physical-source ANN runs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from code.paths import PROJECT_ROOT, RESULT_ROOT, saccade_cache_root

DEFAULT_CACHE = saccade_cache_root() / "user15/eye0"
RESULT_PARENT = RESULT_ROOT / "saccade/fixed_event_window/ann/user15/eye0"
LOSS_WEIGHTINGS = ("ordinary", "target_sqrt")


def default_result_root(loss_weighting: str) -> Path:
    if loss_weighting not in LOSS_WEIGHTINGS:
        raise ValueError(loss_weighting)
    return RESULT_PARENT / (
        "formal_visual_fixation_400event_notail_trainstride200_"
        f"all_physical_source_recombination_{loss_weighting}_epoch200"
    )


DEFAULT_RESULT_ROOT = default_result_root("ordinary")
BATCHES = {
    1: (
        {"seed": 2021, "gpu": 0},
        {"seed": 2022, "gpu": 1},
    ),
    2: ({"seed": 2023, "gpu": 0},),
}


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=sorted(BATCHES), required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--result-root", type=Path)
    parser.add_argument(
        "--loss-weighting", choices=LOSS_WEIGHTINGS, default="ordinary"
    )
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    result_root = args.result_root or default_result_root(args.loss_weighting)

    jobs = [dict(job) for job in BATCHES[args.batch]]
    for job in jobs:
        job["status"] = "ready"
        job["output_dir"] = str(result_root / f"seed{job['seed']}")
        job["log"] = str(result_root / "logs" / f"seed{job['seed']}.log")
        if Path(job["output_dir"]).exists():
            raise FileExistsError(job["output_dir"])

    result_root.mkdir(parents=True, exist_ok=True)
    (result_root / "logs").mkdir(exist_ok=True)
    state = {
        "status": "running",
        "batch": args.batch,
        "started_unix": time.time(),
        "cache_dir": str(args.cache_dir),
        "result_root": str(result_root),
        "epochs": args.epochs,
        "loss_weighting": args.loss_weighting,
        "source_mode": "all_physical",
        "augmentation": "same-target online 200+200 from all physical train sources",
        "launch_policy": "immediate_manual_batch",
        "jobs": jobs,
    }
    configuration_path = result_root / f"batch{args.batch}_configuration.json"
    progress_path = result_root / f"batch{args.batch}_progress.json"
    atomic_json(configuration_path, state)
    atomic_json(progress_path, state)
    lock = threading.Lock()

    def update(job: dict, **values: object) -> None:
        with lock:
            job.update(values)
            atomic_json(progress_path, state)

    def worker(job: dict) -> None:
        command = [
            sys.executable,
            "-m",
            "code.saccade.fixed_event_window.ann.training",
            "train",
            "--cache-dir",
            str(args.cache_dir),
            "--output-dir",
            job["output_dir"],
            "--device",
            "cuda:0",
            "--seed",
            str(job["seed"]),
            "--epochs",
            str(args.epochs),
            "--loss-weighting",
            args.loss_weighting,
            "--same-target-recombination-all-sources",
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
        with Path(job["log"]).open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            update(
                job,
                status="running",
                pid=process.pid,
                started_unix=time.time(),
                command=command,
            )
            return_code = process.wait()
        update(
            job,
            status="complete" if return_code == 0 else "failed",
            return_code=return_code,
            finished_unix=time.time(),
        )

    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = [executor.submit(worker, job) for job in jobs]
        for future in futures:
            future.result()

    state["status"] = (
        "complete" if all(job["status"] == "complete" for job in jobs) else "failed"
    )
    state["finished_unix"] = time.time()
    atomic_json(progress_path, state)
    return 0 if state["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
