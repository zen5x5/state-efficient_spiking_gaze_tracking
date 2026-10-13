"""Frozen launcher for ordinary-loss grid-count SNN runs.

Seed2021 is retained for reproducibility. Do not launch another seed unless the
user explicitly reopens this research route.
"""

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
DEFAULT_RESULT_ROOT = RESULT_ROOT / (
    "saccade/fixed_event_window/snn/user15/eye0/"
    "formal_visual_fixation_400event_notail_trainstride200_"
    "all_physical_source_recombination_t2_alpha0p75_"
    "grid_spike_count_output_ordinary_epoch200"
)
BATCHES = {
    1: ({"seed": 2021, "gpu": 0}, {"seed": 2022, "gpu": 1}),
    2: ({"seed": 2023, "gpu": 0},),
}


def jobs_for_batch(batch: int) -> list[dict]:
    return [dict(job) for job in BATCHES[batch]]


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
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    jobs = jobs_for_batch(args.batch)
    for job in jobs:
        job["status"] = "ready"
        run_dir = args.result_root / f"seed{job['seed']}"
        job["output_dir"] = str(run_dir)
        job["log"] = str(
            args.result_root / "logs" / f"seed{job['seed']}.log"
        )
        if run_dir.exists():
            raise FileExistsError(run_dir)

    args.result_root.mkdir(parents=True, exist_ok=True)
    (args.result_root / "logs").mkdir(exist_ok=True)
    state = {
        "status": "running",
        "batch": args.batch,
        "started_unix": time.time(),
        "cache_dir": str(args.cache_dir),
        "result_root": str(args.result_root),
        "epochs": args.epochs,
        "timesteps": 2,
        "alpha": 0.75,
        "source_mode": "all_physical",
        "loss_weighting": "ordinary",
        "output_encoding": "grid_spike_count_xy",
        "launch_policy": "immediate_manual_batch",
        "jobs": jobs,
    }
    configuration_path = args.result_root / f"batch{args.batch}_configuration.json"
    progress_path = args.result_root / f"batch{args.batch}_progress.json"
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
            "code.saccade.fixed_event_window.snn.training",
            "train",
            "--cache-dir",
            str(args.cache_dir),
            "--output-dir",
            job["output_dir"],
            "--device",
            "cuda:0",
            "--backend",
            "exodus",
            "--seed",
            str(job["seed"]),
            "--epochs",
            str(args.epochs),
            "--loss-weighting",
            "ordinary",
            "--output-encoding",
            "grid_spike_count_xy",
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
