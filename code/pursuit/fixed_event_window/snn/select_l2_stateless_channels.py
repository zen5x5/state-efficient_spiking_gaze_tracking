"""Select L2 stateless channels from a frozen epoch-100 checkpoint.

L1 is fixed to all-stateless.  L2 channels are then added conditionally, one
at a time, using validation angular error only.  The unchanged L1 prefix is
cached so candidate evaluations do not recompute it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import torch

from ..ann.metrics import angular_error_deg, pixel_error
from .dataloader import make_snn_dataloader, prepare_event_bins
from .model import SpikingPursuitSNN
from .training import load_persistent_model_state_dict, write_json


L1_ALL = list(range(8))
L2_WIDTH = 16


class FrozenL2Evaluator:
    """Evaluate L2 masks while reusing the frozen L1 output currents."""

    def __init__(
        self,
        baseline_dir: Path,
        cache_dir: Path,
        device: torch.device,
    ) -> None:
        configuration = json.loads(
            (baseline_dir / "configuration.json").read_text(encoding="utf-8")
        )
        self.seed = int(configuration["training"]["seed"])
        self.device = device
        _, loader = make_snn_dataloader(
            cache_dir,
            "validation",
            16,
            self.seed,
            "repeat_full_400",
        )
        self.batches = list(loader)
        self.targets = [
            batch["target_row_col_px"].to(device) for batch in self.batches
        ]
        self.model = SpikingPursuitSNN(alpha=0.75, backend="exodus").to(device)
        state = torch.load(
            baseline_dir / "training_state_epoch100.pt",
            map_location="cpu",
            weights_only=False,
        )
        if int(state["epoch"]) != 100:
            raise ValueError("expected an epoch-100 training snapshot")
        load_persistent_model_state_dict(self.model, state["model"])
        self.model.eval()
        self.l2_currents: list[torch.Tensor] = []

    def metrics(self, predictions: list[torch.Tensor]) -> dict[str, float]:
        angular = torch.cat(
            [
                angular_error_deg(prediction, target)
                for prediction, target in zip(predictions, self.targets)
            ]
        )
        pixels = torch.cat(
            [
                pixel_error(prediction, target)
                for prediction, target in zip(predictions, self.targets)
            ]
        )
        return {
            "mean_pixel_error": float(pixels.double().mean()),
            "mean_angular_error_deg": float(angular.double().mean()),
        }

    @torch.no_grad()
    def full_predictions(
        self,
        mask: dict[str, list[int]],
        *,
        capture_l2_current: bool = False,
    ) -> list[torch.Tensor]:
        self.model.set_stateless_channels(mask)
        captured: list[torch.Tensor] = []

        def capture(_module, inputs):
            captured.append(inputs[0].detach())

        hook = None
        if capture_l2_current:
            hook = self.model.blocks[1].spike.register_forward_pre_hook(capture)
        predictions = []
        try:
            for batch in self.batches:
                values = prepare_event_bins(
                    batch,
                    2,
                    "repeat_full_400",
                ).to(self.device)
                predictions.append(self.model(values))
        finally:
            if hook is not None:
                hook.remove()
        if capture_l2_current:
            if len(captured) != len(self.batches):
                raise AssertionError("failed to capture one L2 current per batch")
            self.l2_currents = captured
        return predictions

    @torch.no_grad()
    def suffix_predictions(self, l2_channels: list[int]) -> list[torch.Tensor]:
        if not self.l2_currents:
            raise RuntimeError("L2 currents have not been prepared")
        self.model.set_stateless_channels(
            {"L1": L1_ALL, "L2": sorted(l2_channels)}
        )
        predictions = []
        for current in self.l2_currents:
            self.model.reset_states()
            values = self.model.blocks[1].spike(current)
            for layer_index in range(2, 8):
                values = self.model.blocks[layer_index](values)
            values = values.reshape(values.shape[0], 2, 480)
            values = self.model.hidden2(self.model.hidden1(values))
            predictions.append(self.model.readout(values.mean(dim=1)))
        return predictions


def select_channels(
    baseline_dir: Path,
    cache_dir: Path,
    output_path: Path,
    device: torch.device,
    num_l2_channels: int = 4,
) -> dict:
    if not 1 <= num_l2_channels <= L2_WIDTH:
        raise ValueError(
            f"num_l2_channels must be in [1, {L2_WIDTH}], got {num_l2_channels}"
        )
    started = time.perf_counter()
    evaluator = FrozenL2Evaluator(baseline_dir, cache_dir, device)
    all_state = evaluator.metrics(evaluator.full_predictions({}))
    l1_predictions = evaluator.full_predictions(
        {"L1": L1_ALL},
        capture_l2_current=True,
    )
    l1_only = evaluator.metrics(l1_predictions)
    cached_l1 = evaluator.suffix_predictions([])
    cache_max_difference = max(
        float((full - cached).abs().max())
        for full, cached in zip(l1_predictions, cached_l1)
    )
    if cache_max_difference > 1e-7:
        raise AssertionError(
            f"cached L2 suffix differs from full forward: {cache_max_difference}"
        )

    selected: list[int] = []
    rounds = []
    for selection_round in range(1, num_l2_channels + 1):
        candidates = []
        for channel in range(L2_WIDTH):
            if channel in selected:
                continue
            mask = sorted([*selected, channel])
            metrics = evaluator.metrics(evaluator.suffix_predictions(mask))
            candidates.append({"channel_added": channel, "L2": mask, **metrics})
        chosen = min(
            candidates,
            key=lambda row: (row["mean_angular_error_deg"], row["channel_added"]),
        )
        selected = list(chosen["L2"])
        rounds.append(
            {
                "round": selection_round,
                "selected": selected,
                "chosen_metrics": {
                    "mean_pixel_error": chosen["mean_pixel_error"],
                    "mean_angular_error_deg": chosen[
                        "mean_angular_error_deg"
                    ],
                },
                "candidates": candidates,
            }
        )
        print(
            f"seed{evaluator.seed} selection_round={selection_round} "
            f"L2={selected} "
            f"validation_angle={chosen['mean_angular_error_deg']:.6f}",
            flush=True,
        )

    selected_suffix = evaluator.suffix_predictions(selected)
    selected_full = evaluator.full_predictions({"L1": L1_ALL, "L2": selected})
    selected_max_difference = max(
        float((full - cached).abs().max())
        for full, cached in zip(selected_full, selected_suffix)
    )
    if selected_max_difference > 1e-7:
        raise AssertionError(
            "selected-mask cached suffix differs from full forward: "
            f"{selected_max_difference}"
        )

    random_l2 = sorted(
        random.Random(evaluator.seed).sample(range(L2_WIDTH), num_l2_channels)
    )
    random_validation = evaluator.metrics(evaluator.suffix_predictions(random_l2))

    result = {
        "seed": evaluator.seed,
        "checkpoint": str(
            (baseline_dir / "training_state_epoch100.pt").resolve()
        ),
        "checkpoint_epoch": 100,
        "selection_split": "validation",
        "test_used": False,
        "method": (
            f"conditional greedy addition of {num_l2_channels} L2 channels "
            "with L1 fully stateless"
        ),
        "objective": "minimum validation mean angular error at each round",
        "selected_l2_channel_count": num_l2_channels,
        "candidate_evaluations": sum(
            L2_WIDTH - index for index in range(num_l2_channels)
        ),
        "all_state_validation": all_state,
        "l1_all_stateless_validation": l1_only,
        "selected_mask": {"L1": L1_ALL, "L2": selected},
        "selected_validation": evaluator.metrics(selected_full),
        "random_control_mask": {"L1": L1_ALL, "L2": random_l2},
        "random_control_validation": random_validation,
        "rounds": rounds,
        "cache_equivalence": {
            "l1_only_max_output_difference": cache_max_difference,
            "selected_max_output_difference": selected_max_difference,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output_path, result)
    return result


def select_four(
    baseline_dir: Path,
    cache_dir: Path,
    output_path: Path,
    device: torch.device,
) -> dict:
    """Backward-compatible four-channel entry point."""
    return select_channels(
        baseline_dir,
        cache_dir,
        output_path,
        device,
        num_l2_channels=4,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-l2-channels", type=int, default=4)
    args = parser.parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    select_channels(
        args.baseline_dir,
        args.cache_dir,
        args.output_path,
        torch.device(args.device),
        num_l2_channels=args.num_l2_channels,
    )


if __name__ == "__main__":
    main()
