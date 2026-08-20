from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.dataset as arrow_dataset
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]

from loitering_munition_damage_twin.surrogate.artifacts import sha256_file
from loitering_munition_damage_twin.surrogate.dataset import FEATURE_COLUMNS
from loitering_munition_damage_twin.surrogate.features import (
    COMPONENT_PROXY_FEATURE_COLUMNS,
    augment_terminal_physics_features,
)
from loitering_munition_damage_twin.surrogate.model import (
    DEFAULT_ORDINAL_APPLICABILITY,
    component_probabilities_to_ordinal,
)

from loitering_munition_damage_twin.experiments.analyze_validation_threshold_feasibility import (
    evaluate_cell_threshold_feasibility,
)


REPORT_SCHEMA = "stage0_fragment_proxy_quality_v1"
MUNITION_NAMES = ("Small", "Med-LM", "Med-RD", "Heavy")
TASK_NAMES = ("K", "M", "F", "C")


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def _read_validation(dataset_path: Path):
    mechanism_columns = [
        f"{mechanism}_{task}_ge{level}_prob"
        for mechanism in ("fragment", "shock")
        for task in TASK_NAMES
        for level in (1, 2)
    ]
    columns = list(dict.fromkeys([
        *FEATURE_COLUMNS,
        "sample_id",
        "split_role",
        "munition_id",
        *[f"{task}_level" for task in TASK_NAMES],
        *mechanism_columns,
    ]))
    dataset = arrow_dataset.dataset(
        str(dataset_path), format="parquet")
    frame = dataset.to_table(
        columns=columns,
        filter=arrow_dataset.field("split_role") == "val",
    ).to_pandas()
    if frame.empty or not frame["split_role"].eq("val").all():
        raise RuntimeError(
            "Validation-only Parquet predicate was not enforced.")
    return frame


def _task_mechanism_targets(frame, mechanism: str) -> np.ndarray:
    return np.stack([
        frame[f"{mechanism}_{task}_ge{level}_prob"].to_numpy(
            dtype=np.float64)
        for task in TASK_NAMES
        for level in (1, 2)
    ], axis=1).reshape(len(frame), len(TASK_NAMES), 2)


def _fragment_component_tree(frame, armor_aware: bool) -> np.ndarray:
    augmented = augment_terminal_physics_features(
        frame,
        copy=True,
        include_component_proxies=True,
        armor_aware_fragment_proxies=bool(armor_aware),
    )
    columns = [
        name for name in COMPONENT_PROXY_FEATURE_COLUMNS
        if "_fragment_" in name
    ]
    values = augmented[columns].to_numpy(dtype=np.float32)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(values), 4096):
            outputs.append(
                component_probabilities_to_ordinal(
                    torch.from_numpy(values[start:start + 4096])
                ).numpy()
            )
    return np.concatenate(outputs, axis=0).astype(np.float64)


def _probability_fit(predicted: np.ndarray, target: np.ndarray) -> dict:
    residual = np.asarray(predicted) - np.asarray(target)
    nonzero = np.asarray(target) > 0.0
    hard_positive = np.asarray(target) >= 0.5
    return {
        "elements": int(target.size),
        "target_mean": float(np.asarray(target).mean()),
        "predicted_mean": float(np.asarray(predicted).mean()),
        "mean_bias": float(residual.mean()),
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.square(residual).mean())),
        "nonzero_target_mae": (
            float(np.abs(residual[nonzero]).mean())
            if np.any(nonzero) else None
        ),
        "hard_positive_count": int(hard_positive.sum()),
        "hard_positive_recall_at_0p5": (
            float((predicted[hard_positive] >= 0.5).mean())
            if np.any(hard_positive) else None
        ),
    }


def _goal_feasibility(
        frame,
        probabilities: np.ndarray,
        threshold_grid: np.ndarray) -> dict:
    munition_ids = frame["munition_id"].to_numpy(dtype=np.int64)
    feasible = []
    cells = {}
    for munition_id, munition in enumerate(MUNITION_NAMES):
        mask = munition_ids == munition_id
        cells[munition] = {}
        for task_id, task in enumerate(TASK_NAMES):
            maximum_fp = (
                0.005
                if munition_id == 0 and task_id == 0
                else 0.025 if task_id == 3 else None
            )
            result = evaluate_cell_threshold_feasibility(
                probabilities[mask, task_id, 0],
                probabilities[mask, task_id, 1],
                frame.loc[mask, f"{task}_level"].to_numpy(
                    dtype=np.int64),
                threshold_grid,
                tuple(DEFAULT_ORDINAL_APPLICABILITY[
                    munition_id][task_id]),
                minimum_accuracy=0.94,
                minimum_recall=0.90,
                minimum_support=100,
                maximum_l0_false_positive_rate=maximum_fp,
            )
            cells[munition][task] = result
            if result["metric_goal_feasible"]:
                feasible.append(f"{munition}/{task}")
    return {
        "feasible_count": int(len(feasible)),
        "feasible_cells": feasible,
        "infeasible_cells": [
            f"{munition}/{task}"
            for munition in MUNITION_NAMES
            for task in TASK_NAMES
            if f"{munition}/{task}" not in feasible
        ],
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only comparison of legacy and armor-aware deployable "
            "fragment component proxies."))
    parser.add_argument(
        "--data", default="output/damage_dataset.parquet")
    parser.add_argument(
        "--threshold-step", type=float, default=0.01)
    parser.add_argument(
        "--output",
        default="output/experiments/fragment_proxy_quality.json")
    args = parser.parse_args()
    if not 0.0 < args.threshold_step <= 0.10:
        raise ValueError("threshold-step must be in (0,0.10].")

    dataset_path = Path(args.data).resolve()
    frame = _read_validation(dataset_path)
    target_fragment = _task_mechanism_targets(frame, "fragment")
    target_shock = _task_mechanism_targets(frame, "shock")
    threshold_grid = np.arange(
        args.threshold_step,
        1.0 + 0.5 * args.threshold_step,
        args.threshold_step,
        dtype=np.float64,
    )
    sources = {}
    for name, armor_aware in (
        ("legacy_geometric", False),
        ("armor_aware", True),
    ):
        fragment = _fragment_component_tree(frame, armor_aware)
        hybrid = 1.0 - (1.0 - fragment) * (1.0 - target_shock)
        hybrid[..., 1] = np.minimum(hybrid[..., 1], hybrid[..., 0])
        sources[name] = {
            "fragment_target_fit": _probability_fit(
                fragment, target_fragment),
            "hybrid_with_target_shock": _goal_feasibility(
                frame, hybrid, threshold_grid),
        }

    payload = {
        "schema": REPORT_SCHEMA,
        "status": "COMPLETE",
        "split": "validation",
        "test_labels_used": False,
        "validation_scan_predicate": "split_role == 'val'",
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "validation_rows": int(len(frame)),
        "sources": sources,
    }
    output_path = Path(args.output).resolve()
    _write_json_atomic(output_path, payload)
    print(json.dumps({
        "status": payload["status"],
        "validation_rows": payload["validation_rows"],
        "sources": {
            name: {
                "fragment_mae": result[
                    "fragment_target_fit"]["mae"],
                "hybrid_feasible_cells": result[
                    "hybrid_with_target_shock"]["feasible_count"],
                "hybrid_infeasible_cells": result[
                    "hybrid_with_target_shock"]["infeasible_cells"],
            }
            for name, result in sources.items()
        },
        "output": str(output_path),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
