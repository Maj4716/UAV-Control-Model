from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.dataset as arrow_dataset


REPO_ROOT = Path(__file__).resolve().parents[3]

from loitering_munition_damage_twin.surrogate.artifacts import sha256_file
from loitering_munition_damage_twin.surrogate.model import DEFAULT_ORDINAL_APPLICABILITY

from loitering_munition_damage_twin.experiments.analyze_validation_threshold_feasibility import (
    evaluate_cell_threshold_feasibility,
)


REPORT_SCHEMA = "stage0_mechanism_oracle_feasibility_v1"
MUNITION_NAMES = ("Small", "Med-LM", "Med-RD", "Heavy")
TASK_NAMES = ("K", "M", "F", "C")


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def _read_validation(dataset_path: Path):
    columns = [
        "sample_id", "root_seed_id", "split_role", "munition_id",
        *[f"{task}_level" for task in TASK_NAMES],
        *[
            f"{mechanism}_{task}_ge{level}_prob"
            for mechanism in ("fragment", "shock")
            for task in TASK_NAMES
            for level in (1, 2)
        ],
    ]
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


def _source_probabilities(frame, source: str) -> np.ndarray:
    output = np.empty(
        (len(frame), len(TASK_NAMES), 2),
        dtype=np.float64)
    for task_index, task in enumerate(TASK_NAMES):
        fragment = np.column_stack((
            frame[f"fragment_{task}_ge1_prob"].to_numpy(
                dtype=np.float64),
            frame[f"fragment_{task}_ge2_prob"].to_numpy(
                dtype=np.float64),
        ))
        shock = np.column_stack((
            frame[f"shock_{task}_ge1_prob"].to_numpy(
                dtype=np.float64),
            frame[f"shock_{task}_ge2_prob"].to_numpy(
                dtype=np.float64),
        ))
        if source == "fragment":
            values = fragment
        elif source == "shock":
            values = shock
        elif source == "ordinal_or":
            values = 1.0 - (1.0 - fragment) * (1.0 - shock)
        else:
            raise ValueError(f"Unknown mechanism source: {source}")
        values[:, 1] = np.minimum(values[:, 1], values[:, 0])
        output[:, task_index] = np.clip(values, 0.0, 1.0)
    return output


def _evaluate_source(
        frame,
        probabilities: np.ndarray,
        threshold_grid: np.ndarray) -> dict:
    cells = {}
    feasible_cells = []
    evidence_failures = []
    for munition_id, munition_name in enumerate(
            MUNITION_NAMES):
        cells[munition_name] = {}
        munition_mask = (
            frame["munition_id"].to_numpy(dtype=np.int64)
            == munition_id)
        for task_index, task in enumerate(TASK_NAMES):
            maximum_fp = (
                0.005
                if munition_id == 0 and task == "K"
                else 0.025 if task == "C" else None)
            cell = evaluate_cell_threshold_feasibility(
                probabilities[
                    munition_mask, task_index, 0],
                probabilities[
                    munition_mask, task_index, 1],
                frame.loc[
                    munition_mask,
                    f"{task}_level",
                ].to_numpy(dtype=np.int64),
                threshold_grid,
                tuple(
                    bool(value)
                    for value in
                    DEFAULT_ORDINAL_APPLICABILITY[
                        munition_id][task_index]
                ),
                minimum_accuracy=0.94,
                minimum_recall=0.90,
                minimum_support=100,
                maximum_l0_false_positive_rate=maximum_fp,
            )
            cells[munition_name][task] = cell
            cell_name = f"{munition_name}/{task}"
            if cell["metric_goal_feasible"]:
                feasible_cells.append(cell_name)
            if not cell["evidence_sufficient"]:
                evidence_failures.append({
                    "cell": cell_name,
                    "failures": cell["evidence_failures"],
                })
    return {
        "metric_goal_feasible_cells": feasible_cells,
        "metric_goal_feasible_count": len(feasible_cells),
        "evidence_failure_cells": evidence_failures,
        "evidence_sufficient": not evidence_failures,
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only mechanism oracle threshold feasibility audit. "
            "It diagnoses simulator labels and never exposes test rows."))
    parser.add_argument(
        "--data", default="output/damage_dataset.parquet")
    parser.add_argument(
        "--threshold-step", type=float, default=0.01)
    parser.add_argument(
        "--output",
        default=(
            "output/experiments/"
            "mechanism_oracle_feasibility.json"))
    args = parser.parse_args()
    if not 0.0 < args.threshold_step <= 0.10:
        raise ValueError(
            "threshold-step must be in (0,0.10].")
    dataset_path = Path(args.data).resolve()
    frame = _read_validation(dataset_path)
    threshold_grid = np.arange(
        float(args.threshold_step),
        1.0 + 0.5 * float(args.threshold_step),
        float(args.threshold_step),
        dtype=np.float64,
    )
    sources = {}
    for source in ("fragment", "shock", "ordinal_or"):
        sources[source] = _evaluate_source(
            frame,
            _source_probabilities(frame, source),
            threshold_grid,
        )
    payload = {
        "schema": REPORT_SCHEMA,
        "status": "COMPLETE",
        "split": "validation",
        "test_labels_used": False,
        "validation_scan_predicate": "split_role == 'val'",
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "validation_rows": int(len(frame)),
        "requirements": {
            "minimum_cell_3class_accuracy_percent": 94.0,
            "minimum_applicable_class_diagonal_recall_percent": 90.0,
            "minimum_class_support": 100,
            "small_k0_max_false_positive_percent": 0.5,
            "per_cell_c0_max_false_positive_percent": 2.5,
        },
        "threshold_grid": {
            "minimum": float(threshold_grid.min()),
            "maximum": float(threshold_grid.max()),
            "step": float(args.threshold_step),
        },
        "source_semantics": {
            "fragment": (
                "simulator fragment-only damage-tree oracle; "
                "diagnostic, not deployable input"),
            "shock": (
                "simulator shock-only damage-tree oracle; "
                "diagnostic, not deployable input"),
            "ordinal_or": (
                "probabilistic OR of fragment/shock ordinal tree outputs; "
                "diagnostic approximation because authoritative combination "
                "occurs at component level"),
        },
        "sources": sources,
    }
    output_path = Path(args.output).resolve()
    _write_json_atomic(output_path, payload)
    print(json.dumps({
        "status": payload["status"],
        "split": payload["split"],
        "test_labels_used": payload["test_labels_used"],
        "validation_rows": payload["validation_rows"],
        "feasible_cells": {
            source: result["metric_goal_feasible_count"]
            for source, result in sources.items()
        },
        "output": str(output_path),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
