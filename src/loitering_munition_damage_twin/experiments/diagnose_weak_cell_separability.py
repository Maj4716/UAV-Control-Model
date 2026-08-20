from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

from loitering_munition_damage_twin.surrogate.artifacts import sha256_file
from loitering_munition_damage_twin.surrogate.dataset import FEATURE_COLUMNS
from loitering_munition_damage_twin.surrogate.features import (
    TERMINAL_PHYSICS_FEATURE_COLUMNS,
    TERMINAL_PHYSICS_FEATURE_VERSION,
    augment_terminal_physics_features,
)


ENTRY_CELLS = (
    ("Small/K", 0, "K_level", 0.005),
    ("Med-LM/K", 1, "K_level", 0.025),
    ("Med-RD/K", 2, "K_level", 0.025),
    ("Heavy/K", 3, "K_level", 0.025),
    ("Small/C", 0, "C_level", 0.025),
    ("Med-LM/C", 1, "C_level", 0.025),
    ("Med-RD/C", 2, "C_level", 0.025),
)


def _recall_at_fpr_cap(
        target: np.ndarray,
        score: np.ndarray,
        maximum_fpr: float) -> dict:
    fpr, tpr, thresholds = roc_curve(target, score)
    feasible = np.flatnonzero(fpr <= maximum_fpr + 1e-12)
    if feasible.size == 0:
        return {
            "maximum_recall": 0.0,
            "observed_false_positive_rate": 0.0,
            "threshold": float("inf"),
        }
    candidate_tpr = tpr[feasible]
    best_local = int(np.argmax(candidate_tpr))
    best = int(feasible[best_local])
    return {
        "maximum_recall": float(tpr[best]),
        "observed_false_positive_rate": float(fpr[best]),
        "threshold": float(thresholds[best]),
    }


def _fit_and_score(
        train_frame: pd.DataFrame,
        validation_frame: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        target_level: int,
        munition_id: int,
        maximum_fpr: float,
        conditional_damaged_only: bool,
        seed: int,
        estimators: int) -> dict:
    train_mask = train_frame["munition_id"].to_numpy() == munition_id
    validation_mask = (
        validation_frame["munition_id"].to_numpy() == munition_id)
    if conditional_damaged_only:
        train_mask &= (
            train_frame[target_column].to_numpy() >= 1)
        validation_mask &= (
            validation_frame[target_column].to_numpy() >= 1)
    train_cell = train_frame.loc[train_mask]
    validation_cell = validation_frame.loc[validation_mask]
    train_target = (
        train_cell[target_column].to_numpy(dtype=np.int64)
        >= target_level
    ).astype(np.int64)
    validation_target = (
        validation_cell[target_column].to_numpy(dtype=np.int64)
        >= target_level
    ).astype(np.int64)
    if np.unique(train_target).size < 2:
        raise RuntimeError(
            f"Training cell {munition_id}/{target_column} has one class.")
    if np.unique(validation_target).size < 2:
        raise RuntimeError(
            f"Validation cell {munition_id}/{target_column} has one class.")

    model = ExtraTreesClassifier(
        n_estimators=int(estimators),
        criterion="entropy",
        max_features=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=int(seed),
    )
    train_weight = np.clip(
        train_cell["loss_weight"].to_numpy(dtype=np.float64),
        0.05,
        20.0,
    )
    train_weight /= max(float(np.mean(train_weight)), 1e-12)
    model.fit(
        train_cell[feature_columns].to_numpy(dtype=np.float32),
        train_target,
        sample_weight=train_weight,
    )
    score = model.predict_proba(
        validation_cell[feature_columns].to_numpy(dtype=np.float32)
    )[:, 1]
    return {
        "model": "ExtraTreesClassifier",
        "training_rows": int(len(train_cell)),
        "validation_rows": int(len(validation_cell)),
        "train_positive_support": int(train_target.sum()),
        "validation_positive_support": int(validation_target.sum()),
        "full_auc": float(roc_auc_score(validation_target, score)),
        "standardized_partial_auc": float(roc_auc_score(
            validation_target, score, max_fpr=maximum_fpr)),
        "average_precision": float(
            average_precision_score(validation_target, score)),
        "maximum_false_positive_rate": float(maximum_fpr),
        "recall_at_fpr_cap": _recall_at_fpr_cap(
            validation_target, score, maximum_fpr),
    }


def _write_json_atomic(path: str, payload: dict) -> None:
    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    os.replace(temporary_path, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate weak-cell separability using only train/validation "
            "rows and the same 13 deployable features."
        )
    )
    parser.add_argument(
        "--data", default="output/damage_dataset.parquet")
    parser.add_argument(
        "--baseline-report",
        default=(
            "output/experiments/"
            "A19_bounded_class1_floor_calibration/seed42/"
            "recalibration_report.json"
        ),
    )
    parser.add_argument(
        "--output",
        default="output/experiments/weak_cell_separability_diagnostic.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--estimators", type=int, default=400)
    parser.add_argument(
        "--feature-set",
        choices=("terminal", "terminal_physics"),
        default="terminal",
        help=(
            "Use the original 13 terminal features or append deterministic "
            "target-geometry physics features."),
    )
    args = parser.parse_args()

    dataset_path = str(Path(args.data).resolve())
    required_columns = (
        FEATURE_COLUMNS
        + ["munition_id", "split_role", "loss_weight", "K_level",
           "M_level", "C_level"]
    )
    frame = pd.read_parquet(
        dataset_path, columns=required_columns, engine="pyarrow")
    train_frame = frame.loc[frame["split_role"] == "train"].copy()
    validation_frame = frame.loc[
        frame["split_role"] == "val"].copy()
    if train_frame.empty or validation_frame.empty:
        raise RuntimeError(
            "Dataset must contain explicit train and val split_role rows.")
    if args.feature_set == "terminal_physics":
        train_frame = augment_terminal_physics_features(
            train_frame, copy=False)
        validation_frame = augment_terminal_physics_features(
            validation_frame, copy=False)
        feature_columns = (
            list(FEATURE_COLUMNS)
            + list(TERMINAL_PHYSICS_FEATURE_COLUMNS)
        )
        feature_contract = TERMINAL_PHYSICS_FEATURE_VERSION
    else:
        feature_columns = list(FEATURE_COLUMNS)
        feature_contract = "terminal_base_v1"

    with open(args.baseline_report, "r", encoding="utf-8") as stream:
        baseline_payload = json.load(stream)
    baseline_report = baseline_payload.get(
        "selection_validation_report", baseline_payload)
    if (
        baseline_report.get("split") != "validation"
        or baseline_report.get("test_labels_used") is not False
    ):
        raise RuntimeError(
            "Baseline report is not validation-only.")
    baseline_diagnostics = baseline_report[
        "targeted_probability_diagnostics"]

    cells = {}
    for name, munition_id, target_column, maximum_fpr in ENTRY_CELLS:
        diagnostic = _fit_and_score(
            train_frame,
            validation_frame,
            feature_columns=feature_columns,
            target_column=target_column,
            target_level=1,
            munition_id=munition_id,
            maximum_fpr=maximum_fpr,
            conditional_damaged_only=False,
            seed=args.seed,
            estimators=args.estimators,
        )
        baseline_partial_auc = float(
            baseline_diagnostics[name][
                "entry_standardized_partial_auc"])
        diagnostic["neural_baseline_standardized_partial_auc"] = (
            baseline_partial_auc)
        diagnostic["partial_auc_delta_vs_neural"] = (
            diagnostic["standardized_partial_auc"]
            - baseline_partial_auc
        )
        cells[name] = diagnostic
        print(
            f"[SEPARABILITY] {name}: pAUC="
            f"{diagnostic['standardized_partial_auc']:.4f} "
            f"(NN {baseline_partial_auc:.4f}, "
            f"delta {diagnostic['partial_auc_delta_vs_neural']:+.4f}) "
            f"recall@FPRcap="
            f"{diagnostic['recall_at_fpr_cap']['maximum_recall']:.2%}"
        )

    conditional = _fit_and_score(
        train_frame,
        validation_frame,
        feature_columns=feature_columns,
        target_column="M_level",
        target_level=2,
        munition_id=2,
        maximum_fpr=0.10,
        conditional_damaged_only=True,
        seed=args.seed,
        estimators=args.estimators,
    )
    baseline_conditional_auc = float(
        baseline_diagnostics["Med-RD/M_L1_vs_L2"][
            "conditional_auc"])
    conditional["neural_baseline_full_auc"] = (
        baseline_conditional_auc)
    conditional["full_auc_delta_vs_neural"] = (
        conditional["full_auc"] - baseline_conditional_auc)
    cells["Med-RD/M_L1_vs_L2"] = conditional
    print(
        "[SEPARABILITY] Med-RD/M_L1_vs_L2: AUC="
        f"{conditional['full_auc']:.4f} "
        f"(NN {baseline_conditional_auc:.4f}, "
        f"delta {conditional['full_auc_delta_vs_neural']:+.4f})"
    )

    entry_deltas = [
        cells[name]["partial_auc_delta_vs_neural"]
        for name, *_ in ENTRY_CELLS
    ]
    result = {
        "schema": "stage0_nn_weak_cell_separability_v1",
        "status": "COMPLETE",
        "split": "validation",
        "test_labels_used": False,
        "dataset": dataset_path,
        "dataset_sha256": sha256_file(dataset_path),
        "feature_contract": feature_contract,
        "features": feature_columns,
        "feature_count": len(feature_columns),
        "model_role": (
            "diagnostic upper-bound probe; not a deployment candidate"),
        "random_seed": int(args.seed),
        "estimators": int(args.estimators),
        "cells": cells,
        "summary": {
            "mean_entry_partial_auc_delta_vs_neural": float(
                np.mean(entry_deltas)),
            "entry_cells_better_than_neural_by_0p01": int(sum(
                delta >= 0.01 for delta in entry_deltas)),
            "entry_cells_worse_than_neural_by_0p01": int(sum(
                delta <= -0.01 for delta in entry_deltas)),
        },
    }
    _write_json_atomic(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "test_labels_used": result["test_labels_used"],
        "summary": result["summary"],
        "output": str(Path(args.output).resolve()),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
