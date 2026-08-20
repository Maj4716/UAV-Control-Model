from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as arrow_dataset
from sklearn.isotonic import IsotonicRegression


REPO_ROOT = Path(__file__).resolve().parents[3]

from loitering_munition_damage_twin.stage0.component_supervision import COMPONENT_TARGET_COLUMNS
from loitering_munition_damage_twin.surrogate.artifacts import sha256_file
from loitering_munition_damage_twin.surrogate.dataset import FEATURE_COLUMNS
from loitering_munition_damage_twin.surrogate.features import (
    COMPONENT_PROXY_FEATURE_COLUMNS,
    augment_terminal_physics_features,
)

from loitering_munition_damage_twin.experiments.analyze_a35_component_branch import (
    MUNITION_NAMES,
    TASK_NAMES,
    _component_tree,
)
from loitering_munition_damage_twin.experiments.analyze_validation_threshold_feasibility import (
    evaluate_cell_threshold_feasibility,
)


REPORT_SCHEMA = "stage0_calibrated_component_proxy_probe_v1"


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def _load_non_test_dataset(dataset_path: Path) -> pd.DataFrame:
    columns = list(dict.fromkeys(
        list(FEATURE_COLUMNS)
        + [
            "munition_id",
            "sample_id",
            "split_role",
            "K_level",
            "M_level",
            "F_level",
            "C_level",
        ]
    ))
    dataset = arrow_dataset.dataset(
        str(dataset_path), format="parquet")
    frame = dataset.to_table(
        columns=columns,
        filter=arrow_dataset.field("split_role") != "test",
    ).to_pandas()
    if (
        frame.empty
        or not frame["split_role"].isin(
            ["train", "val"]).all()
        or frame["sample_id"].astype(str).duplicated().any()
    ):
        raise RuntimeError(
            "Train/validation scan or sample identity is invalid.")
    return frame


def _load_component_targets(
        sidecar_path: Path,
        sample_ids: np.ndarray,
) -> np.ndarray:
    sidecar = pd.read_parquet(
        sidecar_path,
        columns=["sample_id", *COMPONENT_TARGET_COLUMNS],
        engine="pyarrow",
    )
    if sidecar["sample_id"].astype(str).duplicated().any():
        raise RuntimeError(
            "Component sidecar contains duplicate sample IDs.")
    sidecar["sample_id"] = sidecar[
        "sample_id"].astype(str)
    sidecar = sidecar.set_index("sample_id", verify_integrity=True)
    requested = pd.Index(
        np.asarray(sample_ids, dtype=str))
    missing = requested.difference(sidecar.index)
    if len(missing):
        raise RuntimeError(
            "Component sidecar is missing requested samples.")
    return sidecar.loc[
        requested, list(COMPONENT_TARGET_COLUMNS)
    ].to_numpy(dtype=np.float32).reshape(
        len(requested), 2, -1)


def _fit_isotonic_component_calibrators(
        proxy: np.ndarray,
        target: np.ndarray,
        train_mask: np.ndarray,
        validation_mask: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    if proxy.shape != target.shape or proxy.ndim != 3:
        raise ValueError(
            "Proxy and target must share shape (N,2,C).")
    calibrated = np.empty(
        (int(validation_mask.sum()), *proxy.shape[1:]),
        dtype=np.float32,
    )
    contracts = []
    for mechanism_index in range(proxy.shape[1]):
        for component_index in range(proxy.shape[2]):
            model = IsotonicRegression(
                y_min=0.0,
                y_max=1.0,
                increasing=True,
                out_of_bounds="clip",
            )
            train_x = proxy[
                train_mask, mechanism_index, component_index]
            train_y = target[
                train_mask, mechanism_index, component_index]
            if np.ptp(train_x) <= 1e-12:
                prediction = np.full(
                    int(validation_mask.sum()),
                    float(np.mean(train_y)),
                    dtype=np.float32,
                )
                knot_count = 1
            else:
                model.fit(train_x, train_y)
                prediction = model.predict(
                    proxy[
                        validation_mask,
                        mechanism_index,
                        component_index,
                    ]
                ).astype(np.float32)
                knot_count = int(len(model.X_thresholds_))
            calibrated[
                :, mechanism_index, component_index] = (
                    np.clip(prediction, 0.0, 1.0))
            contracts.append({
                "mechanism_index": int(mechanism_index),
                "component_index": int(component_index),
                "knot_count": int(knot_count),
            })
    return calibrated, contracts


def _probability_error(
        predicted: np.ndarray,
        target: np.ndarray) -> dict:
    residual = (
        predicted.astype(np.float64)
        - target.astype(np.float64)
    )
    return {
        "elements": int(residual.size),
        "mean_bias": float(residual.mean()),
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.square(residual).mean())),
        "brier": float(np.square(residual).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train-only monotone calibration of deployable per-component "
            "physics proxies, followed by a validation-only damage-tree "
            "feasibility audit. Test rows are excluded at the scan boundary."
        ))
    parser.add_argument(
        "--data", default="output/damage_dataset.parquet")
    parser.add_argument(
        "--component-sidecar",
        default="output/component_supervision.parquet")
    parser.add_argument(
        "--output",
        default=(
            "output/experiments/"
            "calibrated_component_proxy_tree_diagnostic.json"))
    parser.add_argument(
        "--threshold-step", type=float, default=0.02)
    args = parser.parse_args()
    if not 0.0 < args.threshold_step <= 0.25:
        raise ValueError(
            "threshold-step must be in (0,0.25].")

    dataset_path = Path(args.data).resolve()
    sidecar_path = Path(args.component_sidecar).resolve()
    frame = _load_non_test_dataset(dataset_path)
    component_targets = _load_component_targets(
        sidecar_path,
        frame["sample_id"].astype(str).to_numpy(),
    )
    validation_count = int(
        frame["split_role"].eq("val").sum())
    validation_component_prediction = np.empty(
        (validation_count, 2, len(COMPONENT_TARGET_COLUMNS) // 2),
        dtype=np.float32,
    )
    validation_component_raw = np.empty_like(
        validation_component_prediction)
    validation_component_target = np.empty_like(
        validation_component_prediction)
    validation_levels = np.empty(
        (validation_count, 4), dtype=np.int64)
    validation_munition = np.empty(
        validation_count, dtype=np.int64)
    validation_cursor = 0
    calibration_contract = {}
    raw_error_by_munition = {}
    calibrated_error_by_munition = {}

    for munition_index, munition_name in enumerate(
            MUNITION_NAMES):
        munition_mask = (
            frame["munition_id"].to_numpy(dtype=np.int64)
            == munition_index
        )
        cell = frame.loc[munition_mask].copy().reset_index(
            drop=True)
        targets = component_targets[munition_mask]
        cell = augment_terminal_physics_features(
            cell,
            copy=False,
            include_component_proxies=True,
        )
        proxy = cell[
            COMPONENT_PROXY_FEATURE_COLUMNS
        ].to_numpy(dtype=np.float32).reshape(
            len(cell), 2, -1)
        train_mask = cell[
            "split_role"].eq("train").to_numpy()
        validation_mask = cell[
            "split_role"].eq("val").to_numpy()
        calibrated, contracts = (
            _fit_isotonic_component_calibrators(
                proxy,
                targets,
                train_mask,
                validation_mask,
            )
        )
        count = int(validation_mask.sum())
        destination = slice(
            validation_cursor,
            validation_cursor + count,
        )
        validation_component_prediction[
            destination] = calibrated
        validation_component_raw[
            destination] = proxy[validation_mask]
        validation_component_target[
            destination] = targets[validation_mask]
        validation_levels[destination] = cell.loc[
            validation_mask,
            [f"{task}_level" for task in TASK_NAMES],
        ].to_numpy(dtype=np.int64)
        validation_munition[destination] = munition_index
        raw_error_by_munition[munition_name] = (
            _probability_error(
                proxy[validation_mask],
                targets[validation_mask],
            )
        )
        calibrated_error_by_munition[munition_name] = (
            _probability_error(
                calibrated,
                targets[validation_mask],
            )
        )
        calibration_contract[munition_name] = {
            "training_rows": int(train_mask.sum()),
            "validation_rows": count,
            "calibrators": contracts,
        }
        validation_cursor += count

    if validation_cursor != validation_count:
        raise RuntimeError(
            "Validation assembly row count mismatch.")
    ordinal_probability = _component_tree(
        validation_component_prediction)
    threshold_grid = np.arange(
        float(args.threshold_step),
        1.0 + 0.5 * float(args.threshold_step),
        float(args.threshold_step),
        dtype=np.float64,
    )
    cells = {}
    all_metric_feasible = True
    all_evidence_sufficient = True
    c_minimum_false_positives = []
    c_l0_support = 0
    applicability = (
        ((True, False), (True, True), (True, True), (True, False)),
        ((True, True), (True, True), (True, True), (True, True)),
        ((True, True), (True, True), (True, True), (True, True)),
        ((True, True), (True, True), (True, True), (True, True)),
    )
    for munition_index, munition_name in enumerate(
            MUNITION_NAMES):
        mask = validation_munition == munition_index
        cells[munition_name] = {}
        for task_index, task_name in enumerate(TASK_NAMES):
            result = evaluate_cell_threshold_feasibility(
                ordinal_probability[mask, task_index, 0],
                ordinal_probability[mask, task_index, 1],
                validation_levels[mask, task_index],
                threshold_grid,
                applicability[munition_index][task_index],
                maximum_l0_false_positive_rate=(
                    0.005
                    if munition_index == 0 and task_index == 0
                    else None
                ),
            )
            cells[munition_name][task_name] = result
            all_metric_feasible &= bool(
                result["metric_goal_feasible"])
            all_evidence_sufficient &= bool(
                result["evidence_sufficient"])
            if task_index == 3:
                c_l0_support += int(
                    result["class_support"][0])
                if result["goal_candidate"] is not None:
                    c_minimum_false_positives.append(
                        int(result["goal_candidate"][
                            "l0_false_positive_count"]))

    c_budget = int(math.floor(0.025 * c_l0_support + 1e-12))
    c_joint_count = (
        int(sum(c_minimum_false_positives))
        if len(c_minimum_false_positives) == 4
        else None
    )
    global_c_feasible = bool(
        c_joint_count is not None
        and c_joint_count <= c_budget
    )
    all_metric_feasible &= global_c_feasible

    payload = {
        "schema": REPORT_SCHEMA,
        "status": "COMPLETE",
        "split": "validation",
        "test_labels_used": False,
        "scan_predicate": "split_role != 'test'",
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "component_sidecar": str(sidecar_path),
        "component_sidecar_sha256": sha256_file(sidecar_path),
        "calibration_fit_split": "train",
        "calibration_method": (
            "per-munition per-mechanism per-component monotone isotonic"),
        "validation_rows": int(validation_count),
        "component_probability_error": {
            "raw_proxy_overall": _probability_error(
                validation_component_raw,
                validation_component_target,
            ),
            "raw_proxy_by_munition": raw_error_by_munition,
            "calibrated_overall": _probability_error(
                validation_component_prediction,
                validation_component_target,
            ),
            "calibrated_by_munition": (
                calibrated_error_by_munition),
        },
        "metric_goal_threshold_feasible": bool(
            all_metric_feasible),
        "evidence_sufficient": bool(
            all_evidence_sufficient),
        "global_c0_joint_feasibility": {
            "l0_support": int(c_l0_support),
            "maximum_false_positive_count": int(c_budget),
            "minimum_false_positive_count_for_local_goals": (
                c_joint_count),
            "feasible": global_c_feasible,
        },
        "cells": cells,
        "calibration_contract": calibration_contract,
    }
    output_path = Path(args.output).resolve()
    _write_json_atomic(output_path, payload)
    print(json.dumps({
        "status": payload["status"],
        "metric_goal_threshold_feasible": payload[
            "metric_goal_threshold_feasible"],
        "evidence_sufficient": payload[
            "evidence_sufficient"],
        "calibrated_component_error": payload[
            "component_probability_error"][
                "calibrated_overall"],
        "infeasible_cells": [
            f"{munition}/{task}"
            for munition, tasks in cells.items()
            for task, result in tasks.items()
            if not result["metric_goal_feasible"]
        ],
        "output": str(output_path),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
