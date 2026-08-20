from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as arrow_dataset


REPO_ROOT = Path(__file__).resolve().parents[3]

from loitering_munition_damage_twin.surrogate.artifacts import sha256_file
from loitering_munition_damage_twin.surrogate.dataset import FEATURE_COLUMNS
from loitering_munition_damage_twin.simulation.engine import (
    DamageEngine,
    EncounterCondition,
    create_heavy_loitering_munition,
    create_medium_loitering_munition,
    create_medium_rear_det,
    create_small_loitering_munition,
    load_armor_plates,
    load_vehicle_model,
)


REPORT_SCHEMA = "stage0_high_mc_label_stability_v2"
TASK_NAMES = ("K", "M", "F", "C")
MUNITION_NAMES = ("Small", "Med-LM", "Med-RD", "Heavy")
_WORKER_ENGINE = None
_WORKER_COMPONENTS = None
_WORKER_PROJECTILES = None
_WORKER_REPLICATES = None
_WORKER_COMPARISON_REPLICATES = None
_WORKER_ANTITHETIC = None


def _stable_seed(sample_id: str, replicate_index: int) -> int:
    digest = hashlib.sha256(
        (
            "stage0-high-mc-audit-v1|"
            f"{sample_id}|{int(replicate_index)}"
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _init_worker(
        replicates: int,
        comparison_replicates: int,
        antithetic: bool) -> None:
    global _WORKER_ENGINE
    global _WORKER_COMPONENTS
    global _WORKER_PROJECTILES
    global _WORKER_REPLICATES
    global _WORKER_COMPARISON_REPLICATES
    global _WORKER_ANTITHETIC
    _WORKER_ENGINE = DamageEngine(
        armor_plates=load_armor_plates())
    _WORKER_COMPONENTS = load_vehicle_model()
    _WORKER_PROJECTILES = (
        create_small_loitering_munition(),
        create_medium_loitering_munition(),
        create_medium_rear_det(),
        create_heavy_loitering_munition(),
    )
    _WORKER_REPLICATES = int(replicates)
    _WORKER_COMPARISON_REPLICATES = int(
        comparison_replicates)
    _WORKER_ANTITHETIC = bool(antithetic)


def _audit_worker(
        row: tuple,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (
        x_cm, y_cm, z_cm,
        vx_ms, vy_ms, vz_ms,
        sin_yaw, cos_yaw,
        sin_pitch, cos_pitch,
        sin_roll, cos_roll,
        munition_id, sample_id,
    ) = row
    encounter = EncounterCondition(
        dx=float(x_cm),
        dy=float(y_cm),
        dz=float(z_cm),
        vx=float(vx_ms),
        vy=float(vy_ms),
        vz=float(vz_ms),
        yaw_deg=float(np.degrees(np.arctan2(
            sin_yaw, cos_yaw))),
        pitch_deg=float(np.degrees(np.arctan2(
            sin_pitch, cos_pitch))),
        roll_deg=float(np.degrees(np.arctan2(
            sin_roll, cos_roll))),
    )
    replicate_results = []
    for replicate_index in range(_WORKER_REPLICATES):
        if _WORKER_ANTITHETIC:
            seed_index = replicate_index // 2
            spread_sign = (
                1.0 if replicate_index % 2 == 0 else -1.0)
        else:
            seed_index = replicate_index
            spread_sign = 1.0
        replicate_results.append(
            _WORKER_ENGINE.evaluate(
                _WORKER_PROJECTILES[int(munition_id)],
                encounter,
                _WORKER_COMPONENTS,
                rng_seed=_stable_seed(
                    str(sample_id), seed_index),
                fragment_spread_sign=spread_sign,
            ).damage_tree.ordinal_probability_vector
        )
    samples = np.stack(replicate_results).astype(np.float64)
    comparison = samples[:_WORKER_COMPARISON_REPLICATES]
    return (
        samples.mean(axis=0),
        samples.std(axis=0),
        comparison.mean(axis=0),
        comparison.std(axis=0),
    )


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def _select_audit_rows(
        frame: pd.DataFrame,
        rows_per_munition: int,
        seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    probability_columns = [
        f"{task}_ge{level}_prob"
        for task in TASK_NAMES
        for level in (1, 2)
    ]
    selected = []
    for munition_id in range(4):
        cell = frame.loc[
            frame["munition_id"].eq(munition_id)
        ].copy()
        if len(cell) < rows_per_munition:
            raise RuntimeError(
                f"m_id={munition_id} has fewer validation rows "
                "than requested.")
        probabilities = cell[
            probability_columns].to_numpy(dtype=np.float64)
        distance = np.min(
            np.abs(probabilities - 0.5), axis=1)
        exact_l1 = np.column_stack([
            cell[f"{task}_level"].to_numpy(
                dtype=np.int64) == 1
            for task in TASK_NAMES
        ]).any(axis=1)
        informative_count = int(
            rows_per_munition // 2)
        ranking = np.lexsort((
            cell["sample_id"].astype(str).to_numpy(),
            distance,
            ~exact_l1,
        ))
        informative_index = ranking[:informative_count]
        remaining_index = np.setdiff1d(
            np.arange(len(cell)),
            informative_index,
            assume_unique=False,
        )
        random_count = (
            rows_per_munition - informative_count)
        random_index = rng.choice(
            remaining_index,
            size=random_count,
            replace=False,
        )
        chosen = cell.iloc[np.concatenate((
            informative_index,
            random_index,
        ))].copy()
        chosen["audit_selection"] = np.concatenate((
            np.repeat("boundary_or_l1", informative_count),
            np.repeat("random_validation", random_count),
        ))
        selected.append(chosen)
    output = pd.concat(
        selected, ignore_index=True)
    if output["sample_id"].astype(str).duplicated().any():
        raise RuntimeError(
            "Audit selection contains duplicate samples.")
    return output


def _confusion(original: np.ndarray,
               audited: np.ndarray) -> list[list[int]]:
    matrix = np.zeros((3, 3), dtype=np.int64)
    np.add.at(matrix, (original, audited), 1)
    return matrix.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a bounded, validation-only stratified sample with many "
            "independent Monte-Carlo fragment realizations. The test split "
            "is excluded at the Arrow scan predicate."
        ))
    parser.add_argument(
        "--data", default="output/damage_dataset.parquet")
    parser.add_argument(
        "--rows-per-munition", type=int, default=256)
    parser.add_argument(
        "--replicates", type=int, default=64)
    parser.add_argument(
        "--comparison-replicates", type=int, default=32,
        help=(
            "Prefix budget compared with the full replay, e.g. 32 vs 64."))
    parser.add_argument(
        "--antithetic", action="store_true",
        help=(
            "Pair each Gaussian fragment spread with its sign-reversed "
            "counterpart. Both budgets must therefore be even."))
    parser.add_argument(
        "--workers", type=int, default=max(
            1, min(12, os.cpu_count() or 1)))
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--output",
        default=(
            "output/experiments/"
            "high_mc_label_stability_r64.json"))
    args = parser.parse_args()
    if args.rows_per_munition < 16:
        raise ValueError(
            "rows-per-munition must be at least 16.")
    if args.replicates < 8:
        raise ValueError(
            "replicates must be at least 8.")
    if not (
        8 <= args.comparison_replicates < args.replicates
    ):
        raise ValueError(
            "comparison-replicates must be at least 8 and smaller "
            "than replicates.")
    if args.antithetic and (
        args.replicates % 2 != 0
        or args.comparison_replicates % 2 != 0
    ):
        raise ValueError(
            "Antithetic replicate budgets must be even.")
    if args.workers < 1:
        raise ValueError("workers must be positive.")

    dataset_path = Path(args.data).resolve()
    probability_columns = [
        f"{task}_ge{level}_prob"
        for task in TASK_NAMES
        for level in (1, 2)
    ]
    std_columns = [
        f"{task}_ge{level}_prob_std"
        for task in TASK_NAMES
        for level in (1, 2)
    ]
    required = list(dict.fromkeys(
        list(FEATURE_COLUMNS[:12])
        + [
            "munition_id",
            "sample_id",
            "split_role",
            "label_mc_replicates",
        ]
        + [f"{task}_level" for task in TASK_NAMES]
        + probability_columns
        + std_columns
    ))
    dataset = arrow_dataset.dataset(
        str(dataset_path), format="parquet")
    validation = dataset.to_table(
        columns=required,
        filter=arrow_dataset.field("split_role") == "val",
    ).to_pandas()
    if (
        validation.empty
        or not validation["split_role"].eq("val").all()
    ):
        raise RuntimeError(
            "Validation-only scan predicate was not enforced.")
    audit = _select_audit_rows(
        validation,
        rows_per_munition=int(args.rows_per_munition),
        seed=int(args.seed),
    )

    worker_rows = audit[[
        "x_cm", "y_cm", "z_cm",
        "vx_ms", "vy_ms", "vz_ms",
        "sin_yaw", "cos_yaw",
        "sin_pitch", "cos_pitch",
        "sin_roll", "cos_roll",
        "munition_id", "sample_id",
    ]].itertuples(index=False, name=None)
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=int(args.workers),
        initializer=_init_worker,
        initargs=(
            int(args.replicates),
            int(args.comparison_replicates),
            bool(args.antithetic),
        ),
    ) as pool:
        replay = list(pool.map(
            _audit_worker,
            worker_rows,
            chunksize=2,
        ))
    elapsed = time.perf_counter() - started
    high_mean = np.stack(
        [item[0] for item in replay])
    high_std = np.stack(
        [item[1] for item in replay])
    comparison_mean = np.stack(
        [item[2] for item in replay])
    comparison_std = np.stack(
        [item[3] for item in replay])
    high_mean[:, 1::2] = np.minimum(
        high_mean[:, 1::2],
        high_mean[:, 0::2],
    )
    comparison_mean[:, 1::2] = np.minimum(
        comparison_mean[:, 1::2],
        comparison_mean[:, 0::2],
    )
    original_mean = audit[
        probability_columns].to_numpy(
            dtype=np.float64)
    original_std = audit[
        std_columns].to_numpy(
            dtype=np.float64)
    original_levels = audit[
        [f"{task}_level" for task in TASK_NAMES]
    ].to_numpy(dtype=np.int64)
    high_levels = np.stack([
        np.where(
            high_mean[:, 2 * task_index + 1] >= 0.5,
            2,
            np.where(
                high_mean[:, 2 * task_index] >= 0.5,
                1,
                0,
            ),
        )
        for task_index in range(4)
    ], axis=1)
    comparison_levels = np.stack([
        np.where(
            comparison_mean[:, 2 * task_index + 1] >= 0.5,
            2,
            np.where(
                comparison_mean[:, 2 * task_index] >= 0.5,
                1,
                0,
            ),
        )
        for task_index in range(4)
    ], axis=1)

    cells = {}
    for munition_id, munition_name in enumerate(
            MUNITION_NAMES):
        munition_mask = audit[
            "munition_id"].to_numpy(dtype=np.int64) == munition_id
        cells[munition_name] = {}
        for task_index, task_name in enumerate(TASK_NAMES):
            old = original_levels[
                munition_mask, task_index]
            new = high_levels[
                munition_mask, task_index]
            comparison_level = comparison_levels[
                munition_mask, task_index]
            probability_slice = slice(
                2 * task_index, 2 * task_index + 2)
            residual = (
                high_mean[munition_mask, probability_slice]
                - original_mean[
                    munition_mask, probability_slice]
            )
            cells[munition_name][task_name] = {
                "rows": int(munition_mask.sum()),
                "original_support": np.bincount(
                    old, minlength=3).astype(int).tolist(),
                "high_mc_support": np.bincount(
                    new, minlength=3).astype(int).tolist(),
                "original_rows_high_mc_columns_confusion": (
                    _confusion(old, new)),
                "hard_level_agreement_percent": float(
                    np.mean(old == new) * 100.0),
                "probability_mae": float(
                    np.abs(residual).mean()),
                "probability_rmse": float(
                    np.sqrt(np.square(residual).mean())),
                "high_mc_standard_error_mean": float(
                    (
                        high_std[
                            munition_mask, probability_slice]
                        / np.sqrt(float(args.replicates))
                    ).mean()),
                "comparison_to_full_hard_level_agreement_percent": float(
                    np.mean(comparison_level == new) * 100.0),
                "comparison_to_full_probability_mae": float(
                    np.abs(
                        comparison_mean[
                            munition_mask, probability_slice]
                        - high_mean[
                            munition_mask, probability_slice]
                    ).mean()),
            }

    original_standard_error = (
        original_std
        / np.sqrt(
            audit["label_mc_replicates"].to_numpy(
                dtype=np.float64)[:, None])
    )
    high_standard_error = (
        high_std / np.sqrt(float(args.replicates)))
    comparison_standard_error = (
        comparison_std
        / np.sqrt(float(args.comparison_replicates)))
    payload = {
        "schema": REPORT_SCHEMA,
        "status": "COMPLETE",
        "split": "validation",
        "test_labels_used": False,
        "validation_scan_predicate": "split_role == 'val'",
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "seed_namespace": "stage0-high-mc-audit-v1",
        "antithetic_fragment_spread": bool(
            args.antithetic),
        "selection_seed": int(args.seed),
        "selection": {
            "rows_per_munition": int(
                args.rows_per_munition),
            "total_rows": int(len(audit)),
            "roles": {
                str(key): int(value)
                for key, value in audit[
                    "audit_selection"].value_counts().items()
            },
            "role": (
                "diagnostic stratified sample; not an unbiased estimate "
                "of the full validation distribution"),
        },
        "high_mc_replicates": int(args.replicates),
        "comparison_replicates": int(
            args.comparison_replicates),
        "workers": int(args.workers),
        "elapsed_seconds": float(elapsed),
        "simulator_evaluations": int(
            len(audit) * int(args.replicates)),
        "rows_per_second": float(
            len(audit) / max(elapsed, 1e-12)),
        "overall": {
            "hard_cell_agreement_percent": float(
                np.mean(
                    original_levels == high_levels) * 100.0),
            "ordinal_probability_mae": float(
                np.abs(high_mean - original_mean).mean()),
            "ordinal_probability_rmse": float(
                np.sqrt(np.square(
                    high_mean - original_mean).mean())),
            "original_standard_error_mean": float(
                original_standard_error.mean()),
            "high_mc_standard_error_mean": float(
                high_standard_error.mean()),
            "standard_error_reduction_percent": float(
                (
                    1.0
                    - high_standard_error.mean()
                    / max(
                        original_standard_error.mean(),
                        1e-12,
                    )
                ) * 100.0),
            "comparison_to_full_hard_cell_agreement_percent": float(
                np.mean(
                    comparison_levels == high_levels) * 100.0),
            "comparison_to_full_ordinal_probability_mae": float(
                np.abs(
                    comparison_mean - high_mean).mean()),
            "comparison_standard_error_mean": float(
                comparison_standard_error.mean()),
        },
        "cells": cells,
    }
    output_path = Path(args.output).resolve()
    _write_json_atomic(output_path, payload)
    print(json.dumps({
        "status": payload["status"],
        "test_labels_used": payload[
            "test_labels_used"],
        "rows": payload["selection"]["total_rows"],
        "replicates": payload[
            "high_mc_replicates"],
        "elapsed_seconds": payload[
            "elapsed_seconds"],
        "overall": payload["overall"],
        "output": str(output_path),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
