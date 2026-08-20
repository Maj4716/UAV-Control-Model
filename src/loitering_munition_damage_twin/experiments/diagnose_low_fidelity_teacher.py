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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, roc_curve

REPO_ROOT = Path(__file__).resolve().parents[3]

from loitering_munition_damage_twin.surrogate.artifacts import sha256_file
from loitering_munition_damage_twin.surrogate.dataset import FEATURE_COLUMNS
from loitering_munition_damage_twin.surrogate.features import (
    TERMINAL_PHYSICS_FEATURE_COLUMNS,
    augment_terminal_physics_features,
)
from loitering_munition_damage_twin.simulation.engine import (
    DamageEngine,
    EncounterCondition,
    create_small_loitering_munition,
    load_armor_plates,
    load_vehicle_model,
)


_WORKER_ENGINE = None
_WORKER_COMPONENTS = None
_WORKER_PROJECTILE = None
_WORKER_SEEDS = None

BASE_TEACHER_FEATURE_NAMES = tuple(
    f"{mechanism}_{task}_ge{level}"
    for mechanism in ("combined", "fragment", "shock")
    for task in ("K", "M", "F", "C")
    for level in (1, 2)
) + (
    "fuel_fragment_damage",
    "fuel_shock_damage",
    "fuel_combined_damage",
    "fuel_fragment_hits",
    "fuel_fragment_penetrations",
    "fuel_mean_penetration_margin",
    "fuel_max_penetration_margin",
    "total_hits",
    "total_penetrations",
)


def _fixed_seed(index: int) -> int:
    digest = hashlib.sha256(
        f"stage0-low-fidelity-teacher-v1|{index}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _init_teacher_worker(replicates: int) -> None:
    global _WORKER_ENGINE
    global _WORKER_COMPONENTS
    global _WORKER_PROJECTILE
    global _WORKER_SEEDS
    _WORKER_COMPONENTS = load_vehicle_model()
    _WORKER_ENGINE = DamageEngine(armor_plates=load_armor_plates())
    _WORKER_PROJECTILE = create_small_loitering_munition()
    _WORKER_SEEDS = tuple(_fixed_seed(index) for index in range(replicates))


def _single_result_features(result) -> np.ndarray:
    vectors = []
    for attribute in (
        "damage_tree",
        "damage_tree_fragment",
        "damage_tree_shockwave",
    ):
        tree = getattr(result, attribute)
        vectors.extend(tree.ordinal_probability_vector.tolist())

    by_id = {
        int(component.component_id): component
        for component in result.component_results
    }
    fuel = by_id[3]
    margins = np.asarray(
        fuel.penetration_margins, dtype=np.float64)
    vectors.extend((
        float(fuel.fragment_damage_prob),
        float(fuel.shockwave_damage_prob),
        float(fuel.combined_damage_prob),
        float(fuel.fragment_hits),
        float(fuel.fragment_penetrations),
        float(margins.mean()) if margins.size else 0.0,
        float(margins.max()) if margins.size else 0.0,
        float(result.total_hits),
        float(result.total_penetrations),
    ))
    return np.asarray(vectors, dtype=np.float64)


def _teacher_worker(row: tuple[float, ...]) -> np.ndarray:
    (
        x_cm, y_cm, z_cm,
        vx_ms, vy_ms, vz_ms,
        sin_yaw, cos_yaw,
        sin_pitch, cos_pitch,
        sin_roll, cos_roll,
    ) = row
    encounter = EncounterCondition(
        dx=float(x_cm),
        dy=float(y_cm),
        dz=float(z_cm),
        vx=float(vx_ms),
        vy=float(vy_ms),
        vz=float(vz_ms),
        yaw_deg=float(np.degrees(np.arctan2(sin_yaw, cos_yaw))),
        pitch_deg=float(np.degrees(np.arctan2(
            sin_pitch, cos_pitch))),
        roll_deg=float(np.degrees(np.arctan2(sin_roll, cos_roll))),
    )
    samples = np.stack([
        _single_result_features(_WORKER_ENGINE.evaluate(
            _WORKER_PROJECTILE,
            encounter,
            _WORKER_COMPONENTS,
            rng_seed=seed,
        ))
        for seed in _WORKER_SEEDS
    ])
    return np.concatenate((samples.mean(axis=0), samples.std(axis=0)))


def _recall_at_fpr(
        target: np.ndarray,
        score: np.ndarray,
        maximum_fpr: float,
) -> dict[str, float]:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        target, score)
    valid = np.flatnonzero(false_positive_rate <= maximum_fpr + 1e-12)
    selected = int(valid[np.argmax(true_positive_rate[valid])])
    return {
        "recall": float(true_positive_rate[selected]),
        "false_positive_rate": float(false_positive_rate[selected]),
        "threshold": float(thresholds[selected]),
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only probe for a deterministic low-fidelity physics "
            "teacher. Test rows are excluded at the Parquet scan boundary."))
    parser.add_argument(
        "--data", default="output/damage_dataset.parquet")
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--train-negative-limit", type=int, default=20000)
    parser.add_argument("--estimators", type=int, default=400)
    parser.add_argument("--workers", type=int, default=max(
        1, min(12, os.cpu_count() or 1)))
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--output",
        default=(
            "output/experiments/"
            "low_fidelity_teacher_small_k_diagnostic.json"))
    args = parser.parse_args()
    if args.replicates < 1:
        raise ValueError("replicates must be at least one.")
    if args.train_negative_limit < 1:
        raise ValueError("train-negative-limit must be positive.")

    dataset_path = Path(args.data).resolve()
    required = list(dict.fromkeys(
        list(FEATURE_COLUMNS)
        + ["munition_id", "split_role", "loss_weight", "K_level"]
    ))
    dataset = arrow_dataset.dataset(str(dataset_path), format="parquet")
    frame = dataset.to_table(
        columns=required,
        filter=(
            (arrow_dataset.field("split_role") != "test")
            & (arrow_dataset.field("munition_id") == 0)
        ),
    ).to_pandas()

    target = (
        frame["K_level"].to_numpy(dtype=np.int64) >= 1
    ).astype(np.int64)
    train_mask = frame["split_role"].eq("train").to_numpy()
    validation_mask = frame["split_role"].eq("val").to_numpy()
    train_positive = np.flatnonzero(train_mask & (target == 1))
    train_negative = np.flatnonzero(train_mask & (target == 0))
    rng = np.random.default_rng(args.seed)
    if len(train_negative) > args.train_negative_limit:
        train_negative = rng.choice(
            train_negative,
            size=args.train_negative_limit,
            replace=False,
        )
    selected = np.sort(np.concatenate((
        train_positive,
        train_negative,
        np.flatnonzero(validation_mask),
    )))
    frame = frame.iloc[selected].copy().reset_index(drop=True)
    target = target[selected]
    train_mask = frame["split_role"].eq("train").to_numpy()
    validation_mask = frame["split_role"].eq("val").to_numpy()

    teacher_inputs = frame[[
        "x_cm", "y_cm", "z_cm",
        "vx_ms", "vy_ms", "vz_ms",
        "sin_yaw", "cos_yaw",
        "sin_pitch", "cos_pitch",
        "sin_roll", "cos_roll",
    ]].itertuples(index=False, name=None)
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=int(args.workers),
        initializer=_init_teacher_worker,
        initargs=(int(args.replicates),),
    ) as pool:
        teacher_values = np.stack(list(pool.map(
            _teacher_worker,
            teacher_inputs,
            chunksize=16,
        )))
    teacher_seconds = time.perf_counter() - started
    teacher_columns = [
        f"teacher_{statistic}_{name}"
        for statistic in ("mean", "std")
        for name in BASE_TEACHER_FEATURE_NAMES
    ]
    if teacher_values.shape[1] != len(teacher_columns):
        raise RuntimeError("Teacher feature shape does not match contract.")
    teacher_frame = pd.DataFrame(
        teacher_values.astype(np.float32),
        columns=teacher_columns,
        index=frame.index,
    )
    frame = pd.concat((frame, teacher_frame), axis=1)

    frame = augment_terminal_physics_features(frame, copy=False)
    feature_sets = {
        "terminal_geometry_v2": (
            list(FEATURE_COLUMNS)
            + list(TERMINAL_PHYSICS_FEATURE_COLUMNS)
        ),
        "terminal_geometry_v2_plus_low_fidelity_teacher": (
            list(FEATURE_COLUMNS)
            + list(TERMINAL_PHYSICS_FEATURE_COLUMNS)
            + teacher_columns
        ),
    }
    results = {}
    for name, columns in feature_sets.items():
        model = ExtraTreesClassifier(
            n_estimators=int(args.estimators),
            criterion="entropy",
            max_features=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=int(args.seed),
        )
        weights = np.clip(
            frame.loc[
                train_mask, "loss_weight"
            ].to_numpy(dtype=np.float64),
            0.05,
            20.0,
        )
        model.fit(
            frame.loc[
                train_mask, columns
            ].to_numpy(dtype=np.float32),
            target[train_mask],
            sample_weight=weights,
        )
        score = model.predict_proba(
            frame.loc[
                validation_mask, columns
            ].to_numpy(dtype=np.float32)
        )[:, 1]
        results[name] = {
            "feature_count": int(len(columns)),
            "full_auc": float(
                roc_auc_score(target[validation_mask], score)),
            "standardized_partial_auc_at_fpr_0p005": float(
                roc_auc_score(
                    target[validation_mask],
                    score,
                    max_fpr=0.005,
                )),
            "recall_at_fpr_0p005": _recall_at_fpr(
            target[validation_mask], score, 0.005),
        }

    # The production dataset currently derives fragment-spread seeds from
    # sample_id.  That makes neighbouring terminal states receive unrelated
    # Monte-Carlo quadrature points and injects row-wise label noise into the
    # response surface.  The teacher above deliberately uses the same fixed
    # quadrature seeds for every row.  Treating its combined K>=1 mean as a
    # provisional CRN label lets this validation-only probe answer a separate
    # question: would the unchanged deployable terminal features become
    # learnable if dataset labels used common random numbers?
    combined_k_ge1_index = BASE_TEACHER_FEATURE_NAMES.index(
        "combined_K_ge1")
    crn_probability = teacher_values[:, combined_k_ge1_index]
    crn_target = (crn_probability >= 0.5).astype(np.int64)
    crn_model = ExtraTreesClassifier(
        n_estimators=int(args.estimators),
        criterion="entropy",
        max_features=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=int(args.seed),
    )
    crn_model.fit(
        frame.loc[
            train_mask,
            feature_sets["terminal_geometry_v2"],
        ].to_numpy(dtype=np.float32),
        crn_target[train_mask],
        sample_weight=weights,
    )
    crn_score = crn_model.predict_proba(
        frame.loc[
            validation_mask,
            feature_sets["terminal_geometry_v2"],
        ].to_numpy(dtype=np.float32),
    )[:, 1]
    crn_relabel_diagnostic = {
        "role": (
            "validation-only counterfactual; no production labels changed"),
        "seed_mode": "common_random_numbers",
        "replicates": int(args.replicates),
        "train_positive_rows": int(
            crn_target[train_mask].sum()),
        "validation_positive_rows": int(
            crn_target[validation_mask].sum()),
        "validation_agreement_with_current_label": float(
            np.mean(
                crn_target[validation_mask]
                == target[validation_mask]
            )
        ),
        "terminal_geometry_v2": {
            "full_auc": float(roc_auc_score(
                crn_target[validation_mask], crn_score)),
            "standardized_partial_auc_at_fpr_0p005": float(
                roc_auc_score(
                    crn_target[validation_mask],
                    crn_score,
                    max_fpr=0.005,
                )
            ),
            "recall_at_fpr_0p005": _recall_at_fpr(
                crn_target[validation_mask],
                crn_score,
                0.005,
            ),
        },
        "fixed_seed_teacher_probability": {
            "full_auc": float(roc_auc_score(
                crn_target[validation_mask],
                crn_probability[validation_mask],
            )),
            "recall_at_fpr_0p005": _recall_at_fpr(
                crn_target[validation_mask],
                crn_probability[validation_mask],
                0.005,
            ),
        },
    }

    payload = {
        "schema": "stage0_low_fidelity_teacher_probe_v1",
        "status": "COMPLETE",
        "split": "validation",
        "test_labels_used": False,
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(str(dataset_path)),
        "fixed_seed_namespace": "stage0-low-fidelity-teacher-v1",
        "replicates": int(args.replicates),
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "validation_positive_rows": int(
            target[validation_mask].sum()),
        "teacher_feature_count": int(len(teacher_columns)),
        "teacher_compute_seconds": float(teacher_seconds),
        "teacher_rows_per_second": float(
            len(frame) / max(teacher_seconds, 1e-9)),
        "results": results,
        "common_random_number_relabel_diagnostic": (
            crn_relabel_diagnostic),
    }
    output = Path(args.output).resolve()
    _write_json_atomic(output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
