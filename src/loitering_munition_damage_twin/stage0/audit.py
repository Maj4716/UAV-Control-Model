"""Deep audit utilities for a Stage-0 Parquet dataset.

Unlike the lightweight gate, this script first checks every physical Parquet
column independently.  This makes a damaged column diagnosable even when a
normal whole-table read fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import ks_2samp

from loitering_munition_damage_twin.stage0.component_supervision import (
    COMPONENT_SUPERVISION_PROFILE_FILENAME,
    COMPONENT_TARGET_COLUMNS,
    sha256_text_sequence,
)

TASKS = "KMFC"
SPLITS = ("train", "val", "test")
OBSERVABLE_FEATURES = [
    "x_cm", "y_cm", "z_cm", "vx_ms", "vy_ms", "vz_ms",
    "sin_yaw", "cos_yaw", "sin_pitch", "cos_pitch",
    "sin_roll", "cos_roll", "norm_velocity",
]


def scan_physical_columns(path: Path) -> dict:
    parquet = pq.ParquetFile(path)
    failures = {}
    try:
        for column in parquet.schema_arrow.names:
            try:
                table = parquet.read(columns=[column], use_threads=False)
                if table.num_rows != parquet.metadata.num_rows:
                    failures[column] = (
                        f"row count {table.num_rows} != {parquet.metadata.num_rows}")
            except Exception as exc:  # Arrow exposes several version-specific errors.
                failures[column] = f"{type(exc).__name__}: {exc}"
        return {
            "pyarrow_version": pa.__version__,
            "rows": parquet.metadata.num_rows,
            "row_groups": parquet.metadata.num_row_groups,
            "columns": parquet.metadata.num_columns,
            "column_failures": failures,
        }
    finally:
        parquet.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _counts(series: pd.Series) -> dict:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).sort_index().items()}


def _quantiles(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=float)
    keys = ("min", "p01", "p05", "p50", "p95", "p99", "max")
    if arr.size == 0:
        return {"count": 0, **{key: None for key in keys}}
    result = {"count": int(arr.size)}
    result.update({
        key: float(value)
        for key, value in zip(
            keys,
            np.quantile(arr, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]),
        )
    })
    return result


def _audit_component_supervision(
        dataset_path: Path,
        dataset_profile: dict,
        dataset_sample_ids: pd.Series) -> dict:
    profile_path = dataset_path.with_name(
        COMPONENT_SUPERVISION_PROFILE_FILENAME)
    if not profile_path.is_file():
        return {"status": "NOT_PRESENT_OPTIONAL"}
    with profile_path.open("r", encoding="utf-8") as stream:
        profile = json.load(stream)
    artifact = profile.get("artifact", {})
    sidecar_path = dataset_path.with_name(
        str(artifact.get("path", "")))
    checks = {
        "schema_match": (
            profile.get("schema")
            == "stage0_component_supervision_v1"),
        "base_sha256_match": (
            profile.get("base_dataset", {}).get("sha256")
            == dataset_profile.get("artifact", {}).get("sha256")),
        "base_rows_match": (
            int(profile.get("base_dataset", {}).get("rows", -1))
            == len(dataset_sample_ids)),
        "artifact_exists": sidecar_path.is_file(),
    }
    range_violations = None
    if sidecar_path.is_file():
        checks["artifact_size_match"] = (
            int(artifact.get("size_bytes", -1))
            == sidecar_path.stat().st_size)
        checks["artifact_sha256_match"] = (
            artifact.get("sha256") == _sha256(sidecar_path))
        sidecar = pd.read_parquet(
            sidecar_path, engine="pyarrow",
            columns=["sample_id", *COMPONENT_TARGET_COLUMNS])
        sidecar_ids = sidecar["sample_id"].astype(str)
        checks["artifact_rows_match"] = (
            len(sidecar) == len(dataset_sample_ids))
        checks["sample_id_order_match"] = (
            sidecar_ids.tolist()
            == dataset_sample_ids.astype(str).tolist())
        order_hash = sha256_text_sequence(sidecar_ids)
        checks["sample_id_hash_match"] = (
            order_hash
            == artifact.get("sample_id_order_sha256")
            == profile.get("base_dataset", {}).get(
                "sample_id_order_sha256"))
        values = sidecar[
            list(COMPONENT_TARGET_COLUMNS)].to_numpy(
                dtype=np.float32)
        range_violations = int((
            ~np.isfinite(values)
            | (values < 0.0)
            | (values > 1.0)
        ).sum())
        checks["probability_range_valid"] = (
            range_violations == 0)
    passed = bool(checks) and all(checks.values())
    return {
        "status": (
            "CURRENT_V1" if passed else "CONTRACT_MISMATCH"),
        "profile": str(profile_path),
        "artifact": str(sidecar_path),
        "checks": checks,
        "probability_range_violations": range_violations,
        "component_ids": profile.get(
            "target_contract", {}).get("component_ids"),
        "target_count": profile.get(
            "target_contract", {}).get("target_count"),
    }


def _read_with_compatible_engine(path: Path) -> tuple[pd.DataFrame, str, str | None]:
    try:
        return pd.read_parquet(path, engine="pyarrow"), "pyarrow", None
    except Exception as exc:
        pyarrow_error = f"{type(exc).__name__}: {exc}"
    try:
        return pd.read_parquet(path, engine="fastparquet"), "fastparquet", pyarrow_error
    except Exception as exc:
        raise RuntimeError(
            f"pyarrow failed ({pyarrow_error}); fastparquet failed "
            f"({type(exc).__name__}: {exc})") from exc


def _level_grid(df: pd.DataFrame) -> dict:
    result = {}
    for split in SPLITS:
        split_df = df[df["split_role"] == split]
        result[split] = {}
        for munition_id in range(4):
            cell = split_df[split_df["munition_id"] == munition_id]
            result[split][str(munition_id)] = {
                task: [
                    int((cell[f"{task}_level"] == level).sum())
                    for level in range(3)
                ]
                for task in TASKS
            }
    return result


def _required_exact_level(
        applicability: dict, munition_id: int,
        task: str, level: int) -> bool:
    if level == 0:
        return True
    munition = applicability.get(
        str(munition_id), applicability.get(munition_id, {}))
    flags = munition.get(task)
    return bool(
        isinstance(flags, list)
        and len(flags) == 2
        and flags[level - 1])


def _audit_exact_level_evidence(
        df: pd.DataFrame, profile: dict) -> dict:
    """Audit usable class evidence without trusting profile counts."""
    applicability = profile.get("ordinal_applicability", {})
    evaluation_contract = profile.get(
        "evaluation_exact_level_support")
    training_contract = profile.get(
        "training_exact_level_support", {})
    contract_present = isinstance(
        evaluation_contract, dict)
    evaluation_enforced = bool(
        evaluation_contract.get("enforced", False)
        if contract_present else False)
    minimum_eval_rows = int(
        evaluation_contract.get("minimum_rows", 100)
        if contract_present else 100)
    minimum_eval_roots = int(
        evaluation_contract.get(
            "minimum_root_families", 16)
        if contract_present else 16)
    minimum_train_rows = int(
        training_contract.get("minimum_rows", 128))
    minimum_train_roots = int(
        training_contract.get(
            "minimum_root_families", 16))
    reported_cells = (
        evaluation_contract.get("cells", {})
        if contract_present else {})

    cells = {}
    gaps = []
    profile_mismatches = []
    structural_zero_violations = []
    munition_values = df["munition_id"].to_numpy(
        dtype=np.int64)
    roles = df["split_role"].astype(str).to_numpy()
    roots = df["root_seed_id"].astype(str)
    for munition_id in range(4):
        cells[str(munition_id)] = {}
        munition_mask = munition_values == munition_id
        for task in TASKS:
            cells[str(munition_id)][task] = {}
            if f"{task}_level" in df.columns:
                levels = df[f"{task}_level"].to_numpy(
                    dtype=np.int8)
            else:
                levels = (
                    (df[f"{task}_ge1_prob"].to_numpy(
                        dtype=float) >= 0.5).astype(np.int8)
                    + (df[f"{task}_ge2_prob"].to_numpy(
                        dtype=float) >= 0.5).astype(np.int8)
                )
            for level in (0, 1, 2):
                required = _required_exact_level(
                    applicability, munition_id, task, level)
                level_rows = int(
                    (munition_mask & (levels == level)).sum())
                if not required:
                    if level_rows:
                        structural_zero_violations.append({
                            "munition_id": munition_id,
                            "task": task,
                            "level": level,
                            "rows": level_rows,
                        })
                    continue
                cells[str(munition_id)][task][str(level)] = {}
                for split_role in SPLITS:
                    mask = (
                        munition_mask
                        & (levels == level)
                        & (roles == split_role)
                    )
                    rows = int(mask.sum())
                    root_families = int(
                        roots[mask].nunique())
                    minimum_rows = (
                        minimum_train_rows
                        if split_role == "train"
                        else minimum_eval_rows)
                    minimum_roots = (
                        minimum_train_roots
                        if split_role == "train"
                        else minimum_eval_roots)
                    passed = bool(
                        rows >= minimum_rows
                        and root_families >= minimum_roots)
                    cell = {
                        "rows": rows,
                        "root_families": root_families,
                        "minimum_rows": minimum_rows,
                        "minimum_root_families": minimum_roots,
                        "passed": passed,
                    }
                    cells[str(munition_id)][task][
                        str(level)][split_role] = cell
                    if not passed:
                        gaps.append({
                            "split": split_role,
                            "munition_id": munition_id,
                            "task": task,
                            "level": level,
                            **cell,
                        })
                    if (
                        contract_present
                        and split_role in ("val", "test")
                    ):
                        reported = (
                            reported_cells
                            .get(str(munition_id), {})
                            .get(task, {})
                            .get(str(level), {})
                            .get(split_role)
                        )
                        if (
                            not isinstance(reported, dict)
                            or int(reported.get("rows", -1))
                            != rows
                            or int(reported.get(
                                "root_families", -1))
                            != root_families
                        ):
                            profile_mismatches.append({
                                "split": split_role,
                                "munition_id": munition_id,
                                "task": task,
                                "level": level,
                                "observed_rows": rows,
                                "observed_root_families":
                                    root_families,
                                "reported": reported,
                            })
    schema_current = bool(
        profile.get("profile_schema")
        == "stage0_lineage_v2"
        and profile.get("dataset_schema")
        == "stage0_lineage_v2")
    contract_ready = bool(
        schema_current
        and contract_present
        and evaluation_enforced
        and minimum_eval_rows >= 100
        and minimum_eval_roots >= 16
        and minimum_train_rows >= 128
        and not gaps
        and not profile_mismatches
        and not structural_zero_violations)
    return {
        "status": (
            "PASS" if contract_ready
            else "EVIDENCE_GAP_OR_CONTRACT_MISMATCH"),
        "contract_ready": contract_ready,
        "contract_present": contract_present,
        "evaluation_enforced": evaluation_enforced,
        "minimum_train_rows": minimum_train_rows,
        "minimum_train_root_families": minimum_train_roots,
        "minimum_eval_rows": minimum_eval_rows,
        "minimum_eval_root_families": minimum_eval_roots,
        "gap_count": len(gaps),
        "gaps": gaps,
        "profile_mismatch_count": len(
            profile_mismatches),
        "profile_mismatches": profile_mismatches,
        "structural_zero_violation_count": len(
            structural_zero_violations),
        "structural_zero_violations":
            structural_zero_violations,
        "cells": cells,
    }


def audit_statistics(path: Path, profile_path: Path) -> dict:
    df, read_engine, pyarrow_error = _read_with_compatible_engine(path)
    with profile_path.open("r", encoding="utf-8") as stream:
        profile = json.load(stream)
    component_supervision = _audit_component_supervision(
        path, profile, df["sample_id"])

    numeric = df.select_dtypes(include=[np.number])
    nonfinite = {
        column: int((~np.isfinite(numeric[column].to_numpy(dtype=float))).sum())
        for column in numeric.columns
    }
    nonfinite = {column: count for column, count in nonfinite.items() if count}
    missing = {column: int(count) for column, count in df.isna().sum().items() if count}

    roles = df["split_role"].astype(str)
    root_role_counts = df.groupby("root_seed_id")["split_role"].nunique()
    root_split_counts = (
        df[["root_seed_id", "split_role"]]
        .drop_duplicates("root_seed_id")["split_role"]
        .value_counts()
    )
    crawled = df["crawl_stage"].to_numpy(dtype=int) > 0
    parent = df["parent_id"].fillna("").astype(str)
    sample_ids = set(df["sample_id"].astype(str))
    crawled_parent = parent[crawled]
    parent_present = crawled_parent.isin(sample_ids)

    lineage = {
        "sample_id_duplicates": int(df["sample_id"].astype(str).duplicated().sum()),
        "root_families": int(df["root_seed_id"].nunique()),
        "root_split_counts": {str(k): int(v) for k, v in root_split_counts.items()},
        "cross_split_root_families": int((root_role_counts > 1).sum()),
        "root_family_size": _quantiles(df.groupby("root_seed_id").size().to_numpy()),
        "crawl_stage_counts": _counts(df["crawl_stage"]),
        "is_crawled_counts": _counts(df["is_crawled"]),
        "crawl_flag_stage_mismatch": int(
            ((df["is_crawled"].to_numpy(dtype=int) > 0) != crawled).sum()),
        "crawled_missing_parent_id": int((crawled_parent.str.len() == 0).sum()),
        "crawled_parent_not_in_final_table": int((~parent_present).sum()),
        "crawled_parent_in_final_table": int(parent_present.sum()),
        "crawled_rows_outside_train": int((roles[crawled] != "train").sum()),
        "families_over_10": int((df.groupby("root_seed_id").size() > 10).sum()),
        "families_over_100": int((df.groupby("root_seed_id").size() > 100).sum()),
        "families_over_1000": int((df.groupby("root_seed_id").size() > 1000).sum()),
        "largest_families": {
            str(root): int(size)
            for root, size in df.groupby("root_seed_id").size().nlargest(10).items()
        },
    }

    alias_pairs = [
        ("x_cm", "x"), ("y_cm", "y"), ("z_cm", "z"),
        ("vx_ms", "vx"), ("vy_ms", "vy"), ("vz_ms", "vz"),
        ("munition_id", "m_id"),
    ]
    alias_max_error = {
        f"{left}__{right}": float(np.max(np.abs(
            df[left].to_numpy(dtype=float) - df[right].to_numpy(dtype=float))))
        for left, right in alias_pairs
    }
    speed = np.sqrt(df["vx_ms"] ** 2 + df["vy_ms"] ** 2 + df["vz_ms"] ** 2)
    trig_errors = {}
    for angle in ("yaw", "pitch", "roll"):
        radians = np.radians(df[angle].to_numpy(dtype=float))
        trig_errors[f"sin_{angle}"] = float(np.max(np.abs(
            df[f"sin_{angle}"].to_numpy(dtype=float) - np.sin(radians))))
        trig_errors[f"cos_{angle}"] = float(np.max(np.abs(
            df[f"cos_{angle}"].to_numpy(dtype=float) - np.cos(radians))))

    yaw = np.radians(df["yaw"].to_numpy(dtype=float))
    pitch = np.radians(df["pitch"].to_numpy(dtype=float))
    heading = np.column_stack((
        np.cos(pitch) * np.sin(yaw),
        np.cos(pitch) * np.cos(yaw),
        np.sin(pitch),
    ))
    velocity = df[["vx_ms", "vy_ms", "vz_ms"]].to_numpy(dtype=float)
    cos_aoa = np.clip(np.einsum("ij,ij->i", heading, velocity) / speed.to_numpy(), -1.0, 1.0)
    aoa_deg = np.degrees(np.arccos(cos_aoa))

    position = df[["x_cm", "y_cm", "z_cm"]].to_numpy(dtype=float)
    target = df[["target_x", "target_y", "target_z"]].to_numpy(dtype=float)
    line = target - position
    line_norm = np.linalg.norm(line, axis=1)
    closing_cos = np.einsum("ij,ij->i", line, velocity) / (line_norm * speed.to_numpy())
    kinematics = {
        "alias_max_abs_error": alias_max_error,
        "trig_max_abs_error": trig_errors,
        "norm_velocity_max_abs_error": float(np.max(np.abs(speed - df["norm_velocity"]))),
        "aoa_deg": _quantiles(aoa_deg),
        "aoa_over_30_deg": int((aoa_deg > 30.0 + 1e-8).sum()),
        "closing_cosine": _quantiles(closing_cos),
        "not_pointing_toward_sampler_target": int((closing_cos <= 0.0).sum()),
        "stored_impact_cosine_max_abs_error": float(np.max(np.abs(
            closing_cos - df["impact_cosine"].to_numpy(dtype=float)))),
        "position_ranges_cm": {
            axis: _quantiles(df[axis].to_numpy()) for axis in ("x_cm", "y_cm", "z_cm")
        },
        "speed_mps": _quantiles(speed.to_numpy()),
    }

    probability_errors = {}
    for task in TASKS:
        p1 = df[f"{task}1_prob"].to_numpy(dtype=float)
        p2 = df[f"{task}2_prob"].to_numpy(dtype=float)
        expected_ge1 = p1 if task == "C" else 1.0 - (1.0 - p1) * (1.0 - p2)
        expected_ge2 = np.minimum(p2, expected_ge1)
        probability_errors[task] = {
            "ge1_formula_max_abs_error": float(np.max(np.abs(
                df[f"{task}_ge1_prob"].to_numpy(dtype=float) - expected_ge1))),
            "ge2_formula_max_abs_error": float(np.max(np.abs(
                df[f"{task}_ge2_prob"].to_numpy(dtype=float) - expected_ge2))),
            "ordinal_violations": int((
                df[f"{task}_ge2_prob"].to_numpy(dtype=float)
                > df[f"{task}_ge1_prob"].to_numpy(dtype=float) + 1e-12
            ).sum()),
            "level_mismatch": int((
                (df[f"{task}_ge1_prob"].to_numpy(dtype=float) >= 0.5).astype(int)
                + (df[f"{task}_ge2_prob"].to_numpy(dtype=float) >= 0.5).astype(int)
                != df[f"{task}_level"].to_numpy(dtype=int)
            ).sum()),
            "ge1_near_0_5": int((np.abs(
                df[f"{task}_ge1_prob"].to_numpy(dtype=float) - 0.5) <= 0.05).sum()),
            "ge2_near_0_5": int((np.abs(
                df[f"{task}_ge2_prob"].to_numpy(dtype=float) - 0.5) <= 0.05).sum()),
        }

    std_summary = {}
    for task in TASKS:
        for level in (1, 2):
            column = f"{task}{level}_prob_std"
            values = df[column].to_numpy(dtype=float)
            std_summary[column] = {
                "mean": float(values.mean()),
                "p95": float(np.quantile(values, 0.95)),
                "max": float(values.max()),
                "over_0_20": int((values > 0.20).sum()),
                "over_0_40": int((values > 0.40).sum()),
                "nonzero": int((values > 1e-12).sum()),
            }

    level_grid = _level_grid(df)
    exact_level_evidence = _audit_exact_level_evidence(
        df, profile)
    positive_family_diversity = {}
    train_rows = df[roles == "train"]
    for munition_id in range(4):
        positive_family_diversity[str(munition_id)] = {}
        for task in TASKS:
            positive_family_diversity[str(munition_id)][task] = {}
            for level in (1, 2):
                family_sizes = (
                    train_rows[
                        (train_rows["munition_id"] == munition_id)
                        & (train_rows[f"{task}_level"] == level)
                    ]
                    .groupby("root_seed_id")
                    .size()
                    .sort_values(ascending=False)
                )
                total = int(family_sizes.sum())
                roots = int(len(family_sizes))
                top = int(family_sizes.iloc[0]) if roots else 0
                effective_roots = (
                    float(total ** 2 / np.square(family_sizes.to_numpy(dtype=float)).sum())
                    if roots else 0.0
                )
                positive_family_diversity[str(munition_id)][task][str(level)] = {
                    "rows": total,
                    "root_families": roots,
                    "largest_family_rows": top,
                    "largest_family_share": float(top / max(total, 1)),
                    "effective_root_families": effective_roots,
                }
    rare_cells = []
    for split in SPLITS:
        for munition_id in range(4):
            for task in TASKS:
                for level, count in enumerate(level_grid[split][str(munition_id)][task]):
                    if count < 100:
                        rare_cells.append({
                            "split": split, "munition_id": munition_id,
                            "task": task, "level": level, "count": count,
                        })

    label_rates = {}
    for split in SPLITS:
        subset = df[roles == split]
        label_rates[split] = {
            f"{task}_ge{level}": float(
                (subset[f"{task}_ge{level}_prob"] >= 0.5).mean())
            for task in TASKS for level in (1, 2)
        }

    no_hits = df["total_hits"].to_numpy(dtype=float) <= 1e-12
    no_hit_damage = {
        "rows": int(no_hits.sum()),
        "share": float(no_hits.mean()),
        "positive_rates": {
            f"{task}_ge{level}": float(
                (df.loc[no_hits, f"{task}_ge{level}_prob"] >= 0.5).mean())
            for task in TASKS for level in (1, 2)
        },
        "by_munition": {
            str(munition_id): {
                "rows": int((no_hits & (df["munition_id"].to_numpy() == munition_id)).sum()),
                **{
                    f"{task}_level_ge1_rate": float((
                        df.loc[
                            no_hits & (df["munition_id"].to_numpy() == munition_id),
                            f"{task}_level",
                        ] >= 1
                    ).mean())
                    for task in TASKS
                },
            }
            for munition_id in range(4)
        },
    }

    ks = {}
    train = df[roles == "train"]
    test = df[roles == "test"]
    for column in OBSERVABLE_FEATURES:
        result = ks_2samp(
            train[column].to_numpy(), test[column].to_numpy(), method="asymp")
        ks[column] = {"statistic": float(result.statistic), "pvalue": float(result.pvalue)}

    exact_feature_duplicates = int(df.duplicated(OBSERVABLE_FEATURES + ["munition_id"]).sum())
    rounded = pd.DataFrame({
        "x": np.round(df["x_cm"], 0), "y": np.round(df["y_cm"], 0),
        "z": np.round(df["z_cm"], 0), "vx": np.round(df["vx_ms"], 1),
        "vy": np.round(df["vy_ms"], 1), "vz": np.round(df["vz_ms"], 1),
        "yaw": np.round(df["yaw"], 1), "pitch": np.round(df["pitch"], 1),
        "roll": np.round(df["roll"], 1), "m": df["munition_id"].to_numpy(),
    })
    rounded_hash = pd.util.hash_pandas_object(rounded, index=False)
    rounded_frame = pd.DataFrame({"signature": rounded_hash, "split": roles})
    rounded_cross = rounded_frame.groupby("signature")["split"].nunique()

    weights = df["loss_weight"].to_numpy(dtype=float)
    training_cap = 20.0 if profile.get("dataset_schema") == "stage0_lineage_v2" else 200.0
    clipped_weights = np.clip(weights, 0.05, training_cap)

    def _ess_summary(values: np.ndarray) -> dict:
        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            return {
                "count": 0,
                "effective_sample_size": 0.0,
                "effective_sample_size_ratio": 0.0,
            }
        ess = float(values.sum() ** 2 / np.square(values).sum())
        return {
            "count": int(len(values)),
            "effective_sample_size": ess,
            "effective_sample_size_ratio": float(ess / len(values)),
        }

    weighting = {
        "loss_weight": _quantiles(weights),
        "nonpositive": int((weights <= 0).sum()),
        "effective_sample_size": float(weights.sum() ** 2 / np.square(weights).sum()),
        "below_training_floor_0_05": int((weights < 0.05).sum()),
        "training_cap": training_cap,
        "above_training_cap": int((weights > training_cap).sum()),
        "clipped_effective_sample_size": float(
            clipped_weights.sum() ** 2 / np.square(clipped_weights).sum()),
        "by_split": {
            split: _quantiles(df.loc[roles == split, "loss_weight"].to_numpy())
            for split in SPLITS
        },
        "ess_by_split": {
            split: _ess_summary(
                df.loc[roles == split, "loss_weight"].to_numpy(dtype=float))
            for split in SPLITS
        },
        "by_crawl_flag": {
            str(flag): _quantiles(df.loc[df["is_crawled"] == flag, "loss_weight"].to_numpy())
            for flag in (0, 1)
        },
        "K_task_weight_values": _counts(df["K_task_weight"]),
        "C_task_weight_values": _counts(df["C_task_weight"]),
    }
    for component in (
        "aoa_accept_prob", "aoa_ipw", "physics_weight", "active_sampling_weight",
        "family_weight", "class_balance_weight", "loss_weight_raw",
    ):
        if component in df.columns:
            weighting[component] = _quantiles(df[component].to_numpy(dtype=float))

    mc_histogram = _counts(df["label_mc_replicates"].astype(int))
    expected_mc_histogram = profile.get("label_mc", {}).get("replicate_histogram")
    if expected_mc_histogram is None and profile.get("label_mc_replicates") is not None:
        expected_mc_histogram = {
            str(int(profile["label_mc_replicates"])): int(len(df))
        }
    mc_resolved_columns = [
        f"{task}_ge{level}_mc_resolved"
        for task in TASKS
        for level in (1, 2)
    ]
    mc_standard_error_columns = [
        f"{task}_ge{level}_mc_standard_error"
        for task in TASKS
        for level in (1, 2)
    ]
    mc_convergence_available = all(
        column in df.columns
        for column in (
            mc_resolved_columns
            + mc_standard_error_columns
            + [
                "label_mc_all_resolved",
                "label_mc_max_reached",
            ]
        )
    )
    if mc_convergence_available:
        all_resolved = df[
            "label_mc_all_resolved"].astype(bool)
        maximum_reached = df[
            "label_mc_max_reached"].astype(bool)
        mc_convergence = {
            "available": True,
            "all_resolved_rows": int(all_resolved.sum()),
            "all_resolved_ratio": float(all_resolved.mean()),
            "maximum_reached_rows": int(maximum_reached.sum()),
            "maximum_reached_ratio": float(maximum_reached.mean()),
            "resolved_ratio_by_head": {
                column: float(df[column].astype(bool).mean())
                for column in mc_resolved_columns
            },
            "standard_error_by_head": {
                column: _quantiles(
                    df[column].to_numpy(dtype=float))
                for column in mc_standard_error_columns
            },
        }
    else:
        mc_convergence = {"available": False}

    profile_checks = {
        "rows_match": int(profile.get("artifact", {}).get("rows", -1)) == len(df),
        "size_match": int(profile.get("artifact", {}).get("size_bytes", -1)) == path.stat().st_size,
        "sha256_match": str(profile.get("artifact", {}).get("sha256", "")) == _sha256(path),
        "split_counts_match": profile.get("split_counts", {}) == _counts(df["split_role"]),
        "label_mc_replicates_match": mc_histogram == expected_mc_histogram,
        "label_mc_convergence_match": (
            (
                int(profile.get("label_mc", {}).get(
                    "all_resolved_rows", -1))
                == mc_convergence["all_resolved_rows"]
                and int(profile.get("label_mc", {}).get(
                    "maximum_reached_rows", -1))
                == mc_convergence["maximum_reached_rows"]
            )
            if mc_convergence_available else None
        ),
    }
    k2_positive_rows = int((df["K_ge2_prob"] >= 0.5).sum())
    k2_observed_ratio = float(k2_positive_rows / max(len(df), 1))
    k2_contract = profile.get("k2_ratio_contract", {})
    k2_ratio_audit = {
        "enforced": bool(k2_contract.get("enforced", True)),
        "phase2_stop_ratio": k2_contract.get("phase2_stop_ratio"),
        "final_max_ratio": k2_contract.get("final_max_ratio"),
        "observed_positive_rows": k2_positive_rows,
        "observed_final_ratio": k2_observed_ratio,
        "profile_rows_match": (
            int(k2_contract.get("observed_positive_rows", -1)) ==
            k2_positive_rows
        ),
        "profile_ratio_match": (
            abs(float(k2_contract.get("observed_final_ratio", -1.0)) -
                k2_observed_ratio) <= 1e-12
        ),
        "within_final_max": (
            k2_contract.get("final_max_ratio") is not None and
            k2_observed_ratio <= float(k2_contract["final_max_ratio"]) + 1e-12
        ),
    }

    return {
        "artifact_identity": {
            "profile_schema": profile.get("profile_schema"),
            "dataset_schema_profile": profile.get("dataset_schema"),
            "dataset_schema_columns": sorted(
                df["dataset_schema"].astype(str).unique().tolist())
                if "dataset_schema" in df.columns else [],
            "expected_current_schema": "stage0_lineage_v2",
            "current_schema_match": bool(
                profile.get("profile_schema") == "stage0_lineage_v2" and
                profile.get("dataset_schema") == "stage0_lineage_v2" and
                set(df["dataset_schema"].astype(str).unique()) == {"stage0_lineage_v2"}
            ) if "dataset_schema" in df.columns else False,
            "artifact_sha256": _sha256(path),
        },
        "read_engine": read_engine,
        "pyarrow_read_error": pyarrow_error,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_values": missing,
        "nonfinite_values": nonfinite,
        "split_counts": _counts(df["split_role"]),
        "munition_counts": _counts(df["munition_id"]),
        "layer_counts": _counts(df["layer_type"]),
        "lineage": lineage,
        "kinematics": kinematics,
        "probability_consistency": probability_errors,
        "mc_std_summary": std_summary,
        "label_mc_replicate_histogram": mc_histogram,
        "label_mc_convergence": mc_convergence,
        "mechanism_probability_summary": {
            mechanism: {
                f"{task}_ge{level}": _quantiles(
                    df[f"{mechanism}_{task}_ge{level}_prob"].to_numpy(dtype=float))
                for task in TASKS for level in (1, 2)
                if f"{mechanism}_{task}_ge{level}_prob" in df.columns
            }
            for mechanism in ("fragment", "shock")
        },
        "component_supervision": component_supervision,
        "level_grid": level_grid,
        "exact_level_evidence": exact_level_evidence,
        "positive_family_diversity": positive_family_diversity,
        "cells_below_100": rare_cells,
        "label_positive_rates": label_rates,
        "no_hit_damage": no_hit_damage,
        "train_test_feature_ks": ks,
        "exact_observable_feature_duplicates": exact_feature_duplicates,
        "rounded_cross_split_signatures": int((rounded_cross > 1).sum()),
        "weighting": weighting,
        "k2_ratio_contract": k2_ratio_audit,
        "hits": _quantiles(df["total_hits"].to_numpy()),
        "penetrations": _quantiles(df["total_penetrations"].to_numpy()),
        "penetrations_over_hits": int((df["total_penetrations"] > df["total_hits"] + 1e-12).sum()),
        "overall_score": _quantiles(df["overall_score"].to_numpy()),
        "profile_checks": profile_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="深度审计 Stage-0 Parquet 数据集。")
    parser.add_argument("parquet_path")
    parser.add_argument(
        "--physical-only", action="store_true",
        help="只逐列扫描 Parquet 物理可读性。",
    )
    parser.add_argument(
        "--output", default="output/stage0_dataset_audit.json",
        help="完整审计 JSON 输出路径。",
    )
    args = parser.parse_args()
    report = scan_physical_columns(Path(args.parquet_path).resolve())
    if args.physical_only:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if report["column_failures"]:
            raise SystemExit(2)
        return

    profile_path = Path(args.parquet_path).resolve().with_name("generation_profile.json")
    statistical = audit_statistics(Path(args.parquet_path).resolve(), profile_path)
    contract_status = (
        (
            "CURRENT_V2"
            if statistical["exact_level_evidence"]["contract_ready"]
            else "CURRENT_V2_EVIDENCE_GAP"
        )
        if statistical["artifact_identity"]["current_schema_match"]
        else "LEGACY_OR_SCHEMA_MISMATCH"
    )
    report["status"] = "AUDIT_COMPLETE"
    report["contract_status"] = contract_status
    report["statistics"] = statistical
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print(json.dumps({
        "status": "AUDIT_COMPLETE",
        "contract_status": contract_status,
        "artifact_identity": statistical["artifact_identity"],
        "output": str(output_path),
        "read_engine": statistical["read_engine"],
        "rows": statistical["rows"],
        "cells_below_100": len(statistical["cells_below_100"]),
        "exact_level_evidence": {
            key: statistical["exact_level_evidence"][key]
            for key in (
                "status",
                "contract_ready",
                "contract_present",
                "evaluation_enforced",
                "gap_count",
                "profile_mismatch_count",
                "structural_zero_violation_count",
            )
        },
        "cross_split_root_families": statistical["lineage"]["cross_split_root_families"],
        "train_weight_ess_ratio": (
            statistical["weighting"]
            .get("ess_by_split", {})
            .get("train", {})
            .get("effective_sample_size_ratio")
        ),
        "k2_ratio_contract": statistical["k2_ratio_contract"],
        "label_mc_convergence": (
            {
                key: statistical["label_mc_convergence"].get(key)
                for key in (
                    "available",
                    "all_resolved_ratio",
                    "maximum_reached_ratio",
                )
                if key in statistical["label_mc_convergence"]
            }
        ),
        "component_supervision_status": (
            statistical["component_supervision"]["status"]),
        "profile_checks": statistical["profile_checks"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
