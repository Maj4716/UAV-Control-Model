"""Validate a Stage-0 damage dataset without importing PyTorch.

The validator is intentionally independent from ``nn_dataset.py`` so a newly
generated artifact can be accepted or rejected on a lightweight simulation
machine before it is copied to the training environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from loitering_munition_damage_twin.simulation.coordinate_frames import FRAME_CONVENTION_VERSION
from loitering_munition_damage_twin.stage0.component_supervision import (
    COMPONENT_SUPERVISION_PROFILE_FILENAME,
    COMPONENT_TARGET_COLUMNS,
    sha256_text_sequence,
    validate_component_supervision_profile,
)


PROFILE_SCHEMA = "stage0_lineage_v2"
DATASET_SCHEMA = "stage0_lineage_v2"
ALLOWED_SPLITS = {"train", "val", "test"}
LINEAGE_COLUMNS = {
    "sample_id",
    "root_seed_id",
    "parent_id",
    "crawl_stage",
    "split_role",
    "frame_version",
    "dataset_schema",
    "label_mc_replicates",
    "label_mc_min_replicates",
    "label_mc_max_replicates",
}
KINEMATIC_COLUMNS = {
    "x_cm", "y_cm", "z_cm", "vx_ms", "vy_ms", "vz_ms",
    "sin_yaw", "cos_yaw", "sin_pitch", "cos_pitch",
    "sin_roll", "cos_roll", "norm_velocity", "munition_id",
}
ORDINAL_COLUMNS = {
    f"{task}_{level}_prob"
    for task in "KMFC"
    for level in ("ge1", "ge2")
}
ORDINAL_STD_COLUMNS = {f"{column}_std" for column in ORDINAL_COLUMNS}
MC_RESOLUTION_COLUMNS = {
    f"{task}_{level}_mc_resolved"
    for task in "KMFC"
    for level in ("ge1", "ge2")
}
MC_STANDARD_ERROR_COLUMNS = {
    f"{task}_{level}_mc_standard_error"
    for task in "KMFC"
    for level in ("ge1", "ge2")
}
MECHANISM_COLUMNS = {
    f"{mechanism}_{task}_{level}_prob"
    for mechanism in ("fragment", "shock")
    for task in "KMFC"
    for level in ("ge1", "ge2")
}
WEIGHT_COLUMNS = {
    "loss_weight", "aoa_accept_prob", "aoa_ipw", "physics_weight",
    "active_sampling_weight", "family_weight", "class_balance_weight",
}


class Stage0ValidationError(RuntimeError):
    """Raised when a dataset violates a Stage-0 data contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage0ValidationError(message)


def _required_exact_level(
    applicability: dict[str, Any],
    munition_id: int,
    task: str,
    level: int,
) -> bool:
    """Return whether an exact ordinal level belongs to the data contract."""
    if level == 0:
        return True
    munition = applicability.get(
        str(munition_id), applicability.get(munition_id, {}))
    task_applicability = munition.get(task)
    _require(
        isinstance(task_applicability, list)
        and len(task_applicability) == 2,
        f"profile ordinal_applicability 缺少 m_id={munition_id}:{task}。",
    )
    return bool(task_applicability[level - 1])


def _validate_exact_level_evidence(
    df: pd.DataFrame,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Independently verify production train/val/test class evidence."""
    gate = profile.get("usability_gate", {})
    contract = profile.get("evaluation_exact_level_support")
    if not bool(gate.get("enforced", False)):
        return {
            "enforced": False,
            "passed": True,
            "minimum_train_rows": None,
            "minimum_eval_rows": None,
            "minimum_eval_root_families": None,
            "checked_cells": 0,
        }

    _require(
        isinstance(contract, dict),
        "正式数据集缺少 evaluation_exact_level_support 合同；"
        "该数据集必须由当前生成器重新构建。",
    )
    _require(
        bool(contract.get("enforced", False)),
        "正式数据集的 evaluation_exact_level_support 未启用。",
    )
    minimum_eval_rows = int(contract.get("minimum_rows", 0))
    minimum_eval_roots = int(
        contract.get("minimum_root_families", 0))
    _require(
        minimum_eval_rows >= 100 and minimum_eval_roots >= 16,
        "评估证据合同不得低于每个适用单元 100 行、16 个 root。",
    )
    minimum_train_rows = int(
        profile.get("training_exact_level_support", {}).get(
            "minimum_rows", 128))
    minimum_train_roots = int(
        profile.get("training_exact_level_support", {}).get(
            "minimum_root_families", 16))
    _require(
        minimum_train_rows >= 128 and minimum_train_roots >= 16,
        "训练精确等级证据合同不得低于每个适用单元 "
        "128 行、16 个 root。",
    )
    applicability = profile.get("ordinal_applicability")
    _require(
        isinstance(applicability, dict),
        "profile 缺少 ordinal_applicability。",
    )
    reported_cells = contract.get("cells")
    _require(
        isinstance(reported_cells, dict),
        "evaluation_exact_level_support 缺少 cells。",
    )

    checked_cells = 0
    observed_support: dict[str, Any] = {}
    munition_values = df["munition_id"].to_numpy(dtype=np.int64)
    roles = df["split_role"].astype(str).to_numpy()
    roots = df["root_seed_id"].astype(str)
    for munition_id in range(4):
        observed_support[str(munition_id)] = {}
        munition_mask = munition_values == munition_id
        for task in "KMFC":
            observed_support[str(munition_id)][task] = {}
            inferred_level = (
                (df[f"{task}_ge1_prob"].to_numpy(dtype=np.float64) >= 0.5)
                .astype(np.int8)
                + (df[f"{task}_ge2_prob"].to_numpy(dtype=np.float64) >= 0.5)
                .astype(np.int8)
            )
            if f"{task}_level" in df.columns:
                stored_level = df[f"{task}_level"].to_numpy(
                    dtype=np.int8)
                _require(
                    np.array_equal(stored_level, inferred_level),
                    f"{task}_level 与序数概率阈值不一致。",
                )
            for level in (0, 1, 2):
                if not _required_exact_level(
                    applicability, munition_id, task, level
                ):
                    continue
                observed_support[str(munition_id)][task][str(level)] = {}
                for split_role in ("train", "val", "test"):
                    mask = (
                        munition_mask
                        & (roles == split_role)
                        & (inferred_level == level)
                    )
                    rows = int(mask.sum())
                    root_families = int(roots[mask].nunique())
                    observed = {
                        "rows": rows,
                        "root_families": root_families,
                    }
                    observed_support[str(munition_id)][task][
                        str(level)][split_role] = observed
                    if split_role == "train":
                        _require(
                            rows >= minimum_train_rows,
                            f"m_id={munition_id}:{task}=L{level} "
                            f"train 行数={rows} < {minimum_train_rows}。",
                        )
                        _require(
                            root_families >= minimum_train_roots,
                            f"m_id={munition_id}:{task}=L{level} "
                            f"train root={root_families} < "
                            f"{minimum_train_roots}。",
                        )
                        continue
                    checked_cells += 1
                    _require(
                        rows >= minimum_eval_rows,
                        f"m_id={munition_id}:{task}=L{level} "
                        f"{split_role} 行数={rows} < "
                        f"{minimum_eval_rows}。",
                    )
                    _require(
                        root_families >= minimum_eval_roots,
                        f"m_id={munition_id}:{task}=L{level} "
                        f"{split_role} root={root_families} < "
                        f"{minimum_eval_roots}。",
                    )
                    reported = (
                        reported_cells
                        .get(str(munition_id), {})
                        .get(task, {})
                        .get(str(level), {})
                        .get(split_role)
                    )
                    _require(
                        isinstance(reported, dict),
                        f"profile 缺少 m_id={munition_id}:{task}=L{level} "
                        f"{split_role} 证据统计。",
                    )
                    _require(
                        int(reported.get("rows", -1)) == rows
                        and int(reported.get(
                            "root_families", -1)) == root_families,
                        f"profile 与 Parquet 的 m_id={munition_id}:"
                        f"{task}=L{level} {split_role} 证据不一致。",
                    )
    return {
        "enforced": True,
        "passed": True,
        "minimum_train_rows": minimum_train_rows,
        "minimum_train_root_families": minimum_train_roots,
        "minimum_eval_rows": minimum_eval_rows,
        "minimum_eval_root_families": minimum_eval_roots,
        "checked_cells": checked_cells,
        "observed": observed_support,
    }


def validate_stage0_dataset(
    parquet_path: str | os.PathLike[str],
    profile_path: str | os.PathLike[str] | None = None,
    verify_hash: bool = True,
) -> dict[str, Any]:
    """Validate schema, lineage isolation, probabilities and artifact hash."""
    dataset_path = Path(parquet_path).resolve()
    profile_file = (
        Path(profile_path).resolve()
        if profile_path is not None
        else dataset_path.with_name("generation_profile.json")
    )
    _require(dataset_path.is_file(), f"数据集不存在: {dataset_path}")
    _require(profile_file.is_file(), f"generation profile 不存在: {profile_file}")

    with profile_file.open("r", encoding="utf-8") as stream:
        profile = json.load(stream)

    _require(
        profile.get("profile_schema") == PROFILE_SCHEMA,
        f"profile_schema={profile.get('profile_schema')!r}，期望 {PROFILE_SCHEMA!r}",
    )
    _require(
        profile.get("dataset_schema") == DATASET_SCHEMA,
        f"dataset_schema={profile.get('dataset_schema')!r}，期望 {DATASET_SCHEMA!r}",
    )
    _require(
        profile.get("frame_convention") == FRAME_CONVENTION_VERSION,
        "坐标系版本不匹配；旧坐标数据不得进入 Stage-0 训练。",
    )
    _require(
        profile.get("phase2_mode") == "per_munition_topoff",
        "phase2_mode 必须为 per_munition_topoff。",
    )

    required = (LINEAGE_COLUMNS | KINEMATIC_COLUMNS | ORDINAL_COLUMNS |
                ORDINAL_STD_COLUMNS | MECHANISM_COLUMNS | WEIGHT_COLUMNS)
    parquet_engine = "pyarrow"
    try:
        df = pd.read_parquet(dataset_path, engine="pyarrow")
    except (OSError, ValueError, TypeError) as pyarrow_error:
        # Files written by a newer Arrow may occasionally hit an older Arrow
        # decoder bug.  fastparquet is an independent compatibility path; the
        # artifact hash still guarantees that no alternate file was validated.
        try:
            df = pd.read_parquet(dataset_path, engine="fastparquet")
            parquet_engine = "fastparquet_fallback"
        except (ImportError, OSError, ValueError, TypeError) as fallback_error:
            raise Stage0ValidationError(
                f"Parquet 回读失败；pyarrow={pyarrow_error}; "
                f"fastparquet={fallback_error}") from fallback_error
    missing = sorted(required - set(df.columns))
    _require(not missing, f"数据表缺少 Stage-0 字段: {missing}")
    _require(len(df) > 0, "数据表为空。")

    _require(not df["sample_id"].isna().any(), "sample_id 存在空值。")
    _require(not df["sample_id"].astype(str).duplicated().any(), "sample_id 不唯一。")
    _require(not df["root_seed_id"].isna().any(), "root_seed_id 存在空值。")
    _require((df["root_seed_id"].astype(str).str.len() > 0).all(), "root_seed_id 存在空字符串。")
    _require((df["crawl_stage"] >= 0).all(), "crawl_stage 必须为非负整数。")

    roles = set(df["split_role"].astype(str).unique())
    _require(roles <= ALLOWED_SPLITS, f"发现未知 split_role: {sorted(roles - ALLOWED_SPLITS)}")
    _require(roles == ALLOWED_SPLITS, f"train/val/test 必须全部存在，当前为 {sorted(roles)}")
    root_split_counts = df.groupby("root_seed_id", dropna=False)["split_role"].nunique()
    cross_split_roots = int((root_split_counts > 1).sum())
    _require(cross_split_roots == 0, f"发现 {cross_split_roots} 个 root family 跨 split。")

    crawled = df["crawl_stage"].to_numpy(dtype=np.int64) > 0
    parent_ids = df["parent_id"].fillna("").astype(str).to_numpy(dtype=str)
    if crawled.any():
        _require((np.char.str_len(parent_ids[crawled]) > 0).all(), "爬行样本缺少 parent_id。")

    frame_values = set(df["frame_version"].astype(str).unique())
    schema_values = set(df["dataset_schema"].astype(str).unique())
    _require(frame_values == {FRAME_CONVENTION_VERSION}, f"frame_version 混杂: {frame_values}")
    _require(schema_values == {DATASET_SCHEMA}, f"dataset_schema 混杂: {schema_values}")
    mc_actual = df["label_mc_replicates"].astype(int)
    mc_minimum = df["label_mc_min_replicates"].astype(int)
    mc_maximum = df["label_mc_max_replicates"].astype(int)
    _require(
        not ((mc_actual < 1) | (mc_minimum < 1) | (mc_maximum < mc_minimum) |
             (mc_actual < mc_minimum) | (mc_actual > mc_maximum)).any(),
        "自适应 MC 次数不满足 min <= actual <= max。",
    )
    mc_histogram = {
        str(int(k)): int(v) for k, v in mc_actual.value_counts().sort_index().items()
    }
    _require(
        profile.get("label_mc", {}).get("replicate_histogram") == mc_histogram,
        "profile 与数据表的自适应 MC 次数直方图不一致。",
    )

    label_mc_profile = profile.get("label_mc", {})
    resolution_ratio = None
    maximum_reached_ratio = None
    if int(label_mc_profile.get("maximum_configured", 0)) >= 64:
        required_mc_quality_columns = (
            MC_RESOLUTION_COLUMNS
            | MC_STANDARD_ERROR_COLUMNS
            | {
                "label_mc_all_resolved",
                "label_mc_max_reached",
                "label_mc_max_standard_error",
            }
        )
        missing_mc_quality = sorted(
            required_mc_quality_columns - set(df.columns))
        _require(
            not missing_mc_quality,
            "64-replicate MC contract is missing convergence columns: "
            f"{missing_mc_quality}")
        _require(
            int(label_mc_profile.get("minimum_configured", 0)) >= 8,
            "64-replicate MC contract requires minimum_configured >= 8.")
        _require(
            float(label_mc_profile.get(
                "standard_error_target", 1.0)) <= 0.02,
            "64-replicate MC contract requires standard_error_target <= 0.02.")
        _require(
            label_mc_profile.get("antithetic_pairs") is True,
            "64-replicate MC contract requires antithetic_pairs=true.")
        resolved_column_order = [
            f"{task}_{level}_mc_resolved"
            for task in "KMFC"
            for level in ("ge1", "ge2")
        ]
        standard_error_column_order = [
            f"{task}_{level}_mc_standard_error"
            for task in "KMFC"
            for level in ("ge1", "ge2")
        ]
        resolved_matrix = df[
            resolved_column_order].to_numpy(dtype=bool)
        standard_error_matrix = df[
            standard_error_column_order].to_numpy(dtype=np.float64)
        _require(
            np.isfinite(standard_error_matrix).all()
            and (standard_error_matrix >= 0.0).all(),
            "MC standard-error diagnostics contain invalid values.")
        all_resolved = df[
            "label_mc_all_resolved"].to_numpy(dtype=bool)
        max_reached = df[
            "label_mc_max_reached"].to_numpy(dtype=bool)
        _require(
            np.array_equal(
                all_resolved, resolved_matrix.all(axis=1)),
            "label_mc_all_resolved does not match per-head diagnostics.")
        _require(
            np.array_equal(
                max_reached,
                mc_actual.to_numpy() >= mc_maximum.to_numpy()),
            "label_mc_max_reached does not match actual/max replicates.")
        reported_max_se = df[
            "label_mc_max_standard_error"
        ].to_numpy(dtype=np.float64)
        _require(
            np.allclose(
                reported_max_se,
                standard_error_matrix.max(axis=1),
                rtol=1e-6,
                atol=1e-8,
            ),
            "label_mc_max_standard_error is inconsistent.")
        resolution_ratio = float(all_resolved.mean())
        maximum_reached_ratio = float(max_reached.mean())
        _require(
            int(label_mc_profile.get(
                "all_resolved_rows", -1))
            == int(all_resolved.sum()),
            "profile label_mc.all_resolved_rows mismatch.")
        _require(
            int(label_mc_profile.get(
                "maximum_reached_rows", -1))
            == int(max_reached.sum()),
            "profile label_mc.maximum_reached_rows mismatch.")

    numeric_columns = sorted(
        KINEMATIC_COLUMNS | ORDINAL_COLUMNS | ORDINAL_STD_COLUMNS |
        MECHANISM_COLUMNS | WEIGHT_COLUMNS)
    numeric = df[numeric_columns].to_numpy(dtype=np.float64)
    _require(np.isfinite(numeric).all(), "运动学或概率字段含 NaN/Inf。")
    _require(df["munition_id"].isin([0, 1, 2, 3]).all(), "munition_id 必须位于 [0,3]。")

    monotonic_violations = 0
    for task in "KMFC":
        ge1 = df[f"{task}_ge1_prob"].to_numpy(dtype=np.float64)
        ge2 = df[f"{task}_ge2_prob"].to_numpy(dtype=np.float64)
        _require(((0.0 <= ge1) & (ge1 <= 1.0)).all(), f"{task}_ge1_prob 超出 [0,1]。")
        _require(((0.0 <= ge2) & (ge2 <= 1.0)).all(), f"{task}_ge2_prob 超出 [0,1]。")
        monotonic_violations += int((ge2 > ge1 + 1e-12).sum())
    _require(monotonic_violations == 0, f"发现 {monotonic_violations} 个序数概率保序违规。")
    _require((df[list(WEIGHT_COLUMNS)] > 0).all().all(), "样本权重字段必须全部为正。")

    family_profile = profile.get("family_distribution", {})
    configured_cap = int(family_profile.get("maximum_rows_per_root_configured", -1))
    observed_max_family = int(df["root_seed_id"].astype(str).value_counts().max())
    _require(configured_cap > 0, "profile 缺少有效的 root family 上限。")
    _require(observed_max_family <= configured_cap,
             f"单 root 最大 {observed_max_family} 行，超过配置上限 {configured_cap}。")
    gate = profile.get("usability_gate", {})
    _require(not gate.get("enforced") or gate.get("passed"),
             f"训练可用性门禁失败: {gate.get('failures', [])[:5]}")
    exact_level_evidence = _validate_exact_level_evidence(
        df, profile)

    train_weights = df.loc[
        df["split_role"].astype(str) == "train", "loss_weight"
    ].to_numpy(dtype=np.float64)
    train_weight_ess = float(
        train_weights.sum() ** 2 / np.square(train_weights).sum())
    train_weight_ess_ratio = float(
        train_weight_ess / max(len(train_weights), 1))
    minimum_ess_ratio = float(
        profile.get("weighting", {}).get(
            "minimum_effective_sample_size_ratio", 0.0))
    _require(
        minimum_ess_ratio > 0.0,
        "profile 缺少有效的训练权重 ESS 下限。",
    )
    _require(
        train_weight_ess_ratio + 1e-12 >= minimum_ess_ratio,
        f"train loss_weight ESS 比例={train_weight_ess_ratio:.6f} < "
        f"{minimum_ess_ratio:.6f}。",
    )

    k2_contract = profile.get("k2_ratio_contract", {})
    k2_phase2_stop = float(k2_contract.get("phase2_stop_ratio", -1.0))
    k2_final_max = float(k2_contract.get("final_max_ratio", -1.0))
    _require(
        0.0 < k2_phase2_stop <= k2_final_max < 1.0,
        "profile 缺少有效的 K2 比例合同。",
    )
    observed_k2_rows = int((df["K_ge2_prob"] >= 0.5).sum())
    observed_k2_ratio = float(observed_k2_rows / len(df))
    if bool(k2_contract.get("enforced", True)):
        _require(
            observed_k2_ratio <= k2_final_max + 1e-12,
            f"最终 K2 比例={observed_k2_ratio:.6f} > "
            f"{k2_final_max:.6f}。",
        )
    _require(
        int(k2_contract.get("observed_positive_rows", -1)) ==
        observed_k2_rows,
        "profile 的 K2 正例行数与 Parquet 不一致。",
    )
    _require(
        abs(float(k2_contract.get("observed_final_ratio", -1.0)) -
            observed_k2_ratio) <= 1e-12,
        "profile 的 K2 最终比例与 Parquet 不一致。",
    )

    artifact = profile.get("artifact", {})
    actual_hash = None
    if artifact:
        _require(int(artifact.get("rows", -1)) == len(df), "profile 记录的行数与 Parquet 不一致。")
        _require(
            int(artifact.get("size_bytes", -1)) == dataset_path.stat().st_size,
            "profile 记录的文件大小与 Parquet 不一致。",
        )
        if verify_hash:
            expected_hash = str(artifact.get("sha256", ""))
            _require(len(expected_hash) == 64, "profile 缺少有效 SHA-256。")
            actual_hash = _sha256(dataset_path)
            _require(actual_hash == expected_hash, "Parquet SHA-256 校验失败。")
    else:
        raise Stage0ValidationError("profile 缺少 artifact 完整性记录。")

    component_profile_file = dataset_path.with_name(
        COMPONENT_SUPERVISION_PROFILE_FILENAME)
    embedded_component = profile.get(
        "component_supervision")
    component_status = "not_present_optional"
    if embedded_component is not None or component_profile_file.is_file():
        _require(
            component_profile_file.is_file(),
            "generation profile 声明了部件监督，但缺少独立 profile。")
        with component_profile_file.open(
                "r", encoding="utf-8") as stream:
            component_profile = json.load(stream)
        try:
            validate_component_supervision_profile(
                component_profile,
                base_dataset_path=str(dataset_path),
                base_dataset_sha256=str(artifact["sha256"]),
                base_dataset_rows=len(df),
                base_dataset_schema=DATASET_SCHEMA,
                frame_convention=FRAME_CONVENTION_VERSION,
            )
        except RuntimeError as exc:
            raise Stage0ValidationError(str(exc)) from exc
        component_artifact = component_profile.get(
            "artifact", {})
        component_path = dataset_path.with_name(
            str(component_artifact.get("path", "")))
        _require(
            component_path.is_file(),
            f"部件监督 Parquet 不存在: {component_path}")
        _require(
            int(component_artifact.get("rows", -1))
            == len(df),
            "部件监督未完整覆盖主数据集。")
        _require(
            int(component_artifact.get("size_bytes", -1))
            == component_path.stat().st_size,
            "部件监督文件大小与 profile 不一致。")
        if verify_hash:
            _require(
                _sha256(component_path)
                == component_artifact.get("sha256"),
                "部件监督 Parquet SHA-256 校验失败。")
        component_frame = pd.read_parquet(
            component_path, engine="pyarrow",
            columns=["sample_id", *COMPONENT_TARGET_COLUMNS])
        _require(
            list(component_frame.columns)
            == ["sample_id", *COMPONENT_TARGET_COLUMNS],
            "部件监督列名或顺序与合同不一致。")
        _require(
            component_frame["sample_id"].astype(str).tolist()
            == df["sample_id"].astype(str).tolist(),
            "部件监督 sample_id 顺序与主数据集不一致。")
        component_values = component_frame[
            list(COMPONENT_TARGET_COLUMNS)].to_numpy(
                dtype=np.float64)
        _require(
            np.isfinite(component_values).all()
            and np.all(component_values >= 0.0)
            and np.all(component_values <= 1.0),
            "部件监督包含非有限值或超出 [0,1]。")
        order_hash = sha256_text_sequence(
            component_frame["sample_id"].astype(str))
        _require(
            order_hash
            == component_artifact.get(
                "sample_id_order_sha256"),
            "部件监督 sample_id 顺序哈希不一致。")
        if embedded_component is not None:
            _require(
                embedded_component.get("schema")
                == component_profile.get("schema"),
                "主 generation profile 与部件监督 schema 不一致。")
            _require(
                embedded_component.get("artifact")
                == component_profile.get("artifact"),
                "主 generation profile 与部件监督 artifact 不一致。")
        component_status = "sha256_match"

    adjustment_file = dataset_path.with_name("logit_adjustment.json")
    adjustment_status = "not_present_optional"
    if adjustment_file.is_file():
        with adjustment_file.open("r", encoding="utf-8") as stream:
            adjustment = json.load(stream)
        adjustment_hash = adjustment.get("__meta__", {}).get("dataset_sha256")
        expected_hash = str(artifact.get("sha256", ""))
        _require(
            adjustment_hash == expected_hash,
            "logit_adjustment.json 与当前 Parquet SHA-256 不匹配。",
        )
        adjustment_status = "sha256_match"

    split_counts = {
        role: int((df["split_role"].astype(str) == role).sum())
        for role in ("train", "val", "test")
    }
    return {
        "status": "PASS",
        "dataset": str(dataset_path),
        "profile": str(profile_file),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "root_families": int(df["root_seed_id"].nunique()),
        "cross_split_root_families": cross_split_roots,
        "split_counts": split_counts,
        "frame_convention": FRAME_CONVENTION_VERSION,
        "dataset_schema": DATASET_SCHEMA,
        "sha256_verified": bool(verify_hash),
        "ordinal_monotonicity_violations": monotonic_violations,
        "label_mc_replicate_histogram": mc_histogram,
        "label_mc_all_resolved_ratio": resolution_ratio,
        "label_mc_maximum_reached_ratio": maximum_reached_ratio,
        "maximum_rows_per_root": observed_max_family,
        "train_weight_ess_ratio": train_weight_ess_ratio,
        "minimum_weight_ess_ratio": minimum_ess_ratio,
        "k2_positive_rows": observed_k2_rows,
        "k2_final_ratio": observed_k2_ratio,
        "k2_final_max_ratio": k2_final_max,
        "parquet_read_engine": parquet_engine,
        "usability_gate_passed": bool(gate.get("passed", False)),
        "exact_level_evidence": {
            key: value
            for key, value in exact_level_evidence.items()
            if key != "observed"
        },
        "logit_adjustment_status": adjustment_status,
        "component_supervision_status": component_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="校验 Stage-0 仿真数据集门禁。")
    parser.add_argument("parquet_path", help="待校验的 Parquet 文件。")
    parser.add_argument("--profile", help="generation_profile.json 路径；默认取数据集同目录。")
    parser.add_argument("--skip-hash", action="store_true", help="跳过 SHA-256（仅用于超大文件快速诊断）。")
    args = parser.parse_args()
    try:
        report = validate_stage0_dataset(
            args.parquet_path,
            profile_path=args.profile,
            verify_hash=not args.skip_hash,
        )
    except (Stage0ValidationError, FileNotFoundError, ValueError, OSError) as exc:
        print(json.dumps({"status": "FAIL", "reason": str(exc)}, indent=2, ensure_ascii=False))
        sys.exit(2)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
