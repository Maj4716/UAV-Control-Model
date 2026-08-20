from __future__ import annotations

import json
import hashlib
import os
from typing import Dict

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.preprocessing import MinMaxScaler

from loitering_munition_damage_twin.stage0.component_supervision import (
    COMPONENT_SUPERVISION_FILENAME,
    COMPONENT_SUPERVISION_PROFILE_FILENAME,
    COMPONENT_SUPERVISION_SCHEMA,
    COMPONENT_TARGET_COLUMNS,
    CRITICAL_COMPONENT_IDS,
    sha256_file,
    sha256_text_sequence,
    validate_component_supervision_profile,
)
from loitering_munition_damage_twin.surrogate.model import DEFAULT_ORDINAL_APPLICABILITY
from loitering_munition_damage_twin.surrogate.features import (
    COMPONENT_PROXY_FEATURE_COLUMNS,
    TERMINAL_PHYSICS_FEATURE_COLUMNS,
    augment_terminal_physics_features,
    terminal_physics_contract_metadata,
)
from loitering_munition_damage_twin.paths import PROJECT_ROOT

REQUIRED_GENERATION_PROFILE_SCHEMA = "stage0_lineage_v2"
REQUIRED_DATASET_SCHEMA = "stage0_lineage_v2"
REQUIRED_FRAME_VERSION = "stage0_ned_frd_v1"

FEATURE_COLUMNS = [
    "x_cm", "y_cm", "z_cm",        # 空间坐标
    "vx_ms", "vy_ms", "vz_ms",     # 速度分量
    "sin_yaw", "cos_yaw",
    "sin_pitch", "cos_pitch",
    "sin_roll", "cos_roll",
    "norm_velocity",               # 通用物理衍生 1: 速度模长 (能量基底)
]

# Generation-only aim-point metadata.  These columns are permanently excluded
# because the selected internal component is not an observable terminal-state
# variable and leaks the active-sampling policy.
PHYSICS_FEATURE_COLUMNS = {"los_distance", "impact_cosine"}


def _cfg_section(ablation_config: dict | None, name: str) -> dict:
    if not ablation_config:
        return {}
    section = ablation_config.get(name, {})
    return section if isinstance(section, dict) else {}


def get_feature_columns(ablation_config: dict | None = None) -> list[str]:
    """Return the active feature list for a normal run or an ablation."""
    data_cfg = _cfg_section(ablation_config, "data")
    forbidden = set(data_cfg.get("extra_features", [])) & PHYSICS_FEATURE_COLUMNS
    if forbidden:
        raise ValueError(
            "生成阶段瞄准点元数据禁止进入代理模型: "
            f"{sorted(forbidden)}")
    candidate_features = list(FEATURE_COLUMNS)
    use_terminal_physics_features = bool(
        data_cfg.get("use_terminal_physics_features", False))
    use_component_proxy_features = bool(
        data_cfg.get("use_component_proxy_features", False))
    use_armor_aware_fragment_proxies = bool(
        data_cfg.get("use_armor_aware_fragment_proxies", False))
    if use_component_proxy_features and not use_terminal_physics_features:
        raise ValueError(
            "data.use_component_proxy_features requires "
            "data.use_terminal_physics_features=True.")
    if (
        use_armor_aware_fragment_proxies
        and not use_component_proxy_features
    ):
        raise ValueError(
            "data.use_armor_aware_fragment_proxies requires "
            "data.use_component_proxy_features=True.")
    if use_terminal_physics_features:
        candidate_features.extend(TERMINAL_PHYSICS_FEATURE_COLUMNS)
    if use_component_proxy_features:
        candidate_features.extend(COMPONENT_PROXY_FEATURE_COLUMNS)
    drop_features = set(data_cfg.get("drop_features", []))
    unknown = drop_features - set(candidate_features)
    if unknown:
        raise ValueError(f"data.drop_features 含未知特征: {sorted(unknown)}")
    active = [
        column for column in candidate_features
        if column not in drop_features
    ]
    if not active:
        raise ValueError("data.drop_features 不能移除全部模型输入。")
    return active


class DamageDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 mun_ids: np.ndarray, loss_weights: np.ndarray,
                 k_task_weights: np.ndarray,
                 c_task_weights: np.ndarray,
                 m_task_weights: np.ndarray,
                 labels_soft: np.ndarray = None,
                 label_confidence: np.ndarray = None,
                 sample_ids: np.ndarray = None,
                 root_seed_ids: np.ndarray = None,
                 mechanism_targets_soft: np.ndarray = None,
                 component_targets_soft: np.ndarray = None):
        """
        features: numpy array shape (N, F)
        labels: numpy array shape (N, 4, 2)              — 硬 0/1 标签 (用于评估)
        mun_ids: numpy array shape (N,)
        loss_weights: numpy array shape (N,)             — 全任务统一样本权重 (CB × IPW)
        k_task_weights: numpy array shape (N,)           — K 分支专用权重 (m_id 条件化)
        c_task_weights: numpy array shape (N,)           — [P0-2] C 分支专用权重 (m_id 条件化)
        m_task_weights: numpy array shape (N,)           — [P3-M] M 分支专用权重
        labels_soft: numpy array shape (N, 4, 2)         — [R20] 软标签 (M1_prob 等连续概率)
                                                           用于 BCE 训练替代硬标签, 缓解 Small × M
                                                           的 0.5 阈值切割人为噪声; 评估仍用硬 labels.
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.mun_ids = torch.tensor(mun_ids, dtype=torch.long)
        self.loss_weights = torch.tensor(loss_weights, dtype=torch.float32)
        self.k_task_weights = torch.tensor(k_task_weights, dtype=torch.float32)
        self.c_task_weights = torch.tensor(c_task_weights, dtype=torch.float32)
        self.m_task_weights = torch.tensor(m_task_weights, dtype=torch.float32)
        # [R20] 软标签 fallback: 旧调用未传时退化为硬标签 (旧行为)
        if labels_soft is None:
            labels_soft = labels
        self.labels_soft = torch.tensor(labels_soft, dtype=torch.float32)
        if label_confidence is None:
            label_confidence = np.ones_like(labels, dtype=np.float32)
        self.label_confidence = torch.tensor(
            label_confidence, dtype=torch.float32)
        n = len(features)
        self.sample_ids = np.asarray(
            sample_ids if sample_ids is not None
            else [str(i) for i in range(n)], dtype=str)
        self.root_seed_ids = np.asarray(
            root_seed_ids if root_seed_ids is not None
            else self.sample_ids, dtype=str)
        self.mechanism_targets_soft = (
            None
            if mechanism_targets_soft is None
            else torch.tensor(
                mechanism_targets_soft, dtype=torch.float32)
        )
        self.component_targets_soft = (
            None
            if component_targets_soft is None
            else torch.as_tensor(
                component_targets_soft, dtype=torch.float32)
        )

        assert self.features.dtype == torch.float32
        assert self.labels.dtype == torch.float32
        assert self.labels_soft.dtype == torch.float32
        assert self.label_confidence.dtype == torch.float32
        assert not torch.isnan(self.features).any()
        assert not torch.isnan(self.labels).any()
        assert not torch.isnan(self.labels_soft).any()
        assert torch.isfinite(self.label_confidence).all()
        assert ((self.label_confidence > 0.0)
                & (self.label_confidence <= 1.0)).all()
        if self.mechanism_targets_soft is not None:
            assert tuple(self.mechanism_targets_soft.shape) == (
                n, 2, 4, 2)
            assert torch.isfinite(self.mechanism_targets_soft).all()
            assert ((self.mechanism_targets_soft >= 0.0)
                    & (self.mechanism_targets_soft <= 1.0)).all()
            assert torch.all(
                self.mechanism_targets_soft[..., 1]
                <= self.mechanism_targets_soft[..., 0])
        if self.component_targets_soft is not None:
            assert tuple(self.component_targets_soft.shape) == (
                len(self.features), 2,
                len(CRITICAL_COMPONENT_IDS))
            assert torch.isfinite(
                self.component_targets_soft).all()
            assert (
                (self.component_targets_soft >= 0.0)
                & (self.component_targets_soft <= 1.0)
            ).all()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        values = (
            self.features[idx], self.labels[idx],
            self.mun_ids[idx], self.loss_weights[idx],
            self.k_task_weights[idx],
            self.c_task_weights[idx],
            self.m_task_weights[idx],
            self.labels_soft[idx],
            self.label_confidence[idx],
            self.sample_ids[idx],
            self.root_seed_ids[idx],
        )
        if self.mechanism_targets_soft is not None:
            values = values + (self.mechanism_targets_soft[idx],)
        if self.component_targets_soft is not None:
            values = values + (self.component_targets_soft[idx],)
        return values


def _load_generation_profile(parquet_path: str) -> dict:
    profile_path = os.path.join(os.path.dirname(parquet_path), "generation_profile.json")
    if not os.path.exists(profile_path):
        raise FileNotFoundError(
            f"缺少 generation_profile.json：{profile_path}。"
            f"请先用新的按弹型 quota 管线重建数据集。")

    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    schema = profile.get("profile_schema")
    phase2_mode = profile.get("phase2_mode")
    dataset_schema = profile.get("dataset_schema")
    frame_version = profile.get("frame_convention")
    if (schema != REQUIRED_GENERATION_PROFILE_SCHEMA
            or dataset_schema != REQUIRED_DATASET_SCHEMA
            or frame_version != REQUIRED_FRAME_VERSION
            or phase2_mode != "per_munition_topoff"):
        raise RuntimeError(
            f"数据集不满足 Stage-0 门禁: profile_schema={schema!r}, "
            f"dataset_schema={dataset_schema!r}, frame={frame_version!r}, "
            f"phase2_mode={phase2_mode!r}。旧坐标系数据会被明确拒绝；"
            f"请重新运行 generate_dataset.py。")

    artifact = profile.get("artifact", {})
    expected_hash = str(artifact.get("sha256", ""))
    if (int(artifact.get("size_bytes", -1)) != os.path.getsize(parquet_path)
            or len(expected_hash) != 64):
        raise RuntimeError("Stage-0 数据完整性门禁失败：profile artifact 元数据缺失或文件大小不符。")
    digest = hashlib.sha256()
    with open(parquet_path, "rb") as dataset_file:
        for chunk in iter(lambda: dataset_file.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_hash:
        raise RuntimeError("Stage-0 数据完整性门禁失败：Parquet SHA-256 不匹配。")

    gate = profile.get("usability_gate", {})
    if gate.get("enforced") and not gate.get("passed"):
        raise RuntimeError(
            "Stage-0 训练可用性门禁失败: " +
            "; ".join(str(v) for v in gate.get("failures", [])[:8]))

    print(f"[Dataset] generation_profile 已加载: {profile_path} (schema={schema})")
    return profile


def _ordinal_applicability_matrix(profile: dict) -> np.ndarray:
    """Return and validate the Stage-0 applicability contract as (4,4,2)."""
    task_names = ("K", "M", "F", "C")
    raw = profile.get("ordinal_applicability", {})
    matrix = np.zeros((4, 4, 2), dtype=bool)
    for munition_id in range(4):
        munition = raw.get(str(munition_id), {})
        for task_index, task_name in enumerate(task_names):
            flags = munition.get(task_name)
            if not isinstance(flags, list) or len(flags) != 2:
                raise RuntimeError(
                    "generation_profile.ordinal_applicability 缺少 "
                    f"m_id={munition_id}/{task_name} 的两个序数标志。")
            matrix[munition_id, task_index] = np.asarray(flags, dtype=bool)

    expected = np.asarray(DEFAULT_ORDINAL_APPLICABILITY, dtype=bool)
    if not np.array_equal(matrix, expected):
        raise RuntimeError(
            "数据集 ordinal_applicability 与当前代理模型合同不一致；"
            "请显式升级模型合同后再训练。")
    return matrix


def _build_ordinal_targets(df: pd.DataFrame,
                           use_soft_labels: bool = True,
                           use_label_uncertainty: bool = False,
                           uncertainty_scale: float = 0.10,
                           confidence_floor: float = 0.25):
    """Build hard labels, full MC-mean targets and bounded reliability weights.

    Unlike the historical one-sided clamp, probabilities below 0.5 remain
    probabilities.  BCE against the complete MC mean is a proper scoring rule.
    The confidence term only reduces the contribution of labels whose MC mean
    remains noisy; it never changes the target itself.
    """
    if uncertainty_scale <= 0.0:
        raise ValueError("label_uncertainty_scale must be positive.")
    if not (0.0 < confidence_floor <= 1.0):
        raise ValueError("label_confidence_floor must be in (0,1].")

    tasks = ("K", "M", "F", "C")
    y = np.zeros((len(df), 4, 2), dtype=np.float32)
    y_soft = np.zeros_like(y)
    confidence = np.ones_like(y)
    replicates = df["label_mc_replicates"].to_numpy(dtype=np.float32)
    replicates = np.maximum(replicates, 1.0)

    for task_index, task_name in enumerate(tasks):
        levels = df[f"{task_name}_level"].to_numpy(dtype=np.int64)
        for level_index, level in enumerate((1, 2)):
            probability_column = f"{task_name}_ge{level}_prob"
            std_column = f"{task_name}_ge{level}_prob_std"
            y[:, task_index, level_index] = (levels >= level).astype(
                np.float32)
            y_soft[:, task_index, level_index] = df[
                probability_column].to_numpy(dtype=np.float32)
            if use_label_uncertainty:
                mc_standard_error_column = (
                    f"{task_name}_ge{level}_mc_standard_error")
                if mc_standard_error_column in df.columns:
                    standard_error = df[
                        mc_standard_error_column
                    ].to_numpy(dtype=np.float32)
                else:
                    standard_error = (
                        df[std_column].to_numpy(dtype=np.float32)
                        / np.sqrt(replicates)
                    )
                reliability = 1.0 / (
                    1.0 + np.square(standard_error / uncertainty_scale))
                mc_resolved_column = (
                    f"{task_name}_ge{level}_mc_resolved")
                if mc_resolved_column in df.columns:
                    resolved = df[
                        mc_resolved_column
                    ].to_numpy(dtype=bool)
                    # An estimate that exhausted the MC budget without
                    # resolving its confidence interval is not equivalent to
                    # a precise hard label. Retain its soft probability while
                    # capping its loss contribution at the configured floor.
                    reliability = np.where(
                        resolved,
                        reliability,
                        np.minimum(reliability, confidence_floor),
                    )
                confidence[:, task_index, level_index] = np.clip(
                    reliability, confidence_floor, 1.0)

    y_soft[:, :, 1] = np.minimum(y_soft[:, :, 1], y_soft[:, :, 0])
    y_soft = np.clip(y_soft, 0.0, 1.0).astype(np.float32)
    if not use_soft_labels:
        y_soft = y.copy()
    if not use_label_uncertainty:
        confidence.fill(1.0)
    return y, y_soft, confidence.astype(np.float32)


def _build_mechanism_targets(df: pd.DataFrame) -> np.ndarray:
    """Build fragment/shock MC-mean ordinal targets as (N,2,4,2).

    These are simulator outputs already present in the Stage-0 Parquet.  They
    are auxiliary supervision only: no realized hit, penetration or component
    damage is exposed to the model input.
    """
    mechanisms = ("fragment", "shock")
    tasks = ("K", "M", "F", "C")
    columns = [
        f"{mechanism}_{task}_ge{level}_prob"
        for mechanism in mechanisms
        for task in tasks
        for level in (1, 2)
    ]
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise RuntimeError(
            "分机制监督已启用，但 Stage-0 数据集缺少目标列: "
            f"{missing}")

    targets = np.zeros((len(df), 2, 4, 2), dtype=np.float32)
    for mechanism_index, mechanism in enumerate(mechanisms):
        for task_index, task in enumerate(tasks):
            level1 = df[
                f"{mechanism}_{task}_ge1_prob"].to_numpy(
                    dtype=np.float32)
            level2 = df[
                f"{mechanism}_{task}_ge2_prob"].to_numpy(
                    dtype=np.float32)
            if (
                not np.isfinite(level1).all()
                or not np.isfinite(level2).all()
                or np.any((level1 < 0.0) | (level1 > 1.0))
                or np.any((level2 < 0.0) | (level2 > 1.0))
            ):
                raise RuntimeError(
                    "分机制概率目标包含非有限值或超出 [0,1]: "
                    f"{mechanism}/{task}")
            if np.any(level2 > level1 + 1e-6):
                raise RuntimeError(
                    "分机制概率目标违反序数单调性: "
                    f"{mechanism}/{task}")
            targets[:, mechanism_index, task_index, 0] = level1
            targets[:, mechanism_index, task_index, 1] = np.minimum(
                level2, level1)
    return targets


def _load_component_targets(
        df: pd.DataFrame,
        parquet_path: str,
        generation_profile: dict,
        data_cfg: dict) -> tuple[np.ndarray, dict]:
    """Load a SHA-bound label sidecar without exposing it as model input."""
    configured_path = data_cfg.get("component_supervision_path")
    if configured_path:
        sidecar_path = str(configured_path)
        if not os.path.isabs(sidecar_path):
            sidecar_path = str((PROJECT_ROOT / sidecar_path).resolve())
    else:
        sidecar_path = os.path.join(
            os.path.dirname(os.path.abspath(parquet_path)),
            COMPONENT_SUPERVISION_FILENAME,
        )
    configured_profile = data_cfg.get(
        "component_supervision_profile_path")
    if configured_profile:
        profile_path = str(configured_profile)
        if not os.path.isabs(profile_path):
            profile_path = str((PROJECT_ROOT / profile_path).resolve())
    else:
        profile_path = os.path.join(
            os.path.dirname(sidecar_path),
            COMPONENT_SUPERVISION_PROFILE_FILENAME,
        )
    if not os.path.isfile(sidecar_path):
        raise FileNotFoundError(
            "部件级监督已启用，但缺少旁路 Parquet: "
            f"{sidecar_path}。请运行 build_component_supervision.py。")
    if not os.path.isfile(profile_path):
        raise FileNotFoundError(
            "部件级监督已启用，但缺少旁路 profile: "
            f"{profile_path}。")
    with open(profile_path, "r", encoding="utf-8") as stream:
        component_profile = json.load(stream)
    validate_component_supervision_profile(
        component_profile,
        base_dataset_path=parquet_path,
        base_dataset_sha256=str(
            generation_profile["artifact"]["sha256"]),
        base_dataset_rows=len(df),
        base_dataset_schema=REQUIRED_DATASET_SCHEMA,
        frame_convention=REQUIRED_FRAME_VERSION,
    )
    artifact = component_profile.get("artifact", {})
    if artifact.get("path") != os.path.basename(sidecar_path):
        raise RuntimeError(
            "部件级监督 profile 的 artifact.path 与请求文件不一致。")
    if int(artifact.get("rows", -1)) != len(df):
        raise RuntimeError(
            "部件级监督必须覆盖主数据集的全部 sample_id。")
    if int(artifact.get("columns", -1)) != (
            1 + len(COMPONENT_TARGET_COLUMNS)):
        raise RuntimeError(
            "部件级监督 profile 的列数不符合合同。")
    if (
        int(artifact.get("size_bytes", -1))
        != os.path.getsize(sidecar_path)
        or artifact.get("sha256") != sha256_file(sidecar_path)
    ):
        raise RuntimeError(
            "部件级监督旁路文件大小或 SHA-256 与 profile 不一致。")

    sidecar = pd.read_parquet(
        sidecar_path, engine="pyarrow",
        columns=["sample_id", *COMPONENT_TARGET_COLUMNS])
    if list(sidecar.columns) != [
            "sample_id", *COMPONENT_TARGET_COLUMNS]:
        raise RuntimeError(
            "部件级监督旁路列名或顺序与合同不一致。")
    base_sample_ids = df["sample_id"].astype(str).to_numpy()
    sidecar_sample_ids = sidecar["sample_id"].astype(str).to_numpy()
    if (
        len(sidecar_sample_ids) != len(base_sample_ids)
        or not np.array_equal(
            sidecar_sample_ids, base_sample_ids)
    ):
        raise RuntimeError(
            "部件级监督 sample_id 顺序与主数据集不完全一致；"
            "拒绝可能的错位标签。")
    order_hash = sha256_text_sequence(sidecar_sample_ids)
    if (
        artifact.get("sample_id_order_sha256") != order_hash
        or component_profile.get("base_dataset", {}).get(
            "sample_id_order_sha256") != order_hash
    ):
        raise RuntimeError(
            "部件级监督 sample_id 顺序哈希与 profile 不一致。")

    flat = sidecar[list(COMPONENT_TARGET_COLUMNS)].to_numpy(
        dtype=np.float32)
    if (
        not np.isfinite(flat).all()
        or np.any(flat < 0.0)
        or np.any(flat > 1.0)
    ):
        raise RuntimeError(
            "部件级监督包含非有限值或超出 [0,1] 的概率。")
    targets = flat.reshape(
        len(df), 2, len(CRITICAL_COMPONENT_IDS))
    contract = {
        "schema": COMPONENT_SUPERVISION_SCHEMA,
        "path": os.path.abspath(sidecar_path),
        "profile_path": os.path.abspath(profile_path),
        "sha256": str(artifact["sha256"]),
        "rows": int(len(sidecar)),
        "component_ids": list(CRITICAL_COMPONENT_IDS),
        "mechanisms": ["fragment", "shock"],
        "target_columns": list(COMPONENT_TARGET_COLUMNS),
        "sample_id_order_sha256": order_hash,
        "model_input_allowed": False,
    }
    return targets.astype(np.float32, copy=False), contract


def _ordinal_levels(y_arr: np.ndarray, task_idx: int) -> np.ndarray:
    return y_arr[:, task_idx, 0].astype(np.int64) + y_arr[:, task_idx, 1].astype(np.int64)


def _count_cell_level(y_arr: np.ndarray, mun_arr: np.ndarray,
                      mun_id: int, task_idx: int, level: int) -> int:
    levels = _ordinal_levels(y_arr, task_idx)
    return int(((mun_arr == mun_id) & (levels == level)).sum())


def _print_split_level_grid(split_name: str, y_arr: np.ndarray, mun_arr: np.ndarray) -> None:
    mun_names = ["Small", "Med-LM", "Med-RD", "Heavy"]
    task_names = ["K", "M", "F", "C"]
    print(f"[Dataset] [V5 Stats] {split_name} task×munition level counts:")
    for task_idx, task_name in enumerate(task_names):
        levels = _ordinal_levels(y_arr, task_idx)
        parts = []
        for mun_id, mun_name in enumerate(mun_names):
            counts = np.bincount(levels[mun_arr == mun_id], minlength=3)[:3].astype(int).tolist()
            parts.append(f"{mun_name}=L0/L1/L2{counts}")
        print(f"  {task_name}: " + " | ".join(parts))


def _validate_dataset_usability(profile: dict,
                                y_val: np.ndarray, mun_val: np.ndarray,
                                y_test: np.ndarray = None,
                                mun_test: np.ndarray = None) -> None:
    """Enforce train-family diversity; report natural holdout scarcity only.

    Exact count floors in validation/test forced active-sampling descendants into
    holdouts and made natural prevalence impossible to measure.  The v2 profile
    gates applicable training heads by independent root families.  Holdouts stay
    root-independent and naturally sampled; zero-positive applicable cells are a
    warning that should be covered by a separately generated challenge set.
    """
    gate = profile.get("usability_gate", {})
    if gate.get("enforced") and not gate.get("passed"):
        raise RuntimeError(
            "Stage-0 v2 可用性门禁失败: " +
            "; ".join(str(v) for v in gate.get("failures", [])[:8]))

    applicability = profile.get("ordinal_applicability", {})
    task_names = ("K", "M", "F", "C")
    warnings = []
    holdouts = [("Val", y_val, mun_val)]
    if y_test is not None and mun_test is not None:
        holdouts.append(("Test", y_test, mun_test))
    for split_name, y_split, mun_split in holdouts:
        for mun_id in range(4):
            for task_idx, task_name in enumerate(task_names):
                flags = applicability.get(str(mun_id), {}).get(task_name, [True, True])
                for level_idx in range(2):
                    if not bool(flags[level_idx]):
                        continue
                    count = int(((mun_split == mun_id) &
                                 (y_split[:, task_idx, level_idx] == 1)).sum())
                    if count == 0:
                        warnings.append(
                            f"{split_name} m_id={mun_id} {task_name}>={level_idx + 1} 无正例")
    print("[Dataset] [V2 Gate] 训练集独立 root 可用性: "
          f"{'PASS' if gate.get('passed') else '仅诊断（小规模数据）'}")
    if warnings:
        print("[Dataset] WARNING: 自然 holdout 存在稀有空格；不污染切分，"
              "请用独立 challenge set 补充评估: " + "; ".join(warnings[:12]))


def _split_manifest_path(parquet_path: str) -> str:
    return os.path.join(os.path.dirname(parquet_path), "split_manifest.json")


def _dataset_file_signature(parquet_path: str) -> dict:
    return {
        "rows": None,
        "size_bytes": int(os.path.getsize(parquet_path)),
        "mtime_ns": int(os.stat(parquet_path).st_mtime_ns),
    }


def _load_or_create_split_indices(parquet_path: str, X: np.ndarray, y: np.ndarray,
                                  groups: np.ndarray, random_state: int,
                                  split_roles: np.ndarray = None,
                                  root_seed_ids: np.ndarray = None):
    manifest_path = _split_manifest_path(parquet_path)
    current_sig = _dataset_file_signature(parquet_path)
    current_sig["rows"] = int(len(X))

    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if (
            manifest.get("rows") == current_sig["rows"]
            and manifest.get("size_bytes") == current_sig["size_bytes"]
            and manifest.get("mtime_ns") == current_sig["mtime_ns"]
            and manifest.get("split_strategy") == "preassigned_root_seed_v1"
        ):
            print(f"[Split] 复用已存在的 split_manifest: {manifest_path}")
            return (
                np.asarray(manifest["train_idx"], dtype=np.int64),
                np.asarray(manifest["val_idx"], dtype=np.int64),
                np.asarray(manifest["test_idx"], dtype=np.int64),
            )
        print(f"[Split] 检测到数据文件已变化，重建 split_manifest: {manifest_path}")

    if split_roles is None or root_seed_ids is None:
        raise RuntimeError(
            "Stage-0 数据集必须包含 split_role 与 root_seed_id；"
            "禁止退回空间桶近似切分。")

    roles = np.asarray(split_roles).astype(str)
    roots = np.asarray(root_seed_ids).astype(str)
    allowed = {"train", "val", "test"}
    unexpected = set(np.unique(roles)) - allowed
    if unexpected:
        raise RuntimeError(f"split_role 含非法值: {sorted(unexpected)}")
    train_idx = np.where(roles == "train")[0]
    val_idx = np.where(roles == "val")[0]
    test_idx = np.where(roles == "test")[0]
    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise RuntimeError(
            f"预分配 split 为空: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    root_sets = {
        "train": set(roots[train_idx].tolist()),
        "val": set(roots[val_idx].tolist()),
        "test": set(roots[test_idx].tolist()),
    }
    overlaps = {
        "train_val": len(root_sets["train"] & root_sets["val"]),
        "train_test": len(root_sets["train"] & root_sets["test"]),
        "val_test": len(root_sets["val"] & root_sets["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"root_seed_id 跨 split 泄漏: {overlaps}")

    manifest = {
        "rows": current_sig["rows"],
        "size_bytes": current_sig["size_bytes"],
        "mtime_ns": current_sig["mtime_ns"],
        "random_state": int(random_state),
        "split_strategy": "preassigned_root_seed_v1",
        "root_overlap": overlaps,
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    print(f"[Split] 已写出 split_manifest: {manifest_path}")
    return train_idx, val_idx, test_idx


class BalancedMunitionBatchSampler(Sampler[list[int]]):
    """Yield equal-munition batches without repeating a row in one epoch.

    Historical sampling used ``replace=True`` and exposed only about 63% of the
    training pool per epoch.  Here each munition receives a weighted random
    ordering without replacement; at most the incomplete tail is discarded.
    """

    def __init__(self, mun_ids: np.ndarray, sampler_weights: np.ndarray,
                 batch_size: int, num_munitions: int = 4, random_state: int = 42):
        if batch_size % num_munitions != 0:
            raise ValueError(
                f"BalancedMunitionBatchSampler 要求 batch_size 可被 {num_munitions} 整除，"
                f"当前收到 {batch_size}")

        self.batch_size = int(batch_size)
        self.num_munitions = int(num_munitions)
        self.per_mun_batch = self.batch_size // self.num_munitions
        self.base_seed = int(random_state)
        self._epoch = 0
        self.indices_by_mun = {}
        self.weights_by_mun = {}

        for m_id in range(self.num_munitions):
            idx = np.where(mun_ids == m_id)[0]
            if len(idx) < self.per_mun_batch:
                raise ValueError(f"弹型 {m_id} 在训练集中没有样本，无法做等曝光 batch 采样。")

            weights = sampler_weights[idx].astype(np.float64)
            if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
                weights = np.ones(len(idx), dtype=np.float64)
            self.indices_by_mun[m_id] = idx
            self.weights_by_mun[m_id] = np.maximum(weights, 1e-12)

        minimum_pool = min(len(v) for v in self.indices_by_mun.values())
        self.num_batches = int(minimum_pool // self.per_mun_batch)
        if self.num_batches <= 0:
            raise ValueError("训练样本不足以构造一个四弹型等量 batch。")

    def __len__(self):
        return self.num_batches

    @property
    def expected_draws_per_munition(self) -> Dict[int, int]:
        return {m_id: self.per_mun_batch * self.num_batches for m_id in range(self.num_munitions)}

    def __iter__(self):
        rng = np.random.default_rng(self.base_seed + self._epoch)
        self._epoch += 1
        ordered_by_mun = {}
        for m_id in range(self.num_munitions):
            indices = self.indices_by_mun[m_id]
            weights = self.weights_by_mun[m_id]
            # Exponential-race ranking is a weighted permutation: large weights
            # are seen earlier, but no physical row is duplicated in the epoch.
            keys = -np.log(np.maximum(rng.random(len(indices)), 1e-15)) / weights
            ordered_by_mun[m_id] = indices[np.argsort(keys, kind="stable")]

        for batch_index in range(self.num_batches):
            batch = []
            for m_id in range(self.num_munitions):
                start = batch_index * self.per_mun_batch
                stop = start + self.per_mun_batch
                chosen = ordered_by_mun[m_id][start:stop]
                batch.extend(chosen.tolist())
            rng.shuffle(batch)
            yield batch


def _build_group_keys(df: pd.DataFrame) -> np.ndarray:
    """[P1 #7] 为 GroupShuffleSplit 构造组键，防止 P2 爬行近邻簇泄漏到 val/test。

    策略：
      - 非爬行样本 (is_crawled=0)  → 每个样本独占一个组 (正整数)
      - 爬行样本   (is_crawled=1)  → 按粗糙位置/速度桶聚合 (相邻爬行点同组)

    桶大小：x/y/z 50 cm × vx/vy/vz 20 m/s。同桶的爬行点会被强制分到同一 split。
    这是种子级真分组的高保真近似——理想方案是 generate_dataset 输出 seed_id；
    在没有该字段时，空间桶哈希足以拦截 P2 高斯抖动 (σ ≤ 30cm) 产生的近邻簇。
    """
    if "root_seed_id" in df.columns:
        roots = df["root_seed_id"].astype(str)
        if (roots == "").any() or roots.isna().any():
            raise RuntimeError("root_seed_id 存在空值，无法执行无泄漏切分。")
        codes, uniques = pd.factorize(roots, sort=True)
        print(f"[Split] Stage-0 谱系分组: {len(df)} 行 → {len(uniques)} 个 root_seed 家族")
        return codes.astype(np.int64)

    n = len(df)
    keys = np.arange(n, dtype=np.int64) + 1  # 非爬行：每行唯一组 (从 1 起，避开 0)

    if "is_crawled" not in df.columns:
        return keys

    is_crawled = df["is_crawled"].values.astype(bool)
    if not is_crawled.any():
        return keys

    crawled_idx = np.where(is_crawled)[0]
    cx = df["x"].values[crawled_idx]
    cy = df["y"].values[crawled_idx]
    cz = df["z"].values[crawled_idx]
    cvx = df["vx"].values[crawled_idx]
    cvy = df["vy"].values[crawled_idx]
    cvz = df["vz"].values[crawled_idx]

    bx = np.floor(cx / 50.0).astype(np.int64)
    by = np.floor(cy / 50.0).astype(np.int64)
    bz = np.floor(cz / 50.0).astype(np.int64)
    bvx = np.floor(cvx / 20.0).astype(np.int64)
    bvy = np.floor(cvy / 20.0).astype(np.int64)
    bvz = np.floor(cvz / 20.0).astype(np.int64)

    # 6D 桶坐标 → 单整数哈希 (大质数异或，碰撞极少)
    h = ((bx * 73856093) ^ (by * 19349663) ^ (bz * 83492791) ^
         (bvx * 49979687) ^ (bvy * 86028157) ^ (bvz * 51234517))
    # 取负值：与非爬行样本的正整数键不冲突
    keys[crawled_idx] = -(np.abs(h) + 1)

    n_crawl_groups = len(np.unique(keys[crawled_idx]))
    print(f"[Split] 爬行点 {is_crawled.sum()} 枚 → 聚合为 {n_crawl_groups} 个空间桶组 "
          f"(平均每组 {is_crawled.sum()/max(n_crawl_groups,1):.2f} 枚)")
    return keys


def _compute_pos_weight(y_train: np.ndarray, cap: float = 100.0) -> np.ndarray:
    """[P1 #9] 基于训练集 (4 任务 × 2 等级) 标签计算 BCE 的 pos_weight = n_neg / n_pos

    cap：上限钳制，避免极稀疏类 (n_pos < 10) 把权重拉到 1e4 量级
    """
    pw = np.zeros((4, 2), dtype=np.float32)
    for i in range(4):
        for j in range(2):
            n_pos = float((y_train[:, i, j] == 1).sum())
            n_neg = float((y_train[:, i, j] == 0).sum())
            pw[i, j] = 1.0 if n_pos == 0 else min(n_neg / n_pos, cap)
    return pw


def _compute_per_mun_pos_weight(y_train: np.ndarray, mun_train: np.ndarray,
                                num_munitions: int = 4,
                                hi_cap: float = 3.0,
                                lo_cap: float = 1.0 / 3.0) -> np.ndarray:
    """[R16 + R17] 按 (munition_id, task, level) 三维网格统计 BCE pos_weight.

    R17 改动: 改用 sqrt-softened 公式并对称钳制到 [1/3, 3.0].
        pw = clip( sqrt(n_neg / n_pos),  1/3, 3.0 )

    R16 第一版直接用了 n_neg/n_pos 全比例 (Small × M_L1 ≈ 4.0, Small × M_L2 ≈ 9.0).
    与 user 既有的 class1_alpha=1.5 + m_task_weight Small=1.20 + BalancedMunition
    BatchSampler 的 class-1 重采样 (~2×) 乘性叠加后, Small × M=1 正样本的有效
    梯度被放大到 ~14×, 远超 Heavy × M=2 等"自然多数"的 ~1×; 导致模型对 Small 的
    M-positive 过度自信、对 M-negative 信号不足, Small × M=0 边界样本被错推到
    L=1, Small × M 3-class 准确率卡在 89-90%, 阈值搜索被迫把 thr1 拉到 0.62
    才能勉强维持 ——这是模型自身概率分布塌陷在阈值附近、纯靠阈值已经救不动的
    典型迹象.

    sqrt 软化后:
        Small × M_L1: ratio=5.7 → sqrt≈2.39 (保留 ~2.4× 正样本提升, 不再 4×)
        Heavy × M_L1: ratio=0.05 → sqrt≈0.22 → clamp 至 1/3 (放宽下限避免梯度消失)
    既保留了"按真实先验分弹型校准"的核心收益, 又把乘性叠加的暴击效应削平,
    让模型有机会学到 L=0 vs L=1 的真实概率分隔.

    上下限 [1/3, 3.0] 是对称钳制 (log-space 对称), 防止极稀疏 / 极稠密单元
    把 BCE 信号推向数值不稳定区.

    Args:
        y_train:  (N, 4, 2) 二值阈限标签
        mun_train: (N,) munition_id ∈ [0, num_munitions)
        hi_cap:   上限 (默认 3.0); 对应 ratio ≤ 9 时不饱和
        lo_cap:   下限 (默认 1/3 ≈ 0.33); 防止极稠密正样本梯度消失
    Returns:
        pw_per_mun: (num_munitions, 4, 2) numpy float32
    """
    pw = np.ones((num_munitions, 4, 2), dtype=np.float32)
    for m in range(num_munitions):
        mask = (mun_train == m)
        if mask.sum() == 0:
            continue  # 该弹型无样本, 保持 1.0 占位
        y_m = y_train[mask]
        for i in range(4):
            for j in range(2):
                n_pos = float((y_m[:, i, j] == 1).sum())
                n_neg = float((y_m[:, i, j] == 0).sum())
                if n_pos == 0:
                    pw[m, i, j] = 1.0
                    continue
                ratio = n_neg / max(n_pos, 1.0)
                # sqrt 软化 + 对称钳制: 保留方向性但削平乘性叠加暴击
                pw[m, i, j] = float(np.clip(np.sqrt(ratio), lo_cap, hi_cap))
    return pw


def _compute_adaptive_train_balance(y_train: np.ndarray,
                                    mun_train: np.ndarray):
    """Build train-split-driven balancing for sampler and loss weights.

    Goals:
      1. Mildly rebalance total munition exposure.
      2. Increase visibility of scarce class-1 tasks, especially K1/M1/C1.
      3. Further lift rare (munition, task, class=1) cells without hard-coding
         only a few specific combinations.

    Returns:
      sampler_weights: (N,) float64, mean-normalized for WeightedRandomSampler
      loss_balance:    (N,) float32, mean-normalized multiplier for train loss_weight
      diag:            dict with human-readable statistics
    """
    task_names = ["K", "M", "F", "C"]
    mun_names = ["Small", "Med-LM", "Med-RD", "Heavy"]

    # Ordinal 0/1/2 labels reconstructed from the two ordinal heads.
    true_level = y_train[:, :, 0].astype(np.int64) + y_train[:, :, 1].astype(np.int64)
    n = len(y_train)

    # ------------------------------------------------------------------
    # 1) Munition-level factor is intentionally neutralized.
    #    四种弹型的总曝光平衡由 BalancedMunitionBatchSampler 强约束实现；
    #    这里不再用 count-based sampler / loss factor 轻微推拉 munition 暴露。
    # ------------------------------------------------------------------
    mun_counts = np.bincount(mun_train, minlength=4).astype(np.float64)
    mun_sampler_factor = np.ones(4, dtype=np.float64)
    mun_loss_factor = np.ones(4, dtype=np.float64)

    # ------------------------------------------------------------------
    # 2) Task-level class-1 scarcity: K/M/C are the current weak branches,
    #    but we still compute all 4 tasks from train statistics.
    # ------------------------------------------------------------------
    class1_counts = np.array(
        [(true_level[:, i] == 1).sum() for i in range(4)],
        dtype=np.float64,
    )
    nonzero_task = class1_counts[class1_counts > 0]
    task_target = float(nonzero_task.mean()) if nonzero_task.size > 0 else 1.0
    task_priority = np.array([1.10, 1.05, 1.00, 1.05], dtype=np.float64)
    task_class1_factor = np.ones(4, dtype=np.float64)
    for i in range(4):
        if class1_counts[i] > 0:
            scarcity = (task_target / class1_counts[i]) ** 0.30
            task_class1_factor[i] = np.clip(scarcity * task_priority[i], 1.0, 2.00)

    # ------------------------------------------------------------------
    # 3) Cell-level class-1 scarcity: each (task, munition) cell gets its own
    #    boost derived from the train split rather than fixed manual rules.
    # ------------------------------------------------------------------
    cell_class1_counts = np.zeros((4, 4), dtype=np.float64)
    cell_class1_factor = np.ones((4, 4), dtype=np.float64)
    # [R18] Small × M cell 单独把 sampler class-1 重采样上限压低到 1.20:
    #   per-mun pos_weight (R16, sqrt 软化后 ~2.0×) 与 class1_alpha=1.5 已经在
    #   BCE 内部给 Small × M=1 充足的正样本梯度. sampler 这一层再 2.40× 重采样
    #   会把 Small × M=1 的有效 per-epoch 梯度推到 ~7×, 而 Small × M=0 仍 1×;
    #   模型被迫把决策面整体右移 → L=0 边界样本被错推到 L=1, Small × M 3-class
    #   准确率卡在 89-90%. 把这一格 sampler 上限单独压到 1.20 (其它格保持 2.40,
    #   特别是 Heavy × M 仍需要 sampler 重采样补偿其 L=1 极稀缺).
    M_TASK_IDX = 1
    SMALL_MUN_ID = 0
    for i in range(4):
        row_counts = np.array(
            [((true_level[:, i] == 1) & (mun_train == m)).sum() for m in range(4)],
            dtype=np.float64,
        )
        cell_class1_counts[i] = row_counts
        nonzero_cells = row_counts[row_counts > 0]
        if nonzero_cells.size == 0:
            continue
        row_target = float(nonzero_cells.mean())
        for m in range(4):
            if row_counts[m] > 0:
                scarcity = (row_target / row_counts[m]) ** 0.45
                # 仅 Small × M cell 单独压低; 其它格维持 2.40
                clamp_hi = 1.20 if (i == M_TASK_IDX and m == SMALL_MUN_ID) else 2.40
                cell_class1_factor[i, m] = np.clip(scarcity, 1.0, clamp_hi)

    # ------------------------------------------------------------------
    # 4) Per-sample factor composition: for a sample that is class-1 on any
    #    task, use the strongest matching task/cell boost to avoid explosions.
    # ------------------------------------------------------------------
    rare_factor = np.ones(n, dtype=np.float64)
    for i in range(4):
        is_class1 = (true_level[:, i] == 1)
        if not np.any(is_class1):
            continue
        for m in range(4):
            mask = is_class1 & (mun_train == m)
            if not np.any(mask):
                continue
            boost = max(task_class1_factor[i], cell_class1_factor[i, m])
            rare_factor[mask] = np.maximum(rare_factor[mask], boost)

    # [R18] 移除前 pos_weight 时代的 Small × M=1 surgical 1.35× boost.
    #   该 boost 是在 (munition, task, level) per-cell pos_weight 引入之前
    #   作为定向补偿加上去的; 现在 BCE 内已有 sqrt-softened pos_weight (~2×) +
    #   class1_alpha (1.5×) 双重照顾 Small × M=1 正样本梯度, 再叠 sampler-side
    #   1.35 boost 是重复加权, 只会把模型推得更过自信. 显式删除此通路.

    sampler_weights = mun_sampler_factor[mun_train] * rare_factor
    sampler_weights = np.clip(sampler_weights, 0.50, 3.00)
    sampler_weights = sampler_weights / max(float(sampler_weights.mean()), 1e-8)

    loss_balance = mun_loss_factor[mun_train] * np.power(rare_factor, 0.35)
    loss_balance = np.clip(loss_balance, 0.85, 1.60)
    loss_balance = loss_balance / max(float(loss_balance.mean()), 1e-8)

    diag = {
        "task_names": task_names,
        "mun_names": mun_names,
        "mun_counts": mun_counts.astype(int),
        "mun_sampler_factor": mun_sampler_factor,
        "mun_loss_factor": mun_loss_factor,
        "class1_counts": class1_counts.astype(int),
        "task_class1_factor": task_class1_factor,
        "cell_class1_counts": cell_class1_counts.astype(int),
        "cell_class1_factor": cell_class1_factor,
    }
    return sampler_weights.astype(np.float64), loss_balance.astype(np.float32), diag


def _task_weight_vector(data_cfg: dict, key: str) -> np.ndarray:
    raw = data_cfg.get(key, [1.0, 1.0, 1.0, 1.0])
    values = np.asarray(raw, dtype=np.float32)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError(f"data.{key} must contain four finite numbers.")
    if (values <= 0.0).any():
        raise ValueError(f"data.{key} values must be positive.")
    return values


def _task_munition_matrix(config: dict, key: str,
                          default: float = 1.0,
                          minimum: float = 0.0,
                          maximum: float | None = None) -> np.ndarray:
    raw = config.get(
        key, np.full((4, 4), default, dtype=np.float32).tolist())
    values = np.asarray(raw, dtype=np.float32)
    if values.shape != (4, 4) or not np.isfinite(values).all():
        raise ValueError(
            f"data.{key} must be a finite 4x4 matrix "
            "(rows=K/M/F/C, columns=Small/Med-LM/Med-RD/Heavy).")
    if (values < minimum).any():
        raise ValueError(
            f"data.{key} values must be >= {minimum}.")
    if maximum is not None and (values > maximum).any():
        raise ValueError(
            f"data.{key} values must be <= {maximum}.")
    return values


def _apply_label_confidence_strength(
        confidence: np.ndarray,
        munition_ids: np.ndarray,
        strength: np.ndarray) -> np.ndarray:
    if confidence.ndim != 3 or confidence.shape[1:] != (4, 2):
        raise ValueError(
            "confidence must have shape (N, 4, 2).")
    if munition_ids.shape != (confidence.shape[0],):
        raise ValueError(
            "munition_ids must have shape (N,).")
    if strength.shape != (4, 4):
        raise ValueError("strength must have shape (4, 4).")
    adjusted = confidence.astype(np.float32, copy=True)
    for task_index in range(4):
        strength_for_rows = strength[
            task_index, munition_ids][:, None]
        adjusted[:, task_index, :] = (
            1.0
            - strength_for_rows
            * (1.0 - adjusted[:, task_index, :])
        )
    return adjusted


def get_dataloaders(parquet_path: str, batch_size: int = 256,
                    random_state: int = 42,
                    persist_scaler: bool = True,
                    ablation_config: dict | None = None,
                    scaler_dir: str = "./output/models",
                    scaler_override=None,
                    load_test_split: bool = True):
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Cannot find dataset at {parquet_path}")

    data_cfg = _cfg_section(ablation_config, "data")
    feature_columns = get_feature_columns(ablation_config)
    use_terminal_physics_features = bool(
        data_cfg.get("use_terminal_physics_features", False))
    use_component_proxy_features = bool(
        data_cfg.get("use_component_proxy_features", False))
    use_armor_aware_fragment_proxies = bool(
        data_cfg.get("use_armor_aware_fragment_proxies", False))
    use_mechanism_supervision = bool(
        data_cfg.get("use_mechanism_supervision", False))
    use_component_supervision = bool(
        data_cfg.get("use_component_supervision", False))
    if len(feature_columns) != len(FEATURE_COLUMNS):
        print(
            "[Dataset][Ablation] active feature count="
            f"{len(feature_columns)} "
            f"(terminal_physics={use_terminal_physics_features})")

    profile = _load_generation_profile(parquet_path)
    ordinal_applicability = _ordinal_applicability_matrix(profile)
    try:
        df = pd.read_parquet(parquet_path, engine="pyarrow")
        parquet_engine = "pyarrow"
    except (OSError, ValueError, TypeError) as pyarrow_error:
        try:
            df = pd.read_parquet(parquet_path, engine="fastparquet")
            parquet_engine = "fastparquet_fallback"
            print(f"[Dataset] pyarrow 回读不兼容，已由 fastparquet 完整回读: {pyarrow_error}")
        except (ImportError, OSError, ValueError, TypeError) as fallback_error:
            raise RuntimeError(
                f"Parquet 回读失败；pyarrow={pyarrow_error}; "
                f"fastparquet={fallback_error}") from fallback_error
    print(f"[Dataset] Parquet read engine: {parquet_engine}")
    if len(df) != int(profile.get("artifact", {}).get("rows", -1)):
        raise RuntimeError("Stage-0 数据完整性门禁失败：Parquet 行数与 profile artifact 不一致。")
    if profile.get("target_total") is not None and len(df) != int(profile["target_total"]):
        print(f"[Dataset] WARNING: parquet 行数 {len(df)} 与 profile.target_total={profile['target_total']} 不一致")

    required_lineage = {
        "sample_id", "root_seed_id", "parent_id", "crawl_stage", "split_role",
        "frame_version", "dataset_schema", "label_mc_replicates",
        "label_mc_min_replicates", "label_mc_max_replicates",
    }
    missing_lineage = sorted(required_lineage - set(df.columns))
    if missing_lineage:
        raise RuntimeError(f"Stage-0 数据集缺少谱系/坐标字段: {missing_lineage}")
    if set(df["frame_version"].astype(str).unique()) != {REQUIRED_FRAME_VERSION}:
        raise RuntimeError("parquet frame_version 与 Stage-0 坐标约定不一致。")
    if set(df["dataset_schema"].astype(str).unique()) != {REQUIRED_DATASET_SCHEMA}:
        raise RuntimeError("parquet dataset_schema 与 Stage-0 schema 不一致。")
    if df["sample_id"].isna().any() or df["sample_id"].astype(str).duplicated().any():
        raise RuntimeError("Stage-0 数据集 sample_id 存在空值或重复。")
    mc_actual = df["label_mc_replicates"].astype(int)
    mc_minimum = df["label_mc_min_replicates"].astype(int)
    mc_maximum = df["label_mc_max_replicates"].astype(int)
    if ((mc_actual < 1) | (mc_minimum < 1) | (mc_maximum < mc_minimum) |
            (mc_actual < mc_minimum) | (mc_actual > mc_maximum)).any():
        raise RuntimeError("自适应 MC 次数不满足 min <= actual <= max。")
    mc_histogram = {
        str(int(k)): int(v) for k, v in mc_actual.value_counts().sort_index().items()
    }
    if profile.get("label_mc", {}).get("replicate_histogram") != mc_histogram:
        raise RuntimeError("profile 与 parquet 的自适应 MC 次数直方图不一致。")
    family_cap = int(profile.get("family_distribution", {}).get(
        "maximum_rows_per_root_configured", -1))
    observed_max_family = int(df["root_seed_id"].astype(str).value_counts().max())
    if family_cap <= 0 or observed_max_family > family_cap:
        raise RuntimeError(
            f"root family 上限门禁失败: observed={observed_max_family}, cap={family_cap}")

    # Task multipliers are explicit experiment configuration.  The previous
    # hard-coded K/C maps silently overrode the Parquet and compounded IPW,
    # pos_weight, focal loss and resampling.  The credible baseline is all ones.
    mun_ids = df["munition_id"].to_numpy(dtype=np.int64)
    k_vector = _task_weight_vector(data_cfg, "k_task_weights")
    c_vector = _task_weight_vector(data_cfg, "c_task_weights")
    m_vector = _task_weight_vector(data_cfg, "m_task_weights")
    k_task_weights = k_vector[mun_ids]
    c_task_weights = c_vector[mun_ids]
    m_task_weights = m_vector[mun_ids]

    if use_terminal_physics_features:
        df = augment_terminal_physics_features(
            df,
            copy=False,
            include_component_proxies=use_component_proxy_features,
            armor_aware_fragment_proxies=(
                use_armor_aware_fragment_proxies),
        )
    X = df[feature_columns].values.astype(np.float32)
    y, y_soft, label_confidence = _build_ordinal_targets(
        df,
        use_soft_labels=bool(data_cfg.get("use_soft_labels", True)),
        use_label_uncertainty=bool(
            data_cfg.get("use_label_uncertainty", False)),
        uncertainty_scale=float(
            data_cfg.get("label_uncertainty_scale", 0.10)),
        confidence_floor=float(
            data_cfg.get("label_confidence_floor", 0.25)),
    )
    mechanism_targets = (
        _build_mechanism_targets(df)
        if use_mechanism_supervision else None
    )
    if use_component_supervision:
        component_targets, component_supervision_contract = (
            _load_component_targets(
                df, parquet_path, profile, data_cfg)
        )
    else:
        component_targets = None
        component_supervision_contract = None
    confidence_strength = _task_munition_matrix(
        data_cfg,
        "label_confidence_strength_by_task_munition",
        default=1.0,
        minimum=0.0,
        maximum=1.0,
    )
    if bool(data_cfg.get("use_label_uncertainty", False)):
        # strength=1 retains the MC reliability weight; strength=0 bypasses
        # it for a cell. Intermediate values shrink the weight toward one.
        label_confidence = _apply_label_confidence_strength(
            label_confidence, mun_ids, confidence_strength)
        print(
            "[Dataset] MC confidence strength "
            "(rows=K/M/F/C, cols=Small/Med-LM/Med-RD/Heavy): "
            f"{confidence_strength.tolist()}")
    loss_weights = df["loss_weight"].values.astype(np.float32)
    sample_ids = df["sample_id"].astype(str).to_numpy()
    root_seed_ids = df["root_seed_id"].astype(str).to_numpy()

    # =======================================================================
    # [P1 #7] GroupShuffleSplit：按"种子组"切分，防止爬行近邻簇泄漏
    # =======================================================================
    groups = _build_group_keys(df)

    train_idx, val_idx, test_idx = _load_or_create_split_indices(
        parquet_path, X, y, groups, random_state,
        split_roles=df["split_role"].values,
        root_seed_ids=df["root_seed_id"].values,
    )

    def _take(idx):
        values = (
            X[idx], y[idx], y_soft[idx], label_confidence[idx],
            mun_ids[idx], loss_weights[idx], k_task_weights[idx],
            c_task_weights[idx], m_task_weights[idx],
            sample_ids[idx], root_seed_ids[idx],
        )
        if mechanism_targets is not None:
            values = values + (mechanism_targets[idx],)
        if component_targets is not None:
            values = values + (component_targets[idx],)
        return values

    train_parts = _take(train_idx)
    val_parts = _take(val_idx)
    (X_train, y_train, ys_train, conf_train, mun_train, w_train,
     kw_train, cw_train, mw_train, sid_train, rid_train) = train_parts[:11]
    (X_val, y_val, ys_val, conf_val, mun_val, w_val,
     kw_val, cw_val, mw_val, sid_val, rid_val) = val_parts[:11]
    auxiliary_cursor = 11
    if use_mechanism_supervision:
        mechanism_train = train_parts[auxiliary_cursor]
        mechanism_val = val_parts[auxiliary_cursor]
        auxiliary_cursor += 1
    else:
        mechanism_train = mechanism_val = None
    if use_component_supervision:
        component_train = train_parts[auxiliary_cursor]
        component_val = val_parts[auxiliary_cursor]
    else:
        component_train = component_val = None
    if load_test_split:
        test_parts = _take(test_idx)
        (X_test, y_test, ys_test, conf_test, mun_test, w_test,
         kw_test, cw_test, mw_test, sid_test, rid_test) = test_parts[:11]
        test_cursor = 11
        if use_mechanism_supervision:
            mechanism_test = test_parts[test_cursor]
            test_cursor += 1
        else:
            mechanism_test = None
        component_test = (
            test_parts[test_cursor]
            if use_component_supervision else None)
    else:
        X_test = y_test = ys_test = conf_test = mun_test = None
        w_test = kw_test = cw_test = mw_test = None
        sid_test = rid_test = mechanism_test = None
        component_test = None

    if component_train is not None:
        positive_mass = component_train.sum(
            axis=0, dtype=np.float64)
        negative_mass = (
            float(len(component_train)) - positive_mass)
        # Square-root tempering avoids making sparse auxiliary labels dominate
        # the authoritative ordinal task while still preventing an all-zero
        # component predictor from minimizing the dense 102-target loss.
        component_positive_weight = np.clip(
            np.sqrt(
                (negative_mass + 1.0)
                / (positive_mass + 1.0)),
            1.0,
            10.0,
        ).astype(np.float32)
    else:
        component_positive_weight = None

    _validate_dataset_usability(
        profile,
        y_val, mun_val,
        y_test, mun_test,
    )

    # Training fits the scaler on train only; evaluation/deployment must inject
    # the fitted artifact through scaler_override and may never refit it.
    if scaler_override is None:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    else:
        scaler = scaler_override
        expected_features = getattr(scaler, "n_features_in_", len(feature_columns))
        if int(expected_features) != len(feature_columns):
            raise RuntimeError(
                "Scaler feature count does not match the active feature contract.")
        X_train_scaled = scaler.transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = (
        scaler.transform(X_test).astype(np.float32)
        if load_test_split else None)

    # [P2 #18] 持久化 scaler 到磁盘，供 eval / 部署侧加载 —— 线上推理前必须复用同一条 MinMax
    # 映射，否则特征分布会被二次缩放，Logit 行为完全失真 (这是 R1~R10 未覆盖到的工程漏洞)。
    if persist_scaler:
        if scaler_override is not None:
            raise RuntimeError("Evaluation must not overwrite the fitted scaler.")
        import pickle
        os.makedirs(scaler_dir, exist_ok=True)
        scaler_path = os.path.join(scaler_dir, "minmax_scaler.pkl")
        with open(scaler_path, "wb") as _fh:
            pickle.dump(scaler, _fh)
        # 同步导出一份 JSON，便于非 Python 推理端 (C++/ONNXRuntime) 直接读取 min/scale
        scaler_meta = {
            "feature_names": feature_columns,
            "feature_range": [-1.0, 1.0],
            "data_min_": scaler.data_min_.tolist(),
            "data_max_": scaler.data_max_.tolist(),
            "scale_": scaler.scale_.tolist(),
            "min_": scaler.min_.tolist(),
        }
        import json as _json
        with open(os.path.join(scaler_dir, "minmax_scaler.json"), "w", encoding="utf-8") as _fh:
            _json.dump(scaler_meta, _fh, indent=2, ensure_ascii=False)
        print(f"[Dataset] MinMaxScaler 已固化: {scaler_path}  (+ 同名 .json 元数据)")

    # [R16] pos_weight 升级为 per-(munition, task, level) 三维形状 (4, 4, 2)
    # 仅基于 train 集统计 (杜绝 test 信号回流). 旧的全局 _compute_pos_weight
    # 也保留供调试 / fallback 时打印用.
    pos_weight_global = _compute_pos_weight(y_train, cap=100.0)
    pos_weight_per_mun = _compute_per_mun_pos_weight(y_train, mun_train)
    pos_weight_mode = str(data_cfg.get("pos_weight_mode", "ones")).lower()
    if pos_weight_mode == "per_munition":
        pos_weight = pos_weight_per_mun
    elif pos_weight_mode == "global":
        pos_weight = pos_weight_global
        print("[Dataset][Ablation] pos_weight_mode=global; per-munition BCE prior disabled.")
    elif pos_weight_mode == "ones":
        pos_weight = np.ones((4, 2), dtype=np.float32)
        print("[Dataset][Ablation] pos_weight_mode=ones; BCE positive prior disabled.")
    else:
        raise ValueError(f"Unknown data.pos_weight_mode: {pos_weight_mode}")
    sampler_weights, train_loss_balance, balance_diag = \
        _compute_adaptive_train_balance(y_train, mun_train)
    if data_cfg.get("use_adaptive_loss_balance", False):
        w_train = (w_train.astype(np.float32) * train_loss_balance).astype(np.float32)
    else:
        print("[Dataset] adaptive train loss balance 默认关闭，避免与 pos_weight/focal 重复放大。")
    if not data_cfg.get("use_adaptive_sampler_balance", False):
        sampler_weights = np.ones_like(sampler_weights, dtype=np.float64)
        print("[Dataset] 弹型内稀有类重采样默认关闭；仅保留四弹型等量 batch。")

    train_dataset = DamageDataset(
        X_train_scaled, y_train, mun_train, w_train, kw_train, cw_train,
        mw_train, ys_train, conf_train, sid_train, rid_train,
        mechanism_train, component_train)
    val_dataset = DamageDataset(
        X_val_scaled, y_val, mun_val, w_val, kw_val, cw_val,
        mw_val, ys_val, conf_val, sid_val, rid_val,
        mechanism_val, component_val)
    test_dataset = (
        DamageDataset(
            X_test_scaled, y_test, mun_test, w_test, kw_test, cw_test,
            mw_test, ys_test, conf_test, sid_test, rid_test,
            mechanism_test, component_test)
        if load_test_split else None)

    # ================================================================
    # 强平衡 batch 采样
    # ================================================================
    # 每个 batch 固定 4 种弹型等量曝光，弹型内再由 rare-factor 决定采样概率。
    # 这样弹型平衡由 batch 结构硬约束，稀有类放大则只发生在弹型内部。
    # ================================================================
    is_k1 = (y_train[:, 0, 0] == 1) & (y_train[:, 0, 1] == 0)
    is_m1 = (y_train[:, 1, 0] == 1) & (y_train[:, 1, 1] == 0)
    is_c1 = (y_train[:, 3, 0] == 1) & (y_train[:, 3, 1] == 0)
    any_class1_kmc = is_k1 | is_m1 | is_c1

    # Diagnostics: expected sampler exposure for the weak class-1 slices.
    total_w = sampler_weights.sum()
    c1_share = sampler_weights[any_class1_kmc].sum() / total_w
    small_k1 = int((is_k1 & (mun_train == 0)).sum())
    small_c1 = int((is_c1 & (mun_train == 0)).sum())
    heavy_m1 = int((is_m1 & (mun_train == 3)).sum())
    print("[Dataset] [Adaptive] sampler diagnostics")
    print(f"  class-1 (K/M/C any) 样本占比: {any_class1_kmc.sum()/len(y_train)*100:.2f}% "
          f"→ 加权后 batch 期望占比: {c1_share*100:.2f}%")
    print(f"  结构性稀缺格样本: Small×K1={small_k1}, Small×C1={small_c1}, "
          f"Heavy×M1={heavy_m1}")

    weighted_mun_share = []
    for m in range(4):
        mask = (mun_train == m)
        weighted_mun_share.append(float(sampler_weights[mask].sum() / max(total_w, 1e-8) * 100.0))
    print(f"[Dataset] [Adaptive] munition raw counts: {balance_diag['mun_counts'].tolist()}")
    print(f"[Dataset] [Adaptive] munition sampler factors: "
          f"{[round(float(v), 3) for v in balance_diag['mun_sampler_factor']]}")
    print(f"[Dataset] [Adaptive] munition loss factors: "
          f"{[round(float(v), 3) for v in balance_diag['mun_loss_factor']]}")
    print(f"[Dataset] [Adaptive] raw weight mass by munition (%，仅诊断，不代表 batch 曝光): "
          f"{[round(v, 2) for v in weighted_mun_share]}")
    print(f"[Dataset] [Adaptive] class-1 task counts: "
          f"{dict(zip(balance_diag['task_names'], balance_diag['class1_counts'].tolist()))}")
    print(f"[Dataset] [Adaptive] class-1 task boosts: "
          f"{dict(zip(balance_diag['task_names'], [round(float(v), 3) for v in balance_diag['task_class1_factor']]))}")
    for i, t_name in enumerate(balance_diag['task_names']):
        row_counts = balance_diag['cell_class1_counts'][i].tolist()
        row_facts = [round(float(v), 3) for v in balance_diag['cell_class1_factor'][i]]
        print(f"[Dataset] [Adaptive] {t_name} class-1 by munition: "
              f"{dict(zip(balance_diag['mun_names'], row_counts))} | boosts={row_facts}")

    batch_sampler = BalancedMunitionBatchSampler(
        mun_ids=mun_train,
        sampler_weights=sampler_weights,
        batch_size=batch_size,
        random_state=random_state,
    )
    expected_draws = batch_sampler.expected_draws_per_munition
    print(f"[Dataset] [Balanced] per-batch munition exposure: {batch_sampler.per_mun_batch} × 4")
    print(f"[Dataset] [Balanced] per-epoch expected draws: {[expected_draws[m] for m in range(4)]}")

    if data_cfg.get("use_balanced_sampler", False):
        train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)
    else:
        print("[Dataset] Baseline loader uses shuffle=True without replacement.")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = (
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        if load_test_split else None)

    # 控制台报告
    print(
        f"[Dataset] 切分完成 → Train:{len(train_dataset)} | "
        f"Val:{len(val_dataset)}"
        + (
            f" | Test:{len(test_dataset)}"
            if load_test_split else " | Test:SEALED"
        )
    )
    if "per_munition" in profile:
        target_quota = {
            int(m): int(meta.get("target_quota", 0))
            for m, meta in profile["per_munition"].items()
        }
        print(f"[Dataset] generation_profile 目标配额: {target_quota}")
        small_profile = profile["per_munition"].get("0", {})
        small_pos = small_profile.get("positive_counts", {})
        if "C1_only" in small_pos:
            print(f"[Dataset] generation_profile Small C1_only={int(small_pos['C1_only'])}")
    report_splits = [
        ("Train", (mun_train, w_train, kw_train, cw_train, mw_train, y_train)),
        ("Val",   (mun_val,   w_val,   kw_val,   cw_val,   mw_val,   y_val)),
    ]
    if load_test_split:
        report_splits.append(
            ("Test", (
                mun_test, w_test, kw_test, cw_test, mw_test, y_test)))
    for split_name, (m, w, kw, cw, mw, ys) in report_splits:
        mun_counts = np.bincount(m, minlength=4)
        print(f"  {split_name:5s}  m_id 分布: {mun_counts.tolist()}  | "
              f"K2_pos: {int(ys[:,0,1].sum()):>5} | "
              f"loss_w∈[{w.min():.3f},{w.max():.2f}] | "
              f"k_task_w∈[{kw.min():.2f},{kw.max():.2f}] | "
              f"c_task_w∈[{cw.min():.2f},{cw.max():.2f}] | "
              f"m_task_w∈[{mw.min():.2f},{mw.max():.2f}]")
        _print_split_level_grid(split_name, ys, m)

    print(f"[Dataset] [R16] 全局任务级 pos_weight (历史口径, n_neg/n_pos, capped@100):")
    for i, name in enumerate(["K", "M", "F", "C"]):
        print(f"  {name}: lvl1={pos_weight_global[i,0]:.2f}  lvl2={pos_weight_global[i,1]:.2f}")
    print(f"[Dataset] [R16] per-munition × task × level pos_weight (新口径, 形状 (4,4,2)):")
    pos_weight_print = (
        pos_weight
        if pos_weight.ndim == 3
        else np.broadcast_to(pos_weight[None, :, :], (4, 4, 2))
    )
    mun_names = ["Small", "Med-LM", "Med-RD", "Heavy"]
    task_names = ["K", "M", "F", "C"]
    print(f"  {'弹型':<7}", end="")
    for tn in task_names:
        print(f" | {tn}_L1   {tn}_L2  ", end="")
    print()
    for m, mn in enumerate(mun_names):
        print(f"  {mn:<7}", end="")
        for i in range(4):
            print(f" | {pos_weight_print[m,i,0]:6.2f} {pos_weight_print[m,i,1]:6.2f} ", end="")
        print()

    mc_resolution_columns = [
        f"{task}_ge{level}_mc_resolved"
        for task in ("K", "M", "F", "C")
        for level in (1, 2)
    ]
    mc_resolution_available = all(
        column in df.columns
        for column in mc_resolution_columns
    )
    data_contract = {
        "contract_schema": "stage0_nn_data_v1",
        "dataset_path": os.path.abspath(parquet_path),
        "dataset_sha256": str(profile["artifact"]["sha256"]),
        "dataset_rows": int(len(df)),
        "dataset_schema": REQUIRED_DATASET_SCHEMA,
        "frame_convention": REQUIRED_FRAME_VERSION,
        "feature_names": list(feature_columns),
        "feature_count": int(len(feature_columns)),
        "terminal_physics_contract": (
            terminal_physics_contract_metadata(
                include_component_proxies=(
                    use_component_proxy_features),
                armor_aware_fragment_proxies=(
                    use_armor_aware_fragment_proxies))
            if use_terminal_physics_features else None
        ),
        "ordinal_applicability": ordinal_applicability.astype(bool).tolist(),
        "split_counts": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "target_mode": (
            "full_mc_mean" if data_cfg.get("use_soft_labels", True)
            else "hard_ordinal"
        ),
        "label_uncertainty_enabled": bool(
            data_cfg.get("use_label_uncertainty", False)),
        "mc_resolution_supervision_available": bool(
            mc_resolution_available),
        "mc_resolution_rate_by_ordinal_head": (
            {
                column: float(
                    df[column].astype(bool).mean())
                for column in mc_resolution_columns
            }
            if mc_resolution_available else None
        ),
        "label_confidence_strength_by_task_munition": (
            confidence_strength.tolist()),
        "mechanism_supervision_enabled": use_mechanism_supervision,
        "mechanism_target_schema": (
            "fragment_shock_ordinal_mc_mean_v1"
            if use_mechanism_supervision else None
        ),
        "component_supervision_enabled": use_component_supervision,
        "component_supervision_contract": (
            component_supervision_contract
            if use_component_supervision else None
        ),
        "component_positive_weight": (
            component_positive_weight.tolist()
            if component_positive_weight is not None else None
        ),
        "pos_weight_mode": pos_weight_mode,
        "balanced_sampler": bool(
            data_cfg.get("use_balanced_sampler", False)),
    }
    return (
        train_loader, val_loader, test_loader, scaler, pos_weight,
        data_contract,
    )
