from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import argparse
import json
import time
import sys
import random
from contextlib import redirect_stderr, redirect_stdout
from typing import Optional
from torch.utils.tensorboard import SummaryWriter

# 开启 CUDNN 基准寻找以加速训练
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

from loitering_munition_damage_twin.surrogate.model import (
    DamageAssessmentMTL,
    DEFAULT_ORDINAL_APPLICABILITY,
    component_probabilities_to_ordinal,
)
from loitering_munition_damage_twin.surrogate.dataset import get_dataloaders, get_feature_columns
from loitering_munition_damage_twin.surrogate.artifacts import sha256_file, write_model_manifest
from loitering_munition_damage_twin.paths import PROJECT_ROOT

try:
    from loitering_munition_damage_twin.experiments.ablation_config import (
        load_ablation_config,
        resolve_output_dir,
        write_resolved_config,
    )
except ImportError:
    load_ablation_config = None
    resolve_output_dir = None
    write_resolved_config = None


# Focal Loss γ —— 与 generate_dataset.py CONFIG["FOCAL_LOSS_GAMMA"] 对齐
FOCAL_LOSS_GAMMA = 2.0

# Final research objective.  Keep this separate from the historical 85%-L1
# diagnostic gate so old experiments remain comparable while a run cannot be
# mistaken for satisfying the stricter thesis acceptance contract.
GOAL_MIN_CELL_3CLASS_ACCURACY_PERCENT = 94.0
GOAL_MIN_CLASS_DIAGONAL_RECALL_PERCENT = 90.0
GOAL_MIN_CLASS_SUPPORT = 100


def _cfg_section(ablation_config: dict | None, name: str) -> dict:
    if not ablation_config:
        return {}
    section = ablation_config.get(name, {})
    return section if isinstance(section, dict) else {}


def _resolve_repo_relative_path(path: str, seed: int) -> str:
    rendered = str(path).format(seed=int(seed))
    if os.path.isabs(rendered):
        return os.path.abspath(rendered)
    return str((PROJECT_ROOT / rendered).resolve())


def _json_values_equal(left, right) -> bool:
    return json.dumps(
        left, sort_keys=True, separators=(",", ":")
    ) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def _load_verified_warm_start(
        model: nn.Module,
        training_cfg: dict,
        data_contract: dict,
        resolved_model_config: dict,
        seed: int) -> dict | None:
    checkpoint_cfg = training_cfg.get("initial_checkpoint")
    if not checkpoint_cfg:
        return None
    checkpoint_path = _resolve_repo_relative_path(
        str(checkpoint_cfg), seed)
    manifest_cfg = training_cfg.get("initial_manifest")
    manifest_path = (
        _resolve_repo_relative_path(str(manifest_cfg), seed)
        if manifest_cfg
        else os.path.join(
            os.path.dirname(checkpoint_path), "model_manifest.json")
    )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Warm-start checkpoint is missing: {checkpoint_path}")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"Warm-start manifest is missing: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as stream:
        source_manifest = json.load(stream)
    if source_manifest.get("schema") != "stage0_nn_artifact_v1":
        raise RuntimeError(
            "Warm-start manifest has unsupported schema: "
            f"{source_manifest.get('schema')!r}")
    source_contract = source_manifest.get("data_contract", {})
    source_feature_names = list(
        source_contract.get("feature_names", []))
    current_feature_names = list(
        data_contract.get("feature_names", []))
    base_input_dim = int(
        resolved_model_config.get(
            "base_input_dim", len(current_feature_names)))
    feature_extension = bool(
        len(current_feature_names) > len(source_feature_names)
        and base_input_dim == len(source_feature_names)
        and current_feature_names[:base_input_dim]
        == source_feature_names
    )
    for contract_key in (
            "dataset_sha256", "dataset_schema", "frame_convention",
            "ordinal_applicability"):
        if not _json_values_equal(
                source_contract.get(contract_key),
                data_contract.get(contract_key)):
            raise RuntimeError(
                "Warm-start data contract mismatch for "
                f"{contract_key}.")
    if (
        current_feature_names != source_feature_names
        and not feature_extension
    ):
        raise RuntimeError(
            "Warm-start data contract mismatch for feature_names. "
            "A permitted extension must preserve the complete source "
            "feature prefix and set model.base_input_dim to its length.")

    checkpoint_name = os.path.basename(checkpoint_path)
    sealed_checkpoint = (
        source_manifest.get("artifacts", {}).get(checkpoint_name, {}))
    sealed_checkpoint_hash = str(
        sealed_checkpoint.get("sha256", ""))
    observed_checkpoint_hash = sha256_file(checkpoint_path)
    if (
        not sealed_checkpoint_hash
        or sealed_checkpoint_hash != observed_checkpoint_hash
    ):
        raise RuntimeError(
            "Warm-start checkpoint SHA-256 does not match its manifest.")

    current_scaler_path = "./output/models/minmax_scaler.pkl"
    source_scaler_hash = str(
        source_manifest.get("artifacts", {})
        .get("minmax_scaler.pkl", {}).get("sha256", ""))
    source_scaler_path = os.path.join(
        os.path.dirname(manifest_path), "minmax_scaler.pkl")
    if (
        not os.path.isfile(current_scaler_path)
        or not os.path.isfile(source_scaler_path)
        or not source_scaler_hash
        or sha256_file(source_scaler_path) != source_scaler_hash
    ):
        raise RuntimeError(
            "Warm-start source/current scaler artifact is missing or "
            "the source hash differs from its manifest.")
    if feature_extension:
        source_scaler_json_path = os.path.join(
            os.path.dirname(manifest_path), "minmax_scaler.json")
        current_scaler_json_path = os.path.join(
            os.path.dirname(current_scaler_path), "minmax_scaler.json")
        if (
            not os.path.isfile(source_scaler_json_path)
            or not os.path.isfile(current_scaler_json_path)
        ):
            raise RuntimeError(
                "Feature-extension warm start requires both scaler JSON "
                "contracts.")
        with open(
                source_scaler_json_path, "r",
                encoding="utf-8") as stream:
            source_scaler_contract = json.load(stream)
        with open(
                current_scaler_json_path, "r",
                encoding="utf-8") as stream:
            current_scaler_contract = json.load(stream)
        if (
            list(source_scaler_contract.get(
                "feature_names", []))
            != current_feature_names[:base_input_dim]
            or list(current_scaler_contract.get(
                "feature_names", []))[:base_input_dim]
            != source_feature_names
        ):
            raise RuntimeError(
                "Feature-extension scaler feature prefix mismatch.")
        import numpy as _np_warm_start
        for scaler_key in (
                "data_min_", "data_max_", "scale_", "min_"):
            source_values = _np_warm_start.asarray(
                source_scaler_contract.get(scaler_key, []),
                dtype=_np_warm_start.float64)
            current_values = _np_warm_start.asarray(
                current_scaler_contract.get(scaler_key, []),
                dtype=_np_warm_start.float64)[:base_input_dim]
            if (
                source_values.shape != current_values.shape
                or not _np_warm_start.allclose(
                    source_values, current_values,
                    rtol=0.0, atol=1e-12)
            ):
                raise RuntimeError(
                    "Feature-extension warm start changed the scaler "
                    f"prefix for {scaler_key}.")
    elif sha256_file(current_scaler_path) != source_scaler_hash:
        raise RuntimeError(
            "Warm-start scaler differs from the scaler fitted for this run.")

    adapter_config_keys = {
        "residual_adapter_cells",
        "residual_adapter_hidden_dim",
        "residual_adapter_feature_indices",
        "residual_adapter_frequencies",
        "residual_adapter_max_logit",
        # The fusion matrix is a non-persistent calibration buffer.  It has
        # no checkpoint parameters, so a continuation experiment may
        # pre-register a new matrix without weakening state-dict checks.
        "component_tree_fusion_alpha",
    }
    source_model_config = source_manifest.get("model_config", {})
    current_independent_component_branch = bool(
        resolved_model_config.get(
            "use_component_auxiliary_heads", False)
        and str(resolved_model_config.get(
            "component_branch_mode", "shared_auxiliary")
        ).strip().lower() == "independent_experts"
    )
    source_independent_component_branch = bool(
        source_model_config.get(
            "use_component_auxiliary_heads", False)
        and str(source_model_config.get(
            "component_branch_mode", "shared_auxiliary")
        ).strip().lower() == "independent_experts"
    )
    independent_component_extension = bool(
        current_independent_component_branch
        and not source_independent_component_branch
    )
    independent_component_continuation = bool(
        current_independent_component_branch
        and source_independent_component_branch
    )
    independent_component_feature_extension = bool(
        independent_component_continuation
        and feature_extension
    )
    component_extension_config_keys = {
        "use_component_auxiliary_heads",
        "component_ids",
    }
    for key, source_value in source_model_config.items():
        if key in adapter_config_keys:
            continue
        if (
            independent_component_feature_extension
            and key == "in_dim"
        ):
            if int(source_value) != base_input_dim:
                raise RuntimeError(
                    "Warm-start component feature extension does not "
                    "preserve the source input width.")
            continue
        if (
            independent_component_extension
            and key in component_extension_config_keys
        ):
            continue
        if not _json_values_equal(
                source_value, resolved_model_config.get(key)):
            raise RuntimeError(
                f"Warm-start base model config mismatch for {key}.")

    source_state = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True)
    migrated_component_input_key = None
    if independent_component_feature_extension:
        component_input_key = (
            "independent_component_branch.encoder.0.weight")
        if (
            component_input_key not in source_state
            or component_input_key not in model.state_dict()
        ):
            raise RuntimeError(
                "Independent component input layer is missing from the "
                "feature-extension warm start.")
        source_weight = source_state[component_input_key]
        current_weight = model.state_dict()[
            component_input_key].clone()
        source_embedding_dim = int(
            source_model_config.get(
                "component_branch_munition_emb_dim", 16))
        current_embedding_dim = int(
            resolved_model_config.get(
                "component_branch_munition_emb_dim", 16))
        if source_embedding_dim != current_embedding_dim:
            raise RuntimeError(
                "Component feature extension cannot change its munition "
                "embedding width.")
        expected_source_width = (
            base_input_dim + source_embedding_dim)
        expected_current_width = (
            len(current_feature_names)
            + current_embedding_dim)
        if (
            tuple(source_weight.shape)
            != (current_weight.shape[0], expected_source_width)
            or current_weight.shape[1] != expected_current_width
        ):
            raise RuntimeError(
                "Independent component input weight shape is incompatible "
                "with the verified prefix migration.")
        # Preserve the learned terminal-state and munition-embedding weights
        # exactly. New component-proxy columns begin at zero, so the extended
        # model is functionally identical to its source before fine-tuning.
        current_weight.zero_()
        current_weight[:, :base_input_dim].copy_(
            source_weight[:, :base_input_dim])
        current_weight[
            :, len(current_feature_names):
        ].copy_(
            source_weight[:, base_input_dim:])
        source_state[component_input_key] = current_weight
        migrated_component_input_key = component_input_key
    incompatible = model.load_state_dict(source_state, strict=False)
    allowed_missing_prefixes = (
        "residual_feature_expansion.",
        "residual_adapters.",
        "residual_adapter_munitions",
        "residual_adapter_basis",
    )
    if independent_component_extension:
        allowed_missing_prefixes = (
            allowed_missing_prefixes
            + ("independent_component_branch.",)
        )
    unexpected_missing = [
        key for key in incompatible.missing_keys
        if not key.startswith(allowed_missing_prefixes)
    ]
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Warm-start state dict is not a strict base-model subset: "
            f"missing={unexpected_missing}, "
            f"unexpected={incompatible.unexpected_keys}")
    if (
        not getattr(model, "residual_adapters", None)
        and not independent_component_extension
        and not independent_component_continuation
    ):
        raise RuntimeError(
            "Warm-start fine-tuning requires a registered extension "
            "(residual adapter or independent component branch).")

    return {
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": observed_checkpoint_hash,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "source_model_variant": (
            source_manifest.get("training_config", {})
            .get("model_variant", "unknown")),
        "expected_missing_adapter_keys": list(
            incompatible.missing_keys),
        "independent_component_extension": bool(
            independent_component_extension),
        "independent_component_continuation": bool(
            independent_component_continuation),
        "feature_extension": bool(feature_extension),
        "independent_component_feature_extension": bool(
            independent_component_feature_extension),
        "migrated_component_input_key": (
            migrated_component_input_key),
    }


class TeeStream:
    """Write console output to both the terminal and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _default_console_log_path() -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(
        "./output/runs/damage_model",
        f"training_console_{timestamp}.txt",
    )


def ordinal_class_distribution_nll(
        logits: torch.Tensor,
        cumulative_targets: torch.Tensor,
        eps: float = 1e-7) -> torch.Tensor:
    """Return per-row NLL for the ordinal three-class probability mass.

    ``logits`` and ``cumulative_targets`` have shape ``(N, 2)`` and represent
    ``P(L>=1)`` and ``P(L>=2)``.  The induced class distribution is
    ``[1-p1, p1-p2, p2]``.  Training this distribution explicitly supplies a
    direct gradient for the narrow middle class while preserving the monotone
    cumulative output contract.
    """
    if logits.ndim != 2 or logits.shape[-1] != 2:
        raise ValueError(
            f"logits must have shape (N, 2), got {tuple(logits.shape)}")
    if cumulative_targets.shape != logits.shape:
        raise ValueError(
            "cumulative_targets must have the same shape as logits, got "
            f"{tuple(cumulative_targets.shape)} vs {tuple(logits.shape)}")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")

    probabilities = torch.sigmoid(logits)
    p1 = probabilities[:, 0]
    p2 = torch.minimum(probabilities[:, 1], p1)
    class_probabilities = torch.stack(
        (1.0 - p1, p1 - p2, p2), dim=1)
    class_probabilities = class_probabilities.clamp_min(eps)
    class_probabilities = (
        class_probabilities
        / class_probabilities.sum(dim=1, keepdim=True)
    )

    target1 = cumulative_targets[:, 0].clamp(0.0, 1.0)
    target2 = torch.minimum(
        cumulative_targets[:, 1].clamp(0.0, 1.0), target1)
    class_targets = torch.stack(
        (1.0 - target1, target1 - target2, target2), dim=1)
    return -(class_targets * class_probabilities.log()).sum(dim=1)


def hard_negative_pairwise_ranking_loss(
        scores: torch.Tensor,
        positive_mask: torch.Tensor,
        negative_mask: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
        margin: float = 0.5,
        hard_negative_fraction: float = 0.10) -> torch.Tensor:
    """Rank positives above the highest-scoring fraction of negatives.

    This is a differentiable surrogate for improving the low-FPR part of the
    ROC curve.  Returning ``scores.sum() * 0`` for a one-class mini-batch keeps
    the graph valid without inventing synthetic pairs.
    """
    if scores.ndim != 1:
        raise ValueError("scores must be a one-dimensional tensor.")
    if positive_mask.shape != scores.shape or negative_mask.shape != scores.shape:
        raise ValueError("ranking masks must match scores.")
    if margin < 0.0:
        raise ValueError("ranking margin must be non-negative.")
    if not 0.0 < hard_negative_fraction <= 1.0:
        raise ValueError("hard_negative_fraction must be in (0, 1].")
    positive_mask = positive_mask.bool()
    negative_mask = negative_mask.bool()
    positive_scores = scores[positive_mask]
    negative_scores = scores[negative_mask]
    if positive_scores.numel() == 0 or negative_scores.numel() == 0:
        return scores.sum() * 0.0

    hard_count = max(
        1,
        int(torch.ceil(torch.tensor(
            negative_scores.numel() * hard_negative_fraction)).item()),
    )
    hard_count = min(hard_count, negative_scores.numel())
    _, hard_indices = torch.topk(
        negative_scores.detach(), k=hard_count, largest=True)
    hard_negative_scores = negative_scores[hard_indices]
    pair_loss = F.softplus(
        margin
        - positive_scores.unsqueeze(1)
        + hard_negative_scores.unsqueeze(0)
    )

    if sample_weights is None:
        return pair_loss.mean()
    if sample_weights.shape != scores.shape:
        raise ValueError("sample_weights must match scores.")
    positive_weights = sample_weights[positive_mask].clamp_min(0.0)
    negative_weights = sample_weights[negative_mask][
        hard_indices].clamp_min(0.0)
    positive_weights = positive_weights / positive_weights.mean().clamp_min(1e-8)
    negative_weights = negative_weights / negative_weights.mean().clamp_min(1e-8)
    pair_weights = (
        positive_weights.unsqueeze(1)
        * negative_weights.unsqueeze(0)
    )
    return (
        (pair_loss * pair_weights).sum()
        / pair_weights.sum().clamp_min(1e-8)
    )


class FocalUncertaintyOrdinalLoss(nn.Module):
    """Multi-task probabilistic ordinal loss.

    The improved baseline is full-MC-mean BCE with IPW plus a proper
    three-class distribution loss.  MC confidence weighting, focal modulation,
    positive priors and legacy class-1 shaping remain explicit isolated
    ablation switches.  Kendall task uncertainty is kept separate from all
    row/task weights.
    """
    def __init__(self, num_tasks: int = 4, penalty_weight: float = 10.0,
                 gamma: float = FOCAL_LOSS_GAMMA,
                 pos_weight: torch.Tensor = None,
                 class1_margin: float = 0.15,
                 class1_margin_weight: float = 0.5,
                 class1_alpha: float = 1.0,
                 cell_class1_alpha: torch.Tensor = None,
                 class_distribution_weight: float = 0.25,
                 hard_level_classification_weight: float = 0.0,
                 middle_class_distribution_multiplier:
                 torch.Tensor = None,
                 entry_ranking_weight: torch.Tensor = None,
                 conditional_l1_l2_ranking_weight: torch.Tensor = None,
                 ranking_margin: float = 0.5,
                 hard_negative_fraction: float = 0.10):
        super().__init__()
        if class_distribution_weight < 0.0:
            raise ValueError("class_distribution_weight must be non-negative.")
        if hard_level_classification_weight < 0.0:
            raise ValueError(
                "hard_level_classification_weight must be non-negative.")
        # [P0 #3] 初始化为 0.5（precision = e^-0.5 ≈ 0.6 起步），不再卡死在 1.0
        self.log_vars = nn.Parameter(torch.full((num_tasks,), 0.5))
        self.gamma = gamma
        self.penalty_weight = penalty_weight
        self.class_distribution_weight = float(class_distribution_weight)
        self.hard_level_classification_weight = float(
            hard_level_classification_weight)
        if middle_class_distribution_multiplier is None:
            middle_class_distribution_multiplier = torch.ones(
                num_tasks, 4, dtype=torch.float32)
        elif middle_class_distribution_multiplier.shape != (num_tasks, 4):
            raise ValueError(
                "middle_class_distribution_multiplier must have shape "
                f"({num_tasks}, 4), got "
                f"{tuple(middle_class_distribution_multiplier.shape)}")
        if (
            not torch.isfinite(
                middle_class_distribution_multiplier).all()
            or (middle_class_distribution_multiplier < 1.0).any()
        ):
            raise ValueError(
                "middle_class_distribution_multiplier values must be "
                "finite and >= 1.")
        self.register_buffer(
            "middle_class_distribution_multiplier",
            middle_class_distribution_multiplier.float(),
        )
        if ranking_margin < 0.0:
            raise ValueError("ranking_margin must be non-negative.")
        if not 0.0 < hard_negative_fraction <= 1.0:
            raise ValueError(
                "hard_negative_fraction must be in (0, 1].")
        self.ranking_margin = float(ranking_margin)
        self.hard_negative_fraction = float(hard_negative_fraction)
        for name, matrix in (
            ("entry_ranking_weight", entry_ranking_weight),
            ("conditional_l1_l2_ranking_weight",
             conditional_l1_l2_ranking_weight),
        ):
            if matrix is None:
                matrix = torch.zeros(
                    num_tasks, 4, dtype=torch.float32)
            if matrix.shape != (num_tasks, 4):
                raise ValueError(
                    f"{name} must have shape ({num_tasks}, 4), got "
                    f"{tuple(matrix.shape)}")
            if (
                not torch.isfinite(matrix).all()
                or (matrix < 0.0).any()
                or (matrix > 1.0).any()
            ):
                raise ValueError(
                    f"{name} values must be finite and in [0, 1].")
            self.register_buffer(name, matrix.float())
        self._entry_ranking_specs = [
            [
                (munition_id, float(
                    self.entry_ranking_weight[
                        task_id, munition_id].item()))
                for munition_id in range(4)
                if float(self.entry_ranking_weight[
                    task_id, munition_id].item()) > 0.0
            ]
            for task_id in range(num_tasks)
        ]
        self._conditional_ranking_specs = [
            [
                (munition_id, float(
                    self.conditional_l1_l2_ranking_weight[
                        task_id, munition_id].item()))
                for munition_id in range(4)
                if float(self.conditional_l1_l2_ranking_weight[
                    task_id, munition_id].item()) > 0.0
            ]
            for task_id in range(num_tasks)
        ]
        # [P1-D] class-1 margin loss 超参
        self.class1_margin = class1_margin
        self.class1_margin_weight = class1_margin_weight
        # [P1-A+] class-1 样本的 BCE 乘法因子（类别权重，不走采样端）
        #   对真实 y=(1,0) 样本把 BCE loss 乘以 alpha，让 class-1 在 loss 上获得
        #   更多梯度关注，而不需要依赖 WeightedRandomSampler 过度倾斜。
        #   alpha=2.0 → class-1 loss 权重翻倍；class-0 / class-2 不受影响。
        self.class1_alpha = class1_alpha
        if cell_class1_alpha is None:
            cell_class1_alpha = torch.ones(num_tasks, 4, dtype=torch.float32)
        elif cell_class1_alpha.shape != (num_tasks, 4):
            raise ValueError(
                f"cell_class1_alpha must have shape ({num_tasks}, 4), "
                f"got {tuple(cell_class1_alpha.shape)}")
        self.register_buffer("cell_class1_alpha", cell_class1_alpha.float())

        # [R16] pos_weight 接受两种形状, 内部统一为 3D (4_mun, 4_task, 2_lvl):
        #   * (4_task, 2_lvl)         — 旧口径全局先验, 自动广播到所有弹型
        #   * (4_mun, 4_task, 2_lvl)  — 新口径 per-munition 先验, 关键修正点
        # 二者对外接口完全一致, 只是 forward 阶段按 m_ids gather 出 (N, T, L).
        if pos_weight is None:
            pos_weight = torch.ones(4, num_tasks, 2)
        elif pos_weight.dim() == 2:
            # 旧 (4, 2) 自动扩为 (4_mun, 4_task, 2_lvl): 同一份权重广播给 4 个弹型
            pos_weight = pos_weight.unsqueeze(0).expand(4, -1, -1).contiguous()
        elif pos_weight.dim() != 3:
            raise ValueError(
                f"pos_weight must be 2D (T, L) or 3D (M, T, L), got shape {tuple(pos_weight.shape)}")
        # 用 buffer 注册，跟随 .to(device) 自动迁移；不参与优化
        self.register_buffer("pos_weight", pos_weight)

        # [P0-4] 差异化 log_var 下限：K 任务单独放宽到 -2.0，M/F/C 维持 -1.0。
        #   动机：K 是稀疏 hard task，全局下限 -1.0 让 precision 封顶 e^1≈2.72，
        #   K 的梯度份额被 M/F 挤压；放 K 到 -2.0 后 precision 上限 e^2≈7.4。
        #   与已回退的 R14-D 关键区别：R14-D 把**全部 4 任务**放到 -3.0 → M/F
        #   跑到 -2.97 反把 K 挤没；本方案**仅放 K**，M/F/C 仍卡在 -1.0 不能逃。
        self.register_buffer(
            "log_var_lower",
            torch.tensor([-2.0, -1.0, -1.0, -1.0], dtype=torch.float32),
        )

    def forward(self, logits, targets, ipw_weights,
                k_task_weight=None, c_task_weight=None, m_task_weight=None, m_ids=None,
                targets_soft=None, target_confidence=None):
        """
        logits: (N, 4, 2)
        targets: (N, 4, 2) 硬 0/1 标签 (仅用于离散class-1约束)
        ipw_weights: (N,)
        k_task_weight: (N,) 仅作用于 K 分支；None 时退化为全 1
        c_task_weight: (N,) [P0-2] 仅作用于 C 分支；None 时退化为全 1
        m_task_weight: (N,) [P3-M] 仅作用于 M 分支；当前用于定向提升 Small×M
        m_ids: (N,) [R16] 用于按样本 gather per-munition pos_weight (4,4,2)→(N,4,2);
               与 R14-C mfc_munition_weight 不同 — 此处只调正/负样本相对权重(BCE 内),
               不影响 task 间相对权重, 与 Kendall log_var 完全解耦.
        targets_soft: (N, 4, 2) 完整MC均值；缺省时使用硬标签.
        target_confidence: (N, 4, 2) 由MC标准误得到的有界可信度.
        """
        loss_total = 0.0
        # [P0-4] 差异化 log_var 下限：K=-2.0, M/F/C=-1.0。
        #   K precision 上限 e^2≈7.4，给 hard task 更大权重空间；
        #   M/F/C precision 上限 e^1≈2.72，防止"简单任务奖励"把 K 挤没。
        #   上限 2.5 对所有任务统一，允许需要时退权 (precision e^-2.5 ≈ 0.08)。
        safe_log_vars = torch.maximum(self.log_vars, self.log_var_lower).clamp(max=2.5)
        task_losses = []

        # [R16] 按样本 gather per-munition pos_weight: (4,4,2) → (N,4,2)
        # 若 m_ids 缺省 (推理 / 调用兼容), 退化为按弹型 0 取一份 (等价于全局).
        if m_ids is not None:
            pw_per_sample = self.pos_weight[m_ids]         # (N, 4_task, 2_lvl)
        else:
            pw_per_sample = self.pos_weight[0:1].expand(
                logits.shape[0], -1, -1)                   # 占位 fallback

        for i in range(4):
            task_logits = logits[:, i, :]      # [N, 2]
            task_targets = targets[:, i, :]    # [N, 2] 硬标签 (用于 mask 与 focal_factor)
            # [R20] BCE 计算用的目标: 优先软标签, 否则退化为硬标签
            if targets_soft is not None:
                bce_targets = targets_soft[:, i, :]   # [N, 2] 连续 ∈ [0, 1]
            else:
                bce_targets = task_targets

            # Proper soft-target BCE.  For an optional positive prior, interpolate
            # by the probability target itself instead of the thresholded label.
            pw_i = pw_per_sample[:, i, :]                                  # [N, 2]
            weight_factor = 1.0 + (pw_i - 1.0) * bce_targets
            bce_raw = F.binary_cross_entropy_with_logits(
                task_logits, bce_targets, reduction='none')                # [N, 2]
            p = torch.sigmoid(task_logits)
            p_t = p * bce_targets + (1.0 - p) * (1.0 - bce_targets)
            focal_factor = (1.0 - p_t).pow(self.gamma)
            confidence_i = (
                target_confidence[:, i, :]
                if target_confidence is not None
                else torch.ones_like(bce_raw)
            )
            bce_focal = focal_factor * bce_raw * weight_factor * confidence_i
            bce_sample = bce_focal.mean(dim=1)                # [N]
            is_class1 = (
                (task_targets[:, 0] > 0.5)
                & (task_targets[:, 1] < 0.5)
            )

            # Explicit proper three-class distribution objective.  It is not
            # multiplied by the optional MC-confidence factor: that factor is
            # class-correlated in the current dataset and the A13 ablation must
            # isolate its effect to the historical cumulative BCE path.
            if m_ids is not None:
                middle_multiplier = (
                    self.middle_class_distribution_multiplier[i, m_ids])
            else:
                middle_multiplier = torch.full_like(
                    bce_sample,
                    float(
                        self.middle_class_distribution_multiplier[
                            i, 0].item()),
                )
            if self.class_distribution_weight > 0.0:
                class_distribution_sample = ordinal_class_distribution_nll(
                    task_logits, bce_targets)
                class_distribution_sample = (
                    class_distribution_sample
                    * torch.where(
                        is_class1,
                        middle_multiplier,
                        torch.ones_like(middle_multiplier),
                    )
                )
            else:
                class_distribution_sample = torch.zeros_like(bce_sample)

            # The continuous damage scores and the thresholded ordinal label
            # are related but not interchangeable targets.  In particular,
            # p_ge1-p_ge2 can be below 0.5 for a row whose deterministic
            # thresholded class is L1.  Keep the proper soft probability loss
            # above, then add an explicit hard-level NLL when requested.
            if self.hard_level_classification_weight > 0.0:
                hard_level_classification_sample = (
                    ordinal_class_distribution_nll(
                        task_logits, task_targets)
                )
                hard_level_classification_sample = (
                    hard_level_classification_sample
                    * torch.where(
                        is_class1,
                        middle_multiplier,
                        torch.ones_like(middle_multiplier),
                    )
                )
            else:
                hard_level_classification_sample = torch.zeros_like(
                    bce_sample)

            # ---------- 2. 物理保序性惩罚 ----------
            p1 = torch.sigmoid(task_logits[:, 0])
            p2 = torch.sigmoid(task_logits[:, 1])
            penalty_sample = F.relu(p2 - p1).pow(2)           # [N]

            # ---------- 2b. [P1-D] class-1 margin loss ----------
            margin_gap = p1 - p2                                                   # [N]
            margin_deficit = F.relu(self.class1_margin - margin_gap)              # [N]
            margin_sample = torch.where(
                is_class1,
                margin_deficit.pow(2),
                torch.zeros_like(margin_deficit),
            )                                                                       # [N]

            # [P1-A+] 2c. class-1 样本的 BCE alpha 放大（loss 端类别权重）
            # 与 P1-D margin loss 协同：margin loss 塑形概率间隔、alpha 提升 BCE 信号强度。
            # class-1 loss 权重 = alpha；class-0 / class-2 权重 = 1.0（不变）。
            if m_ids is not None:
                alpha_for_task = self.cell_class1_alpha[i, m_ids] * self.class1_alpha
            else:
                alpha_for_task = torch.full_like(bce_sample, self.class1_alpha)
            alpha_factor = torch.where(
                is_class1,
                alpha_for_task,
                torch.ones_like(bce_sample),
            )                                                                       # [N]
            bce_sample = bce_sample * alpha_factor

            inner_loss_sample = (bce_sample
                                 + self.class_distribution_weight
                                 * class_distribution_sample
                                 + self.hard_level_classification_weight
                                 * hard_level_classification_sample
                                 + self.penalty_weight * penalty_sample
                                 + self.class1_margin_weight * margin_sample)

            # ---------- 3. 加权 ----------
            sample_w = ipw_weights
            if i == 0 and k_task_weight is not None:
                # K 分支: 在 IPW 之上额外乘 K_task_weight (m_id 条件化)
                sample_w = sample_w * k_task_weight
            if i == 1 and m_task_weight is not None:
                # [P3-M] M 分支: 仅对 Small 样本加一档权重，提升 Small×M 的表征质量。
                sample_w = sample_w * m_task_weight
            if i == 3 and c_task_weight is not None:
                # [P0-2] C 分支: 对称于 K，乘 C_task_weight (m_id 条件化)
                # 仅微抬 Small/Med 的 C 梯度 + 微降 Heavy (0.85)，绝不触碰 M/F。
                sample_w = sample_w * c_task_weight
            # [R15] 去掉 R14-C 的 mfc_munition_weight —— 实证导致 Heavy × M
            # class-1 recall 崩到 22-28%、Small × M 3-class 掉到 88.2%。
            # m_ids 参数保留做签名兼容，但 M/F 分支不再特殊加权。
            inner_weighted = inner_loss_sample * sample_w

            # ---------- 4. 批聚合 ----------
            Loss_inner_aggr = inner_weighted.mean()
            if m_ids is not None:
                ranking_loss = task_logits.sum() * 0.0
                for munition_id, entry_weight in (
                        self._entry_ranking_specs[i]):
                    cell_mask = (m_ids == munition_id)
                    if not bool(cell_mask.any()):
                        continue
                    cell_weights = sample_w[cell_mask]
                    cell_targets = task_targets[cell_mask]
                    entry_confidence = confidence_i[
                        cell_mask, 0]
                    ranking_loss = ranking_loss + (
                        entry_weight
                        * hard_negative_pairwise_ranking_loss(
                            task_logits[cell_mask, 0],
                            positive_mask=(
                                cell_targets[:, 0] > 0.5),
                            negative_mask=(
                                cell_targets[:, 0] < 0.5),
                            sample_weights=(
                                cell_weights * entry_confidence),
                            margin=self.ranking_margin,
                            hard_negative_fraction=(
                                self.hard_negative_fraction),
                        )
                    )
                for munition_id, conditional_weight in (
                        self._conditional_ranking_specs[i]):
                    cell_mask = (m_ids == munition_id)
                    if not bool(cell_mask.any()):
                        continue
                    cell_weights = sample_w[cell_mask]
                    cell_targets = task_targets[cell_mask]
                    level1_mask = (
                        (cell_targets[:, 0] > 0.5)
                        & (cell_targets[:, 1] < 0.5)
                    )
                    level2_mask = cell_targets[:, 1] > 0.5
                    conditional_confidence = confidence_i[
                        cell_mask, 1]
                    ranking_loss = ranking_loss + (
                        conditional_weight
                        * hard_negative_pairwise_ranking_loss(
                            task_logits[cell_mask, 1],
                            positive_mask=level2_mask,
                            negative_mask=level1_mask,
                            sample_weights=(
                                cell_weights
                                * conditional_confidence),
                            margin=self.ranking_margin,
                            hard_negative_fraction=(
                                self.hard_negative_fraction),
                        )
                    )
                Loss_inner_aggr = Loss_inner_aggr + ranking_loss

            # ---------- 5. Kendall 不确定性 (log_var 正则不被任何 W 污染) ----------
            precision = torch.exp(-safe_log_vars[i])
            task_loss_final = 0.5 * precision * Loss_inner_aggr + 0.5 * safe_log_vars[i]

            task_losses.append(task_loss_final)
            loss_total = loss_total + task_loss_final

        return loss_total, torch.stack(task_losses)


def mechanism_auxiliary_loss(
        fragment_logits: torch.Tensor,
        shock_logits: torch.Tensor,
        mechanism_targets: torch.Tensor,
        row_weights: torch.Tensor,
        applicability: torch.Tensor,
        class_distribution_weight: float = 0.25,
        branch_weights: torch.Tensor | None = None,
        boundary_focus_weight: float = 0.0,
        boundary_focus_bandwidth: float = 0.15,
        hard_classification_weight: float = 0.0,
        use_dataset_row_weights: bool = True) -> torch.Tensor:
    """Proper auxiliary risk for fragment/shock mechanism probabilities.

    The simulator already records MC-mean cumulative probabilities for each
    mechanism.  Supervising those latent physical mechanisms prevents the
    shared combined head from letting the easier shock-dominated M/F tasks
    overwrite fragment-dominated K/C representations.
    """
    if tuple(fragment_logits.shape[1:]) != (4, 2):
        raise ValueError("fragment_logits must have shape (N,4,2).")
    if shock_logits.shape != fragment_logits.shape:
        raise ValueError(
            "shock_logits must have the same shape as fragment_logits.")
    if tuple(mechanism_targets.shape) != (
            fragment_logits.shape[0], 2, 4, 2):
        raise ValueError(
            "mechanism_targets must have shape (N,2,4,2).")
    if tuple(applicability.shape) != (
            fragment_logits.shape[0], 4, 2):
        raise ValueError("applicability must have shape (N,4,2).")
    if class_distribution_weight < 0.0:
        raise ValueError(
            "class_distribution_weight must be non-negative.")
    if boundary_focus_weight < 0.0:
        raise ValueError(
            "boundary_focus_weight must be non-negative.")
    if boundary_focus_bandwidth <= 0.0:
        raise ValueError(
            "boundary_focus_bandwidth must be positive.")
    if hard_classification_weight < 0.0:
        raise ValueError(
            "hard_classification_weight must be non-negative.")

    mechanism_logits = torch.stack(
        (fragment_logits, shock_logits), dim=1)
    target = mechanism_targets.to(
        dtype=mechanism_logits.dtype)
    valid = applicability.unsqueeze(1).expand_as(
        mechanism_logits).to(mechanism_logits.dtype)
    if branch_weights is None:
        branch_weights = torch.ones(
            2,
            dtype=mechanism_logits.dtype,
            device=mechanism_logits.device,
        )
    else:
        branch_weights = torch.as_tensor(
            branch_weights,
            dtype=mechanism_logits.dtype,
            device=mechanism_logits.device,
        )
    if tuple(branch_weights.shape) != (2,):
        raise ValueError("branch_weights must have shape (2,).")
    if (
        not torch.isfinite(branch_weights).all()
        or (branch_weights <= 0.0).any()
    ):
        raise ValueError(
            "branch_weights must be finite and positive.")
    element_weight = (
        valid * branch_weights.view(1, 2, 1, 1)
    )
    if boundary_focus_weight > 0.0:
        boundary_distance = (
            (target - 0.5) / float(boundary_focus_bandwidth)
        )
        element_weight = element_weight * (
            1.0
            + float(boundary_focus_weight)
            * torch.exp(-0.5 * boundary_distance.square())
        )
    bce = F.binary_cross_entropy_with_logits(
        mechanism_logits, target, reduction="none")
    element_denominator = element_weight.sum(
        dim=(1, 2, 3)).clamp_min(1.0)
    per_sample = (
        (bce * element_weight).sum(dim=(1, 2, 3))
        / element_denominator
    )

    if hard_classification_weight > 0.0:
        hard_target = (target >= 0.5).to(mechanism_logits.dtype)
        hard_bce = F.binary_cross_entropy_with_logits(
            mechanism_logits, hard_target, reduction="none")
        per_sample = per_sample + float(
            hard_classification_weight) * (
                (hard_bce * element_weight).sum(dim=(1, 2, 3))
                / element_denominator
            )

    if class_distribution_weight > 0.0:
        distribution_terms = []
        for mechanism_index in range(2):
            for task_index in range(4):
                # A task is either applicable at L1 (and optionally L2), or a
                # structural zero.  The latter is excluded from auxiliary risk.
                task_valid = applicability[:, task_index, 0].to(
                    mechanism_logits.dtype)
                term = ordinal_class_distribution_nll(
                    mechanism_logits[
                        :, mechanism_index, task_index, :],
                    target[:, mechanism_index, task_index, :],
                ) * task_valid * branch_weights[mechanism_index]
                distribution_terms.append(term)
        per_sample = per_sample + class_distribution_weight * (
            torch.stack(distribution_terms, dim=1).sum(dim=1)
            / applicability[:, :, 0].sum(
                dim=1).to(mechanism_logits.dtype).clamp_min(1.0)
            / branch_weights.sum()
        )

    effective_row_weights = (
        row_weights
        if bool(use_dataset_row_weights)
        else torch.ones_like(row_weights)
    )
    return (per_sample * effective_row_weights).mean()


def component_auxiliary_loss(
        component_logits: torch.Tensor,
        component_targets: torch.Tensor,
        row_weights: torch.Tensor,
        positive_weight: torch.Tensor,
        ordinal_targets: torch.Tensor,
        applicability: torch.Tensor,
        deployed_logits: torch.Tensor | None = None,
        target_tree_teacher_weight: float = 0.0,
        rule_consistency_weight: float = 0.05,
        distribution_weight: float = 0.10,
        munition_ids: torch.Tensor | None = None,
        rule_entry_ranking_weight: torch.Tensor | None = None,
        rule_conditional_l1_l2_ranking_weight:
        torch.Tensor | None = None,
        ranking_margin: float = 0.5,
        hard_negative_fraction: float = 0.10) -> torch.Tensor:
    """Dense component risk plus a low-weight damage-tree consistency term.

    The authoritative deployed output remains the direct ordinal head.
    Component targets are simulator labels used only to shape the shared
    representation.  Square-root-tempered positive weights are computed from
    the training split and sealed in the data contract.
    """
    if (
        component_logits.ndim != 3
        or component_logits.shape[1] != 2
    ):
        raise ValueError(
            "component_logits must have shape (N,2,C).")
    if component_targets.shape != component_logits.shape:
        raise ValueError(
            "component_targets must match component_logits.")
    if positive_weight.shape != component_logits.shape[1:]:
        raise ValueError(
            "positive_weight must have shape (2,C).")
    if tuple(ordinal_targets.shape) != (
            component_logits.shape[0], 4, 2):
        raise ValueError(
            "ordinal_targets must have shape (N,4,2).")
    if tuple(applicability.shape) != (
            component_logits.shape[0], 4, 2):
        raise ValueError(
            "applicability must have shape (N,4,2).")
    if rule_consistency_weight < 0.0:
        raise ValueError(
            "rule_consistency_weight must be non-negative.")
    if distribution_weight < 0.0:
        raise ValueError(
            "distribution_weight must be non-negative.")
    if target_tree_teacher_weight < 0.0:
        raise ValueError(
            "target_tree_teacher_weight must be non-negative.")
    if target_tree_teacher_weight > 0.0:
        if deployed_logits is None:
            raise ValueError(
                "deployed_logits are required for target-tree teaching.")
        if tuple(deployed_logits.shape) != (
                component_logits.shape[0], 4, 2):
            raise ValueError(
                "deployed_logits must have shape (N,4,2).")
    ranking_matrices = (
        rule_entry_ranking_weight,
        rule_conditional_l1_l2_ranking_weight,
    )
    for matrix in ranking_matrices:
        if matrix is not None and tuple(matrix.shape) != (4, 4):
            raise ValueError(
                "Component-rule ranking weights must have shape (4,4).")
        if matrix is not None and bool(
                (matrix < 0.0).any() or (matrix > 1.0).any()):
            raise ValueError(
                "Component-rule ranking weights must be in [0,1].")
    ranking_enabled = any(
        matrix is not None and bool((matrix > 0.0).any())
        for matrix in ranking_matrices
    )
    if ranking_enabled:
        if munition_ids is None:
            raise ValueError(
                "munition_ids are required for component-rule ranking.")
        if tuple(munition_ids.shape) != (
                component_logits.shape[0],):
            raise ValueError(
                "munition_ids must have shape (N,).")

    targets = component_targets.to(
        dtype=component_logits.dtype)
    positive_weight = positive_weight.to(
        dtype=component_logits.dtype)
    element_loss = F.binary_cross_entropy_with_logits(
        component_logits,
        targets,
        reduction="none",
        pos_weight=positive_weight,
    )
    per_sample = element_loss.mean(dim=(1, 2))

    rule_probability = None
    if rule_consistency_weight > 0.0 or ranking_enabled:
        mechanism_probability = torch.sigmoid(
            component_logits.float())
        combined_component_probability = (
            1.0
            - (1.0 - mechanism_probability[:, 0])
            * (1.0 - mechanism_probability[:, 1])
        )
        rule_probability = component_probabilities_to_ordinal(
            combined_component_probability).to(
                component_logits.dtype)
        if rule_consistency_weight > 0.0:
            valid = applicability.to(component_logits.dtype)
            rule_element = F.binary_cross_entropy(
                rule_probability.clamp(1e-6, 1.0 - 1e-6),
                ordinal_targets.to(component_logits.dtype),
                reduction="none",
            )
            rule_per_sample = (
                (rule_element * valid).sum(dim=(1, 2))
                / valid.sum(dim=(1, 2)).clamp_min(1.0)
            )
            per_sample = (
                per_sample
                + float(rule_consistency_weight)
                * rule_per_sample
            )

    weighted_loss = (per_sample * row_weights).mean()
    if target_tree_teacher_weight > 0.0:
        target_combined_component_probability = (
            1.0
            - (1.0 - targets[:, 0])
            * (1.0 - targets[:, 1])
        )
        target_tree_probability = (
            component_probabilities_to_ordinal(
                target_combined_component_probability.float())
            .to(component_logits.dtype)
        )
        valid = applicability.to(component_logits.dtype)
        teacher_element = F.binary_cross_entropy_with_logits(
            deployed_logits.to(component_logits.dtype),
            target_tree_probability,
            reduction="none",
        )
        teacher_per_sample = (
            (teacher_element * valid).sum(dim=(1, 2))
            / valid.sum(dim=(1, 2)).clamp_min(1.0)
        )
        weighted_loss = (
            weighted_loss
            + float(target_tree_teacher_weight)
            * (teacher_per_sample * row_weights).mean()
        )
    if distribution_weight > 0.0:
        predicted_mean = torch.sigmoid(
            component_logits.float()).mean(dim=0)
        target_mean = targets.float().mean(dim=0)
        distribution_loss = F.smooth_l1_loss(
            predicted_mean, target_mean)
        weighted_loss = (
            weighted_loss
            + float(distribution_weight) * distribution_loss
        )
    if ranking_enabled:
        rule_logits = torch.logit(
            rule_probability.clamp(1e-6, 1.0 - 1e-6)
        ).to(component_logits.dtype)
        ordinal_targets_for_ranking = ordinal_targets.to(
            component_logits.dtype)
        munition_ids = munition_ids.to(
            device=component_logits.device)
        row_weights_for_ranking = row_weights.to(
            dtype=component_logits.dtype)
        ranking_loss = component_logits.sum() * 0.0
        if rule_entry_ranking_weight is not None:
            entry_weights = rule_entry_ranking_weight.to(
                device=component_logits.device,
                dtype=component_logits.dtype,
            )
            for task_index in range(4):
                for munition_index in range(4):
                    current_weight = float(
                        entry_weights[
                            task_index, munition_index].item())
                    if current_weight <= 0.0:
                        continue
                    cell_mask = (
                        munition_ids == munition_index)
                    if not bool(cell_mask.any()):
                        continue
                    cell_targets = ordinal_targets_for_ranking[
                        cell_mask, task_index]
                    ranking_loss = ranking_loss + (
                        current_weight
                        * hard_negative_pairwise_ranking_loss(
                            rule_logits[
                                cell_mask, task_index, 0],
                            positive_mask=(
                                cell_targets[:, 0] > 0.5),
                            negative_mask=(
                                cell_targets[:, 0] < 0.5),
                            sample_weights=row_weights_for_ranking[
                                cell_mask],
                            margin=ranking_margin,
                            hard_negative_fraction=(
                                hard_negative_fraction),
                        )
                    )
        if rule_conditional_l1_l2_ranking_weight is not None:
            conditional_weights = (
                rule_conditional_l1_l2_ranking_weight.to(
                    device=component_logits.device,
                    dtype=component_logits.dtype,
                )
            )
            for task_index in range(4):
                for munition_index in range(4):
                    current_weight = float(
                        conditional_weights[
                            task_index, munition_index].item())
                    if current_weight <= 0.0:
                        continue
                    cell_mask = (
                        munition_ids == munition_index)
                    if not bool(cell_mask.any()):
                        continue
                    cell_targets = ordinal_targets_for_ranking[
                        cell_mask, task_index]
                    ranking_loss = ranking_loss + (
                        current_weight
                        * hard_negative_pairwise_ranking_loss(
                            rule_logits[
                                cell_mask, task_index, 1],
                            positive_mask=(
                                cell_targets[:, 1] > 0.5),
                            negative_mask=(
                                (cell_targets[:, 0] > 0.5)
                                & (cell_targets[:, 1] < 0.5)
                            ),
                            sample_weights=row_weights_for_ranking[
                                cell_mask],
                            margin=ranking_margin,
                            hard_negative_fraction=(
                                hard_negative_fraction),
                        )
                    )
        weighted_loss = weighted_loss + ranking_loss
    return weighted_loss


def _batch_auxiliary_targets(
        batch,
        mechanism_outputs_enabled: bool,
        component_outputs_enabled: bool):
    cursor = 11
    mechanism_targets = None
    component_targets = None
    if mechanism_outputs_enabled:
        mechanism_targets = batch[cursor]
        cursor += 1
    if component_outputs_enabled:
        component_targets = batch[cursor]
    return mechanism_targets, component_targets


def _forward_with_training_auxiliaries(
        model: nn.Module,
        features: torch.Tensor,
        munition_ids: torch.Tensor,
        mechanism_outputs_enabled: bool,
        component_outputs_enabled: bool):
    if mechanism_outputs_enabled:
        logits, fragment_logits, shock_logits = (
            model.forward_with_mechanisms(
                features, munition_ids))
        return (
            logits, fragment_logits, shock_logits, None)
    if component_outputs_enabled:
        logits, component_logits = (
            model.forward_with_components(
                features, munition_ids))
        return logits, None, None, component_logits
    return model(features, munition_ids), None, None, None


# 兼容旧引用名
UncertaintyOrdinalLoss = FocalUncertaintyOrdinalLoss


TASK_NAMES = ["K", "M", "F", "C"]
MUN_NAMES = ["Small", "Med-LM", "Med-RD", "Heavy"]
HEAD_NAMES = ["K1", "K2", "M1", "M2", "F1", "F2", "C1", "C2"]
CURRENT_THRESHOLD_SCHEMA = "v8_exact_l1_floor_constrained"
TARGET_CELL_MASK = torch.zeros(4, 4, dtype=torch.bool)
TARGET_CELL_MASK[0, 0] = True  # Small x K
TARGET_CELL_MASK[1, 0] = True  # Small x M
TARGET_CELL_MASK[3, 0] = True  # Small x C
DEFAULT_PER_MUN_MIN_POS = 30
CLASS1_FLOOR_MIN_POS = 100
CLASS1_FLOOR_RECALL = 85.0
CLASS1_FLOOR_MAX_ACCURACY_DROP = 0.005
GLOBAL_C0_MAX_FP_RATE = 0.025
RARE_CELL_CONFIG = {
    (0, 0): {
        "name": "SmallK1",
        "min_pos_l1": 15,
        "fixed_l2_threshold": 0.90,
        "max_fp_rate": 0.005,
        "recall_weight": 0.75,
        "objective": "max 0.75*recall + 0.25*F1 under Small x K0 FP <= 0.5%",
    },
    (3, 0): {
        "name": "Small_C",
        "max_fp_rate": GLOBAL_C0_MAX_FP_RATE,
        "recall_weight": 0.70,
        "objective": "maximize C1 recall/F1 under C0 FP <= 2.5%",
    },
    (3, 1): {
        "name": "Med-LM_C",
        "max_fp_rate": GLOBAL_C0_MAX_FP_RATE,
        "recall_weight": 0.70,
        "objective": "maximize C1 recall/F1 under C0 FP <= 2.5%",
    },
    (3, 2): {
        "name": "Med-RD_C",
        "max_fp_rate": GLOBAL_C0_MAX_FP_RATE,
        "recall_weight": 0.70,
        "objective": "maximize C1 recall/F1 under C0 FP <= 2.5%",
    },
    (3, 3): {
        "name": "Heavy_C",
        "max_fp_rate": GLOBAL_C0_MAX_FP_RATE,
        "recall_weight": 0.70,
        "objective": "maximize C1 recall/F1 under C0 FP <= 2.5%",
    },
}


def _task_munition_loss_matrix(
        loss_config: dict,
        key: str,
        device: torch.device,
        default: float = 1.0,
        minimum: float = 1.0) -> torch.Tensor:
    raw = loss_config.get(key, [[default] * 4 for _ in range(4)])
    matrix = torch.as_tensor(
        raw, dtype=torch.float32, device=device)
    if matrix.shape != (4, 4):
        raise ValueError(
            f"loss.{key} must be a 4x4 matrix "
            "(rows=K/M/F/C, columns=Small/Med-LM/Med-RD/Heavy).")
    if not torch.isfinite(matrix).all() or (matrix < minimum).any():
        raise ValueError(
            f"loss.{key} values must be finite and >= {minimum}.")
    return matrix


def _cell_l1_min_pos(task_idx: int, mun_id: int) -> int:
    return int(RARE_CELL_CONFIG.get((task_idx, mun_id), {}).get(
        "min_pos_l1", DEFAULT_PER_MUN_MIN_POS))


def _cell_l1_search_params(task_idx: int, mun_id: int):
    cfg = RARE_CELL_CONFIG.get((task_idx, mun_id), {})
    if cfg:
        return cfg.get("max_fp_rate"), float(cfg.get("recall_weight", 0.7))
    return None, 0.7


def _binary_f1(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction.long()
    target = target.long()
    tp = int(((prediction == 1) & (target == 1)).sum().item())
    fp = int(((prediction == 1) & (target == 0)).sum().item())
    fn = int(((prediction == 0) & (target == 1)).sum().item())
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2.0 * precision * recall / max(precision + recall, 1e-9)


def _search_l1_threshold(
        probabilities: torch.Tensor,
        targets: torch.Tensor,
        threshold_grid: torch.Tensor,
        max_fp_rate: float | None = None,
        recall_weight: float = 0.7,
        minimum_recall: float = CLASS1_FLOOR_RECALL / 100.0,
        maximum_accuracy_drop: float | None = (
            CLASS1_FLOOR_MAX_ACCURACY_DROP),
        minimum_accuracy: float | None = None,
        minimum_negative_recall: float | None = None):
    """Calibrate one binary threshold under optional safety/recall constraints.

    The recall floor is prioritized only when it is attainable without
    violating ``max_fp_rate``.  Otherwise calibration falls back to the best
    safety-feasible candidate and records a genuine capability shortfall
    instead of silently breaking the false-positive contract.
    """
    target = targets.long()
    n_pos = int(target.sum().item())
    n_neg = int(target.numel() - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.0, 0.0, 0.0, 0.0

    best_any = None
    best_guarded = None
    guarded_floor_candidates = []
    goal_candidates = []
    for threshold in threshold_grid:
        threshold_value = float(threshold)
        prediction = (probabilities >= threshold).long()
        tp = int(((prediction == 1) & (target == 1)).sum().item())
        fp = int(((prediction == 1) & (target == 0)).sum().item())
        fn = int(((prediction == 0) & (target == 1)).sum().item())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        accuracy = (tp + n_neg - fp) / max(target.numel(), 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
        fp_rate = fp / max(n_neg, 1)
        objective = recall_weight * recall + (1.0 - recall_weight) * f1
        candidate = (
            threshold_value, objective, precision, recall, fp_rate)
        candidate_key = (objective, -fp_rate, threshold_value)
        if best_any is None or candidate_key > best_any[0]:
            best_any = (candidate_key, candidate)
        if max_fp_rate is None or fp_rate <= max_fp_rate:
            if best_guarded is None or candidate_key > best_guarded[0]:
                best_guarded = (candidate_key, candidate)
            if recall >= minimum_recall:
                guarded_floor_candidates.append(
                    (candidate_key, candidate, accuracy))
                goal_margins = []
                if minimum_accuracy is not None:
                    goal_margins.append(
                        accuracy - float(minimum_accuracy))
                if minimum_negative_recall is not None:
                    goal_margins.append(
                        (1.0 - fp_rate)
                        - float(minimum_negative_recall))
                if (
                    goal_margins
                    and min(goal_margins) >= -1e-12
                ):
                    goal_key = (
                        min(goal_margins), accuracy, objective,
                        -fp_rate, threshold_value,
                    )
                    goal_candidates.append((goal_key, candidate))
    baseline = best_guarded if best_guarded is not None else best_any
    selected = baseline
    if goal_candidates:
        selected = max(goal_candidates, key=lambda item: item[0])
    elif guarded_floor_candidates:
        baseline_prediction = (
            probabilities >= float(baseline[1][0])).long()
        baseline_accuracy = float(
            (baseline_prediction == target).float().mean().item())
        minimum_accuracy = (
            -float("inf")
            if maximum_accuracy_drop is None
            else baseline_accuracy - maximum_accuracy_drop
        )
        admissible = [
            candidate for candidate in guarded_floor_candidates
            if candidate[2] >= minimum_accuracy - 1e-12
        ]
        if admissible:
            selected = max(admissible, key=lambda item: item[0])
    return selected[1]


def _search_joint_ordinal_thresholds(
        p1_vec: torch.Tensor,
        p2_vec: torch.Tensor,
        t1_vec: torch.Tensor,
        t2_vec: torch.Tensor,
        threshold_grid: torch.Tensor,
        alpha: float = 0.80,
        thr2_min_slack: float = 0.10,
        max_l0_fp_rate: float | None = None,
        minimum_exact_l1_recall: float = (
            CLASS1_FLOOR_RECALL / 100.0),
        maximum_accuracy_drop: float | None = (
            CLASS1_FLOOR_MAX_ACCURACY_DROP),
        minimum_three_class_accuracy: float | None = None,
        minimum_class_diagonal_recall: float | None = None):
    """Shared joint ordinal calibration used during and after training.

    The optional ``max_l0_fp_rate`` is a hard feasibility constraint on
    predicting any positive level for true class-0 rows.  Candidate ordering is
    deterministic: objective, then lower FPR, then higher thresholds.
    """
    p2_clamped = torch.minimum(p2_vec, p1_vec)
    t1_long = t1_vec.long()
    t2_long = t2_vec.long()
    true_level = t1_long + t2_long
    n_l0 = int((true_level == 0).sum().item())
    n1_pos = int(t1_long.sum().item())
    n2_pos = int(t2_long.sum().item())

    if (n1_pos == 0 or n1_pos == t1_long.numel()
            or n2_pos == 0 or n2_pos == t2_long.numel()):
        threshold1, *_ = _search_l1_threshold(
            p1_vec, t1_vec, threshold_grid,
            max_fp_rate=max_l0_fp_rate, recall_weight=0.7,
            minimum_recall=minimum_exact_l1_recall,
            maximum_accuracy_drop=maximum_accuracy_drop,
            minimum_accuracy=minimum_three_class_accuracy,
            minimum_negative_recall=(
                minimum_class_diagonal_recall))
        threshold2, *_ = _search_l1_threshold(
            p2_clamped, t2_vec, threshold_grid)
        threshold2 = max(threshold2, threshold1 - thr2_min_slack)
        pass1 = (p1_vec >= threshold1).long()
        pass2 = (
            (p1_vec >= threshold1) & (p2_clamped >= threshold2)
        ).long()
        pred_level = pass1 + pass2
        acc3 = float((pred_level == true_level).float().mean().item())
        f1_h1 = _binary_f1(pass1, t1_long)
        f1_h2 = _binary_f1(pass2, t2_long)
        score = alpha * acc3 + (1.0 - alpha) * 0.5 * (f1_h1 + f1_h2)
        return threshold1, threshold2, score, acc3, f1_h1, f1_h2

    best_any = None
    best_guarded = None
    guarded_floor_candidates = []
    goal_candidates = []
    true_class1 = (true_level == 1)
    n_class1 = int(true_class1.sum().item())
    true_class2 = (true_level == 2)
    n_class2 = int(true_class2.sum().item())
    for threshold1_tensor in threshold_grid:
        threshold1 = float(threshold1_tensor)
        pass1_mask = p1_vec >= threshold1_tensor
        pass1_long = pass1_mask.long()
        f1_h1 = _binary_f1(pass1_long, t1_long)
        fp_rate = (
            float((pass1_mask & (true_level == 0)).sum().item())
            / max(n_l0, 1)
        )
        for threshold2_tensor in threshold_grid:
            threshold2 = float(threshold2_tensor)
            if threshold2 < threshold1 - thr2_min_slack:
                continue
            pass2_long = (
                pass1_mask & (p2_clamped >= threshold2_tensor)
            ).long()
            f1_h2 = _binary_f1(pass2_long, t2_long)
            pred_level = pass1_long + pass2_long
            exact_l1_recall = (
                float(
                    ((pred_level == 1) & true_class1).sum().item())
                / max(n_class1, 1)
            )
            acc3 = float((pred_level == true_level).float().mean().item())
            class0_recall = 1.0 - fp_rate
            exact_l2_recall = (
                float(
                    ((pred_level == 2) & true_class2).sum().item())
                / max(n_class2, 1)
            )
            score = (
                alpha * acc3
                + (1.0 - alpha) * 0.5 * (f1_h1 + f1_h2)
            )
            candidate = (
                threshold1, threshold2, score, acc3, f1_h1, f1_h2)
            candidate_key = (
                score, -fp_rate, threshold1, threshold2)
            if best_any is None or candidate_key > best_any[0]:
                best_any = (candidate_key, candidate)
            if max_l0_fp_rate is None or fp_rate <= max_l0_fp_rate:
                if best_guarded is None or candidate_key > best_guarded[0]:
                    best_guarded = (candidate_key, candidate)
                if (
                    n_class1 > 0
                    and exact_l1_recall >= minimum_exact_l1_recall
                ):
                    guarded_floor_candidates.append(
                        (candidate_key, candidate))
                    goal_margins = []
                    if minimum_three_class_accuracy is not None:
                        goal_margins.append(
                            acc3 - float(minimum_three_class_accuracy))
                    if minimum_class_diagonal_recall is not None:
                        recall_floor = float(
                            minimum_class_diagonal_recall)
                        if n_l0 > 0:
                            goal_margins.append(
                                class0_recall - recall_floor)
                        if n_class1 > 0:
                            goal_margins.append(
                                exact_l1_recall - recall_floor)
                        if n_class2 > 0:
                            goal_margins.append(
                                exact_l2_recall - recall_floor)
                    if (
                        goal_margins
                        and min(goal_margins) >= -1e-12
                    ):
                        goal_key = (
                            min(goal_margins), acc3, score,
                            -fp_rate, threshold1, threshold2,
                        )
                        goal_candidates.append((goal_key, candidate))
    baseline = best_guarded if best_guarded is not None else best_any
    selected = baseline
    if goal_candidates:
        selected = max(goal_candidates, key=lambda item: item[0])
    elif guarded_floor_candidates:
        baseline_accuracy = float(baseline[1][3])
        minimum_accuracy = (
            -float("inf")
            if maximum_accuracy_drop is None
            else baseline_accuracy - maximum_accuracy_drop
        )
        admissible = [
            candidate for candidate in guarded_floor_candidates
            if float(candidate[1][3]) >= minimum_accuracy - 1e-12
        ]
        if admissible:
            selected = max(admissible, key=lambda item: item[0])
    if selected is None:
        raise RuntimeError("Threshold grid produced no valid ordinal candidate.")
    return selected[1]


def _is_applicable(task_idx: int, mun_id: int, level_idx: int) -> bool:
    return bool(DEFAULT_ORDINAL_APPLICABILITY[mun_id][task_idx][level_idx])


def _rare_cell_meta(task_idx: int, mun_id: int, n_samples: int,
                    n_pos_1: int, n_pos_2: int, mode: str,
                    thr1: float, thr2: float) -> dict:
    cfg = RARE_CELL_CONFIG.get((task_idx, mun_id), {})
    return {
        "name": cfg.get(
            "name", f"{MUN_NAMES[mun_id]}_{TASK_NAMES[task_idx]}"),
        "task": TASK_NAMES[task_idx],
        "munition": MUN_NAMES[mun_id],
        "n_samples": int(n_samples),
        "n_pos_L1": int(n_pos_1),
        "n_pos_L2": int(n_pos_2),
        "min_pos_L1": int(cfg.get(
            "min_pos_l1", DEFAULT_PER_MUN_MIN_POS)),
        "min_pos_L2": int(DEFAULT_PER_MUN_MIN_POS),
        "applicable_L1": _is_applicable(task_idx, mun_id, 0),
        "applicable_L2": _is_applicable(task_idx, mun_id, 1),
        "mode": mode,
        "threshold_L1": float(thr1),
        "threshold_L2": float(thr2),
        "max_fp_rate": (
            float(cfg["max_fp_rate"]) if "max_fp_rate" in cfg else None),
        "recall_weight": float(cfg.get("recall_weight", 0.7)),
        "objective": cfg.get(
            "objective",
            "cellwise joint calibration when support is sufficient; "
            "otherwise inherit the global validation threshold"),
    }


def _build_cell_diag(true_level: torch.Tensor, pred_level: torch.Tensor,
                     probs_all: torch.Tensor, mids_all: torch.Tensor,
                     task_idx: int, mun_id: int) -> dict:
    mask = (mids_all == mun_id)
    diag = {
        "n": int(mask.sum().item()),
        "acc": 0.0,
        "cm": torch.zeros(3, 3, dtype=torch.long),
        "true_counts": torch.zeros(3, dtype=torch.long),
        "pred_counts": torch.zeros(3, dtype=torch.long),
        "l0_to_l1": 0.0,
        "l0_to_l2": 0.0,
        "l1_to_l0": 0.0,
        "l1_to_l2": 0.0,
        "l2_to_l1": 0.0,
        "prob_by_true": {},
    }
    if diag["n"] <= 0:
        return diag

    cell_true = true_level[mask, task_idx].long()
    cell_pred = pred_level[mask, task_idx].long()
    cm = torch.bincount(cell_true * 3 + cell_pred, minlength=9).reshape(3, 3)
    true_counts = cm.sum(dim=1)
    pred_counts = cm.sum(dim=0)

    def _cell_rate(t_cls: int, p_cls: int) -> float:
        denom = int(true_counts[t_cls].item())
        if denom <= 0:
            return 0.0
        return float(cm[t_cls, p_cls].item()) / denom * 100.0

    diag.update({
        "acc": float((cell_pred == cell_true).float().mean().item()) * 100.0,
        "cm": cm,
        "true_counts": true_counts,
        "pred_counts": pred_counts,
        "l0_to_l1": _cell_rate(0, 1),
        "l0_to_l2": _cell_rate(0, 2),
        "l1_to_l0": _cell_rate(1, 0),
        "l1_to_l2": _cell_rate(1, 2),
        "l2_to_l1": _cell_rate(2, 1),
    })
    p1_cell = probs_all[mask, task_idx, 0]
    p2_cell = torch.minimum(probs_all[mask, task_idx, 1], p1_cell)
    prob_by_true = {}
    for lv in range(3):
        lv_mask = (cell_true == lv)
        if int(lv_mask.sum().item()) == 0:
            continue
        p1_lv = p1_cell[lv_mask]
        p2_lv = p2_cell[lv_mask]
        prob_by_true[lv] = {
            "p1_mean": float(p1_lv.mean().item()),
            "p1_p50": float(torch.quantile(p1_lv, 0.50).item()),
            "p1_p90": float(torch.quantile(p1_lv, 0.90).item()),
            "p2_mean": float(p2_lv.mean().item()),
            "p2_p50": float(torch.quantile(p2_lv, 0.50).item()),
            "p2_p90": float(torch.quantile(p2_lv, 0.90).item()),
        }
    diag["prob_by_true"] = prob_by_true
    return diag


def _collect_low_cls1_cells(cls1_recall_per_mun: torch.Tensor,
                            cls1_count_per_mun: torch.Tensor,
                            mask: torch.Tensor,
                            floor: float = CLASS1_FLOOR_RECALL,
                            min_pos: int = CLASS1_FLOOR_MIN_POS) -> list[dict]:
    lows = []
    for task_idx in range(4):
        for mun_id in range(4):
            if not bool(mask[task_idx, mun_id].item()):
                continue
            n_pos = int(cls1_count_per_mun[task_idx, mun_id].item())
            rec = float(cls1_recall_per_mun[task_idx, mun_id].item())
            if n_pos >= min_pos and rec < floor:
                lows.append({
                    "task": TASK_NAMES[task_idx],
                    "munition": MUN_NAMES[mun_id],
                    "n_pos": n_pos,
                    "recall": rec,
                    "floor": float(floor),
                    "deficit": float(floor - rec),
                })
    lows.sort(key=lambda x: x["deficit"], reverse=True)
    return lows


def _minimum_supported_class1_recall(
        cls1_recall_per_mun: torch.Tensor,
        cls1_count_per_mun: torch.Tensor,
        minimum_support: int = GOAL_MIN_CLASS_SUPPORT) -> float:
    """Return the worst validation class-1 recall with adequate evidence.

    Recall tensors are expressed in percent.  Cells below the production
    evidence contract are deliberately excluded here; the separate Stage-0
    support gate must fail such an artifact rather than assigning a misleading
    zero-recall training penalty to an unsupported cell.
    """
    supported = cls1_count_per_mun >= int(minimum_support)
    if not bool(supported.any().item()):
        return 0.0
    return float(cls1_recall_per_mun[supported].min().item()) / 100.0


def _minimum_supported_diagonal_recall(
        predicted_levels: torch.Tensor,
        true_levels: torch.Tensor,
        munition_ids: torch.Tensor,
        minimum_support: int = GOAL_MIN_CLASS_SUPPORT) -> float:
    """Return the worst supported applicable confusion diagonal recall."""
    recalls = []
    predicted = predicted_levels.detach().cpu().long()
    targets = true_levels.detach().cpu().long()
    munition = munition_ids.detach().cpu().long()
    for munition_id in range(4):
        munition_mask = munition == munition_id
        for task_id in range(4):
            for level in range(3):
                applicable = (
                    True if level == 0 else bool(
                        DEFAULT_ORDINAL_APPLICABILITY[
                            munition_id][task_id][level - 1])
                )
                if not applicable:
                    continue
                level_mask = (
                    munition_mask & (targets[:, task_id] == level))
                support = int(level_mask.sum().item())
                if support < int(minimum_support):
                    continue
                correct = int((
                    predicted[level_mask, task_id] == level
                ).sum().item())
                recalls.append(correct / support)
    return min(recalls) if recalls else 0.0


def _goal_candidate_sort_key(candidate: dict) -> tuple:
    """Lexicographic validation-only key aligned with the thesis contract."""
    if (
        "min_cell_acc_3class" not in candidate
        or "min_supported_class1_recall" not in candidate
    ):
        return (0, -float("inf"), -float("inf"),
                float(candidate.get("selection_score", -float("inf"))))
    minimum_accuracy = float(candidate["min_cell_acc_3class"])
    minimum_l1_recall = float(candidate["min_supported_class1_recall"])
    minimum_diagonal_recall = float(candidate.get(
        "min_supported_class_diagonal_recall",
        minimum_l1_recall,
    ))
    accuracy_margin = (
        minimum_accuracy
        - GOAL_MIN_CELL_3CLASS_ACCURACY_PERCENT / 100.0)
    recall_margin = (
        minimum_diagonal_recall
        - GOAL_MIN_CLASS_DIAGONAL_RECALL_PERCENT / 100.0)
    small_k0_fp_rate = float(candidate.get("small_k0_fp_rate", 0.0))
    global_c0_fp_rate = float(candidate.get("c0_fp_rate", 0.0))
    safety_margin = min(
        0.005 - small_k0_fp_rate,
        GLOBAL_C0_MAX_FP_RATE - global_c0_fp_rate,
    )
    # Safety is a hard feasibility gate, not a reward for driving an already
    # safe false-positive rate ever lower at the expense of accuracy/recall.
    worst_margin = min(accuracy_margin, recall_margin)
    total_deficit = (
        max(0.0, -accuracy_margin)
        + max(0.0, -recall_margin)
        + max(0.0, -safety_margin)
    )
    passed = int(
        accuracy_margin >= 0.0
        and recall_margin >= 0.0
        and safety_margin >= 0.0
    )
    return (
        passed,
        worst_margin,
        -total_deficit,
        float(candidate.get("selection_score", -float("inf"))),
    )


def _safe_torch_save(obj, path: str, retries: int = 4) -> None:
    """原子写 + 短重试，吸收 Windows 上对同名 .pth 的瞬时文件锁
    （torch.load 残留句柄 / 杀软扫描 / 文件系统缓存）。"""
    import gc as _gc
    import time as _time
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = path + ".tmp"
    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    last_err: Optional[BaseException] = None
    for attempt in range(retries):
        try:
            torch.save(obj, tmp_path)
            if os.path.exists(path):
                try:
                    os.chmod(path, 0o666)
                except OSError:
                    pass
            os.replace(tmp_path, path)
            return
        except (RuntimeError, OSError, PermissionError) as e:
            last_err = e
            _gc.collect()
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            _time.sleep(0.5 * (attempt + 1))
    raise last_err if last_err is not None else RuntimeError(
        f"_safe_torch_save failed for {path}")


def _write_json_atomic(path: str, payload: dict) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def _validation_report_from_metrics(
        metrics: dict,
        model_variant: str,
        dataset_sha256: str,
        model_sha256: str,
        threshold_sha256: str) -> dict:
    recall = metrics["cls1_recall_per_mun"]
    support = metrics["cls1_count_per_mun"]
    accuracy = metrics["acc_per_munition"]
    configured_class1_floor_percent = 100.0 * float(metrics.get(
        "minimum_exact_l1_recall",
        CLASS1_FLOOR_RECALL / 100.0,
    ))
    cells = {}
    failures = []
    for munition_id, munition in enumerate(MUN_NAMES):
        cells[munition] = {}
        for task_id, task in enumerate(TASK_NAMES):
            cell = {
                "samples": int(
                    metrics["samples_per_munition"][munition_id].item()),
                "three_class_accuracy_percent": float(
                    accuracy[task_id, munition_id].item()),
                "class1_support": int(
                    support[task_id, munition_id].item()),
                "class1_recall_percent": float(
                    recall[task_id, munition_id].item()),
            }
            cells[munition][task] = cell
            if (
                cell["class1_support"] >= CLASS1_FLOOR_MIN_POS
                and cell["class1_recall_percent"]
                < configured_class1_floor_percent
            ):
                failures.append(
                    f"{munition}/{task} class-1 recall "
                    f"{cell['class1_recall_percent']:.2f}% < "
                    f"{configured_class1_floor_percent:.2f}% "
                    f"(n={cell['class1_support']})")
    small_k0_fp = float(metrics["small_k0_fp_rate"]) * 100.0
    global_c0_fp = float(metrics["c0_fp_rate"]) * 100.0
    if small_k0_fp > 0.5:
        failures.insert(
            0, f"Small/K level-0 FP {small_k0_fp:.3f}% > 0.500%")
    if global_c0_fp > GLOBAL_C0_MAX_FP_RATE * 100.0:
        failures.insert(
            0, f"global C level-0 FP {global_c0_fp:.3f}% > "
            f"{GLOBAL_C0_MAX_FP_RATE * 100.0:.3f}%")

    # The user-facing objective is stronger than the historical performance
    # gate: every munition/task cell must reach at least 94% three-class accuracy and
    # every applicable confusion-matrix diagonal must reach 90% recall.  A
    # rare class with fewer than 100 validation examples is not silently
    # treated as passing; it is explicitly marked insufficient evidence.
    goal_failures = []
    if small_k0_fp > 0.5:
        goal_failures.append(
            f"Small/K level-0 FP {small_k0_fp:.3f}% > 0.500%")
    if global_c0_fp > GLOBAL_C0_MAX_FP_RATE * 100.0:
        goal_failures.append(
            f"global C level-0 FP {global_c0_fp:.3f}% > "
            f"{GLOBAL_C0_MAX_FP_RATE * 100.0:.3f}%")
    evidence_failures = []
    goal_cells = {}
    predicted_levels = metrics.get("pred_level")
    true_levels = metrics.get("true_level")
    munition_ids = metrics.get("munition_ids")
    has_confusion_evidence = (
        isinstance(predicted_levels, torch.Tensor)
        and isinstance(true_levels, torch.Tensor)
        and isinstance(munition_ids, torch.Tensor)
        and predicted_levels.shape == true_levels.shape
        and predicted_levels.ndim == 2
        and predicted_levels.shape[1] == len(TASK_NAMES)
        and munition_ids.ndim == 1
        and munition_ids.shape[0] == predicted_levels.shape[0]
    )
    if not has_confusion_evidence:
        evidence_failures.append(
            "full validation predictions or munition ids are unavailable")
    else:
        predicted_levels = predicted_levels.detach().cpu().long()
        true_levels = true_levels.detach().cpu().long()
        munition_ids = munition_ids.detach().cpu().long()
        for munition_id, munition in enumerate(MUN_NAMES):
            goal_cells[munition] = {}
            munition_mask = munition_ids == munition_id
            for task_id, task in enumerate(TASK_NAMES):
                cell_true = true_levels[munition_mask, task_id]
                cell_pred = predicted_levels[munition_mask, task_id]
                confusion = torch.bincount(
                    cell_true * 3 + cell_pred, minlength=9
                ).reshape(3, 3)
                class_support = confusion.sum(dim=1)
                class_recall = []
                class_status = []
                for level in range(3):
                    support_value = int(class_support[level].item())
                    applicable = (
                        True if level == 0
                        else bool(
                            DEFAULT_ORDINAL_APPLICABILITY[
                                munition_id][task_id][level - 1])
                    )
                    recall_value = (
                        None if support_value == 0
                        else float(
                            confusion[level, level].item()
                            / support_value * 100.0)
                    )
                    if not applicable:
                        status = "NOT_APPLICABLE"
                    elif support_value < GOAL_MIN_CLASS_SUPPORT:
                        status = "INSUFFICIENT_EVIDENCE"
                        evidence_failures.append(
                            f"{munition}/{task}/L{level} support "
                            f"{support_value} < {GOAL_MIN_CLASS_SUPPORT}")
                    elif (
                        recall_value
                        < GOAL_MIN_CLASS_DIAGONAL_RECALL_PERCENT
                    ):
                        status = "FAIL"
                        goal_failures.append(
                            f"{munition}/{task}/L{level} diagonal recall "
                            f"{recall_value:.2f}% < "
                            f"{GOAL_MIN_CLASS_DIAGONAL_RECALL_PERCENT:.2f}% "
                            f"(n={support_value})")
                    else:
                        status = "PASS"
                    class_recall.append(recall_value)
                    class_status.append(status)

                accuracy_value = float(
                    accuracy[task_id, munition_id].item())
                if (
                    accuracy_value
                    < GOAL_MIN_CELL_3CLASS_ACCURACY_PERCENT
                ):
                    goal_failures.append(
                        f"{munition}/{task} three-class accuracy "
                        f"{accuracy_value:.2f}% < "
                        f"{GOAL_MIN_CELL_3CLASS_ACCURACY_PERCENT:.2f}%")
                goal_cells[munition][task] = {
                    "confusion_matrix_rows_true_columns_predicted": (
                        confusion.tolist()),
                    "class_support": class_support.tolist(),
                    "class_diagonal_recall_percent": class_recall,
                    "class_status": class_status,
                    "three_class_accuracy_percent": accuracy_value,
                }

    goal_passed = not goal_failures and not evidence_failures
    return {
        "schema": "stage0_nn_validation_selection_v2",
        "status": "COMPLETE",
        "split": "validation",
        "test_labels_used": False,
        "model_variant": model_variant,
        "selected_epoch": int(metrics["epoch"]),
        "selection_score": float(metrics["selection_score"]),
        "validation_rows": int(
            metrics["samples_per_munition"].sum().item()),
        "average_3class_accuracy_percent": (
            float(metrics["acc3_mean"]) * 100.0),
        "task_3class_accuracy_percent": {
            task: float(metrics["acc_per_task"][task_id])
            for task_id, task in enumerate(TASK_NAMES)
        },
        "small_k0_false_positive_percent": small_k0_fp,
        "global_c0_false_positive_percent": global_c0_fp,
        "cell_metrics": cells,
        "targeted_probability_diagnostics": metrics.get(
            "targeted_probability_diagnostics", {}),
        "confidence_resolved_diagnostics": metrics.get(
            "confidence_resolved_diagnostics", {}),
        "performance_gate": {
            "passed": not failures,
            "failure_count": len(failures),
            "failures": failures,
        },
        "goal_performance_gate": {
            "passed": goal_passed,
            "status": (
                "PASS" if goal_passed
                else "FAIL" if goal_failures
                else "INSUFFICIENT_EVIDENCE"
            ),
            "requirements": {
                "minimum_cell_3class_accuracy_percent": (
                    GOAL_MIN_CELL_3CLASS_ACCURACY_PERCENT),
                "minimum_applicable_class_diagonal_recall_percent": (
                    GOAL_MIN_CLASS_DIAGONAL_RECALL_PERCENT),
                "minimum_class_support": GOAL_MIN_CLASS_SUPPORT,
                "small_k0_max_false_positive_percent": 0.5,
                "global_c0_max_false_positive_percent": (
                    GLOBAL_C0_MAX_FP_RATE * 100.0),
            },
            "metric_failure_count": len(goal_failures),
            "metric_failures": goal_failures,
            "evidence_failure_count": len(evidence_failures),
            "evidence_failures": evidence_failures,
            "cell_confusion_metrics": goal_cells,
        },
        "artifact_identity": {
            "dataset_sha256": dataset_sha256,
            "model_sha256": model_sha256,
            "threshold_sha256": threshold_sha256,
        },
    }


def _targeted_probability_diagnostics(
        probabilities: torch.Tensor,
        true_levels: torch.Tensor,
        munition_ids: torch.Tensor) -> dict:
    """Validation-only ranking diagnostics for pre-registered weak cells."""
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
    )

    probabilities_np = probabilities.detach().cpu().numpy()
    levels_np = true_levels.detach().cpu().numpy()
    munition_np = munition_ids.detach().cpu().numpy()
    output = {}
    entry_targets = (
        ("Small/K", 0, 0, 0.005),
        ("Med-LM/K", 0, 1, 0.025),
        ("Med-RD/K", 0, 2, 0.025),
        ("Heavy/K", 0, 3, 0.025),
        ("Small/C", 3, 0, 0.025),
        ("Med-LM/C", 3, 1, 0.025),
        ("Med-RD/C", 3, 2, 0.025),
    )
    for name, task_id, munition_id, max_fpr in entry_targets:
        mask = munition_np == munition_id
        target = (levels_np[mask, task_id] >= 1).astype("int64")
        score = probabilities_np[mask, task_id, 0]
        if len(set(target.tolist())) < 2:
            output[name] = {
                "entry_auc": None,
                "entry_standardized_partial_auc": None,
                "entry_average_precision": None,
                "maximum_false_positive_rate": max_fpr,
            }
            continue
        output[name] = {
            "entry_auc": float(roc_auc_score(target, score)),
            "entry_standardized_partial_auc": float(
                roc_auc_score(target, score, max_fpr=max_fpr)),
            "entry_average_precision": float(
                average_precision_score(target, score)),
            "maximum_false_positive_rate": max_fpr,
            "positive_support": int(target.sum()),
            "negative_support": int((target == 0).sum()),
        }

    damaged = (
        (munition_np == 2)
        & (levels_np[:, 1] >= 1)
    )
    conditional_target = (
        levels_np[damaged, 1] >= 2).astype("int64")
    conditional_score = probabilities_np[damaged, 1, 1]
    if len(set(conditional_target.tolist())) < 2:
        output["Med-RD/M_L1_vs_L2"] = {
            "conditional_auc": None,
            "conditional_average_precision": None,
        }
    else:
        output["Med-RD/M_L1_vs_L2"] = {
            "conditional_auc": float(roc_auc_score(
                conditional_target, conditional_score)),
            "conditional_average_precision": float(
                average_precision_score(
                    conditional_target, conditional_score)),
            "level1_support": int(
                (conditional_target == 0).sum()),
            "level2_support": int(conditional_target.sum()),
        }
    return output


def _confidence_resolved_diagnostics(
        probabilities: torch.Tensor,
        soft_targets: torch.Tensor,
        label_confidence: torch.Tensor,
        true_levels: torch.Tensor,
        predicted_levels: torch.Tensor,
        munition_ids: torch.Tensor,
        uncertainty_scale: float = 0.10,
        z_value: float = 1.96) -> dict:
    """Report decisions whose MC uncertainty resolves the 0.5 boundaries.

    ``label_confidence`` follows
    ``1 / (1 + (standard_error / uncertainty_scale)^2)``.  Inverting that
    relation yields the MC standard error used to form an approximate 95%
    interval around each cumulative probability.  Ambiguous boundary rows
    remain in proper-score metrics but are not mislabeled as high-confidence
    hard decisions.
    """
    confidence = torch.clamp(
        label_confidence.float(), min=1e-6, max=1.0)
    standard_error = float(uncertainty_scale) * torch.sqrt(
        torch.clamp(1.0 / confidence - 1.0, min=0.0))
    lower = soft_targets.float() - float(z_value) * standard_error
    upper = soft_targets.float() + float(z_value) * standard_error
    cells = {}
    for munition_id, munition_name in enumerate(MUN_NAMES):
        cells[munition_name] = {}
        munition_mask = munition_ids == munition_id
        for task_id, task_name in enumerate(TASK_NAMES):
            true_task = true_levels[:, task_id]
            predicted_task = predicted_levels[:, task_id]
            exact_l1 = munition_mask & (true_task == 1)
            resolved_l1 = (
                exact_l1
                & (lower[:, task_id, 0] >= 0.5)
                & (upper[:, task_id, 1] < 0.5)
            )
            l0 = munition_mask & (true_task == 0)
            resolved_l0 = (
                l0 & (upper[:, task_id, 0] < 0.5)
            )
            l2 = munition_mask & (true_task == 2)
            resolved_l2 = (
                l2 & (lower[:, task_id, 1] >= 0.5)
            )

            def _class_metric(full_mask, resolved_mask, class_id):
                full_support = int(full_mask.sum().item())
                resolved_support = int(resolved_mask.sum().item())
                recall = (
                    float(
                        (predicted_task[resolved_mask] == class_id)
                        .float().mean().item()) * 100.0
                    if resolved_support > 0 else None
                )
                return {
                    "full_support": full_support,
                    "resolved_support": resolved_support,
                    "resolved_fraction_percent": (
                        100.0 * resolved_support / full_support
                        if full_support > 0 else None
                    ),
                    "resolved_recall_percent": recall,
                }

            cell_mask = munition_mask
            soft_cell = soft_targets[cell_mask, task_id]
            probability_cell = probabilities[cell_mask, task_id]
            soft_brier = (
                float(torch.mean(
                    torch.square(probability_cell - soft_cell)).item())
                if int(cell_mask.sum().item()) > 0 else None
            )
            cells[munition_name][task_name] = {
                "L0": _class_metric(l0, resolved_l0, 0),
                "L1": _class_metric(exact_l1, resolved_l1, 1),
                "L2": _class_metric(l2, resolved_l2, 2),
                "soft_target_brier": soft_brier,
            }
    return {
        "method": "approximate_mc_95pct_interval",
        "uncertainty_scale": float(uncertainty_scale),
        "z_value": float(z_value),
        "boundary": 0.5,
        "cells": cells,
    }


def _clone_state_dict_cpu(model: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _average_state_dicts(state_dicts: list[dict]) -> dict:
    if not state_dicts:
        raise ValueError("state_dicts must not be empty")
    averaged = {}
    for key in state_dicts[0]:
        vals = [sd[key] for sd in state_dicts]
        first = vals[0]
        if torch.is_floating_point(first):
            avg = torch.stack([v.float() for v in vals], dim=0).mean(dim=0)
            averaged[key] = avg.to(dtype=first.dtype)
        else:
            averaged[key] = first.clone()
    return averaged


def _insert_topk_candidate(topk: list[dict], candidate: dict, k: int = 5) -> list[dict]:
    topk = topk + [candidate]
    topk.sort(key=_goal_candidate_sort_key, reverse=True)
    return topk[:k]


def _threshold_json_from_metrics(metrics: dict, model_variant: str,
                                 raw_best_epoch: int = None,
                                 soup_epochs: list[int] = None) -> dict:
    best_thr_matrix = metrics["best_thr_matrix"]
    best_thr_matrix_perm = metrics["best_thr_matrix_perm"]
    per_mun_dict = {}
    for head in HEAD_NAMES:
        i = TASK_NAMES.index(head[0])
        j = int(head[1]) - 1
        per_mun_dict[head] = {
            str(m): float(best_thr_matrix_perm[i, m, j].item())
            for m in range(4)
        }
    return {
        "K1": float(best_thr_matrix[0, 0].item()),
        "K2": float(best_thr_matrix[0, 1].item()),
        "M1": float(best_thr_matrix[1, 0].item()),
        "M2": float(best_thr_matrix[1, 1].item()),
        "F1": float(best_thr_matrix[2, 0].item()),
        "F2": float(best_thr_matrix[2, 1].item()),
        "C1": float(best_thr_matrix[3, 0].item()),
        "C2": float(best_thr_matrix[3, 1].item()),
        "per_munition": per_mun_dict,
        "_schema": CURRENT_THRESHOLD_SCHEMA,
        "_note": "Checkpoint-aligned thresholds saved at the final selected model variant.",
        "_calibration_policy": {
            "joint_alpha": 0.80,
            "threshold_grid_min": 0.02,
            "threshold_grid_max": 1.0,
            "threshold_grid_step": 0.02,
            "c0_max_false_positive_rate": GLOBAL_C0_MAX_FP_RATE,
            "minimum_exact_class1_recall": (
                float(metrics.get(
                    "minimum_exact_l1_recall",
                    CLASS1_FLOOR_RECALL / 100.0))),
            "maximum_class1_floor_accuracy_drop": metrics.get(
                "maximum_class1_floor_accuracy_drop",
                CLASS1_FLOOR_MAX_ACCURACY_DROP),
            "goal_aware_cell_search": bool(
                metrics.get("minimum_goal_cell_accuracy") is not None
                or metrics.get(
                    "minimum_goal_class_diagonal_recall") is not None),
            "minimum_cell_accuracy": metrics.get(
                "minimum_goal_cell_accuracy"),
            "minimum_class_diagonal_recall": metrics.get(
                "minimum_goal_class_diagonal_recall"),
            "recall_floor_policy": (
                "prefer recall-feasible thresholds without violating the "
                "false-positive cap; otherwise retain the best safety-"
                "feasible threshold"),
        },
        "_model_variant": model_variant,
        "_best_epoch": int(metrics["epoch"]),
        "_raw_best_epoch": int(raw_best_epoch if raw_best_epoch is not None else metrics["epoch"]),
        "_soup_epochs": [int(e) for e in (soup_epochs or [])],
        "_selection_score": float(metrics["selection_score"]),
        "_val_loss": float(metrics["val_loss"]),
        "_small_m_acc": float(metrics["small_m_acc_score"]),
        "_small_k1_recall": float(metrics["small_k1_recall_score"]),
        "_small_c1_recall": float(metrics["small_c1_recall_score"]),
        "_small_k0_false_positive": float(metrics["small_k0_fp_rate"]),
        "_c0_false_positive": float(metrics["c0_fp_rate"]),
        "_small_c0_false_positive": float(metrics["small_c0_fp_rate"]),
        "_guardrail_penalty": float(metrics["guardrail_penalty"]),
        "_non_target_cell_acc_mean": float(metrics["non_target_cell_acc_mean"]),
        "_rare_cell_thresholds": metrics.get("rare_cell_thresholds", {}),
        "_low_class1_cells": metrics.get("low_cls1_cells", []),
    }


def _write_model_artifacts(model: nn.Module, metrics: dict, model_variant: str,
                           raw_best_epoch: int = None,
                           soup_epochs: list[int] = None) -> None:
    _safe_torch_save(model.state_dict(), "./output/models/best_model.pth")
    thr_dict = _threshold_json_from_metrics(
        metrics,
        model_variant=model_variant,
        raw_best_epoch=raw_best_epoch,
        soup_epochs=soup_epochs,
    )
    with open("./output/models/best_thresholds.json", "w", encoding="utf-8") as _fh:
        json.dump(thr_dict, _fh, indent=2, ensure_ascii=False)


def _metrics_to_best_predictions(metrics: dict, model_variant: str) -> dict:
    import numpy as _np
    out = {
        "K": None, "M": None, "F": None, "C": None,
        "K_true": None, "M_true": None, "F_true": None, "C_true": None,
        "epoch": int(metrics["epoch"]),
        "val_loss": float(metrics["val_loss"]),
        "selection_score": float(metrics["selection_score"]),
        "model_variant": model_variant,
        "composite": float(metrics["composite"]),
        "small_m_diag": dict(metrics["small_m_diag"]),
        "small_k_diag": dict(metrics["small_k_diag"]),
        "munition_acc_matrix": metrics["acc_per_munition"].clone(),
        "munition_samples": metrics["samples_per_munition"].clone(),
        "munition_thresholds": metrics["best_thr_matrix"].clone(),
        "per_mun_thresholds": metrics["best_thr_matrix_perm"].clone(),
        "cls1_recall_vec": metrics["cls1_recall_vec"].clone(),
        "cls1_recall_per_mun": metrics["cls1_recall_per_mun"].clone(),
        "cls1_count_per_mun": metrics["cls1_count_per_mun"].clone(),
        "composite_breakdown": dict(metrics["composite_breakdown"]),
        "tuned_f1_matrix": metrics["tuned_f1_matrix"].clone(),
        "acc_per_task": list(metrics["acc_per_task"]),
        "low_cls1_cells": list(metrics.get("low_cls1_cells", [])),
    }
    for ti, name in enumerate(TASK_NAMES):
        out[name] = _np.concatenate(metrics["epoch_pred_levels"][ti])
        out[name + "_true"] = _np.concatenate(metrics["epoch_true_levels"][ti])
    return out


def _evaluate_selection_snapshot(model: nn.Module,
                                 criterion: FocalUncertaintyOrdinalLoss,
                                 val_loader,
                                 device,
                                 use_amp: bool,
                                 epoch_label: int,
                                 train_loss: float = float("nan"),
                                 mechanism_auxiliary_weight: float = 0.0,
                                 mechanism_class_distribution_weight:
                                 float = 0.25,
                                 mechanism_loss_options: dict | None = None,
                                 component_auxiliary_weight: float = 0.0,
                                 component_target_tree_teacher_weight:
                                 float = 0.0,
                                 component_rule_consistency_weight:
                                 float = 0.05,
                                 component_distribution_weight:
                                 float = 0.10,
                                 component_positive_weight:
                                 torch.Tensor | None = None,
                                 component_rule_entry_ranking_weight:
                                 torch.Tensor | None = None,
                                 component_rule_conditional_ranking_weight:
                                 torch.Tensor | None = None,
                                 component_rule_ranking_margin:
                                 float = 0.5,
                                 component_rule_hard_negative_fraction:
                                 float = 0.10,
                                 minimum_exact_l1_recall: float = (
                                     CLASS1_FLOOR_RECALL / 100.0),
                                 maximum_class1_floor_accuracy_drop:
                                 float | None = (
                                     CLASS1_FLOOR_MAX_ACCURACY_DROP),
                                 minimum_goal_cell_accuracy:
                                 float | None = None,
                                 minimum_goal_class_diagonal_recall:
                                 float | None = None) -> dict:
    model.eval()
    val_loss = 0.0
    val_task_losses = torch.zeros(4, device=device)
    all_probs, all_tgts, all_mids = [], [], []
    all_soft_targets, all_label_confidence = [], []
    mechanism_outputs_enabled = bool(getattr(
        model, "has_mechanism_outputs", False))
    component_outputs_enabled = bool(getattr(
        model, "has_component_outputs", False))

    with torch.no_grad():
        for batch in val_loader:
            (x, y, m_ids, weights, k_w, c_w, m_w, y_soft,
             label_confidence, _sample_ids, _root_ids) = batch[:11]
            mechanism_targets, component_targets = (
                _batch_auxiliary_targets(
                    batch,
                    mechanism_outputs_enabled,
                    component_outputs_enabled,
                )
            )
            x = x.to(device); y = y.to(device); m_ids = m_ids.to(device)
            weights = weights.to(device); k_w = k_w.to(device); c_w = c_w.to(device); m_w = m_w.to(device)
            y_soft = y_soft.to(device)
            label_confidence = label_confidence.to(device)
            if mechanism_targets is not None:
                mechanism_targets = mechanism_targets.to(device)
            if component_targets is not None:
                component_targets = component_targets.to(device)
            weights_clipped = torch.clamp(weights, min=0.05, max=200.0)
            if use_amp:
                with torch.amp.autocast('cuda'):
                    (
                        logits, fragment_logits, shock_logits,
                        component_logits,
                    ) = _forward_with_training_auxiliaries(
                        model, x, m_ids,
                        mechanism_outputs_enabled,
                        component_outputs_enabled,
                    )
                loss, task_losses = criterion(
                    logits.float(), y, weights_clipped,
                    k_task_weight=k_w, c_task_weight=c_w, m_task_weight=m_w,
                    m_ids=m_ids, targets_soft=y_soft,
                    target_confidence=label_confidence)
                if mechanism_targets is not None:
                    loss = loss + mechanism_auxiliary_weight * (
                        mechanism_auxiliary_loss(
                            fragment_logits.float(),
                            shock_logits.float(),
                            mechanism_targets,
                            weights_clipped,
                            model.ordinal_applicability[m_ids],
                            class_distribution_weight=(
                                mechanism_class_distribution_weight),
                            **(mechanism_loss_options or {}),
                        )
                    )
                if component_targets is not None:
                    if component_positive_weight is None:
                        raise RuntimeError(
                            "Component positive weights are missing.")
                    loss = loss + component_auxiliary_weight * (
                        component_auxiliary_loss(
                            component_logits.float(),
                            component_targets,
                            weights_clipped,
                            component_positive_weight,
                            y_soft,
                            model.ordinal_applicability[m_ids],
                            deployed_logits=logits.float(),
                            target_tree_teacher_weight=(
                                component_target_tree_teacher_weight),
                            rule_consistency_weight=(
                                component_rule_consistency_weight),
                            distribution_weight=(
                                component_distribution_weight),
                            munition_ids=m_ids,
                            rule_entry_ranking_weight=(
                                component_rule_entry_ranking_weight),
                            rule_conditional_l1_l2_ranking_weight=(
                                component_rule_conditional_ranking_weight),
                            ranking_margin=(
                                component_rule_ranking_margin),
                            hard_negative_fraction=(
                                component_rule_hard_negative_fraction),
                        )
                    )
            else:
                (
                    logits, fragment_logits, shock_logits,
                    component_logits,
                ) = _forward_with_training_auxiliaries(
                    model, x, m_ids,
                    mechanism_outputs_enabled,
                    component_outputs_enabled,
                )
                loss, task_losses = criterion(
                    logits, y, weights_clipped,
                    k_task_weight=k_w, c_task_weight=c_w, m_task_weight=m_w,
                    m_ids=m_ids, targets_soft=y_soft,
                    target_confidence=label_confidence)
                if mechanism_targets is not None:
                    loss = loss + mechanism_auxiliary_weight * (
                        mechanism_auxiliary_loss(
                            fragment_logits,
                            shock_logits,
                            mechanism_targets,
                            weights_clipped,
                            model.ordinal_applicability[m_ids],
                            class_distribution_weight=(
                                mechanism_class_distribution_weight),
                            **(mechanism_loss_options or {}),
                        )
                    )
                if component_targets is not None:
                    if component_positive_weight is None:
                        raise RuntimeError(
                            "Component positive weights are missing.")
                    loss = loss + component_auxiliary_weight * (
                        component_auxiliary_loss(
                            component_logits,
                            component_targets,
                            weights_clipped,
                            component_positive_weight,
                            y_soft,
                            model.ordinal_applicability[m_ids],
                            deployed_logits=logits,
                            target_tree_teacher_weight=(
                                component_target_tree_teacher_weight),
                            rule_consistency_weight=(
                                component_rule_consistency_weight),
                            distribution_weight=(
                                component_distribution_weight),
                            munition_ids=m_ids,
                            rule_entry_ranking_weight=(
                                component_rule_entry_ranking_weight),
                            rule_conditional_l1_l2_ranking_weight=(
                                component_rule_conditional_ranking_weight),
                            ranking_margin=(
                                component_rule_ranking_margin),
                            hard_negative_fraction=(
                                component_rule_hard_negative_fraction),
                        )
                    )
            val_loss += loss.item()
            val_task_losses += task_losses.detach()
            all_probs.append(torch.sigmoid(logits).float().cpu())
            all_tgts.append(y.float().cpu())
            all_mids.append(m_ids.cpu())
            all_soft_targets.append(y_soft.float().cpu())
            all_label_confidence.append(
                label_confidence.float().cpu())

    val_loss /= len(val_loader)
    val_task_losses /= len(val_loader)
    probs_all = torch.cat(all_probs, dim=0)
    tgts_all = torch.cat(all_tgts, dim=0)
    mids_all = torch.cat(all_mids, dim=0)
    soft_targets_all = torch.cat(all_soft_targets, dim=0)
    label_confidence_all = torch.cat(
        all_label_confidence, dim=0)
    n_val = probs_all.size(0)
    # Include the full useful probability range and a closed-head fallback at
    # 1.0 so hard false-positive constraints always have a feasible candidate.
    thr_grid = [i / 100.0 for i in range(2, 100, 2)] + [1.0]

    def _best_f1_thr(p_vec, t_vec):
        n_pos = int(t_vec.sum().item())
        n_neg = t_vec.numel() - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5, 0.0, 0.0, 0.0
        t = t_vec.long()
        best = (0.5, 0.0, 0.0, 0.0)
        for thr in thr_grid:
            pred = (p_vec >= thr).long()
            tp_ = int(((pred == 1) & (t == 1)).sum().item())
            fp_ = int(((pred == 1) & (t == 0)).sum().item())
            fn_ = int(((pred == 0) & (t == 1)).sum().item())
            p_ = tp_ / max(tp_ + fp_, 1)
            r_ = tp_ / max(tp_ + fn_, 1)
            f_ = 2 * p_ * r_ / max(p_ + r_, 1e-9)
            if f_ > best[1]:
                best = (thr, f_, p_, r_)
        return best

    def _bin_f1(pred_long, t_long):
        tp_ = int(((pred_long == 1) & (t_long == 1)).sum().item())
        fp_ = int(((pred_long == 1) & (t_long == 0)).sum().item())
        fn_ = int(((pred_long == 0) & (t_long == 1)).sum().item())
        if tp_ + fp_ == 0 or tp_ + fn_ == 0:
            return 0.0
        p_ = tp_ / (tp_ + fp_)
        r_ = tp_ / (tp_ + fn_)
        return 2 * p_ * r_ / max(p_ + r_, 1e-9)

    def _best_joint_thr(p1_vec, p2_vec, t1_vec, t2_vec,
                        alpha=0.80, thr2_min_slack=0.10,
                        max_l0_fp_rate=None):
        return _search_joint_ordinal_thresholds(
            p1_vec, p2_vec, t1_vec, t2_vec, thr_grid,
            alpha=alpha,
            thr2_min_slack=thr2_min_slack,
            max_l0_fp_rate=max_l0_fp_rate,
            minimum_exact_l1_recall=minimum_exact_l1_recall,
            maximum_accuracy_drop=(
                maximum_class1_floor_accuracy_drop),
            minimum_three_class_accuracy=(
                minimum_goal_cell_accuracy),
            minimum_class_diagonal_recall=(
                minimum_goal_class_diagonal_recall),
        )

    def _best_l1_only_thr(p1_vec, t1_vec, max_fp_rate=None, recall_weight=0.7):
        return _search_l1_threshold(
            p1_vec, t1_vec, thr_grid,
            max_fp_rate=max_fp_rate,
            recall_weight=recall_weight,
            minimum_recall=minimum_exact_l1_recall,
            maximum_accuracy_drop=(
                maximum_class1_floor_accuracy_drop),
            minimum_accuracy=minimum_goal_cell_accuracy,
            minimum_negative_recall=(
                minimum_goal_class_diagonal_recall),
        )

    viol_mask_all = probs_all[:, :, 1] > probs_all[:, :, 0]
    violation_count = int(viol_mask_all.sum().item())
    violation_per_task = viol_mask_all.sum(dim=0).long()
    violation_rate = violation_count / (n_val * 4) * 100

    best_thr_matrix = torch.full((4, 2), 0.5, dtype=torch.float32)
    tuned_f1_matrix = torch.zeros((4, 2), dtype=torch.float32)
    tuned_p_matrix = torch.zeros((4, 2), dtype=torch.float32)
    tuned_r_matrix = torch.zeros((4, 2), dtype=torch.float32)
    for i in range(4):
        p1_i = probs_all[:, i, 0]
        p2_i = probs_all[:, i, 1]
        t1_i = tgts_all[:, i, 0]
        t2_i = tgts_all[:, i, 1]
        thr1_j, thr2_j, *_ = _best_joint_thr(
            p1_i, p2_i, t1_i, t2_i,
            max_l0_fp_rate=(
                GLOBAL_C0_MAX_FP_RATE if i == 3 else None))
        best_thr_matrix[i, 0] = thr1_j
        best_thr_matrix[i, 1] = thr2_j
        p2_cl = torch.minimum(p2_i, p1_i)
        pred1 = (p1_i >= thr1_j).long()
        pred2 = ((p1_i >= thr1_j) & (p2_cl >= thr2_j)).long()
        for j, (pred_j, t_j) in enumerate([(pred1, t1_i.long()), (pred2, t2_i.long())]):
            tp_ = int(((pred_j == 1) & (t_j == 1)).sum().item())
            fp_ = int(((pred_j == 1) & (t_j == 0)).sum().item())
            fn_ = int(((pred_j == 0) & (t_j == 1)).sum().item())
            p_ = tp_ / max(tp_ + fp_, 1)
            r_ = tp_ / max(tp_ + fn_, 1)
            f_ = 2 * p_ * r_ / max(p_ + r_, 1e-9)
            tuned_f1_matrix[i, j] = f_
            tuned_p_matrix[i, j] = p_
            tuned_r_matrix[i, j] = r_

    best_thr_matrix_perm = best_thr_matrix.unsqueeze(1).expand(-1, 4, -1).clone()
    rare_cell_thresholds = {}
    for i in range(4):
        for m_id in range(4):
            mask = (mids_all == m_id)
            n_cell = int(mask.sum().item())
            min_pos_l1 = _cell_l1_min_pos(i, m_id)
            if n_cell < 30:
                meta = _rare_cell_meta(
                    i, m_id, n_cell, 0, 0, "fallback_too_few_samples",
                    float(best_thr_matrix_perm[i, m_id, 0].item()),
                    float(best_thr_matrix_perm[i, m_id, 1].item()))
                if meta:
                    rare_cell_thresholds[meta["name"]] = meta
                continue
            p1_m = probs_all[mask, i, 0]
            p2_m = probs_all[mask, i, 1]
            t1_m = tgts_all[mask, i, 0]
            t2_m = tgts_all[mask, i, 1]
            n_pos_1 = int(t1_m.sum().item())
            n_pos_2 = int(t2_m.sum().item())
            if not _is_applicable(i, m_id, 1):
                if n_pos_2:
                    raise RuntimeError(
                        f"Structural-zero cell contains positives: "
                        f"{MUN_NAMES[m_id]}/{TASK_NAMES[i]}>=2")
                if n_pos_1 >= min_pos_l1:
                    max_fp, recall_weight = _cell_l1_search_params(i, m_id)
                    thr1_m, *_ = _best_l1_only_thr(
                        p1_m, t1_m, max_fp_rate=max_fp,
                        recall_weight=recall_weight)
                    best_thr_matrix_perm[i, m_id, 0] = thr1_m
                best_thr_matrix_perm[i, m_id, 1] = 1.0
                meta = _rare_cell_meta(
                    i, m_id, n_cell, n_pos_1, n_pos_2,
                    "structural_zero_L2",
                    float(best_thr_matrix_perm[i, m_id, 0].item()), 1.0)
                rare_cell_thresholds[meta["name"]] = meta
                continue
            if n_pos_1 < min_pos_l1:
                meta = _rare_cell_meta(
                    i, m_id, n_cell, n_pos_1, n_pos_2, "fallback_too_few_L1",
                    float(best_thr_matrix_perm[i, m_id, 0].item()),
                    float(best_thr_matrix_perm[i, m_id, 1].item()))
                if meta:
                    rare_cell_thresholds[meta["name"]] = meta
                continue
            if n_pos_2 < DEFAULT_PER_MUN_MIN_POS:
                max_fp, recall_weight = _cell_l1_search_params(i, m_id)
                thr1_m, *_ = _best_l1_only_thr(
                    p1_m, t1_m, max_fp_rate=max_fp, recall_weight=recall_weight)
                best_thr_matrix_perm[i, m_id, 0] = thr1_m
                meta = _rare_cell_meta(
                    i, m_id, n_cell, n_pos_1, n_pos_2,
                    "l1_only_global_L2",
                    float(best_thr_matrix_perm[i, m_id, 0].item()),
                    float(best_thr_matrix_perm[i, m_id, 1].item()))
                if meta:
                    rare_cell_thresholds[meta["name"]] = meta
                continue
            max_fp, _ = _cell_l1_search_params(i, m_id)
            thr1_m, thr2_m, *_ = _best_joint_thr(
                p1_m, p2_m, t1_m, t2_m,
                max_l0_fp_rate=max_fp)
            best_thr_matrix_perm[i, m_id, 0] = thr1_m
            best_thr_matrix_perm[i, m_id, 1] = thr2_m
            meta = _rare_cell_meta(
                i, m_id, n_cell, n_pos_1, n_pos_2, "joint",
                float(best_thr_matrix_perm[i, m_id, 0].item()),
                float(best_thr_matrix_perm[i, m_id, 1].item()))
            if meta:
                rare_cell_thresholds[meta["name"]] = meta

    preds_05 = (probs_all > 0.5).long()
    tgt_long = tgts_all.long()
    preds_flat = preds_05.reshape(n_val, -1)
    tgt_flat = tgt_long.reshape(n_val, -1)
    tp = ((preds_flat == 1) & (tgt_flat == 1)).sum(dim=0).float()
    fp = ((preds_flat == 1) & (tgt_flat == 0)).sum(dim=0).float()
    fn = ((preds_flat == 0) & (tgt_flat == 1)).sum(dim=0).float()
    precision = tp / (tp + fp).clamp(min=1.0)
    recall = tp / (tp + fn).clamp(min=1.0)
    f1_default = 2 * precision * recall / (precision + recall).clamp(min=1e-9)

    thr_gather = best_thr_matrix_perm[:, mids_all, :]
    thr_per_sample = thr_gather.permute(1, 0, 2).contiguous()
    p1 = probs_all[:, :, 0]
    p2 = torch.minimum(probs_all[:, :, 1], p1)
    pass1 = p1 >= thr_per_sample[:, :, 0]
    pass2 = (p1 >= thr_per_sample[:, :, 0]) & (p2 >= thr_per_sample[:, :, 1])
    pred_level = pass1.long() + pass2.long()
    true_level = tgts_all[:, :, 0].long() + tgts_all[:, :, 1].long()
    targeted_probability_diagnostics = (
        _targeted_probability_diagnostics(
            probs_all, true_level, mids_all))
    confidence_resolved_diagnostics = (
        _confidence_resolved_diagnostics(
            probs_all,
            soft_targets_all,
            label_confidence_all,
            true_level,
            pred_level,
            mids_all,
        )
    )
    c0_mask = (true_level[:, 3] == 0)
    c0_fp_rate = float((pred_level[c0_mask, 3] > 0).float().mean().item()) if int(c0_mask.sum().item()) > 0 else 0.0
    small_c0_mask = c0_mask & (mids_all == 0)
    small_c0_fp_rate = float((pred_level[small_c0_mask, 3] > 0).float().mean().item()) if int(small_c0_mask.sum().item()) > 0 else 0.0
    correct_mask = (pred_level == true_level)
    correct_per_task = correct_mask.sum(dim=0).long()
    acc_per_task = (correct_per_task.float() / max(n_val, 1) * 100).tolist()

    cls1_recall_vec = torch.zeros(4, dtype=torch.float32)
    for i in range(4):
        mask_c1 = (true_level[:, i] == 1)
        if int(mask_c1.sum().item()) > 0:
            cls1_recall_vec[i] = (pred_level[mask_c1, i] == 1).float().mean() * 100.0

    cls1_recall_per_mun = torch.zeros(4, 4, dtype=torch.float32)
    cls1_count_per_mun = torch.zeros(4, 4, dtype=torch.long)
    for i in range(4):
        for m_id in range(4):
            mask_c1m = (true_level[:, i] == 1) & (mids_all == m_id)
            n_c1m = int(mask_c1m.sum().item())
            cls1_count_per_mun[i, m_id] = n_c1m
            if n_c1m > 0:
                cls1_recall_per_mun[i, m_id] = (pred_level[mask_c1m, i] == 1).float().mean() * 100.0

    acc_per_munition = torch.zeros(4, 4, dtype=torch.float32)
    samples_per_munition = torch.zeros(4, dtype=torch.long)
    for m_id in range(4):
        mask = (mids_all == m_id)
        n_m = int(mask.sum().item())
        samples_per_munition[m_id] = n_m
        if n_m > 0:
            acc_per_munition[:, m_id] = correct_mask[mask].sum(dim=0).float() / n_m * 100.0

    small_m_mask = (mids_all == 0)
    small_m_diag = {
        "n": int(small_m_mask.sum().item()),
        "acc": 0.0,
        "cm": torch.zeros(3, 3, dtype=torch.long),
        "true_counts": torch.zeros(3, dtype=torch.long),
        "pred_counts": torch.zeros(3, dtype=torch.long),
        "l0_to_l1": 0.0, "l0_to_l2": 0.0, "l1_to_l0": 0.0,
        "l1_to_l2": 0.0, "l2_to_l1": 0.0,
    }
    if small_m_diag["n"] > 0:
        sm_true = true_level[small_m_mask, 1].long()
        sm_pred = pred_level[small_m_mask, 1].long()
        sm_cm = torch.bincount(sm_true * 3 + sm_pred, minlength=9).reshape(3, 3)
        sm_true_counts = sm_cm.sum(dim=1)
        sm_pred_counts = sm_cm.sum(dim=0)

        def _cell_rate(t_cls, p_cls):
            denom = int(sm_true_counts[t_cls].item())
            return 0.0 if denom <= 0 else float(sm_cm[t_cls, p_cls].item()) / denom * 100.0

        small_m_diag.update({
            "acc": float((sm_pred == sm_true).float().mean().item()) * 100.0,
            "cm": sm_cm,
            "true_counts": sm_true_counts,
            "pred_counts": sm_pred_counts,
            "l0_to_l1": _cell_rate(0, 1),
            "l0_to_l2": _cell_rate(0, 2),
            "l1_to_l0": _cell_rate(1, 0),
            "l1_to_l2": _cell_rate(1, 2),
            "l2_to_l1": _cell_rate(2, 1),
        })
    small_k_diag = _build_cell_diag(
        true_level, pred_level, probs_all, mids_all, task_idx=0, mun_id=0)

    acc3_mean = float(correct_mask.float().mean().item())
    cls1_recall_mean = float(cls1_recall_vec.mean().item()) / 100.0
    f1_mean = float(tuned_f1_matrix.mean().item())
    min_cell_acc_3class = float(acc_per_munition.min().item()) / 100.0
    min_cell_penalty = 0.30 * max(0.0, 0.95 - min_cell_acc_3class)
    composite = (0.60 * acc3_mean + 0.15 * cls1_recall_mean + 0.20 * f1_mean
                 - 0.05 * violation_rate / 100.0 - min_cell_penalty)
    small_m_acc_score = small_m_diag["acc"] / 100.0
    small_k1_recall_score = float(cls1_recall_per_mun[0, 0].item()) / 100.0
    small_c1_recall_score = float(cls1_recall_per_mun[3, 0].item()) / 100.0
    non_target_mask = ~TARGET_CELL_MASK
    non_target_cell_acc_mean = float(acc_per_munition[non_target_mask].mean().item()) / 100.0
    # Model selection must be stationary.  Historical-best-relative penalties
    # made an epoch's score depend on which candidates happened to precede it
    # and gave final candidate re-evaluation a different objective.
    non_target_drop_penalty = 0.0
    non_target_cls1_drop_penalty = 0.0
    low_cls1_cells = _collect_low_cls1_cells(
        cls1_recall_per_mun, cls1_count_per_mun, non_target_mask,
        floor=100.0 * float(minimum_exact_l1_recall))
    class1_floor_penalty = (
        max((cell["deficit"] for cell in low_cls1_cells), default=0.0) / 100.0)
    min_supported_class1_recall = _minimum_supported_class1_recall(
        cls1_recall_per_mun, cls1_count_per_mun)
    min_supported_class_diagonal_recall = (
        _minimum_supported_diagonal_recall(
            pred_level, true_level, mids_all))
    goal_class1_floor_penalty = max(
        0.0,
        GOAL_MIN_CLASS_DIAGONAL_RECALL_PERCENT / 100.0
        - min_supported_class1_recall,
    )
    goal_cell_accuracy_penalty = 2.0 * max(
        0.0,
        GOAL_MIN_CELL_3CLASS_ACCURACY_PERCENT / 100.0
        - min_cell_acc_3class,
    )
    goal_diagonal_recall_penalty = max(
        0.0,
        GOAL_MIN_CLASS_DIAGONAL_RECALL_PERCENT / 100.0
        - min_supported_class_diagonal_recall,
    )
    c0_guard_rate = max(c0_fp_rate, small_c0_fp_rate)
    c0_fp_penalty = max(0.0, c0_guard_rate - 0.025) * 2.0
    small_k0_mask = (mids_all == 0) & (true_level[:, 0] == 0)
    small_k0_fp_rate = (
        float((pred_level[small_k0_mask, 0] > 0).float().mean().item())
        if int(small_k0_mask.sum().item()) > 0 else 0.0)
    small_k0_fp_penalty = max(0.0, small_k0_fp_rate - 0.005) * 2.0
    guardrail_penalty = (
        non_target_drop_penalty + non_target_cls1_drop_penalty
        + class1_floor_penalty + goal_class1_floor_penalty
        + goal_cell_accuracy_penalty + goal_diagonal_recall_penalty
        + c0_fp_penalty + small_k0_fp_penalty)
    selection_score = (0.37 * acc3_mean + 0.14 * f1_mean + 0.09 * cls1_recall_mean
                       + 0.11 * small_m_acc_score + 0.11 * small_c1_recall_score
                       + 0.08 * small_k1_recall_score
                       + 0.10 * non_target_cell_acc_mean - guardrail_penalty)
    epoch_pred_levels = {ti: [pred_level[:, ti].numpy()] for ti in range(4)}
    epoch_true_levels = {ti: [true_level[:, ti].numpy()] for ti in range(4)}
    composite_breakdown = {
        "acc3": acc3_mean, "cls1": cls1_recall_mean, "f1": f1_mean,
        "viol": violation_rate, "min_cell": min_cell_acc_3class,
        "min_pen": min_cell_penalty, "small_m_acc": small_m_acc_score,
        "small_k1_recall": small_k1_recall_score,
        "small_c1_recall": small_c1_recall_score,
        "non_target_cell_acc_mean": non_target_cell_acc_mean,
        "c0_fp": c0_fp_rate, "small_c0_fp": small_c0_fp_rate,
        "small_k0_fp": small_k0_fp_rate,
        "guard_pen": guardrail_penalty,
        "non_target_drop_pen": non_target_drop_penalty,
        "non_target_cls1_drop_pen": non_target_cls1_drop_penalty,
        "class1_floor_pen": class1_floor_penalty,
        "goal_class1_floor_pen": goal_class1_floor_penalty,
        "goal_cell_accuracy_pen": goal_cell_accuracy_penalty,
        "min_supported_class1_recall": min_supported_class1_recall,
        "goal_diagonal_recall_pen": goal_diagonal_recall_penalty,
        "min_supported_class_diagonal_recall": (
            min_supported_class_diagonal_recall),
        "small_k0_fp_pen": small_k0_fp_penalty,
        "selection": selection_score,
    }
    return {
        "epoch": int(epoch_label), "train_loss": float(train_loss),
        "val_loss": float(val_loss), "val_task_losses": val_task_losses.cpu(),
        "violation_rate": float(violation_rate),
        "violation_per_task": violation_per_task.cpu(),
        "precision_default": precision.cpu(), "recall_default": recall.cpu(),
        "f1_default": f1_default.cpu(), "best_thr_matrix": best_thr_matrix,
        "best_thr_matrix_perm": best_thr_matrix_perm,
        "tuned_f1_matrix": tuned_f1_matrix, "tuned_p_matrix": tuned_p_matrix,
        "tuned_r_matrix": tuned_r_matrix, "pred_level": pred_level,
        "true_level": true_level, "acc_per_task": acc_per_task,
        "munition_ids": mids_all,
        "acc_per_munition": acc_per_munition,
        "samples_per_munition": samples_per_munition,
        "cls1_recall_vec": cls1_recall_vec,
        "cls1_recall_per_mun": cls1_recall_per_mun,
        "cls1_count_per_mun": cls1_count_per_mun,
        "small_m_diag": small_m_diag, "small_k_diag": small_k_diag,
        "c0_fp_rate": c0_fp_rate, "small_k0_fp_rate": small_k0_fp_rate,
        "small_c0_fp_rate": small_c0_fp_rate, "acc3_mean": acc3_mean,
        "cls1_recall_mean": cls1_recall_mean, "f1_mean": f1_mean,
        "min_cell_acc_3class": min_cell_acc_3class,
        "min_supported_class1_recall": min_supported_class1_recall,
        "min_supported_class_diagonal_recall": (
            min_supported_class_diagonal_recall),
        "min_cell_penalty": min_cell_penalty, "composite": composite,
        "small_m_acc_score": small_m_acc_score,
        "small_k1_recall_score": small_k1_recall_score,
        "small_c1_recall_score": small_c1_recall_score,
        "non_target_cell_acc_mean": non_target_cell_acc_mean,
        "guardrail_penalty": guardrail_penalty,
        "selection_score": selection_score,
        "composite_breakdown": composite_breakdown,
        "rare_cell_thresholds": rare_cell_thresholds,
        "low_cls1_cells": low_cls1_cells,
        "minimum_exact_l1_recall": float(
            minimum_exact_l1_recall),
        "maximum_class1_floor_accuracy_drop": (
            None if maximum_class1_floor_accuracy_drop is None
            else float(maximum_class1_floor_accuracy_drop)),
        "minimum_goal_cell_accuracy": (
            None if minimum_goal_cell_accuracy is None
            else float(minimum_goal_cell_accuracy)),
        "minimum_goal_class_diagonal_recall": (
            None if minimum_goal_class_diagonal_recall is None
            else float(minimum_goal_class_diagonal_recall)),
        "targeted_probability_diagnostics": (
            targeted_probability_diagnostics),
        "confidence_resolved_diagnostics": (
            confidence_resolved_diagnostics),
        "epoch_pred_levels": epoch_pred_levels,
        "epoch_true_levels": epoch_true_levels,
    }


def train_model(parquet_path: str, smoke_test: bool = False, seed: int = 42,
                ablation_config: dict | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Booting on Device: {device}")
    if ablation_config:
        print(f"[Training][Ablation] experiment_id={ablation_config.get('experiment_id', 'unknown')}")
        print(f"[Training][Ablation] description={ablation_config.get('description', '')}")
    model_cfg = _cfg_section(ablation_config, "model")
    loss_cfg = _cfg_section(ablation_config, "loss")
    training_cfg = _cfg_section(ablation_config, "training")
    calibration_cfg = _cfg_section(ablation_config, "calibration")
    minimum_exact_l1_recall = float(
        calibration_cfg.get(
            "minimum_exact_class1_recall",
            CLASS1_FLOOR_RECALL / 100.0,
        )
    )
    raw_maximum_accuracy_drop = calibration_cfg.get(
        "maximum_class1_floor_accuracy_drop",
        CLASS1_FLOOR_MAX_ACCURACY_DROP,
    )
    maximum_class1_floor_accuracy_drop = (
        None
        if raw_maximum_accuracy_drop is None
        else float(raw_maximum_accuracy_drop)
    )
    goal_aware_cell_search = bool(
        calibration_cfg.get("goal_aware_cell_search", False))
    minimum_goal_cell_accuracy = (
        float(calibration_cfg.get("minimum_cell_accuracy", 0.94))
        if goal_aware_cell_search else None
    )
    minimum_goal_class_diagonal_recall = (
        float(calibration_cfg.get(
            "minimum_class_diagonal_recall", 0.90))
        if goal_aware_cell_search else None
    )
    for name, value in (
        ("minimum_exact_class1_recall", minimum_exact_l1_recall),
        ("minimum_cell_accuracy", minimum_goal_cell_accuracy),
        ("minimum_class_diagonal_recall",
         minimum_goal_class_diagonal_recall),
    ):
        if value is not None and not (0.0 <= value <= 1.0):
            raise ValueError(f"calibration.{name} must be in [0,1].")
    if (
        maximum_class1_floor_accuracy_drop is not None
        and not (0.0 <= maximum_class1_floor_accuracy_drop <= 1.0)
    ):
        raise ValueError(
            "calibration.maximum_class1_floor_accuracy_drop must be "
            "null or in [0,1].")
    feature_columns = get_feature_columns(ablation_config)
    random.seed(seed)
    import numpy as _np_seed
    _np_seed.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[Training] Random seed: {seed}")

    # [P1 #9] 现在 dataloader 同时返回 pos_weight (4, 2) 张量
    (train_loader, val_loader, test_loader, scaler, pos_weight_np,
     data_contract) = \
        get_dataloaders(parquet_path, batch_size=256,
                        ablation_config=ablation_config,
                        load_test_split=False)
    if test_loader is not None:
        raise RuntimeError("Training must keep the test split sealed.")
    pos_weight = torch.from_numpy(pos_weight_np).to(device)
    expected_draws_per_mun = None
    batch_sampler = getattr(train_loader, "batch_sampler", None)
    if batch_sampler is not None and hasattr(batch_sampler, "expected_draws_per_munition"):
        expected_draws_per_mun = batch_sampler.expected_draws_per_munition
        print(f"[Training] Expected balanced munition draws / epoch: "
              f"{[expected_draws_per_mun[m] for m in range(4)]}")

    # 装载按 Stage-0 可观测特征列配置的 MTL Cond. Shared Bottom 网络
    resolved_model_config = {
        "in_dim": len(feature_columns),
        "base_input_dim": int(
            model_cfg.get("base_input_dim", len(feature_columns))),
        "num_munitions": 4,
        "munition_emb_dim": int(model_cfg.get("munition_emb_dim", 16)),
        "use_munition_embedding": bool(
            model_cfg.get("use_munition_embedding", True)),
        "use_munition_experts": bool(
            model_cfg.get("use_munition_experts", True)),
        "use_physics_skip": bool(model_cfg.get("use_physics_skip", True)),
        "use_k_cascade": bool(model_cfg.get("use_k_cascade", True)),
        "deep_m_branch": bool(model_cfg.get("deep_m_branch", True)),
        "ordinal_parameterization": str(
            model_cfg.get(
                "ordinal_parameterization", "cumulative_logits")),
        "use_mechanism_decomposition": bool(
            model_cfg.get("use_mechanism_decomposition", False)),
        "use_mechanism_auxiliary_heads": bool(
            model_cfg.get("use_mechanism_auxiliary_heads", False)),
        "mechanism_encoder_mode": str(
            model_cfg.get("mechanism_encoder_mode", "shared")),
        "use_component_auxiliary_heads": bool(
            model_cfg.get(
                "use_component_auxiliary_heads", False)),
        "component_ids": (
            data_contract.get(
                "component_supervision_contract", {})
            .get("component_ids")
            if bool(model_cfg.get(
                "use_component_auxiliary_heads", False))
            else None
        ),
        "component_branch_mode": str(
            model_cfg.get(
                "component_branch_mode", "shared_auxiliary")),
        "component_branch_munition_emb_dim": int(
            model_cfg.get(
                "component_branch_munition_emb_dim", 16)),
        "component_tree_fusion_alpha": model_cfg.get(
            "component_tree_fusion_alpha"),
        "residual_adapter_cells": list(
            model_cfg.get("residual_adapter_cells", [])),
        "residual_adapter_hidden_dim": int(
            model_cfg.get("residual_adapter_hidden_dim", 64)),
        "residual_adapter_feature_indices": list(
            model_cfg.get(
                "residual_adapter_feature_indices",
                [0, 1, 2, 3, 4, 5])),
        "residual_adapter_frequencies": list(
            model_cfg.get(
                "residual_adapter_frequencies",
                [1.0, 2.0, 4.0, 8.0])),
        "residual_adapter_max_logit": float(
            model_cfg.get("residual_adapter_max_logit", 2.0)),
        "ordinal_applicability": data_contract["ordinal_applicability"],
    }
    use_component_proxy_features = bool(
        _cfg_section(ablation_config, "data").get(
            "use_component_proxy_features", False))
    if use_component_proxy_features:
        from loitering_munition_damage_twin.surrogate.features import COMPONENT_PROXY_FEATURE_COLUMNS
        expected_base_input_dim = (
            len(feature_columns)
            - len(COMPONENT_PROXY_FEATURE_COLUMNS)
        )
        resolved_base_input_dim = int(
            resolved_model_config["base_input_dim"])
        isolated_component_branch = (
            resolved_base_input_dim == expected_base_input_dim
        )
        direct_proxy_path = (
            resolved_base_input_dim == len(feature_columns)
        )
        allow_direct_proxy_path = bool(
            model_cfg.get(
                "allow_component_proxy_direct_path", False))
        if not (
            isolated_component_branch
            or (direct_proxy_path and allow_direct_proxy_path)
        ):
            raise ValueError(
                "Component proxy features require either A38-style branch "
                f"isolation (base_input_dim={expected_base_input_dim}) or "
                "an explicit from-scratch direct-path ablation "
                f"(base_input_dim={len(feature_columns)} and "
                "model.allow_component_proxy_direct_path=true).")
        if (
            isolated_component_branch
            and not (
                bool(resolved_model_config[
                    "use_component_auxiliary_heads"])
                and str(resolved_model_config[
                    "component_branch_mode"]).strip().lower()
                == "independent_experts"
            )
        ):
            raise ValueError(
                "An isolated component-proxy extension requires the "
                "independent component branch.")
        if direct_proxy_path:
            if training_cfg.get("initial_checkpoint"):
                raise ValueError(
                    "Direct component-proxy experiments must train from "
                    "scratch; feature-expanding a checkpoint is not a "
                    "pre-registered warm-start migration.")
            if bool(training_cfg.get("freeze_base_model", False)):
                raise ValueError(
                    "Direct component-proxy experiments cannot freeze the "
                    "base model.")
    mechanism_decomposition_enabled = bool(
        resolved_model_config["use_mechanism_decomposition"])
    mechanism_auxiliary_heads_enabled = bool(
        resolved_model_config["use_mechanism_auxiliary_heads"])
    if (
        mechanism_decomposition_enabled
        and mechanism_auxiliary_heads_enabled
    ):
        raise ValueError(
            "model.use_mechanism_decomposition and "
            "model.use_mechanism_auxiliary_heads are mutually exclusive.")
    mechanism_outputs_enabled = bool(
        mechanism_decomposition_enabled
        or mechanism_auxiliary_heads_enabled)
    component_outputs_enabled = bool(
        resolved_model_config[
            "use_component_auxiliary_heads"])
    if mechanism_outputs_enabled and component_outputs_enabled:
        raise ValueError(
            "Task-level mechanism and component auxiliary experiments "
            "must remain factorially isolated.")
    if mechanism_outputs_enabled != bool(
            data_contract.get("mechanism_supervision_enabled", False)):
        raise ValueError(
            "Exactly one mechanism-output model mode "
            "(decomposition or auxiliary-only heads) and "
            "data.use_mechanism_supervision must be enabled together.")
    if component_outputs_enabled != bool(
            data_contract.get(
                "component_supervision_enabled", False)):
        raise ValueError(
            "model.use_component_auxiliary_heads and "
            "data.use_component_supervision must be enabled together.")
    mechanism_auxiliary_weight = float(
        loss_cfg.get("mechanism_auxiliary_weight", 0.50))
    mechanism_class_distribution_weight = float(
        loss_cfg.get(
            "mechanism_class_distribution_weight", 0.25))
    mechanism_branch_weights_raw = list(
        loss_cfg.get("mechanism_branch_weights", [1.0, 1.0]))
    if len(mechanism_branch_weights_raw) != 2:
        raise ValueError(
            "loss.mechanism_branch_weights must contain "
            "[fragment, shock].")
    mechanism_branch_weights = torch.as_tensor(
        mechanism_branch_weights_raw,
        dtype=torch.float32,
        device=device,
    )
    if (
        not torch.isfinite(mechanism_branch_weights).all()
        or (mechanism_branch_weights <= 0.0).any()
    ):
        raise ValueError(
            "loss.mechanism_branch_weights must be finite and positive.")
    mechanism_boundary_focus_weight = float(
        loss_cfg.get("mechanism_boundary_focus_weight", 0.0))
    mechanism_boundary_focus_bandwidth = float(
        loss_cfg.get("mechanism_boundary_focus_bandwidth", 0.15))
    mechanism_hard_classification_weight = float(
        loss_cfg.get("mechanism_hard_classification_weight", 0.0))
    mechanism_use_dataset_row_weights = bool(
        loss_cfg.get("mechanism_use_dataset_row_weights", True))
    if mechanism_auxiliary_weight < 0.0:
        raise ValueError(
            "loss.mechanism_auxiliary_weight must be non-negative.")
    if mechanism_class_distribution_weight < 0.0:
        raise ValueError(
            "loss.mechanism_class_distribution_weight must be non-negative.")
    if mechanism_boundary_focus_weight < 0.0:
        raise ValueError(
            "loss.mechanism_boundary_focus_weight must be non-negative.")
    if mechanism_boundary_focus_bandwidth <= 0.0:
        raise ValueError(
            "loss.mechanism_boundary_focus_bandwidth must be positive.")
    if mechanism_hard_classification_weight < 0.0:
        raise ValueError(
            "loss.mechanism_hard_classification_weight must be non-negative.")
    if (
        mechanism_outputs_enabled
        and mechanism_auxiliary_weight <= 0.0
    ):
        raise ValueError(
            "Mechanism supervision requires a positive auxiliary weight.")
    mechanism_loss_options = {
        "branch_weights": mechanism_branch_weights,
        "boundary_focus_weight": mechanism_boundary_focus_weight,
        "boundary_focus_bandwidth": mechanism_boundary_focus_bandwidth,
        "hard_classification_weight": (
            mechanism_hard_classification_weight),
        "use_dataset_row_weights": (
            mechanism_use_dataset_row_weights),
    }
    component_auxiliary_weight = float(
        loss_cfg.get("component_auxiliary_weight", 0.10))
    component_target_tree_teacher_weight = float(
        loss_cfg.get(
            "component_target_tree_teacher_weight", 0.0))
    component_rule_consistency_weight = float(
        loss_cfg.get(
            "component_rule_consistency_weight", 0.05))
    component_distribution_weight = float(
        loss_cfg.get(
            "component_distribution_weight", 0.10))
    for name, value in (
            ("component_auxiliary_weight",
             component_auxiliary_weight),
            ("component_target_tree_teacher_weight",
             component_target_tree_teacher_weight),
            ("component_rule_consistency_weight",
             component_rule_consistency_weight),
            ("component_distribution_weight",
             component_distribution_weight)):
        if value < 0.0:
            raise ValueError(
                f"loss.{name} must be non-negative.")
    if (
        component_outputs_enabled
        and component_auxiliary_weight <= 0.0
    ):
        raise ValueError(
            "Component supervision requires a positive auxiliary weight.")
    component_rule_entry_ranking_weight = (
        _task_munition_loss_matrix(
            loss_cfg,
            "component_rule_entry_ranking_weight",
            device=device,
            default=0.0,
            minimum=0.0,
        )
    )
    component_rule_conditional_ranking_weight = (
        _task_munition_loss_matrix(
            loss_cfg,
            "component_rule_conditional_l1_l2_ranking_weight",
            device=device,
            default=0.0,
            minimum=0.0,
        )
    )
    if (
        (component_rule_entry_ranking_weight > 1.0).any()
        or (component_rule_conditional_ranking_weight > 1.0).any()
    ):
        raise ValueError(
            "Component-rule ranking loss weights must be in [0,1].")
    component_rule_ranking_margin = float(
        loss_cfg.get(
            "component_rule_ranking_margin",
            loss_cfg.get("ranking_margin", 0.5),
        )
    )
    component_rule_hard_negative_fraction = float(
        loss_cfg.get(
            "component_rule_hard_negative_fraction",
            loss_cfg.get("hard_negative_fraction", 0.10),
        )
    )
    if component_rule_ranking_margin < 0.0:
        raise ValueError(
            "loss.component_rule_ranking_margin must be non-negative.")
    if not (
        0.0 < component_rule_hard_negative_fraction <= 1.0
    ):
        raise ValueError(
            "loss.component_rule_hard_negative_fraction must be in (0,1].")
    if component_outputs_enabled:
        raw_component_positive_weight = data_contract.get(
            "component_positive_weight")
        if raw_component_positive_weight is None:
            raise RuntimeError(
                "Component positive-weight contract is missing.")
        component_positive_weight = torch.as_tensor(
            raw_component_positive_weight,
            dtype=torch.float32,
            device=device,
        )
        expected_component_shape = (
            2,
            len(resolved_model_config["component_ids"]),
        )
        if tuple(component_positive_weight.shape) != (
                expected_component_shape):
            raise RuntimeError(
                "Component positive-weight shape mismatch: "
                f"expected={expected_component_shape}, "
                f"observed={tuple(component_positive_weight.shape)}")
        component_positive_weight_enabled = bool(
            loss_cfg.get(
                "component_use_positive_weight", True))
        if not component_positive_weight_enabled:
            # A probability that feeds a nonlinear damage tree must be trained
            # with a proper scoring rule. BCE pos_weight changes its optimum
            # and caused A34 to over-predict mean component risk by 28.5%.
            component_positive_weight = torch.ones_like(
                component_positive_weight)
    else:
        component_positive_weight = None
        component_positive_weight_enabled = False
    model = DamageAssessmentMTL(
        **resolved_model_config,
    ).to(device)
    warm_start_provenance = _load_verified_warm_start(
        model,
        training_cfg=training_cfg,
        data_contract=data_contract,
        resolved_model_config=resolved_model_config,
        seed=seed,
    )
    if (
        component_outputs_enabled
        and resolved_model_config["component_branch_mode"]
        == "independent_experts"
    ):
        component_targets_train = (
            train_loader.dataset.component_targets_soft)
        component_munitions_train = (
            train_loader.dataset.mun_ids)
        if component_targets_train is None:
            raise RuntimeError(
                "Independent component branch requires train component "
                "targets.")
        continuation = bool(
            warm_start_provenance
            and warm_start_provenance.get(
                "independent_component_continuation", False)
        )
        if continuation:
            print(
                "[Training][Component] retained independent expert "
                "weights from the verified warm-start checkpoint.")
        else:
            component_priors = torch.stack([
                component_targets_train[
                    component_munitions_train == munition_index
                ].mean(dim=0)
                for munition_index in range(4)
            ])
            model.initialize_independent_component_priors(
                component_priors)
            print(
                "[Training][Component] independent experts initialized "
                "from train-only soft-label priors.")
    initial_model_state = None
    freeze_base_model = bool(
        training_cfg.get("freeze_base_model", False))
    if freeze_base_model and warm_start_provenance is None:
        raise ValueError(
            "training.freeze_base_model requires initial_checkpoint.")
    if warm_start_provenance is not None:
        initial_model_state = _clone_state_dict_cpu(model)
        print(
            "[Training][WarmStart] verified checkpoint="
            f"{warm_start_provenance['checkpoint_path']}")
    if freeze_base_model:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.residual_adapters.parameters():
            parameter.requires_grad_(True)
        independent_component_branch = getattr(
            model, "independent_component_branch", None)
        if independent_component_branch is not None:
            for parameter in (
                    independent_component_branch.parameters()):
                parameter.requires_grad_(True)
        trainable_extension_names = []
        if len(model.residual_adapters):
            trainable_extension_names.append(
                f"{len(model.residual_adapters)} residual adapters")
        if independent_component_branch is not None:
            trainable_extension_names.append(
                "independent component experts")
        print(
            "[Training][WarmStart] base model frozen; trainable="
            + ", ".join(trainable_extension_names))
    # [P1 #9 + P1 #10] 配置 Focal-BCE + pos_weight + 保序混合损失
    cell_class1_alpha = torch.ones(4, 4, dtype=torch.float32, device=device)
    if bool(loss_cfg.get("use_cell_class1_alpha", False)):
        cell_class1_alpha[0, 0] = 1.25  # v5.2: Small x K1 only; keep K task weights unchanged.
        cell_class1_alpha[3, 0] = 1.8  # v5.1: Small x C1 only; keep Small x M1 neutral.
    loss_gamma = FOCAL_LOSS_GAMMA if bool(
        loss_cfg.get("use_focal_loss", False)) else 0.0
    penalty_weight = 10.0 if bool(
        loss_cfg.get("use_ordinal_penalty", False)) else 0.0
    class1_margin_weight = 0.5 if bool(
        loss_cfg.get("use_class1_margin", False)) else 0.0
    class_distribution_weight = (
        float(loss_cfg.get("class_distribution_weight", 0.25))
        if bool(loss_cfg.get("use_class_distribution_loss", True))
        else 0.0
    )
    if class_distribution_weight < 0.0:
        raise ValueError("loss.class_distribution_weight must be non-negative.")
    hard_level_classification_weight = float(
        loss_cfg.get("hard_level_classification_weight", 0.0))
    if hard_level_classification_weight < 0.0:
        raise ValueError(
            "loss.hard_level_classification_weight must be non-negative.")
    middle_class_distribution_multiplier = _task_munition_loss_matrix(
        loss_cfg,
        "middle_class_distribution_multiplier",
        device=device,
        default=1.0,
        minimum=1.0,
    )
    entry_ranking_weight = _task_munition_loss_matrix(
        loss_cfg,
        "entry_ranking_weight",
        device=device,
        default=0.0,
        minimum=0.0,
    )
    conditional_l1_l2_ranking_weight = _task_munition_loss_matrix(
        loss_cfg,
        "conditional_l1_l2_ranking_weight",
        device=device,
        default=0.0,
        minimum=0.0,
    )
    if (
        (entry_ranking_weight > 1.0).any()
        or (conditional_l1_l2_ranking_weight > 1.0).any()
    ):
        raise ValueError("ranking loss weights must be in [0, 1].")
    ranking_margin = float(loss_cfg.get("ranking_margin", 0.5))
    hard_negative_fraction = float(
        loss_cfg.get("hard_negative_fraction", 0.10))
    criterion = FocalUncertaintyOrdinalLoss(
        num_tasks=4, penalty_weight=penalty_weight,
        gamma=loss_gamma,
        pos_weight=pos_weight,
        class1_margin_weight=class1_margin_weight,
        cell_class1_alpha=cell_class1_alpha,
        class_distribution_weight=class_distribution_weight,
        hard_level_classification_weight=(
            hard_level_classification_weight),
        middle_class_distribution_multiplier=(
            middle_class_distribution_multiplier),
        entry_ranking_weight=entry_ranking_weight,
        conditional_l1_l2_ranking_weight=(
            conditional_l1_l2_ranking_weight),
        ranking_margin=ranking_margin,
        hard_negative_fraction=hard_negative_fraction,
    ).to(device)
    freeze_criterion_uncertainty = bool(
        training_cfg.get(
            "freeze_criterion_uncertainty",
            freeze_base_model))
    if freeze_criterion_uncertainty:
        for parameter in criterion.parameters():
            parameter.requires_grad_(False)

    lr_base = (
        1e-4 if smoke_test
        else float(training_cfg.get("learning_rate", 1e-3))
    )
    if lr_base <= 0.0:
        raise ValueError("training.learning_rate must be positive.")
    # [P0 #3] 把 criterion 的 log_vars 单独放进 weight_decay=0 的参数组，
    # 防止 AdamW 的 L2 衰减把它持续往 0 拖、导致 MTL 退化为 1:1 暴力相加
    trainable_model_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    trainable_criterion_parameters = [
        parameter for parameter in criterion.parameters()
        if parameter.requires_grad
    ]
    if not trainable_model_parameters:
        raise RuntimeError("No trainable model parameters remain.")
    optimizer_groups = [{
        "params": trainable_model_parameters,
        "weight_decay": float(
            training_cfg.get("weight_decay", 5e-4)),
    }]
    if trainable_criterion_parameters:
        optimizer_groups.append({
            "params": trainable_criterion_parameters,
            "weight_decay": 0.0,
        })
    optimizer = optim.AdamW(optimizer_groups, lr=lr_base)

    epochs = (
        10 if smoke_test
        else int(training_cfg.get("epochs", 45))
    )
    if epochs <= 0:
        raise ValueError("training.epochs must be positive.")

    if not smoke_test:
        # [60-epoch 诊断后替换] CosineAnnealingWarmRestarts → CosineAnnealingLR
        # 理由：原 warm-restart 在 epoch 10/30 LR 从 1e-5 暴跳回 1e-3，
        # 直接摧毁 epoch 24 才积累下来的最佳收敛 (Val Loss -2.90 → -2.66)，
        # 此后再也回不去。纯余弦退火让 LR 平稳收敛到 eta_min=1e-5。
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-5)
    else:
        scheduler = None

    os.makedirs("./output/models", exist_ok=True)
    os.makedirs("./output/runs/damage_model", exist_ok=True)
    # Final business checkpoint is driven by the strict goal-aware key.
    # Val Loss is kept only as a diagnostic and an optional shadow checkpoint.
    best_val_loss = float('inf')
    best_val_epoch = None
    best_selection_score = -float('inf')
    best_selection_key = None
    best_selection_epoch = None
    best_composite_epoch = None
    # [60-epoch 诊断后替换] 废弃 best_val_loss 选择：
    #   Loss 混入 `0.5*log_var` 正则项后会跑到负值，Val Loss ≠ 模型质量。
    #   改用复合指标 composite = F1_K2 + F1_M2 + F1_F2 + F1_C2 - 5 * violation_rate
    #   （高价值 L2 毁伤判定 F1 为主，保序性违反率为软约束扣分）
    best_composite = -float('inf')
    # Raw-best saving and early stopping use the same lexicographic 94%/90%
    # plus safety key as top-k/final selection. Val Loss remains diagnostic.
    min_selection_epochs = (
        0 if smoke_test
        else int(training_cfg.get(
            "minimum_selection_epochs", min(32, epochs)))
    )
    patience = (
        3 if smoke_test
        else int(training_cfg.get("selection_patience", 10))
    )
    if not (0 <= min_selection_epochs <= epochs):
        raise ValueError(
            "training.minimum_selection_epochs must be in [0, epochs].")
    if patience <= 0:
        raise ValueError(
            "training.selection_patience must be positive.")
    stale_epochs = 0
    topk_selection_candidates = []
    topk_k = 5

    # 初始化条件 TensorBoard 写入引擎
    writer = SummaryWriter(log_dir="./output/runs/damage_model")

    # 初始化 AMP 混合精度梯度缩放器加速训练
    scaler_amp = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    print(f"\n{'='*50}")
    print(f"  Starting Epochs: {epochs} {'[SMOKE TEST MODE]' if smoke_test else ''}")
    print(f"  Penalty Weight:  {criterion.penalty_weight}")
    print(f"  Class-dist NLL:  {criterion.class_distribution_weight}")
    print(
        "  Hard-level NLL: "
        f"{criterion.hard_level_classification_weight}")
    print(
        "  Middle-class distribution multiplier "
        "(K/M/F/C x Small/Med-LM/Med-RD/Heavy): "
        f"{criterion.middle_class_distribution_multiplier.detach().cpu().tolist()}")
    print(
        "  Entry ranking weights: "
        f"{criterion.entry_ranking_weight.detach().cpu().tolist()}")
    print(
        "  Conditional L1/L2 ranking weights: "
        f"{criterion.conditional_l1_l2_ranking_weight.detach().cpu().tolist()}")
    print(
        f"  Ranking margin/fraction: {criterion.ranking_margin}/"
        f"{criterion.hard_negative_fraction}")
    print(
        "  MC Confidence:   "
        f"{bool(data_contract.get('label_uncertainty_enabled', False))}"
    )
    print(
        "  Mechanism supervision: "
        f"{mechanism_outputs_enabled}"
        + (
            " "
            f"(mode={'fixed_or' if mechanism_decomposition_enabled else 'auxiliary_only'}, "
            f"encoder={resolved_model_config['mechanism_encoder_mode']}, "
            f"aux={mechanism_auxiliary_weight}, "
            f"branch={mechanism_branch_weights.detach().cpu().tolist()}, "
            f"boundary={mechanism_boundary_focus_weight}, "
            f"hard={mechanism_hard_classification_weight}, "
            f"row_weight={mechanism_use_dataset_row_weights}, "
            f"class-dist={mechanism_class_distribution_weight})"
            if mechanism_outputs_enabled else ""
        )
    )
    print(
        "  Component supervision: "
        f"{component_outputs_enabled}"
        + (
            " "
            f"(targets={component_positive_weight.numel()}, "
            f"aux={component_auxiliary_weight}, "
            "target-tree-teacher="
            f"{component_target_tree_teacher_weight}, "
            f"rule={component_rule_consistency_weight}, "
            f"distribution={component_distribution_weight}, "
            "entry-ranking="
            f"{component_rule_entry_ranking_weight.detach().cpu().tolist()}, "
            "conditional-ranking="
            f"{component_rule_conditional_ranking_weight.detach().cpu().tolist()})"
            if component_outputs_enabled else ""
        )
    )
    print(f"  AMP Enabled:     {scaler_amp is not None}")
    print(f"  Selection Patience: {patience} (min epoch {min_selection_epochs})")
    print(f"{'='*50}\n")

    # [NEW] Track Metrics History for Plotting
    # [R12 + R13] 可观测维度：
    #   - 三类准确率：用"校准后阈值"反推 0/1/2 整体 accuracy（R13 改口径）
    #   - 全任务 F1 @thr=0.5（保留原始口径）+ F1 @tuned thr（R13 新增，业务展示用）
    #   - log_var 演化 / 各任务违反率 / LR / 单 epoch 耗时
    #   - [R13 新增] 全 8 头的 best threshold 轨迹（观察阈值是否稳定）
    #   - [R13 新增] per-munition × per-task 3-class accuracy
    history = {
        'train_loss': [], 'val_loss': [],
        'recall_K1': [], 'recall_K2': [], 'recall_M1': [], 'recall_M2': [],
        'violation_rate': [],
        # [R12 新增]
        'acc_K': [], 'acc_M': [], 'acc_F': [], 'acc_C': [],         # 3 类整体准确率（R13 起已切换到校准阈值）
        'f1_M1': [], 'f1_M2': [], 'f1_F1': [], 'f1_F2': [],         # F1 @ thr=0.5（保留原始口径）
        'f1_C1': [], 'f1_C2': [],
        'logvar_K': [], 'logvar_M': [], 'logvar_F': [], 'logvar_C': [],
        'viol_K': [], 'viol_M': [], 'viol_F': [], 'viol_C': [],
        'lr': [], 'epoch_time': [],
        # [R13 新增] 全 8 头的 tuned F1 与 best threshold
        'f1_K1_tuned_all': [], 'f1_K2_tuned': [],
        'f1_M1_tuned': [],     'f1_M2_tuned': [],
        'f1_F1_tuned': [],     'f1_F2_tuned': [],
        'f1_C1_tuned': [],     'f1_C2_tuned': [],
        'thr_K2': [], 'thr_M1': [], 'thr_M2': [],
        'thr_F1': [], 'thr_F2': [], 'thr_C1': [], 'thr_C2': [],
        # [R14 B] class-1 对角线 recall + composite 分项（用于绘图和 TensorBoard）
        'cls1_rec_K': [], 'cls1_rec_M': [], 'cls1_rec_F': [], 'cls1_rec_C': [],
        'composite_acc3': [], 'composite_cls1': [], 'composite_f1': [],
        'composite_total': [],
        'selection_score': [],
    }
    # [R12] 缓存最佳 epoch 的预测/标签，用于训练结束后绘制 4 张混淆矩阵
    best_predictions = {"K": None, "M": None, "F": None, "C": None,
                        "K_true": None, "M_true": None, "F_true": None, "C_true": None,
                        "epoch": None,
                        "val_loss": None,
                        "selection_score": None,
                        "model_variant": None,
                        "composite": None,
                        "small_m_diag": None,
                        "small_k_diag": None,
                        # [R13 新增] 最佳 epoch 的 per-munition × per-task 准确率矩阵 (4 tasks × 4 munitions)
                        "munition_acc_matrix": None,
                        "munition_samples": None,
                        "munition_thresholds": None,
                        "low_cls1_cells": []}

    t0 = time.time()
    for epoch in range(epochs):
        epoch_t0 = time.time()  # [R12] 记录单 epoch 耗时
        model.train()
        if freeze_base_model:
            # Frozen BatchNorm statistics and dropout masks must stay in
            # inference mode; only the newly added adapters are optimized.
            model.eval()
            model.residual_adapters.train()
        train_loss = 0.0
        train_task_losses = torch.zeros(4, device=device)
        train_mun_exposure = torch.zeros(4, dtype=torch.long)

        # Train Loop 更新
        # [R20] DataLoader 末位追加 y_soft (软标签). 解包顺序与 DamageDataset.__getitem__ 一致.
        for batch_idx, batch in enumerate(train_loader):
            (x, y, m_ids, weights, k_w, c_w, m_w, y_soft,
             label_confidence, _sample_ids, _root_ids) = batch[:11]
            mechanism_targets, component_targets = (
                _batch_auxiliary_targets(
                    batch,
                    mechanism_outputs_enabled,
                    component_outputs_enabled,
                ))
            x = x.to(device); y = y.to(device); m_ids = m_ids.to(device)
            weights = weights.to(device); k_w = k_w.to(device); c_w = c_w.to(device); m_w = m_w.to(device)
            y_soft = y_soft.to(device)
            label_confidence = label_confidence.to(device)
            if mechanism_targets is not None:
                mechanism_targets = mechanism_targets.to(device)
            if component_targets is not None:
                component_targets = component_targets.to(device)
            train_mun_exposure += torch.bincount(m_ids.detach().cpu(), minlength=4)
            # [P1 #5] 钳制范围放宽到 [0.05, 200]：
            #   - min=0.05 让 CB 中的"常见类降权"信号能保留 (原 min=1.0 抹平)
            #   - max=200 让极稀疏类 (n_pos<1000) 的高 CB 权重不被截断
            weights_clipped = torch.clamp(weights, min=0.05, max=200.0)

            optimizer.zero_grad()

            # [P2 #11 推荐] 仅模型前向走 AMP，criterion 含 log_vars 等 fp32 敏感参数
            # 显式退出 autocast 后用 fp32 计算 loss
            if scaler_amp is not None:
                with torch.amp.autocast('cuda'):
                    (
                        logits, fragment_logits, shock_logits,
                        component_logits,
                    ) = _forward_with_training_auxiliaries(
                        model, x, m_ids,
                        mechanism_outputs_enabled,
                        component_outputs_enabled,
                    )
                # criterion 在 fp32 下计算，避免 exp(-log_var) 的 fp16 精度损失
                # [P0-2] 透传 c_task_weight 给 C 分支
                # [R20] targets_soft=y_soft 让 BCE 直接用连续概率, y(硬) 仍用于 class-1 mask
                loss, task_losses = criterion(
                    logits.float(), y, weights_clipped,
                    k_task_weight=k_w, c_task_weight=c_w, m_task_weight=m_w, m_ids=m_ids,
                    targets_soft=y_soft,
                    target_confidence=label_confidence)
                if mechanism_outputs_enabled:
                    loss = loss + mechanism_auxiliary_weight * (
                        mechanism_auxiliary_loss(
                            fragment_logits.float(),
                            shock_logits.float(),
                            mechanism_targets,
                            weights_clipped,
                            model.ordinal_applicability[m_ids],
                            class_distribution_weight=(
                                mechanism_class_distribution_weight),
                            **mechanism_loss_options,
                        )
                    )
                if component_outputs_enabled:
                    loss = loss + component_auxiliary_weight * (
                        component_auxiliary_loss(
                            component_logits.float(),
                            component_targets,
                            weights_clipped,
                            component_positive_weight,
                            y_soft,
                            model.ordinal_applicability[m_ids],
                            deployed_logits=logits.float(),
                            target_tree_teacher_weight=(
                                component_target_tree_teacher_weight),
                            rule_consistency_weight=(
                                component_rule_consistency_weight),
                            distribution_weight=(
                                component_distribution_weight),
                            munition_ids=m_ids,
                            rule_entry_ranking_weight=(
                                component_rule_entry_ranking_weight),
                            rule_conditional_l1_l2_ranking_weight=(
                                component_rule_conditional_ranking_weight),
                            ranking_margin=(
                                component_rule_ranking_margin),
                            hard_negative_fraction=(
                                component_rule_hard_negative_fraction),
                        )
                    )
                scaler_amp.scale(loss).backward()
                # 解除缩放梯度的锁定，以便实施剪裁
                scaler_amp.unscale_(optimizer)
            else:
                (
                    logits, fragment_logits, shock_logits,
                    component_logits,
                ) = _forward_with_training_auxiliaries(
                    model, x, m_ids,
                    mechanism_outputs_enabled,
                    component_outputs_enabled,
                )
                loss, task_losses = criterion(
                    logits, y, weights_clipped,
                    k_task_weight=k_w, c_task_weight=c_w, m_task_weight=m_w, m_ids=m_ids,
                    targets_soft=y_soft,
                    target_confidence=label_confidence)
                if mechanism_outputs_enabled:
                    loss = loss + mechanism_auxiliary_weight * (
                        mechanism_auxiliary_loss(
                            fragment_logits,
                            shock_logits,
                            mechanism_targets,
                            weights_clipped,
                            model.ordinal_applicability[m_ids],
                            class_distribution_weight=(
                                mechanism_class_distribution_weight),
                            **mechanism_loss_options,
                        )
                    )
                if component_outputs_enabled:
                    loss = loss + component_auxiliary_weight * (
                        component_auxiliary_loss(
                            component_logits,
                            component_targets,
                            weights_clipped,
                            component_positive_weight,
                            y_soft,
                            model.ordinal_applicability[m_ids],
                            deployed_logits=logits,
                            target_tree_teacher_weight=(
                                component_target_tree_teacher_weight),
                            rule_consistency_weight=(
                                component_rule_consistency_weight),
                            distribution_weight=(
                                component_distribution_weight),
                            munition_ids=m_ids,
                            rule_entry_ranking_weight=(
                                component_rule_entry_ranking_weight),
                            rule_conditional_l1_l2_ranking_weight=(
                                component_rule_conditional_ranking_weight),
                            ranking_margin=(
                                component_rule_ranking_margin),
                            hard_negative_fraction=(
                                component_rule_hard_negative_fraction),
                        )
                    )
                loss.backward()

            # [修复点] 合并模型和独立 criterion 的参数空间进行统一梯度截断
            all_params = (
                trainable_model_parameters
                + trainable_criterion_parameters
            )
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)

            if scaler_amp is not None:
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                optimizer.step()

            train_loss += loss.item()
            train_task_losses += task_losses.detach()

        train_loss /= len(train_loader)

        if scheduler is not None:
            scheduler.step()

        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_task_losses = torch.zeros(4, device=device)

        # [R13] 重构：一次性收集全 val 集的 probs / targets / munition_ids 到 CPU，
        #   随后对全 8 头做阈值搜索，再用校准阈值统一计算 P/R/F1/3-class Acc/per-munition Acc。
        #   原因：R7 只校准 K1，导致 C1/M1/F1 等 pos_weight != 1 的任务在 thr=0.5 下
        #         F1 被低估（K1 默认 83.8% vs 校准 92.7%）。此处扩展到全 8 头一起校准。
        all_probs = []   # list of (B, 4, 2)
        all_tgts  = []   # list of (B, 4, 2)
        all_mids  = []   # list of (B,)

        with torch.no_grad():
            # [R20] val_loader 也要解包 y_soft (位置一致); 但 val loss 与 BCE 用什么标签无所谓,
            # 关键是别让 unpacking 出错. 这里把 y_soft 也传给 criterion 保持口径一致.
            for batch in val_loader:
                (x, y, m_ids, weights, k_w, c_w, m_w, y_soft,
                 label_confidence, _sample_ids, _root_ids) = batch[:11]
                mechanism_targets, component_targets = (
                    _batch_auxiliary_targets(
                        batch,
                        mechanism_outputs_enabled,
                        component_outputs_enabled,
                    ))
                x = x.to(device); y = y.to(device); m_ids = m_ids.to(device)
                weights = weights.to(device); k_w = k_w.to(device); c_w = c_w.to(device); m_w = m_w.to(device)
                y_soft = y_soft.to(device)
                label_confidence = label_confidence.to(device)
                if mechanism_targets is not None:
                    mechanism_targets = mechanism_targets.to(device)
                if component_targets is not None:
                    component_targets = component_targets.to(device)
                weights_clipped = torch.clamp(weights, min=0.05, max=200.0)

                if scaler_amp is not None:
                    with torch.amp.autocast('cuda'):
                        (
                            logits, fragment_logits, shock_logits,
                            component_logits,
                        ) = _forward_with_training_auxiliaries(
                            model, x, m_ids,
                            mechanism_outputs_enabled,
                            component_outputs_enabled,
                        )
                    # [P0-2] 透传 c_task_weight 给 C 分支
                    # [R20] val 也透传 y_soft, 与训练口径一致
                    loss, task_losses = criterion(
                        logits.float(), y, weights_clipped,
                        k_task_weight=k_w, c_task_weight=c_w, m_task_weight=m_w, m_ids=m_ids,
                        targets_soft=y_soft,
                        target_confidence=label_confidence)
                    if mechanism_outputs_enabled:
                        loss = loss + mechanism_auxiliary_weight * (
                            mechanism_auxiliary_loss(
                                fragment_logits.float(),
                                shock_logits.float(),
                                mechanism_targets,
                                weights_clipped,
                                model.ordinal_applicability[m_ids],
                                class_distribution_weight=(
                                    mechanism_class_distribution_weight),
                                **mechanism_loss_options,
                            )
                        )
                    if component_outputs_enabled:
                        loss = loss + component_auxiliary_weight * (
                            component_auxiliary_loss(
                                component_logits.float(),
                                component_targets,
                                weights_clipped,
                                component_positive_weight,
                                y_soft,
                                model.ordinal_applicability[m_ids],
                                deployed_logits=logits.float(),
                                target_tree_teacher_weight=(
                                    component_target_tree_teacher_weight),
                                rule_consistency_weight=(
                                    component_rule_consistency_weight),
                                distribution_weight=(
                                    component_distribution_weight),
                                munition_ids=m_ids,
                                rule_entry_ranking_weight=(
                                    component_rule_entry_ranking_weight),
                                rule_conditional_l1_l2_ranking_weight=(
                                    component_rule_conditional_ranking_weight),
                                ranking_margin=(
                                    component_rule_ranking_margin),
                                hard_negative_fraction=(
                                    component_rule_hard_negative_fraction),
                            )
                        )
                else:
                    (
                        logits, fragment_logits, shock_logits,
                        component_logits,
                    ) = _forward_with_training_auxiliaries(
                        model, x, m_ids,
                        mechanism_outputs_enabled,
                        component_outputs_enabled,
                    )
                    loss, task_losses = criterion(
                        logits, y, weights_clipped,
                        k_task_weight=k_w, c_task_weight=c_w, m_task_weight=m_w, m_ids=m_ids,
                        targets_soft=y_soft,
                        target_confidence=label_confidence)
                    if mechanism_outputs_enabled:
                        loss = loss + mechanism_auxiliary_weight * (
                            mechanism_auxiliary_loss(
                                fragment_logits,
                                shock_logits,
                                mechanism_targets,
                                weights_clipped,
                                model.ordinal_applicability[m_ids],
                                class_distribution_weight=(
                                    mechanism_class_distribution_weight),
                                **mechanism_loss_options,
                            )
                        )
                    if component_outputs_enabled:
                        loss = loss + component_auxiliary_weight * (
                            component_auxiliary_loss(
                                component_logits,
                                component_targets,
                                weights_clipped,
                                component_positive_weight,
                                y_soft,
                                model.ordinal_applicability[m_ids],
                                deployed_logits=logits,
                                target_tree_teacher_weight=(
                                    component_target_tree_teacher_weight),
                                rule_consistency_weight=(
                                    component_rule_consistency_weight),
                                distribution_weight=(
                                    component_distribution_weight),
                                munition_ids=m_ids,
                                rule_entry_ranking_weight=(
                                    component_rule_entry_ranking_weight),
                                rule_conditional_l1_l2_ranking_weight=(
                                    component_rule_conditional_ranking_weight),
                                ranking_margin=(
                                    component_rule_ranking_margin),
                                hard_negative_fraction=(
                                    component_rule_hard_negative_fraction),
                            )
                        )

                val_loss += loss.item()
                val_task_losses += task_losses.detach()

                all_probs.append(torch.sigmoid(logits).float().cpu())   # (B, 4, 2)
                all_tgts.append(y.float().cpu())                         # (B, 4, 2)
                all_mids.append(m_ids.cpu())                             # (B,)

        val_loss /= len(val_loader)
        val_task_losses /= len(val_loader)

        # === [R13] 统一张量：一次 cat 之后所有指标都是切片运算 ===
        probs_all = torch.cat(all_probs, dim=0)    # (N_val, 4, 2)
        tgts_all  = torch.cat(all_tgts,  dim=0)    # (N_val, 4, 2)
        mids_all  = torch.cat(all_mids,  dim=0)    # (N_val,)
        N_val     = probs_all.size(0)
        t_samples = N_val  # 维持旧变量名，下游打印仍在用

        # ---- 物理违反率：直接在 probs 上算（在任何阈值/钳制之前的原始情况）----
        viol_mask_all   = probs_all[:, :, 1] > probs_all[:, :, 0]   # (N, 4)
        violation_count = int(viol_mask_all.sum().item())
        violation_per_task = viol_mask_all.sum(dim=0).long()        # (4,)
        violation_rate = violation_count / (N_val * 4) * 100
        viol_K, viol_M, viol_F, viol_C = [
            violation_per_task[i].item() / max(N_val, 1) * 100 for i in range(4)]

        # ---- K2 全预测正比率（基于 thr=0.5 的老口径，保持历史曲线可比）----
        k2_pred_pos = int(((probs_all[:, 0, 1] > 0.5)).sum().item())
        k2_pred_ratio = k2_pred_pos / max(N_val, 1) * 100

        # ================================================================
        # [R13] 阈值搜索：对全 8 头 (K1/K2/M1/M2/F1/F2/C1/C2) 各自扫 0.10~0.90
        # ================================================================
        TASK_NAMES = ["K", "M", "F", "C"]
        # Must match _evaluate_selection_snapshot exactly.
        thr_grid = [i / 100.0 for i in range(2, 100, 2)] + [1.0]

        def _best_f1_thr(p_vec: torch.Tensor, t_vec: torch.Tensor):
            """返回 (best_thr, best_f1, best_p, best_r)；空正样本时回退 0.5。"""
            n_pos = int(t_vec.sum().item())
            n_neg = t_vec.numel() - n_pos
            if n_pos == 0 or n_neg == 0:
                return 0.5, 0.0, 0.0, 0.0
            t = t_vec.long()
            best = (0.5, 0.0, 0.0, 0.0)
            for thr in thr_grid:
                pred = (p_vec >= thr).long()
                tp_ = int(((pred == 1) & (t == 1)).sum().item())
                fp_ = int(((pred == 1) & (t == 0)).sum().item())
                fn_ = int(((pred == 0) & (t == 1)).sum().item())
                p_  = tp_ / max(tp_ + fp_, 1)
                r_  = tp_ / max(tp_ + fn_, 1)
                f_  = 2 * p_ * r_ / max(p_ + r_, 1e-9)
                if f_ > best[1]:
                    best = (thr, f_, p_, r_)
            return best

        def _bin_f1(pred_long: torch.Tensor, t_long: torch.Tensor) -> float:
            """向量化二分类 F1。"""
            tp_ = int(((pred_long == 1) & (t_long == 1)).sum().item())
            fp_ = int(((pred_long == 1) & (t_long == 0)).sum().item())
            fn_ = int(((pred_long == 0) & (t_long == 1)).sum().item())
            if tp_ + fp_ == 0 or tp_ + fn_ == 0:
                return 0.0
            p_ = tp_ / (tp_ + fp_)
            r_ = tp_ / (tp_ + fn_)
            return 2 * p_ * r_ / max(p_ + r_, 1e-9)

        def _best_joint_thr(p1_vec: torch.Tensor, p2_vec: torch.Tensor,
                            t1_vec: torch.Tensor, t2_vec: torch.Tensor,
                            alpha: float = 0.80,
                            thr2_min_slack: float = 0.10,
                            max_l0_fp_rate: float = None):
            """Delegate to the one shared training/final calibration policy."""
            return _search_joint_ordinal_thresholds(
                p1_vec, p2_vec, t1_vec, t2_vec, thr_grid,
                alpha=alpha,
                thr2_min_slack=thr2_min_slack,
                max_l0_fp_rate=max_l0_fp_rate,
                minimum_exact_l1_recall=minimum_exact_l1_recall,
                maximum_accuracy_drop=(
                    maximum_class1_floor_accuracy_drop),
                minimum_three_class_accuracy=(
                    minimum_goal_cell_accuracy),
                minimum_class_diagonal_recall=(
                    minimum_goal_class_diagonal_recall),
            )

        # best_thr_matrix 维度：行=K/M/F/C 列=L1/L2
        def _best_l1_only_thr(p1_vec: torch.Tensor, t1_vec: torch.Tensor,
                              max_fp_rate: float = None,
                              recall_weight: float = 0.7):
            """Search a single L1 threshold when L2 positives are absent/sparse."""
            return _search_l1_threshold(
                p1_vec, t1_vec, thr_grid,
                max_fp_rate=max_fp_rate,
                recall_weight=recall_weight,
                minimum_recall=minimum_exact_l1_recall,
                maximum_accuracy_drop=(
                    maximum_class1_floor_accuracy_drop),
                minimum_accuracy=minimum_goal_cell_accuracy,
                minimum_negative_recall=(
                    minimum_goal_class_diagonal_recall),
            )

        best_thr_matrix = torch.full((4, 2), 0.5, dtype=torch.float32)
        tuned_f1_matrix = torch.zeros((4, 2), dtype=torch.float32)
        tuned_p_matrix  = torch.zeros((4, 2), dtype=torch.float32)
        tuned_r_matrix  = torch.zeros((4, 2), dtype=torch.float32)
        # [R14 A] 联合阈值搜索：按任务维度搜 (thr1, thr2) 最大化 3-class Acc + F1 平均
        for i in range(4):
            p1_i = probs_all[:, i, 0]
            p2_i = probs_all[:, i, 1]
            t1_i = tgts_all[:, i, 0]
            t2_i = tgts_all[:, i, 1]
            # [R16] α 0.70 → 0.80: 与 per-munition pos_weight 配套, 让阈值搜索
            # 进一步偏向 3-class accuracy. 之前 α=0.70 在 M 任务下选出全 ≤0.50
            # 的低阈值 (Small×M1=0.46, Heavy×M2=0.34), 把 Small × M=0 的边界样本
            # 大量误判为 L≥1, 拉低 Small×M acc 到 ~89%. 提到 0.80 后, 阈值会
            # 略微抬升 → 减少 L0 假阳, 同时 F1 项保留 20% 权重防退化为全预测 0.
            thr1_j, thr2_j, J_j, acc3_j, f1_h1_j, f1_h2_j = _best_joint_thr(
                p1_i, p2_i, t1_i, t2_i, alpha=0.80,
                max_l0_fp_rate=(
                    GLOBAL_C0_MAX_FP_RATE if i == 3 else None))
            best_thr_matrix[i, 0] = thr1_j
            best_thr_matrix[i, 1] = thr2_j
            # 用联合最优阈值反算 P/R/F1，保证日志数值与 3-class Acc 口径自洽
            p2_cl = torch.minimum(p2_i, p1_i)
            pred1 = (p1_i >= thr1_j).long()
            pred2 = ((p1_i >= thr1_j) & (p2_cl >= thr2_j)).long()
            for j, (pred_j, t_j) in enumerate([(pred1, t1_i.long()), (pred2, t2_i.long())]):
                tp_ = int(((pred_j == 1) & (t_j == 1)).sum().item())
                fp_ = int(((pred_j == 1) & (t_j == 0)).sum().item())
                fn_ = int(((pred_j == 0) & (t_j == 1)).sum().item())
                p_  = tp_ / max(tp_ + fp_, 1)
                r_  = tp_ / max(tp_ + fn_, 1)
                f_  = 2 * p_ * r_ / max(p_ + r_, 1e-9)
                tuned_f1_matrix[i, j] = f_
                tuned_p_matrix[i, j]  = p_
                tuned_r_matrix[i, j]  = r_

        # 为了向后兼容保留 best_thr / best_k1_* 几个老变量（打印 / composite 会用到）
        best_thr     = float(best_thr_matrix[0, 0].item())          # K1 的最佳阈值
        best_k1_f1   = float(tuned_f1_matrix[0, 0].item())
        best_k1_p    = float(tuned_p_matrix[0, 0].item())
        best_k1_r    = float(tuned_r_matrix[0, 0].item())

        # ================================================================
        # [P0-3 + P0-3-fix] Per-munition 阈值搜索：在每个 (task, m_id) 单元上
        #   独立搜 (thr1, thr2)，但正样本数过少时必须回退到全局阈值。
        #   原初版本仅检查该弹型总样本数 (n_m >= 30)，对 Heavy×M1 这种
        #   "弹型总样本大、但 M1 正样本仅 71 个" 的情况会搜出不稳定的 thr
        #   (thr2=0.32 < thr1=0.36)，让 class-1 区间只剩 4% 宽，Heavy×M1 从
        #   33% 掉到 16.7%。
        #   P0-3-fix：改为 **要求 head1 和 head2 的正样本数都 ≥ 50** 才做 per-mun
        #   搜索，否则回退全局阈值。
        # best_thr_matrix_perm shape: (4 tasks, 4 munitions, 2 levels)
        # ================================================================
        best_thr_matrix_perm = best_thr_matrix.unsqueeze(1).expand(-1, 4, -1).clone()
        rare_cell_thresholds = {}
        perm_fallback_hits = 0
        perm_l1_only_hits = 0
        for i in range(4):
            for m_id in range(4):
                mask = (mids_all == m_id)
                n_cell = int(mask.sum().item())
                min_pos_l1 = _cell_l1_min_pos(i, m_id)
                if n_cell < 30:
                    meta = _rare_cell_meta(
                        i, m_id, n_cell, 0, 0, "fallback_too_few_samples",
                        float(best_thr_matrix_perm[i, m_id, 0].item()),
                        float(best_thr_matrix_perm[i, m_id, 1].item()))
                    if meta:
                        rare_cell_thresholds[meta["name"]] = meta
                    perm_fallback_hits += 1
                    continue
                p1_m = probs_all[mask, i, 0]
                p2_m = probs_all[mask, i, 1]
                t1_m = tgts_all[mask, i, 0]
                t2_m = tgts_all[mask, i, 1]
                n_pos_1 = int(t1_m.sum().item())
                n_pos_2 = int(t2_m.sum().item())
                if not _is_applicable(i, m_id, 1):
                    if n_pos_2:
                        raise RuntimeError(
                            f"Structural-zero cell contains positives: "
                            f"{MUN_NAMES[m_id]}/{TASK_NAMES[i]}>=2")
                    if n_pos_1 >= min_pos_l1:
                        max_fp, recall_weight = _cell_l1_search_params(i, m_id)
                        thr1_m, *_ = _best_l1_only_thr(
                            p1_m, t1_m, max_fp_rate=max_fp,
                            recall_weight=recall_weight)
                        best_thr_matrix_perm[i, m_id, 0] = thr1_m
                    best_thr_matrix_perm[i, m_id, 1] = 1.0
                    meta = _rare_cell_meta(
                        i, m_id, n_cell, n_pos_1, n_pos_2,
                        "structural_zero_L2",
                        float(best_thr_matrix_perm[i, m_id, 0].item()), 1.0)
                    rare_cell_thresholds[meta["name"]] = meta
                    continue
                # V5: if L1 has enough positives but L2 does not, calibrate L1
                # independently and keep L2 effectively closed.
                if n_pos_1 < min_pos_l1:
                    meta = _rare_cell_meta(
                        i, m_id, n_cell, n_pos_1, n_pos_2, "fallback_too_few_L1",
                        float(best_thr_matrix_perm[i, m_id, 0].item()),
                        float(best_thr_matrix_perm[i, m_id, 1].item()))
                    if meta:
                        rare_cell_thresholds[meta["name"]] = meta
                    perm_fallback_hits += 1
                    continue
                if n_pos_2 < DEFAULT_PER_MUN_MIN_POS:
                    max_fp, recall_weight = _cell_l1_search_params(i, m_id)
                    thr1_m, *_ = _best_l1_only_thr(
                        p1_m, t1_m, max_fp_rate=max_fp, recall_weight=recall_weight)
                    best_thr_matrix_perm[i, m_id, 0] = thr1_m
                    meta = _rare_cell_meta(
                        i, m_id, n_cell, n_pos_1, n_pos_2,
                        "l1_only_global_L2",
                        float(best_thr_matrix_perm[i, m_id, 0].item()),
                        float(best_thr_matrix_perm[i, m_id, 1].item()))
                    if meta:
                        rare_cell_thresholds[meta["name"]] = meta
                    perm_l1_only_hits += 1
                    continue
                # [R16] 与全局搜索一致, α 提到 0.80
                max_fp, _ = _cell_l1_search_params(i, m_id)
                thr1_m, thr2_m, *_ = _best_joint_thr(
                    p1_m, p2_m, t1_m, t2_m, alpha=0.80,
                    max_l0_fp_rate=max_fp)
                best_thr_matrix_perm[i, m_id, 0] = thr1_m
                best_thr_matrix_perm[i, m_id, 1] = thr2_m
                meta = _rare_cell_meta(
                    i, m_id, n_cell, n_pos_1, n_pos_2, "joint",
                    float(best_thr_matrix_perm[i, m_id, 0].item()),
                    float(best_thr_matrix_perm[i, m_id, 1].item()))
                if meta:
                    rare_cell_thresholds[meta["name"]] = meta

        # ================================================================
        # [R13] P/R/F1 @ thr=0.5（为保持历史曲线可比沿用，但不再作为业务展示主口径）
        # ================================================================
        preds_05 = (probs_all > 0.5).long()                        # (N, 4, 2)
        tgt_long = tgts_all.long()
        preds_flat = preds_05.reshape(N_val, -1)                    # (N, 8)  K1 K2 M1 M2 F1 F2 C1 C2
        tgt_flat   = tgt_long.reshape(N_val, -1)
        tp = ((preds_flat == 1) & (tgt_flat == 1)).sum(dim=0).float()
        fp = ((preds_flat == 1) & (tgt_flat == 0)).sum(dim=0).float()
        fn = ((preds_flat == 0) & (tgt_flat == 1)).sum(dim=0).float()
        precision = tp / (tp + fp).clamp(min=1.0)
        recall    = tp / (tp + fn).clamp(min=1.0)
        f1        = 2 * precision * recall / (precision + recall).clamp(min=1e-9)
        p_K1, p_K2, p_M1, p_M2, p_F1, p_F2, p_C1, p_C2 = precision.tolist()
        r_K1, r_K2, r_M1, r_M2, r_F1, r_F2, r_C1, r_C2 = recall.tolist()
        f_K1, f_K2, f_M1, f_M2, f_F1, f_F2, f_C1, f_C2 = f1.tolist()

        # ================================================================
        # [P0-3] 3-class Ordinal Acc —— 用 per-munition 阈值重算
        #   每个样本根据其 m_id 查对应 (task, mun) 的 (thr1, thr2)
        #   shape 链：best_thr_matrix_perm (4,4,2) → gather 得 (4, N, 2) → permute (N, 4, 2)
        # ================================================================
        thr_gather = best_thr_matrix_perm[:, mids_all, :]           # (4, N, 2)
        thr_per_sample = thr_gather.permute(1, 0, 2).contiguous()   # (N, 4, 2)
        thr1_ps = thr_per_sample[:, :, 0]                           # (N, 4)
        thr2_ps = thr_per_sample[:, :, 1]                           # (N, 4)
        p1 = probs_all[:, :, 0]                                     # (N, 4)
        p2 = torch.minimum(probs_all[:, :, 1], p1)                  # 物理保序硬钳制
        pass1 = (p1 >= thr1_ps)                                     # (N, 4)
        pass2 = (p1 >= thr1_ps) & (p2 >= thr2_ps)
        pred_level = pass1.long() + pass2.long()                    # (N, 4), ∈{0,1,2}
        true_level = tgts_all[:, :, 0].long() + tgts_all[:, :, 1].long()
        c0_mask = (true_level[:, 3] == 0)
        if int(c0_mask.sum().item()) > 0:
            c0_fp_rate = float((pred_level[c0_mask, 3] > 0).float().mean().item())
        else:
            c0_fp_rate = 0.0
        small_c0_mask = c0_mask & (mids_all == 0)
        if int(small_c0_mask.sum().item()) > 0:
            small_c0_fp_rate = float((pred_level[small_c0_mask, 3] > 0).float().mean().item())
        else:
            small_c0_fp_rate = 0.0

        correct_mask = (pred_level == true_level)                   # (N, 4) bool
        correct_per_task = correct_mask.sum(dim=0).long()           # (4,)

        # ================================================================
        # [R14 B] class-1 对角线 recall  (4,)  ——  新 composite 指标核心分量
        #   在 R13 训练中观察到 K/M/C 的 class-1 对角线塌陷到 31% / 68.7% / 78.9%；
        #   composite 若只奖励 per-head F1 根本感知不到这个塌陷。此处显式算出
        #   class-1 样本的 recall，作为早停 / best-epoch 的新分项。
        # ================================================================
        cls1_recall_vec = torch.zeros(4, dtype=torch.float32)
        for i in range(4):
            mask_c1 = (true_level[:, i] == 1)
            n_c1 = int(mask_c1.sum().item())
            if n_c1 > 0:
                cls1_recall_vec[i] = (pred_level[mask_c1, i] == 1).float().mean() * 100.0

        # ================================================================
        # [R14 B] 按弹型 × 任务的 class-1 对角线 recall 热力图矩阵 (4 task, 4 mun)
        #   暴露 "哪一个弹型在哪一个任务上" 的中间类塌陷最严重
        # ================================================================
        cls1_recall_per_mun = torch.zeros(4, 4, dtype=torch.float32)  # [task, mun]
        cls1_count_per_mun = torch.zeros(4, 4, dtype=torch.long)
        for i in range(4):
            for m_id in range(4):
                mask_c1m = (true_level[:, i] == 1) & (mids_all == m_id)
                n_c1m = int(mask_c1m.sum().item())
                cls1_count_per_mun[i, m_id] = n_c1m
                if n_c1m > 0:
                    cls1_recall_per_mun[i, m_id] = \
                        (pred_level[mask_c1m, i] == 1).float().mean() * 100.0

        # ================================================================
        # [R13 NEW] 按弹种拆分的任务准确率矩阵  acc_per_munition[task, munition]
        #   [R15] 提前到 composite 之前计算，因为 composite 现在要用 min-cell 惩罚
        # ================================================================
        acc_per_munition = torch.zeros(4, 4, dtype=torch.float32)
        samples_per_munition = torch.zeros(4, dtype=torch.long)
        for m_id in range(4):
            mask = (mids_all == m_id)
            n_m = int(mask.sum().item())
            samples_per_munition[m_id] = n_m
            if n_m > 0:
                corr = correct_mask[mask].sum(dim=0).float()          # (4,)
                acc_per_munition[:, m_id] = corr / n_m * 100.0

        # Small x M is the most sensitive boundary cell. Track its full
        # 3-class confusion pattern so L0->L1 drift is visible during training.
        small_m_mask = (mids_all == 0)
        small_m_diag = {
            "n": int(small_m_mask.sum().item()),
            "acc": 0.0,
            "cm": torch.zeros(3, 3, dtype=torch.long),
            "true_counts": torch.zeros(3, dtype=torch.long),
            "pred_counts": torch.zeros(3, dtype=torch.long),
            "l0_to_l1": 0.0,
            "l0_to_l2": 0.0,
            "l1_to_l0": 0.0,
            "l1_to_l2": 0.0,
            "l2_to_l1": 0.0,
        }
        if small_m_diag["n"] > 0:
            sm_true = true_level[small_m_mask, 1].long()
            sm_pred = pred_level[small_m_mask, 1].long()
            sm_cm = torch.bincount(sm_true * 3 + sm_pred, minlength=9).reshape(3, 3)
            sm_true_counts = sm_cm.sum(dim=1)
            sm_pred_counts = sm_cm.sum(dim=0)

            def _cell_rate(t_cls: int, p_cls: int) -> float:
                denom = int(sm_true_counts[t_cls].item())
                if denom <= 0:
                    return 0.0
                return float(sm_cm[t_cls, p_cls].item()) / denom * 100.0

            small_m_diag.update({
                "acc": float((sm_pred == sm_true).float().mean().item()) * 100.0,
                "cm": sm_cm,
                "true_counts": sm_true_counts,
                "pred_counts": sm_pred_counts,
                "l0_to_l1": _cell_rate(0, 1),
                "l0_to_l2": _cell_rate(0, 2),
                "l1_to_l0": _cell_rate(1, 0),
                "l1_to_l2": _cell_rate(1, 2),
                "l2_to_l1": _cell_rate(2, 1),
            })
            sm_p1 = probs_all[small_m_mask, 1, 0]
            sm_p2 = torch.minimum(probs_all[small_m_mask, 1, 1], sm_p1)
            prob_by_true = {}
            for lv in range(3):
                lv_mask = (sm_true == lv)
                if int(lv_mask.sum().item()) == 0:
                    continue
                p1_lv = sm_p1[lv_mask]
                p2_lv = sm_p2[lv_mask]
                prob_by_true[lv] = {
                    "p1_mean": float(p1_lv.mean().item()),
                    "p1_p50": float(torch.quantile(p1_lv, 0.50).item()),
                    "p1_p90": float(torch.quantile(p1_lv, 0.90).item()),
                    "p2_mean": float(p2_lv.mean().item()),
                    "p2_p50": float(torch.quantile(p2_lv, 0.50).item()),
                    "p2_p90": float(torch.quantile(p2_lv, 0.90).item()),
                }
            small_m_diag["prob_by_true"] = prob_by_true
        small_k_diag = _build_cell_diag(
            true_level, pred_level, probs_all, mids_all, task_idx=0, mun_id=0)

        # ================================================================
        # [R15] Composite 指标重新平衡（替换 R14-B 的 0.50/0.25/0.20/-0.05 配方）
        #   R14-B 实证下：best@epoch 16 时 3-class Acc 全线低于 R13；Heavy × M
        #   class-1=22.2% / Small × M 3-class=88.2% 严重塌陷没有被感知，因为
        #   composite 只看宏平均 cls1=66%（仍然算"进步"）。
        #   R15 做两件事：
        #   (1) 把 acc3 权重从 0.50 → 0.60，cls1 从 0.25 → 0.15 —— 3-class Acc
        #       是用户的首要目标，cls1 已经被 Part-A 的联合阈值单调约束间接
        #       保护，权重不需要那么高。
        #   (2) 加 "最弱格子惩罚"：对 16 格 (task × munition) 里最低的 3-class
        #       Acc 格，若低于 95% 就线性惩罚 composite，系数 0.30；这样
        #       Heavy × M 一塌陷 composite 立刻往下拉，best-epoch 不会选中。
        # ================================================================
        acc3_mean = float(correct_mask.float().mean().item())              # [0,1]
        cls1_recall_mean = float(cls1_recall_vec.mean().item()) / 100.0    # [0,1]
        f1_mean = float(tuned_f1_matrix.mean().item())                     # [0,1]
        min_cell_acc_3class = float(acc_per_munition.min().item()) / 100.0  # 16 格中的最低
        min_cell_penalty = 0.30 * max(0.0, 0.95 - min_cell_acc_3class)     # 线性 hinge
        composite_breakdown = {
            "acc3":     acc3_mean,
            "cls1":     cls1_recall_mean,
            "f1":       f1_mean,
            "viol":     violation_rate,
            "min_cell": min_cell_acc_3class,
            "min_pen":  min_cell_penalty,
        }
        composite = (0.60 * acc3_mean
                     + 0.15 * cls1_recall_mean
                     + 0.20 * f1_mean
                     - 0.05 * violation_rate / 100.0
                     - min_cell_penalty)

        small_m_acc_score = small_m_diag["acc"] / 100.0
        small_k1_recall_score = float(cls1_recall_per_mun[0, 0].item()) / 100.0
        small_c1_recall_score = float(cls1_recall_per_mun[3, 0].item()) / 100.0
        non_target_mask = ~TARGET_CELL_MASK
        non_target_cell_acc_mean = float(acc_per_munition[non_target_mask].mean().item()) / 100.0
        non_target_drop_penalty = 0.0
        non_target_cls1_drop_penalty = 0.0
        low_cls1_cells = _collect_low_cls1_cells(
            cls1_recall_per_mun, cls1_count_per_mun, non_target_mask,
            floor=100.0 * float(minimum_exact_l1_recall))
        class1_floor_penalty = (
            max((cell["deficit"] for cell in low_cls1_cells), default=0.0) / 100.0)
        min_supported_class1_recall = _minimum_supported_class1_recall(
            cls1_recall_per_mun, cls1_count_per_mun)
        min_supported_class_diagonal_recall = (
            _minimum_supported_diagonal_recall(
                pred_level, true_level, mids_all))
        goal_class1_floor_penalty = max(
            0.0,
            GOAL_MIN_CLASS_DIAGONAL_RECALL_PERCENT / 100.0
            - min_supported_class1_recall,
        )
        goal_cell_accuracy_penalty = 2.0 * max(
            0.0,
            GOAL_MIN_CELL_3CLASS_ACCURACY_PERCENT / 100.0
            - min_cell_acc_3class,
        )
        goal_diagonal_recall_penalty = max(
            0.0,
            GOAL_MIN_CLASS_DIAGONAL_RECALL_PERCENT / 100.0
            - min_supported_class_diagonal_recall,
        )
        c0_guard_rate = max(c0_fp_rate, small_c0_fp_rate)
        c0_fp_penalty = max(0.0, c0_guard_rate - 0.025) * 2.0
        small_k0_mask = (mids_all == 0) & (true_level[:, 0] == 0)
        small_k0_fp_rate = (
            float((pred_level[small_k0_mask, 0] > 0).float().mean().item())
            if int(small_k0_mask.sum().item()) > 0 else 0.0)
        small_k0_fp_penalty = max(0.0, small_k0_fp_rate - 0.005) * 2.0
        guardrail_penalty = (
            non_target_drop_penalty + non_target_cls1_drop_penalty
            + class1_floor_penalty + goal_class1_floor_penalty
            + goal_cell_accuracy_penalty + goal_diagonal_recall_penalty
            + c0_fp_penalty + small_k0_fp_penalty)
        selection_score = (0.37 * acc3_mean
                           + 0.14 * f1_mean
                           + 0.09 * cls1_recall_mean
                           + 0.11 * small_m_acc_score
                           + 0.11 * small_c1_recall_score
                           + 0.08 * small_k1_recall_score
                           + 0.10 * non_target_cell_acc_mean
                           - guardrail_penalty)
        composite_breakdown.update({
            "small_m_acc": small_m_acc_score,
            "small_k1_recall": small_k1_recall_score,
            "small_c1_recall": small_c1_recall_score,
            "non_target_cell_acc_mean": non_target_cell_acc_mean,
            "c0_fp": c0_fp_rate,
            "small_c0_fp": small_c0_fp_rate,
            "small_k0_fp": small_k0_fp_rate,
            "guard_pen": guardrail_penalty,
            "non_target_drop_pen": non_target_drop_penalty,
            "non_target_cls1_drop_pen": non_target_cls1_drop_penalty,
            "class1_floor_pen": class1_floor_penalty,
            "goal_class1_floor_pen": goal_class1_floor_penalty,
            "goal_cell_accuracy_pen": goal_cell_accuracy_penalty,
            "min_supported_class1_recall": min_supported_class1_recall,
            "goal_diagonal_recall_pen": goal_diagonal_recall_penalty,
            "min_supported_class_diagonal_recall": (
                min_supported_class_diagonal_recall),
            "small_k0_fp_pen": small_k0_fp_penalty,
            "selection": selection_score,
        })

        # [R12] 缓存供 epoch 末构建混淆矩阵（最佳 epoch 用上）
        epoch_pred_levels = {ti: [pred_level[:, ti].numpy()] for ti in range(4)}
        epoch_true_levels = {ti: [true_level[:, ti].numpy()] for ti in range(4)}

        history.setdefault('best_thr_K1', []).append(best_thr)
        history.setdefault('f1_K1_tuned', []).append(best_k1_f1 * 100)

        # Append histories
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['recall_K1'].append(r_K1 * 100)
        history['recall_K2'].append(r_K2 * 100)
        history['recall_M1'].append(r_M1 * 100)
        history['recall_M2'].append(r_M2 * 100)
        history['violation_rate'].append(violation_rate)
        history.setdefault('precision_K2', []).append(p_K2 * 100)
        history.setdefault('f1_K2', []).append(f_K2 * 100)
        history.setdefault('k2_pred_ratio', []).append(k2_pred_ratio)
        history.setdefault('f1_K1_default', []).append(f_K1 * 100)

        # [R12] === 新维度 1：3 类整体准确率 ===
        acc_per_task = (correct_per_task.float() / max(t_samples, 1) * 100).tolist()
        history['acc_K'].append(acc_per_task[0])
        history['acc_M'].append(acc_per_task[1])
        history['acc_F'].append(acc_per_task[2])
        history['acc_C'].append(acc_per_task[3])

        # [R12] === 新维度 2：补全 M/F/C × L1/L2 的 F1 ===
        history['f1_M1'].append(f_M1 * 100)
        history['f1_M2'].append(f_M2 * 100)
        history['f1_F1'].append(f_F1 * 100)
        history['f1_F2'].append(f_F2 * 100)
        history['f1_C1'].append(f_C1 * 100)
        history['f1_C2'].append(f_C2 * 100)

        # [R12] === 新维度 3：log_var 演化（Kendall 不确定性收敛轨迹）===
        history['logvar_K'].append(criterion.log_vars[0].item())
        history['logvar_M'].append(criterion.log_vars[1].item())
        history['logvar_F'].append(criterion.log_vars[2].item())
        history['logvar_C'].append(criterion.log_vars[3].item())

        # [R12] === 新维度 4：单任务违反率 ===
        history['viol_K'].append(viol_K)
        history['viol_M'].append(viol_M)
        history['viol_F'].append(viol_F)
        history['viol_C'].append(viol_C)

        # [R12] === 新维度 5：学习率（Cosine 退火）===
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # [R13] === 新维度 6：全 8 头的 tuned F1 + best threshold ===
        # best_thr_matrix / tuned_f1_matrix 形状 (4, 2)  行=K/M/F/C  列=L1/L2
        history['f1_K1_tuned_all'].append(tuned_f1_matrix[0, 0].item() * 100)
        history['f1_K2_tuned']    .append(tuned_f1_matrix[0, 1].item() * 100)
        history['f1_M1_tuned']    .append(tuned_f1_matrix[1, 0].item() * 100)
        history['f1_M2_tuned']    .append(tuned_f1_matrix[1, 1].item() * 100)
        history['f1_F1_tuned']    .append(tuned_f1_matrix[2, 0].item() * 100)
        history['f1_F2_tuned']    .append(tuned_f1_matrix[2, 1].item() * 100)
        history['f1_C1_tuned']    .append(tuned_f1_matrix[3, 0].item() * 100)
        history['f1_C2_tuned']    .append(tuned_f1_matrix[3, 1].item() * 100)
        history['thr_K2'].append(best_thr_matrix[0, 1].item())
        history['thr_M1'].append(best_thr_matrix[1, 0].item())
        history['thr_M2'].append(best_thr_matrix[1, 1].item())
        history['thr_F1'].append(best_thr_matrix[2, 0].item())
        history['thr_F2'].append(best_thr_matrix[2, 1].item())
        history['thr_C1'].append(best_thr_matrix[3, 0].item())
        history['thr_C2'].append(best_thr_matrix[3, 1].item())

        # [R14 B] === 新维度：class-1 对角线 recall + composite 分项曲线 ===
        history['cls1_rec_K'].append(float(cls1_recall_vec[0].item()))
        history['cls1_rec_M'].append(float(cls1_recall_vec[1].item()))
        history['cls1_rec_F'].append(float(cls1_recall_vec[2].item()))
        history['cls1_rec_C'].append(float(cls1_recall_vec[3].item()))
        # composite 分项用于 training_history.png 的一个独立面板
        history['composite_acc3'].append(acc3_mean * 100)            # 百分制便于看图
        history['composite_cls1'].append(cls1_recall_mean * 100)
        history['composite_f1'].append(f1_mean * 100)
        history['composite_total'].append(composite)                 # 原始 [0,1] 浮点
        history['selection_score'].append(selection_score)
        history.setdefault('small_m_acc', []).append(small_m_diag["acc"])
        history.setdefault('small_m_l0_to_l1', []).append(small_m_diag["l0_to_l1"])
        history.setdefault('small_m_l0_to_l2', []).append(small_m_diag["l0_to_l2"])
        history.setdefault('small_m_l1_to_l0', []).append(small_m_diag["l1_to_l0"])
        history.setdefault('small_m_l1_to_l2', []).append(small_m_diag["l1_to_l2"])
        history.setdefault('small_m_l2_to_l1', []).append(small_m_diag["l2_to_l1"])
        history.setdefault('small_k_acc', []).append(small_k_diag["acc"])
        history.setdefault('small_k_l0_to_l1', []).append(small_k_diag["l0_to_l1"])
        history.setdefault('small_k_l1_to_l0', []).append(small_k_diag["l1_to_l0"])
        history.setdefault('small_k1_recall', []).append(small_k1_recall_score * 100.0)
        history.setdefault('small_k0_fp', []).append(small_k0_fp_rate * 100.0)


        # 汇报学习进度，含方差衍生与召回指标
        # 同时打印 raw / safe，按真实的分任务下限与统一上限显示。
        def _fmt_logvar(v, task_index):
            raw = v.item()
            lower = float(criterion.log_var_lower[task_index].item())
            safe = max(lower, min(2.5, raw))
            return f"{raw:+.2f}/{safe:+.2f}"
        variance_logstr = " | ".join(
            [f"s_{i}:{_fmt_logvar(criterion.log_vars[i], i)}" for i in range(4)])
        lr_current = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:03d}/{epochs} | LR: {lr_current:.2e} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Var: [{variance_logstr}]")

        # [NEW] 拆解任务损失与核心指标召回/精准率/F1
        val_tL = val_task_losses.tolist()
        print(f"    -> Val Tasks | Val_K:{val_tL[0]:.4f} Val_M:{val_tL[1]:.4f} Val_F:{val_tL[2]:.4f} Val_C:{val_tL[3]:.4f}")
        # [P1-A+] 取消 thr=0.5 口径的 K/M P/R/F1 打印；K2_pred_ratio 提示迁移到 tuned 阈值上
        tuned_p_pct  = (tuned_p_matrix  * 100).tolist()
        tuned_r_pct  = (tuned_r_matrix  * 100).tolist()
        tuned_f1_pct = (tuned_f1_matrix * 100).tolist()
        tuned_thr_lst = best_thr_matrix.tolist()
        k2_thr_tuned = tuned_thr_lst[0][1]
        k2_pred_ratio_tuned = float((probs_all[:, 0, 1] >= k2_thr_tuned).float().mean().item()) * 100
        k2_val_prevalence = float(tgts_all[:, 0, 1].float().mean().item()) * 100
        print(f"    -> K  | P/R/F1 @tuned  lvl1:{tuned_p_pct[0][0]:5.1f}/{tuned_r_pct[0][0]:5.1f}/{tuned_f1_pct[0][0]:5.1f}"
              f"   lvl2:{tuned_p_pct[0][1]:5.1f}/{tuned_r_pct[0][1]:5.1f}/{tuned_f1_pct[0][1]:5.1f}"
              f"   K2_pred_ratio@{k2_thr_tuned:.2f}={k2_pred_ratio_tuned:5.1f}%"
              f"   val_observed={k2_val_prevalence:5.1f}%")
        print(f"    -> M  | P/R/F1 @tuned  lvl1:{tuned_p_pct[1][0]:5.1f}/{tuned_r_pct[1][0]:5.1f}/{tuned_f1_pct[1][0]:5.1f}"
              f"   lvl2:{tuned_p_pct[1][1]:5.1f}/{tuned_r_pct[1][1]:5.1f}/{tuned_f1_pct[1][1]:5.1f}")
        # 全 8 头 tuned-threshold F1 浓缩行 — 一眼看哪个头需要校准
        print(f"    -> Tuned F1 | "
              f"K1:{tuned_f1_pct[0][0]:5.1f}@{tuned_thr_lst[0][0]:.2f} "
              f"K2:{tuned_f1_pct[0][1]:5.1f}@{tuned_thr_lst[0][1]:.2f} | "
              f"M1:{tuned_f1_pct[1][0]:5.1f}@{tuned_thr_lst[1][0]:.2f} "
              f"M2:{tuned_f1_pct[1][1]:5.1f}@{tuned_thr_lst[1][1]:.2f} | "
              f"F1:{tuned_f1_pct[2][0]:5.1f}@{tuned_thr_lst[2][0]:.2f} "
              f"F2:{tuned_f1_pct[2][1]:5.1f}@{tuned_thr_lst[2][1]:.2f} | "
              f"C1:{tuned_f1_pct[3][0]:5.1f}@{tuned_thr_lst[3][0]:.2f} "
              f"C2:{tuned_f1_pct[3][1]:5.1f}@{tuned_thr_lst[3][1]:.2f}")
        # [P2 #19] 单任务拆分 — 帮助定位哪一任务在"L2 概率翻越 L1"
        print(f"    -> Violation: {violation_rate:.2f}% "
              f"(K:{viol_K:.2f} | M:{viol_M:.2f} | F:{viol_F:.2f} | C:{viol_C:.2f})")
        # [R13] 3-class Acc（tuned 阈值口径）+ 按弹型拆解
        print(f"    -> 3-class Acc (tuned) | K:{acc_per_task[0]:5.2f}% "
              f"M:{acc_per_task[1]:5.2f}% F:{acc_per_task[2]:5.2f}% "
              f"C:{acc_per_task[3]:5.2f}%")
        # [R14 B] class-1 对角线 recall —— 直接暴露中间类塌陷，是本轮修复的核心指标
        print(f"    -> Class-1 Recall | "
              f"K:{cls1_recall_vec[0].item():5.2f}% "
              f"M:{cls1_recall_vec[1].item():5.2f}% "
              f"F:{cls1_recall_vec[2].item():5.2f}% "
              f"C:{cls1_recall_vec[3].item():5.2f}%  "
              f"(mean={cls1_recall_mean*100:.2f}%)")
        # [R14 B / R15] composite 分项 —— 让调参者直观看懂每项贡献
        #   R15 新增 min_cell（16 格最弱者）与 min_pen（最弱格子惩罚）
        print(f"    -> Composite | acc3={acc3_mean:.4f} cls1={cls1_recall_mean:.4f} "
              f"f1={f1_mean:.4f} viol={violation_rate/100:.4f} "
              f"min_cell={min_cell_acc_3class:.4f} min_pen={min_cell_penalty:.4f} → "
              f"total={composite:.4f}")
        print(f"    -> Selection | score={selection_score:.4f} "
              f"SmallM_acc={small_m_acc_score*100:5.2f}% "
              f"SmallK1_rec={small_k1_recall_score*100:5.2f}% "
              f"SmallC1_rec={small_c1_recall_score*100:5.2f}% "
              f"NonTargetAcc={non_target_cell_acc_mean*100:5.2f}% "
              f"SmallK0_FP={small_k0_fp_rate*100:4.2f}% "
              f"C0_FP={c0_fp_rate*100:4.2f}% "
              f"SmallC0_FP={small_c0_fp_rate*100:4.2f}% "
              f"cls1_floor_pen={class1_floor_penalty:.4f} "
              f"guard_pen={guardrail_penalty:.4f}")
        mun_labels = ["Small", "Med-LM", "Med-RD", "Heavy"]
        mun_acc_lines = []
        for m_id in range(4):
            n_m = int(samples_per_munition[m_id].item())
            if n_m == 0:
                mun_acc_lines.append(f"{mun_labels[m_id]}(0) -")
                continue
            avg4 = acc_per_munition[:, m_id].mean().item()
            mun_acc_lines.append(f"{mun_labels[m_id]}({n_m}):{avg4:5.2f}%")
        print(f"    -> By munition (task-avg) | " + "  ".join(mun_acc_lines))
        sm_cm_list = small_m_diag["cm"].tolist()
        sm_true_counts = small_m_diag["true_counts"].tolist()
        sm_pred_counts = small_m_diag["pred_counts"].tolist()
        print(f"    -> Small x M detail | n={small_m_diag['n']} "
              f"acc={small_m_diag['acc']:5.2f}% "
              f"true={sm_true_counts} pred={sm_pred_counts} "
              f"L0->L1={small_m_diag['l0_to_l1']:4.1f}% "
              f"L0->L2={small_m_diag['l0_to_l2']:4.1f}% "
              f"L1->L0={small_m_diag['l1_to_l0']:4.1f}% "
              f"L1->L2={small_m_diag['l1_to_l2']:4.1f}% "
              f"L2->L1={small_m_diag['l2_to_l1']:4.1f}%")
        print(f"       Small x M CM rows=true[0,1,2] cols=pred[0,1,2] | {sm_cm_list}")
        if small_m_diag.get("prob_by_true"):
            prob_lines = []
            for lv in range(3):
                stats = small_m_diag["prob_by_true"].get(lv)
                if stats is None:
                    continue
                prob_lines.append(
                    f"L{lv}:p1_mu={stats['p1_mean']:.3f}/p50={stats['p1_p50']:.3f}/p90={stats['p1_p90']:.3f} "
                    f"p2_mu={stats['p2_mean']:.3f}/p50={stats['p2_p50']:.3f}/p90={stats['p2_p90']:.3f}")
            print("       Small x M prob by true | " + " | ".join(prob_lines))
        sk_cm_list = small_k_diag["cm"].tolist()
        sk_true_counts = small_k_diag["true_counts"].tolist()
        sk_pred_counts = small_k_diag["pred_counts"].tolist()
        print(f"    -> Small x K detail | n={small_k_diag['n']} "
              f"acc={small_k_diag['acc']:5.2f}% "
              f"true={sk_true_counts} pred={sk_pred_counts} "
              f"L0->L1={small_k_diag['l0_to_l1']:4.1f}% "
              f"L1->L0={small_k_diag['l1_to_l0']:4.1f}% "
              f"SmallK1_rec={small_k1_recall_score*100:5.1f}% "
              f"SmallK0_FP={small_k0_fp_rate*100:4.2f}%")
        print(f"       Small x K CM rows=true[0,1,2] cols=pred[0,1,2] | {sk_cm_list}")
        if small_k_diag.get("prob_by_true"):
            prob_lines = []
            for lv in range(3):
                stats = small_k_diag["prob_by_true"].get(lv)
                if stats is None:
                    continue
                prob_lines.append(
                    f"L{lv}:p1_mu={stats['p1_mean']:.3f}/p50={stats['p1_p50']:.3f}/p90={stats['p1_p90']:.3f} "
                    f"p2_mu={stats['p2_mean']:.3f}/p50={stats['p2_p50']:.3f}/p90={stats['p2_p90']:.3f}")
            print("       Small x K prob by true | " + " | ".join(prob_lines))
        exposure_lines = []
        for m_id, label in enumerate(mun_labels):
            actual = int(train_mun_exposure[m_id].item())
            if expected_draws_per_mun is not None:
                exposure_lines.append(f"{label}:{actual}/{expected_draws_per_mun[m_id]}")
            else:
                exposure_lines.append(f"{label}:{actual}")
        print(f"    -> Train exposure | " + "  ".join(exposure_lines))
        cls1_mun_lines = []
        for m_id, label in enumerate(mun_labels):
            cls1_mun_lines.append(
                f"{label}:K={cls1_recall_per_mun[0, m_id].item():4.1f}% "
                f"M={cls1_recall_per_mun[1, m_id].item():4.1f}% "
                f"F={cls1_recall_per_mun[2, m_id].item():4.1f}% "
                f"C={cls1_recall_per_mun[3, m_id].item():4.1f}%")
        print(f"    -> Class-1 Recall by munition | " + "  ".join(cls1_mun_lines))
        if low_cls1_cells:
            low_str = " | ".join(
                f"{c['munition']}x{c['task']}={c['recall']:.1f}%/n{c['n_pos']}"
                for c in low_cls1_cells[:6])
            print(f"    -> Class-1 floor alerts (<{CLASS1_FLOOR_RECALL:.0f}%, "
                  f"n>={CLASS1_FLOOR_MIN_POS}) | {low_str}")
        # 同步到 TensorBoard 的 Violation 命名空间
        writer.add_scalar("Violation/Overall", violation_rate, epoch)
        for i, d_name in enumerate(["K", "M", "F", "C"]):
            writer.add_scalar(f"Violation/{d_name}",
                              violation_per_task[i].item() / max(t_samples, 1) * 100,
                              epoch)

        # [优化点] Epoch 级别的条件 TensorBoard 写入，防止影响密集步进前推性能
        writer.add_scalar("Loss/Total_Train", train_loss, epoch)
        writer.add_scalar("Loss/Total_Val", val_loss, epoch)
        writer.add_scalar("LearningRate", lr_current, epoch)
        # [R14 B] Composite 分项早停/best-epoch 驱动，单独 namespace 便于 TB 对比
        writer.add_scalar("Composite/acc3_mean", acc3_mean, epoch)
        writer.add_scalar("Composite/cls1_recall_mean", cls1_recall_mean, epoch)
        writer.add_scalar("Composite/f1_mean", f1_mean, epoch)
        writer.add_scalar("Composite/total", composite, epoch)
        writer.add_scalar("Selection/score", selection_score, epoch)
        writer.add_scalar("Selection/SmallK1_recall", small_k1_recall_score * 100.0, epoch)
        writer.add_scalar("Selection/SmallC1_recall", small_c1_recall_score * 100.0, epoch)
        writer.add_scalar("Selection/NonTargetCellAccMean", non_target_cell_acc_mean * 100.0, epoch)
        writer.add_scalar("Selection/Class1FloorPenalty", class1_floor_penalty, epoch)
        writer.add_scalar("Selection/C0_false_positive", c0_fp_rate * 100.0, epoch)
        writer.add_scalar("Selection/SmallC0_false_positive", small_c0_fp_rate * 100.0, epoch)
        writer.add_scalar("SmallK/Acc_3class", small_k_diag["acc"], epoch)
        writer.add_scalar("SmallK/Class1_recall", small_k1_recall_score * 100.0, epoch)
        writer.add_scalar("SmallK/K0_false_positive", small_k0_fp_rate * 100.0, epoch)
        writer.add_scalar("SmallK/L0_to_L1", small_k_diag["l0_to_l1"], epoch)
        writer.add_scalar("SmallK/L1_to_L0", small_k_diag["l1_to_l0"], epoch)
        writer.add_scalar("SmallM/Acc_3class", small_m_diag["acc"], epoch)
        writer.add_scalar("SmallM/L0_to_L1", small_m_diag["l0_to_l1"], epoch)
        writer.add_scalar("SmallM/L0_to_L2", small_m_diag["l0_to_l2"], epoch)
        writer.add_scalar("SmallM/L1_to_L0", small_m_diag["l1_to_l0"], epoch)
        writer.add_scalar("SmallM/L1_to_L2", small_m_diag["l1_to_l2"], epoch)
        writer.add_scalar("SmallM/L2_to_L1", small_m_diag["l2_to_l1"], epoch)
        for i, d_name in enumerate(["K", "M", "F", "C"]):
            writer.add_scalar(f"Cls1Recall/{d_name}",
                              cls1_recall_vec[i].item(), epoch)
        dim_names = ["K", "M", "F", "C"]
        for i, d_name in enumerate(dim_names):
            # train_task_losses 在循环里只累加未除，这里 / len(train_loader) 才是 per-batch 均值
            writer.add_scalar(f"Loss_Task/Train_{d_name}",
                              train_task_losses[i].item() / len(train_loader), epoch)
            # [P1 #6 修复] val_task_losses 在 L233 已 /= len(val_loader)，此处不再除
            writer.add_scalar(f"Loss_Task/Val_{d_name}",
                              val_task_losses[i].item(), epoch)
            writer.add_scalar(f"Variance_s/Dim_{d_name}",
                              criterion.log_vars[i].item(), epoch)

        # [R12] 单 epoch 耗时（含 train + val + plot 同步开销）
        epoch_dt = time.time() - epoch_t0
        history['epoch_time'].append(epoch_dt)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch + 1
            if not smoke_test:
                _safe_torch_save(model.state_dict(), "./output/models/best_val_model.pth")
                print(f"    [*] Saved best_val_model.pth @ min Val Loss: "
                      f"epoch={best_val_epoch} val_loss={best_val_loss:.4f}")

        selection_candidate_summary = {
            "selection_score": float(selection_score),
            "min_cell_acc_3class": float(min_cell_acc_3class),
            "min_supported_class1_recall": float(
                min_supported_class1_recall),
            "min_supported_class_diagonal_recall": float(
                min_supported_class_diagonal_recall),
            "small_k0_fp_rate": float(small_k0_fp_rate),
            "c0_fp_rate": float(c0_fp_rate),
        }
        selection_key = _goal_candidate_sort_key(
            selection_candidate_summary)
        improved_selection = (
            best_selection_key is None
            or selection_key > best_selection_key
        )

        # Final business artifacts follow the strict goal-aware selection key.
        # best_val_model.pth remains as a shadow diagnostic checkpoint.
        if improved_selection:
            best_selection_score = selection_score
            best_selection_key = selection_key
            best_selection_epoch = epoch + 1
            import numpy as _np
            for ti, name in enumerate(["K", "M", "F", "C"]):
                best_predictions[name] = _np.concatenate(epoch_pred_levels[ti])
                best_predictions[name + "_true"] = _np.concatenate(epoch_true_levels[ti])
            best_predictions["epoch"] = epoch + 1
            best_predictions["val_loss"] = val_loss
            best_predictions["selection_score"] = selection_score
            best_predictions["model_variant"] = "raw_best"
            best_predictions["composite"] = composite
            best_predictions["small_m_diag"] = dict(small_m_diag)
            best_predictions["small_k_diag"] = dict(small_k_diag)
            best_predictions["munition_acc_matrix"] = acc_per_munition.clone()
            best_predictions["munition_samples"]    = samples_per_munition.clone()
            best_predictions["munition_thresholds"] = best_thr_matrix.clone()
            best_predictions["per_mun_thresholds"]  = best_thr_matrix_perm.clone()
            best_predictions["cls1_recall_vec"]     = cls1_recall_vec.clone()
            best_predictions["cls1_recall_per_mun"] = cls1_recall_per_mun.clone()
            best_predictions["cls1_count_per_mun"]  = cls1_count_per_mun.clone()
            best_predictions["tuned_f1_matrix"]     = tuned_f1_matrix.clone()
            best_predictions["acc_per_task"]        = list(acc_per_task)
            best_predictions["low_cls1_cells"]      = list(low_cls1_cells)
            best_predictions["composite_breakdown"] = dict(composite_breakdown)

            if (not smoke_test) or ablation_config:
                _safe_torch_save(model.state_dict(), "./output/models/best_model.pth")
                HEAD_NAMES = ["K1", "K2", "M1", "M2", "F1", "F2", "C1", "C2"]
                per_mun_dict = {}
                for head in HEAD_NAMES:
                    i = ["K", "M", "F", "C"].index(head[0])
                    j = int(head[1]) - 1
                    per_mun_dict[head] = {
                        str(m): float(best_thr_matrix_perm[i, m, j].item())
                        for m in range(4)
                    }
                thr_dict = {
                    "K1": float(best_thr_matrix[0, 0].item()),
                    "K2": float(best_thr_matrix[0, 1].item()),
                    "M1": float(best_thr_matrix[1, 0].item()),
                    "M2": float(best_thr_matrix[1, 1].item()),
                    "F1": float(best_thr_matrix[2, 0].item()),
                    "F2": float(best_thr_matrix[2, 1].item()),
                    "C1": float(best_thr_matrix[3, 0].item()),
                    "C2": float(best_thr_matrix[3, 1].item()),
                    "per_munition": per_mun_dict,
                    "_schema": CURRENT_THRESHOLD_SCHEMA,
                    "_note": "Checkpoint-aligned thresholds saved at the best goal-aware selection epoch.",
                    "_calibration_policy": {
                        "joint_alpha": 0.80,
                        "threshold_grid_min": 0.02,
                        "threshold_grid_max": 1.0,
                        "threshold_grid_step": 0.02,
                        "c0_max_false_positive_rate": GLOBAL_C0_MAX_FP_RATE,
                        "minimum_exact_class1_recall": (
                            float(minimum_exact_l1_recall)),
                        "maximum_class1_floor_accuracy_drop": (
                            None
                            if maximum_class1_floor_accuracy_drop is None
                            else float(
                                maximum_class1_floor_accuracy_drop)),
                        "goal_aware_cell_search": bool(
                            goal_aware_cell_search),
                        "minimum_cell_accuracy": (
                            minimum_goal_cell_accuracy),
                        "minimum_class_diagonal_recall": (
                            minimum_goal_class_diagonal_recall),
                        "recall_floor_policy": (
                            "prefer recall-feasible thresholds without "
                            "violating the false-positive cap; otherwise "
                            "retain the best safety-feasible threshold"),
                    },
                    "_model_variant": "raw_best",
                    "_best_epoch": int(best_selection_epoch),
                    "_raw_best_epoch": int(best_selection_epoch),
                    "_soup_epochs": [],
                    "_selection_score": float(selection_score),
                    "_val_loss": float(val_loss),
                    "_small_m_acc": float(small_m_acc_score),
                    "_small_k1_recall": float(small_k1_recall_score),
                    "_small_c1_recall": float(small_c1_recall_score),
                    "_small_k0_false_positive": float(small_k0_fp_rate),
                    "_c0_false_positive": float(c0_fp_rate),
                    "_small_c0_false_positive": float(small_c0_fp_rate),
                    "_guardrail_penalty": float(guardrail_penalty),
                    "_non_target_cell_acc_mean": float(non_target_cell_acc_mean),
                    "_rare_cell_thresholds": rare_cell_thresholds,
                    "_low_class1_cells": low_cls1_cells,
                }
                with open("./output/models/best_thresholds.json", "w", encoding="utf-8") as _fh:
                    json.dump(thr_dict, _fh, indent=2, ensure_ascii=False)
                print(f"    [*] Saved best_model.pth @ goal-aware selection: "
                      f"epoch={best_selection_epoch} score={best_selection_score:.4f} "
                      f"SmallM={small_m_acc_score*100:.2f}% "
                      f"SmallK1={small_k1_recall_score*100:.2f}% "
                      f"SmallC1={small_c1_recall_score*100:.2f}%")

        if not smoke_test:
            candidate = {
                "epoch": epoch + 1,
                "selection_score": float(selection_score),
                "min_cell_acc_3class": float(min_cell_acc_3class),
                "min_supported_class1_recall": float(
                    min_supported_class1_recall),
                "min_supported_class_diagonal_recall": float(
                    min_supported_class_diagonal_recall),
                "small_k0_fp_rate": float(small_k0_fp_rate),
                "c0_fp_rate": float(c0_fp_rate),
                "guardrail_penalty": float(guardrail_penalty),
                "small_m_acc": float(small_m_acc_score),
                "small_k1_recall": float(small_k1_recall_score),
                "small_c1_recall": float(small_c1_recall_score),
                "non_target_cell_acc_mean": float(non_target_cell_acc_mean),
                "state_dict": _clone_state_dict_cpu(model),
            }
            topk_selection_candidates = _insert_topk_candidate(
                topk_selection_candidates, candidate, k=topk_k)

        # [60-epoch 诊断后替换] 复合评价函数：L2 F1 总和 − 5× 保序违反率(%)
        # [R7 叠加] 若 K1 校准 F1 比默认 0.5 高，加入 composite，给真实业务指标更多权重
        # [R13] composite 升级：使用全 8 头的 tuned F1 总和 − 违反率扣分
        # [R14 B] 第三次重写：composite = 0.50·acc3_mean + 0.25·cls1_recall_mean
        #   + 0.20·f1_mean − 0.05·violation_rate/100（分项已在上方 cls1_recall_vec
        #   后统一计算，此处只做 best/early-stop 决策）
        if composite > best_composite:
            best_composite = composite
            best_composite_epoch = epoch + 1

        if improved_selection:
            stale_epochs = 0
        else:
            stale_epochs += 1
            if (epoch + 1) >= min_selection_epochs and stale_epochs >= patience:
                print(
                    f"\n[EARLY STOP] 94%/90%+安全目标选择键已连续 "
                    f"{patience} 个 epoch 未刷新，终止训练。")
                print(
                    f"              best_selection_score = "
                    f"{best_selection_score:.4f} @ epoch "
                    f"{best_selection_epoch}")
                print(f"              best_val_loss = {best_val_loss:.4f} @ epoch {best_val_epoch}")
                break

    if not smoke_test and best_predictions["epoch"] is not None:
        final_candidates = []
        topk_dir = os.path.join("./output/models", f"topk_selection_seed{seed}")
        os.makedirs(topk_dir, exist_ok=True)
        topk_selection_candidates.sort(
            key=_goal_candidate_sort_key, reverse=True)
        soup_epochs = [int(c["epoch"]) for c in topk_selection_candidates[:topk_k]]

        for rank, cand in enumerate(topk_selection_candidates[:topk_k], start=1):
            _safe_torch_save({
                "epoch": int(cand["epoch"]),
                "selection_score": float(cand["selection_score"]),
                "min_cell_acc_3class": float(
                    cand["min_cell_acc_3class"]),
                "min_supported_class1_recall": float(
                    cand["min_supported_class1_recall"]),
                "min_supported_class_diagonal_recall": float(
                    cand["min_supported_class_diagonal_recall"]),
                "small_k0_fp_rate": float(cand["small_k0_fp_rate"]),
                "c0_fp_rate": float(cand["c0_fp_rate"]),
                "guardrail_penalty": float(cand["guardrail_penalty"]),
                "small_m_acc": float(cand["small_m_acc"]),
                "small_k1_recall": float(cand["small_k1_recall"]),
                "small_c1_recall": float(cand["small_c1_recall"]),
                "non_target_cell_acc_mean": float(cand["non_target_cell_acc_mean"]),
                "model_state": cand["state_dict"],
            }, os.path.join(topk_dir, f"rank{rank}_epoch{int(cand['epoch']):03d}.pth"))

        def _evaluate_final_candidate(name: str, state_dict: dict, epoch_label: int,
                                      soup_epoch_list: list[int] = None):
            model.load_state_dict(state_dict)
            metrics = _evaluate_selection_snapshot(
                model, criterion, val_loader, device, scaler_amp is not None,
                epoch_label=epoch_label,
                mechanism_auxiliary_weight=(
                    mechanism_auxiliary_weight
                    if mechanism_outputs_enabled else 0.0),
                mechanism_class_distribution_weight=(
                    mechanism_class_distribution_weight),
                mechanism_loss_options=mechanism_loss_options,
                component_auxiliary_weight=(
                    component_auxiliary_weight
                    if component_outputs_enabled else 0.0),
                component_target_tree_teacher_weight=(
                    component_target_tree_teacher_weight),
                component_rule_consistency_weight=(
                    component_rule_consistency_weight),
                component_distribution_weight=(
                    component_distribution_weight),
                component_positive_weight=(
                    component_positive_weight),
                component_rule_entry_ranking_weight=(
                    component_rule_entry_ranking_weight),
                component_rule_conditional_ranking_weight=(
                    component_rule_conditional_ranking_weight),
                component_rule_ranking_margin=(
                    component_rule_ranking_margin),
                component_rule_hard_negative_fraction=(
                    component_rule_hard_negative_fraction),
                minimum_exact_l1_recall=minimum_exact_l1_recall,
                maximum_class1_floor_accuracy_drop=(
                    maximum_class1_floor_accuracy_drop),
                minimum_goal_cell_accuracy=(
                    minimum_goal_cell_accuracy),
                minimum_goal_class_diagonal_recall=(
                    minimum_goal_class_diagonal_recall),
            )
            final_candidates.append({
                "name": name,
                "state_dict": _clone_state_dict_cpu(model),
                "metrics": metrics,
                "soup_epochs": [int(e) for e in (soup_epoch_list or [])],
            })

        raw_state = torch.load("./output/models/best_model.pth", map_location=device, weights_only=True)
        _evaluate_final_candidate("raw_best", raw_state, int(best_selection_epoch))

        if (
            initial_model_state is not None
            and bool(training_cfg.get(
                "include_initial_candidate", True))
        ):
            _evaluate_final_candidate(
                "verified_warm_start",
                initial_model_state,
                0,
            )

        if best_val_epoch is not None and os.path.exists("./output/models/best_val_model.pth"):
            best_val_state = torch.load("./output/models/best_val_model.pth", map_location=device, weights_only=True)
            _evaluate_final_candidate("best_val", best_val_state, int(best_val_epoch))

        for rank, cand in enumerate(topk_selection_candidates[:topk_k], start=1):
            _evaluate_final_candidate(
                f"rank{rank}_epoch{int(cand['epoch']):03d}",
                cand["state_dict"],
                int(cand["epoch"]),
            )

        if len(topk_selection_candidates) >= 2:
            soup_state = _average_state_dicts([c["state_dict"] for c in topk_selection_candidates[:topk_k]])
            _safe_torch_save({
                "soup_epochs": soup_epochs,
                "model_state": soup_state,
            }, os.path.join(topk_dir, "top5_soup.pth"))
            _evaluate_final_candidate(
                f"top{len(soup_epochs)}_soup",
                soup_state,
                int(best_selection_epoch),
                soup_epoch_list=soup_epochs,
            )

        # 释放对 best_model.pth / best_val_model.pth 的句柄，
        # 否则 Windows 下后续 torch.save 会因独占写锁失败。
        # final_candidates 内部已是 _clone_state_dict_cpu 深拷贝，原始 state_dict 不再需要。
        raw_state = None
        if 'best_val_state' in locals():
            best_val_state = None
        import gc
        gc.collect()

        final_candidates.sort(
            key=lambda c: _goal_candidate_sort_key({
                "selection_score": float(
                    c["metrics"]["selection_score"]),
                "min_cell_acc_3class": float(
                    c["metrics"]["min_cell_acc_3class"]),
                "min_supported_class1_recall": float(
                    c["metrics"]["min_supported_class1_recall"]),
                "min_supported_class_diagonal_recall": float(
                    c["metrics"][
                        "min_supported_class_diagonal_recall"]),
                "small_k0_fp_rate": float(
                    c["metrics"]["small_k0_fp_rate"]),
                "c0_fp_rate": float(
                    c["metrics"]["c0_fp_rate"]),
            }),
            reverse=True,
        )
        print("\n[FINAL-CANDIDATES] Re-evaluated with final selection policy:")
        for cand in final_candidates:
            m = cand["metrics"]
            print(f"  {cand['name']:<18s} score={float(m['selection_score']):.4f} "
                  f"guard={float(m['guardrail_penalty']):.4f} "
                  f"MinAcc={100.0*float(m['min_cell_acc_3class']):5.2f}% "
                  f"MinL1={100.0*float(m['min_supported_class1_recall']):5.2f}% "
                  f"MinDiag={100.0*float(m['min_supported_class_diagonal_recall']):5.2f}% "
                  f"SmallK1={float(m['cls1_recall_per_mun'][0, 0].item()):5.1f}% "
                  f"SmallK0_FP={float(m['small_k0_fp_rate']*100):4.2f}% "
                  f"SmallM={float(m['small_m_diag']['acc']):5.2f}% "
                  f"SmallC1={float(m['cls1_recall_per_mun'][3, 0].item()):5.1f}% "
                  f"LowCls1={len(m.get('low_cls1_cells', []))}")

        selected = final_candidates[0]
        model.load_state_dict(selected["state_dict"])
        _write_model_artifacts(
            model,
            selected["metrics"],
            model_variant=selected["name"],
            raw_best_epoch=best_selection_epoch,
            soup_epochs=selected["soup_epochs"],
        )
        best_predictions = _metrics_to_best_predictions(selected["metrics"], selected["name"])
        best_selection_score = float(selected["metrics"]["selection_score"])
        print(f"[FINAL-CANDIDATES] Selected {selected['name']} as final best_model.pth")

    manifest_path = write_model_manifest(
        "./output/models",
        data_contract=data_contract,
        model_config=resolved_model_config,
        training_config={
            "loss": {
                "focal_gamma": float(loss_gamma),
                "ordinal_penalty_weight": float(penalty_weight),
                "class1_margin_weight": float(class1_margin_weight),
                "cell_class1_alpha_enabled": bool(
                    loss_cfg.get("use_cell_class1_alpha", False)),
                "class_distribution_weight": float(
                    class_distribution_weight),
                "class_distribution_loss_enabled": bool(
                    class_distribution_weight > 0.0),
                "hard_level_classification_weight": float(
                    hard_level_classification_weight),
                "middle_class_distribution_multiplier": (
                    middle_class_distribution_multiplier.detach()
                    .cpu().tolist()),
                "entry_ranking_weight": (
                    entry_ranking_weight.detach().cpu().tolist()),
                "conditional_l1_l2_ranking_weight": (
                    conditional_l1_l2_ranking_weight.detach()
                    .cpu().tolist()),
                "ranking_margin": float(ranking_margin),
                "hard_negative_fraction": float(
                    hard_negative_fraction),
                "mechanism_auxiliary_weight": float(
                    mechanism_auxiliary_weight
                    if mechanism_outputs_enabled else 0.0),
                "mechanism_class_distribution_weight": float(
                    mechanism_class_distribution_weight),
                "mechanism_branch_weights": (
                    mechanism_branch_weights.detach().cpu().tolist()),
                "mechanism_boundary_focus_weight": float(
                    mechanism_boundary_focus_weight),
                "mechanism_boundary_focus_bandwidth": float(
                    mechanism_boundary_focus_bandwidth),
                "mechanism_hard_classification_weight": float(
                    mechanism_hard_classification_weight),
                "mechanism_use_dataset_row_weights": bool(
                    mechanism_use_dataset_row_weights),
                "component_auxiliary_weight": float(
                    component_auxiliary_weight
                    if component_outputs_enabled else 0.0),
                "component_target_tree_teacher_weight": float(
                    component_target_tree_teacher_weight
                    if component_outputs_enabled else 0.0),
                "component_rule_consistency_weight": float(
                    component_rule_consistency_weight),
                "component_distribution_weight": float(
                    component_distribution_weight),
                "component_rule_entry_ranking_weight": (
                    component_rule_entry_ranking_weight.detach()
                    .cpu().tolist()),
                "component_rule_conditional_l1_l2_ranking_weight": (
                    component_rule_conditional_ranking_weight.detach()
                    .cpu().tolist()),
                "component_rule_ranking_margin": float(
                    component_rule_ranking_margin),
                "component_rule_hard_negative_fraction": float(
                    component_rule_hard_negative_fraction),
                "component_positive_weight_enabled": bool(
                    component_positive_weight_enabled),
            },
            "epochs_requested": int(epochs),
            "smoke_test": bool(smoke_test),
            "optimization": {
                "learning_rate": float(lr_base),
                "weight_decay": float(
                    training_cfg.get("weight_decay", 5e-4)),
                "minimum_selection_epochs": int(
                    min_selection_epochs),
                "selection_patience": int(patience),
                "freeze_base_model": bool(freeze_base_model),
                "freeze_criterion_uncertainty": bool(
                    freeze_criterion_uncertainty),
                "trainable_model_parameters": int(sum(
                    parameter.numel()
                    for parameter in trainable_model_parameters)),
            },
            "warm_start": warm_start_provenance,
        },
        seed=seed,
    )
    print(f"[Training] Sealed model/data artifact contract: {manifest_path}")
    if not smoke_test and "selected" in locals():
        validation_report = _validation_report_from_metrics(
            selected["metrics"],
            model_variant=selected["name"],
            dataset_sha256=str(data_contract["dataset_sha256"]),
            model_sha256=sha256_file(
                "./output/models/best_model.pth"),
            threshold_sha256=sha256_file(
                "./output/models/best_thresholds.json"),
        )
        validation_report_path = (
            "./output/validation/selection_metrics.json")
        _write_json_atomic(
            validation_report_path, validation_report)
        print(
            "[VALIDATION] "
            f"acc={validation_report['average_3class_accuracy_percent']:.2f}% "
            f"gate={'PASS' if validation_report['performance_gate']['passed'] else 'FAIL'} "
            f"failures={validation_report['performance_gate']['failure_count']} "
            f"report={os.path.abspath(validation_report_path)}"
        )

    elapsed = time.time() - t0
    writer.close()

    # =========================================================
    # [R12] 训练可视化重写：原 2×2 → 3×3 主面板 + 独立混淆矩阵图 + 最终摘要卡
    #       目标：业务侧打开图就能看懂"模型训成什么样了"，不必再翻 train.txt。
    # =========================================================
    import matplotlib.pyplot as plt
    import numpy as np
    # [R13] 字体回退链：DejaVu Sans 在 Windows 下不带 CJK 字形，中文会跳出大量 Glyph missing
    # 警告并显示成豆腐方块。把 Microsoft YaHei / SimHei 放在首位即可本机复用系统字体。
    # 同时对 matplotlib 的默认后端做一次静音：即便图中残留 CJK 也不再 print warning。
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = [
        'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    import warnings as _warnings
    _warnings.filterwarnings("ignore",
                             category=UserWarning,
                             message=".*Glyph.*missing from font.*")

    actual_epochs = len(history['train_loss'])
    ep = list(range(1, actual_epochs + 1))

    # 调色板：K=红 M=蓝 F=绿 C=橙（贯穿全图保持一致）
    TASK_COLORS = {"K": "#e74c3c", "M": "#3498db", "F": "#2ecc71", "C": "#f39c12"}
    final_variant = best_predictions.get("model_variant", "raw_best")
    final_epoch = best_predictions.get("epoch")
    final_label = f"{final_variant}@epoch {final_epoch}" if final_epoch is not None else "none"

    def _as_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    fig, axes = plt.subplots(3, 3, figsize=(20, 14))
    fig.suptitle(f'Damage Assessment MTL — Training Diagnostics '
                 f'(epochs={actual_epochs}, final={final_label})',
                 fontsize=15, fontweight='bold', y=0.995)

    # ============== [Row 0, Col 0] Loss curves ==============
    ax = axes[0, 0]
    ax.plot(ep, history['train_loss'], label='Train Loss', color='#2c7bb6', marker='o', ms=3)
    ax.plot(ep, history['val_loss'],   label='Val Loss',   color='#d7191c', marker='s', ms=3)
    if best_predictions["epoch"] is not None:
        ax.axvline(best_predictions["epoch"], color='gray', linestyle='--', alpha=0.6,
                   label=f'Selection best @ ep {best_predictions["epoch"]}')
    ax.set_title('Loss Curves (Kendall-regularized, reference)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    # ============== [Row 0, Col 1] 3-class overall accuracy (KEY metric) ==============
    ax = axes[0, 1]
    for name in ["K", "M", "F", "C"]:
        ax.plot(ep, history[f'acc_{name}'], label=f'{name}-task Acc',
                color=TASK_COLORS[name], marker='o', ms=3)
    ax.set_title('3-Class Overall Accuracy (0 / 1 / 2) — primary business metric')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(top=100.5)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9, loc='lower right')

    # ============== [Row 0, Col 2] All-task tuned F1 evolution ==============
    # [P1-A+] 取消 thr=0.5 曲线（pos_weight 下本就不公平），统一画 tuned threshold F1
    ax = axes[0, 2]
    tuned_keys_row = {
        "K": ("f1_K1_tuned_all", "f1_K2_tuned"),
        "M": ("f1_M1_tuned",     "f1_M2_tuned"),
        "F": ("f1_F1_tuned",     "f1_F2_tuned"),
        "C": ("f1_C1_tuned",     "f1_C2_tuned"),
    }
    for name, (k1_key, k2_key) in tuned_keys_row.items():
        ax.plot(ep, history[k1_key],
                color=TASK_COLORS[name], linestyle='-',  marker='.', ms=3,
                label=f'F1 {name}1 (≥1)')
        ax.plot(ep, history[k2_key],
                color=TASK_COLORS[name], linestyle='--', marker='x', ms=3,
                label=f'F1 {name}2 (≥2)')
    ax.set_title('F1 — All 8 Heads @ tuned threshold (solid=L1, dashed=L2)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('F1 (%)')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7, ncol=2, loc='lower right')

    # ============== [Row 1, Col 0] [P1-A+] Composite breakdown trend ==============
    # 取消 thr=0.5 Recall 曲线（pos_weight 使之不公平）；改画 composite 三分量随 epoch 的轨迹
    # 这样可以直接看出 best-epoch 是被 acc3 / cls1 / f1 中的哪个主导
    ax = axes[1, 0]
    ax.plot(ep, history['composite_acc3'], label='acc3 mean (%)',  color='#2c7bb6', marker='.', ms=3)
    ax.plot(ep, history['composite_cls1'], label='cls1 recall (%)', color='#d7191c', marker='x', ms=3)
    ax.plot(ep, history['composite_f1'],   label='tuned F1 (%)',   color='#2ecc71', marker='s', ms=3)
    ax2 = ax.twinx()
    ax2.plot(ep, history['composite_total'], label='composite total', color='#9b59b6',
             linestyle=':', linewidth=2.0, marker='^', ms=3)
    ax2.set_ylabel('Composite total', color='#9b59b6')
    ax2.tick_params(axis='y', labelcolor='#9b59b6')
    ax.set_title('Composite Breakdown (reference trend)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Component (%)')
    ax.grid(True, alpha=0.3)
    lns1 = ax.get_lines(); lns2 = ax2.get_lines()
    ax.legend(lns1 + lns2, [l.get_label() for l in lns1 + lns2], fontsize=8, loc='lower right')

    # ============== [Row 1, Col 1] [R14 E-2] Class-1 recall curves per task ==============
    # 原 K2 Diagnostic 让位给 R14 首要目标：中间类对角线召回。
    # （K2 相关曲线仍可在 [Row 0, Col 2] 的 8-head F1 面板中观察。）
    ax = axes[1, 1]
    for name in ["K", "M", "F", "C"]:
        key = f'cls1_rec_{name}'
        if key in history and len(history[key]) == len(ep):
            ax.plot(ep, history[key], label=f'{name} class-1 recall',
                    color=TASK_COLORS[name], marker='o', ms=3)
    ax.axhline(y=85.0, color='gray', linestyle='--', alpha=0.5, label='R14 target 85%')
    ax.set_title('Class-1 Diagonal Recall per Task (R14 primary target)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Recall (%)')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc='lower right')

    # ============== [Row 1, Col 2] [P1-A+] All 8-head tuned threshold trajectories ==============
    # 原显示 K1 @thr=0.5 对比已取消（thr=0.5 在 pos_weight 下本就不公平）
    # 改为绘制全 8 头 tuned threshold 的 epoch-by-epoch 轨迹，直接暴露哪些头阈值抖动严重
    ax = axes[1, 2]
    thr_lines = [
        ('best_thr_K1', 'K1', '-'),  ('thr_K2', 'K2', '--'),
        ('thr_M1',     'M1', '-'),  ('thr_M2', 'M2', '--'),
        ('thr_F1',     'F1', '-'),  ('thr_F2', 'F2', '--'),
        ('thr_C1',     'C1', '-'),  ('thr_C2', 'C2', '--'),
    ]
    for key, name, ls in thr_lines:
        if key in history and len(history[key]) == len(ep):
            ax.plot(ep, history[key], label=name,
                    color=TASK_COLORS[name[0]], linestyle=ls, marker='.', ms=3)
    ax.set_title('Tuned Threshold Evolution per Head (stability check)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Threshold')
    ax.set_ylim(0.05, 0.95)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7, ncol=4, loc='lower center')

    # ============== [Row 2, Col 0] Per-task violation rate (P2 #19) ==============
    ax = axes[2, 0]
    for name in ["K", "M", "F", "C"]:
        ax.plot(ep, history[f'viol_{name}'], label=f'Viol {name} (%)',
                color=TASK_COLORS[name], marker='o', ms=3)
    ax.plot(ep, history['violation_rate'], label='Aggregate', color='black', linestyle=':', marker='.', ms=3)
    ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5, label='Target ≤ 0.3%')
    ax.set_title('Physics Violation Rate per Task (P(L2)>P(L1) before clamp)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Violation Rate (%)')
    ax.set_yscale('symlog', linthresh=0.01)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # ============== [Row 2, Col 1] log_var evolution (Kendall) ==============
    ax = axes[2, 1]
    for name in ["K", "M", "F", "C"]:
        ax.plot(ep, history[f'logvar_{name}'], label=f's_{name}',
                color=TASK_COLORS[name], marker='o', ms=3)
    # [R14 D] 下限从 -1.0 放宽到 -3.0（precision 上限 e^3 ≈ 20），解冻 Kendall 机制
    ax.axhline(y=-1.0, color='gray', linestyle='--', alpha=0.5, label='M/F/C lower clamp -1.0')
    ax.axhline(y=-2.0, color='black', linestyle=':', alpha=0.5, label='K lower clamp -2.0')
    ax.axhline(y= 2.5, color='gray', linestyle='--', alpha=0.5, label='Upper clamp +2.5')
    ax.set_title('Kendall log_var per Task  (precision = e^{-s})')
    ax.set_xlabel('Epoch'); ax.set_ylabel('log_var (s)')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc='best')

    # ============== [Row 2, Col 2] LR + epoch time (dual axis) ==============
    ax = axes[2, 2]
    ax.plot(ep, history['lr'], color='#9b59b6', marker='o', ms=3, label='Learning Rate')
    ax.set_yscale('log')
    ax.set_xlabel('Epoch'); ax.set_ylabel('LR (log)')
    ax2 = ax.twinx()
    ax2.bar(ep, history['epoch_time'], alpha=0.25, color='#34495e', label='Epoch Time (s)')
    ax2.set_ylabel('Sec / Epoch')
    ax.set_title(f'Cosine LR Schedule + Wall-Clock (total {elapsed:.0f}s)')
    ax.grid(True, alpha=0.3)
    lns = ax.get_lines() + [ax2.containers[0]]
    ax.legend(lns, [l.get_label() for l in lns], fontsize=8, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    history_png = "./output/runs/damage_model/training_history.png"
    plt.savefig(history_png, dpi=110)
    plt.close(fig)
    print(f"\n[+] Diagnostic dashboard (3×3) → {history_png}")

    # =========================================================
    # [R12] 第二张图：4 张混淆矩阵（K/M/F/C），来自 best epoch 的 val 集预测
    # =========================================================
    if best_predictions["K"] is not None:
        fig_cm, axes_cm = plt.subplots(2, 2, figsize=(12, 10))
        fig_cm.suptitle(f'Confusion Matrices @ Final Selection '
                        f'({final_label}) — Validation Set',
                        fontsize=14, fontweight='bold')
        for idx, name in enumerate(["K", "M", "F", "C"]):
            ax_cm = axes_cm[idx // 2, idx % 2]
            y_pred = best_predictions[name]
            y_true = best_predictions[name + "_true"]
            cm = np.zeros((3, 3), dtype=np.int64)
            for t, p in zip(y_true, y_pred):
                cm[int(t), int(p)] += 1
            # 按行归一化（recall 视角：真值=t，模型把它分到哪些类）
            row_sum = cm.sum(axis=1, keepdims=True).clip(min=1)
            cm_norm = cm / row_sum * 100.0
            im = ax_cm.imshow(cm_norm, cmap='Blues', vmin=0, vmax=100)
            ax_cm.set_title(f'{name}-Task  (acc = '
                            f'{(np.trace(cm) / cm.sum() * 100):.2f}%)',
                            color=TASK_COLORS[name], fontweight='bold')
            ax_cm.set_xlabel('Predicted Level')
            ax_cm.set_ylabel('True Level')
            ax_cm.set_xticks([0, 1, 2]); ax_cm.set_yticks([0, 1, 2])
            ax_cm.set_xticklabels(['0 (none)', '1 (partial)', '2 (catastrophic)'])
            ax_cm.set_yticklabels(['0 (none)', '1 (partial)', '2 (catastrophic)'])
            # 在每个格子里写"绝对计数 / 行百分比"
            for i in range(3):
                for j in range(3):
                    txt_color = "white" if cm_norm[i, j] > 50 else "black"
                    ax_cm.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.1f}%)',
                               ha='center', va='center', color=txt_color, fontsize=10)
            plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04, label='Row-norm %')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        cm_png = "./output/runs/damage_model/confusion_matrices.png"
        plt.savefig(cm_png, dpi=110)
        plt.close(fig_cm)
        print(f"[+] Confusion matrices (best epoch) → {cm_png}")

    # =========================================================
    # [R12 + R13] 第三张图：最终指标摘要 (final_metrics_summary.png)
    #   R13 升级：
    #     (1) F1 条形图默认改用 tuned threshold（而非 thr=0.5），K1 从 83% → 92%
    #     (2) 每根柱子上方标注校准阈值（e.g. "92.7\n@0.76"）
    #     (3) 新增第 3 个面板：per-munition × per-task 3-class accuracy 热力图
    # =========================================================
    if best_predictions["epoch"] is not None:
        be = max(0, min(best_predictions["epoch"] - 1, actual_epochs - 1))   # history index
        # [R14 E-3] 1×3 → 1×4：新增 class-1 recall by munition × task 热力图
        fig_sum, ax_sum = plt.subplots(1, 4, figsize=(28, 6))

        # --- 左图：8 路 F1 @ tuned threshold 条形 ---
        # [P1-A+] 取消 default (thr=0.5) 叠层对比柱（pos_weight 下 thr=0.5 本就不公平）
        names   = ['K1', 'K2', 'M1', 'M2', 'F1', 'F2', 'C1', 'C2']
        f1_tuned_keys = ['f1_K1_tuned_all', 'f1_K2_tuned',
                         'f1_M1_tuned',     'f1_M2_tuned',
                         'f1_F1_tuned',     'f1_F2_tuned',
                         'f1_C1_tuned',     'f1_C2_tuned']
        thr_keys = ['best_thr_K1', 'thr_K2', 'thr_M1', 'thr_M2',
                    'thr_F1', 'thr_F2', 'thr_C1', 'thr_C2']
        f1_matrix = best_predictions.get("tuned_f1_matrix")
        if f1_matrix is not None:
            f1_np = _as_numpy(f1_matrix) * 100.0
            f1_vals = [float(f1_np[0, 0]), float(f1_np[0, 1]),
                       float(f1_np[1, 0]), float(f1_np[1, 1]),
                       float(f1_np[2, 0]), float(f1_np[2, 1]),
                       float(f1_np[3, 0]), float(f1_np[3, 1])]
        else:
            f1_vals = [history[k][be] for k in f1_tuned_keys]
        thr_matrix = best_predictions.get("munition_thresholds")
        if thr_matrix is not None:
            thr_np = _as_numpy(thr_matrix)
            thr_vals = [float(thr_np[0, 0]), float(thr_np[0, 1]),
                        float(thr_np[1, 0]), float(thr_np[1, 1]),
                        float(thr_np[2, 0]), float(thr_np[2, 1]),
                        float(thr_np[3, 0]), float(thr_np[3, 1])]
        else:
            thr_vals = [history[k][be] for k in thr_keys]
        bar_colors = [TASK_COLORS[n[0]] for n in names]
        bar_alpha  = [0.7 if n[1] == '1' else 1.0 for n in names]
        bars_tun = ax_sum[0].bar(names, f1_vals, color=bar_colors, alpha=None,
                                 label='F1 @ tuned threshold')
        for b, a in zip(bars_tun, bar_alpha):
            b.set_alpha(a)
        # 数值标注：tuned F1 + 使用的阈值（双行）
        for b, v, t in zip(bars_tun, f1_vals, thr_vals):
            ax_sum[0].text(b.get_x() + b.get_width() / 2, v + 0.8,
                           f'{v:.1f}\n@{t:.2f}',
                           ha='center', fontsize=8, fontweight='bold')
        ax_sum[0].axhline(y=95.0, color='gray', linestyle='--', alpha=0.5,
                          label='95% baseline')
        ax_sum[0].set_title(f'F1 per Head @ Final Selection ({final_label}) '
                            f'— tuned threshold (L1 浅色 / L2 深色)')
        ax_sum[0].set_ylabel('F1 (%)'); ax_sum[0].set_ylim(0, 108)
        ax_sum[0].grid(True, alpha=0.3, axis='y')
        ax_sum[0].legend(fontsize=8, loc='lower right')

        # --- 中图：3 类整体准确率（tuned 口径） ---
        acc_names  = ['K', 'M', 'F', 'C']
        acc_from_best = best_predictions.get("acc_per_task")
        if acc_from_best is not None:
            acc_vals = [float(v) for v in acc_from_best]
        else:
            acc_vals = [history[f'acc_{n}'][be] for n in acc_names]
        acc_colors = [TASK_COLORS[n] for n in acc_names]
        bars = ax_sum[1].bar(acc_names, acc_vals, color=acc_colors, alpha=0.85)
        for b, v in zip(bars, acc_vals):
            ax_sum[1].text(b.get_x() + b.get_width() / 2, v + 0.4,
                           f'{v:.2f}%', ha='center', fontsize=11, fontweight='bold')
        ax_sum[1].axhline(y=95.0, color='gray', linestyle='--', alpha=0.5,
                          label='95% baseline')
        ax_sum[1].set_title(f'3-Class Overall Accuracy @ Final Selection ({final_label})')
        ax_sum[1].set_ylabel('Accuracy (%)'); ax_sum[1].set_ylim(0, 105)
        ax_sum[1].grid(True, alpha=0.3, axis='y'); ax_sum[1].legend()

        # --- 右图：Per-Munition × Per-Task 3-class accuracy 热力图 ---
        mat = best_predictions.get("munition_acc_matrix")   # (4 tasks, 4 munitions)
        samples = best_predictions.get("munition_samples")   # (4,)
        if mat is not None:
            mat_np = _as_numpy(mat)
            im = ax_sum[2].imshow(mat_np, cmap='RdYlGn', vmin=80.0, vmax=100.0,
                                  aspect='auto')
            MUN_LABELS = [f"Small\n(n={int(samples[0].item())})",
                          f"Med-LM\n(n={int(samples[1].item())})",
                          f"Med-RD\n(n={int(samples[2].item())})",
                          f"Heavy\n(n={int(samples[3].item())})"]
            TASK_LABELS = ["K  (Catastrophic)", "M  (Mobility)",
                           "F  (Firepower)",    "C  (Command)"]
            ax_sum[2].set_xticks(range(4)); ax_sum[2].set_xticklabels(MUN_LABELS, fontsize=9)
            ax_sum[2].set_yticks(range(4)); ax_sum[2].set_yticklabels(TASK_LABELS, fontsize=9)
            for i in range(4):
                for j in range(4):
                    val = mat_np[i, j]
                    txt_color = 'white' if val < 90 else 'black'
                    ax_sum[2].text(j, i, f'{val:.2f}%', ha='center', va='center',
                                   color=txt_color, fontsize=10, fontweight='bold')
            ax_sum[2].set_title('3-Class Accuracy by Munition × Task (tuned thr)')
            cbar = plt.colorbar(im, ax=ax_sum[2], fraction=0.046, pad=0.04)
            cbar.set_label('Accuracy (%)')
        else:
            ax_sum[2].text(0.5, 0.5, '(munition matrix unavailable)',
                           ha='center', va='center', transform=ax_sum[2].transAxes)
            ax_sum[2].set_axis_off()

        # --- [R14 E-3] 第 4 面板：Class-1 Recall by Munition × Task 热力图 ---
        # 直接暴露中间类（class-1）在各弹型上的召回表现 — 这是 R14 的首要修复目标。
        cls1_mat = best_predictions.get("cls1_recall_per_mun")  # (4 tasks, 4 munitions)，百分比
        if cls1_mat is not None:
            cls1_np = _as_numpy(cls1_mat)
            # vmin 设为 50（class-1 容忍度更低：50% 召回是"勉强能看"的起点）
            im4 = ax_sum[3].imshow(cls1_np, cmap='RdYlGn', vmin=50.0, vmax=100.0,
                                   aspect='auto')
            # 若前面 munition_acc 分支已绑定 MUN_LABELS / TASK_LABELS 则复用，否则重建
            _samples = best_predictions.get("munition_samples")
            if _samples is not None:
                MUN_LABELS_4 = [f"Small\n(n={int(_samples[0].item())})",
                                f"Med-LM\n(n={int(_samples[1].item())})",
                                f"Med-RD\n(n={int(_samples[2].item())})",
                                f"Heavy\n(n={int(_samples[3].item())})"]
            else:
                MUN_LABELS_4 = ['Small', 'Med-LM', 'Med-RD', 'Heavy']
            TASK_LABELS_4 = ["K  (Catastrophic)", "M  (Mobility)",
                             "F  (Firepower)",    "C  (Command)"]
            ax_sum[3].set_xticks(range(4)); ax_sum[3].set_xticklabels(MUN_LABELS_4, fontsize=9)
            ax_sum[3].set_yticks(range(4)); ax_sum[3].set_yticklabels(TASK_LABELS_4, fontsize=9)
            for i in range(4):
                for j in range(4):
                    val = cls1_np[i, j]
                    # <75 用白字（红色底），>=75 用黑字（黄绿底）
                    txt_color = 'white' if val < 75 else 'black'
                    # 若某任务×弹型的 class-1 样本数为 0，val 将是 0 — 明确标出
                    display = f'{val:.1f}%' if val > 0.01 else 'n/a'
                    ax_sum[3].text(j, i, display, ha='center', va='center',
                                   color=txt_color, fontsize=10, fontweight='bold')
            ax_sum[3].set_title('Class-1 Recall by Munition × Task (tuned thr) — R14')
            cbar4 = plt.colorbar(im4, ax=ax_sum[3], fraction=0.046, pad=0.04)
            cbar4.set_label('Class-1 Recall (%)')
        else:
            ax_sum[3].text(0.5, 0.5, '(class-1 recall matrix unavailable)',
                           ha='center', va='center', transform=ax_sum[3].transAxes)
            ax_sum[3].set_axis_off()

        plt.tight_layout()
        sum_png = "./output/runs/damage_model/final_metrics_summary.png"
        plt.savefig(sum_png, dpi=110)
        plt.close(fig_sum)
        print(f"[+] Final metrics summary → {sum_png}")

    # =========================================================
    # [R13] 第四张图：per-munition × per-task 3-class accuracy 的**时间演化**
    #   对 heatmap 的补充：若某个弹型在某任务上长期偏低，需要数据增强 or 权重调整
    # =========================================================
    if best_predictions.get("munition_acc_matrix") is not None:
        fig_mun, ax_mun = plt.subplots(1, 1, figsize=(10, 6))
        mat_np = _as_numpy(best_predictions["munition_acc_matrix"])
        samples = _as_numpy(best_predictions["munition_samples"])
        MUN_COLORS = ['#3498db', '#f39c12', '#9b59b6', '#e74c3c']
        MUN_LABELS_SHORT = ['Small', 'Med-LM', 'Med-RD', 'Heavy']
        task_names = ["K", "M", "F", "C"]
        # 每个弹型一组 4 条（对应 4 个任务），共 4×4 = 16 个柱
        bar_w = 0.2
        xs = np.arange(4)  # 4 个任务
        for m_id in range(4):
            vals = mat_np[:, m_id]
            offset = (m_id - 1.5) * bar_w
            bars = ax_mun.bar(xs + offset, vals, width=bar_w,
                              color=MUN_COLORS[m_id], alpha=0.85,
                              label=f'{MUN_LABELS_SHORT[m_id]} (n={int(samples[m_id])})')
            for b, v in zip(bars, vals):
                ax_mun.text(b.get_x() + b.get_width() / 2, v + 0.3,
                            f'{v:.1f}', ha='center', fontsize=7)
        ax_mun.set_xticks(xs)
        ax_mun.set_xticklabels([f'{n}-task' for n in task_names], fontsize=11)
        ax_mun.axhline(y=95.0, color='gray', linestyle='--', alpha=0.5, label='95% target')
        ax_mun.set_ylim(80, 102)
        ax_mun.set_ylabel('3-Class Accuracy (%)')
        ax_mun.set_title(f'Per-Munition × Per-Task Accuracy Breakdown '
                         f'@ Final Selection ({final_label})')
        ax_mun.grid(True, alpha=0.3, axis='y')
        ax_mun.legend(loc='lower right', fontsize=9, ncol=2)
        plt.tight_layout()
        mun_png = "./output/runs/damage_model/munition_breakdown.png"
        plt.savefig(mun_png, dpi=110)
        plt.close(fig_mun)
        print(f"[+] Munition breakdown → {mun_png}")

    print(f"\n{'='*60}")
    print(f"  Training Finished! Elapsed: {elapsed:.1f}s")
    if best_predictions["epoch"] is not None:
        be = max(0, min(best_predictions["epoch"] - 1, actual_epochs - 1))
        acc_from_best = best_predictions.get("acc_per_task")
        if acc_from_best is not None:
            acc_console = [float(v) for v in acc_from_best]
        else:
            acc_console = [history['acc_K'][be], history['acc_M'][be],
                           history['acc_F'][be], history['acc_C'][be]]
        print(f"  Final model variant: {best_predictions.get('model_variant', 'raw_best')}")
        print(f"  Best epoch (selection_score): {best_predictions['epoch']}")
        print(f"  Best selection score: {best_predictions['selection_score']:.4f}")
        print(f"  Val Loss at selection best: {best_predictions['val_loss']:.4f}")
        print(f"  Best Val Loss shadow: {best_val_loss:.4f} @ epoch {best_val_epoch}")
        print(f"  3-class Acc: K={acc_console[0]:.2f}%  "
              f"M={acc_console[1]:.2f}%  "
              f"F={acc_console[2]:.2f}%  "
              f"C={acc_console[3]:.2f}%")
        print(f"  Best composite metric: {best_composite:.4f} @ epoch {best_composite_epoch}")
        # [R14 E-5] class-1 对角线召回块 — R14 的首要目标
        if best_predictions.get("cls1_recall_vec") is not None:
            cvec = best_predictions["cls1_recall_vec"]
            print(f"  Class-1 diagonal recall (tuned thr):")
            print(f"    K={float(cvec[0].item()):.2f}%  "
                  f"M={float(cvec[1].item()):.2f}%  "
                  f"F={float(cvec[2].item()):.2f}%  "
                  f"C={float(cvec[3].item()):.2f}%  "
                  f"(mean={float(cvec.mean().item()):.2f}%)")
        if best_predictions.get("composite_breakdown"):
            cb = best_predictions["composite_breakdown"]
            print(f"  Composite breakdown: "
                  f"acc3={cb.get('acc3', 0.0):.4f}  "
                  f"cls1={cb.get('cls1', 0.0):.4f}  "
                  f"f1={cb.get('f1', 0.0):.4f}  "
                  f"viol={cb.get('viol', 0.0):.4f}  "
                  f"min_cell={cb.get('min_cell', 0.0):.4f}  "
                  f"min_pen={cb.get('min_pen', 0.0):.4f}  "
                  f"non_target_acc={cb.get('non_target_cell_acc_mean', 0.0):.4f}  "
                  f"small_k1={cb.get('small_k1_recall', 0.0):.4f}  "
                  f"small_k0_fp={cb.get('small_k0_fp', 0.0):.4f}  "
                  f"cls1_floor_pen={cb.get('class1_floor_pen', 0.0):.4f}  "
                  f"guard_pen={cb.get('guard_pen', 0.0):.4f}")
        # [R14 E-5] class-1 × munition 子块（若矩阵存在）
        if best_predictions.get("cls1_recall_per_mun") is not None:
            cls1_mat = best_predictions["cls1_recall_per_mun"]
            print(f"  Class-1 recall by munition × task:")
            MUN_NAMES_CLS1 = ['Small', 'Med-LM', 'Med-RD', 'Heavy']
            for m_id, name in enumerate(MUN_NAMES_CLS1):
                per_task = "  ".join(
                    f"{t}={float(cls1_mat[i, m_id].item()):.1f}%"
                    for i, t in enumerate(['K', 'M', 'F', 'C']))
                print(f"    {name:7s}: {per_task}")
        if best_predictions.get("munition_acc_matrix") is not None:
            mat = best_predictions["munition_acc_matrix"]
            samples = best_predictions["munition_samples"]
            print(f"  Munition breakdown (task-avg 3-class Acc):")
            MUN_NAMES_CONSOLE = ['Small', 'Med-LM', 'Med-RD', 'Heavy']
            for m_id, name in enumerate(MUN_NAMES_CONSOLE):
                n_m = int(samples[m_id].item())
                if n_m > 0:
                    task_avg = float(mat[:, m_id].mean().item())
                    per_task = "  ".join(f"{t}={float(mat[i, m_id].item()):.1f}%"
                                         for i, t in enumerate(['K', 'M', 'F', 'C']))
                    print(f"    {name:7s} (n={n_m:>4}): avg={task_avg:.2f}%  |  {per_task}")
        if best_predictions.get("small_m_diag") is not None:
            sm = best_predictions["small_m_diag"]
            print(f"  Small x M confusion detail:")
            print(f"    n={sm['n']}  acc={sm['acc']:.2f}%  "
                  f"true={sm['true_counts'].tolist()}  pred={sm['pred_counts'].tolist()}")
            print(f"    L0->L1={sm['l0_to_l1']:.1f}%  L0->L2={sm['l0_to_l2']:.1f}%  "
                  f"L1->L0={sm['l1_to_l0']:.1f}%  L1->L2={sm['l1_to_l2']:.1f}%  "
                  f"L2->L1={sm['l2_to_l1']:.1f}%")
            print(f"    CM rows=true[0,1,2] cols=pred[0,1,2] | {sm['cm'].tolist()}")
        if best_predictions.get("small_k_diag") is not None:
            sk = best_predictions["small_k_diag"]
            print(f"  Small x K confusion detail:")
            print(f"    n={sk['n']}  acc={sk['acc']:.2f}%  "
                  f"true={sk['true_counts'].tolist()}  pred={sk['pred_counts'].tolist()}")
            print(f"    L0->L1={sk['l0_to_l1']:.2f}%  L1->L0={sk['l1_to_l0']:.1f}%  "
                  f"L1->L2={sk['l1_to_l2']:.1f}%")
            print(f"    CM rows=true[0,1,2] cols=pred[0,1,2] | {sk['cm'].tolist()}")
        if best_predictions.get("low_cls1_cells"):
            print(f"  Class-1 floor alerts (<{CLASS1_FLOOR_RECALL:.0f}%, n>={CLASS1_FLOOR_MIN_POS}):")
            for cell in best_predictions["low_cls1_cells"][:8]:
                print(f"    {cell['munition']} x {cell['task']}: "
                      f"recall={cell['recall']:.1f}% n={cell['n_pos']} "
                      f"deficit={cell['deficit']:.1f}pt")
        if best_predictions.get("munition_thresholds") is not None:
            thr_mat = best_predictions["munition_thresholds"]
            print(f"  Global calibrated thresholds (all 8 heads):")
            for i, t in enumerate(['K', 'M', 'F', 'C']):
                print(f"    {t}1={float(thr_mat[i, 0].item()):.2f}  "
                      f"{t}2={float(thr_mat[i, 1].item()):.2f}")
        # [P0-3] per-munition 阈值矩阵（行=head, 列=m_id）
        if best_predictions.get("per_mun_thresholds") is not None:
            pm = best_predictions["per_mun_thresholds"]
            print(f"  Per-munition thresholds (P0-3, row=head, col=Small/Med-LM/Med-RD/Heavy):")
            HEADS = ['K1', 'K2', 'M1', 'M2', 'F1', 'F2', 'C1', 'C2']
            for h in HEADS:
                i = ['K', 'M', 'F', 'C'].index(h[0])
                j = int(h[1]) - 1
                vals = "  ".join(f"{float(pm[i, m, j].item()):.2f}" for m in range(4))
                print(f"    {h}: {vals}")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--smoke_test", action="store_true", help="10 epochs fast test")
    parser.add_argument("--seed", type=int, default=42, help="Training random seed")
    parser.add_argument("--ablation-config", type=str, default=None,
                        help="JSON config for a controlled ablation run.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Run directory. Relative ./output artifacts are written inside it.")
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path for a txt copy of all console output. Defaults to output/runs/damage_model/training_console_TIMESTAMP.txt",
    )
    args = parser.parse_args()

    repo_cwd = os.getcwd()
    ablation_config = {}
    if args.ablation_config:
        if load_ablation_config is None:
            raise RuntimeError("abli_exp.ablation_config could not be imported.")
        ablation_config = load_ablation_config(args.ablation_config)

    data_from_cfg = _cfg_section(ablation_config, "paths").get("data")
    data_path = os.path.abspath(args.data or data_from_cfg or "./output/damage_dataset.parquet")

    output_dir = args.output_dir
    if output_dir is None and ablation_config and resolve_output_dir is not None:
        output_dir = resolve_output_dir(ablation_config, args.seed, repo_root=repo_cwd)
    if output_dir:
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(output_dir)
        if write_resolved_config is not None:
            write_resolved_config(
                ablation_config,
                os.path.join(output_dir, "config_resolved.json"),
                extra={"seed": args.seed, "data": data_path, "output_dir": output_dir},
            )

    log_file = args.log_file or _default_console_log_path()
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    with open(log_file, "w", encoding="utf-8", buffering=1) as _log_fh:
        tee_out = TeeStream(sys.stdout, _log_fh)
        tee_err = TeeStream(sys.stderr, _log_fh)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print(f"[Training] Console log tee enabled: {log_file}")
            print(f"[Training] Working directory: {os.getcwd()}")
            train_model(data_path, args.smoke_test, seed=args.seed,
                        ablation_config=ablation_config)
