from __future__ import annotations

import copy
import json
import os
from typing import Any

from loitering_munition_damage_twin.paths import ABLATION_CONFIG_ROOT


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment_id": "A0_full",
    "description": "Credible Stage-0 v2 baseline without stacked balancing mechanisms.",
    "paths": {
        "data": "output/damage_dataset.parquet",
        "result_root": "output/experiments",
    },
    "data": {
        "use_soft_labels": True,
        # The MC standard-error weight is disabled in the improved baseline.
        # It is strongly class-correlated in the current dataset and therefore
        # changes the conditional probability estimated by weighted BCE.
        # A13_with_label_confidence restores the historical behavior as a
        # single-variable ablation.
        "use_label_uncertainty": False,
        "label_uncertainty_scale": 0.10,
        "label_confidence_floor": 0.25,
        # Rows are K/M/F/C; columns are Small/Med-LM/Med-RD/Heavy.
        # A value of 0 bypasses the MC-confidence multiplier for one cell,
        # while 1 retains it fully.
        "label_confidence_strength_by_task_munition": [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        "use_terminal_physics_features": False,
        "use_component_proxy_features": False,
        "use_armor_aware_fragment_proxies": False,
        "use_mechanism_supervision": False,
        "use_component_supervision": False,
        "use_balanced_sampler": False,
        "use_adaptive_sampler_balance": False,
        "use_adaptive_loss_balance": False,
        "pos_weight_mode": "ones",
        "k_task_weights": [1.0, 1.0, 1.0, 1.0],
        "m_task_weights": [1.0, 1.0, 1.0, 1.0],
        "c_task_weights": [1.0, 1.0, 1.0, 1.0],
    },
    "model": {
        "munition_emb_dim": 16,
        "use_munition_embedding": True,
        "use_munition_experts": True,
        "use_physics_skip": True,
        "use_k_cascade": True,
        "deep_m_branch": True,
        "ordinal_parameterization": "cumulative_logits",
        "use_mechanism_decomposition": False,
        "use_mechanism_auxiliary_heads": False,
        "mechanism_encoder_mode": "shared",
        "use_component_auxiliary_heads": False,
        "component_branch_mode": "shared_auxiliary",
        "component_branch_munition_emb_dim": 16,
        "component_tree_fusion_alpha": None,
        # Per-component analytic proxies are deployable terminal-state
        # features, but direct-path use is an explicit from-scratch ablation.
        # The default keeps A38's strict branch isolation contract.
        "allow_component_proxy_direct_path": False,
        "residual_adapter_cells": [],
        "residual_adapter_hidden_dim": 64,
        "residual_adapter_feature_indices": [0, 1, 2, 3, 4, 5],
        "residual_adapter_frequencies": [1.0, 2.0, 4.0, 8.0],
        "residual_adapter_max_logit": 2.0,
    },
    "loss": {
        "use_focal_loss": False,
        "use_ordinal_penalty": False,
        "use_class1_margin": False,
        "use_cell_class1_alpha": False,
        # Proper three-class distribution induced by the cumulative heads:
        # q0=1-p_ge1, q1=p_ge1-p_ge2, q2=p_ge2.  This directly trains the
        # narrow middle class without abandoning the monotone ordinal heads.
        "use_class_distribution_loss": True,
        "class_distribution_weight": 0.25,
        # The simulator exposes continuous damage probabilities, while the
        # reported ordinal class is defined by thresholding those probabilities
        # at 0.5.  The soft cumulative objectives preserve probability
        # calibration; this optional second objective supervises the resulting
        # hard L0/L1/L2 decision without pretending that p_ge1-p_ge2 is the
        # posterior probability of the thresholded middle class.
        "hard_level_classification_weight": 0.0,
        "middle_class_distribution_multiplier": [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        "entry_ranking_weight": [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        "conditional_l1_l2_ranking_weight": [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        "ranking_margin": 0.5,
        "hard_negative_fraction": 0.10,
        "mechanism_auxiliary_weight": 0.50,
        "mechanism_class_distribution_weight": 0.25,
        "mechanism_branch_weights": [1.0, 1.0],
        "mechanism_boundary_focus_weight": 0.0,
        "mechanism_boundary_focus_bandwidth": 0.15,
        "mechanism_hard_classification_weight": 0.0,
        "mechanism_use_dataset_row_weights": True,
        "component_auxiliary_weight": 0.10,
        "component_target_tree_teacher_weight": 0.0,
        "component_rule_consistency_weight": 0.05,
        "component_distribution_weight": 0.10,
        "component_use_positive_weight": True,
        "component_rule_entry_ranking_weight": [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        "component_rule_conditional_l1_l2_ranking_weight": [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        "component_rule_ranking_margin": 0.5,
        "component_rule_hard_negative_fraction": 0.10,
    },
    "calibration": {
        "threshold_strategy": "per_munition",
        "minimum_exact_class1_recall": 0.85,
        "maximum_class1_floor_accuracy_drop": 0.005,
        "goal_aware_cell_search": False,
        "minimum_cell_accuracy": 0.94,
        "minimum_class_diagonal_recall": 0.90,
    },
    "training": {
        "epochs": 45,
        "learning_rate": 0.001,
        "weight_decay": 0.0005,
        "minimum_selection_epochs": 32,
        "selection_patience": 10,
        "freeze_base_model": False,
        "freeze_criterion_uncertainty": False,
        "include_initial_candidate": True,
    },
    "ablation": {
        "components": [],
        "note": "No component removed.",
    },
}


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _config_dir() -> str:
    return str(ABLATION_CONFIG_ROOT)


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ablation_config(path: str | None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    base_path = os.path.join(_config_dir(), "_base.json")
    if os.path.exists(base_path):
        _deep_update(cfg, _load_json(base_path))
    if path:
        if not os.path.isabs(path):
            candidate = os.path.abspath(path)
            if not os.path.exists(candidate):
                candidate = os.path.join(_config_dir(), path)
                if not candidate.endswith(".json"):
                    candidate += ".json"
            path = candidate
        _deep_update(cfg, _load_json(path))
    return cfg


def resolve_output_dir(config: dict[str, Any], seed: int, repo_root: str | None = None) -> str:
    repo_root = os.path.abspath(repo_root or os.getcwd())
    paths = config.get("paths", {})
    result_root = paths.get("result_root", "output/experiments")
    if not os.path.isabs(result_root):
        result_root = os.path.join(repo_root, result_root)
    exp_id = config.get("experiment_id", "unnamed_ablation")
    return os.path.abspath(os.path.join(result_root, exp_id, f"seed{int(seed)}"))


def write_resolved_config(config: dict[str, Any], path: str,
                          extra: dict[str, Any] | None = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = copy.deepcopy(config)
    if extra:
        payload.setdefault("resolved", {}).update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
