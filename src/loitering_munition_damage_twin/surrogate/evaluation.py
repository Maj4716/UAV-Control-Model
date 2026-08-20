from __future__ import annotations

import torch
import numpy as np
import os
import json
import shutil
import argparse
import csv
import pickle
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    log_loss,
)

from loitering_munition_damage_twin.surrogate.model import DamageAssessmentMTL
from loitering_munition_damage_twin.surrogate.dataset import get_dataloaders, get_feature_columns
from loitering_munition_damage_twin.surrogate.artifacts import (
    data_contracts_match,
    load_and_verify_manifest,
    sha256_file,
)

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

# Strict Stage-0 v2 consumer:
# verifies the sealed model/data/scaler/threshold contract, reports both
# classification and probability quality, validates ONNX Runtime parity, and
# only then assembles a complete deployment bundle.


TASK_NAMES = ["K", "M", "F", "C"]
MUN_NAMES = ["Small", "Med-LM", "Med-RD", "Heavy"]
GLOBAL_C0_MAX_FP_RATE = 0.025
ONNX_PARITY_ATOL = 2e-5
ONNX_PARITY_RTOL = 1e-4
TASK_DESCS = {
    "K": "灾难级核心毁伤 (K_Level)",
    "M": "机动履带瘫痪 (M_Level)",
    "F": "火力系统损毁 (F_Level)",
    "C": "指挥控制损毁 (C_Level)",
}


def _write_json_atomic(path: str, payload: dict) -> None:
    absolute_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
    temporary_path = absolute_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    os.replace(temporary_path, absolute_path)


def _authorize_test_evaluation(
    promotion_report: str,
    run_dir: str,
    experiment_id: str | None = None,
) -> dict:
    """Require an in-run, test-blind validation PASS before reading test."""
    report_path = Path(promotion_report).resolve()
    resolved_run_dir = Path(run_dir).resolve()
    if not report_path.is_file():
        raise RuntimeError(
            f"Test evaluation refused: promotion report does not exist: "
            f"{report_path}")
    try:
        report_path.relative_to(resolved_run_dir)
    except ValueError as exc:
        raise RuntimeError(
            "Test evaluation refused: promotion report must be inside "
            f"the current run directory {resolved_run_dir}.") from exc
    with report_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if (
        payload.get("status") != "PASS"
        or payload.get("test_metrics_read") is not False
    ):
        raise RuntimeError(
            "Test evaluation refused: promotion report must declare "
            "status=PASS and test_metrics_read=false.")
    candidate = payload.get("candidate")
    if (
        experiment_id
        and str(candidate) != str(experiment_id)
    ):
        raise RuntimeError(
            "Test evaluation refused: promotion candidate "
            f"{candidate!r} does not match experiment "
            f"{experiment_id!r}.")
    return {
        "promotion_report": str(report_path),
        "promotion_report_sha256": sha256_file(report_path),
        "promotion_schema": payload.get("schema"),
        "promotion_candidate": candidate,
        "test_metrics_read_at_promotion": False,
    }


def _onnx_parity_stats(reference: np.ndarray,
                       candidate: np.ndarray) -> dict:
    if reference.shape != candidate.shape:
        return {
            "passed": False,
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "maximum_absolute_error": float("inf"),
            "maximum_normalized_error": float("inf"),
        }
    absolute_error = np.abs(reference - candidate)
    tolerance = (
        ONNX_PARITY_ATOL
        + ONNX_PARITY_RTOL * np.abs(reference)
    )
    normalized_error = absolute_error / np.maximum(tolerance, 1e-30)
    return {
        "passed": bool(np.all(absolute_error <= tolerance)),
        "shape_match": True,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "maximum_absolute_error": float(np.max(absolute_error)),
        "maximum_normalized_error": float(np.max(normalized_error)),
    }


def _cfg_section(ablation_config: dict | None, name: str) -> dict:
    if not ablation_config:
        return {}
    section = ablation_config.get(name, {})
    return section if isinstance(section, dict) else {}


def _load_logit_shifts(adj_path: str, dataset_sha256: str):
    """读 ordinal logit_adjustment.json（*_ge1/2_prob × log_adjust）。

    返回 ``(shifts, found_all, enabled)``。只有显式的
    ``ordinal_exceedance_v2 + enabled=true`` 才允许启用。
    """
    logit_shifts = np.zeros((4, 2), dtype=np.float32)
    if not os.path.exists(adj_path):
        raise FileNotFoundError(
            f"Missing dataset logit-adjustment metadata: {adj_path}")

    with open(adj_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    found_all = True
    for i, t in enumerate(TASK_NAMES):
        for j, lvl in enumerate(["1", "2"]):
            key = f"{t}_ge{lvl}_prob"
            if key in stats and "log_adjust" in stats[key]:
                logit_shifts[i, j] = float(stats[key]["log_adjust"])
            else:
                found_all = False
    meta = stats.get("__meta__", {})
    if meta.get("dataset_sha256") != dataset_sha256:
        raise RuntimeError(
            "logit_adjustment.json belongs to a different Parquet artifact.")
    enabled = bool(
        found_all
        and meta.get("schema") == "ordinal_exceedance_v2"
        and meta.get("enabled") is True
    )
    return logit_shifts, found_all, enabled


def _load_thresholds(thr_path: str):
    """Load global and optional per-munition calibrated thresholds."""
    if not os.path.exists(thr_path):
        raise FileNotFoundError(
            f"Missing calibrated thresholds: {thr_path}")

    with open(thr_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    schema_ver = loaded.get("_schema")
    supported_schemas = {
        "v7_monotone_fpr_constrained",
        "v8_exact_l1_floor_constrained",
    }
    if schema_ver not in supported_schemas:
        raise RuntimeError(
            f"Unsupported threshold schema {schema_ver!r}; retrain the model.")
    heads = ("K1", "K2", "M1", "M2", "F1", "F2", "C1", "C2")
    default = {}
    for head in heads:
        if head not in loaded:
            raise RuntimeError(f"Threshold file is missing {head}.")
        value = float(loaded[head])
        if not 0.0 <= value <= 1.0:
            raise RuntimeError(f"Threshold {head} is outside [0,1].")
        default[head] = value
    raw_pm = loaded.get("per_munition")
    if not isinstance(raw_pm, dict):
        raise RuntimeError("Threshold file lacks per_munition calibration.")
    per_mun_dict = {}
    for head in heads:
        head_dict = raw_pm.get(head)
        if not isinstance(head_dict, dict):
            raise RuntimeError(f"Missing per-munition thresholds for {head}.")
        per_mun_dict[head] = {}
        for munition_id in range(4):
            if str(munition_id) not in head_dict:
                raise RuntimeError(
                    f"Missing threshold {head}/munition={munition_id}.")
            value = float(head_dict[str(munition_id)])
            if not 0.0 <= value <= 1.0:
                raise RuntimeError(
                    f"Threshold {head}/munition={munition_id} outside [0,1].")
            per_mun_dict[head][munition_id] = value
    return default, per_mun_dict, schema_ver


def _load_threshold_metadata(thr_path: str) -> dict:
    if not os.path.exists(thr_path):
        return {}
    try:
        with open(thr_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        return {k: v for k, v in loaded.items() if k.startswith("_")}
    except Exception as exc:
        print(f"[EVAL] !!提示!! 读取阈值元数据失败: {exc}")
        return {}


def _print_threshold_matrix(thr_dict, per_mun_dict=None):
    print("[EVAL] 已加载全局阈值矩阵（行=task, 列=level）：")
    for t in TASK_NAMES:
        print(f"    {t}: L1={thr_dict[f'{t}1']:.3f}   L2={thr_dict[f'{t}2']:.3f}")
    if per_mun_dict is not None:
        print("[EVAL] 已加载 per-munition 阈值 (行=head, 列=Small/Med-LM/Med-RD/Heavy):")
        for head in ["K1", "K2", "M1", "M2", "F1", "F2", "C1", "C2"]:
            vals = "  ".join(f"{per_mun_dict[head][m]:.2f}" for m in range(4))
            print(f"    {head}: {vals}")


def _copy_if_exists(src: str, dst: str, label: str) -> bool:
    """Copy a required deployment artifact or fail closed."""
    if not os.path.exists(src):
        raise FileNotFoundError(f"Missing required {label}: {src}")
    shutil.copy2(src, dst)
    print(f"[DEPLOY] 已打包 {label:<22s} → {dst}")
    return True


def _evaluate_c2_challenge(challenge_path: str, model: torch.nn.Module,
                           scaler, feature_columns: list[str],
                           per_mun_thr: dict, task_thresholds: dict,
                           logit_shifts_tensor: torch.Tensor,
                           dataset_sha256: str, eval_dir: str) -> dict:
    """Evaluate root-independent rare C2 discrimination without calibration."""
    challenge = Path(challenge_path).resolve()
    profile_path = challenge.with_suffix(".profile.json")
    if not challenge.is_file() or not profile_path.is_file():
        raise FileNotFoundError(
            f"Missing C2 challenge artifact/profile: {challenge}")
    with profile_path.open("r", encoding="utf-8") as stream:
        profile = json.load(stream)
    artifact = profile.get("artifact", {})
    if profile.get("profile_schema") != "stage0_c2_challenge_v1":
        raise RuntimeError("Unsupported C2 challenge schema.")
    if int(artifact.get("size_bytes", -1)) != challenge.stat().st_size:
        raise RuntimeError("C2 challenge size does not match its profile.")
    if artifact.get("sha256") != sha256_file(str(challenge)):
        raise RuntimeError("C2 challenge SHA-256 mismatch.")
    if profile.get("source_dataset", {}).get("sha256") != dataset_sha256:
        raise RuntimeError(
            "C2 challenge was not isolated against the evaluated dataset.")
    if profile.get("purpose") != (
            "root_independent_rare_event_discrimination_not_calibration"):
        raise RuntimeError("C2 challenge purpose contract is missing.")

    frame = pd.read_parquet(challenge, engine="pyarrow")
    if len(frame) != int(artifact.get("rows", -1)):
        raise RuntimeError("C2 challenge row count mismatch.")
    required = set(feature_columns) | {
        "sample_id", "root_seed_id", "challenge_target",
        "challenge_munition_id",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"C2 challenge lacks fields: {missing}")
    if frame["root_seed_id"].astype(str).duplicated().any():
        raise RuntimeError("C2 challenge contains repeated root families.")

    x = scaler.transform(
        frame[feature_columns].to_numpy(dtype=np.float32)).astype(np.float32)
    munitions = frame["challenge_munition_id"].to_numpy(dtype=np.int64)
    targets = frame["challenge_target"].to_numpy(dtype=np.int64)
    with torch.no_grad():
        logits = model(torch.from_numpy(x), torch.from_numpy(munitions))
        probabilities = torch.sigmoid(
            logits - logit_shifts_tensor).numpy()[:, 3, 1]
    thresholds = np.asarray([
        (per_mun_thr["C2"][int(munition_id)]
         if per_mun_thr is not None else task_thresholds["C2"])
        for munition_id in munitions
    ])
    predictions = (probabilities >= thresholds).astype(np.int64)

    def _metrics(mask: np.ndarray) -> dict:
        scoped_targets = targets[mask]
        scoped_probabilities = np.clip(
            probabilities[mask], 1e-7, 1.0 - 1e-7)
        scoped_predictions = predictions[mask]
        tp = int(np.sum((scoped_predictions == 1) & (scoped_targets == 1)))
        fp = int(np.sum((scoped_predictions == 1) & (scoped_targets == 0)))
        fn = int(np.sum((scoped_predictions == 0) & (scoped_targets == 1)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        return {
            "rows": int(mask.sum()),
            "positive_rows": int(scoped_targets.sum()),
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-9),
            "average_precision": float(average_precision_score(
                scoped_targets, scoped_probabilities)),
            "brier": float(np.mean(np.square(
                scoped_probabilities - scoped_targets))),
            "nll": float(log_loss(
                scoped_targets, scoped_probabilities, labels=[0, 1])),
        }

    report = {
        "purpose": profile["purpose"],
        "selection_bias": profile.get("selection_bias"),
        "challenge_sha256": artifact["sha256"],
        "source_dataset_sha256": dataset_sha256,
        "overall": _metrics(np.ones(len(frame), dtype=bool)),
        "per_munition": {
            str(munition_id): _metrics(munitions == munition_id)
            for munition_id in sorted(np.unique(munitions))
        },
    }
    output_path = os.path.join(eval_dir, "c2_challenge_metrics.json")
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print(f"[EVAL] Independent C2 challenge metrics written to {output_path}")
    return report


def evaluate_and_export(parquet_path: str, model_path: str,
                        out_onnx: str = "./output/models/deployable_damage_model.onnx",
                        ablation_config: dict | None = None,
                        threshold_strategy: str | None = None,
                        challenge_path: str | None = None):
    device = torch.device("cpu")
    calibration_cfg = _cfg_section(ablation_config, "calibration")
    threshold_strategy = (
        threshold_strategy
        or calibration_cfg.get("threshold_strategy")
        or "per_munition"
    )
    threshold_strategy = str(threshold_strategy).lower()
    feature_columns = get_feature_columns(ablation_config)
    model_path = os.path.abspath(model_path)
    model_dir = os.path.dirname(model_path)
    if os.path.basename(model_path) != "best_model.pth":
        raise RuntimeError(
            "Evaluation accepts only the manifest-sealed best_model.pth artifact.")

    profile_path = os.path.join(
        os.path.dirname(os.path.abspath(parquet_path)),
        "generation_profile.json")
    if not os.path.isfile(profile_path):
        raise FileNotFoundError(f"Missing generation profile: {profile_path}")
    with open(profile_path, "r", encoding="utf-8") as stream:
        generation_profile = json.load(stream)
    dataset_sha256 = str(
        generation_profile.get("artifact", {}).get("sha256", ""))
    manifest = load_and_verify_manifest(
        model_dir,
        dataset_sha256=dataset_sha256,
        feature_names=feature_columns,
    )
    model_cfg = manifest["model_config"]

    scaler_path = os.path.join(model_dir, "minmax_scaler.pkl")
    with open(scaler_path, "rb") as stream:
        fitted_scaler = pickle.load(stream)

    # ------------------------------------------------------------------
    # 1. 加载测试流；严格复用训练时拟合并封存的 scaler
    # ------------------------------------------------------------------
    print(f"[EVAL] 正在加载数据集并复用训练 scaler: {parquet_path}")
    (_, _, test_loader, scaler, _pos_weight,
     data_contract) = get_dataloaders(
        parquet_path, batch_size=256, persist_scaler=False,
        ablation_config=ablation_config,
        scaler_override=fitted_scaler)
    if not data_contracts_match(
            data_contract, manifest["data_contract"]):
        raise RuntimeError(
            "Runtime data contract differs from the sealed training contract.")

    # ------------------------------------------------------------------
    # 2. 挂载模型权重
    # ------------------------------------------------------------------
    model = DamageAssessmentMTL(**model_cfg)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True),
        strict=True)
    print(f"[EVAL] 成功读入合同校验后的最优权重: {model_path}")
    model.eval()

    # ------------------------------------------------------------------
    # 3. 载入 Logit Adjustment 先验参数 (4, 2)
    #    schema: K_ge1_prob/.../C_ge2_prob → log_adjust
    #
    # Stage-0 合同：生成端写出显式 ordinal exceedance 先验；只有
    # logit_adjustment.json 标记 ordinal_exceedance_v2 且 enabled=true 时才启用。
    # 当前生成端默认 enabled=false，直到 shift 与阈值在同一概率空间联合校准。
    # ------------------------------------------------------------------
    adj_path = os.path.join(os.path.dirname(parquet_path), "logit_adjustment.json")
    if not os.path.exists(adj_path):
        adj_path = "./output/logit_adjustment.json"
    raw_shifts, adj_found_all, adjustment_enabled = _load_logit_shifts(
        adj_path, dataset_sha256)
    if adj_found_all and not adjustment_enabled:
        print("[EVAL] Logit adjustment 元数据存在但未联合校准，按 schema 明确禁用。")
    logit_shifts = raw_shifts if adjustment_enabled else np.zeros((4, 2), dtype=np.float32)
    logit_shifts_tensor = torch.tensor(logit_shifts, dtype=torch.float32)

    # ------------------------------------------------------------------
    # 4. 载入校准阈值字典 (R10/R13/R14/R15/P0-3: 全 8 头 + 可选 per-munition)
    # ------------------------------------------------------------------
    thr_path = os.path.join(model_dir, "best_thresholds.json")
    task_thresholds, per_mun_thr, thr_schema_ver = _load_thresholds(thr_path)
    thr_metadata = _load_threshold_metadata(thr_path)
    if threshold_strategy == "global":
        per_mun_thr = None
        print("[EVAL][Ablation] threshold_strategy=global; per-munition thresholds ignored.")
    elif threshold_strategy in {"fixed_0_5", "fixed_0.5", "fixed"}:
        task_thresholds = {k: 0.5 for k in ("K1", "K2", "M1", "M2", "F1", "F2", "C1", "C2")}
        per_mun_thr = None
        thr_schema_ver = "fixed_0_5"
        print("[EVAL][Ablation] threshold_strategy=fixed_0_5; all thresholds forced to 0.5.")
    elif threshold_strategy not in {"per_munition", "per-mun", "cellwise"}:
        raise ValueError(f"Unknown calibration.threshold_strategy: {threshold_strategy}")
    thr_model_variant = thr_metadata.get("_model_variant", "unknown")
    print(f"[EVAL] threshold model_variant='{thr_model_variant}' "
          f"best_epoch={thr_metadata.get('_best_epoch', 'unknown')} "
          f"selection_score={thr_metadata.get('_selection_score', 'unknown')}")
    _print_threshold_matrix(task_thresholds, per_mun_thr)

    # 构建 per-task 阈值张量（L1 与 L2 分别广播）
    lvl1_thr_global = torch.tensor([task_thresholds[f"{t}1"] for t in TASK_NAMES],
                                   dtype=torch.float32)  # (4,)
    lvl2_thr_global = torch.tensor([task_thresholds[f"{t}2"] for t in TASK_NAMES],
                                   dtype=torch.float32)  # (4,)

    # [P0-3] 若有 per-munition 阈值，构建 (4 tasks, 4 muns, 2 levels) 张量用于按 m_id gather
    use_per_mun = per_mun_thr is not None
    if use_per_mun:
        per_mun_mat = torch.zeros((4, 4, 2), dtype=torch.float32)
        for i, t in enumerate(TASK_NAMES):
            for m in range(4):
                per_mun_mat[i, m, 0] = per_mun_thr[f"{t}1"][m]
                per_mun_mat[i, m, 1] = per_mun_thr[f"{t}2"][m]
        print(f"[EVAL] 推理将使用 per-munition 阈值 (detected from best_thresholds.json)")
    else:
        per_mun_mat = None
        print(f"[EVAL] 推理使用全局阈值 (no per_munition thresholds present)")
    print(f"[EVAL] uses_per_munition_thresholds={str(bool(use_per_mun)).lower()}")

    # ------------------------------------------------------------------
    # 5. 前向评估：5 步严格计算图（逐 batch 累加预测 & 真值）
    # ------------------------------------------------------------------
    all_preds = {t: [] for t in TASK_NAMES}
    all_trues = {t: [] for t in TASK_NAMES}
    all_mids = []
    all_prob_matrices = []
    all_soft_targets = []
    all_sample_ids = []
    all_root_ids = []

    print("[EVAL] 开始推演评估（logit-shift + sigmoid + 保序钳制 + 校准阈值判定）...")
    with torch.no_grad():
        # [R1 #2] 5-tuple 解包：训练侧 loss_w / k_task_w 在 eval 阶段不需要
        for batch in test_loader:
            x, y, m_ids = batch[0], batch[1], batch[2]
            y_soft = batch[7]
            sample_ids, root_ids = batch[9], batch[10]
            x = x.to(device)
            m_ids = m_ids.to(device)

            # 真值等级 (0/1/2) 来自 [ge_1, ge_2] 叠加
            # y shape: (N, 4, 2)
            y_level = y[:, :, 0].long() + y[:, :, 1].long()  # (N, 4)

            # -> Step 1: 模型原生 logits
            raw_logits = model(x, m_ids)                     # (N, 4, 2)
            # -> Step 2: Logit 落差补偿（广播 (4,2)）
            adj_logits = raw_logits - logit_shifts_tensor
            # -> Step 3: Sigmoid 投射
            probs = torch.sigmoid(adj_logits)                # (N, 4, 2)
            # -> Step 4: 物理保序钳制 — P(y>=2) 必是 P(y>=1) 的绝对子集
            p_ge_1 = probs[:, :, 0]                          # (N, 4)
            p_ge_2 = probs[:, :, 1]
            if torch.any(p_ge_2 > p_ge_1 + 1e-6):
                raise RuntimeError(
                    "Model violated its monotone ordinal output contract.")
            p_ge_2_clamped = p_ge_2
            # -> Step 5: 阈值分界判定（L1/L2 逐任务阈值；v3 schema 时按 m_id 分派）
            if use_per_mun:
                # [P0-3] gather: per_mun_mat (4,4,2) → (4, N, 2) → (N, 4, 2)
                thr_g = per_mun_mat[:, m_ids, :]                     # (4, N, 2)
                thr_ps = thr_g.permute(1, 0, 2).contiguous()         # (N, 4, 2)
                pass_lvl1 = p_ge_1 >= thr_ps[:, :, 0]                # (N, 4)
                pass_lvl2 = p_ge_2_clamped >= thr_ps[:, :, 1]        # (N, 4)
            else:
                pass_lvl1 = p_ge_1 >= lvl1_thr_global.unsqueeze(0)   # (N, 4)
                pass_lvl2 = p_ge_2_clamped >= lvl2_thr_global.unsqueeze(0)
            pred_level = pass_lvl1.long() + (pass_lvl1 & pass_lvl2).long()  # (N,4)

            for i, t in enumerate(TASK_NAMES):
                all_preds[t].extend(pred_level[:, i].cpu().numpy().tolist())
                all_trues[t].extend(y_level[:, i].cpu().numpy().tolist())
            all_mids.extend(m_ids.cpu().numpy().tolist())
            all_prob_matrices.append(probs.cpu().numpy())
            all_soft_targets.append(y_soft.numpy())
            all_sample_ids.extend(list(sample_ids))
            all_root_ids.extend(list(root_ids))

    # ------------------------------------------------------------------
    # 6. 打印科学分类指标（R13：4 任务全打印，不再只打 K/M）
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print(f" [Classification Report @ calibrated thresholds, schema={thr_schema_ver}] ")
    print("=" * 64)
    for t in TASK_NAMES:
        if use_per_mun:
            # per-mun 下单一阈值不成立；打印 4 弹型阈值供对照
            thr_str = "per_mun " + "/".join(
                f"{per_mun_thr[f'{t}1'][m]:.2f}→{per_mun_thr[f'{t}2'][m]:.2f}"
                for m in range(4))
            print(f"\n>>> [{t}_Level — {TASK_DESCS[t]}]  {thr_str}")
        else:
            print(f"\n>>> [{t}_Level — {TASK_DESCS[t]}]  thr1={task_thresholds[f'{t}1']:.3f}  "
                  f"thr2={task_thresholds[f'{t}2']:.3f}")
        print(classification_report(all_trues[t], all_preds[t],
                                    labels=[0, 1, 2], zero_division=0))
    print("=" * 64 + "\n")

    pred_mat = np.stack([np.asarray(all_preds[t], dtype=np.int64)
                         for t in TASK_NAMES], axis=1)
    true_mat = np.stack([np.asarray(all_trues[t], dtype=np.int64)
                         for t in TASK_NAMES], axis=1)
    mid_arr = np.asarray(all_mids, dtype=np.int64)
    probability_matrix = np.concatenate(all_prob_matrices, axis=0)
    soft_target_matrix = np.concatenate(all_soft_targets, axis=0)

    def _ece(probabilities: np.ndarray, targets: np.ndarray,
             bins: int = 10) -> float:
        edges = np.linspace(0.0, 1.0, bins + 1)
        value = 0.0
        for bin_index in range(bins):
            lower, upper = edges[bin_index], edges[bin_index + 1]
            mask = ((probabilities >= lower) &
                    (probabilities < upper if bin_index < bins - 1
                     else probabilities <= upper))
            if np.any(mask):
                value += (
                    np.mean(mask)
                    * abs(float(np.mean(probabilities[mask]))
                          - float(np.mean(targets[mask])))
                )
        return float(value)

    probability_metrics = {}
    for task_index, task in enumerate(TASK_NAMES):
        for level_index in range(2):
            head = f"{task}{level_index + 1}"
            probabilities = np.clip(
                probability_matrix[:, task_index, level_index],
                1e-7, 1.0 - 1e-7)
            hard_targets = (
                true_mat[:, task_index] >= level_index + 1).astype(np.float64)
            soft_targets = soft_target_matrix[:, task_index, level_index]
            probability_metrics[head] = {
                "positive_count": int(hard_targets.sum()),
                "brier_hard": float(
                    np.mean(np.square(probabilities - hard_targets))),
                "nll_hard": float(log_loss(
                    hard_targets, probabilities, labels=[0, 1])),
                "ece_hard_10bin": _ece(probabilities, hard_targets),
                "average_precision": (
                    float(average_precision_score(
                        hard_targets, probabilities))
                    if 0 < hard_targets.sum() < len(hard_targets) else None
                ),
                "brier_mc_mean": float(
                    np.mean(np.square(probabilities - soft_targets))),
                "cross_entropy_mc_mean": float(np.mean(
                    -soft_targets * np.log(probabilities)
                    -(1.0 - soft_targets) * np.log(1.0 - probabilities))),
                "ece_mc_mean_10bin": _ece(probabilities, soft_targets),
            }

    def _root_cluster_accuracy_ci(repetitions: int = 500) -> dict:
        roots, inverse = np.unique(
            np.asarray(all_root_ids, dtype=str), return_inverse=True)
        root_counts = np.bincount(inverse).astype(np.float64)
        root_correct = np.zeros((len(roots), 4), dtype=np.float64)
        for task_index in range(4):
            np.add.at(
                root_correct[:, task_index],
                inverse,
                (pred_mat[:, task_index] == true_mat[:, task_index]).astype(
                    np.float64),
            )
        rng = np.random.default_rng(20260725)
        bootstrap = np.zeros((repetitions, 4), dtype=np.float64)
        for repetition in range(repetitions):
            sampled_roots = rng.integers(0, len(roots), size=len(roots))
            denominator = root_counts[sampled_roots].sum()
            bootstrap[repetition] = (
                root_correct[sampled_roots].sum(axis=0)
                / max(denominator, 1.0)
            )
        return {
            task: {
                "estimate": float(np.mean(
                    pred_mat[:, task_index] == true_mat[:, task_index])),
                "ci95_low": float(np.quantile(
                    bootstrap[:, task_index], 0.025)),
                "ci95_high": float(np.quantile(
                    bootstrap[:, task_index], 0.975)),
                "bootstrap_repetitions": int(repetitions),
                "root_clusters": int(len(roots)),
            }
            for task_index, task in enumerate(TASK_NAMES)
        }

    root_cluster_accuracy_ci = _root_cluster_accuracy_ci()

    def _cell_acc(task_idx: int, mun_id: int) -> float:
        mask = mid_arr == mun_id
        if not np.any(mask):
            return 0.0
        return float(np.mean(pred_mat[mask, task_idx] == true_mat[mask, task_idx]) * 100.0)

    def _class1_metrics(task_idx: int, mun_id: int):
        mask = mid_arr == mun_id
        true1 = mask & (true_mat[:, task_idx] == 1)
        pred1 = mask & (pred_mat[:, task_idx] == 1)
        tp = int(np.sum(true1 & pred1))
        fp = int(np.sum((mask & (true_mat[:, task_idx] != 1)) & pred1))
        fn = int(np.sum(true1 & (pred_mat[:, task_idx] != 1)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        return int(np.sum(true1)), precision * 100.0, recall * 100.0, f1 * 100.0

    cell_metrics = {}
    for mun_id, munition in enumerate(MUN_NAMES):
        cell_metrics[munition] = {}
        munition_mask = mid_arr == mun_id
        for task_idx, task in enumerate(TASK_NAMES):
            n_pos, precision, recall, f1 = _class1_metrics(
                task_idx, mun_id)
            level0_mask = (
                munition_mask & (true_mat[:, task_idx] == 0))
            level0_fp = (
                float(np.mean(pred_mat[level0_mask, task_idx] > 0) * 100.0)
                if np.any(level0_mask) else 0.0
            )
            cell_metrics[munition][task] = {
                "samples": int(np.sum(munition_mask)),
                "class1_positive_count": int(n_pos),
                "class1_precision": float(precision),
                "class1_recall": float(recall),
                "class1_f1": float(f1),
                "three_class_accuracy": float(
                    _cell_acc(task_idx, mun_id)),
                "level0_false_positive_rate": float(level0_fp),
            }

    small_k_n, small_k_p, small_k_r, small_k_f = _class1_metrics(0, 0)
    small_k0_mask = (mid_arr == 0) & (true_mat[:, 0] == 0)
    small_k0_fp = (
        float(np.mean(pred_mat[small_k0_mask, 0] > 0) * 100.0)
        if np.any(small_k0_mask) else 0.0)
    small_c_n, _, small_c_r, _ = _class1_metrics(3, 0)
    small_c0_mask = (mid_arr == 0) & (true_mat[:, 3] == 0)
    c0_mask = true_mat[:, 3] == 0
    small_c0_fp = (
        float(np.mean(pred_mat[small_c0_mask, 3] > 0) * 100.0)
        if np.any(small_c0_mask) else 0.0)
    c0_fp = float(np.mean(pred_mat[c0_mask, 3] > 0) * 100.0) if np.any(c0_mask) else 0.0

    low_cls1_cells = []
    target_cells = {(0, 0), (1, 0), (3, 0)}
    for task_idx, task in enumerate(TASK_NAMES):
        for mun_id, mun in enumerate(MUN_NAMES):
            if (task_idx, mun_id) in target_cells:
                continue
            n_pos, _p, recall, _f = _class1_metrics(task_idx, mun_id)
            if n_pos >= 100 and recall < 85.0:
                low_cls1_cells.append((mun, task, n_pos, recall))
    low_cls1_cells.sort(key=lambda item: item[3])

    performance_gate_failures = []
    if small_k0_fp > 0.5:
        performance_gate_failures.append(
            f"Small/K level-0 FP {small_k0_fp:.3f}% > 0.500%")
    if c0_fp > GLOBAL_C0_MAX_FP_RATE * 100.0:
        performance_gate_failures.append(
            f"global C level-0 FP {c0_fp:.3f}% > "
            f"{GLOBAL_C0_MAX_FP_RATE * 100.0:.3f}%")
    for munition in MUN_NAMES:
        for task in TASK_NAMES:
            cell = cell_metrics[munition][task]
            if (
                int(cell["class1_positive_count"]) >= 100
                and float(cell["class1_recall"]) < 85.0
            ):
                performance_gate_failures.append(
                    f"{munition}/{task} class-1 recall "
                    f"{cell['class1_recall']:.2f}% < 85.00% "
                    f"(n={cell['class1_positive_count']})")
    performance_gate = {
        "passed": not performance_gate_failures,
        "criteria": {
            "minimum_class1_positive_support": 100,
            "minimum_class1_recall_percent": 85.0,
            "maximum_small_k0_false_positive_percent": 0.5,
            "maximum_global_c0_false_positive_percent": (
                GLOBAL_C0_MAX_FP_RATE * 100.0),
        },
        "failures": performance_gate_failures,
    }

    print("[EVAL] Cell-level guardrail check @ test:")
    print(f"    Small x K1: n={small_k_n}  precision={small_k_p:.2f}%  "
          f"recall={small_k_r:.2f}%  F1={small_k_f:.2f}%")
    print(f"    Small x K: 3-class acc={_cell_acc(0, 0):.2f}%  "
          f"K0_FP={small_k0_fp:.3f}%")
    print(f"    Small x M: 3-class acc={_cell_acc(1, 0):.2f}%")
    print(f"    Small x C1: n={small_c_n}  recall={small_c_r:.2f}%  "
          f"C0_FP={c0_fp:.3f}%  SmallC0_FP={small_c0_fp:.3f}%")
    if low_cls1_cells:
        print("    Non-target class-1 recall below 85% (n>=100):")
        for mun, task, n_pos, recall in low_cls1_cells[:8]:
            print(f"      {mun} x {task}: recall={recall:.2f}%  n={n_pos}")
    else:
        print("    Non-target class-1 recall floor: all cells >=85% where n>=100")
    print(
        "    Performance gate: "
        f"{'PASS' if performance_gate['passed'] else 'FAIL'}"
    )
    for failure in performance_gate_failures[:12]:
        print(f"      - {failure}")
    print("")

    eval_dir = "./output/eval"
    os.makedirs(eval_dir, exist_ok=True)
    overall_acc = {
        task: float(np.mean(pred_mat[:, i] == true_mat[:, i]) * 100.0)
        for i, task in enumerate(TASK_NAMES)
    }
    metrics_out = {
        "dataset_sha256": dataset_sha256,
        "model_sha256": sha256_file(model_path),
        "model_manifest_schema": manifest["schema"],
        "threshold_strategy": threshold_strategy,
        "threshold_schema": thr_schema_ver,
        "overall_acc": overall_acc,
        "avg_acc": float(np.mean(list(overall_acc.values()))),
        "probability_metrics": probability_metrics,
        "root_cluster_accuracy_ci": root_cluster_accuracy_ci,
        "cell_metrics": cell_metrics,
        "performance_gate": performance_gate,
        "small_k1": {
            "n": int(small_k_n),
            "precision": float(small_k_p),
            "recall": float(small_k_r),
            "f1": float(small_k_f),
            "k0_fp": float(small_k0_fp),
        },
        "small_m": {
            "acc": float(_cell_acc(1, 0)),
        },
        "small_c1": {
            "n": int(small_c_n),
            "recall": float(small_c_r),
            "c0_fp": float(c0_fp),
            "small_c0_fp": float(small_c0_fp),
        },
        "low_cls1_cells": [
            {"munition": mun, "task": task, "n_pos": int(n_pos), "recall": float(recall)}
            for mun, task, n_pos, recall in low_cls1_cells
        ],
    }
    if challenge_path:
        metrics_out["c2_challenge"] = _evaluate_c2_challenge(
            challenge_path=challenge_path,
            model=model,
            scaler=scaler,
            feature_columns=feature_columns,
            per_mun_thr=per_mun_thr,
            task_thresholds=task_thresholds,
            logit_shifts_tensor=logit_shifts_tensor,
            dataset_sha256=dataset_sha256,
            eval_dir=eval_dir,
        )
    with open(os.path.join(eval_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)
    with open(os.path.join(eval_dir, "predictions.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "root_seed_id", "munition_id"] +
                        [f"true_{t}" for t in TASK_NAMES] +
                        [f"pred_{t}" for t in TASK_NAMES] +
                        [f"prob_{t}{level}" for t in TASK_NAMES
                         for level in (1, 2)] +
                        [f"mc_target_{t}{level}" for t in TASK_NAMES
                         for level in (1, 2)])
        for row_index, (mid, true_row, pred_row) in enumerate(
                zip(mid_arr, true_mat, pred_mat)):
            writer.writerow([
                all_sample_ids[row_index],
                all_root_ids[row_index],
                int(mid),
            ] +
                             [int(v) for v in true_row.tolist()] +
                             [int(v) for v in pred_row.tolist()] +
                             [float(v) for v in
                              probability_matrix[row_index].reshape(-1)] +
                             [float(v) for v in
                              soft_target_matrix[row_index].reshape(-1)])
    print(f"[EVAL] Machine-readable metrics written to {eval_dir}")

    # ------------------------------------------------------------------
    # 7. ONNX 导出 [R11 #15]：dummy batch=8（覆盖 4 种 m_id 且 >1，防 trace 退化）
    #     Windows 加固：路径 normalize+absolute，并主动清理同名旧文件
    #     避免 torch.onnx.export 内部 open() 命中 Errno 22 (Invalid argument)
    # ------------------------------------------------------------------
    out_onnx = os.path.abspath(os.path.normpath(out_onnx))
    os.makedirs(os.path.dirname(out_onnx), exist_ok=True)
    if os.path.exists(out_onnx):
        try:
            os.remove(out_onnx)
        except OSError as exc:
            print(f"[EXPORT][WARN] 旧 ONNX 文件 {out_onnx} 无法删除 ({exc})，"
                  f"可能被另一进程占用；继续尝试直接覆盖")
    dummy_x = torch.randn(8, len(feature_columns), dtype=torch.float32)
    dummy_m = torch.randint(0, 4, (8,), dtype=torch.long)
    torch.onnx.export(
        model,
        (dummy_x, dummy_m),
        out_onnx,
        export_params=True,
        opset_version=14,
        input_names=['encounter_features', 'munition_id'],
        output_names=['KMFC_logits_4x2'],
        dynamic_axes={
            'encounter_features':     {0: 'batch_size'},
            'munition_id':            {0: 'batch_size'},
            'KMFC_logits_4x2':        {0: 'batch_size'},
        },
    )
    print(f"[EXPORT] Network exported to ONNX static graph: {out_onnx}")
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(out_onnx))
    runtime = ort.InferenceSession(
        out_onnx, providers=["CPUExecutionProvider"])
    maximum_parity_error = 0.0
    maximum_normalized_parity_error = 0.0
    parity_passed = True
    rng = np.random.default_rng(20260725)
    for parity_batch_size in (1, 4, 7):
        parity_x = rng.normal(
            size=(parity_batch_size, len(feature_columns))).astype(np.float32)
        parity_m = rng.integers(
            0, 4, size=parity_batch_size, dtype=np.int64)
        with torch.no_grad():
            pytorch_logits = model(
                torch.from_numpy(parity_x),
                torch.from_numpy(parity_m)).numpy()
        onnx_logits = runtime.run(None, {
            "encounter_features": parity_x,
            "munition_id": parity_m,
        })[0]
        parity_stats = _onnx_parity_stats(
            pytorch_logits, onnx_logits)
        parity_passed = parity_passed and bool(
            parity_stats["passed"])
        maximum_parity_error = max(
            maximum_parity_error,
            float(parity_stats["maximum_absolute_error"]),
        )
        maximum_normalized_parity_error = max(
            maximum_normalized_parity_error,
            float(parity_stats["maximum_normalized_error"]),
        )
    if not parity_passed:
        raise RuntimeError(
            "ONNXRuntime parity failed: "
            f"max_abs_error={maximum_parity_error:.3e}, "
            f"max_normalized_error={maximum_normalized_parity_error:.3f}, "
            f"atol={ONNX_PARITY_ATOL:.1e}, rtol={ONNX_PARITY_RTOL:.1e}")
    print(
        f"[EXPORT] ONNX checker + Runtime parity PASS "
        f"(max_abs_error={maximum_parity_error:.3e}, "
        f"normalized={maximum_normalized_parity_error:.3f}, "
        f"atol={ONNX_PARITY_ATOL:.1e}, rtol={ONNX_PARITY_RTOL:.1e})")

    # ------------------------------------------------------------------
    # 8. Ship-to-Prod 部署包 [R11 #17 + R14 + R15]
    #    结构：./output/deploy/ = {onnx, scaler.pkl, scaler.json,
    #                              logit_adjustment.json, best_thresholds.json,
    #                              deploy_config.json}
    # ------------------------------------------------------------------
    deploy_dir = os.path.abspath(os.path.normpath("./output/deploy"))
    os.makedirs(deploy_dir, exist_ok=True)

    deploy_onnx_path   = os.path.join(deploy_dir, "damage_model.onnx")
    deploy_pytorch_path = os.path.join(deploy_dir, "best_model.pth")
    deploy_scaler_pkl  = os.path.join(deploy_dir, "minmax_scaler.pkl")
    deploy_scaler_json = os.path.join(deploy_dir, "minmax_scaler.json")
    deploy_adj_path    = os.path.join(deploy_dir, "logit_adjustment.json")
    deploy_thr_path    = os.path.join(deploy_dir, "best_thresholds.json")
    deploy_effective_thr_path = os.path.join(
        deploy_dir, "effective_thresholds.json")
    deploy_manifest_path = os.path.join(deploy_dir, "model_manifest.json")
    deploy_cfg_path    = os.path.join(deploy_dir, "deploy_config.json")
    effective_threshold_payload = {
        "_schema": "stage0_effective_thresholds_v1",
        "strategy": threshold_strategy,
        "global": task_thresholds,
        "per_munition": per_mun_thr if use_per_mun else None,
    }
    with open(
            deploy_effective_thr_path, "w", encoding="utf-8") as stream:
        json.dump(
            effective_threshold_payload, stream, indent=2,
            ensure_ascii=False)

    status = {
        "damage_model.onnx":     _copy_if_exists(out_onnx, deploy_onnx_path,
                                                 "ONNX model"),
        "best_model.pth": _copy_if_exists(
            model_path, deploy_pytorch_path, "PyTorch reference model"),
        "minmax_scaler.pkl":     _copy_if_exists(os.path.join(model_dir, "minmax_scaler.pkl"),
                                                 deploy_scaler_pkl, "MinMax scaler (pkl)"),
        "minmax_scaler.json":    _copy_if_exists(os.path.join(model_dir, "minmax_scaler.json"),
                                                 deploy_scaler_json, "MinMax scaler (json)"),
        "logit_adjustment.json": _copy_if_exists(adj_path, deploy_adj_path,
                                                 "Logit adjustment"),
        "best_thresholds.json":  _copy_if_exists(thr_path, deploy_thr_path,
                                                 "Best thresholds"),
        "effective_thresholds.json": True,
        "model_manifest.json": _copy_if_exists(
            os.path.join(model_dir, "model_manifest.json"),
            deploy_manifest_path, "Model manifest"),
    }
    if not all(status.values()):
        raise RuntimeError("Deployment bundle is incomplete.")

    deploy_config = {
        "version": "stage0-nn-deploy-v1",
        "dataset_sha256": dataset_sha256,
        "model_manifest_schema": manifest["schema"],
        "ordinal_applicability": data_contract["ordinal_applicability"],
        "onnx_validation": {
            "checker_passed": True,
            "runtime_parity_passed": True,
            "maximum_absolute_error": maximum_parity_error,
            "maximum_normalized_error": maximum_normalized_parity_error,
            "absolute_tolerance": ONNX_PARITY_ATOL,
            "relative_tolerance": ONNX_PARITY_RTOL,
        },
        "threshold_schema": thr_schema_ver,
        "threshold_strategy": threshold_strategy,
        "model_variant": thr_model_variant,
        "threshold_best_epoch": thr_metadata.get("_best_epoch"),
        "threshold_raw_best_epoch": thr_metadata.get("_raw_best_epoch"),
        "threshold_soup_epochs": thr_metadata.get("_soup_epochs", []),
        "threshold_selection_score": thr_metadata.get("_selection_score"),
        "threshold_metadata": thr_metadata,
        "rare_cell_thresholds": thr_metadata.get("_rare_cell_thresholds", {}),
        "low_class1_cells": thr_metadata.get("_low_class1_cells", []),
        "uses_per_munition_thresholds": bool(use_per_mun),
        "logit_adjustment_enabled": bool(adjustment_enabled),
        "performance_gate": metrics_out.get("performance_gate", {}),
        "description": (
            "Stage-0 v2 monotone ordinal damage surrogate. "
            "MinMax缩放、模型、结构零、阈值和全部哈希均由同一产物合同封存。"
        ),
        "input_schema": {
            "feature_count": len(feature_columns),
            "features_active_order": feature_columns,
            "munition_id_range": [0, 3],
            "munition_id_map": {
                "0": "Small",
                "1": "Med-LM",
                "2": "Med-RD",
                "3": "Heavy",
            },
        },
        "output_schema": {
            "logits_shape": [4, 2],
            "task_order": TASK_NAMES,
            "level_order": ["P(y>=1)", "P(y>=2)"],
        },
        "artifacts": {
            "onnx_model":         "damage_model.onnx",
            "pytorch_reference":  "best_model.pth",
            "minmax_scaler_pkl":  "minmax_scaler.pkl",
            "minmax_scaler_json": "minmax_scaler.json",
            "logit_adjustment":   "logit_adjustment.json",
            "thresholds":         "best_thresholds.json",
            "effective_thresholds": "effective_thresholds.json",
            "model_manifest":      "model_manifest.json",
        },
        "artifact_status": status,
        # 即使散文件丢失，以下硬编码嵌入 JSON 可独立复原推理链路
        "logit_shifts_matrix": logit_shifts.tolist(),
        "thresholds": task_thresholds,
        "per_munition_thresholds": per_mun_thr,  # None or {head: {m_id: thr}}
        "inference_steps": [
            "1. features = MinMaxScaler.transform(raw_features)         "
            "# 严格沿用训练 fit 的 (data_min_, data_max_)",
            "2. logits = onnx.run({encounter_features, munition_id})     "
            "# 输出 (N, 4, 2)",
            "3. logits_adj = logits - logit_shifts_matrix               "
            "# 仅在同一概率空间联合校准后启用；当前通常为零矩阵",
            "4. probs = sigmoid(logits_adj)                             "
            "# (N, 4, 2)",
            "5. assert probs[:, :, 1] <= probs[:, :, 0]                 "
            "# 模型输出头天然保序，运行时只验证",
            "6. pred_level = 1·(p1>=thr1) + 1·((p1>=thr1) & (p2>=thr2)) "
            "# 返回 0/1/2 毁伤等级",
        ],
        "metrics_snapshot_note": (
            "概率指标同时报告hard-label与MC均值口径；低支持单元继承全局L2阈值，"
            "结构零由模型掩码；可选C2 challenge仅用于独立root稀有事件判别，"
            "禁止用于阈值校准或部署先验估计。"
        ),
    }

    deploy_config["artifact_sha256"] = {
        name: sha256_file(os.path.join(deploy_dir, name))
        for name in status
    }
    with open(deploy_cfg_path, "w", encoding="utf-8") as f:
        json.dump(deploy_config, f, indent=2, ensure_ascii=False)
    print(f"[DEPLOY] 部署包索引已写入: {deploy_cfg_path}")
    print(f"[DEPLOY] 部署目录内容:")
    for name, ok in status.items():
        tag = "OK " if ok else "!! "
        print(f"    [{tag}] {name}")
    print(f"[DEPLOY] version='{deploy_config['version']}'  "
          f"threshold_schema='{thr_schema_ver}'  "
          f"model_variant='{thr_model_variant}'  "
          f"uses_per_munition_thresholds={str(bool(use_per_mun)).lower()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  type=str, default=None)
    parser.add_argument("--model", type=str, default="./output/models/best_model.pth")
    parser.add_argument("--ablation-config", type=str, default=None,
                        help="JSON config for a controlled ablation run.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Run directory. Relative ./output artifacts are read/written inside it.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold-strategy", type=str, default=None,
                        choices=["per_munition", "global", "fixed_0_5"])
    parser.add_argument(
        "--challenge-data", type=str, default=None,
        help="Optional root-independent stage0_c2_challenge.parquet.")
    parser.add_argument(
        "--promotion-report", type=str, required=True,
        help=(
            "Validation-only PASS report inside the current run directory. "
            "The report must declare test_metrics_read=false."))
    args = parser.parse_args()

    repo_cwd = os.getcwd()
    ablation_config = {}
    if args.ablation_config:
        if load_ablation_config is None:
            raise RuntimeError("abli_exp.ablation_config could not be imported.")
        ablation_config = load_ablation_config(args.ablation_config)

    data_from_cfg = _cfg_section(ablation_config, "paths").get("data")
    data_path = os.path.abspath(args.data or data_from_cfg or "./output/damage_dataset.parquet")
    challenge_path = (
        os.path.abspath(args.challenge_data) if args.challenge_data else None)

    output_dir = args.output_dir
    if output_dir is None and ablation_config and resolve_output_dir is not None:
        output_dir = resolve_output_dir(ablation_config, args.seed, repo_root=repo_cwd)
    output_dir = os.path.abspath(output_dir or repo_cwd)
    test_authorization = _authorize_test_evaluation(
        args.promotion_report,
        output_dir,
        ablation_config.get("experiment_id"),
    )
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(output_dir)
        if write_resolved_config is not None:
            write_resolved_config(
                ablation_config,
                os.path.join(output_dir, "config_resolved.json"),
                extra={"seed": args.seed, "data": data_path, "output_dir": output_dir},
            )

    evaluation_status_path = os.path.abspath(
        "./output/eval/evaluation_status.json")
    evaluation_identity = {
        "schema": "stage0_nn_evaluation_status_v1",
        "experiment_id": ablation_config.get("experiment_id"),
        "seed": int(args.seed),
        "test_authorization": test_authorization,
    }
    _write_json_atomic(evaluation_status_path, {
        **evaluation_identity,
        "status": "RUNNING",
    })
    try:
        evaluate_and_export(
            data_path,
            args.model,
            ablation_config=ablation_config,
            threshold_strategy=args.threshold_strategy,
            challenge_path=challenge_path,
        )
        metrics_path = os.path.abspath(
            "./output/eval/test_metrics.json")
        deploy_config_path = os.path.abspath(
            "./output/deploy/deploy_config.json")
        if not os.path.isfile(metrics_path):
            raise RuntimeError(
                f"Evaluation completed without metrics: {metrics_path}")
        if not os.path.isfile(deploy_config_path):
            raise RuntimeError(
                "Evaluation completed without a sealed deployment config: "
                f"{deploy_config_path}")
        with open(metrics_path, "r", encoding="utf-8") as stream:
            completed_metrics = json.load(stream)
        _write_json_atomic(evaluation_status_path, {
            **evaluation_identity,
            "status": "COMPLETE",
            "metrics_path": metrics_path,
            "metrics_sha256": sha256_file(metrics_path),
            "deploy_config_path": deploy_config_path,
            "deploy_config_sha256": sha256_file(deploy_config_path),
            "performance_gate_passed": bool(
                completed_metrics.get(
                    "performance_gate", {}).get("passed", False)),
        })
        print(
            "[EVAL] Pipeline completion marker: "
            f"{evaluation_status_path}")
    except Exception as exc:
        _write_json_atomic(evaluation_status_path, {
            **evaluation_identity,
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        raise
