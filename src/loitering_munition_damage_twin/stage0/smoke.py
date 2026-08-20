"""Build and validate a tiny Stage-0 dataset using the real damage engine."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from loitering_munition_damage_twin.stage0.generation import (
    CONFIG,
    PhysicsAwareSampler,
    _build_generation_profile,
    _emit_logit_adjustment,
    _finalize_sample_weights,
    _init_worker,
    _process_single_encounter,
    _write_dataset_with_profile,
)
from loitering_munition_damage_twin.stage0.validation import (
    validate_stage0_dataset,
)


def build_smoke_dataset(rows: int, mc_replicates: int, output_dir: str) -> dict:
    if rows < 16:
        raise ValueError("--rows 至少为 16，才能稳定覆盖四弹型和 train/val/test。")
    if mc_replicates < 1:
        raise ValueError("--mc-replicates 必须 >= 1。")

    np.random.seed(CONFIG["RANDOM_SEED"])
    sampler = PhysicsAwareSampler()

    # 采样多于目标行数，物理过滤后再截取；保留生成器真实的坐标、姿态和 split 合同。
    candidates = sampler.generate_phase_1(max(rows * 3, 64))
    if candidates.empty:
        raise RuntimeError("冒烟采样未产生可用候选。")
    candidates = candidates.copy()
    # Smoke mode uses a fixed count by setting the adaptive bounds equal.
    candidates["label_mc_min_replicates"] = int(mc_replicates)
    candidates["label_mc_max_replicates"] = int(mc_replicates)

    # 小规模验证故意串行，避免测试命令占满整机 CPU；生产管线仍使用多进程。
    _init_worker(sampler.components, sampler.plates)
    results = [
        _process_single_encounter((i, row))
        for i, row in enumerate(candidates.to_dict("records"))
    ]
    result_df = pd.DataFrame(results)

    # 确保验证产物稳定覆盖三个引用 split，同时尽量接近请求行数。
    selected_parts = []
    for role in ("train", "val", "test"):
        role_df = result_df[result_df["split_role"] == role]
        if role_df.empty:
            raise RuntimeError(f"冒烟样本未覆盖 {role} split，请增加 --rows。")
        selected_parts.append(role_df.iloc[[0]])
    selected_ids = {part.iloc[0]["sample_id"] for part in selected_parts}
    remainder = result_df[~result_df["sample_id"].isin(selected_ids)]
    selected_parts.append(remainder.head(max(0, rows - 3)))
    final_df = pd.concat(selected_parts, ignore_index=True).head(rows).copy()
    final_df = _finalize_sample_weights(
        final_df, float(CONFIG["VALID_PROB_STRICT"]))

    counts = {
        m_id: int((final_df["munition_id"] == m_id).sum())
        for m_id in range(4)
    }
    profile = _build_generation_profile(
        final_df=final_df,
        final_quota=counts,
        phase1_kept_counts=counts,
        phase2_task_counts={m_id: {} for m_id in range(4)},
        seed_th=float(CONFIG["SEED_PROB_RELAX"]),
        valid_th=float(CONFIG["VALID_PROB_STRICT"]),
        target_total=len(final_df),
        phase1_ratio=1.0,
    )
    profile["artifact_kind"] = "stage0_smoke_not_for_training"
    final_df.attrs["generation_profile"] = profile

    output_dir = os.path.abspath(output_dir)
    dataset_path = os.path.join(output_dir, "damage_dataset.parquet")
    adjustment_path = os.path.join(output_dir, "logit_adjustment.json")
    profile_path = _write_dataset_with_profile(final_df, dataset_path)
    with open(profile_path, "r", encoding="utf-8") as profile_handle:
        written_profile = json.load(profile_handle)
    _emit_logit_adjustment(
        final_df,
        float(CONFIG["VALID_PROB_STRICT"]),
        CONFIG["PHYSICAL_PRIOR"],
        adjustment_path,
        dataset_sha256=written_profile["artifact"]["sha256"],
    )
    return validate_stage0_dataset(dataset_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 Stage-0 小规模端到端冒烟数据集。")
    parser.add_argument("--rows", type=int, default=24, help="最终行数，默认 24。")
    parser.add_argument("--mc-replicates", type=int, default=2, help="每状态蒙特卡洛重复次数，默认 2。")
    parser.add_argument("--output-dir", default="output/stage0_smoke", help="独立输出目录。")
    args = parser.parse_args()
    report = build_smoke_dataset(args.rows, args.mc_replicates, args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
