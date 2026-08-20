# -*- coding: utf-8 -*-
"""
数据存储模块 — 面向深度学习的数据集持久化
==========================================

功能:
  - 将仿真结果构建为结构化 DataFrame
  - 支持 Parquet (pyarrow) 和 HDF5 格式持久化
  - 支持加载已保存的数据集
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

# from batch_runner import SimResult
from loitering_munition_damage_twin.simulation.engine import Projectile


# ============================================================================
#  DataFrame 构建
# ============================================================================

# 12维输入特征列名
FEATURE_COLUMNS = [
    "x_cm", "y_cm", "z_cm",        # 空间坐标
    "vx_ms", "vy_ms", "vz_ms",     # 速度分量
    "sin_yaw", "cos_yaw",
    "sin_pitch", "cos_pitch",
    "sin_roll", "cos_roll"
]

# 毁伤树 8 维概率向量列名
LEVEL_VECTOR_COLUMNS = [
    "K1_prob", "K2_prob",
    "M1_prob", "M2_prob",
    "F1_prob", "F2_prob",
    "C1_prob", "C2_prob",
]

# 输出标签列名
OUTPUT_COLUMNS = [
    "overall_score",
    *LEVEL_VECTOR_COLUMNS,
    "total_hits", "total_penetrations",
    "damaged_count", "total_components",
    "K_level", "M_level", "F_level", "C_level",
    "sample_weight"
]


def build_dataframe(
    sim_results: List[SimResult],
    projectile: Projectile,
) -> pd.DataFrame:
    """
    将仿真结果列表转换为结构化 DataFrame。

    参数:
        sim_results: 单个弹型下的仿真结果列表
        projectile:  对应弹药参数

    返回:
        DataFrame, 列序:
          弹药参数 | 9维输入特征 | overall_score | 8维level_vector |
          total_hits | total_penetrations | damaged_count | ...
    """
    rows = []
    w = projectile.warhead
    fb = w.fragment_bed

    for r in sim_results:
        row = {
            "sample_index": r.sample_index,
            # ---- 弹药参数 (标识列) ----
            "projectile_name": projectile.name,
            "charge_mass_kg": w.charge_mass_kg,
            "tnt_equivalent": w.tnt_equivalent,
            "frag_count": fb.total_count,
            "frag_mass_g": fb.single_mass_g,
            "detonation_point": w.detonation_point.value,

            # ---- 9 维输入特征 ----
            **{col: val for col, val in zip(FEATURE_COLUMNS, r.encounter_features)},

            # ---- 速度标量 (派生特征) ----
            "speed_ms": float(np.linalg.norm(r.encounter_features[3:6])),

            # ---- 输出标签 ----
            "overall_score": r.overall_score,
            **{col: val for col, val in zip(LEVEL_VECTOR_COLUMNS, r.level_vector)},
            "total_hits": r.total_hits,
            "total_penetrations": r.total_penetrations,
            "damaged_count": r.damaged_count,
            "total_components": r.total_components,
            "K_level": r.K_level,
            "M_level": r.M_level,
            "F_level": r.F_level,
            "C_level": r.C_level,
            "sample_weight": r.sample_weight,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if df.isnull().values.any():
        num_nans = df.isnull().sum().sum()
        raise ValueError(f"CRITICAL DATA CORRUPTION: Found {num_nans} NaN or null values in dataset. Upstream simulation pipeline failure.")

    return df


def build_multi_projectile_dataframe(
    all_results: Dict[str, List[SimResult]],
    projectiles: List[Projectile],
) -> pd.DataFrame:
    """
    合并多个弹型的仿真结果为单个 DataFrame。

    参数:
        all_results: {弹型名称: SimResult 列表}
        projectiles: 弹药参数列表 (与 all_results 顺序对应)

    返回:
        合并后的 DataFrame
    """
    proj_map = {p.name: p for p in projectiles}
    dfs = []
    for name, results in all_results.items():
        proj = proj_map[name]
        df = build_dataframe(results, proj)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# ============================================================================
#  持久化: 保存 & 加载
# ============================================================================

def save_dataset(
    df: pd.DataFrame,
    output_path: str,
    fmt: str = "parquet",
) -> str:
    """
    将 DataFrame 持久化到文件。

    参数:
        df:          数据集
        output_path: 输出文件路径 (不含扩展名则自动追加)
        fmt:         格式, "parquet" 或 "hdf5"

    返回:
        实际保存的文件路径
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if fmt == "parquet":
        if not output_path.endswith(".parquet"):
            output_path += ".parquet"
        df.to_parquet(output_path, engine="pyarrow", index=False)
    elif fmt in ("hdf5", "h5"):
        if not output_path.endswith(".h5"):
            output_path += ".h5"
        df.to_hdf(output_path, key="damage_dataset", mode="w", complevel=5)
    else:
        raise ValueError(f"不支持的格式: {fmt}, 可选: parquet, hdf5")

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"[数据存储] 保存 {len(df)} 行 × {len(df.columns)} 列 → {output_path} ({size_mb:.2f} MB)")
    return output_path


def load_dataset(path: str) -> pd.DataFrame:
    """
    加载已保存的数据集。

    参数:
        path: 文件路径 (.parquet 或 .h5)

    返回:
        DataFrame
    """
    if path.endswith(".parquet"):
        df = pd.read_parquet(path, engine="pyarrow")
    elif path.endswith(".h5"):
        df = pd.read_hdf(path, key="damage_dataset")
    else:
        raise ValueError(f"不支持的文件扩展名: {path}")

    print(f"[数据存储] 加载 {len(df)} 行 × {len(df.columns)} 列 ← {path}")
    return df
