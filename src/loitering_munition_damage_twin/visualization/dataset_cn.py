# -*- coding: utf-8 -*-
"""
仿真数据集多维诊断与深度学习体检脚本

模块 1-4：原物理分布 / IPW / 弹药平权诊断（修复了 R 区间错误、过密散点、
          调色板写法、相关矩阵 cbar 等问题）。
模块 5-8：面向深度学习管线（nn_dataset.py + nn_train.py）追加的体检：
          特征边缘分布、特征-标签预测效力、(munition×task×class) 稀有格子、
          loss_weight 对数尾部 / ESS、layer_type 配额、train/val/test 切分一致性。
"""

import json
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# 【学术绘图全局样式设置】
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
    "Arial Unicode MS", "DejaVu Sans"
]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["figure.dpi"] = 150

# 与 nn_dataset.FEATURE_COLUMNS 保持一致；此脚本避免额外引入 torch 依赖。
FEATURE_COLUMNS = [
    "x_cm", "y_cm", "z_cm",
    "vx_ms", "vy_ms", "vz_ms",
    "sin_yaw", "cos_yaw",
    "sin_pitch", "cos_pitch",
    "sin_roll", "cos_roll",
    "norm_velocity",
    "los_distance",
    "impact_cosine",
]
LABEL_COLUMNS = ["K_level", "M_level", "F_level", "C_level"]
PROB_COLUMNS = [
    "K1_prob", "K2_prob", "M1_prob", "M2_prob",
    "F1_prob", "F2_prob", "C1_prob", "C2_prob",
]
MUN_NAMES = {0: "小型", 1: "中型-前起爆", 2: "中型-后起爆", 3: "重型"}
SPLIT_NAMES = {"train": "训练", "val": "验证", "test": "测试"}


def reconstruct_aerodynamic_aoa(df: pd.DataFrame) -> np.ndarray:
    """与 generate_dataset.py 一致的 AoA 重构口径（基于 body x-axis）。"""
    v = df["norm_velocity"].clip(lower=1e-6)
    dir_x = df["vx_ms"] / v
    dir_y = df["vy_ms"] / v
    dir_z = df["vz_ms"] / v
    h_x = df["cos_pitch"] * df["cos_yaw"]
    h_y = df["cos_pitch"] * df["sin_yaw"]
    h_z = -df["sin_pitch"]
    cos_aoa = np.clip(dir_x * h_x + dir_y * h_y + dir_z * h_z, -1.0, 1.0)
    return np.degrees(np.arccos(cos_aoa))


def integrity_check(df: pd.DataFrame) -> dict:
    """事前体检：NaN/Inf、sin^2+cos^2=1、norm_velocity ≡ ‖v‖。"""
    report = {}
    relevant = FEATURE_COLUMNS + LABEL_COLUMNS + PROB_COLUMNS + ["loss_weight"]
    nan_counts = df[relevant].isna().sum()
    inf_counts = df[relevant].apply(lambda s: np.isinf(s).sum())
    report["nan_total"] = int(nan_counts.sum())
    report["inf_total"] = int(inf_counts.sum())
    report["nan_per_col"] = nan_counts[nan_counts > 0].to_dict()
    report["inf_per_col"] = inf_counts[inf_counts > 0].to_dict()

    for tag in ("yaw", "pitch", "roll"):
        s = df[f"sin_{tag}"] ** 2 + df[f"cos_{tag}"] ** 2
        report[f"{tag}_sincos_max_dev"] = float((s - 1).abs().max())

    v_recon = np.sqrt(df["vx_ms"] ** 2 + df["vy_ms"] ** 2 + df["vz_ms"] ** 2)
    report["norm_velocity_max_dev"] = float((v_recon - df["norm_velocity"]).abs().max())
    return report


def _safe_sample(df: pd.DataFrame, n: int = 8000, seed: int = 0) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


# =========================================================================
# 模块一：空间保体积填充测度（修复半球壳层理论 PDF）
# =========================================================================
def module_01_spatial(df: pd.DataFrame, out_dir: Path):
    fig = plt.figure(figsize=(18, 5))
    sub = _safe_sample(df, 8000, seed=1)

    ax1 = fig.add_subplot(131)
    sns.scatterplot(x="x_cm", y="y_cm", data=sub, s=6, alpha=0.5,
                    edgecolor=None, color="#2c7bb6", ax=ax1)
    ax1.set_title(f"起爆点水平顶视分布 (X-Y)  [抽样 n={len(sub)}]")
    ax1.set_aspect("equal", adjustable="datalim")

    ax2 = fig.add_subplot(132)
    sns.scatterplot(x="x_cm", y="z_cm", data=sub, s=6, alpha=0.5,
                    edgecolor=None, color="#d7191c", ax=ax2)
    ax2.set_title(f"起爆点垂直侧视分布 (X-Z)  [抽样 n={len(sub)}]")
    ax2.set_aspect("equal", adjustable="datalim")

    ax3 = fig.add_subplot(133)
    r = df["radial_dist"].values
    r_min, r_max = float(r.min()), float(r.max())
    sns.histplot(r, bins=60, stat="density", color="gray", alpha=0.45, ax=ax3)
    # 半球壳层均匀体积分布的理论 PDF：f(r) = 3 r^2 / (R^3 - r0^3)
    r_grid = np.linspace(r_min, r_max, 200)
    pdf_shell = 3 * r_grid ** 2 / (r_max ** 3 - r_min ** 3)
    pdf_uniform = np.full_like(r_grid, 1.0 / (r_max - r_min))
    ax3.plot(r_grid, pdf_shell, "k--", lw=2,
             label=fr"半球壳体积均匀 $f(r)=3r^2/(R^3-r_0^3)$，$r_0={r_min:.0f},R={r_max:.0f}$")
    ax3.plot(r_grid, pdf_uniform, color="#888", lw=1.5, linestyle=":",
             label="径向均匀参考")
    ax3.set_title("径向距离密度检查")
    ax3.set_xlabel("径向距离 R (cm)")
    ax3.set_ylabel("密度")
    ax3.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_dir / "diag_01_spatial_volumetric.png")
    plt.close(fig)
    print("[+] diag_01_spatial_volumetric.png")


# =========================================================================
# 模块二：运动学流形与特征解耦正交性
# =========================================================================
def module_02_kinematic(df: pd.DataFrame, out_dir: Path):
    fig = plt.figure(figsize=(12, 5))
    sub = _safe_sample(df, 8000, seed=2)

    ax1 = fig.add_subplot(121)
    sns.histplot(df["norm_velocity"], bins=40, kde=True, color="#fdae61", ax=ax1)
    ax1.set_title("速度模长分布")
    ax1.set_xlabel("速度 (m/s)")
    ax1.set_ylabel("计数")

    ax2 = fig.add_subplot(122)
    sns.scatterplot(x="cos_yaw", y="sin_yaw", data=sub, s=6, alpha=0.6,
                    color="#1a9641", ax=ax2)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax2.plot(np.cos(theta), np.sin(theta), "k--", lw=1, alpha=0.7,
             label="单位圆")
    ax2.set_title("偏航角相位流形 (cos, sin)")
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out_dir / "diag_02_kinematic_manifold.png")
    plt.close(fig)
    print("[+] diag_02_kinematic_manifold.png")


# =========================================================================
# 模块三：气动软衰减与 IPW 机制验证
# =========================================================================
def module_03_aero_ipw(df: pd.DataFrame, out_dir: Path):
    fig = plt.figure(figsize=(14, 5))
    sub = _safe_sample(df, 12000, seed=3)

    ax1 = fig.add_subplot(121)
    sns.histplot(df["aerodynamic_aoa"], bins=60, color="#d7191c",
                 stat="density", kde=False, ax=ax1)
    ax1.axvline(15, color="k", linestyle="--", label="15° 软拒止起点")
    ax1.axvline(30, color="k", linestyle="-", label="30° 硬拒止边界")
    ax1.set_title("气动攻角边缘分布")
    ax1.set_xlabel("重构气动攻角 (deg)")
    ax1.set_ylabel("密度")
    ax1.legend()

    ax2 = fig.add_subplot(122)
    sns.scatterplot(x="aerodynamic_aoa", y="loss_weight", data=sub,
                    s=10, alpha=0.45, color="#2b83ba", ax=ax2)
    ax2.set_yscale("log")
    ax2.set_title("IPW 损失权重随气动攻角的响应（纵轴对数）")
    ax2.set_xlabel("气动攻角 (deg)")
    ax2.set_ylabel("loss_weight（对数尺度）")

    plt.tight_layout()
    fig.savefig(out_dir / "diag_03_aerodynamic_ipw.png")
    plt.close(fig)
    print("[+] diag_03_aerodynamic_ipw.png")


# =========================================================================
# 模块四：弹药平权与联合毁伤效能
# =========================================================================
def module_04_damage(df: pd.DataFrame, out_dir: Path):
    fig = plt.figure(figsize=(18, 5))

    ax1 = fig.add_subplot(131)
    sns.countplot(x="munition_id", hue="munition_id", data=df,
                  palette="Set2", legend=False, ax=ax1)
    ax1.set_title("弹型编号分配均衡性")
    ax1.set_xlabel("弹型编号 (0..3)")
    ax1.set_ylabel("计数")

    ax2 = fig.add_subplot(132)
    sns.violinplot(x="munition_id", y="overall_score", hue="munition_id",
                   data=df, palette="Set2", inner="quartile",
                   legend=False, ax=ax2)
    ax2.set_title("综合毁伤评分分层分布")
    ax2.set_xlabel("弹型")

    ax3 = fig.add_subplot(133)
    corr_cols = ["norm_velocity", "aerodynamic_aoa",
                 "total_penetrations", "overall_score",
                 "K_level", "M_level", "F_level", "C_level"]
    corr_mat = df[corr_cols].corr(method="spearman")
    sns.heatmap(corr_mat, annot=True, cmap="coolwarm", fmt=".2f",
                vmin=-1, vmax=1, square=True, cbar=True, ax=ax3)
    ax3.set_title("物理诊断量与毁伤等级的 Spearman 相关")

    plt.tight_layout()
    fig.savefig(out_dir / "diag_04_damage_efficacy.png")
    plt.close(fig)
    print("[+] diag_04_damage_efficacy.png")


# =========================================================================
# 模块五：15 维特征边缘分布 + 一致性体检（DL 归一化前的体检）
# =========================================================================
def module_05_feature_marginal(df: pd.DataFrame, integrity: dict, out_dir: Path):
    fig, axes = plt.subplots(4, 4, figsize=(18, 14))
    axes = axes.flatten()

    for i, col in enumerate(FEATURE_COLUMNS):
        ax = axes[i]
        sns.histplot(df[col], bins=50, color="#3690c0",
                     stat="density", kde=False, ax=ax)
        skew = stats.skew(df[col].values)
        kurt = stats.kurtosis(df[col].values)
        ax.set_title(f"{col}\n偏度={skew:+.2f}  峰度={kurt:+.2f}", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")

    # 最后一个子图：把一致性体检结果做成文字面板，剩余空轴隐藏。
    panel_idx = len(FEATURE_COLUMNS)
    ax = axes[panel_idx]
    ax.axis("off")
    lines = [
        "完整性检查",
        "-" * 26,
        f"NaN 总数 : {integrity['nan_total']}",
        f"Inf 总数 : {integrity['inf_total']}",
        f"|sin^2+cos^2 - 1| 最大偏差:",
        f"  yaw   = {integrity['yaw_sincos_max_dev']:.2e}",
        f"  pitch = {integrity['pitch_sincos_max_dev']:.2e}",
        f"  roll  = {integrity['roll_sincos_max_dev']:.2e}",
        f"||v|| - norm_velocity 最大偏差:",
        f"  = {integrity['norm_velocity_max_dev']:.2e}",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.6", fc="#f7f7f7", ec="#bbb"))
    for spare_ax in axes[panel_idx + 1:]:
        spare_ax.axis("off")
    fig.suptitle("特征边缘分布（15 维）与完整性审计",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    fig.savefig(out_dir / "diag_05_feature_marginal.png", bbox_inches="tight")
    plt.close(fig)
    print("[+] diag_05_feature_marginal.png")


# =========================================================================
# 模块六：特征-标签预测效力 + 特征-特征共线性
# =========================================================================
def module_06_feature_label_corr(df: pd.DataFrame, out_dir: Path):
    fig = plt.figure(figsize=(18, 6))

    # 左：15 特征 × 4 ordinal 标签 Spearman
    ax1 = fig.add_subplot(121)
    sp = pd.DataFrame(index=FEATURE_COLUMNS, columns=LABEL_COLUMNS, dtype=float)
    for f in FEATURE_COLUMNS:
        for lab in LABEL_COLUMNS:
            rho, _ = stats.spearmanr(df[f].values, df[lab].values)
            sp.loc[f, lab] = rho
    sns.heatmap(sp.astype(float), annot=True, fmt=".2f", cmap="RdBu_r",
                vmin=-0.6, vmax=0.6, cbar=True, ax=ax1)
    ax1.set_title("Spearman 秩相关：特征 x 等级标签")

    # 右：15 特征 Pearson 共线性
    ax2 = fig.add_subplot(122)
    feat_corr = df[FEATURE_COLUMNS].corr(method="pearson")
    sns.heatmap(feat_corr, cmap="coolwarm", vmin=-1, vmax=1,
                square=True, cbar=True, annot=False, ax=ax2)
    ax2.set_title("Pearson 相关：特征 x 特征（共线性）")

    plt.tight_layout()
    fig.savefig(out_dir / "diag_06_feature_label_correlation.png")
    plt.close(fig)
    print("[+] diag_06_feature_label_correlation.png")


# =========================================================================
# 模块七：(munition × task × class) 稀有格子 + 软标签分布
# =========================================================================
def module_07_label_balance(df: pd.DataFrame, out_dir: Path):
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4)

    # 上排：4 个任务 × 4 弹型 × 3 等级 计数热力图（每个任务一张）
    for i, lab in enumerate(LABEL_COLUMNS):
        ax = fig.add_subplot(gs[0, i])
        ct = pd.crosstab(df["munition_id"], df[lab])
        ct = ct.reindex(index=[0, 1, 2, 3], columns=[0, 1, 2], fill_value=0)
        ct.index = [MUN_NAMES[i] for i in ct.index]
        sns.heatmap(ct, annot=True, fmt="d", cmap="YlOrRd",
                    cbar=True, ax=ax)
        ax.set_title(f"计数：{lab} 按弹型分组")
        ax.set_xlabel("等级")
        ax.set_ylabel("弹型")

    # 下排：8 个 *_prob 软标签的边缘 KDE，每行 4 个，每个子图画 2 个 prob
    sub_axes = [fig.add_subplot(gs[1, j]) for j in range(4)]
    pairs = [("K1_prob", "K2_prob"), ("M1_prob", "M2_prob"),
             ("F1_prob", "F2_prob"), ("C1_prob", "C2_prob")]
    palette = sns.color_palette("Set1", n_colors=2)
    for ax, (a_col, b_col) in zip(sub_axes, pairs):
        sns.kdeplot(df[a_col], ax=ax, color=palette[0], lw=1.6,
                    clip=(0, 1), label=a_col)
        sns.kdeplot(df[b_col], ax=ax, color=palette[1], lw=1.6,
                    clip=(0, 1), label=b_col)
        ax.set_xlim(0, 1)
        ax.set_xlabel("概率")
        ax.set_ylabel("密度")
        ax.set_title(f"软标签：{a_col[0]} 类任务")
        ax.legend(fontsize=9)

    fig.suptitle("标签平衡诊断（硬标签 + 软标签）", fontsize=14, y=1.0)
    plt.tight_layout()
    fig.savefig(out_dir / "diag_07_label_balance.png", bbox_inches="tight")
    plt.close(fig)
    print("[+] diag_07_label_balance.png")


# =========================================================================
# 模块八：loss_weight 尾部 / ESS / layer_type 配额 / split 一致性
# =========================================================================
def module_08_weight_split(df: pd.DataFrame, out_dir: Path,
                           split_manifest_path: Path):
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3)

    # (1,1) loss_weight 对数分布 + ESS / 极端尾部
    ax = fig.add_subplot(gs[0, 0])
    w = df["loss_weight"].values.astype(np.float64)
    sns.histplot(np.log10(w), bins=80, color="#762a83", ax=ax)
    ess = (w.sum() ** 2) / (w ** 2).sum()
    ess_ratio = ess / len(w)
    p99 = np.quantile(w, 0.99)
    p999 = np.quantile(w, 0.999)
    txt = (f"ESS = {ess:,.0f}  (占 N 的 {ess_ratio*100:.1f}%)\n"
           f"p99 = {p99:.1f}\n"
           f"p99.9 = {p999:.1f}\n"
           f"最大值 = {w.max():.1f}")
    ax.text(0.98, 0.97, txt, transform=ax.transAxes, va="top", ha="right",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f7", ec="#bbb"))
    ax.set_xlabel("log10(loss_weight)")
    ax.set_ylabel("计数")
    ax.set_title("loss_weight 尾部分布与有效样本数")

    # (1,2) loss_weight × munition × K_level（验证 IPW 真的提升稀有类）
    ax = fig.add_subplot(gs[0, 1])
    plot_df = df[["munition_id", "K_level", "loss_weight"]].copy()
    plot_df["munition_id"] = plot_df["munition_id"].map(MUN_NAMES)
    sns.boxplot(x="munition_id", y="loss_weight", hue="K_level",
                data=plot_df, palette="Set2",
                showfliers=False, ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel("弹型")
    ax.set_ylabel("loss_weight")
    ax.set_title("loss_weight 按弹型与 K_level 分组")

    # (1,3) layer_type × munition_id 配额
    ax = fig.add_subplot(gs[0, 2])
    layer_ct = pd.crosstab(df["layer_type"], df["munition_id"])
    layer_ct.columns = [MUN_NAMES[c] for c in layer_ct.columns]
    sns.heatmap(layer_ct, annot=True, fmt="d", cmap="Blues",
                cbar=True, ax=ax)
    ax.set_xlabel("弹型")
    ax.set_ylabel("layer_type")
    ax.set_title("采样配额：layer_type x 弹型")

    # (2,1) is_crawled 空间散点对照
    ax = fig.add_subplot(gs[1, 0])
    sub = _safe_sample(df, 12000, seed=8)
    sns.scatterplot(x="x_cm", y="z_cm", hue="is_crawled", data=sub,
                    palette={0: "#9999cc", 1: "#e34a33"},
                    s=8, alpha=0.55, edgecolor=None, ax=ax)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("分组泄漏检查：is_crawled 空间叠加 (X-Z)")

    # (2,2) train/val/test 在 munition 上的分布
    ax = fig.add_subplot(gs[1, 1])
    split_idx = None
    if split_manifest_path.exists():
        with open(split_manifest_path, "r", encoding="utf-8") as f:
            mani = json.load(f)
        split_idx = {k: np.asarray(mani[k], dtype=np.int64)
                     for k in ("train_idx", "val_idx", "test_idx")
                     if k in mani}
        rows = []
        for split, idx in split_idx.items():
            sub_df = df.iloc[idx]
            for mid, c in sub_df["munition_id"].value_counts().items():
                split_key = split.replace("_idx", "")
                rows.append({"split": SPLIT_NAMES.get(split_key, split_key),
                             "munition": MUN_NAMES[mid],
                             "count": int(c)})
        plot_df = pd.DataFrame(rows)
        sns.barplot(x="munition", y="count", hue="split",
                    data=plot_df, palette="Set1", ax=ax)
        ax.set_xlabel("弹型")
        ax.set_ylabel("计数")
        ax.legend(title="子集")
        ax.set_title("切分一致性：各子集弹型分布")
    else:
        ax.text(0.5, 0.5, "未找到 split_manifest.json",
                ha="center", va="center")
        ax.set_axis_off()

    # (2,3) train/val/test 在 4 个 ordinal label 上的 class=1/class=2 占比
    ax = fig.add_subplot(gs[1, 2])
    if split_idx is not None:
        rows = []
        for split, idx in split_idx.items():
            sub_df = df.iloc[idx]
            for lab in LABEL_COLUMNS:
                for cls in (1, 2):
                    rate = float((sub_df[lab] == cls).mean()) * 100
                    split_key = split.replace("_idx", "")
                    rows.append({"split": SPLIT_NAMES.get(split_key, split_key),
                                 "label": f"{lab[0]}={cls}",
                                 "rate(%)": rate})
        plot_df = pd.DataFrame(rows)
        sns.barplot(x="label", y="rate(%)", hue="split",
                    data=plot_df, palette="Set1", ax=ax)
        ax.set_xlabel("标签")
        ax.set_ylabel("正例率(%)")
        ax.legend(title="子集")
        ax.set_title("切分一致性：各标签正例率")
    else:
        ax.text(0.5, 0.5, "未找到 split_manifest.json",
                ha="center", va="center")
        ax.set_axis_off()

    plt.tight_layout()
    fig.savefig(out_dir / "diag_08_weight_split.png")
    plt.close(fig)
    print("[+] diag_08_weight_split.png")


# =========================================================================
# 主入口
# =========================================================================
def run_diagnostic_visualization(parquet_path: str = "output/damage_dataset.parquet",
                                 out_dir: str = "vis_outputs_cn"):
    print(f"[*] 加载仿真数据集: {parquet_path}")
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        # pyarrow 与生成端版本不一致时偶发 histogram 报错，回退 fastparquet
        print(f"    pyarrow 读取失败 ({e.__class__.__name__})，回退 fastparquet")
        df = pd.read_parquet(parquet_path, engine="fastparquet")
    print(f"    样本数 = {len(df)},  列数 = {df.shape[1]}")

    # 衍生特征
    df["aerodynamic_aoa"] = reconstruct_aerodynamic_aoa(df)
    df["radial_dist"] = np.sqrt(df["x_cm"] ** 2 + df["y_cm"] ** 2 + df["z_cm"] ** 2)

    # 数据完整性体检
    integrity = integrity_check(df)
    print(f"[*] 完整性体检 NaN={integrity['nan_total']} Inf={integrity['inf_total']}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    module_01_spatial(df, out_path)
    module_02_kinematic(df, out_path)
    module_03_aero_ipw(df, out_path)
    module_04_damage(df, out_path)
    module_05_feature_marginal(df, integrity, out_path)
    module_06_feature_label_corr(df, out_path)
    module_07_label_balance(df, out_path)

    split_manifest_path = Path(parquet_path).with_name("split_manifest.json")
    module_08_weight_split(df, out_path, split_manifest_path)

    # ------------------------------------------------------------------
    # 文本总结面板
    # ------------------------------------------------------------------
    w = df["loss_weight"].values.astype(np.float64)
    ess = (w.sum() ** 2) / (w ** 2).sum()
    print("\n" + "=" * 60)
    print(" 仿真数据集检验诊断简报 ")
    print("=" * 60)
    print(f"  样本总数                         : {len(df):,}")
    print(f"  特征维度                         : {len(FEATURE_COLUMNS)}")
    print(f"  速度值域                         : "
          f"{df['norm_velocity'].min():.1f} ~ {df['norm_velocity'].max():.1f} m/s")
    print(f"  半径值域                         : "
          f"{df['radial_dist'].min():.1f} ~ {df['radial_dist'].max():.1f} cm")
    print(f"  气动攻角值域                     : "
          f"{df['aerodynamic_aoa'].min():.1f} ~ {df['aerodynamic_aoa'].max():.1f} deg")
    print(f"  IPW权重峰值                      : {w.max():.2f}")
    print(f"  IPW有效样本 (ESS / N)            : "
          f"{ess:,.0f} / {len(w):,}  ({ess/len(w)*100:.1f}%)")
    print(f"  穿透次数均值                     : {df['total_penetrations'].mean():.2f}")
    print(f"  灾难性杀伤率                     : "
          f"{(df['K_level'] > 0).mean()*100:.2f}%")
    print(f"  数据完整性 (NaN / Inf)           : "
          f"{integrity['nan_total']} / {integrity['inf_total']}")
    print(f"  输出目录                         : {out_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    run_diagnostic_visualization()
