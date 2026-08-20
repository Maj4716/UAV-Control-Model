import json
import numpy as np
import pandas as pd
from scipy.stats import qmc
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
import os
import hashlib
import platform
import sys

import pyarrow as pa
import pyarrow.parquet as pq

from loitering_munition_damage_twin.paths import PROJECT_ROOT
from loitering_munition_damage_twin.simulation import (
    coordinate_frames as coordinate_frames_module,
)
from loitering_munition_damage_twin.simulation import engine as engine_module
from loitering_munition_damage_twin.simulation.engine import (
    EncounterCondition, DamageEngine,
    create_small_loitering_munition, create_medium_loitering_munition,
    create_medium_rear_det, create_heavy_loitering_munition,
    bundled_resource_path, load_vehicle_model, load_armor_plates,
    parse_component_geometry,
)
from loitering_munition_damage_twin.simulation.coordinate_frames import FRAME_CONVENTION_VERSION
from loitering_munition_damage_twin.stage0.component_supervision import (
    COMPONENT_SUPERVISION_FILENAME,
    COMPONENT_SUPERVISION_PROFILE_FILENAME,
    COMPONENT_TARGET_COLUMNS,
    build_component_supervision_profile,
    component_means_to_columns,
    extract_component_mc_means,
    sha256_file,
    sha256_text_sequence,
)
from loitering_munition_damage_twin.stage0 import (
    component_supervision as component_supervision_module,
)

# ============================================================================
# 配置文件 (Configuration Externalization)
# ============================================================================
CONFIG = {
    # 涉密/实验测算数据：各弹型特定的致损速度经验阈值 V_threshold (m/s)
    # 0: 小型, 1: 中型, 2: 中型后起爆, 3: 大型
    "V_THRESHOLDS": {
        0: 150.0,
        1: 180.0,
        2: 180.0,
        3: 200.0
    },

    # 涉密/实验测算数据：装甲穿深极度敏感的侧倾角临界区 (度)
    # ===== 修正 (基于 K2 实测画像) =====
    # 旧值 [90, -90] 源自"聚能射流正对侧装甲"的先验，但 1994 个 K2 实测样本 |roll|
    # 分位数 [50.9, 120.5, 146.3]，中位数 120° 表明 K2 主要发生在【侧后方斜射】
    # 而非【正侧方】。同时正面斜射 (~60°) 也是次峰。CRITICAL_ROLL_WIDTH 由 ±7°
    # 放宽到 ±15° 以覆盖侧后方的实测主峰带宽。
    "CRITICAL_ROLLS": [120.0, -120.0, 60.0, -60.0],
    "CRITICAL_ROLL_WIDTH": 15.0,  # 极密区宽度 ±15度 (原 ±7)

    # 采样规模宏观控制
    "N_TARGET": 300000,           # 最终生成的数据行数
    # 当前修复目标不是继续放大全局 K2/C2，而是把最终四种弹型总量拉回接近等量。
    # Phase 1 提高到 50%，让四种弹型都有足够大的基础盘，减少最终分布被 Phase 2 劫持。
    "PHASE1_RATIO": 0.5,
    # Phase-1 is the longest single in-memory stage.  Persist completed engine
    # results in atomic Parquet shards so an external shutdown loses at most
    # one shard instead of the entire multi-hour run.  A shard is reusable only
    # when the full input, CONFIG, generator/physics sources and vehicle assets
    # have identical SHA-256 identities.
    "PHASE1_CHECKPOINT_ENABLED": True,
    "PHASE1_CHECKPOINT_DIR": "output/stage0_phase1_checkpoint_v1",
    "PHASE1_CHECKPOINT_INTERVAL": 1000,

    # 物理重要性边界锐度参数 (控制 sigmoid 权重斜率)
    "LAMBDA_SHARPNESS": 10.0,

    # LHS 采样物理域界限
    "RADIUS_MAX_CM": 500.0,       # 最大起始起爆半径
    "V_MIN": 50.0,
    "V_MAX": 300.0,

    # Phase 2 边界爬行自适应扰动参数 (单位: 弧度 Radians)
    "NOISE_SIGMA_MAX": 0.25,       # 距离判定线远时的顶配高斯步长 (约 14.3 度)
    "NOISE_SIGMA_MIN": 0.01,      # 逼近判定边界时的超精细高斯步长 (约 0.57 度)
    "DIST_NORM_VELOCITY": 35.0,   # 归一化距离跨度

    # 修复 #6 / #7 / #8：可复现性 + Phase 2 软攻角拒止 + 物理过滤超采补偿
    "RANDOM_SEED": 42,             # 全局确定性随机种子，保证多次运行结果可复现
    "CRAWL_OVERSAMPLE": 2.0,       # Phase 2 爬行内部超采倍率，对冲物理掩码 + 软拒止的复合丢弃
    "AOA_SOFT_TAIL_DEG": 15.0,     # 软攻角拒止起始角度 (与 Phase 1 同口径)
    "AOA_HARD_FAIL_DEG": 30.0,     # 软攻角拒止硬切角度
    "AOA_SOFT_SIGMA": 5.0,         # 高斯软衰减带宽

    # --- 方案 A: 宽松种子提取阈值 (放宽黄金种子判定以扩大爬行起点池) ---
    # 逻辑: K2_prob=0.3 的点虽不是正样本，但位于判决面近邻，从它出发爬行命中率高
    "SEED_PROB_RELAX": 0.25,       # 宽松种子阈值 (用于选爬行起点)
    "VALID_PROB_STRICT": 0.5,      # 严格入库阈值 (用于最终数据集过滤 valid_K2 / valid_C2)

    # --- 改进版方案 B: 多阶段梯度引导爬行 ---
    "CRAWL_N_STAGES": 5,               # 总爬行阶段数
    "CRAWL_TOPK_PER_STAGE": 50,        # 每阶段从历史池中选 top-k 个种子 (保持多样性，避免坍缩)
    "CRAWL_BOUNDARY_BAND": [0.3, 0.7], # 种子优先区间：判决面边界带，含信息量最大
    "CRAWL_SIGMA_DECAY": 0.75,         # 每阶段步长衰减系数 (exploration → exploitation)
    "CRAWL_NEIGHBOR_DEDUP_R": 5.0,     # 种子去重半径 (无量纲，在标准化特征空间中)
    # Stage-0 v2 diversity controls.  A row quota is not evidence of independent
    # coverage when thousands of descendants share one active-learning root.
    "MAX_ROWS_PER_ROOT": 64,
    "MAX_CHILDREN_PER_ROOT_PER_STAGE": 8,
    "MIN_BOUNDARY_SEED_ROOTS": 64,
    "FRESH_ROOT_CANDIDATE_MULTIPLIER": 8,
    "FRESH_ROOT_BATCH_SIZE": 1024,
    "FRESH_ROOT_MAX_CANDIDATES_PER_TASK": 8192,
    "FRESH_ROOT_MAX_ROUNDS": 8,
    # C2 must support train/validation/test evidence independently.  Its
    # narrow reachable cone therefore receives a larger root-discovery budget
    # than ordinary top-off cells.
    "C2_FRESH_ROOT_MAX_CANDIDATES": 32768,
    "C2_FRESH_ROOT_MAX_ROUNDS": 32,
    # 稀有 root 探测不再先在球内生成约 90% 会落入车辆 AABB 的无效点，
    # 而是在左右侧装甲外侧生成独立起爆点。范围为相对 AABB 的外侧间距，
    # 切向扰动覆盖组件前后/高低方向；只作用于 Phase-2 fresh-root 探测。
    "FRESH_ROOT_LATERAL_CLEARANCE_RANGE_CM": [16.0, 80.0],
    "FRESH_ROOT_TANGENTIAL_JITTER_CM": [180.0, 120.0],
    # C2 is reachable only through a comparatively narrow crew-access
    # corridor.  A broad side shell spends most candidates over the engine
    # bay/roof edges and made the production reachability probe appear
    # negative even though 29/32 high-score legacy roots remain positive at
    # fixed 64-MC revalidation.  These values alter only the Phase-2 proposal:
    # damage physics and the accepted-label threshold remain unchanged.
    "C2_FRESH_RIGHT_SIDE_PROB": 0.75,
    # Prefer the exposed maximum-y crew clusters while retaining 30% uniform
    # exploration over every valid six-person cluster centroid.
    "C2_FRESH_MAX_Y_CLUSTER_PROB": 0.70,
    # Source-y interval is derived from immutable C2 crew-cluster geometry:
    # [minimum cluster y - first margin, median cluster y + second margin].
    "C2_FRESH_CREW_Y_CORRIDOR_MARGIN_CM": [120.0, 30.0],
    # Source height relative to the selected crew-cluster target height.
    "C2_FRESH_TARGET_Z_OFFSET_RANGE_CM": [-50.0, 110.0],
    "TARGET_STRICT_POSITIVE_ROOTS": 32,
    "C2_TARGET_STRICT_POSITIVE_ROOTS": 128,
    "MIN_TRAIN_POSITIVE_ROWS": 128,
    "MIN_TRAIN_POSITIVE_ROOTS": 16,
    "MIN_TRAIN_NEGATIVE_ROOTS": 16,
    "MIN_TRAIN_LEVEL_ROOTS": 16,
    "MIN_TRAIN_EXACT_LEVEL_ROWS": 128,
    "MIN_EVAL_EXACT_LEVEL_ROWS": 100,
    "MIN_EVAL_EXACT_LEVEL_ROOTS": 16,
    # Evaluation evidence is a production-artifact contract.  Keep its
    # activation threshold independent from the general usability gate so
    # reduced deterministic pipeline tests can exercise the remaining gates
    # without pretending that a few hundred rows are a production dataset.
    "EVALUATION_SUPPORT_GATE_MIN_ROWS": 50000,
    "MIN_EFFECTIVE_POSITIVE_ROOTS": 8.0,
    "MAX_DOMINANT_ROOT_SHARE": 0.25,
    "MAX_POSITIVE_ROWS_PER_ROOT_PER_CELL": 8,
    "USABILITY_GATE_MIN_ROWS": 50000,
    "FINAL_TOPUP_MAX_ROUNDS": 10,
    "FINAL_TOPUP_MIN_BATCH": 256,

    # --- 方案 D: 训练侧加权补偿 (不改采样，在 loss_weight 上强化稀有类) ---
    "RARE_CLASS_AMP_CAP": 150.0,       # 稀有类权重放大上限 (原为 50，放大 3x)
    "FOCAL_LOSS_GAMMA": 2.0,           # 下游 NN focal loss 建议的 gamma (仅写入元数据供训练脚本读取)

    # --- 方案 P1: Aim-and-Shoot 瞄准式速度方向采样 ---
    # 取代原均匀球面采样 (大概率"飞向虚空")，先在车体 AABB 内采靶点，
    # 速度方向 = (靶点 - 起爆点) 单位化 + 切空间高斯噪声
    "AIM_SIGMA_DEG": 5.0,              # 瞄准方向的角散布 (战术级精度，约 ±5°)

    # ============================================================
    # 第三轮：Per-Munition Stratified Sampling 弹型独立分层架构
    # ============================================================
    # 上一轮 M_ID_PROBS=[0.05,0.10,0.10,0.75] + K2_HUNT 100% m_id=3 导致
    # 弹型 3 占样本 ~85%，弹型 0/1/2 各仅 3-5%。模型对轻/中型弹的 M/F/C
    # 物理特性几乎学不到。
    #
    # 本轮：把全局 M_ID_PROBS 退役，改为 per-munition stratified sampling。
    # 每种弹型独立分配预算 + 内部按其物理特性设计专属 _HUNT 层。
    # ============================================================

    # 各弹型的总预算占比：这一轮修复中，Phase 1 与最终数据集都按等量目标执行。
    "MUNITION_BUDGET": {
        0: 0.25,
        1: 0.25,
        2: 0.25,
        3: 0.25,
    },

    # 各弹型内部的层段比例 (在该弹型预算之内进一步分层)
    # 设计原则：每种弹型把 ≥75% 兵力投到其"主毁伤模式"对应的 _HUNT 层
    # [P1-A] 引入 K1_HUNT 层 + 补齐各弹型的 class-1 稀缺格子：
    #   * Small 原缺 C_HUNT（全集仅 38 个 Small×C1）和 K1_HUNT（仅 26 个 Small×K1）
    #   * Med-RD 原缺 M_HUNT（仅 460 个 Med-RD×M1）
    #   * Heavy 原缺 M_HUNT（仅 167 个 Heavy×M1，最严重稀缺格）
    # 对应 train 后 Small×K1=0%, Small×C1=0%, Heavy×M1 曾崩至 16.7%
    "PER_MUNITION_LAYERS": {
        # 弹型 3 重型：K2/K1 主力 + 补 M 稀缺
        3: {
            "K2_HUNT":       0.40,         # 原 0.55，腾出空间给 K1_HUNT/M_HUNT
            "K1_K2":         0.15,         # 原 0.25
            "K1_HUNT":       0.15,         # [P1-A] 新增：K 关键组件 + 次临界速度 → 产 K1
            "M_HUNT":        0.15,         # [P1-A] 新增：Heavy×M1 仅 167 样本，补齐
            "CRITICAL_ROLL": 0.10,         # 原 0.15
            "M_F":           0.05,         # 保持
        },
        # 弹型 0 小型 (V门槛 150，小装药，主打履带/外挂火控)
        # [P3-SM] 训练诊断显示 Small×M 仍弱于其他弹型×任务组合，因此在不改变
        # 小型弹总配额的前提下，将其内部预算进一步向走行系/动力系样本倾斜。
        0: {
            "M_HUNT":        0.40,         # Small×M 主正例来源：瞄准传动/发动机/履带/悬挂
            "M_F":           0.20,         # Small 的 M0/M1/M2 分界负样本与近边界样本
            "K1_HUNT":       0.10,
            "F_HUNT":        0.10,
            "C_HUNT":        0.14,
            "K1_K2":         0.06,
        },
        # 弹型 1 中型：M/F/C 主力
        1: {
            "C_HUNT":        0.25,         # 原 0.30
            "F_HUNT":        0.20,         # 原 0.25
            "M_HUNT":        0.15,         # 原 0.20
            "K1_HUNT":       0.15,         # [P1-A] 新增：Med-LM×K1 仅 238 样本
            "K1_K2":         0.15,         # 原 0.15
            "M_F":           0.10,         # 保持
        },
        # 弹型 2 中型后起爆：F/C 主力
        # [P1-A] 新增 M_HUNT（原完全缺）+ K1_HUNT
        2: {
            "C_HUNT":        0.30,         # 原 0.40
            "F_HUNT":        0.25,         # 原 0.30
            "K1_HUNT":       0.15,         # [P1-A] 新增：Med-RD×K1 仅 225 样本
            "K1_K2":         0.15,         # 原 0.20
            "M_F":           0.10,         # 保持
            "M_HUNT":        0.05,         # [P1-A] 新增少量：Med-RD 原无 M_HUNT，M1 仅 460
        },
    },

    # ============================================================
    # 各任务的关键组件 ID + 物理窄带参数
    # (基于 vehicle_model.json 70 组件名称分类扫描)
    # ============================================================

    # K 类靶点必须与 DamageTree 的实际规则一致：K1 只由油箱(id=3)
    # 触发，K2 只由弹药架(id=46)触发。将辅助弹药/供弹装置混入 K2
    # 靶池会把大部分定向预算投向不可能触发 K2 的组件。
    "K1_CRITICAL_COMPONENT_IDS": [3],
    "K2_CRITICAL_COMPONENT_IDS": [46],
    "K2_HUNT_R_RANGE":     [200.0, 280.0],   # K2 实测 r IQR
    "K2_HUNT_V_RANGE":     [220.0, 300.0],   # K2 实测 v IQR
    "K2_HUNT_ROLL_RANGE":  [50.0, 150.0],    # K2 实测 |roll| 范围
    "K2_AIM_BIAS":         0.6,              # 60% 瞄准关键组件，40% AABB
    "K2_AIM_SIGMA_DEG":    8.0,

    # [P1-A] K1_HUNT 层参数：瞄 K 关键组件但速度"刚过穿甲门槛"（不足以引爆弹药/殉爆）
    # 物理机制：K2 需要高速命中+关键组件链式反应；K1 仅需击穿单个 K 组件（油箱引燃
    # 或单弹药架损毁）。降低速度上限让命中时"只击穿不引爆"，从而产 K1 而非 K2。
    # 同时扩宽 R/ROLL 区间以让各弹型都能适配。与 K2_HUNT 共享 K_CRITICAL 组件。
    "K1_HUNT_R_RANGE":     [180.0, 350.0],   # 比 K2 宽（覆盖更远距离的近似击穿）
    "K1_HUNT_V_RANGE":     [150.0, 220.0],   # 刚过 Small V_th (150)，上限 < K2_HUNT 下限 220
    "K1_HUNT_ROLL_RANGE":  [30.0, 160.0],    # 比 K2 roll 略宽（包含前斜方攻击）
    "K1_AIM_BIAS":         0.65,             # 65% 瞄 K 关键组件
    "K1_AIM_SIGMA_DEG":    10.0,             # 稍大散布，让 "刚打穿" 分布更广

    # M 关键组件 (传动/发动机/履带/轮/悬挂 = 38 个)
    # M 击穿门槛低，主要靠击中走行系；速度可低；roll 影响小
    "M_CRITICAL_COMPONENT_IDS": [
        1, 2,                                          # 传动/发动机
        4, 5, 30, 31,                                  # 主动轮/诱导轮
        6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,    # 负重轮 1-12
        18, 19, 20, 21, 22, 23,                        # 托带轮 1-6
        32, 33, 34, 35, 36, 37, 38, 39,                # 履带各段
        24, 25, 26, 27, 28, 29,                        # 悬挂装置
    ],
    "M_HUNT_R_RANGE":      [150.0, 400.0],   # M 任务对距离不敏感，宽放
    "M_HUNT_V_RANGE":      [120.0, 280.0],   # 中速即可，不必高速
    "M_AIM_BIAS":          0.7,              # 70% 瞄准走行系
    "M_AIM_SIGMA_DEG":     6.0,

    # F 关键组件 (火控/主炮/机炮/烟幕/观瞄/通讯 = 13 个，多在车顶)
    "F_CRITICAL_COMPONENT_IDS": [
        40, 41, 42,                # 测距仪 + 主/辅观瞄
        43, 44,                    # 火控计算机
        45,                        # 主炮
        47, 48,                    # 烟幕弹发射器
        49,                        # 供弹装置 (跨 K2/F)
        50, 51, 52,                # 机炮
        55, 56,                    # 通讯天线
    ],
    "F_HUNT_R_RANGE":      [180.0, 380.0],
    "F_HUNT_V_RANGE":      [150.0, 280.0],
    "F_AIM_BIAS":          0.7,
    "F_AIM_SIGMA_DEG":     6.0,

    # C 关键组件 (乘员 3 + 车载步兵 7 = 10 个，多在车体内部)
    "C_CRITICAL_COMPONENT_IDS": [58, 59, 60, 61, 62, 63, 64, 65, 66, 67],
    # C2 规则要求至少 60%（10 人中的 6 人）毁伤。C2_HUNT 不瞄单人，
    # 而瞄每名乘员及其最近 6 人的簇质心，使主破片锥覆盖多人群组。
    "C2_CLUSTER_SIZE": 6,
    "C_HUNT_R_RANGE":      [180.0, 320.0],   # C 任务需要中等距离形成穿透条件
    "C_HUNT_V_RANGE":      [180.0, 290.0],
    "C_AIM_BIAS":          0.7,
    "C_AIM_SIGMA_DEG":     7.0,

    # sim_engine.generate_fragment_field 的 FRONT 起爆主破片沿机体 -X_B
    # 飞散，REAR 起爆沿 +X_B 飞散。因此定向采样时，前端起爆弹的机体轴
    # 必须背向关键组件，后端起爆弹才应朝向关键组件。该符号只改变采样提案，
    # 不改变毁伤物理：+1=速度/机轴朝靶点，-1=速度/机轴背向靶点。
    "HUNT_AXIS_SIGN_BY_MUNITION": {
        0: -1.0,  # Small, FRONT
        1: -1.0,  # Med-LM, FRONT
        2: 1.0,   # Med-RD, REAR
        3: -1.0,  # Heavy, FRONT
    },
    # 已计入约 250--275 m/s 弹体速度与 Gurney 速度矢量合成后的主破片
    # 有效锥角。HUNT 将真实靶点放在锥面而不是锥轴上。
    "HUNT_FRAGMENT_CONE_DEG_BY_MUNITION": {
        0: 35.0,
        1: 34.0,
        2: 26.0,
        3: 30.0,
    },
    "HUNT_AOA_JITTER_DEG": 5.0,

    # Targeted layers may tighten only the sampling proposal; they do not alter
    # the damage physics.  These overrides create new independent roots for the
    # cells that previously collapsed onto a handful of descendants.
    "HUNT_OVERRIDES": {
        0: {
            "C": {"r_range": [170.0, 250.0], "v_range": [240.0, 300.0],
                  "aim_bias": 0.95, "aim_sigma_deg": 3.0},
            "K1": {"r_range": [170.0, 280.0], "v_range": [190.0, 250.0],
                   "aim_bias": 0.90, "aim_sigma_deg": 5.0},
        },
        1: {
            "C": {"r_range": [170.0, 260.0], "v_range": [240.0, 300.0],
                  "aim_bias": 0.95, "aim_sigma_deg": 3.0},
            # Unlike C1, C2 aims at a six-person cluster centroid.  Keep the
            # target on that centroid and reduce proposal jitter; this is an
            # importance proposal, not a change to the simulator.
            "C2": {"r_range": [170.0, 260.0], "v_range": [240.0, 300.0],
                   "aim_bias": 1.00, "aim_sigma_deg": 3.0,
                   "target_jitter_cm": 6.0},
            "K2": {"r_range": [180.0, 260.0], "v_range": [250.0, 300.0],
                   "aim_bias": 0.90, "aim_sigma_deg": 4.0},
        },
        3: {
            "K2": {"r_range": [190.0, 280.0], "v_range": [240.0, 300.0],
                   "aim_bias": 0.85, "aim_sigma_deg": 5.0},
            "C": {"r_range": [170.0, 270.0], "v_range": [230.0, 300.0],
                  "aim_bias": 0.90, "aim_sigma_deg": 4.0},
        },
    },

    # ============================================================
    # m_id 条件化的 K 任务训练权重 (供下游 nn_train.py 消费)
    # [P0-1] 从 {0.10, 0.20, 0.30, 1.00} 抬到 {0.40, 0.50, 0.55, 1.00}
    #   原因：旧值把 Small/Med 弹型上 K 分支的有效等效样本压到 ~92，
    #   被 Heavy 的 35k 样本完全淹没 → Small×K1 recall=0% 的机械成因。
    #   新值让 Small×K 等效样本抬到 ~369，进入"可学"区间。
    # ============================================================
    "M_ID_K_TASK_WEIGHT": {
        0: 0.40,
        1: 0.50,
        2: 0.55,
        3: 1.00,       # 重型：K 任务全权重训练
    },

    # ============================================================
    # [P0-2] m_id 条件化的 C 任务训练权重 (新增，对称于 K_TASK_WEIGHT)
    #   原因：C 分支此前无弹型权重。Small/Med 弹型的 C1 正样本在源头稀疏，
    #   训练信号被 Heavy 淹没 → Small×C1 recall=0%。本配置对 Small/Med 微抬、
    #   对 Heavy 轻微下调 (0.85)。**严禁对 M/F 分支做类似加权** (R14-C 教训)。
    # ============================================================
    "M_ID_C_TASK_WEIGHT": {
        0: 1.25,
        1: 1.05,
        2: 1.05,
        3: 0.90,
    },

    # ============================================================
    # 物理先验比例 (由领域知识给定，用于 Logit Adjustment 推理校准)
    # 这些是"无任何扩展采样下"的预期标签比例 — 可由一个小 baseline 估算
    # 当前数值是基于 P1+P2 阶段（未做 K2_HUNT/_HUNT 扩展）的实测保守估计
    # ============================================================
    "PHYSICAL_PRIOR": {
        "K1_prob": 0.030,
        "K2_prob": 0.015,
        "M1_prob": 0.450,
        "M2_prob": 0.180,
        "F1_prob": 0.150,
        "F2_prob": 0.040,
        "C1_prob": 0.040,
        "C2_prob": 0.012,
    },
    # Generator-side CB weighting is disabled in v2.  The training pipeline
    # already has balanced batches, focal loss and per-cell pos_weight.
    "APPLY_GENERATOR_CB_WEIGHT": False,
    "CB_LOSS_BETA": 0.999,
    "AOA_IPW_CAP": 20.0,
    "LOSS_WEIGHT_MIN": 0.05,
    "LOSS_WEIGHT_MAX": 20.0,
    "MIN_WEIGHT_ESS_RATIO": 0.50,
    # 温度搜索直接约束真正进入梯度更新的 train split，并额外留出 0.5%
    # 安全余量。旧实现只让全表 ESS 恰好达到 50%，实际 train ESS 为 49.743%。
    "WEIGHT_ESS_TARGET_MARGIN": 0.005,
    "WEIGHT_TEMPER_MIN_ALPHA": 0.25,
    "FAMILY_WEIGHT_REFERENCE_SIZE": 8.0,
    # 最终数据集的弹型目标配额。默认四种弹型等量，避免 Heavy 继续依靠样本数碾压。
    "MUNITION_FINAL_TARGET": {
        0: 0.25,
        1: 0.25,
        2: 0.25,
        3: 0.25,
    },
    # Phase 2 从"全局追 K2/C2"改成"按弹型 quota 追各自的高价值任务"。
    "PHASE2_TOP_OFF_PLAN": {
        # [P3-SM/P4-M1] Small×M 定向增强：Small 的补齐阶段维持 85% M 任务，
        # 但将其中一部分从 M2 转向 M1_only，补足一级机动毁伤边界样本。
        # [V5] 再加入 C1_only，专门补足 Small×C1 稀缺格。
        0: {"M2_prob": 0.35, "M1_only": 0.30, "C1_only": 0.25, "F2_prob": 0.05, "K1_prob": 0.05},
        # [P4-M1] 其他弹型少量注入 M1_only，避免 M 分支只在二级强毁伤附近富集。
        # 锥面感知 HUNT 后，Med-LM×C2 在生产候选中出现 29 个相互独立的
        # train 正例 root，旧结构零假设已被推翻。分配 20% C2 top-off，
        # 使其达到生产门禁要求的正例行数与 root 多样性。
        1: {"C2_prob": 0.20, "K2_prob": 0.15, "F2_prob": 0.20,
            "M2_prob": 0.10, "M1_only": 0.15, "K1_prob": 0.20},
        2: {"C2_prob": 0.35, "F2_prob": 0.20, "M2_prob": 0.20, "M1_only": 0.10, "K1_prob": 0.15},
        3: {"K2_prob": 0.25, "M2_prob": 0.25, "M1_only": 0.10, "K1_prob": 0.20, "C2_prob": 0.20},
    },
    # 3% 是 Phase-2 的“停止继续定向补 K2”阈值，而不是最终数据比例硬目标。
    # Phase-1 的物理采样本身可能已经超过 3%；旧字段名把 3% 误解成最终目标，
    # 但 2026-07-24 正式数据实际为 7.19%。最终另设 8% 安全上限，避免 K2
    # 继续吞噬其他等级覆盖，同时保留稀有致死级学习信号。
    "K2_PHASE2_STOP_RATIO": 0.03,
    "K2_FINAL_MAX_RATIO": 0.08,
    # Cells with no positive observation in the 300k audit are treated as
    # configured structural-zero hypotheses.  Domain users should revise this
    # matrix if higher-fidelity evidence shows that the outcome is reachable.
    "ORDINAL_APPLICABILITY": {
        0: {"K": [True, False], "M": [True, True], "F": [True, True], "C": [True, False]},
        # Med-LM C>=2 已由 29 个独立 train root 证明在当前毁伤引擎中可达。
        1: {"K": [True, True],  "M": [True, True], "F": [True, True], "C": [True, True]},
        2: {"K": [True, True],  "M": [True, True], "F": [True, True], "C": [True, True]},
        3: {"K": [True, True],  "M": [True, True], "F": [True, True], "C": [True, True]},
    },
    "GENERATION_PROFILE_SCHEMA": "stage0_lineage_v2",

    # Stage-0: every sample carries immutable lineage and frame metadata.
    # Repeated fragment realizations estimate aleatoric label variation.
    "FRAME_CONVENTION_VERSION": FRAME_CONVENTION_VERSION,
    "DATASET_SCHEMA": "stage0_lineage_v2",
    # A 3/9-replicate estimate leaves a material amount of sample-ID-specific
    # sparse-fragment noise in hard ordinal labels. A validation-only r64
    # replay found 5.66% disagreement with the legacy labels. More
    # importantly, the same enriched rows still changed hard level in 1.66%
    # of cells between the first 32 and all 64 realizations (up to 3.91% in
    # the weakest cell). Start at eight realizations so a zero sample variance
    # cannot be declared stable after only three coincident fields, and allow
    # unresolved boundary cases to continue to 64.
    "LABEL_MC_MIN_REPLICATES": 8,
    "LABEL_MC_MAX_REPLICATES": 64,
    "LABEL_MC_BOUNDARY_HALF_WIDTH": 0.15,
    "LABEL_MC_STD_TRIGGER": 0.20,
    "LABEL_MC_CONFIDENCE_Z": 1.96,
    "LABEL_MC_STANDARD_ERROR_TARGET": 0.02,
    "LABEL_MC_DECISION_MARGIN": 0.02,
    # Pair every random fragment field with its sign-reflected counterpart.
    # The validation-only r32/r64 audit improved hard-level agreement from
    # 98.34% to 98.54% and reduced probability MAE from 0.00427 to 0.00372
    # without changing the simulator expectation.
    "LABEL_MC_ANTITHETIC": True,
    # Backwards-compatible alias used only by old callers and smoke overrides.
    "LABEL_MC_REPLICATES": 3,
    "PARQUET_ROW_GROUP_SIZE": 50000,
    "REFERENCE_SPLIT_RATIOS": {"train": 0.80, "val": 0.10, "test": 0.10},
}

# ============================================================================
# 环境与几何辅助函数
# ============================================================================

def wrap_angle_rad(angle_rad: np.ndarray) -> np.ndarray:
    """处理弧度周期的 wrap-around (-pi 到 pi) 避免物理流形撕裂"""
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi

def wrap_angle_deg(angle_deg: np.ndarray) -> np.ndarray:
    """处理角度周期的 wrap-around (-180 到 180)"""
    return (angle_deg + 180.0) % 360.0 - 180.0


def _stable_uint32(value: str) -> int:
    """Process- and PYTHONHASHSEED-independent 32-bit identifier hash."""
    digest = hashlib.blake2s(value.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _label_mc_rng_pair(
        sample_id: str, replicate_id: int) -> Tuple[int, float]:
    """Return the deterministic seed and antithetic spread sign."""
    antithetic = bool(
        CONFIG.get("LABEL_MC_ANTITHETIC", False))
    random_index = (
        int(replicate_id) // 2
        if antithetic else int(replicate_id))
    spread_sign = (
        1.0
        if not antithetic or int(replicate_id) % 2 == 0
        else -1.0)
    rng_seed = _stable_uint32(
        f"{CONFIG['RANDOM_SEED']}|{sample_id}|{random_index}")
    return rng_seed, spread_sign


def _reference_split_for_root(root_seed_id: str) -> str:
    """Assign an immutable split before active descendants are generated."""
    u = _stable_uint32(root_seed_id) / float(2**32)
    train_ratio = float(CONFIG["REFERENCE_SPLIT_RATIOS"]["train"])
    val_ratio = float(CONFIG["REFERENCE_SPLIT_RATIOS"]["val"])
    if u < train_ratio:
        return "train"
    if u < train_ratio + val_ratio:
        return "val"
    return "test"

def get_vehicle_aabb(components, plates):
    """从解析后的真实部件几何计算整车 AABB。"""
    min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
    max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')

    for c in components:
        _, geometry = parse_component_geometry(c)
        aabb_min = geometry.aabb_min
        aabb_max = geometry.aabb_max
        min_x, min_y, min_z = min(min_x, aabb_min[0]), min(min_y, aabb_min[1]), min(min_z, aabb_min[2])
        max_x, max_y, max_z = max(max_x, aabb_max[0]), max(max_y, aabb_max[1]), max(max_z, aabb_max[2])

    for p in plates:
        min_x, min_y, min_z = min(min_x, p.aabb_min[0]), min(min_y, p.aabb_min[1]), min(min_z, p.aabb_min[2])
        max_x, max_y, max_z = max(max_x, p.aabb_max[0]), max(max_y, p.aabb_max[1]), max(max_z, p.aabb_max[2])

    return np.array([min_x, min_y, min_z]), np.array([max_x, max_y, max_z])


def _allocate_counts(total: int, ratios: Dict[Any, float]) -> Dict[Any, int]:
    """按给定权重分配整数配额，最后一个键吸收舍入误差。"""
    if total <= 0:
        return {k: 0 for k in ratios}

    items = list(ratios.items())
    alloc = {}
    running = 0
    for idx, (key, ratio) in enumerate(items):
        if idx == len(items) - 1:
            count = total - running
        else:
            count = int(total * ratio)
            running += count
        alloc[key] = max(0, int(count))
    return alloc


def _format_munition_count_dict(counts: Dict[int, int]) -> str:
    labels = {0: "Small", 1: "Med-LM", 2: "Med-RD", 3: "Heavy"}
    return " | ".join(f"{labels[m]}={counts.get(m, 0)}" for m in range(4))


def _cap_dataframe_by_munition(df: pd.DataFrame,
                               quota_map: Dict[int, int],
                               seed: int,
                               keep_phase2_first: bool = True) -> Tuple[pd.DataFrame, Dict[int, Dict[str, int]]]:
    """按弹型配额裁剪，优先保留 Phase 2 稀有样本。"""
    mun_col = "munition_id" if "munition_id" in df.columns else "m_id"
    kept_frames = []
    stats = {}

    for m_id, quota in quota_map.items():
        bucket = df[df[mun_col] == m_id].copy()
        if bucket.empty or quota <= 0:
            stats[m_id] = {"available": int(len(bucket)), "kept": 0, "trimmed": int(len(bucket))}
            continue

        if len(bucket) <= quota:
            keep = bucket
        elif keep_phase2_first and "is_crawled" in bucket.columns:
            if "split_role" in bucket.columns:
                reference = bucket[bucket["split_role"] != "train"].copy()
                train_bucket = bucket[bucket["split_role"] == "train"].copy()
            else:
                reference = bucket.iloc[0:0].copy()
                train_bucket = bucket
            phase2 = train_bucket[train_bucket["is_crawled"] == 1].copy()
            phase1 = train_bucket[train_bucket["is_crawled"] != 1].copy()
            remaining = max(quota - len(reference), 0)
            if remaining == 0:
                keep = reference.sample(n=quota, replace=False, random_state=seed)
            elif len(phase2) >= remaining:
                phase2 = phase2.sample(n=remaining, replace=False, random_state=seed)
                keep = pd.concat([reference, phase2], ignore_index=True)
            else:
                need_phase1 = remaining - len(phase2)
                if len(phase1) > need_phase1:
                    phase1 = phase1.sample(n=need_phase1, replace=False, random_state=seed)
                keep = pd.concat([reference, phase2, phase1], ignore_index=True)
        else:
            keep = bucket.sample(n=quota, replace=False, random_state=seed)

        kept_frames.append(keep)
        stats[m_id] = {
            "available": int(len(bucket)),
            "kept": int(len(keep)),
            "trimmed": int(len(bucket) - len(keep)),
        }

    if not kept_frames:
        return pd.DataFrame(), stats
    return pd.concat(kept_frames, ignore_index=True), stats


def _cap_root_families(df: pd.DataFrame, max_rows_per_root: int,
                       seed: int) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Cap correlated descendants while preferentially retaining the root row."""
    if df.empty or "root_seed_id" not in df.columns or max_rows_per_root <= 0:
        return df.copy(), {"families_capped": 0, "rows_removed": 0}

    kept = []
    families_capped = 0
    rows_removed = 0
    for root_id, family in df.groupby("root_seed_id", sort=False, dropna=False):
        if len(family) <= max_rows_per_root:
            kept.append(family)
            continue
        families_capped += 1
        rows_removed += len(family) - max_rows_per_root
        root_rows = family[
            (family.get("parent_id", pd.Series("", index=family.index)).fillna("").astype(str) == "") |
            (family.get("crawl_stage", pd.Series(1, index=family.index)).fillna(1).astype(int) == 0)
        ]
        if root_rows.empty:
            head = family.sample(
                n=1, random_state=_stable_uint32(f"{seed}|{root_id}|head"))
        else:
            head = root_rows.iloc[[0]]
        remainder = family.drop(index=head.index)
        tail_n = max_rows_per_root - len(head)
        tail = remainder.sample(
            n=tail_n, replace=False,
            random_state=_stable_uint32(f"{seed}|{root_id}|tail"))
        kept.append(pd.concat([head, tail], axis=0))

    result = pd.concat(kept, ignore_index=True) if kept else df.iloc[0:0].copy()
    return result, {
        "families_capped": int(families_capped),
        "rows_removed": int(rows_removed),
    }


def _take_with_root_capacity(existing: pd.DataFrame, candidates: pd.DataFrame,
                             max_rows_per_root: int, seed: int) -> pd.DataFrame:
    """Keep only candidate rows that fit the remaining capacity of each root."""
    if candidates.empty or "root_seed_id" not in candidates.columns:
        return candidates.copy()
    existing_counts = (
        existing["root_seed_id"].astype(str).value_counts().to_dict()
        if not existing.empty and "root_seed_id" in existing.columns else {}
    )
    kept = []
    for root_id, family in candidates.groupby("root_seed_id", sort=False):
        capacity = max_rows_per_root - int(existing_counts.get(str(root_id), 0))
        if capacity <= 0:
            continue
        if len(family) > capacity:
            family = family.sample(
                n=capacity, replace=False,
                random_state=_stable_uint32(f"{seed}|{root_id}|capacity"))
        kept.append(family)
    return pd.concat(kept, ignore_index=True) if kept else candidates.iloc[0:0].copy()


def _take_target_rows_with_capacity(existing: pd.DataFrame,
                                    candidates: pd.DataFrame,
                                    target_col: str,
                                    valid_th: float,
                                    max_rows_per_root: int,
                                    max_positive_rows_per_root: int,
                                    seed: int) -> pd.DataFrame:
    """Cap both the whole lineage family and one target-positive family.

    A global 64-row family cap is too weak for a rare cell containing only a
    few hundred positives.  This function applies a second, task-local cap so
    one successful root cannot dominate that cell.
    """
    if candidates.empty:
        return candidates.copy()
    candidates = candidates[_target_valid_mask(candidates, target_col, valid_th)].copy()
    if candidates.empty:
        return candidates

    existing_root_counts = (
        existing["root_seed_id"].astype(str).value_counts().to_dict()
        if not existing.empty else {}
    )
    if not existing.empty:
        existing_positive = existing[
            _target_valid_mask(existing, target_col, valid_th)
        ]
        existing_positive_counts = (
            existing_positive["root_seed_id"].astype(str).value_counts().to_dict()
        )
    else:
        existing_positive_counts = {}

    kept = []
    for root_id, family in candidates.groupby("root_seed_id", sort=False):
        root_key = str(root_id)
        global_capacity = max_rows_per_root - int(existing_root_counts.get(root_key, 0))
        positive_capacity = (
            max_positive_rows_per_root -
            int(existing_positive_counts.get(root_key, 0))
        )
        capacity = min(global_capacity, positive_capacity)
        if capacity <= 0:
            continue
        if len(family) > capacity:
            family = family.sample(
                n=capacity, replace=False,
                random_state=_stable_uint32(
                    f"{seed}|{target_col}|{root_key}|positive-cap"),
            )
        kept.append(family)
    return pd.concat(kept, ignore_index=True) if kept else candidates.iloc[0:0].copy()


def _cap_all_ordinal_positive_families(df: pd.DataFrame,
                                       valid_th: float,
                                       max_rows_per_root_per_cell: int,
                                       seed: int) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Apply the task-local family cap to incidental positives as well.

    A row accepted for M2 can incidentally be C1-positive.  Limiting only the
    task that admitted the row would therefore still allow C1 family collapse.
    This final Phase-2 pass caps every ordinal positive cell before independent
    Phase-1 top-up restores the exact munition quota.
    """
    result = df.copy()
    removed_by_cell = {}
    for m_id in range(4):
        for dimension in ("K", "M", "F", "C"):
            for level in (1, 2):
                probability_col = f"{dimension}_ge{level}_prob"
                if probability_col not in result.columns:
                    continue
                cell_mask = (
                    (result["munition_id"].astype(int) == m_id) &
                    (result[probability_col] >= valid_th)
                )
                cell = result[cell_mask]
                drop_indices = []
                for root_id, family in cell.groupby("root_seed_id", sort=False):
                    if len(family) <= max_rows_per_root_per_cell:
                        continue
                    root_rows = family[
                        family.get("crawl_stage", pd.Series(
                            1, index=family.index)).fillna(1).astype(int) == 0
                    ]
                    keep_parts = []
                    if not root_rows.empty:
                        keep_parts.append(root_rows.iloc[[0]])
                    remaining_capacity = max_rows_per_root_per_cell - len(keep_parts)
                    remainder = family.drop(
                        index=pd.concat(keep_parts).index if keep_parts else [])
                    if remaining_capacity > 0:
                        # Keep both boundary-informative and strong-positive rows.
                        ranked = remainder.assign(
                            _boundary_distance=np.abs(
                                remainder[probability_col].astype(float) - 0.5)
                        ).sort_values(
                            ["_boundary_distance", probability_col],
                            ascending=[True, False],
                        )
                        keep_parts.append(ranked.head(remaining_capacity).drop(
                            columns=["_boundary_distance"]))
                    keep_index = (
                        pd.concat(keep_parts).index
                        if keep_parts else pd.Index([], dtype=family.index.dtype)
                    )
                    drop_indices.extend(family.index.difference(keep_index).tolist())
                if drop_indices:
                    cell_name = f"m_id={m_id}:{dimension}>={level}"
                    removed_by_cell[cell_name] = int(len(drop_indices))
                    result = result.drop(index=drop_indices)
    return result.reset_index(drop=True), removed_by_cell


def _assign_sampling_weight_components(df: pd.DataFrame,
                                       accept_prob: np.ndarray) -> pd.DataFrame:
    """Store interpretable proposal weights instead of one opaque multiplier."""
    if df.empty:
        return df
    accept_prob = np.asarray(accept_prob, dtype=float)
    if len(accept_prob) != len(df):
        raise ValueError("accept_prob 长度与样本数不一致。")
    accept_prob = np.clip(accept_prob, 1e-12, 1.0)
    aoa_ipw = np.minimum(1.0 / accept_prob, float(CONFIG["AOA_IPW_CAP"]))

    v_norm = np.sqrt(df["vx"].values**2 + df["vy"].values**2 + df["vz"].values**2)
    v_th = np.asarray([CONFIG["V_THRESHOLDS"][int(m)] for m in df["m_id"]], dtype=float)
    physics_weight = 1.0 / (
        1.0 + np.exp(-float(CONFIG["LAMBDA_SHARPNESS"]) * (v_norm / v_th - 1.0))
    )
    in_critical_roll = np.zeros(len(df), dtype=bool)
    for critical_roll in CONFIG["CRITICAL_ROLLS"]:
        in_critical_roll |= (
            np.abs(wrap_angle_deg(df["roll"].values - critical_roll)) <=
            float(CONFIG["CRITICAL_ROLL_WIDTH"])
        )
    physics_weight[~in_critical_roll] *= 0.5

    df["aoa_accept_prob"] = accept_prob
    df["aoa_ipw"] = aoa_ipw
    df["physics_weight"] = physics_weight
    # No defensible density ratio is currently available for the targeted
    # active proposal.  Keep it explicit and neutral rather than pretending the
    # inherited parent weight corrects the child proposal.
    df["active_sampling_weight"] = 1.0
    df["family_weight"] = 1.0
    df["class_balance_weight"] = 1.0
    df["loss_weight"] = np.clip(
        aoa_ipw * physics_weight,
        float(CONFIG["LOSS_WEIGHT_MIN"]),
        float(CONFIG["LOSS_WEIGHT_MAX"]),
    )
    return df


def _task_positive_counts(df: pd.DataFrame, valid_th: float) -> Dict[str, int]:
    counts = {}
    for col in ["K1_prob", "K2_prob", "M1_prob", "M2_prob", "F1_prob", "F2_prob", "C1_prob", "C2_prob"]:
        counts[col] = int((df[col] >= valid_th).sum()) if col in df.columns else 0
    if "M1_prob" in df.columns and "M2_prob" in df.columns:
        counts["M1_only"] = int(((df["M1_prob"] >= valid_th) & (df["M2_prob"] < valid_th)).sum())
    if "C1_prob" in df.columns and "C2_prob" in df.columns:
        counts["C1_only"] = int(((df["C1_prob"] >= valid_th) & (df["C2_prob"] < valid_th)).sum())
    return counts


def _target_score(df: pd.DataFrame, target_col: str) -> np.ndarray:
    """Phase 2 目标分数。

    普通目标直接使用 *_prob；M1_only/C1_only 使用 p1*(1-p2)，
    以偏向“一级毁伤成立、二级毁伤尚未成立”的样本。
    """
    if target_col == "M1_only":
        m1 = df["M1_prob"].values
        m2 = df["M2_prob"].values
        return np.clip(m1 * (1.0 - m2), 0.0, 1.0)
    if target_col == "C1_only":
        c1 = df["C1_prob"].values
        c2 = df["C2_prob"].values
        return np.clip(c1 * (1.0 - c2), 0.0, 1.0)
    return df[target_col].values


def _target_seed_mask(df: pd.DataFrame, target_col: str, seed_th: float, valid_th: float) -> np.ndarray:
    if target_col == "M1_only":
        return ((df["M1_prob"].values >= seed_th) &
                (df["M2_prob"].values < valid_th))
    if target_col == "C1_only":
        return ((df["C1_prob"].values >= seed_th) &
                (df["C2_prob"].values < valid_th))
    return df[target_col].values >= seed_th


def _target_valid_mask(df: pd.DataFrame, target_col: str, valid_th: float) -> np.ndarray:
    if target_col == "M1_only":
        return ((df["M1_prob"].values >= valid_th) &
                (df["M2_prob"].values < valid_th))
    if target_col == "C1_only":
        return ((df["C1_prob"].values >= valid_th) &
                (df["C2_prob"].values < valid_th))
    return df[target_col].values >= valid_th


def _target_applicability_cell(target_col: str) -> Tuple[str, int]:
    if target_col.startswith("K2"):
        return "K", 2
    if target_col.startswith("K"):
        return "K", 1
    if target_col.startswith("M2"):
        return "M", 2
    if target_col.startswith("M"):
        return "M", 1
    if target_col.startswith("F2"):
        return "F", 2
    if target_col.startswith("F"):
        return "F", 1
    if target_col.startswith("C2"):
        return "C", 2
    if target_col.startswith("C"):
        return "C", 1
    raise ValueError(f"未知 top-off 目标: {target_col}")


def _validate_generation_config() -> None:
    """Fail before simulation when quotas and reachability policy conflict."""
    errors = []
    if bool(CONFIG.get("PHASE1_CHECKPOINT_ENABLED", False)):
        checkpoint_dir = str(CONFIG.get(
            "PHASE1_CHECKPOINT_DIR", "")).strip()
        if not checkpoint_dir:
            errors.append("PHASE1_CHECKPOINT_DIR 不能为空")
        if int(CONFIG.get("PHASE1_CHECKPOINT_INTERVAL", 0)) <= 0:
            errors.append("PHASE1_CHECKPOINT_INTERVAL 必须为正整数")
    for name in ("MUNITION_BUDGET", "MUNITION_FINAL_TARGET"):
        values = CONFIG[name]
        if set(values) != {0, 1, 2, 3}:
            errors.append(f"{name} 必须包含且仅包含 m_id=0..3")
        if not np.isclose(sum(float(v) for v in values.values()), 1.0, atol=1e-9):
            errors.append(f"{name} 比例之和必须为 1")

    for m_id in range(4):
        plan = CONFIG["PHASE2_TOP_OFF_PLAN"].get(m_id, {})
        if not np.isclose(sum(float(v) for v in plan.values()), 1.0, atol=1e-9):
            errors.append(f"m_id={m_id} PHASE2_TOP_OFF_PLAN 比例之和必须为 1")
        for target_col, ratio in plan.items():
            if float(ratio) <= 0:
                errors.append(f"m_id={m_id} {target_col} 的 top-off 比例必须为正")
                continue
            dimension, level = _target_applicability_cell(target_col)
            if not bool(CONFIG["ORDINAL_APPLICABILITY"][m_id][dimension][level - 1]):
                errors.append(
                    f"m_id={m_id} {target_col} 指向结构零 {dimension}>={level}")

    if int(CONFIG["MAX_POSITIVE_ROWS_PER_ROOT_PER_CELL"]) <= 0:
        errors.append("MAX_POSITIVE_ROWS_PER_ROOT_PER_CELL 必须为正")
    if (int(CONFIG["MAX_POSITIVE_ROWS_PER_ROOT_PER_CELL"]) >
            int(CONFIG["MAX_ROWS_PER_ROOT"])):
        errors.append("任务正例 root 上限不能大于全局 root 上限")
    if not 0 < float(CONFIG["MIN_WEIGHT_ESS_RATIO"]) <= 1:
        errors.append("MIN_WEIGHT_ESS_RATIO 必须位于 (0,1]")
    ess_margin = float(CONFIG.get("WEIGHT_ESS_TARGET_MARGIN", 0.0))
    if not 0 <= ess_margin <= 1.0 - float(CONFIG["MIN_WEIGHT_ESS_RATIO"]):
        errors.append(
            "WEIGHT_ESS_TARGET_MARGIN 必须位于 "
            "[0, 1-MIN_WEIGHT_ESS_RATIO]")
    k2_phase2_stop = float(CONFIG["K2_PHASE2_STOP_RATIO"])
    k2_final_max = float(CONFIG["K2_FINAL_MAX_RATIO"])
    if not 0.0 < k2_phase2_stop <= k2_final_max < 1.0:
        errors.append(
            "K2 比例合同必须满足 "
            "0 < K2_PHASE2_STOP_RATIO <= K2_FINAL_MAX_RATIO < 1")
    clearance = CONFIG.get("FRESH_ROOT_LATERAL_CLEARANCE_RANGE_CM", [])
    tangent = CONFIG.get("FRESH_ROOT_TANGENTIAL_JITTER_CM", [])
    if (len(clearance) != 2 or float(clearance[0]) <= 15.0 or
            float(clearance[1]) <= float(clearance[0])):
        errors.append("FRESH_ROOT_LATERAL_CLEARANCE_RANGE_CM 必须为递增且下界 > 15 cm")
    if len(tangent) != 2 or min(float(v) for v in tangent) <= 0:
        errors.append("FRESH_ROOT_TANGENTIAL_JITTER_CM 必须包含两个正值")
    c2_right_probability = float(CONFIG.get(
        "C2_FRESH_RIGHT_SIDE_PROB", -1.0))
    if not 0.0 < c2_right_probability < 1.0:
        errors.append("C2_FRESH_RIGHT_SIDE_PROB 必须位于 (0,1)")
    c2_max_y_probability = float(CONFIG.get(
        "C2_FRESH_MAX_Y_CLUSTER_PROB", -1.0))
    if not 0.0 <= c2_max_y_probability <= 1.0:
        errors.append("C2_FRESH_MAX_Y_CLUSTER_PROB 必须位于 [0,1]")
    c2_y_margin = CONFIG.get(
        "C2_FRESH_CREW_Y_CORRIDOR_MARGIN_CM", [])
    if (
        len(c2_y_margin) != 2
        or min(float(value) for value in c2_y_margin) < 0.0
    ):
        errors.append(
            "C2_FRESH_CREW_Y_CORRIDOR_MARGIN_CM 必须包含两个非负值")
    c2_z_offset = CONFIG.get(
        "C2_FRESH_TARGET_Z_OFFSET_RANGE_CM", [])
    if (
        len(c2_z_offset) != 2
        or float(c2_z_offset[1]) <= float(c2_z_offset[0])
    ):
        errors.append(
            "C2_FRESH_TARGET_Z_OFFSET_RANGE_CM 必须为递增二元区间")
    axis_signs = CONFIG.get("HUNT_AXIS_SIGN_BY_MUNITION", {})
    if set(axis_signs) != set(range(4)):
        errors.append("HUNT_AXIS_SIGN_BY_MUNITION 必须覆盖 m_id=0..3")
    for m_id, sign in axis_signs.items():
        if float(sign) not in (-1.0, 1.0):
            errors.append(
                f"HUNT_AXIS_SIGN_BY_MUNITION[{m_id}] 必须为 -1 或 +1")
    cone_angles = CONFIG.get("HUNT_FRAGMENT_CONE_DEG_BY_MUNITION", {})
    if set(cone_angles) != set(range(4)):
        errors.append("HUNT_FRAGMENT_CONE_DEG_BY_MUNITION 必须覆盖 m_id=0..3")
    for m_id, angle in cone_angles.items():
        if not 0.0 < float(angle) < 90.0:
            errors.append(
                f"HUNT_FRAGMENT_CONE_DEG_BY_MUNITION[{m_id}] 必须位于 (0,90)")
    if not 0.0 <= float(CONFIG.get("HUNT_AOA_JITTER_DEG", -1.0)) <= 15.0:
        errors.append("HUNT_AOA_JITTER_DEG 必须位于 [0,15]")
    c2_cluster_size = int(CONFIG.get("C2_CLUSTER_SIZE", 0))
    crew_count = len(CONFIG.get("C_CRITICAL_COMPONENT_IDS", []))
    if c2_cluster_size < 2 or c2_cluster_size > crew_count:
        errors.append(
            "C2_CLUSTER_SIZE 必须位于 [2, C_CRITICAL_COMPONENT_IDS 数量]")
    if (
        int(CONFIG.get("C2_TARGET_STRICT_POSITIVE_ROOTS", 0))
        < 3 * int(CONFIG["MIN_EVAL_EXACT_LEVEL_ROOTS"])
    ):
        errors.append(
            "C2_TARGET_STRICT_POSITIVE_ROOTS 必须至少覆盖三个 split "
            "的独立 root 证据预算")
    if (
        int(CONFIG.get("C2_FRESH_ROOT_MAX_CANDIDATES", 0))
        < int(CONFIG["FRESH_ROOT_MAX_CANDIDATES_PER_TASK"])
    ):
        errors.append(
            "C2_FRESH_ROOT_MAX_CANDIDATES 不能小于普通 fresh-root 预算")
    if (
        int(CONFIG["MIN_TRAIN_EXACT_LEVEL_ROWS"])
        < int(CONFIG["MIN_TRAIN_POSITIVE_ROWS"])
    ):
        errors.append(
            "MIN_TRAIN_EXACT_LEVEL_ROWS 不能低于训练正例行门槛")
    if (
        int(CONFIG["MIN_EVAL_EXACT_LEVEL_ROWS"]) < 100
        or int(CONFIG["MIN_EVAL_EXACT_LEVEL_ROOTS"]) < 16
    ):
        errors.append(
            "评估集每个适用精确等级至少需要 100 行和 16 个 root")
    if int(CONFIG.get("EVALUATION_SUPPORT_GATE_MIN_ROWS", 0)) <= 0:
        errors.append(
            "EVALUATION_SUPPORT_GATE_MIN_ROWS 必须为正整数")
    mc_minimum = int(CONFIG.get("LABEL_MC_MIN_REPLICATES", 0))
    mc_maximum = int(CONFIG.get("LABEL_MC_MAX_REPLICATES", 0))
    if mc_minimum < 4 or mc_maximum < mc_minimum:
        errors.append(
            "标签MC次数必须满足 4 <= MIN <= MAX")
    if bool(CONFIG.get("LABEL_MC_ANTITHETIC", False)) and (
        mc_minimum % 2 != 0 or mc_maximum % 2 != 0
    ):
        errors.append(
            "启用 LABEL_MC_ANTITHETIC 时 MIN/MAX 必须为偶数")
    mc_z = float(CONFIG.get("LABEL_MC_CONFIDENCE_Z", 0.0))
    mc_se_target = float(
        CONFIG.get("LABEL_MC_STANDARD_ERROR_TARGET", 0.0))
    mc_decision_margin = float(
        CONFIG.get("LABEL_MC_DECISION_MARGIN", -1.0))
    if mc_z <= 0.0:
        errors.append("LABEL_MC_CONFIDENCE_Z 必须为正")
    if not 0.0 < mc_se_target < 0.5:
        errors.append(
            "LABEL_MC_STANDARD_ERROR_TARGET 必须位于 (0,0.5)")
    if not 0.0 <= mc_decision_margin < 0.5:
        errors.append(
            "LABEL_MC_DECISION_MARGIN 必须位于 [0,0.5)")

    if errors:
        raise RuntimeError("生成配置一致性检查失败: " + "; ".join(errors))


def _build_generation_profile(final_df: pd.DataFrame,
                              final_quota: Dict[int, int],
                              phase1_kept_counts: Dict[int, int],
                              phase2_task_counts: Dict[int, Dict[str, int]],
                              seed_th: float,
                              valid_th: float,
                              target_total: int,
                              phase1_ratio: float,
                              phase2_discovery_stats: Dict[int, Dict[str, Any]] = None,
                              phase2_cell_cap_stats: Dict[int, Dict[str, Any]] = None) -> Dict[str, Any]:
    def _family_stats(rows: pd.DataFrame) -> Dict[str, Any]:
        if rows.empty:
            return {
                "rows": 0, "root_families": 0, "effective_root_families": 0.0,
                "largest_root_share": 0.0,
            }
        counts = rows["root_seed_id"].astype(str).value_counts().to_numpy(dtype=float)
        return {
            "rows": int(len(rows)),
            "root_families": int(len(counts)),
            "effective_root_families": float(counts.sum() ** 2 / np.square(counts).sum()),
            "largest_root_share": float(counts.max() / counts.sum()),
        }

    def _append_family_failures(stats: Dict[str, Any], cell_name: str,
                                minimum_roots: int) -> None:
        if stats["root_families"] < minimum_roots:
            usability_failures.append(
                f"{cell_name} 独立 root={stats['root_families']} < {minimum_roots}")
        if stats["effective_root_families"] < float(CONFIG["MIN_EFFECTIVE_POSITIVE_ROOTS"]):
            usability_failures.append(
                f"{cell_name} 有效 root={stats['effective_root_families']:.2f} < "
                f"{CONFIG['MIN_EFFECTIVE_POSITIVE_ROOTS']}")
        if stats["largest_root_share"] > float(CONFIG["MAX_DOMINANT_ROOT_SHARE"]):
            usability_failures.append(
                f"{cell_name} 最大 root 占比={stats['largest_root_share']:.3f} > "
                f"{CONFIG['MAX_DOMINANT_ROOT_SHARE']}")

    per_munition = {}
    cell_diversity = {}
    exact_level_diversity = {}
    evaluation_level_support = {}
    usability_failures = []
    evaluation_support_enforced = (
        int(len(final_df))
        >= int(CONFIG["EVALUATION_SUPPORT_GATE_MIN_ROWS"])
    )
    for m_id in range(4):
        bucket = final_df[final_df["munition_id"] == m_id]
        phase2_count = int(bucket["is_crawled"].sum()) if "is_crawled" in bucket.columns else 0
        phase1_count = int(len(bucket) - phase2_count)
        per_munition[str(m_id)] = {
            "target_quota": int(final_quota.get(m_id, 0)),
            "final_count": int(len(bucket)),
            "phase1_count": phase1_count,
            "phase2_count": phase2_count,
            "positive_counts": _task_positive_counts(bucket, valid_th),
            "phase2_task_additions": {k: int(v) for k, v in phase2_task_counts.get(m_id, {}).items()},
        }
        cell_diversity[str(m_id)] = {}
        exact_level_diversity[str(m_id)] = {}
        evaluation_level_support[str(m_id)] = {}
        train_bucket = bucket[bucket["split_role"] == "train"]
        for dimension in ("K", "M", "F", "C"):
            cell_diversity[str(m_id)][dimension] = {}
            exact_level_diversity[str(m_id)][dimension] = {}
            evaluation_level_support[str(m_id)][dimension] = {}
            for level in (1, 2):
                probability_col = f"{dimension}_ge{level}_prob"
                positive_stats = _family_stats(
                    train_bucket[train_bucket[probability_col] >= valid_th])
                negative_stats = _family_stats(
                    train_bucket[train_bucket[probability_col] < valid_th])
                applicable = bool(CONFIG["ORDINAL_APPLICABILITY"][m_id][dimension][level - 1])
                cell_diversity[str(m_id)][dimension][str(level)] = {
                    "applicable": applicable,
                    "positive": positive_stats,
                    "negative": negative_stats,
                }
                cell_name = f"m_id={m_id}:{dimension}>={level}"
                if not applicable:
                    if positive_stats["rows"] > 0:
                        usability_failures.append(
                            f"{cell_name} 配置为结构零但观察到 {positive_stats['rows']} 个正样本")
                    continue
                if positive_stats["rows"] < int(CONFIG["MIN_TRAIN_POSITIVE_ROWS"]):
                    usability_failures.append(
                        f"{cell_name} 正例行数={positive_stats['rows']} < "
                        f"{CONFIG['MIN_TRAIN_POSITIVE_ROWS']}")
                _append_family_failures(
                    positive_stats, f"{cell_name} 正例",
                    int(CONFIG["MIN_TRAIN_POSITIVE_ROOTS"]))
                _append_family_failures(
                    negative_stats, f"{cell_name} 负例",
                    int(CONFIG["MIN_TRAIN_NEGATIVE_ROOTS"]))

            level_values = train_bucket[f"{dimension}_level"].astype(int)
            for exact_level in (0, 1, 2):
                required = (
                    exact_level == 0 or
                    (exact_level == 1 and bool(CONFIG["ORDINAL_APPLICABILITY"][m_id][dimension][0])) or
                    (exact_level == 2 and bool(CONFIG["ORDINAL_APPLICABILITY"][m_id][dimension][1]))
                )
                stats = _family_stats(train_bucket[level_values == exact_level])
                stats["required"] = required
                exact_level_diversity[str(m_id)][dimension][str(exact_level)] = stats
                if required:
                    if (
                        stats["rows"]
                        < int(CONFIG["MIN_TRAIN_EXACT_LEVEL_ROWS"])
                    ):
                        usability_failures.append(
                            f"m_id={m_id}:{dimension}=L{exact_level} "
                            f"训练行数={stats['rows']} < "
                            f"{CONFIG['MIN_TRAIN_EXACT_LEVEL_ROWS']}")
                    _append_family_failures(
                        stats, f"m_id={m_id}:{dimension}=L{exact_level}",
                        int(CONFIG["MIN_TRAIN_LEVEL_ROOTS"]))
                evaluation_level_support[str(m_id)][dimension][
                    str(exact_level)] = {}
                if required:
                    for split_role in ("val", "test"):
                        evaluation_rows = bucket[
                            (bucket["split_role"] == split_role)
                            & (
                                bucket[f"{dimension}_level"].astype(int)
                                == exact_level
                            )
                        ]
                        evaluation_stats = _family_stats(
                            evaluation_rows)
                        evaluation_level_support[str(m_id)][dimension][
                            str(exact_level)][split_role] = (
                                evaluation_stats)
                        if (
                            evaluation_support_enforced
                            and
                            evaluation_stats["rows"]
                            < int(CONFIG[
                                "MIN_EVAL_EXACT_LEVEL_ROWS"])
                        ):
                            usability_failures.append(
                                f"m_id={m_id}:{dimension}=L{exact_level} "
                                f"{split_role} 行数="
                                f"{evaluation_stats['rows']} < "
                                f"{CONFIG['MIN_EVAL_EXACT_LEVEL_ROWS']}")
                        if (
                            evaluation_support_enforced
                            and
                            evaluation_stats["root_families"]
                            < int(CONFIG[
                                "MIN_EVAL_EXACT_LEVEL_ROOTS"])
                        ):
                            usability_failures.append(
                                f"m_id={m_id}:{dimension}=L{exact_level} "
                                f"{split_role} root="
                                f"{evaluation_stats['root_families']} < "
                                f"{CONFIG['MIN_EVAL_EXACT_LEVEL_ROOTS']}")

    mc_counts = final_df["label_mc_replicates"].astype(int).value_counts().sort_index()
    mc_histogram = {str(int(k)): int(v) for k, v in mc_counts.items()}
    root_sizes = final_df["root_seed_id"].astype(str).value_counts()
    loss_weight = pd.to_numeric(
        final_df["loss_weight"], errors="coerce").to_numpy(dtype=float)

    def _weight_ess(values: np.ndarray) -> Tuple[float, float]:
        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            return 0.0, 0.0
        ess = float(values.sum() ** 2 / np.square(values).sum())
        return ess, float(ess / len(values))

    weight_ess, weight_ess_ratio = _weight_ess(loss_weight)
    weight_ess_by_split = {}
    if "split_role" in final_df.columns:
        roles = final_df["split_role"].astype(str).to_numpy()
        for role in ("train", "val", "test"):
            role_weights = loss_weight[roles == role]
            role_ess, role_ratio = _weight_ess(role_weights)
            weight_ess_by_split[role] = {
                "rows": int(len(role_weights)),
                "effective_sample_size": role_ess,
                "effective_sample_size_ratio": role_ratio,
            }
    else:
        weight_ess_by_split["train"] = {
            "rows": int(len(loss_weight)),
            "effective_sample_size": weight_ess,
            "effective_sample_size_ratio": weight_ess_ratio,
        }
    train_weight_ess_ratio = float(
        weight_ess_by_split.get("train", {}).get(
            "effective_sample_size_ratio", 0.0))
    if train_weight_ess_ratio < float(CONFIG["MIN_WEIGHT_ESS_RATIO"]):
        usability_failures.append(
            f"train loss_weight ESS 比例={train_weight_ess_ratio:.3f} < "
            f"{CONFIG['MIN_WEIGHT_ESS_RATIO']}")
    observed_k2_rows = int(
        (final_df["K2_prob"] >= valid_th).sum())
    observed_k2_ratio = float(
        observed_k2_rows / max(len(final_df), 1))
    k2_ratio_enforced = (
        int(len(final_df)) >= int(CONFIG["USABILITY_GATE_MIN_ROWS"]))
    if (
        k2_ratio_enforced and
        observed_k2_ratio > float(CONFIG["K2_FINAL_MAX_RATIO"]) + 1e-12
    ):
        usability_failures.append(
            f"最终 K2 比例={observed_k2_ratio:.4f} > "
            f"{CONFIG['K2_FINAL_MAX_RATIO']:.4f}")
    config_sha256 = hashlib.sha256(
        json.dumps(CONFIG, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    return {
        "profile_schema": CONFIG["GENERATION_PROFILE_SCHEMA"],
        "dataset_schema": CONFIG["DATASET_SCHEMA"],
        "frame_convention": CONFIG["FRAME_CONVENTION_VERSION"],
        "phase2_mode": "per_munition_topoff",
        "target_total": int(target_total),
        "phase1_ratio": float(phase1_ratio),
        "phase1_kept_counts": {str(k): int(v) for k, v in phase1_kept_counts.items()},
        "seed_prob_relax": float(seed_th),
        "valid_prob_strict": float(valid_th),
        "k2_ratio_contract": {
            "enforced": k2_ratio_enforced,
            "phase2_stop_ratio": float(CONFIG["K2_PHASE2_STOP_RATIO"]),
            "final_max_ratio": float(CONFIG["K2_FINAL_MAX_RATIO"]),
            "observed_positive_rows": observed_k2_rows,
            "observed_final_ratio": observed_k2_ratio,
            "semantics": (
                "phase2_stop_is_not_a_final_target; "
                "final_max_is_the_enforced_safety_ceiling"
            ),
        },
        "munition_target_ratios": {str(k): float(v) for k, v in CONFIG["MUNITION_FINAL_TARGET"].items()},
        "per_munition": per_munition,
        "phase2_root_discovery": {
            str(m): phase2_discovery_stats.get(m, {})
            for m in range(4)
        } if phase2_discovery_stats is not None else {},
        "phase2_ordinal_cell_cap_removals": {
            str(m): phase2_cell_cap_stats.get(m, {})
            for m in range(4)
        } if phase2_cell_cap_stats is not None else {},
        "global_positive_counts": _task_positive_counts(final_df, valid_th),
        "split_counts": (
            {str(k): int(v) for k, v in final_df["split_role"].value_counts().items()}
            if "split_role" in final_df.columns else {}
        ),
        "lineage_columns": ["sample_id", "root_seed_id", "parent_id", "crawl_stage"],
        "label_mc": {
            "rng_seed": int(CONFIG["RANDOM_SEED"]),
            "rng_lineage": (
                "blake2s32(random_seed|sample_id|pair_id), "
                "spread_sign=(+1,-1)"
                if bool(CONFIG.get("LABEL_MC_ANTITHETIC", False))
                else "blake2s32(random_seed|sample_id|replicate_id)"
            ),
            "antithetic_pairs": bool(
                CONFIG.get("LABEL_MC_ANTITHETIC", False)),
            "minimum_configured": int(CONFIG["LABEL_MC_MIN_REPLICATES"]),
            "maximum_configured": int(CONFIG["LABEL_MC_MAX_REPLICATES"]),
            "boundary_half_width": float(CONFIG["LABEL_MC_BOUNDARY_HALF_WIDTH"]),
            "std_trigger": float(CONFIG["LABEL_MC_STD_TRIGGER"]),
            "confidence_z": float(CONFIG["LABEL_MC_CONFIDENCE_Z"]),
            "standard_error_target": float(
                CONFIG["LABEL_MC_STANDARD_ERROR_TARGET"]),
            "decision_margin": float(
                CONFIG["LABEL_MC_DECISION_MARGIN"]),
            "stopping_rule": (
                "after_minimum_replicates_stop_only_when_all_ordinal_"
                "standard_errors_meet_target_and_all_simultaneous_"
                "confidence_intervals_exclude_the_0p5_decision_margin"
            ),
            "observed_minimum": int(mc_counts.index.min()),
            "observed_maximum": int(mc_counts.index.max()),
            "observed_mean": float(np.average(mc_counts.index, weights=mc_counts.values)),
            "replicate_histogram": mc_histogram,
            "aggregation": "mean_of_per_replicate_ordinal_probabilities",
            "all_resolved_rows": (
                int(final_df["label_mc_all_resolved"].astype(bool).sum())
                if "label_mc_all_resolved" in final_df.columns else None
            ),
            "all_resolved_ratio": (
                float(final_df["label_mc_all_resolved"].astype(bool).mean())
                if "label_mc_all_resolved" in final_df.columns else None
            ),
            "maximum_reached_rows": (
                int(final_df["label_mc_max_reached"].astype(bool).sum())
                if "label_mc_max_reached" in final_df.columns else None
            ),
            "maximum_reached_ratio": (
                float(final_df["label_mc_max_reached"].astype(bool).mean())
                if "label_mc_max_reached" in final_df.columns else None
            ),
        },
        "ordinal_applicability": {
            str(m): CONFIG["ORDINAL_APPLICABILITY"][m] for m in range(4)
        },
        "positive_family_diversity_train": cell_diversity,
        "exact_level_family_diversity_train": exact_level_diversity,
        "training_exact_level_support": {
            "minimum_rows": int(
                CONFIG["MIN_TRAIN_EXACT_LEVEL_ROWS"]),
            "minimum_root_families": int(
                CONFIG["MIN_TRAIN_LEVEL_ROOTS"]),
            "cells": exact_level_diversity,
        },
        "evaluation_exact_level_support": {
            "enforced": evaluation_support_enforced,
            "minimum_rows": int(
                CONFIG["MIN_EVAL_EXACT_LEVEL_ROWS"]),
            "minimum_root_families": int(
                CONFIG["MIN_EVAL_EXACT_LEVEL_ROOTS"]),
            "cells": evaluation_level_support,
        },
        "evaluation_split_rebalance": final_df.attrs.get(
            "evaluation_split_rebalance", {}),
        "level2_total_support_topoff": final_df.attrs.get(
            "level2_total_support_topoff", {}),
        "family_distribution": {
            "root_families": int(len(root_sizes)),
            "maximum_rows_per_root_configured": int(CONFIG["MAX_ROWS_PER_ROOT"]),
            "observed_maximum_rows_per_root": int(root_sizes.max()),
            "median_rows_per_root": float(root_sizes.median()),
        },
        "weighting": {
            "generator_class_balance_enabled": bool(CONFIG["APPLY_GENERATOR_CB_WEIGHT"]),
            "loss_weight_min": float(loss_weight.min()),
            "loss_weight_max": float(loss_weight.max()),
            "loss_weight_mean": float(loss_weight.mean()),
            "effective_sample_size": weight_ess,
            "effective_sample_size_ratio": weight_ess_ratio,
            "train_effective_sample_size_ratio": train_weight_ess_ratio,
            "by_split": weight_ess_by_split,
            "minimum_effective_sample_size_ratio": float(CONFIG["MIN_WEIGHT_ESS_RATIO"]),
            "target_effective_sample_size_ratio": float(
                final_df.attrs.get(
                    "weight_ess_target_ratio",
                    min(
                        1.0,
                        float(CONFIG["MIN_WEIGHT_ESS_RATIO"]) +
                        float(CONFIG.get("WEIGHT_ESS_TARGET_MARGIN", 0.0)),
                    ),
                )
            ),
            "tempering_alpha": float(final_df.attrs.get("weight_tempering_alpha", 1.0)),
            "floor_count": int(final_df.attrs.get("weight_floor_count", 0)),
            "cap_count": int(final_df.attrs.get("weight_cap_count", 0)),
            "factors": ["aoa_ipw", "physics_weight", "active_sampling_weight",
                        "family_weight", "class_balance_weight"],
        },
        "usability_gate": {
            "passed": not usability_failures,
            "enforced": int(len(final_df)) >= int(CONFIG["USABILITY_GATE_MIN_ROWS"]),
            "failures": usability_failures,
            "scope": (
                "applicable train ordinal cells; production validation/test "
                "exact-level evidence"
            ),
        },
        "provenance": {
            "random_seed": int(CONFIG["RANDOM_SEED"]),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "config_sha256": config_sha256,
            "parquet_writer": "pyarrow",
        },
    }


# ============================================================================
# 多进程 Engine 计算子任务
# ============================================================================

# 修复 #9：每个子进程一次性载入 vehicle 模型常量到模块级全局，避免逐任务 IPC 复制大对象
_WORKER_COMPONENTS = None
_WORKER_PLATES = None


def _init_worker(components, plates):
    """ProcessPoolExecutor initializer：每个 worker 启动时缓存模型，仅 1 次传输"""
    global _WORKER_COMPONENTS, _WORKER_PLATES
    _WORKER_COMPONENTS = components
    _WORKER_PLATES = plates


def _mc_resolution_diagnostics(
        ordinal_samples: np.ndarray) -> Dict[str, Any]:
    """Return per-head precision and decision-margin diagnostics."""
    samples = np.asarray(ordinal_samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 8:
        raise ValueError(
            "ordinal_samples must have shape (replicates,8)")
    if not np.isfinite(samples).all():
        raise ValueError(
            "ordinal_samples contain non-finite values")
    mean = samples.mean(axis=0)
    if samples.shape[0] > 1:
        standard_error = (
            samples.std(axis=0, ddof=1)
            / np.sqrt(float(samples.shape[0]))
        )
    else:
        standard_error = np.full(8, np.inf, dtype=np.float64)
    confidence_radius = (
        float(CONFIG["LABEL_MC_CONFIDENCE_Z"])
        * standard_error
    )
    lower = mean - confidence_radius
    upper = mean + confidence_radius
    margin = float(CONFIG["LABEL_MC_DECISION_MARGIN"])
    decision_resolved = (
        (upper < 0.5 - margin)
        | (lower > 0.5 + margin)
    )
    precision_resolved = (
        standard_error
        <= float(CONFIG["LABEL_MC_STANDARD_ERROR_TARGET"])
    )
    minimum_reached = bool(
        samples.shape[0]
        >= int(CONFIG["LABEL_MC_MIN_REPLICATES"]))
    resolved_mask = (
        decision_resolved
        & precision_resolved
        & minimum_reached
    )
    return {
        "mean": mean,
        "standard_error": standard_error,
        "confidence_lower": lower,
        "confidence_upper": upper,
        "decision_resolved": decision_resolved,
        "precision_resolved": precision_resolved,
        "resolved_mask": resolved_mask,
        "all_resolved": bool(np.all(resolved_mask)),
    }


def _mc_estimate_is_resolved(
        ordinal_samples: np.ndarray) -> bool:
    """Return whether every ordinal decision is precise and margin-resolved."""
    return bool(
        _mc_resolution_diagnostics(
            ordinal_samples)["all_resolved"])


_PHASE1_CHECKPOINT_SCHEMA = "stage0_phase1_checkpoint_v1"
_CHECKPOINT_TASK_INDEX = "__stage0_task_index"
_PHASE1_CHECKPOINT_COMPATIBILITY_SCHEMA = (
    "stage0_phase1_checkpoint_compatibility_v1")
_PHASE1_CHECKPOINT_COMPATIBILITY_FILE = (
    "phase1_checkpoint_compatibility.json")


def _write_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    """Write a JSON control artifact without exposing a partial file."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temporary = f"{path}.{os.getpid()}.tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def _phase1_input_sha256(frame: pd.DataFrame) -> str:
    """Content identity for deterministic Phase-1 proposals."""
    normalized = frame.reset_index(drop=True)
    try:
        row_hashes = pd.util.hash_pandas_object(
            normalized, index=False, categorize=True,
        ).to_numpy(dtype=np.uint64, copy=False)
    except TypeError as exc:
        raise RuntimeError(
            "Phase-1 checkpoint input contains an unhashable column; "
            "checkpoint reuse cannot be proven safe."
        ) from exc
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "columns": [str(column) for column in normalized.columns],
                "dtypes": [str(dtype) for dtype in normalized.dtypes],
                "rows": int(len(normalized)),
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    )
    digest.update(row_hashes.tobytes(order="C"))
    return digest.hexdigest()


def _phase1_checkpoint_identity(
        frame: pd.DataFrame,
        namespace: str = "phase1",
) -> Dict[str, Any]:
    """Bind resumable results to inputs, policy, physics and geometry."""
    tracked_files = {
        "generate_dataset.py": os.path.abspath(__file__),
        "sim_engine.py": os.path.abspath(engine_module.__file__),
        "coordinate_frames.py": os.path.abspath(
            coordinate_frames_module.__file__),
        "component_supervision.py": os.path.abspath(
            component_supervision_module.__file__),
        "vehicle_model.json": bundled_resource_path("vehicle_model.json"),
        "armor.csv": bundled_resource_path("armor.csv"),
    }
    file_hashes = {
        filename: (
            sha256_file(path) if os.path.isfile(path) else None
        )
        for filename, path in tracked_files.items()
    }
    config_sha256 = hashlib.sha256(
        json.dumps(
            CONFIG, sort_keys=True, ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    identity = {
        "schema": _PHASE1_CHECKPOINT_SCHEMA,
        "namespace": str(namespace),
        "rows": int(len(frame)),
        "input_sha256": _phase1_input_sha256(frame),
        "config_sha256": config_sha256,
        "file_sha256": file_hashes,
    }
    identity["signature"] = hashlib.sha256(
        json.dumps(
            identity, sort_keys=True, ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return identity


def _load_phase1_checkpoint_compatibility() -> List[Dict[str, str]]:
    """Load explicitly reviewed post-processing-only source migrations.

    The target generator hash is stored outside ``generate_dataset.py`` so a
    later edit to this file automatically invalidates the exception.  Physics,
    geometry, input and configuration identities must still match exactly.
    """
    path = os.path.join(
        str(PROJECT_ROOT), _PHASE1_CHECKPOINT_COMPATIBILITY_FILE)
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schema") != _PHASE1_CHECKPOINT_COMPATIBILITY_SCHEMA:
        raise RuntimeError(
            "Phase-1 checkpoint compatibility file has an unsupported "
            "schema.")
    migrations = payload.get("migrations", [])
    if not isinstance(migrations, list):
        raise RuntimeError(
            "Phase-1 checkpoint compatibility migrations must be a list.")
    normalized = []
    for migration in migrations:
        if not isinstance(migration, dict):
            raise RuntimeError(
                "Phase-1 checkpoint compatibility entry must be an object.")
        source_hash = str(
            migration.get("from_generator_sha256", "")).lower()
        target_hash = str(
            migration.get("to_generator_sha256", "")).lower()
        reason = str(migration.get("reason", "")).strip()
        if (
            len(source_hash) != 64
            or len(target_hash) != 64
            or any(character not in "0123456789abcdef"
                   for character in source_hash + target_hash)
            or not reason
        ):
            raise RuntimeError(
                "Invalid Phase-1 checkpoint compatibility migration.")
        normalized.append({
            "from_generator_sha256": source_hash,
            "to_generator_sha256": target_hash,
            "reason": reason,
        })
    return normalized


def _phase1_checkpoint_identities_are_compatible(
        stored: Dict[str, Any], current: Dict[str, Any]) -> bool:
    """Accept only a declared source-only migration with identical physics."""
    for field in (
        "schema", "namespace", "rows", "input_sha256", "config_sha256"):
        if stored.get(field) != current.get(field):
            return False

    stored_files = stored.get("file_sha256", {})
    current_files = current.get("file_sha256", {})
    if set(stored_files) != set(current_files):
        return False
    for filename in stored_files:
        if filename == "generate_dataset.py":
            continue
        if stored_files.get(filename) != current_files.get(filename):
            return False

    source_hash = str(stored_files.get(
        "generate_dataset.py", "")).lower()
    target_hash = str(current_files.get(
        "generate_dataset.py", "")).lower()
    return any(
        migration["from_generator_sha256"] == source_hash
        and migration["to_generator_sha256"] == target_hash
        for migration in _load_phase1_checkpoint_compatibility()
    )


def _prepare_phase1_checkpoint(identity: Dict[str, Any]) -> str:
    """Create or strictly validate the signature-specific checkpoint root."""
    base_dir = str(CONFIG["PHASE1_CHECKPOINT_DIR"])
    if not os.path.isabs(base_dir):
        base_dir = os.path.join(str(PROJECT_ROOT), base_dir)
    base_dir = os.path.abspath(base_dir)
    checkpoint_dir = os.path.join(
        base_dir, str(identity["signature"]))
    manifest_path = os.path.join(checkpoint_dir, "manifest.json")
    if not os.path.isfile(manifest_path) and os.path.isdir(base_dir):
        compatible = []
        for name in sorted(os.listdir(base_dir)):
            candidate_dir = os.path.join(base_dir, name)
            candidate_manifest = os.path.join(
                candidate_dir, "manifest.json")
            if not os.path.isfile(candidate_manifest):
                continue
            with open(candidate_manifest, "r", encoding="utf-8") as stream:
                manifest = json.load(stream)
            stored_identity = manifest.get("identity", {})
            if (
                name == str(stored_identity.get("signature", ""))
                and manifest.get("schema") == _PHASE1_CHECKPOINT_SCHEMA
                and bool(manifest.get("complete", False))
                and int(manifest.get("completed_rows", -1))
                == int(identity["rows"])
                and _phase1_checkpoint_identities_are_compatible(
                    stored_identity, identity)
            ):
                compatible.append(candidate_dir)
        if len(compatible) > 1:
            raise RuntimeError(
                "Multiple compatible Phase-1 checkpoints were found; "
                "refusing an ambiguous recovery.")
        if compatible:
            print(
                "  [Checkpoint] 已通过显式源码迁移、输入、配置及物理文件"
                "身份校验，复用完整 Phase-1 检查点。")
            return compatible[0]

    parts_dir = os.path.join(checkpoint_dir, "parts")
    os.makedirs(parts_dir, exist_ok=True)
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("identity") != identity:
            raise RuntimeError(
                "Phase-1 checkpoint manifest identity mismatch; refusing "
                "to combine results from different inputs or physics."
            )
    else:
        _write_json_atomic(manifest_path, {
            "schema": _PHASE1_CHECKPOINT_SCHEMA,
            "identity": identity,
            "complete": False,
            "created_unix_time": float(time.time()),
        })
    return checkpoint_dir


def _checkpoint_part_paths(checkpoint_dir: str) -> List[str]:
    parts_dir = os.path.join(checkpoint_dir, "parts")
    if not os.path.isdir(parts_dir):
        return []
    return sorted(
        os.path.join(parts_dir, name)
        for name in os.listdir(parts_dir)
        if name.startswith("part-") and name.endswith(".parquet")
    )


def _load_simulation_checkpoint(
        checkpoint_dir: str,
        task_count: int,
) -> Dict[int, Dict[str, Any]]:
    """Load only complete atomic shards and reject overlap/corruption."""
    restored: Dict[int, Dict[str, Any]] = {}
    for part_path in _checkpoint_part_paths(checkpoint_dir):
        part = pd.read_parquet(part_path, engine="pyarrow")
        if _CHECKPOINT_TASK_INDEX not in part.columns:
            raise RuntimeError(
                f"Checkpoint part lacks {_CHECKPOINT_TASK_INDEX}: {part_path}")
        indices = part.pop(_CHECKPOINT_TASK_INDEX).astype(np.int64)
        if indices.duplicated().any():
            raise RuntimeError(
                f"Checkpoint part contains duplicate task indices: {part_path}")
        for index, record in zip(indices.tolist(), part.to_dict("records")):
            index = int(index)
            if index < 0 or index >= int(task_count):
                raise RuntimeError(
                    f"Checkpoint task index {index} is outside [0,{task_count}).")
            if index in restored:
                raise RuntimeError(
                    f"Checkpoint task index {index} occurs in multiple parts.")
            restored[index] = record
    return restored


def _write_simulation_checkpoint_part(
        checkpoint_dir: str,
        indexed_records: List[Tuple[int, Dict[str, Any]]],
) -> Optional[str]:
    """Atomically persist one set of completed simulation results."""
    if not indexed_records:
        return None
    ordered = sorted(indexed_records, key=lambda item: int(item[0]))
    rows = []
    indices = []
    for index, record in ordered:
        row = dict(record)
        if _CHECKPOINT_TASK_INDEX in row:
            raise RuntimeError(
                f"Simulation result illegally uses reserved column "
                f"{_CHECKPOINT_TASK_INDEX}.")
        row[_CHECKPOINT_TASK_INDEX] = int(index)
        rows.append(row)
        indices.append(int(index))
    part_frame = pd.DataFrame(rows)
    index_digest = hashlib.sha256(
        np.asarray(indices, dtype=np.int64).tobytes(order="C")
    ).hexdigest()[:16]
    parts_dir = os.path.join(checkpoint_dir, "parts")
    os.makedirs(parts_dir, exist_ok=True)
    part_name = (
        f"part-{min(indices):06d}-{max(indices):06d}-"
        f"{index_digest}.parquet")
    part_path = os.path.join(parts_dir, part_name)
    if os.path.exists(part_path):
        existing = pd.read_parquet(part_path, engine="pyarrow")
        if not existing.equals(part_frame):
            raise RuntimeError(
                f"Checkpoint part collision with different content: {part_path}")
        return part_path
    temporary = f"{part_path}.{os.getpid()}.tmp"
    table = pa.Table.from_pandas(part_frame, preserve_index=False)
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, part_path)
    return part_path


def _mark_phase1_checkpoint_complete(
        checkpoint_dir: str,
        restored_rows: int,
) -> None:
    manifest_path = os.path.join(checkpoint_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    manifest["complete"] = True
    manifest["completed_rows"] = int(restored_rows)
    manifest["completed_unix_time"] = float(time.time())
    _write_json_atomic(manifest_path, manifest)


def _process_single_encounter(args):
    """提取出的单次仿真独立任务"""
    i, row = args

    x_i, y_i, z_i = row['x'], row['y'], row['z']
    vx_i, vy_i, vz_i = row['vx'], row['vy'], row['vz']
    pitch_i, roll_i, yaw_i = row['pitch'], row['roll'], row['yaw']
    m_id = int(row['m_id'])

    enc = EncounterCondition(
        dx=x_i, dy=y_i, dz=z_i,
        vx=vx_i, vy=vy_i, vz=vz_i,
        pitch_deg=pitch_i, roll_deg=roll_i, yaw_deg=yaw_i
    )

    # 修复 #9：从子进程全局读取模型，不再依赖 row 传输 (省去 N 次 IPC 拷贝)
    engine = DamageEngine(armor_plates=_WORKER_PLATES)

    mun_dict = {
        0: create_small_loitering_munition(),
        1: create_medium_loitering_munition(),
        2: create_medium_rear_det(),
        3: create_heavy_loitering_munition()
    }
    proj = mun_dict[m_id]

    # Stage-0: seed derives from immutable sample lineage rather than the batch-local
    # index (which restarted at zero for every batch and silently reused spreads).
    sample_id = str(row.get("sample_id", f"legacy-{i}"))
    # New samples use adaptive MC: easy, stable points stop at the minimum;
    # boundary/noisy points receive more replicates.  A legacy row containing
    # only ``label_mc_replicates`` remains fixed-count for backwards-compatible
    # diagnostics and tests.
    has_adaptive_bounds = (
        "label_mc_min_replicates" in row and
        "label_mc_max_replicates" in row
    )
    if has_adaptive_bounds:
        min_replicates = max(1, int(row.get(
            "label_mc_min_replicates", CONFIG["LABEL_MC_MIN_REPLICATES"])))
        max_replicates = max(min_replicates, int(row.get(
            "label_mc_max_replicates", CONFIG["LABEL_MC_MAX_REPLICATES"])))
    else:
        fixed_replicates = max(1, int(row.get(
            "label_mc_replicates", CONFIG["LABEL_MC_REPLICATES"])))
        min_replicates = fixed_replicates
        max_replicates = fixed_replicates
    replicate_results = []
    shockwave_probability_cache = None
    for replicate_id in range(max_replicates):
        antithetic = bool(
            CONFIG.get("LABEL_MC_ANTITHETIC", False))
        rng_seed, spread_sign = _label_mc_rng_pair(
            sample_id, replicate_id)
        replicate_result = engine.evaluate(
            proj,
            enc,
            _WORKER_COMPONENTS,
            rng_seed=rng_seed,
            fragment_spread_sign=spread_sign,
            shockwave_probability_cache=(
                shockwave_probability_cache),
        )
        replicate_results.append(replicate_result)
        if shockwave_probability_cache is None:
            # Shock and its geometric shielding are deterministic for a fixed
            # encounter.  Cache the first replicate's component probabilities
            # and reuse them while only resampling the fragment spread.
            shockwave_probability_cache = {
                int(component.component_id): float(
                    component.shockwave_damage_prob)
                for component in replicate_result.component_results
            }
        pair_complete = (
            not antithetic or replicate_id % 2 == 1)
        if (
            has_adaptive_bounds
            and pair_complete
            and len(replicate_results) >= min_replicates
        ):
            ordinal_so_far = np.asarray([
                result.damage_tree.ordinal_probability_vector
                if result.damage_tree is not None else np.zeros(8)
                for result in replicate_results
            ], dtype=float)
            if _mc_estimate_is_resolved(ordinal_so_far):
                break
    n_replicates = len(replicate_results)

    env_cols = [x_i, y_i, z_i, vx_i, vy_i, vz_i,
                np.sin(np.radians(yaw_i)), np.cos(np.radians(yaw_i)),
                np.sin(np.radians(pitch_i)), np.cos(np.radians(pitch_i)),
                np.sin(np.radians(roll_i)), np.cos(np.radians(roll_i))]

    v_norm = np.sqrt(vx_i**2 + vy_i**2 + vz_i**2)
    target_x = float(row.get("target_x", 0.0))
    target_y = float(row.get("target_y", 0.0))
    target_z = float(row.get("target_z", 0.0))
    if not (np.isfinite(target_x) and np.isfinite(target_y) and np.isfinite(target_z)):
        target_x, target_y, target_z = 0.0, 0.0, 0.0

    # Stage-0 generation-only diagnostics: these are relative to the sampler's
    # Aim-and-Shoot point and therefore must never enter the surrogate input.
    rel_x = target_x - x_i
    rel_y = target_y - y_i
    rel_z = target_z - z_i
    # They remain in Parquet for sampling diagnostics only.
    d_dist = np.sqrt(rel_x**2 + rel_y**2 + rel_z**2) + 1e-4
    cos_alpha = (rel_x*vx_i + rel_y*vy_i + rel_z*vz_i) / (d_dist * v_norm + 1e-4)
    cos_alpha = float(np.clip(cos_alpha, -1.0, 1.0))

    event_samples = np.asarray([
        result.damage_tree.level_vector if result.damage_tree is not None else np.zeros(8)
        for result in replicate_results
    ], dtype=float)
    event_mean = event_samples.mean(axis=0)
    event_std = event_samples.std(axis=0)
    k1, k2, m1, m2, f1, f2, c1, c2 = event_mean.tolist()

    # Average the per-replicate ordinal probabilities directly.  Applying the
    # non-linear OR to already-averaged raw rule events is biased whenever the
    # two events co-vary across fragment realizations.
    ordinal_samples = np.asarray([
        result.damage_tree.ordinal_probability_vector
        if result.damage_tree is not None else np.zeros(8)
        for result in replicate_results
    ], dtype=float)
    ordinal_mean = ordinal_samples.mean(axis=0)
    ordinal_std = ordinal_samples.std(axis=0)
    mc_resolution = _mc_resolution_diagnostics(
        ordinal_samples)
    ordinal_pairs = {
        name: (ordinal_mean[2 * idx], ordinal_mean[2 * idx + 1])
        for idx, name in enumerate(("K", "M", "F", "C"))
    }
    ordinal_pairs = {
        name: (float(np.clip(ge1, 0.0, 1.0)),
               float(np.clip(min(ge2, ge1), 0.0, 1.0)))
        for name, (ge1, ge2) in ordinal_pairs.items()
    }
    levels = {
        name: 2 if ge2 >= 0.5 else (1 if ge1 >= 0.5 else 0)
        for name, (ge1, ge2) in ordinal_pairs.items()
    }
    k_lev, m_lev, f_lev, c_lev = (
        levels["K"], levels["M"], levels["F"], levels["C"])
    overall_score = float(
        0.35 * 0.5 * sum(ordinal_pairs["K"]) +
        0.25 * 0.5 * sum(ordinal_pairs["F"]) +
        0.22 * 0.5 * sum(ordinal_pairs["M"]) +
        0.18 * 0.5 * sum(ordinal_pairs["C"])
    )

    mechanism_means = {}
    mechanism_scores = {}
    for mechanism, attr_name in (
        ("fragment", "damage_tree_fragment"),
        ("shock", "damage_tree_shockwave"),
    ):
        samples = np.asarray([
            getattr(result, attr_name).ordinal_probability_vector
            if getattr(result, attr_name, None) is not None else np.zeros(8)
            for result in replicate_results
        ], dtype=float)
        mechanism_means[mechanism] = samples.mean(axis=0)
        mechanism_scores[mechanism] = float(np.mean([
            getattr(result, attr_name).overall_score
            if getattr(result, attr_name, None) is not None else 0.0
            for result in replicate_results
        ]))
    component_means = extract_component_mc_means(replicate_results)

    # 第三轮新增：m_id 条件化的 K 任务训练权重
    # [P0-1] 新权重表 {0.40, 0.50, 0.55, 1.00}，非 Heavy 弹型 K 梯度抬 4×
    k_task_w = CONFIG["M_ID_K_TASK_WEIGHT"][m_id]
    # [P0-2] 新增 C 分支弹型权重 {1.00, 1.20, 1.30, 0.85}
    c_task_w = CONFIG["M_ID_C_TASK_WEIGHT"][m_id]

    result_dict = {
        "x_cm": env_cols[0], "y_cm": env_cols[1], "z_cm": env_cols[2],
        "vx_ms": env_cols[3], "vy_ms": env_cols[4], "vz_ms": env_cols[5],
        "sin_yaw": env_cols[6], "cos_yaw": env_cols[7],
        "sin_pitch": env_cols[8], "cos_pitch": env_cols[9],
        "sin_roll": env_cols[10], "cos_roll": env_cols[11],
        "norm_velocity": v_norm,
        # 主动采样诊断元数据；不在 nn_dataset.FEATURE_COLUMNS 中。
        "los_distance": d_dist,
        "impact_cosine": cos_alpha,
        "target_x": target_x, "target_y": target_y, "target_z": target_z,
        "fragment_aim_sign": float(row.get("fragment_aim_sign", 1.0)),
        "cone_aim_angle_deg": float(row.get("cone_aim_angle_deg", 0.0)),
        "sampling_geometry": str(row.get("sampling_geometry", "legacy_unknown")),
        "munition_id": m_id,
        "x": x_i, "y": y_i, "z": z_i,
        "vx": vx_i, "vy": vy_i, "vz": vz_i,
        "pitch": pitch_i, "roll": roll_i, "yaw": yaw_i,
        "m_id": m_id,
        "loss_weight": row.get('loss_weight', 1.0),
        "aoa_accept_prob": row.get('aoa_accept_prob', 1.0),
        "aoa_ipw": row.get('aoa_ipw', 1.0),
        "physics_weight": row.get('physics_weight', 1.0),
        "active_sampling_weight": row.get('active_sampling_weight', 1.0),
        "family_weight": row.get('family_weight', 1.0),
        "class_balance_weight": row.get('class_balance_weight', 1.0),
        # 第三轮：K 任务的 m_id 条件化权重 (供 nn_train.py 元素级乘到 K1/K2 BCE loss)
        "K_task_weight": k_task_w,
        # [P0-2] C 任务的 m_id 条件化权重 (供 nn_train.py 元素级乘到 C1/C2 BCE loss)
        "C_task_weight": c_task_w,
        "layer_type": row.get('layer_type', 'UNKNOWN'),  # 透传层段标签
        "sampling_phase": row.get('sampling_phase', 'phase1'),
        "overall_score": overall_score,
        "is_crawled": row.get('is_crawled', 0),
        "K1_prob": k1, "K2_prob": k2,
        "M1_prob": m1, "M2_prob": m2,
        "F1_prob": f1, "F2_prob": f2,
        "C1_prob": c1, "C2_prob": c2,
        "K_ge1_prob": ordinal_pairs["K"][0], "K_ge2_prob": ordinal_pairs["K"][1],
        "M_ge1_prob": ordinal_pairs["M"][0], "M_ge2_prob": ordinal_pairs["M"][1],
        "F_ge1_prob": ordinal_pairs["F"][0], "F_ge2_prob": ordinal_pairs["F"][1],
        "C_ge1_prob": ordinal_pairs["C"][0], "C_ge2_prob": ordinal_pairs["C"][1],
        "K_ge1_prob_std": ordinal_std[0], "K_ge2_prob_std": ordinal_std[1],
        "M_ge1_prob_std": ordinal_std[2], "M_ge2_prob_std": ordinal_std[3],
        "F_ge1_prob_std": ordinal_std[4], "F_ge2_prob_std": ordinal_std[5],
        "C_ge1_prob_std": ordinal_std[6], "C_ge2_prob_std": ordinal_std[7],
        "K1_prob_std": event_std[0], "K2_prob_std": event_std[1],
        "M1_prob_std": event_std[2], "M2_prob_std": event_std[3],
        "F1_prob_std": event_std[4], "F2_prob_std": event_std[5],
        "C1_prob_std": event_std[6], "C2_prob_std": event_std[7],
        "label_mc_replicates": n_replicates,
        "label_mc_min_replicates": min_replicates,
        "label_mc_max_replicates": max_replicates,
        "label_mc_all_resolved": bool(
            mc_resolution["all_resolved"]),
        "label_mc_max_reached": bool(
            n_replicates >= max_replicates),
        "label_mc_max_standard_error": float(
            np.max(mc_resolution["standard_error"])),
        "fragment_overall_score": mechanism_scores["fragment"],
        "shock_overall_score": mechanism_scores["shock"],
        "total_hits": float(np.mean([r.total_hits for r in replicate_results])),
        "total_penetrations": float(np.mean([r.total_penetrations for r in replicate_results])),
        "K_level": k_lev, "M_level": m_lev, "F_level": f_lev, "C_level": c_lev,
        "sample_id": sample_id,
        "root_seed_id": str(row.get("root_seed_id", sample_id)),
        "parent_id": str(row.get("parent_id", "")),
        "crawl_stage": int(row.get("crawl_stage", 0)),
        "split_role": str(row.get("split_role", _reference_split_for_root(sample_id))),
        "frame_version": str(row.get("frame_version", CONFIG["FRAME_CONVENTION_VERSION"])),
        "dataset_schema": str(row.get("dataset_schema", CONFIG["DATASET_SCHEMA"])),
    }
    for mechanism in ("fragment", "shock"):
        values = mechanism_means[mechanism]
        for idx, name in enumerate(("K", "M", "F", "C")):
            result_dict[f"{mechanism}_{name}_ge1_prob"] = values[2 * idx]
            result_dict[f"{mechanism}_{name}_ge2_prob"] = values[2 * idx + 1]
    for task_index, task_name in enumerate(
            ("K", "M", "F", "C")):
        for level_index, level in enumerate((1, 2)):
            ordinal_index = 2 * task_index + level_index
            result_dict[
                f"{task_name}_ge{level}_mc_resolved"
            ] = bool(
                mc_resolution["resolved_mask"][
                    ordinal_index])
            result_dict[
                f"{task_name}_ge{level}_mc_standard_error"
            ] = float(
                mc_resolution["standard_error"][
                    ordinal_index])
    result_dict.update(component_means_to_columns(component_means))
    return result_dict


# ============================================================================
# 代理生成器: 物理边界感知采样引擎
# ============================================================================

class PhysicsAwareSampler:
    """融合了 LHS 约束分层、边界爬行自适应加噪与零地底过滤机制的主动富化生成器"""

    def __init__(self):
        self.components = load_vehicle_model()
        self.plates = load_armor_plates()
        self.min_aabb, self.max_aabb = get_vehicle_aabb(self.components, self.plates)
        self.k_oversample = 4 # LHS 超采样倍率以应对剔除
        self._sample_counter = 0

        # ----------------------------------------------------------------
        # 第三轮 P4 通用化：为 K/M/F/C 4 类任务预计算关键组件中心点
        # 用于各 _HUNT 层的 Aim-and-Shoot 偏置瞄准
        # ----------------------------------------------------------------
        self.target_centers = {}    # task → np.ndarray (n_target, 3)
        for task, cfg_key in [
            ("K2", "K2_CRITICAL_COMPONENT_IDS"),
            ("K1", "K1_CRITICAL_COMPONENT_IDS"),
            ("M",  "M_CRITICAL_COMPONENT_IDS"),
            ("F",  "F_CRITICAL_COMPONENT_IDS"),
            ("C",  "C_CRITICAL_COMPONENT_IDS"),
        ]:
            crit_ids = set(CONFIG[cfg_key])
            centers = []
            for c in self.components:
                if c.get("id") in crit_ids:
                    pos = c.get("geometry", {}).get("position", {})
                    px, py, pz = pos.get("x"), pos.get("y"), pos.get("z")
                    if px is not None and py is not None and pz is not None:
                        centers.append([float(px), float(py), float(pz)])
            self.target_centers[task] = np.array(centers) if centers else np.empty((0, 3))
            print(f"[Sampler] 已锁定 {len(self.target_centers[task]):>2} 个 {task} 关键组件作为瞄准靶点")

        crew_centers = self.target_centers["C"]
        cluster_size = int(CONFIG["C2_CLUSTER_SIZE"])
        c2_centers = []
        if len(crew_centers) >= cluster_size:
            for anchor in crew_centers:
                nearest = np.argsort(np.linalg.norm(
                    crew_centers - anchor, axis=1))[:cluster_size]
                c2_centers.append(crew_centers[nearest].mean(axis=0))
            c2_centers.append(crew_centers.mean(axis=0))
        if c2_centers:
            # 不同 anchor 可能得到同一最近邻簇；稳定去重避免重复加权。
            rounded = np.round(np.asarray(c2_centers, dtype=float), decimals=6)
            _, unique_indices = np.unique(rounded, axis=0, return_index=True)
            self.target_centers["C2"] = np.asarray(c2_centers, dtype=float)[
                np.sort(unique_indices)]
        else:
            self.target_centers["C2"] = np.empty((0, 3))
        print(f"[Sampler] 已锁定 {len(self.target_centers['C2']):>2} 个 C2 乘员簇质心作为瞄准靶点")

        # 向后兼容（外部如果还引用 self.k2_target_centers）
        self.k2_target_centers = self.target_centers["K2"]

        # _HUNT 层的物理窄带参数表 (task_letter → CONFIG keys)
        self._HUNT_PARAMS = {
            "K2": ("K2_HUNT_R_RANGE", "K2_HUNT_V_RANGE", "K2_HUNT_ROLL_RANGE",
                   "K2_AIM_BIAS", "K2_AIM_SIGMA_DEG"),
            # [P1-A] K1 层：瞄 K 关键组件 + 次临界速度窄带 → 产 K1 而非 K2
            "K1": ("K1_HUNT_R_RANGE", "K1_HUNT_V_RANGE", "K1_HUNT_ROLL_RANGE",
                   "K1_AIM_BIAS", "K1_AIM_SIGMA_DEG"),
            "M":  ("M_HUNT_R_RANGE",  "M_HUNT_V_RANGE",  None,
                   "M_AIM_BIAS",  "M_AIM_SIGMA_DEG"),
            "F":  ("F_HUNT_R_RANGE",  "F_HUNT_V_RANGE",  None,
                   "F_AIM_BIAS",  "F_AIM_SIGMA_DEG"),
            "C":  ("C_HUNT_R_RANGE",  "C_HUNT_V_RANGE",  None,
                   "C_AIM_BIAS",  "C_AIM_SIGMA_DEG"),
            "C2": ("C_HUNT_R_RANGE",  "C_HUNT_V_RANGE",  None,
                   "C_AIM_BIAS",  "C_AIM_SIGMA_DEG"),
        }

    def _allocate_sample_ids(self, n_samples: int, prefix: str) -> np.ndarray:
        start = self._sample_counter
        self._sample_counter += int(n_samples)
        return np.asarray([
            f"{prefix}-{CONFIG['RANDOM_SEED']}-{idx:012d}"
            for idx in range(start, start + int(n_samples))
        ], dtype=object)

    def _hunt_task_from_layer(self, layer_type: str):
        """layer_type='K2_HUNT' → 'K2'；非 _HUNT 层返回 None"""
        if layer_type.endswith("_HUNT"):
            return layer_type[:-5]
        return None

    def _generate_lhs_batch(self, n_samples: int, layer_type: str,
                            force_m_id: int = None,
                            exterior_lateral_shell: bool = False) -> pd.DataFrame:
        """核心策略三：约束条件分层 LHS 采样器

        layer_type:
            "M_F"           — 低速负样本对照层
            "K1_K2"         — 普通高速突防层
            "CRITICAL_ROLL" — 侧倾角加密层 (仅 m_id=3 路径下使用)
            "K2_HUNT"       — K2 关键组件瞄准 + 高速窄带 (m_id=3 主力)
            "K1_HUNT"       — [P1-A] K 关键组件瞄准 + 次临界速度窄带 (产 K1 不产 K2)
            "M_HUNT"        — M 关键组件瞄准 + M 任务窄带
            "F_HUNT"        — F 关键组件瞄准 + F 任务窄带
            "C_HUNT"        — C 关键组件瞄准 + C 任务窄带

        force_m_id: 若给定 (0/1/2/3)，强制本批次所有样本为该弹型
                    (per-munition stratified sampling 主入口)
        """
        # 修复 #6：让 qmc 派生种子也走全局 np.random，被入口播种统一覆盖
        # P1：维度 10 → 11，新增 u[:, 10] 作为 target_z 采样
        lhs_seed = int(np.random.randint(0, 2**31 - 1))
        sampler = qmc.LatinHypercube(d=11, seed=lhs_seed)
        u = sampler.random(n=n_samples)

        # ===================================================================
        # 弹型分配：force_m_id 优先 > 全局均匀 (M_ID_PROBS 已退役)
        # ===================================================================
        if force_m_id is not None:
            m_ids = np.full(n_samples, int(force_m_id), dtype=int)
        else:
            # 兜底：未指定时均匀分配
            m_ids = np.random.randint(0, 4, size=n_samples)

        # 解析 _HUNT 任务字母 ("K2_HUNT" → "K2")
        hunt_task = self._hunt_task_from_layer(layer_type)
        is_hunt = hunt_task is not None and len(self.target_centers.get(hunt_task, [])) > 0
        hunt_override = {}
        if is_hunt and force_m_id is not None:
            munition_overrides = CONFIG["HUNT_OVERRIDES"].get(
                int(force_m_id), {})
            # C2 may define a cluster-specific proposal.  Older/custom
            # configurations that only define C retain the previous fallback.
            if hunt_task == "C2":
                hunt_override = munition_overrides.get(
                    "C2", munition_overrides.get("C", {}))
            else:
                hunt_override = munition_overrides.get(hunt_task, {})

        # ===================================================================
        # 空间分布
        #   _HUNT 层：r 在该任务的窄带范围内均匀采
        #   其他层 ：[0, RADIUS_MAX_CM] 立方根均匀 (球体均匀)
        # ===================================================================
        phi = 2 * np.pi * u[:, 0]
        z_factor = 0.001 + 0.999 * u[:, 1]   # 保 Z>0

        if is_hunt:
            r_key, _, _, _, _ = self._HUNT_PARAMS[hunt_task]
            r_lo, r_hi = hunt_override.get("r_range", CONFIG[r_key])
            r = r_lo + (r_hi - r_lo) * u[:, 2]
        else:
            r = CONFIG["RADIUS_MAX_CM"] * np.cbrt(u[:, 2])
        z = r * z_factor
        r_xy = r * np.sqrt(1 - z_factor**2)
        x = r_xy * np.cos(phi)
        y = r_xy * np.sin(phi)

        # ===================================================================
        # 速度大小分布
        #   _HUNT 层 ：在该任务的速度窄带内均匀采 (与弹型 V门槛 取交)
        #   M_F 层  ：低速段 [V_MIN, V_threshold]，无法产 K2
        #   其他层  ：高速突防段 [V_threshold, V_MAX]
        # ===================================================================
        v_norm = np.zeros(n_samples)
        if is_hunt:
            _, v_key, _, _, _ = self._HUNT_PARAMS[hunt_task]
            v_lo, v_hi = hunt_override.get("v_range", CONFIG[v_key])
            # 与弹型门槛取交集，确保至少在该弹型有穿甲效能区域
            for i in range(4):
                mask = (m_ids == i)
                if not np.any(mask): continue
                v_th = CONFIG["V_THRESHOLDS"][i]
                # K2_HUNT / K1_HUNT 都要求严格高于门槛（K 任务需主装穿透）
                # 区别：K2_HUNT v 上限高 (220-300)，K1_HUNT v 上限低 (150-220) 只击穿不引爆
                # M/F/C_HUNT 允许低于门槛 (这些任务不需要主装穿透)
                lo_eff = max(v_lo, v_th) if hunt_task in ("K2", "K1") else v_lo
                hi_eff = max(v_hi, lo_eff + 1.0)
                v_norm[mask] = lo_eff + (hi_eff - lo_eff) * u[mask, 5]
        else:
            for i in range(4):
                mask = (m_ids == i)
                if not np.any(mask): continue
                v_th = CONFIG["V_THRESHOLDS"][i]
                if layer_type == "M_F":
                    v_norm[mask] = CONFIG["V_MIN"] + (v_th - CONFIG["V_MIN"]) * u[mask, 5]
                else:
                    v_norm[mask] = v_th + (CONFIG["V_MAX"] - v_th) * u[mask, 5]

        # ====================================================================
        # Aim-and-Shoot 瞄准式速度方向采样
        #   _HUNT 层：以 BIAS 概率瞄准该任务关键组件，(1-BIAS) 走 AABB 泛化
        #   其他层  ：均匀采 AABB 内靶点 (P1)
        # ====================================================================
        if is_hunt:
            _, _, _, bias_key, sigma_key = self._HUNT_PARAMS[hunt_task]
            centers_arr = self.target_centers[hunt_task]
            bias_p = float(hunt_override.get("aim_bias", CONFIG[bias_key]))
            mask_crit = np.random.rand(n_samples) < bias_p
            n_crit = int(mask_crit.sum())

            target_x = np.empty(n_samples)
            target_y = np.empty(n_samples)
            target_z = np.empty(n_samples)

            if n_crit > 0:
                comp_idx = np.random.randint(
                    0, len(centers_arr), size=n_crit)
                if hunt_task == "C2" and exterior_lateral_shell:
                    preferred_probability = float(
                        CONFIG["C2_FRESH_MAX_Y_CLUSTER_PROB"])
                    preferred_mask = (
                        np.random.rand(n_crit) < preferred_probability)
                    preferred_indices = np.flatnonzero(np.isclose(
                        centers_arr[:, 1],
                        np.max(centers_arr[:, 1]),
                        rtol=0.0,
                        atol=1e-6,
                    ))
                    if preferred_indices.size and preferred_mask.any():
                        comp_idx[preferred_mask] = np.random.choice(
                            preferred_indices,
                            size=int(preferred_mask.sum()),
                            replace=True,
                        )
                centers = centers_arr[comp_idx]
                jitter_sigma = float(hunt_override.get(
                    "target_jitter_cm",
                    10.0 if hunt_override else 20.0,
                ))
                jitter = np.random.normal(0, jitter_sigma, size=(n_crit, 3))
                target_x[mask_crit] = centers[:, 0] + jitter[:, 0]
                target_y[mask_crit] = centers[:, 1] + jitter[:, 1]
                target_z[mask_crit] = centers[:, 2] + jitter[:, 2]

            mask_aabb = ~mask_crit
            n_aabb = int(mask_aabb.sum())
            if n_aabb > 0:
                target_x[mask_aabb] = self.min_aabb[0] + (self.max_aabb[0] - self.min_aabb[0]) * u[mask_aabb, 3]
                target_y[mask_aabb] = self.min_aabb[1] + (self.max_aabb[1] - self.min_aabb[1]) * u[mask_aabb, 4]
                target_z[mask_aabb] = self.min_aabb[2] + (self.max_aabb[2] - self.min_aabb[2]) * u[mask_aabb, 10]

            sigma_aim = np.radians(
                float(hunt_override.get("aim_sigma_deg", CONFIG[sigma_key])))
        else:
            target_x = self.min_aabb[0] + (self.max_aabb[0] - self.min_aabb[0]) * u[:, 3]
            target_y = self.min_aabb[1] + (self.max_aabb[1] - self.min_aabb[1]) * u[:, 4]
            target_z = self.min_aabb[2] + (self.max_aabb[2] - self.min_aabb[2]) * u[:, 10]
            sigma_aim = np.radians(CONFIG["AIM_SIGMA_DEG"])

        sampling_geometry = np.full(n_samples, "radial_lhs", dtype=object)
        if is_hunt and exterior_lateral_shell:
            clearance_lo, clearance_hi = CONFIG[
                "FRESH_ROOT_LATERAL_CLEARANCE_RANGE_CM"]
            tangent_y, tangent_z = CONFIG[
                "FRESH_ROOT_TANGENTIAL_JITTER_CM"]
            clearance = (
                float(clearance_lo) +
                (float(clearance_hi) - float(clearance_lo)) * u[:, 2]
            )
            if hunt_task == "C2":
                # The crew compartment has strongly asymmetric accessible
                # side geometry.  Preserve a minority of left-side proposals
                # for coverage while concentrating expensive high-MC roots in
                # the physically validated right-side corridor.
                right_probability = float(
                    CONFIG["C2_FRESH_RIGHT_SIDE_PROB"])
                left_side = u[:, 0] >= right_probability
            else:
                left_side = u[:, 0] < 0.5
            x = np.where(
                left_side,
                float(self.min_aabb[0]) - clearance,
                float(self.max_aabb[0]) + clearance,
            )
            if hunt_task == "C2":
                crew_y = np.asarray(
                    self.target_centers["C2"][:, 1], dtype=float)
                margin_low, margin_high = CONFIG[
                    "C2_FRESH_CREW_Y_CORRIDOR_MARGIN_CM"]
                corridor_low = max(
                    float(self.min_aabb[1]),
                    float(np.min(crew_y)) - float(margin_low),
                )
                corridor_high = min(
                    float(self.max_aabb[1]),
                    float(np.median(crew_y)) + float(margin_high),
                )
                y = corridor_low + (
                    corridor_high - corridor_low) * u[:, 1]
                z_offset_low, z_offset_high = CONFIG[
                    "C2_FRESH_TARGET_Z_OFFSET_RANGE_CM"]
                z = np.clip(
                    target_z + float(z_offset_low) +
                    (float(z_offset_high) - float(z_offset_low)) * u[:, 10],
                    0.0, float(self.max_aabb[2]),
                )
                sampling_geometry[:] = "fresh_c2_crew_corridor"
            else:
                y = np.clip(
                    target_y + (2.0 * u[:, 1] - 1.0) * float(tangent_y),
                    float(self.min_aabb[1]), float(self.max_aabb[1]),
                )
                z = np.clip(
                    target_z + (2.0 * u[:, 10] - 1.0) * float(tangent_z),
                    0.0, float(self.max_aabb[2]),
                )
                sampling_geometry[:] = "fresh_lateral_shell"
            # 保留左右外壳坐标不动；若切向分量使总半径超过全局上限，
            # 仅同比收缩 y/z，避免重新落回 AABB。
            tangent_norm = np.sqrt(y**2 + z**2)
            tangent_limit = np.sqrt(np.maximum(
                float(CONFIG["RADIUS_MAX_CM"])**2 - x**2, 0.0)) * 0.999
            scale = np.minimum(1.0, tangent_limit / (tangent_norm + 1e-9))
            y *= scale
            z *= scale

        aim_x = target_x - x
        aim_y = target_y - y
        aim_z = target_z - z
        aim_norm = np.sqrt(aim_x**2 + aim_y**2 + aim_z**2) + 1e-9
        aim_dx = aim_x / aim_norm
        aim_dy = aim_y / aim_norm
        aim_dz = aim_z / aim_norm

        fragment_aim_sign = np.ones(n_samples, dtype=float)
        if is_hunt:
            fragment_aim_sign = np.asarray([
                float(CONFIG["HUNT_AXIS_SIGN_BY_MUNITION"][int(m_id)])
                for m_id in m_ids
            ])
        # FRONT 起爆的破片锥沿 -X_B 飞散，因此其机轴/速度方向应背向
        # 真实靶点；REAR 起爆相反。进一步把锥轴旋转一个有效 Taylor
        # 锥角，使真实靶点落在主破片锥面上，而不是落在无破片的锥轴上。
        base_axis = fragment_aim_sign[:, None] * np.column_stack(
            [aim_dx, aim_dy, aim_dz])
        cone_aim_angle_deg = np.zeros(n_samples, dtype=float)
        if is_hunt:
            cone_aim_angle_deg = np.asarray([
                float(CONFIG["HUNT_FRAGMENT_CONE_DEG_BY_MUNITION"][int(m_id)])
                for m_id in m_ids
            ])
            reference = np.tile(np.array([0.0, 0.0, 1.0]), (n_samples, 1))
            near_vertical = np.abs(base_axis[:, 2]) > 0.9
            reference[near_vertical] = np.array([0.0, 1.0, 0.0])
            tangent_1 = np.cross(base_axis, reference)
            tangent_1 /= np.linalg.norm(tangent_1, axis=1, keepdims=True) + 1e-9
            tangent_2 = np.cross(base_axis, tangent_1)
            cone_azimuth = 2.0 * np.pi * u[:, 4]
            cone_tangent = (
                np.cos(cone_azimuth)[:, None] * tangent_1 +
                np.sin(cone_azimuth)[:, None] * tangent_2
            )
            cone_angle = np.radians(cone_aim_angle_deg)
            base_axis = (
                np.cos(cone_angle)[:, None] * base_axis +
                np.sin(cone_angle)[:, None] * cone_tangent
            )
        axis_dx, axis_dy, axis_dz = base_axis.T

        # 在垂直于锥轴方向的切空间加高斯噪声，覆盖锥角模型误差。
        noise = np.random.normal(0, sigma_aim, size=(n_samples, 3))
        # 投影掉沿 aim 的分量，让噪声仅作用于垂直平面 (保持瞄准方向不变)
        proj = noise[:, 0] * axis_dx + noise[:, 1] * axis_dy + noise[:, 2] * axis_dz
        nx = noise[:, 0] - proj * axis_dx
        ny = noise[:, 1] - proj * axis_dy
        nz = noise[:, 2] - proj * axis_dz

        dir_x = axis_dx + nx
        dir_y = axis_dy + ny
        dir_z = axis_dz + nz
        dnorm = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2) + 1e-9
        dir_x /= dnorm
        dir_y /= dnorm
        dir_z /= dnorm

        vx = dir_x * v_norm
        vy = dir_y * v_norm
        vz = dir_z * v_norm

        # 注：原 mask_away 逆飞翻转逻辑已删除 — Aim-and-Shoot 保证速度天然指向车体

        # --- 姿态初始化 (基于实际速度方向) ---
        v_xy = np.sqrt(vx**2 + vy**2)
        pitch_base = np.degrees(np.arctan2(vz, v_xy))
        yaw_base = np.degrees(np.arctan2(vx, vy))

        # HUNT 的方向已经用锥面几何完成探索，只需保留小攻角扰动；普通
        # Phase-1 层仍使用原 ±25° 范围以覆盖广域姿态。
        attitude_jitter = (
            float(CONFIG["HUNT_AOA_JITTER_DEG"]) if is_hunt else 25.0)
        pitch = pitch_base + (-attitude_jitter + 2.0 * attitude_jitter * u[:, 6])
        yaw = yaw_base + (-attitude_jitter + 2.0 * attitude_jitter * u[:, 7])

        if layer_type == "CRITICAL_ROLL":
            # 专用的边界侧倾角敏感加密打靶区
            zone_idx = (u[:, 8] * len(CONFIG["CRITICAL_ROLLS"])).astype(int)
            zone_idx = np.clip(zone_idx, 0, len(CONFIG["CRITICAL_ROLLS"]) - 1)
            rolls = np.zeros(n_samples)
            for iz, c_roll in enumerate(CONFIG["CRITICAL_ROLLS"]):
                zmask = (zone_idx == iz)
                w = CONFIG["CRITICAL_ROLL_WIDTH"]
                rolls[zmask] = c_roll - w + 2*w * u[zmask, 9]
            roll = wrap_angle_deg(rolls)
        elif layer_type == "K2_HUNT":
            # K2_HUNT：roll 限定在 |roll|∈[50,150]° (K2 实测分布带)，正负各 50%
            roll_lo, roll_hi = CONFIG["K2_HUNT_ROLL_RANGE"]
            roll_abs = roll_lo + (roll_hi - roll_lo) * u[:, 8]
            sign = np.where(u[:, 9] < 0.5, -1.0, 1.0)
            roll = wrap_angle_deg(roll_abs * sign)
        elif layer_type == "K1_HUNT":
            # [P1-A] K1_HUNT：roll 限定在 |roll|∈[30,160]° (比 K2 略宽，含前斜方)
            roll_lo, roll_hi = CONFIG["K1_HUNT_ROLL_RANGE"]
            roll_abs = roll_lo + (roll_hi - roll_lo) * u[:, 8]
            sign = np.where(u[:, 9] < 0.5, -1.0, 1.0)
            roll = wrap_angle_deg(roll_abs * sign)
        else:
            # M/F/C_HUNT 与其他层：roll 全域均匀 (M/F/C 任务对 roll 不像 K2 那样敏感)
            roll = -180.0 + 360.0 * u[:, 8]

        sample_ids = self._allocate_sample_ids(n_samples, "p1")
        split_roles = np.asarray([_reference_split_for_root(sid) for sid in sample_ids])
        return pd.DataFrame({
            "x": x, "y": y, "z": z,
            "vx": vx, "vy": vy, "vz": vz,
            "target_x": target_x, "target_y": target_y, "target_z": target_z,
            "fragment_aim_sign": fragment_aim_sign,
            "cone_aim_angle_deg": cone_aim_angle_deg,
            "sampling_geometry": sampling_geometry,
            "pitch": pitch, "roll": roll, "yaw": yaw,
            "m_id": m_ids,
            "layer_type": layer_type,    # 第三轮新增：保留层段标签供下游分析
            "sample_id": sample_ids,
            "root_seed_id": sample_ids.copy(),
            "parent_id": "",
            "crawl_stage": 0,
            "split_role": split_roles,
            "frame_version": CONFIG["FRAME_CONVENTION_VERSION"],
            "dataset_schema": CONFIG["DATASET_SCHEMA"],
            "label_mc_min_replicates": int(CONFIG["LABEL_MC_MIN_REPLICATES"]),
            "label_mc_max_replicates": int(CONFIG["LABEL_MC_MAX_REPLICATES"]),
        })

    def _apply_phase1_filters_and_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """Phase 1 的物理过滤与初始 loss_weight 计算。"""
        if df.empty:
            return df

        mask_z_safe = df["z"] >= 0

        r_mun = 15.0
        mask_collide = (df["x"] > self.min_aabb[0]-r_mun) & (df["x"] < self.max_aabb[0]+r_mun) & \
                       (df["y"] > self.min_aabb[1]-r_mun) & (df["y"] < self.max_aabb[1]+r_mun) & \
                       (df["z"] > self.min_aabb[2]-r_mun) & (df["z"] < self.max_aabb[2]+r_mun)

        df = df[mask_z_safe & (~mask_collide)].copy()
        if df.empty:
            return df

        h_x = np.cos(np.radians(df["pitch"])) * np.sin(np.radians(df["yaw"]))
        h_y = np.cos(np.radians(df["pitch"])) * np.cos(np.radians(df["yaw"]))
        h_z = np.sin(np.radians(df["pitch"]))

        v_norms = np.sqrt(df["vx"]**2 + df["vy"]**2 + df["vz"]**2)
        dir_x, dir_y, dir_z = df["vx"]/v_norms, df["vy"]/v_norms, df["vz"]/v_norms

        cos_aoa = np.clip(dir_x*h_x + dir_y*h_y + dir_z*h_z, -1.0, 1.0)
        aoa_deg = np.degrees(np.arccos(cos_aoa))

        p_accept = np.ones(len(df))
        mask_tail = (aoa_deg > 15.0) & (aoa_deg <= 30.0)
        mask_fail = (aoa_deg > 30.0)

        sigma = 5.0
        p_accept[mask_tail] = np.exp(-((aoa_deg[mask_tail] - 15.0)**2) / (2 * sigma**2))
        p_accept[mask_fail] = 0.0

        passed = np.random.rand(len(p_accept)) < p_accept
        df = df[passed].copy()
        if df.empty:
            return df

        p_act = p_accept[passed]
        df = _assign_sampling_weight_components(df, p_act)
        df["is_crawled"] = 0
        if "sampling_phase" not in df.columns:
            df["sampling_phase"] = "phase1"
        return df

    def _build_phase1_blocks_for_munition(self, munition_id: int, target_count: int) -> pd.DataFrame:
        """为单个弹型构建 Phase 1 候选池。"""
        if target_count <= 0:
            return pd.DataFrame()

        internal_target = max(target_count * self.k_oversample, target_count)
        layer_sizes = _allocate_counts(internal_target, CONFIG["PER_MUNITION_LAYERS"][munition_id])
        blocks = []
        for layer_type, size in layer_sizes.items():
            if size > 0:
                blocks.append(self._generate_lhs_batch(size, layer_type, force_m_id=munition_id))
        return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()

    def generate_phase1_replenishment(self, munition_id: int, num_needed: int) -> pd.DataFrame:
        """弹型独立的 Phase 1 风格回填，只补本弹型缺口。"""
        if num_needed <= 0:
            return pd.DataFrame()

        df = self._build_phase1_blocks_for_munition(munition_id, num_needed)
        df = self._apply_phase1_filters_and_weights(df)
        if df.empty:
            return df

        if len(df) > num_needed:
            df = df.sample(n=num_needed, replace=False, random_state=CONFIG["RANDOM_SEED"]).reset_index(drop=True)
        return df

    def generate_phase_1(self, target_count: int) -> pd.DataFrame:
        """Phase 1：按弹型等量目标生成基础盘，不再让 Heavy 吃掉其他弹型缺口。"""
        phase1_quota = _allocate_counts(target_count, CONFIG["MUNITION_BUDGET"])
        print(f"[Phase 1] Per-munition phase-1 目标配额: {_format_munition_count_dict(phase1_quota)}")

        blocks = []
        for munition_id, mun_quota in phase1_quota.items():
            block = self._build_phase1_blocks_for_munition(munition_id, mun_quota)
            if not block.empty:
                blocks.append(block)

        df = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()
        df = self._apply_phase1_filters_and_weights(df)
        if df.empty:
            return df

        df, cap_stats = _cap_dataframe_by_munition(df, phase1_quota, CONFIG["RANDOM_SEED"], keep_phase2_first=False)
        print("[Phase 1] 物理过滤后按弹型独立裁剪：")
        for m_id in range(4):
            stat = cap_stats.get(m_id, {"available": 0, "kept": 0, "trimmed": 0})
            print(f"  m_id={m_id} available={stat['available']:>6} kept={stat['kept']:>6} trimmed={stat['trimmed']:>6}")

        return df.sample(frac=1.0, random_state=CONFIG["RANDOM_SEED"]).reset_index(drop=True)

    def discover_fresh_target_roots(self, existing_pool: pd.DataFrame,
                                    munition_id: int,
                                    target_col: str,
                                    seed_th: float,
                                    valid_th: float,
                                    desired_seed_roots: int,
                                    desired_strict_roots: int,
                                    required_split_role: Optional[str] = "train",
                                    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Iteratively discover independent roots for one rare target.

        Production top-off keeps the default ``required_split_role="train"``.
        Standalone challenge builders may pass ``None`` to use a disjoint root
        namespace without discarding 20% of candidates solely due to the
        production split hash.
        """
        target_layer = (
            "K2_HUNT" if target_col == "K2_prob" else
            "K1_HUNT" if target_col.startswith("K") else
            "M_HUNT" if target_col.startswith("M") else
            "F_HUNT" if target_col.startswith("F") else
            "C2_HUNT" if target_col.startswith("C2") else
            "C_HUNT"
        )

        def _root_counts(pool: pd.DataFrame) -> Tuple[int, int]:
            if pool.empty:
                return 0, 0
            scoped = pool
            if required_split_role is not None and "split_role" in scoped.columns:
                scoped = scoped[
                    scoped["split_role"].astype(str) == required_split_role]
            if scoped.empty:
                return 0, 0
            seed_roots = int(scoped.loc[
                _target_seed_mask(scoped, target_col, seed_th, valid_th),
                "root_seed_id",
            ].astype(str).nunique())
            strict_roots = int(scoped.loc[
                _target_valid_mask(scoped, target_col, valid_th),
                "root_seed_id",
            ].astype(str).nunique())
            return seed_roots, strict_roots

        seed_before, strict_before = _root_counts(existing_pool)
        fresh_batches = []
        combined_pool = existing_pool
        requested_total = 0
        simulated_total = 0
        rounds = 0
        is_c2_target = str(target_col).startswith("C2")
        max_candidates = int(CONFIG[
            "C2_FRESH_ROOT_MAX_CANDIDATES"
            if is_c2_target
            else "FRESH_ROOT_MAX_CANDIDATES_PER_TASK"
        ])
        max_rounds = int(CONFIG[
            "C2_FRESH_ROOT_MAX_ROUNDS"
            if is_c2_target else "FRESH_ROOT_MAX_ROUNDS"
        ])

        seed_count, strict_count = seed_before, strict_before
        while ((seed_count < desired_seed_roots or
                strict_count < desired_strict_roots) and
               requested_total < max_candidates and rounds < max_rounds):
            remaining = max_candidates - requested_total
            missing_strict = max(desired_strict_roots - strict_count, 0)
            request_n = min(
                remaining,
                max(
                    int(CONFIG["FRESH_ROOT_BATCH_SIZE"]),
                    missing_strict * int(CONFIG["FRESH_ROOT_CANDIDATE_MULTIPLIER"]),
                ),
            )
            if request_n <= 0:
                break
            rounds += 1
            requested_total += request_n
            print(f"  [FreshRoot][{rounds}/{max_rounds}] m_id={munition_id} "
                  f"{target_col}: seed roots={seed_count}/{desired_seed_roots}, "
                  f"strict roots={strict_count}/{desired_strict_roots}; "
                  f"探测 {request_n} 个 {target_layer} 独立候选。")

            fresh_inputs = self._generate_lhs_batch(
                request_n, target_layer, force_m_id=munition_id,
                exterior_lateral_shell=True)
            fresh_inputs = self._apply_phase1_filters_and_weights(fresh_inputs)
            if fresh_inputs.empty:
                continue
            fresh_inputs["sampling_phase"] = "phase2_fresh_root"
            if required_split_role is not None:
                fresh_inputs = fresh_inputs[
                    fresh_inputs["split_role"].astype(str) ==
                    required_split_role
                ].copy()
            if fresh_inputs.empty:
                continue
            fresh_results = self.run_simulation_batch(fresh_inputs)
            simulated_total += int(len(fresh_results))
            if fresh_results.empty:
                continue
            fresh_batches.append(fresh_results)
            combined_pool = (
                fresh_results.reset_index(drop=True).copy()
                if combined_pool.empty else
                pd.concat([combined_pool, fresh_results], ignore_index=True)
            )
            seed_count, strict_count = _root_counts(combined_pool)

        fresh = (
            pd.concat(fresh_batches, ignore_index=True)
            if fresh_batches else existing_pool.iloc[0:0].copy()
        )
        stats = {
            "target_layer": target_layer,
            "split_scope": (
                required_split_role
                if required_split_role is not None else "all_independent_roots"
            ),
            "rounds": int(rounds),
            "raw_candidates_requested": int(requested_total),
            "train_candidates_simulated": int(simulated_total),
            "seed_roots_before": int(seed_before),
            "strict_roots_before": int(strict_before),
            "seed_roots_after": int(seed_count),
            "strict_roots_after": int(strict_count),
            "desired_seed_roots": int(desired_seed_roots),
            "desired_strict_roots": int(desired_strict_roots),
            "candidate_budget_exhausted": bool(
                requested_total >= max_candidates and
                (seed_count < desired_seed_roots or strict_count < desired_strict_roots)
            ),
        }
        return fresh, stats

    def crawl_boundary_stage_2(self, df_seeds: pd.DataFrame, num_needed: int,
                               sigma_scale: float = 1.0) -> pd.DataFrame:
        """核心策略二：在弧度标量空间的动态步长高斯扰动与严格物理截断

        Args:
            df_seeds: 种子样本集
            num_needed: 目标生成数量 (会经超采率膨胀以对冲丢弃)
            sigma_scale: 步长缩放系数，多阶段爬行中按 CRAWL_SIGMA_DECAY 逐阶段衰减，
                         实现 exploration→exploitation 的退火式搜索
        """
        if df_seeds.empty or num_needed <= 0:
            return pd.DataFrame()

        # 修复 #8：先按超采率扩张候选池，对冲后续物理掩码 + 软攻角拒止的复合丢弃率
        n_internal_requested = int(num_needed * CONFIG["CRAWL_OVERSAMPLE"])
        if "sample_id" not in df_seeds.columns:
            df_seeds = df_seeds.copy()
            df_seeds["sample_id"] = self._allocate_sample_ids(len(df_seeds), "legacy-parent")
        if "root_seed_id" not in df_seeds.columns:
            df_seeds = df_seeds.copy()
            df_seeds["root_seed_id"] = df_seeds["sample_id"].astype(str)
        root_col = "root_seed_id"
        seed_roots = df_seeds.drop_duplicates(root_col, keep="first").reset_index(drop=True)
        max_internal = (
            len(seed_roots) * int(CONFIG["MAX_CHILDREN_PER_ROOT_PER_STAGE"])
        )
        n_internal = min(n_internal_requested, max_internal)
        if n_internal < n_internal_requested:
            print(f"    [Diversity Guard] 本阶段候选由 {n_internal_requested} 限制为 "
                  f"{n_internal}（{len(seed_roots)} 个独立 root × 每 root 最多 "
                  f"{CONFIG['MAX_CHILDREN_PER_ROOT_PER_STAGE']} 个）。")
        if n_internal <= 0:
            return pd.DataFrame()
        print(f"[Crawler] Phase 2: 基于 {len(df_seeds)} 枚黄金种子扩散生成 {num_needed} 个富化扰动点 (内部超采到 {n_internal}, σ×{sigma_scale:.3f})...")

        # Round-robin over unique roots.  Sampling rows with replacement here was
        # the direct cause of several thousand descendants from one family.
        order = np.tile(
            np.random.permutation(len(seed_roots)),
            int(np.ceil(n_internal / max(len(seed_roots), 1))),
        )[:n_internal]
        crawled = seed_roots.iloc[order].copy()
        crawled.reset_index(drop=True, inplace=True)
        if "sample_id" not in crawled.columns:
            # Compatibility only for diagnostics; stage-0 dataset generation
            # always enters through _generate_lhs_batch and has full lineage.
            crawled["sample_id"] = self._allocate_sample_ids(n_internal, "legacy-parent")
        parent_ids = crawled["sample_id"].astype(str).to_numpy(copy=True)
        if "root_seed_id" not in crawled.columns:
            crawled["root_seed_id"] = parent_ids
        crawled["parent_id"] = parent_ids
        crawled["sample_id"] = self._allocate_sample_ids(n_internal, "crawl")
        prior_stage = (
            crawled["crawl_stage"].fillna(0).astype(int).to_numpy()
            if "crawl_stage" in crawled.columns else np.zeros(n_internal, dtype=int)
        )
        crawled["crawl_stage"] = prior_stage + 1
        if "split_role" not in crawled.columns:
            crawled["split_role"] = [
                _reference_split_for_root(str(root)) for root in crawled["root_seed_id"]
            ]
        crawled["frame_version"] = CONFIG["FRAME_CONVENTION_VERSION"]
        crawled["dataset_schema"] = CONFIG["DATASET_SCHEMA"]
        crawled["label_mc_min_replicates"] = int(CONFIG["LABEL_MC_MIN_REPLICATES"])
        crawled["label_mc_max_replicates"] = int(CONFIG["LABEL_MC_MAX_REPLICATES"])
        crawled["sampling_phase"] = "phase2_crawl"

        m_ids = crawled["m_id"].values
        v_ths = np.array([CONFIG["V_THRESHOLDS"][m] for m in m_ids])

        v_x, v_y, v_z = crawled["vx"].values, crawled["vy"].values, crawled["vz"].values
        v_norm = np.sqrt(v_x**2 + v_y**2 + v_z**2) + 1e-9

        dist_to_th = np.abs(v_norm - v_ths) / CONFIG["DIST_NORM_VELOCITY"]
        sigma_rad = CONFIG["NOISE_SIGMA_MIN"] + (CONFIG["NOISE_SIGMA_MAX"] - CONFIG["NOISE_SIGMA_MIN"]) * np.clip(dist_to_th, 0, 1)
        # 多阶段退火：σ 随阶段衰减，后期聚焦精细化搜索
        sigma_rad = sigma_rad * sigma_scale

        # ----------------------------------------------------------------
        # 修复 #1 / #2：将速度方向解耦为球面角 (v_pitch, v_yaw) 后独立加噪，
        # 杜绝原版本"速度方向恒定只改模长"的伪扰动；AoA 校验也自动用上新方向。
        # ----------------------------------------------------------------
        v_xy = np.sqrt(v_x**2 + v_y**2) + 1e-9
        v_pitch_rad = np.arctan2(v_z, v_xy)
        v_yaw_rad   = np.arctan2(v_x, v_y)

        v_pitch_rad += np.random.normal(0, sigma_rad)
        v_yaw_rad   += np.random.normal(0, sigma_rad)
        # 俯仰物理上限 ±90°，越界会让 cos 反号导致方向 x/y 分量反向
        v_pitch_rad = np.clip(v_pitch_rad, -np.pi/2, np.pi/2)
        v_yaw_rad   = wrap_angle_rad(v_yaw_rad)

        # 机体姿态 (独立于速度方向) 同步加噪
        pitch_rad = np.radians(crawled["pitch"].values)
        yaw_rad   = np.radians(crawled["yaw"].values)
        roll_rad  = np.radians(crawled["roll"].values)

        pitch_rad += np.random.normal(0, sigma_rad)
        yaw_rad   += np.random.normal(0, sigma_rad)
        roll_rad  += np.random.normal(0, sigma_rad)

        pitch_rad = wrap_angle_rad(pitch_rad)
        yaw_rad   = wrap_angle_rad(yaw_rad)
        roll_rad  = wrap_angle_rad(roll_rad)

        v_norm += np.random.normal(0, sigma_rad * 20.0)
        v_norm = np.clip(v_norm, CONFIG["V_MIN"], CONFIG["V_MAX"])

        crawled["x"] += np.random.normal(0, 10.0, n_internal)
        crawled["y"] += np.random.normal(0, 10.0, n_internal)
        crawled["z"] += np.random.normal(0, 10.0, n_internal)

        # 用扰动后的速度方向单位向量重建速度矢量 (而非沿用原始方向)
        dir_x = np.cos(v_pitch_rad) * np.sin(v_yaw_rad)
        dir_y = np.cos(v_pitch_rad) * np.cos(v_yaw_rad)
        dir_z = np.sin(v_pitch_rad)

        crawled["vx"] = dir_x * v_norm
        crawled["vy"] = dir_y * v_norm
        crawled["vz"] = dir_z * v_norm

        crawled["pitch"] = np.degrees(pitch_rad)
        crawled["yaw"] = np.degrees(yaw_rad)
        crawled["roll"] = np.degrees(roll_rad)

        # ====================================================================
        # 新增：前置物理法则断言 (过滤所有因高斯扰动而跌出物理合法边界的脏点)
        # ====================================================================

        # 1. 彻底禁止地底点 (抛弃 abs 翻转，直接剔除跌入地底的衍生点)
        mask_z_safe = crawled["z"] >= 0

        # 2. 避免 AABB 穿模幽灵点 (禁止将衍生点生成在装甲模型内部)
        r_mun = 15.0
        mask_collide = (crawled["x"] > self.min_aabb[0]-r_mun) & (crawled["x"] < self.max_aabb[0]+r_mun) & \
                       (crawled["y"] > self.min_aabb[1]-r_mun) & (crawled["y"] < self.max_aabb[1]+r_mun) & \
                       (crawled["z"] > self.min_aabb[2]-r_mun) & (crawled["z"] < self.max_aabb[2]+r_mun)
        mask_out_aabb = ~mask_collide

        # 3. 修复 #7：与 Phase 1 一致的软攻角拒止 (15-30° 高斯软概率衰减，>30° 硬切)
        #    避免两阶段在 AoA 上出现"断崖式"分布差异，影响下游 NN 对边界角度的学习
        h_x = np.cos(pitch_rad) * np.sin(yaw_rad)
        h_y = np.cos(pitch_rad) * np.cos(yaw_rad)
        h_z = np.sin(pitch_rad)

        cos_aoa = np.clip(dir_x*h_x + dir_y*h_y + dir_z*h_z, -1.0, 1.0)
        aoa_deg = np.degrees(np.arccos(cos_aoa))

        soft_th = CONFIG["AOA_SOFT_TAIL_DEG"]
        hard_th = CONFIG["AOA_HARD_FAIL_DEG"]
        sigma_aoa = CONFIG["AOA_SOFT_SIGMA"]

        p_accept = np.ones(len(crawled))
        mask_tail = (aoa_deg > soft_th) & (aoa_deg <= hard_th)
        mask_fail = (aoa_deg > hard_th)
        p_accept[mask_tail] = np.exp(-((aoa_deg[mask_tail] - soft_th)**2) / (2 * sigma_aoa**2))
        p_accept[mask_fail] = 0.0
        mask_aoa_safe = np.random.rand(len(p_accept)) < p_accept

        # 4. 逆向飞弹防护：R25 起以当前样本瞄准点为参考，而非默认坐标原点。
        #    旧数据若缺少 target_* 元数据，则退化为原点口径以保持兼容。
        if all(c in crawled.columns for c in ["target_x", "target_y", "target_z"]):
            aim_x = crawled["target_x"] - crawled["x"]
            aim_y = crawled["target_y"] - crawled["y"]
            aim_z = crawled["target_z"] - crawled["z"]
        else:
            aim_x = -crawled["x"]
            aim_y = -crawled["y"]
            aim_z = -crawled["z"]
        dot_vd = aim_x*crawled["vx"] + aim_y*crawled["vy"] + aim_z*crawled["vz"]
        # 对 FRONT 起爆 HUNT 样本，速度/机轴背向靶点才会让 -X_B 主破片锥
        # 指向靶点。保留该符号后，Phase 2 不会误删物理上正确的后向瞄准后代。
        if "fragment_aim_sign" in crawled.columns:
            aim_sign = crawled["fragment_aim_sign"].fillna(1.0).astype(float)
        else:
            aim_sign = 1.0
        mask_towards = aim_sign * dot_vd >= 0

        # 5. 修复 #5：半径上限校验 — 防止位置高斯扰动让起爆点漂出 LHS 物理域
        #    (原本只校验 AABB 内侧，但忽略了向外漂超过 RADIUS_MAX_CM 的脏点)
        r_new = np.sqrt(crawled["x"].values**2 + crawled["y"].values**2 + crawled["z"].values**2)
        mask_radius = r_new <= CONFIG["RADIUS_MAX_CM"]

        # 综合合法性掩码
        valid_mask = mask_z_safe & mask_out_aabb & mask_aoa_safe & mask_towards & mask_radius

        crawled_valid = crawled[valid_mask].copy()
        crawled_valid = _assign_sampling_weight_components(
            crawled_valid, p_accept[np.asarray(valid_mask, dtype=bool)])
        crawled_valid["is_crawled"] = 1

        dropped = n_internal - len(crawled_valid)
        if dropped > 0:
            print(f"    [Physics Guard] 前置截断丢弃了 {dropped} 个违反物理底线的脏点 (剩余 {len(crawled_valid)} 个)")

        # 修复 #8：超采补偿后若仍超出目标预算，按需修剪到 num_needed (节省下游仿真算力)
        if len(crawled_valid) > num_needed:
            crawled_valid = crawled_valid.sample(n=num_needed).reset_index(drop=True)
            print(f"    [Budget] 物理合法点充裕，修剪至目标量 {num_needed}")

        return crawled_valid

    # ========================================================================
    # 改进版方案 B：多阶段梯度引导爬行 (Staged Guided Crawl)
    # ========================================================================
    def _select_boundary_seeds(self, pool: pd.DataFrame, target_prob_col: str,
                                k: int) -> pd.DataFrame:
        """从候选池中选 top-k 边界带种子。

        设计要点 (对用户原方案的三点修正)：
        1. 不选 prob 最大点，而选 prob 落在 [0.3, 0.7] 边界带的点 — 这些对分类器
           决策面的信息增益最大 (active learning 的核心洞察)
        2. top-k 而非 top-1，避免模式坍缩到单个 K2 簇
        3. 在标准化特征空间做邻域去重，避免"富者愈富"的早熟收敛
        """
        if pool.empty:
            return pd.DataFrame()

        lo, hi = CONFIG["CRAWL_BOUNDARY_BAND"]
        p = _target_score(pool, target_prob_col)

        # 优先级分数：边界带内点按 -|p-0.5| 排序 (越接近判决面分数越高)
        # 带外点按 (p >= hi ? 1 : 0) - |p-0.5| 退化排序 (正样本比负样本优先)
        in_band = (p >= lo) & (p <= hi)
        # 构造复合分数：带内点 [1.0, 2.0]，正样本 [0.0, 1.0)，负样本 < 0
        score = np.where(in_band,
                         2.0 - np.abs(p - 0.5),                    # 边界带：越近 0.5 越高
                         np.where(p >= hi, 1.0 - np.abs(p - 0.5),  # 正样本带外：次优
                                  -np.abs(p - 0.5)))               # 纯负样本：最低

        pool = pool.copy()
        pool["_seed_score"] = score
        # 每个 lineage root 先只保留一枚最优候选，再做空间去重。这样 top-k
        # 表示 k 个独立仿真家族，而不是同一祖先的 k 个近邻后代。
        candidates = pool.sort_values("_seed_score", ascending=False)
        if "root_seed_id" in candidates.columns:
            candidates = candidates.drop_duplicates("root_seed_id", keep="first")
        candidates = candidates.head(k * 5).reset_index(drop=True)

        if candidates.empty:
            return pd.DataFrame()

        # --- 标准化特征空间下的邻域去重 ---
        # 特征：归一化后的位置 + 速度方向 + 姿态角 (全部无量纲)
        feat = np.column_stack([
            candidates["x"].values / CONFIG["RADIUS_MAX_CM"],
            candidates["y"].values / CONFIG["RADIUS_MAX_CM"],
            candidates["z"].values / CONFIG["RADIUS_MAX_CM"],
            candidates["vx"].values / CONFIG["V_MAX"],
            candidates["vy"].values / CONFIG["V_MAX"],
            candidates["vz"].values / CONFIG["V_MAX"],
            candidates["roll"].values / 180.0,
        ])

        selected_idx = []
        dedup_r = CONFIG["CRAWL_NEIGHBOR_DEDUP_R"] * 0.05  # 归一化后的等效半径
        for i in range(len(candidates)):
            if len(selected_idx) >= k:
                break
            if not selected_idx:
                selected_idx.append(i)
                continue
            # 与已选种子计算最小距离
            dists = np.linalg.norm(feat[selected_idx] - feat[i], axis=1)
            if dists.min() > dedup_r:
                selected_idx.append(i)

        selected = candidates.iloc[selected_idx].drop(columns=["_seed_score"]).reset_index(drop=True)
        selected_p = _target_score(selected, target_prob_col)
        in_band_count = int(((selected_p >= lo) & (selected_p <= hi)).sum())
        print(f"    [SeedSel] 从 {len(pool)} 候选中选出 {len(selected)} 个独立 root 种子 "
              f"(边界带内 {in_band_count}/{len(selected)}，家族+空间去重后)")
        return selected

    def crawl_multistage_guided(self, df_seeds_init: pd.DataFrame,
                                 num_target: int,
                                 target_prob_col: str) -> pd.DataFrame:
        """多阶段梯度引导爬行：分 N 阶段，每阶段从历史全池选边界带 top-k 种子扩散。

        相比单轮爬行的三点升级：
        - 分阶段退火：σ 按 CRAWL_SIGMA_DECAY 递减，前期广搜后期精搜
        - 迭代式引导：后一阶段的种子来自前面所有阶段累积的结果 (而非只用 Phase 1 种子)
        - 边界带聚焦：种子选择偏好 K2_prob≈0.5 的边界点，信息增益最大

        Args:
            df_seeds_init: Phase 1 提取的初始黄金种子 (宽松阈值)
            num_target: 本分支爬行的最终目标产出数量 (未经 VALID_PROB_STRICT 过滤前的候选池)
            target_prob_col: 目标概率列名，如 "K2_prob" 或 "C2_prob"
        """
        n_stages = CONFIG["CRAWL_N_STAGES"]
        topk = CONFIG["CRAWL_TOPK_PER_STAGE"]
        decay = CONFIG["CRAWL_SIGMA_DECAY"]
        n_per_stage = max(num_target // n_stages, 1)

        print(f"\n  [MultiStage] 启动 {n_stages} 阶段梯度引导爬行 (目标列: {target_prob_col})")
        print(f"  [MultiStage] 每阶段产出 ~{n_per_stage} 枚，top-k 种子={topk}，σ 衰减={decay}")

        accumulated = []          # 每阶段真实仿真后的结果 (用于下阶段选种)
        current_seeds = df_seeds_init
        sigma_scale = 1.0

        for stage in range(n_stages):
            if current_seeds.empty:
                print(f"  [MultiStage][Stage {stage+1}/{n_stages}] 种子池枯竭，提前终止")
                break

            print(f"\n  [MultiStage][Stage {stage+1}/{n_stages}] 种子={len(current_seeds)}，σ_scale={sigma_scale:.3f}")

            # --- 本阶段扩散 + 仿真 ---
            df_crawled = self.crawl_boundary_stage_2(current_seeds, n_per_stage,
                                                     sigma_scale=sigma_scale)
            if df_crawled.empty:
                print(f"  [MultiStage][Stage {stage+1}] 物理过滤后无合法扰动点，跳过")
                sigma_scale *= decay
                continue

            res = self.run_simulation_batch(df_crawled)
            if res.empty:
                sigma_scale *= decay
                continue

            # 统计本阶段命中
            target_score = _target_score(res, target_prob_col)
            hits = int(_target_valid_mask(res, target_prob_col, CONFIG["VALID_PROB_STRICT"]).sum())
            band_hits = int(((target_score >= CONFIG["CRAWL_BOUNDARY_BAND"][0]) &
                             (target_score <= CONFIG["CRAWL_BOUNDARY_BAND"][1])).sum())
            print(f"  [MultiStage][Stage {stage+1}] 产出 {len(res)} 枚："
                  f"严格命中 {hits}，边界带 {band_hits}")

            accumulated.append(res)

            # --- 从累积全池中为下一阶段选种子 (关键：不是只用本阶段) ---
            if stage < n_stages - 1:
                full_pool = pd.concat(accumulated, ignore_index=True)
                current_seeds = self._select_boundary_seeds(full_pool, target_prob_col, topk)

            sigma_scale *= decay

        if not accumulated:
            return pd.DataFrame()

        merged = pd.concat(accumulated, ignore_index=True)
        print(f"\n  [MultiStage] 完成：{n_stages} 阶段累计 {len(merged)} 枚，"
              f"其中严格命中 "
              f"{int(_target_valid_mask(merged, target_prob_col, CONFIG['VALID_PROB_STRICT']).sum())} 枚")
        return merged

    def run_simulation_batch(self, df_inputs: pd.DataFrame) -> pd.DataFrame:
        """投递至装甲车物理 DamageEngine 进行高并发测算并回收结果"""
        if df_inputs.empty:
            return pd.DataFrame()

        rows = df_inputs.to_dict('records')
        # 修复 #9：tasks 不再携带 components / plates，模型由 worker initializer 一次性载入
        all_tasks = [(i, row) for i, row in enumerate(rows)]
        checkpoint_namespace = str(
            df_inputs.attrs.get("simulation_checkpoint_namespace", "")
        ).strip()
        checkpoint_enabled = bool(
            checkpoint_namespace
            and CONFIG.get("PHASE1_CHECKPOINT_ENABLED", False)
        )
        checkpoint_dir = None
        checkpoint_interval = int(CONFIG.get(
            "PHASE1_CHECKPOINT_INTERVAL", 1000))
        results_by_index: Dict[int, Dict[str, Any]] = {}
        checkpoint_buffer: List[Tuple[int, Dict[str, Any]]] = []
        if checkpoint_enabled:
            identity = _phase1_checkpoint_identity(
                df_inputs, checkpoint_namespace)
            checkpoint_dir = _prepare_phase1_checkpoint(identity)
            results_by_index = _load_simulation_checkpoint(
                checkpoint_dir, len(all_tasks))
            if results_by_index:
                print(
                    f"  [Engine] 已从 {checkpoint_namespace} 断点恢复 "
                    f"{len(results_by_index)}/{len(all_tasks)} 枚。")
        tasks = [
            task for task in all_tasks
            if int(task[0]) not in results_by_index
        ]

        def _flush_checkpoint() -> None:
            if checkpoint_dir is None or not checkpoint_buffer:
                return
            _write_simulation_checkpoint_part(
                checkpoint_dir, checkpoint_buffer)
            checkpoint_buffer.clear()

        def _record_result(
                task_index: int,
                result: Dict[str, Any]) -> None:
            task_index = int(task_index)
            if task_index in results_by_index:
                raise RuntimeError(
                    f"Simulation task {task_index} completed twice.")
            results_by_index[task_index] = result
            if checkpoint_dir is not None:
                checkpoint_buffer.append((task_index, result))
                if len(checkpoint_buffer) >= checkpoint_interval:
                    _flush_checkpoint()
            completed = len(results_by_index)
            if completed > 0 and completed % 1000 == 0:
                print(
                    f"  [Engine] 极速仿真推演中... "
                    f"{completed:05d}/{len(all_tasks)}")

        # 使用 max(1, count-1) 避免把系统的IO与心跳压溃
        n_workers = max(1, multiprocessing.cpu_count() - 1)

        t0 = time.time()
        try:
            if tasks:
                with ProcessPoolExecutor(
                    max_workers=n_workers,
                    initializer=_init_worker,
                    initargs=(self.components, self.plates),
                ) as executor:
                    futures = {
                        executor.submit(
                            _process_single_encounter, task): int(task[0])
                        for task in tasks
                    }
                    for future in as_completed(futures):
                        _record_result(
                            futures[future], future.result())
        except (PermissionError, OSError) as exc:
            print(f"  [Engine] 多进程不可用，回退到单进程序列仿真: {exc}")
            _flush_checkpoint()
            _init_worker(self.components, self.plates)
            for task in all_tasks:
                if int(task[0]) in results_by_index:
                    continue
                _record_result(
                    int(task[0]), _process_single_encounter(task))
        finally:
            _flush_checkpoint()

        missing = sorted(
            set(range(len(all_tasks))) - set(results_by_index))
        if missing:
            raise RuntimeError(
                "Simulation batch finished with missing task indices: "
                f"count={len(missing)}, preview={missing[:10]}")
        results = [
            results_by_index[index]
            for index in range(len(all_tasks))
        ]
        if checkpoint_dir is not None:
            _mark_phase1_checkpoint_complete(
                checkpoint_dir, len(results))
        print(f"  [Engine] 批次仿真完成，共处理 {len(results)} 枚，耗时 {time.time()-t0:.1f}s")
        return pd.DataFrame(results)


def _exact_level_is_required(
        munition_id: int, task: str, level: int) -> bool:
    if int(level) == 0:
        return True
    return bool(
        CONFIG["ORDINAL_APPLICABILITY"][int(munition_id)][
            str(task)][int(level) - 1])


def _exact_level_support(
        frame: pd.DataFrame,
        split_role: str,
        munition_id: int,
        task: str,
        level: int,
) -> Tuple[int, int]:
    rows = frame[
        (frame["split_role"].astype(str) == str(split_role))
        & (frame["munition_id"].astype(int) == int(munition_id))
        & (frame[f"{task}_level"].astype(int) == int(level))
    ]
    return int(len(rows)), int(
        rows["root_seed_id"].astype(str).nunique())


def _minimum_total_exact_level_support() -> Tuple[int, int]:
    """Supply needed before whole-root train/val/test allocation.

    Two per-evaluation-cell root-cap buffers prevent a final root move from
    consuming the rows reserved for training when a family contributes more
    than one exact-level row.
    """
    minimum_rows = (
        int(CONFIG["MIN_TRAIN_EXACT_LEVEL_ROWS"])
        + 2 * int(CONFIG["MIN_EVAL_EXACT_LEVEL_ROWS"])
        + 2 * int(CONFIG["MAX_POSITIVE_ROWS_PER_ROOT_PER_CELL"])
    )
    minimum_roots = (
        int(CONFIG["MIN_TRAIN_LEVEL_ROOTS"])
        + 2 * int(CONFIG["MIN_EVAL_EXACT_LEVEL_ROOTS"])
    )
    return minimum_rows, minimum_roots


def _level2_total_support_deficits(
        frame: pd.DataFrame) -> List[Dict[str, Any]]:
    """Report applicable L2 cells lacking allocatable total evidence."""
    minimum_rows, minimum_roots = _minimum_total_exact_level_support()
    deficits = []
    for munition_id in range(4):
        munition_mask = (
            frame["munition_id"].astype(int) == int(munition_id))
        for task in ("K", "M", "F", "C"):
            if not _exact_level_is_required(munition_id, task, 2):
                continue
            rows = frame[
                munition_mask
                & (frame[f"{task}_level"].astype(int) == 2)
            ]
            row_count = int(len(rows))
            root_count = int(
                rows["root_seed_id"].astype(str).nunique())
            row_deficit = max(minimum_rows - row_count, 0)
            root_deficit = max(minimum_roots - root_count, 0)
            if row_deficit or root_deficit:
                deficits.append({
                    "munition_id": int(munition_id),
                    "task": str(task),
                    "probability_column": f"{task}2_prob",
                    "rows": row_count,
                    "root_families": root_count,
                    "minimum_rows": minimum_rows,
                    "minimum_root_families": minimum_roots,
                    "row_deficit": row_deficit,
                    "root_deficit": root_deficit,
                })
    return sorted(
        deficits,
        key=lambda item: (
            -int(item["row_deficit"]),
            -int(item["root_deficit"]),
            int(item["munition_id"]),
            str(item["task"]),
        ),
    )


def _level2_support_topoff_budget(
        row_deficit: int, root_deficit: int) -> Tuple[int, int]:
    """Return a production-sized C2 request and a rare-cell retry budget.

    A 256-row request is split across five crawl stages and then loses rows to
    geometry/AoA guards, yielding only about 9--11 retained Med-LM C2 rows per
    round in the 2026-08-18 rejected run.  Use the already validated fresh-root
    batch as the minimum proposal scale and the C2 discovery retry budget as
    the maximum number of rounds; neither changes labels or physics.
    """
    request_rows = max(
        int(CONFIG["FINAL_TOPUP_MIN_BATCH"]),
        int(CONFIG["FRESH_ROOT_BATCH_SIZE"]),
        8 * max(int(row_deficit), 0),
        int(CONFIG["MAX_POSITIVE_ROWS_PER_ROOT_PER_CELL"])
        * max(int(root_deficit), 0),
    )
    maximum_rounds = max(
        int(CONFIG["FINAL_TOPUP_MAX_ROUNDS"]),
        int(CONFIG["C2_FRESH_ROOT_MAX_ROUNDS"]),
    )
    return request_rows, maximum_rounds


def _rebalance_evaluation_level_support(
        frame: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Move whole train root families to evidence-deficient eval cells.

    The operation is deterministic, never splits a lineage family and never
    takes a train cell below its own row/root floor.  It intentionally allows
    validation/test sizes to grow slightly: exact rare-class evidence is more
    important than preserving an approximate 80/10/10 row ratio.
    """
    output = frame.copy()
    if len(output) < int(CONFIG["EVALUATION_SUPPORT_GATE_MIN_ROWS"]):
        return output, {
            "enforced": False,
            "passed": True,
            "moves": [],
            "failures": [],
        }
    required_columns = {
        "root_seed_id", "split_role", "munition_id",
        *{f"{task}_level" for task in ("K", "M", "F", "C")},
    }
    missing = sorted(required_columns - set(output.columns))
    if missing:
        raise RuntimeError(
            "Evaluation-support rebalance is missing columns: "
            f"{missing}")

    minimum_eval_rows = int(
        CONFIG["MIN_EVAL_EXACT_LEVEL_ROWS"])
    minimum_eval_roots = int(
        CONFIG["MIN_EVAL_EXACT_LEVEL_ROOTS"])
    minimum_train_rows = int(
        CONFIG["MIN_TRAIN_EXACT_LEVEL_ROWS"])
    minimum_train_roots = int(
        CONFIG["MIN_TRAIN_LEVEL_ROOTS"])
    locked_roots = set()
    exhausted_cells = set()
    moves: List[Dict[str, Any]] = []

    def _deficits() -> List[Tuple[int, int, int, str, str]]:
        result = []
        for split_role in ("val", "test"):
            for munition_id in range(4):
                for task in ("K", "M", "F", "C"):
                    for level in (0, 1, 2):
                        if not _exact_level_is_required(
                                munition_id, task, level):
                            continue
                        rows, roots = _exact_level_support(
                            output, split_role, munition_id,
                            task, level)
                        row_deficit = max(
                            minimum_eval_rows - rows, 0)
                        root_deficit = max(
                            minimum_eval_roots - roots, 0)
                        cell_key = (
                            str(split_role), int(munition_id),
                            str(task), int(level))
                        if (
                            (row_deficit or root_deficit)
                            and cell_key not in exhausted_cells
                        ):
                            result.append((
                                row_deficit, root_deficit,
                                munition_id, task,
                                f"{level}|{split_role}",
                            ))
        return sorted(
            result,
            key=lambda item: (
                -item[0], -item[1], item[2],
                item[3], item[4]),
        )

    maximum_moves = max(
        1, 4 * 4 * 3 * 2
        * (minimum_eval_rows + minimum_eval_roots))
    for _ in range(maximum_moves):
        deficits = _deficits()
        if not deficits:
            break
        _, _, munition_id, task, encoded = deficits[0]
        level_text, target_split = encoded.split("|", 1)
        level = int(level_text)
        target_mask = (
            (output["split_role"].astype(str) == "train")
            & (output["munition_id"].astype(int) == munition_id)
            & (output[f"{task}_level"].astype(int) == level)
            & (~output["root_seed_id"].astype(str).isin(
                locked_roots))
        )
        candidate_roots = output.loc[
            target_mask, "root_seed_id"].astype(str).unique()
        ranked_candidates = []
        for root_id in candidate_roots:
            root_rows = output[
                output["root_seed_id"].astype(str) == root_id]
            target_contribution = int(
                (
                    (root_rows["munition_id"].astype(int)
                     == munition_id)
                    & (root_rows[f"{task}_level"].astype(int)
                       == level)
                ).sum())
            ranked_candidates.append((
                -target_contribution,
                len(root_rows),
                _stable_uint32(
                    f"eval-rebalance|{root_id}|"
                    f"{target_split}|{task}|{level}"),
                root_id,
            ))
        ranked_candidates.sort()

        selected_root = None
        for _, _, _, root_id in ranked_candidates:
            root_rows = output[
                output["root_seed_id"].astype(str) == root_id]
            safe = True
            for current_task in ("K", "M", "F", "C"):
                levels_in_root = root_rows.loc[
                    root_rows["munition_id"].astype(int)
                    == munition_id,
                    f"{current_task}_level",
                ].astype(int).unique()
                for current_level in levels_in_root:
                    if not _exact_level_is_required(
                            munition_id, current_task,
                            int(current_level)):
                        continue
                    train_rows, train_roots = _exact_level_support(
                        output, "train", munition_id,
                        current_task, int(current_level))
                    removed_rows = int((
                        (root_rows["munition_id"].astype(int)
                         == munition_id)
                        & (
                            root_rows[
                                f"{current_task}_level"
                            ].astype(int)
                            == int(current_level)
                        )
                    ).sum())
                    if (
                        train_rows - removed_rows
                        < minimum_train_rows
                        or train_roots - 1
                        < minimum_train_roots
                    ):
                        safe = False
                        break
                if not safe:
                    break
            if safe:
                selected_root = root_id
                break

        if selected_root is None:
            # Exhaust only this target cell.  Locking all of its candidate
            # roots (and aborting the global loop) previously prevented later
            # Med-RD/Heavy C2 cells from using otherwise safe train families.
            exhausted_cells.add((
                str(target_split), int(munition_id),
                str(task), int(level)))
            continue
        root_mask = (
            output["root_seed_id"].astype(str) == selected_root)
        moved_rows = int(root_mask.sum())
        output.loc[root_mask, "split_role"] = target_split
        locked_roots.add(selected_root)
        moves.append({
            "root_seed_id": selected_root,
            "rows": moved_rows,
            "munition_id": int(munition_id),
            "target_task": task,
            "target_level": int(level),
            "from": "train",
            "to": target_split,
        })

    failures = []
    final_support: Dict[str, Any] = {}
    for split_role in ("val", "test"):
        final_support[split_role] = {}
        for munition_id in range(4):
            final_support[split_role][str(munition_id)] = {}
            for task in ("K", "M", "F", "C"):
                final_support[split_role][str(munition_id)][task] = {}
                for level in (0, 1, 2):
                    if not _exact_level_is_required(
                            munition_id, task, level):
                        continue
                    rows, roots = _exact_level_support(
                        output, split_role, munition_id,
                        task, level)
                    final_support[split_role][
                        str(munition_id)][task][str(level)] = {
                            "rows": rows,
                            "root_families": roots,
                        }
                    if (
                        rows < minimum_eval_rows
                        or roots < minimum_eval_roots
                    ):
                        failures.append(
                            f"m_id={munition_id}:{task}=L{level} "
                            f"{split_role} rows/root={rows}/{roots} "
                            f"< {minimum_eval_rows}/"
                            f"{minimum_eval_roots}")

    root_split_counts = output.groupby(
        "root_seed_id")["split_role"].nunique()
    if int((root_split_counts > 1).sum()) != 0:
        raise RuntimeError(
            "Evaluation-support rebalance split a root family.")
    return output, {
        "enforced": True,
        "passed": not failures,
        "minimum_rows": minimum_eval_rows,
        "minimum_root_families": minimum_eval_roots,
        "moves": moves,
        "final_support": final_support,
        "failures": failures,
    }


# ============================================================================
# 统筹管道控制流 (重构版：双路拒止采样)
# ============================================================================

def build_dataset_pipeline(target_total: int = None,
                           phase1_ratio: float = None):
    _validate_generation_config()
    sampler = PhysicsAwareSampler()
    target_total = int(target_total if target_total is not None else CONFIG["N_TARGET"])
    phase1_ratio = float(phase1_ratio if phase1_ratio is not None else CONFIG["PHASE1_RATIO"])
    n_phase1 = int(target_total * phase1_ratio)
    seed_th = CONFIG["SEED_PROB_RELAX"]
    valid_th = CONFIG["VALID_PROB_STRICT"]
    final_quota = _allocate_counts(target_total, CONFIG["MUNITION_FINAL_TARGET"])

    print(f"\n==========================================================")
    print(f"[Phase 1] 启动分层探测 (需求产量: {n_phase1})...")
    print(f"[Phase 1] 种子阈值策略：宽松={seed_th} (爬行起点)，严格={valid_th} (入库)")
    print(f"[Phase 1] 最终弹型目标配额: {_format_munition_count_dict(final_quota)}")
    df_p1 = sampler.generate_phase_1(n_phase1)

    print(f"[Phase 1] 正在将其输送至损伤引擎...")
    # The attribute is control-plane metadata only; it is not sent to workers
    # or written as a dataset feature.  Unit-test simulators that replace
    # run_simulation_batch remain unaffected.
    df_p1.attrs["simulation_checkpoint_namespace"] = "phase1"
    res_p1 = sampler.run_simulation_batch(df_p1)
    if res_p1.empty:
        raise RuntimeError("Phase 1 未产出任何可用样本，无法继续构建数据集。")

    phase1_kept_counts = {
        m_id: int((res_p1["munition_id"] == m_id).sum())
        for m_id in range(4)
    }
    print(f"[Phase 1] 实际保留分布: {_format_munition_count_dict(phase1_kept_counts)}")

    accepted_by_mun = {
        m_id: res_p1[res_p1["munition_id"] == m_id].copy().reset_index(drop=True)
        for m_id in range(4)
    }
    pool_by_mun = {
        m_id: accepted_by_mun[m_id].copy()
        for m_id in range(4)
    }
    phase2_task_counts = {
        m_id: {task: 0 for task in CONFIG["PHASE2_TOP_OFF_PLAN"][m_id]}
        for m_id in range(4)
    }
    phase2_discovery_stats = {
        m_id: {} for m_id in range(4)
    }
    phase2_cell_cap_stats = {
        m_id: {} for m_id in range(4)
    }

    global_k2_target = int(round(
        target_total * CONFIG["K2_PHASE2_STOP_RATIO"]))
    global_k2_final_max = int(np.floor(
        target_total * CONFIG["K2_FINAL_MAX_RATIO"]))
    current_global_k2 = int((res_p1["K2_prob"] >= valid_th).sum())
    print(
        f"[Phase 2] K2 定向补样停止阈值: {global_k2_target}，"
        f"最终安全上限: {global_k2_final_max}，"
        f"Phase 1 已占 {current_global_k2}")

    def _rebalance_heavy_k2_alloc(alloc_map: Dict[str, int]) -> Dict[str, int]:
        nonlocal current_global_k2
        if "K2_prob" not in alloc_map:
            return alloc_map

        remaining_k2 = max(global_k2_target - current_global_k2, 0)
        requested_k2 = alloc_map.get("K2_prob", 0)
        if requested_k2 <= remaining_k2:
            return alloc_map

        overflow = requested_k2 - remaining_k2
        alloc_map["K2_prob"] = remaining_k2
        fallback_weights = {
            task: weight
            for task, weight in CONFIG["PHASE2_TOP_OFF_PLAN"][3].items()
            if task != "K2_prob"
        }
        if overflow > 0 and fallback_weights:
            for task, count in _allocate_counts(overflow, fallback_weights).items():
                alloc_map[task] = alloc_map.get(task, 0) + count
        return alloc_map

    def _run_topoff_task(m_id: int, prob_col: str, num_needed: int):
        nonlocal current_global_k2
        if num_needed <= 0:
            return

        original_num_needed = int(num_needed)
        seed_mask = _target_seed_mask(
            pool_by_mun[m_id], prob_col, seed_th, valid_th)
        if "split_role" in pool_by_mun[m_id].columns:
            seed_mask &= (pool_by_mun[m_id]["split_role"].values == "train")
        seeds = pool_by_mun[m_id][seed_mask].copy()
        desired_seed_roots = min(
            int(CONFIG["MIN_BOUNDARY_SEED_ROOTS"]), max(int(num_needed), 8))
        desired_root_key = (
            "C2_TARGET_STRICT_POSITIVE_ROOTS"
            if str(prob_col).startswith("C2")
            else "TARGET_STRICT_POSITIVE_ROOTS"
        )
        desired_strict_roots = min(
            int(CONFIG[desired_root_key]), max(int(num_needed), 8))
        fresh_results, discovery_stats = sampler.discover_fresh_target_roots(
            pool_by_mun[m_id], m_id, prob_col, seed_th, valid_th,
            desired_seed_roots=desired_seed_roots,
            desired_strict_roots=desired_strict_roots,
        )
        phase2_discovery_stats[m_id][prob_col] = discovery_stats
        discovery_stats["requested_rows"] = original_num_needed
        discovery_stats["fresh_rows_accepted"] = 0
        discovery_stats["crawl_rows_accepted"] = 0
        if not fresh_results.empty:
            pool_by_mun[m_id] = pd.concat(
                [pool_by_mun[m_id], fresh_results], ignore_index=True)
            fresh_valid = fresh_results[
                _target_valid_mask(fresh_results, prob_col, valid_th)
            ].copy()
            if prob_col == "K2_prob" and m_id == 3:
                remaining_k2 = max(global_k2_target - current_global_k2, 0)
                if len(fresh_valid) > remaining_k2:
                    fresh_valid = fresh_valid.sample(
                        n=remaining_k2, replace=False,
                        random_state=CONFIG["RANDOM_SEED"])
            fresh_valid = _take_target_rows_with_capacity(
                accepted_by_mun[m_id], fresh_valid, prob_col, valid_th,
                int(CONFIG["MAX_ROWS_PER_ROOT"]),
                int(CONFIG["MAX_POSITIVE_ROWS_PER_ROOT_PER_CELL"]),
                CONFIG["RANDOM_SEED"],
            )
            if len(fresh_valid) > num_needed:
                # One row per newly discovered root is preferred before any
                # repeated root, preserving the maximum possible diversity.
                fresh_valid = fresh_valid.sort_values("root_seed_id").drop_duplicates(
                    "root_seed_id", keep="first").head(num_needed)
            if not fresh_valid.empty:
                accepted_by_mun[m_id] = pd.concat(
                    [accepted_by_mun[m_id], fresh_valid], ignore_index=True)
                phase2_task_counts[m_id][prob_col] = (
                    phase2_task_counts[m_id].get(prob_col, 0) + int(len(fresh_valid))
                )
                discovery_stats["fresh_rows_accepted"] = int(len(fresh_valid))
                if prob_col == "K2_prob":
                    current_global_k2 += int(len(fresh_valid))
                num_needed -= int(len(fresh_valid))
                print(f"  [FreshRoot] 严格命中并直接保留 {len(fresh_valid)} 枚，"
                      f"剩余爬行目标 {max(num_needed, 0)}。")

        seed_mask = _target_seed_mask(
            pool_by_mun[m_id], prob_col, seed_th, valid_th)
        if "split_role" in pool_by_mun[m_id].columns:
            seed_mask &= (pool_by_mun[m_id]["split_role"].values == "train")
        seeds = pool_by_mun[m_id][seed_mask].copy()

        if num_needed <= 0:
            return
        if seeds.empty:
            print(f"  [Phase 2] m_id={m_id} {prob_col} 新 root 探测后仍无可用种子，跳过。")
            return
        seeds = sampler._select_boundary_seeds(
            seeds, prob_col,
            max(int(CONFIG["CRAWL_TOPK_PER_STAGE"]), desired_seed_roots),
        )

        print(f"\n==========================================================")
        print(f"[Phase 2] m_id={m_id} {prob_col} 种子 {len(seeds)} 枚，目标补齐 {num_needed} 枚。")
        res_topoff = sampler.crawl_multistage_guided(seeds, num_needed, prob_col)
        if res_topoff.empty:
            print(f"  [Phase 2] m_id={m_id} {prob_col} 爬行未产出结果。")
            return

        pool_by_mun[m_id] = pd.concat([pool_by_mun[m_id], res_topoff], ignore_index=True)
        valid_topoff = res_topoff[_target_valid_mask(res_topoff, prob_col, valid_th)].copy()
        valid_topoff = _take_target_rows_with_capacity(
            accepted_by_mun[m_id], valid_topoff, prob_col, valid_th,
            int(CONFIG["MAX_ROWS_PER_ROOT"]),
            int(CONFIG["MAX_POSITIVE_ROWS_PER_ROOT_PER_CELL"]),
            CONFIG["RANDOM_SEED"],
        )
        if valid_topoff.empty:
            print(f"  [Phase 2] m_id={m_id} {prob_col} 严格阈值下无通过样本。")
            return

        if prob_col == "K2_prob" and m_id == 3:
            remaining_k2 = max(global_k2_target - current_global_k2, 0)
            if remaining_k2 <= 0:
                print("  [Phase 2] Heavy 的 K2 全局上限已满足，跳过新增 K2 样本。")
                return
            if len(valid_topoff) > remaining_k2:
                valid_topoff = valid_topoff.sample(
                    n=remaining_k2, replace=False, random_state=CONFIG["RANDOM_SEED"])

        if len(valid_topoff) > num_needed:
            valid_topoff = valid_topoff.sample(
                n=num_needed, replace=False, random_state=CONFIG["RANDOM_SEED"])

        accepted_by_mun[m_id] = pd.concat([accepted_by_mun[m_id], valid_topoff], ignore_index=True)
        phase2_task_counts[m_id][prob_col] = phase2_task_counts[m_id].get(prob_col, 0) + int(len(valid_topoff))
        phase2_discovery_stats[m_id][prob_col]["crawl_rows_accepted"] = int(len(valid_topoff))
        if prob_col == "K2_prob":
            current_global_k2 += int(len(valid_topoff))

        print(f"  [Phase 2] m_id={m_id} {prob_col} 严格通过 {len(valid_topoff)} 枚，"
              f"当前该弹型总量 {len(accepted_by_mun[m_id])} / {final_quota[m_id]}，"
              f"全局 K2={current_global_k2}")

    print(f"\n[Phase 2] 启动按弹型 quota 的 top-off 补齐。")
    for m_id in range(4):
        current_count = len(accepted_by_mun[m_id])
        deficit = max(final_quota[m_id] - current_count, 0)
        if deficit <= 0:
            print(f"[Phase 2] m_id={m_id} 已达配额 {current_count}/{final_quota[m_id]}，跳过。")
            continue

        alloc = _allocate_counts(deficit, CONFIG["PHASE2_TOP_OFF_PLAN"][m_id])
        if m_id == 3:
            alloc = _rebalance_heavy_k2_alloc(alloc)

        print(f"[Phase 2] m_id={m_id} 当前 {current_count}/{final_quota[m_id]}，计划: {alloc}")
        for prob_col, num_needed in alloc.items():
            _run_topoff_task(m_id, prob_col, num_needed)

    # 先做一次硬裁剪，确保任何弹型都不会因某个 top-off 分支过量增长。
    for m_id in range(4):
        accepted_by_mun[m_id], cell_cap_stats = _cap_all_ordinal_positive_families(
            accepted_by_mun[m_id], valid_th,
            int(CONFIG["MAX_POSITIVE_ROWS_PER_ROOT_PER_CELL"]),
            CONFIG["RANDOM_SEED"],
        )
        phase2_cell_cap_stats[m_id] = cell_cap_stats
        if cell_cap_stats:
            print(f"[Diversity Guard] m_id={m_id} 已执行所有序数正类的任务级家族限流: "
                  f"{cell_cap_stats}")
        accepted_by_mun[m_id], family_stats = _cap_root_families(
            accepted_by_mun[m_id], int(CONFIG["MAX_ROWS_PER_ROOT"]),
            CONFIG["RANDOM_SEED"])
        if family_stats["rows_removed"]:
            print(f"[Diversity Guard] m_id={m_id} 裁去 "
                  f"{family_stats['rows_removed']} 个超额同源后代，"
                  f"涉及 {family_stats['families_capped']} 个 root。")
        capped_df, _ = _cap_dataframe_by_munition(
            accepted_by_mun[m_id], {m_id: final_quota[m_id]}, CONFIG["RANDOM_SEED"])
        accepted_by_mun[m_id] = capped_df

    # A split rebalancer cannot manufacture evidence: before the independent
    # fallback consumes most of the build time, guarantee enough total C2
    # rows/roots to retain the train floor and populate both evaluation splits.
    # The rejected 2026-08-17 candidate had 282 Med-LM C2 rows, below the hard
    # 128 + 100 + 100 minimum, while Med-RD/Heavy had sufficient total supply.
    level2_total_support_topoff: Dict[str, Any] = {
        "enforced": (
            int(target_total)
            >= int(CONFIG["EVALUATION_SUPPORT_GATE_MIN_ROWS"])),
        "rounds": [],
    }
    if level2_total_support_topoff["enforced"]:
        minimum_level2_rows, minimum_level2_roots = (
            _minimum_total_exact_level_support())

        def _level2_cell_state(
                munition_id: int, task: str) -> Tuple[int, int]:
            bucket = accepted_by_mun[munition_id]
            rows = bucket[
                bucket[f"{task}_level"].astype(int) == 2]
            return int(len(rows)), int(
                rows["root_seed_id"].astype(str).nunique())

        for m_id in range(4):
            if not _exact_level_is_required(m_id, "C", 2):
                continue
            stagnant_rounds = 0
            _, maximum_support_rounds = _level2_support_topoff_budget(0, 0)
            for round_index in range(maximum_support_rounds):
                before_rows, before_roots = _level2_cell_state(m_id, "C")
                row_deficit = max(
                    minimum_level2_rows - before_rows, 0)
                root_deficit = max(
                    minimum_level2_roots - before_roots, 0)
                if not row_deficit and not root_deficit:
                    break
                request_rows, _ = _level2_support_topoff_budget(
                    row_deficit, root_deficit)
                _run_topoff_task(m_id, "C2_prob", request_rows)
                accepted_by_mun[m_id], support_cell_caps = (
                    _cap_all_ordinal_positive_families(
                        accepted_by_mun[m_id], valid_th,
                        int(CONFIG[
                            "MAX_POSITIVE_ROWS_PER_ROOT_PER_CELL"]),
                        CONFIG["RANDOM_SEED"],
                    ))
                accepted_by_mun[m_id], _ = _cap_root_families(
                    accepted_by_mun[m_id],
                    int(CONFIG["MAX_ROWS_PER_ROOT"]),
                    CONFIG["RANDOM_SEED"],
                )
                accepted_by_mun[m_id], _ = _cap_dataframe_by_munition(
                    accepted_by_mun[m_id],
                    {m_id: final_quota[m_id]},
                    CONFIG["RANDOM_SEED"],
                )
                after_rows, after_roots = _level2_cell_state(m_id, "C")
                level2_total_support_topoff["rounds"].append({
                    "munition_id": int(m_id),
                    "task": "C",
                    "round": int(round_index + 1),
                    "requested_rows": int(request_rows),
                    "before_rows": int(before_rows),
                    "before_root_families": int(before_roots),
                    "after_rows": int(after_rows),
                    "after_root_families": int(after_roots),
                    "cell_cap_removals": support_cell_caps,
                })
                print(
                    f"[Split Supply] m_id={m_id}:C=L2 第 {round_index + 1} 轮 "
                    f"rows/root {before_rows}/{before_roots} -> "
                    f"{after_rows}/{after_roots}，目标 "
                    f"{minimum_level2_rows}/{minimum_level2_roots}")
                if (
                    after_rows <= before_rows
                    and after_roots <= before_roots
                ):
                    stagnant_rounds += 1
                else:
                    stagnant_rounds = 0
                if stagnant_rounds >= 3:
                    break

        combined_for_support = pd.concat(
            [accepted_by_mun[m_id] for m_id in range(4)],
            ignore_index=True,
        )
        remaining_level2_deficits = (
            _level2_total_support_deficits(combined_for_support))
        level2_total_support_topoff["minimum_rows"] = int(
            minimum_level2_rows)
        level2_total_support_topoff["minimum_root_families"] = int(
            minimum_level2_roots)
        level2_total_support_topoff["remaining_deficits"] = (
            remaining_level2_deficits)
        level2_total_support_topoff["passed"] = not bool(
            remaining_level2_deficits)
        if remaining_level2_deficits:
            preview = [
                f"m_id={item['munition_id']}:{item['task']}=L2 "
                f"rows/root={item['rows']}/{item['root_families']} < "
                f"{item['minimum_rows']}/{item['minimum_root_families']}"
                for item in remaining_level2_deficits[:8]
            ]
            raise RuntimeError(
                "Phase-2 总供给门禁失败，停止进入耗时的独立回填："
                f"{preview}")
    else:
        level2_total_support_topoff["passed"] = True

    def _limit_replenishment_k2(
        candidates: pd.DataFrame,
        current_k2_rows: int,
    ) -> pd.DataFrame:
        """Prevent final Phase-1 replenishment from crossing the K2 ceiling."""
        if candidates.empty:
            return candidates
        positive_mask = (
            candidates["K2_prob"].to_numpy(dtype=float) >= valid_th)
        allowed_positive = max(global_k2_final_max - current_k2_rows, 0)
        positive = candidates.loc[positive_mask]
        negative = candidates.loc[~positive_mask]
        if len(positive) > allowed_positive:
            positive = positive.sample(
                n=allowed_positive,
                replace=False,
                random_state=CONFIG["RANDOM_SEED"],
            ) if allowed_positive else positive.iloc[0:0]
        return pd.concat([negative, positive], ignore_index=True).sample(
            frac=1.0, random_state=CONFIG["RANDOM_SEED"]).reset_index(drop=True)

    accepted_k2_before_replenishment = sum(
        int((frame["K2_prob"] >= valid_th).sum())
        for frame in accepted_by_mun.values()
    )
    if accepted_k2_before_replenishment > global_k2_final_max:
        raise RuntimeError(
            "Phase-2 结束时 K2 已超过最终安全上限："
            f"{accepted_k2_before_replenishment}/{global_k2_final_max}。"
            "请降低 K2_HUNT/K1_K2 层预算后重新生成。")

    # 若仍有缺口，只能由该弹型自己的 Phase 1 风格样本回填，禁止跨弹型替补。
    for m_id in range(4):
        retries = 0
        max_retries = int(CONFIG["FINAL_TOPUP_MAX_ROUNDS"])
        while len(accepted_by_mun[m_id]) < final_quota[m_id] and retries < max_retries:
            deficit = final_quota[m_id] - len(accepted_by_mun[m_id])
            request_n = max(deficit, int(CONFIG["FINAL_TOPUP_MIN_BATCH"]))
            print(f"[Fallback] m_id={m_id} 尚缺 {deficit} 枚，"
                  f"第 {retries + 1}/{max_retries} 轮请求 {request_n} 枚独立 Phase 1 root。")
            replenish_inputs = sampler.generate_phase1_replenishment(m_id, request_n)
            retries += 1
            if replenish_inputs.empty:
                print(f"[Fallback] m_id={m_id} 未生成可回填候选。")
                continue

            replenish_results = sampler.run_simulation_batch(replenish_inputs)
            if replenish_results.empty:
                print(f"[Fallback] m_id={m_id} 回填仿真无结果。")
                continue

            current_k2_rows = sum(
                int((frame["K2_prob"] >= valid_th).sum())
                for frame in accepted_by_mun.values()
            )
            replenish_results = _limit_replenishment_k2(
                replenish_results, current_k2_rows)
            if replenish_results.empty:
                print(
                    f"[Fallback] m_id={m_id} 候选全部会使 K2 超过 "
                    f"{CONFIG['K2_FINAL_MAX_RATIO']:.1%} 安全上限。")
                continue

            if len(replenish_results) > deficit:
                replenish_results = replenish_results.sample(
                    n=deficit, replace=False, random_state=CONFIG["RANDOM_SEED"])
            accepted_by_mun[m_id] = pd.concat([accepted_by_mun[m_id], replenish_results], ignore_index=True)

        if len(accepted_by_mun[m_id]) < final_quota[m_id]:
            raise RuntimeError(
                f"精确配额回填失败：m_id={m_id} 最终仅达到 "
                f"{len(accepted_by_mun[m_id])}/{final_quota[m_id]}，"
                f"已尝试 {max_retries} 轮。")

    final_df = pd.concat([accepted_by_mun[m_id] for m_id in range(4)], ignore_index=True)
    final_df, final_family_stats = _cap_root_families(
        final_df, int(CONFIG["MAX_ROWS_PER_ROOT"]), CONFIG["RANDOM_SEED"])
    if final_family_stats["rows_removed"]:
        print(f"[Diversity Guard] 最终再次裁去 {final_family_stats['rows_removed']} 个超额同源后代。")
    final_df, _ = _cap_dataframe_by_munition(final_df, final_quota, CONFIG["RANDOM_SEED"])
    final_counts = {
        m: int((final_df["munition_id"] == m).sum()) for m in range(4)
    }
    if len(final_df) != target_total or final_counts != final_quota:
        raise RuntimeError(
            f"最终配额不变量失败：rows={len(final_df)}/{target_total}, "
            f"munition={final_counts}, expected={final_quota}")
    final_df, evaluation_split_rebalance = (
        _rebalance_evaluation_level_support(final_df))
    final_df.attrs["evaluation_split_rebalance"] = (
        evaluation_split_rebalance)
    final_df.attrs["level2_total_support_topoff"] = (
        level2_total_support_topoff)
    print(
        "[Split Evidence] "
        f"moved_roots={len(evaluation_split_rebalance['moves'])} | "
        f"status={'PASS' if evaluation_split_rebalance['passed'] else 'FAIL'}"
    )
    final_k2_rows = int((final_df["K2_prob"] >= valid_th).sum())
    if final_k2_rows > global_k2_final_max:
        raise RuntimeError(
            f"最终 K2 安全上限失败：{final_k2_rows} > "
            f"{global_k2_final_max} "
            f"({CONFIG['K2_FINAL_MAX_RATIO']:.1%})")
    print(f"\n[System] 最终弹型分布: {_format_munition_count_dict(final_counts)}")

    # ==================================================================
    # 第三轮重构：统一 Class-Balanced 加权 (Cui et al. 2019)
    # 取代原 1/pass_rate × 150 的激进放大 — 那是 *错误* 的：
    #   - 爬行点已被采样过程超采 (in Q distribution)
    #   - 再用 1/pass_rate 加权 = 又一次倾斜 → 双重偏置
    # CB 加权只看样本计数本身，与采样过程解耦
    # ==================================================================
    final_df = _finalize_sample_weights(final_df, valid_th)

    # 在元数据里记录 focal loss 建议的 gamma，供下游训练脚本读取
    final_df.attrs["focal_loss_gamma"] = CONFIG["FOCAL_LOSS_GAMMA"]
    final_df.attrs["valid_prob_strict"] = valid_th
    final_df.attrs["generation_profile"] = _build_generation_profile(
        final_df, final_quota, phase1_kept_counts, phase2_task_counts, seed_th, valid_th,
        target_total=target_total, phase1_ratio=phase1_ratio,
        phase2_discovery_stats=phase2_discovery_stats,
        phase2_cell_cap_stats=phase2_cell_cap_stats)

    return final_df


# ============================================================================
# 第三轮新增：训练侧权重 + 推理校正辅助函数
# ============================================================================

def _apply_class_balanced_weights(df: pd.DataFrame, valid_th: float,
                                   cb_beta: float) -> pd.DataFrame:
    """统一对所有稀有任务做 Class-Balanced 加权 (Cui et al. 2019)

    公式: cb_weight = (1 - β) / (1 - β^n_pos)，β=0.999

    每个样本的 loss_weight 按"在该样本所属正类中最大的 CB 权重"放大。
    这样：
      - 在 8 个任务中只有 K2=positive 的样本，仅按 K2 的 CB 权重放大
      - K2+C2 双正样本，按二者中较大的 CB 权重放大 (避免双重计权)
      - 全负样本，权重不变
    """
    n_all = len(df)
    tasks = ["K1_prob", "K2_prob", "M1_prob", "M2_prob",
             "F1_prob", "F2_prob", "C1_prob", "C2_prob"]

    print(f"\n[CB Weights] 启动 Class-Balanced 加权 (β={cb_beta})...")
    cb_per_task = {}
    for task in tasks:
        n_pos = int((df[task] >= valid_th).sum())
        if n_pos == 0:
            cb_per_task[task] = 1.0
            continue
        # 标准 CB 公式
        cb = (1.0 - cb_beta) / (1.0 - cb_beta ** n_pos)
        # 归一化：让所有任务 CB 权重 × n_pos / n_all 之和约为 1 (避免梯度爆炸)
        cb_normalized = cb * n_all / len(tasks)
        cb_per_task[task] = cb_normalized
        print(f"  {task}: n_pos={n_pos:>6}/{n_all} "
              f"({100*n_pos/n_all:>5.2f}%), CB_weight × {cb_normalized:>6.2f}")

    # 计算每个样本应得的 CB 倍数 (取所有正类任务中的最大值)
    sample_cb_mult = np.ones(len(df))
    for task, w in cb_per_task.items():
        if w > 1.0:
            mask = (df[task] >= valid_th).values
            sample_cb_mult[mask] = np.maximum(sample_cb_mult[mask], w)

    df["class_balance_weight"] = sample_cb_mult
    print(f"[CB Weights] 完成：class_balance_weight 动态范围 "
          f"[{df['class_balance_weight'].min():.3f}, "
          f"{df['class_balance_weight'].max():.2f}]")
    return df


def _finalize_sample_weights(df: pd.DataFrame, valid_th: float) -> pd.DataFrame:
    """Combine documented weight factors once, normalize, and clip safely."""
    df = df.copy()
    defaults = {
        "aoa_accept_prob": 1.0,
        "aoa_ipw": 1.0,
        "physics_weight": 1.0,
        "active_sampling_weight": 1.0,
        "class_balance_weight": 1.0,
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default

    family_sizes = df["root_seed_id"].astype(str).map(
        df["root_seed_id"].astype(str).value_counts())
    df["family_size"] = family_sizes.astype(int)
    df["family_weight"] = np.minimum(
        1.0,
        np.sqrt(float(CONFIG["FAMILY_WEIGHT_REFERENCE_SIZE"]) /
                np.maximum(family_sizes.values.astype(float), 1.0)),
    )

    if bool(CONFIG["APPLY_GENERATOR_CB_WEIGHT"]):
        df = _apply_class_balanced_weights(df, valid_th, CONFIG["CB_LOSS_BETA"])
    else:
        df["class_balance_weight"] = 1.0
        print("\n[Weights] 生成器侧 class-balanced 权重已禁用；训练侧只保留一套类别平衡机制。")

    factors = [
        "aoa_ipw", "physics_weight", "active_sampling_weight",
        "family_weight", "class_balance_weight",
    ]
    raw_weight = np.ones(len(df), dtype=float)
    for column in factors:
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise RuntimeError(f"样本权重分量 {column} 含非正值或非有限值。")
        raw_weight *= values
    df["loss_weight_raw"] = raw_weight

    split_positions = {}
    if "split_role" in df.columns:
        split_positions = {
            str(role): df.index.get_indexer(indices)
            for role, indices in df.groupby("split_role").groups.items()
        }
    elif len(df):
        split_positions = {"train": np.arange(len(df), dtype=int)}

    def _normalized_tempered(alpha: float) -> np.ndarray:
        values = np.power(raw_weight, alpha)
        for positions in split_positions.values():
            mean_weight = float(np.mean(values[positions]))
            if mean_weight > 0:
                values[positions] /= mean_weight
        return np.clip(
            values,
            float(CONFIG["LOSS_WEIGHT_MIN"]),
            float(CONFIG["LOSS_WEIGHT_MAX"]),
        )

    def _ess_ratio(values: np.ndarray) -> float:
        if len(values) == 0:
            return 0.0
        return float(values.sum() ** 2 /
                     (np.square(values).sum() * len(values)))

    optimization_positions = split_positions.get(
        "train", np.arange(len(df), dtype=int))
    target_ess_ratio = min(
        1.0,
        float(CONFIG["MIN_WEIGHT_ESS_RATIO"]) +
        float(CONFIG.get("WEIGHT_ESS_TARGET_MARGIN", 0.0)),
    )
    min_alpha = float(CONFIG["WEIGHT_TEMPER_MIN_ALPHA"])
    alpha = 1.0
    normalized = _normalized_tempered(alpha)
    if _ess_ratio(normalized[optimization_positions]) < target_ess_ratio:
        low_values = _normalized_tempered(min_alpha)
        if _ess_ratio(low_values[optimization_positions]) >= target_ess_ratio:
            low, high = min_alpha, 1.0
            for _ in range(40):
                middle = 0.5 * (low + high)
                middle_values = _normalized_tempered(middle)
                if _ess_ratio(
                    middle_values[optimization_positions]) >= target_ess_ratio:
                    low = middle
                else:
                    high = middle
            alpha = low
            normalized = _normalized_tempered(alpha)
        else:
            alpha = min_alpha
            normalized = low_values

    df["loss_weight"] = normalized
    ess = float(df["loss_weight"].sum() ** 2 /
                np.square(df["loss_weight"]).sum()) if len(df) else 0.0
    ess_ratio = ess / max(len(df), 1)
    ess_by_split = {
        role: _ess_ratio(normalized[positions])
        for role, positions in split_positions.items()
    }
    train_ess_ratio = float(ess_by_split.get("train", ess_ratio))
    df.attrs["weight_tempering_alpha"] = float(alpha)
    df.attrs["weight_ess_ratio"] = float(ess_ratio)
    df.attrs["weight_train_ess_ratio"] = train_ess_ratio
    df.attrs["weight_ess_by_split"] = ess_by_split
    df.attrs["weight_ess_target_ratio"] = float(target_ess_ratio)
    df.attrs["weight_floor_count"] = int(np.isclose(
        normalized, float(CONFIG["LOSS_WEIGHT_MIN"]), rtol=0, atol=1e-12).sum())
    df.attrs["weight_cap_count"] = int(np.isclose(
        normalized, float(CONFIG["LOSS_WEIGHT_MAX"]), rtol=0, atol=1e-12).sum())
    print(f"[Weights] 最终 loss_weight=[{df['loss_weight'].min():.3f}, "
          f"{df['loss_weight'].max():.3f}]，tempering α={alpha:.4f}，"
          f"train ESS={train_ess_ratio:.1%}（目标 {target_ess_ratio:.1%}），"
          f"全表 ESS={ess:.0f}/{len(df)} ({ess_ratio:.1%})。")
    return df


def _emit_logit_adjustment(df: pd.DataFrame, valid_th: float,
                             physical_prior: dict, out_path: str,
                             dataset_sha256: str = None):
    """输出语义正确但默认禁用的 ordinal prior-adjustment 元数据。

    使用方法 (推理侧)：
        logit_corrected = logit_train - log_adjust
        p_physical = sigmoid(logit_corrected)

    这样模型输出的概率会校正回物理先验，避免因训练数据 K2 比例被人为拔高
    导致的推理时假阳性
    """
    import json
    # The supplied domain priors describe raw rule events.  Convert them to the
    # same ordinal exceedance semantics used by the surrogate before recording
    # any shift; mixing these meanings made the old (disabled) file unsafe.
    ordinal_prior = {}
    for dim in ("K", "M", "F", "C"):
        p1 = float(physical_prior[f"{dim}1_prob"])
        p2 = float(physical_prior[f"{dim}2_prob"])
        ge1 = p1 if dim == "C" else 1.0 - (1.0 - p1) * (1.0 - p2)
        ordinal_prior[f"{dim}_ge1_prob"] = float(np.clip(ge1, 0.0, 1.0))
        ordinal_prior[f"{dim}_ge2_prob"] = float(np.clip(min(p2, ge1), 0.0, 1.0))

    train_df = (
        df[df["split_role"] == "train"]
        if "split_role" in df.columns else df
    )
    if train_df.empty:
        raise RuntimeError("Logit adjustment 无可用 train split。")

    adj = {}
    for task, pi_true in ordinal_prior.items():
        if task not in train_df.columns:
            continue
        pi_train = float((train_df[task] >= valid_th).mean())
        log_adjust = float(np.log(max(pi_train, 1e-6) / max(pi_true, 1e-6)))
        adj[task] = {
            "pi_train": pi_train,
            "pi_true": pi_true,
            "log_adjust": log_adjust,
        }
    adj["__meta__"] = {
        "schema": "ordinal_exceedance_v2",
        "enabled": False,
        "disabled_reason": (
            "Thresholds must be calibrated in the same shifted-logit space before enabling."
        ),
        "valid_threshold": valid_th,
        "n_train_samples": int(len(train_df)),
        "n_dataset_samples": int(len(df)),
        "dataset_sha256": dataset_sha256,
        "input_prior_semantics": "raw_rule_events",
        "output_prior_semantics": "ordinal_exceedance",
        "usage": "当前禁用；联合校准后方可执行 logit_corrected = logit_train - log_adjust",
        "reference": "Menon et al. 2020, Long-Tail Learning via Logit Adjustment",
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    temp_path = f"{out_path}.{os.getpid()}.stage0.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(adj, f, indent=2, ensure_ascii=False)
    os.replace(temp_path, out_path)

    print(f"\n[Logit Adjust] 已写入 {out_path}")
    for task, v in adj.items():
        if task.startswith("__"): continue
        print(f"  {task}: pi_train={v['pi_train']:.4f}, "
              f"pi_true={v['pi_true']:.4f}, log_adjust={v['log_adjust']:+.3f}")

def _write_dataset_with_profile(df_final: pd.DataFrame, out_path: str):
    out_path = os.path.abspath(out_path)
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    required = {
        "sample_id", "root_seed_id", "parent_id", "crawl_stage", "split_role",
        "frame_version", "dataset_schema", "label_mc_replicates",
        "label_mc_min_replicates", "label_mc_max_replicates",
        "aoa_accept_prob", "aoa_ipw", "physics_weight", "active_sampling_weight",
        "family_weight", "class_balance_weight", "loss_weight",
    }
    required.update({
        f"{dimension}_ge{level}_prob_std"
        for dimension in ("K", "M", "F", "C") for level in (1, 2)
    })
    required.update({
        f"{mechanism}_{dimension}_ge{level}_prob"
        for mechanism in ("fragment", "shock")
        for dimension in ("K", "M", "F", "C") for level in (1, 2)
    })
    required.update(COMPONENT_TARGET_COLUMNS)
    missing = sorted(required - set(df_final.columns))
    if missing:
        raise RuntimeError(f"Stage-0 写出门禁失败，缺少字段: {missing}")
    if df_final["sample_id"].duplicated().any():
        raise RuntimeError("Stage-0 写出门禁失败：sample_id 不唯一。")
    if set(df_final["frame_version"].astype(str).unique()) != {CONFIG["FRAME_CONVENTION_VERSION"]}:
        raise RuntimeError("Stage-0 写出门禁失败：frame_version 混杂。")
    if set(df_final["dataset_schema"].astype(str).unique()) != {CONFIG["DATASET_SCHEMA"]}:
        raise RuntimeError("Stage-0 写出门禁失败：dataset_schema 混杂。")
    mc_actual = df_final["label_mc_replicates"].astype(int)
    mc_minimum = df_final["label_mc_min_replicates"].astype(int)
    mc_maximum = df_final["label_mc_max_replicates"].astype(int)
    if ((mc_actual < 1) | (mc_actual < mc_minimum) |
            (mc_actual > mc_maximum) | (mc_minimum < 1) |
            (mc_maximum < mc_minimum)).any():
        raise RuntimeError("Stage-0 写出门禁失败：自适应 MC 次数不满足 min <= actual <= max。")
    root_split_counts = df_final.groupby("root_seed_id")["split_role"].nunique()
    if int((root_split_counts > 1).sum()) != 0:
        raise RuntimeError("Stage-0 写出门禁失败：root_seed_id 跨 split。")
    root_sizes = df_final["root_seed_id"].astype(str).value_counts()
    if int(root_sizes.max()) > int(CONFIG["MAX_ROWS_PER_ROOT"]):
        raise RuntimeError(
            f"Stage-0 写出门禁失败：单 root 最大 {int(root_sizes.max())} 行，"
            f"超过 {CONFIG['MAX_ROWS_PER_ROOT']}。")
    profile = dict(df_final.attrs.get("generation_profile", {}))

    def _reject_with_profile(reasons: List[str]) -> None:
        rejected = dict(profile)
        rejected["rejection"] = {
            "status": "REJECTED_NOT_FOR_TRAINING",
            "reasons": [str(reason) for reason in reasons],
            "candidate_rows": int(len(df_final)),
            "candidate_columns": int(len(df_final.columns)),
            "recorded_unix_time": float(time.time()),
        }
        rejected_path = os.path.join(out_dir, "generation_profile.rejected.json")
        rejected_temp = f"{rejected_path}.{os.getpid()}.stage0.tmp"
        with open(rejected_temp, "w", encoding="utf-8") as handle:
            json.dump(rejected, handle, indent=2, ensure_ascii=False)
        os.replace(rejected_temp, rejected_path)
        raise RuntimeError(
            "Stage-0 可用性门禁失败；完整诊断已写入 "
            f"{rejected_path}: {reasons[:8]}")

    invariant_failures = []
    target_total = profile.get("target_total")
    if target_total is not None and int(target_total) != len(df_final):
        invariant_failures.append(
            f"候选总行数={len(df_final)}，期望 {int(target_total)}")
    for m_id, meta in profile.get("per_munition", {}).items():
        expected = int(meta.get("target_quota", -1))
        observed = int((df_final["munition_id"] == int(m_id)).sum())
        if expected >= 0 and observed != expected:
            invariant_failures.append(
                f"m_id={m_id} 行数={observed}，期望 {expected}")
    if invariant_failures:
        _reject_with_profile(invariant_failures)

    usability_gate = profile.get("usability_gate", {})
    if usability_gate.get("enforced") and not usability_gate.get("passed"):
        _reject_with_profile(list(usability_gate.get("failures", [])))

    component_values = df_final[
        list(COMPONENT_TARGET_COLUMNS)].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(component_values).all()
        or np.any(component_values < 0.0)
        or np.any(component_values > 1.0)
    ):
        _reject_with_profile([
            "部件级破片/冲击波监督包含非有限值或超出 [0,1]。"
        ])

    # Component probabilities are simulator labels, not deployable inputs.
    # Keep the Stage-0 v2 table unchanged and seal them in a SHA-bound sidecar.
    base_frame = df_final.drop(
        columns=list(COMPONENT_TARGET_COLUMNS))
    component_frame = pd.DataFrame({
        "sample_id": df_final["sample_id"].astype(str).to_numpy(),
        **{
            column: df_final[column].to_numpy(dtype=np.float32)
            for column in COMPONENT_TARGET_COLUMNS
        },
    })

    temp_path = f"{out_path}.{os.getpid()}.stage0.tmp"
    component_path = os.path.join(
        out_dir, COMPONENT_SUPERVISION_FILENAME)
    component_temp_path = (
        f"{component_path}.{os.getpid()}.stage0.tmp")
    try:
        base_frame.to_parquet(
            temp_path, engine="pyarrow", index=False,
            row_group_size=int(CONFIG["PARQUET_ROW_GROUP_SIZE"]),
        )
        component_frame.to_parquet(
            component_temp_path, engine="pyarrow", index=False,
            row_group_size=int(CONFIG["PARQUET_ROW_GROUP_SIZE"]),
        )
        # Traverse every row group and every column.  The previous five-column
        # readback missed a corrupted nested histogram in another column.
        parquet_file = pq.ParquetFile(temp_path)
        try:
            if parquet_file.schema_arrow.names != list(
                    base_frame.columns):
                raise RuntimeError("Parquet 全列回读校验失败：列名或列顺序发生变化。")
            rows_read = 0
            for batch in parquet_file.iter_batches(batch_size=65536, use_threads=False):
                rows_read += int(batch.num_rows)
        finally:
            parquet_file.close()
        if rows_read != len(base_frame):
            raise RuntimeError(
                f"Parquet 全列回读校验失败：写出 {len(base_frame)} 行，读回 {rows_read} 行")
        component_parquet = pq.ParquetFile(component_temp_path)
        try:
            if component_parquet.schema_arrow.names != list(
                    component_frame.columns):
                raise RuntimeError(
                    "部件监督 Parquet 全列回读校验失败："
                    "列名或顺序发生变化。")
            component_rows_read = 0
            for batch in component_parquet.iter_batches(
                    batch_size=65536, use_threads=False):
                component_rows_read += int(batch.num_rows)
        finally:
            component_parquet.close()
        if component_rows_read != len(component_frame):
            raise RuntimeError(
                "部件监督 Parquet 行数回读失败："
                f"写出 {len(component_frame)}，"
                f"读回 {component_rows_read}")
        # Replace data artifacts only after both temporary files pass their
        # complete readback.  Profiles written below bind their exact hashes.
        os.replace(component_temp_path, component_path)
        os.replace(temp_path, out_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(component_temp_path):
            os.remove(component_temp_path)

    dataset_sha256 = sha256_file(out_path)
    component_sha256 = sha256_file(component_path)
    sample_id_order_sha256 = sha256_text_sequence(
        component_frame["sample_id"].astype(str))
    mc_histogram = {
        str(int(k)): int(v)
        for k, v in mc_actual.value_counts().sort_index().items()
    }
    profile.setdefault("label_mc", {})["replicate_histogram"] = mc_histogram
    artifact_parquet = pq.ParquetFile(out_path)
    try:
        parquet_row_groups = int(artifact_parquet.num_row_groups)
    finally:
        artifact_parquet.close()
    profile["artifact"] = {
        "path": os.path.basename(out_path),
        "rows": int(len(base_frame)),
        "columns": int(len(base_frame.columns)),
        "size_bytes": int(os.path.getsize(out_path)),
        "sha256": dataset_sha256,
        "parquet_row_group_size": int(CONFIG["PARQUET_ROW_GROUP_SIZE"]),
        "parquet_row_groups": parquet_row_groups,
        "all_columns_readback_verified": True,
        "pyarrow_version": pa.__version__,
    }
    component_artifact = pq.ParquetFile(component_path)
    try:
        component_row_groups = int(
            component_artifact.num_row_groups)
    finally:
        component_artifact.close()
    component_profile = build_component_supervision_profile(
        base_dataset_path=out_path,
        base_dataset_sha256=dataset_sha256,
        base_dataset_rows=len(base_frame),
        base_dataset_schema=CONFIG["DATASET_SCHEMA"],
        frame_convention=CONFIG["FRAME_CONVENTION_VERSION"],
        sidecar_path=component_path,
        sidecar_rows=len(component_frame),
        sidecar_size_bytes=os.path.getsize(component_path),
        sidecar_sha256=component_sha256,
        sample_id_order_sha256=sample_id_order_sha256,
        parquet_row_groups=component_row_groups,
        pyarrow_version=pa.__version__,
        label_replay_verified=True,
    )
    component_profile_path = os.path.join(
        out_dir, COMPONENT_SUPERVISION_PROFILE_FILENAME)
    component_profile_temp = (
        f"{component_profile_path}.{os.getpid()}.stage0.tmp")
    with open(
            component_profile_temp, "w",
            encoding="utf-8") as handle:
        json.dump(
            component_profile, handle, indent=2,
            ensure_ascii=False)
    os.replace(component_profile_temp, component_profile_path)
    profile["component_supervision"] = {
        "schema": component_profile["schema"],
        "profile_path": os.path.basename(
            component_profile_path),
        "artifact": component_profile["artifact"],
        "target_contract": component_profile[
            "target_contract"],
    }
    profile_path = os.path.join(out_dir, "generation_profile.json")
    profile_temp_path = f"{profile_path}.{os.getpid()}.stage0.tmp"
    with open(profile_temp_path, "w", encoding="utf-8") as _fh:
        json.dump(profile, _fh, indent=2, ensure_ascii=False)
    os.replace(profile_temp_path, profile_path)
    return profile_path

if __name__ == "__main__":
    t_start = time.time()
    print(f"[System] =============== 反装甲毁伤智能富化代理采样框架 (高纯度版 v2) ===============")
    print(f"[System] 已启用：按弹型精确 quota、稀有 root 发现、Phase 2 top-off、profile 门禁；生成端 CB 已关闭")
    # 修复 #6：全局种子注入 — 让 LHS、Gaussian 扰动、pandas.sample、AoA 接受/拒止全部可复现
    np.random.seed(CONFIG["RANDOM_SEED"])
    print(f"[System] 全局随机种子已锁定为 {CONFIG['RANDOM_SEED']}，本次运行可完整复现。")
    df_final = build_dataset_pipeline()
    out_path = "output/damage_dataset.parquet"
    # The official Parquet/profile contract must pass first.  Writing logit
    # metadata before this gate previously left a new 299999-row adjustment next
    # to an old v1 Parquet after rejection.
    profile_path = _write_dataset_with_profile(df_final, out_path)
    with open(profile_path, "r", encoding="utf-8") as profile_handle:
        written_profile = json.load(profile_handle)
    _emit_logit_adjustment(
        df_final,
        float(df_final.attrs.get("valid_prob_strict", CONFIG["VALID_PROB_STRICT"])),
        CONFIG["PHYSICAL_PRIOR"],
        os.path.join(os.path.dirname(out_path), "logit_adjustment.json"),
        dataset_sha256=written_profile["artifact"]["sha256"],
    )

    # 统计盘点
    c_all = len(df_final)
    c_crawl = df_final['is_crawled'].sum() if 'is_crawled' in df_final.columns else 0
    k2_true = (df_final['K2_prob'] >= CONFIG["VALID_PROB_STRICT"]).sum()
    c2_true = (df_final['C2_prob'] >= CONFIG["VALID_PROB_STRICT"]).sum()
    train_ess_ratio = float(
        written_profile.get("weighting", {}).get(
            "train_effective_sample_size_ratio", float("nan")))

    print(f"\n[System] 构建流圆满收官：总体耗时 {time.time()-t_start:.1f}s")
    print(f"         写出表体积: {c_all} 行 -> {out_path} (已满足精确总量与弹型配额)")
    print(f"         Profile: {profile_path}")
    print(
        "         部件监督: "
        f"{os.path.join(os.path.dirname(out_path), COMPONENT_SUPERVISION_FILENAME)} "
        "(训练期标签旁路，禁止作为模型输入)")
    print(f"         样本群画像: 后期爬行衍生保留 {c_crawl} 份 | "
          f"K2 致死级 {k2_true} 份 ({100*k2_true/max(c_all,1):.2f}%) | "
          f"C2 瘫痪级 {c2_true} 份 ({100*c2_true/max(c_all,1):.2f}%)")
    print(f"         合同校验: train ESS={100*train_ess_ratio:.2f}% | "
          f"K2 Phase-2 停止={100*CONFIG['K2_PHASE2_STOP_RATIO']:.1f}% | "
          f"K2 最终上限={100*CONFIG['K2_FINAL_MAX_RATIO']:.1f}%")
    print(f"         训练建议: focal_loss_gamma={CONFIG['FOCAL_LOSS_GAMMA']}，"
          f"loss_weight 动态范围 [{df_final['loss_weight'].min():.2f}, {df_final['loss_weight'].max():.2f}]\n")
