# 阶段 0 修改与检验记录

日期：2026-07-20
项目目录：`<PROJECT_ROOT>`

## 1. 本阶段结论

阶段 0 的代码改造已完成，并通过静态编译、10 项自动回归测试、原项目场景集成测试、性能基准和真实引擎端到端数据冒烟验证。

本阶段建立了以下基础合同：

1. 数字孪生外部接口统一采用 SI、NED 导航系、FRD 弹体系和标量在前四元数；历史毁伤引擎只在单一适配边界接收 cm。
2. 弹轴统一为 `+X_B`，姿态、速度、破片场和主动采样使用同一方向约定。
3. 装甲厚度文件真正参与破片侵彻与冲击波遮蔽，旋转几何使用真实相交测试。
4. 主动采样数据在爬行前按根种子固定 split，所有后代继承 split，消除近邻家族泄漏。
5. 标签改为多次蒙特卡洛仿真的显式序数超越概率，数据写出带 schema、谱系和 SHA-256 门禁。
6. 代理模型只使用 13 个终端时刻可观测特征；主动采样瞄准点派生量不再作为网络输入。

历史 `output/damage_dataset.parquet`、旧模型和旧部署包未被覆盖。它们使用旧坐标和旧特征合同，现会被 Stage-0 门禁明确拒绝。本次生成的 `output/stage0_smoke/` 仅用于验证，不可代替生产训练数据。

## 2. 修改前确认的问题

| 类别 | 修改前问题 | 风险 |
|---|---|---|
| 坐标与弹轴 | 仿真破片场以弹体 `+Y` 为轴，主动采样和计划中的飞控接口以 `+X` 为轴；`from_speed_and_attitude` 中俯仰角没有正确改变前向速度 | 仿真状态与采样特征错位，无法和六自由度模型可靠耦合 |
| 目标包络 | 数据采样器没有按所有真实部件几何计算车辆 AABB | 瞄准点和空间采样范围错误 |
| 装甲 | `armor.csv` 被加载但厚度没有进入实际侵彻计算 | 有/无装甲参数可能得到相同结果 |
| 几何 | 旋转长方体使用膨胀 AABB，旋转圆柱轴向未完整处理 | 产生假命中、错误法向和错误入射角 |
| 冲击波 | 遮蔽主要依赖部件 ID，而不是爆点到部件的几何视线 | 内外部件超压传播缺少空间一致性 |
| 概率 | 存在“只要有一次穿透，就把部分未穿透事件也纳入毁伤概率”的不连续逻辑；规则事件概率与序数等级概率混用 | 标签语义不稳定，阈值和损失函数含义不一致 |
| 主动采样 | 先爬行、再随机切分；同一根样本的高斯近邻后代可进入不同 split | 严重的家族泄漏和偏高测试指标 |
| 随机性 | 仿真随机种子依赖批内索引，批次重新从 0 计数 | 不同批次静默复用相同随机序列 |
| 特征泄漏 | `los_distance`、`impact_cosine` 由采样器选定的内部瞄准点计算并进入模型 | 网络可学习主动采样策略，而非仅学习终端状态到毁伤的映射 |
| 部署 | Logit adjustment 的先验语义与模型序数输出不一致，且阈值没有在同一偏移空间联合校准 | 离线指标与部署推理可能静默偏离 |
| 工程化 | 缺少统一依赖声明、数据完整性校验和 Stage-0 自动测试；批处理脚本硬编码其他机器的 Python 路径 | 难以复现和验收 |

## 3. 逐文件修改记录

### 3.1 新增文件

#### `coordinate_frames.py`

- 定义坐标版本 `stage0_ned_frd_v1`。
- 明确目标系 `T=[east/right, north/forward, up]`、导航系 NED、弹体系 FRD。
- 实现目标系与 NED 的向量转换。
- 实现 ZYX 欧拉角到 body-to-NED/body-to-target 旋转矩阵。
- 实现标量在前 `[w,x,y,z]` 四元数的归一化、欧拉角构造和旋转矩阵转换。
- 新增不可变 `TerminalEncounterState`：位置、速度、姿态、角速度、目标运动、弹型、起爆延迟和坐标版本均在一个 SI 状态对象中表达。
- 将 m→cm 转换限制在 `to_damage_engine_inputs()` 这一处历史引擎边界。

#### `validate_stage0_dataset.py`

- 新增不依赖 PyTorch 的数据集验收工具。
- 校验 profile/dataset/frame schema、必要字段、运动学有限值、弹型范围和非空数据。
- 校验 `sample_id` 唯一、根谱系不跨 split、爬行样本存在 `parent_id`。
- 校验 train/val/test 均存在。
- 校验八个显式序数概率位于 `[0,1]` 且 `P(L≥2)≤P(L≥1)`。
- 校验蒙特卡洛重复次数在表内一致并与 profile 一致。
- 校验 Parquet 行数、文件大小和 SHA-256。
- CLI 对不合格数据返回结构化 `FAIL` 和非零退出码。

#### `stage0_smoke.py`

- 新增可重复的小规模端到端验收入口。
- 使用真实物理采样器、真实车辆/装甲模型和真实毁伤引擎，串行执行以避免测试占满 CPU。
- 生成独立的 Parquet、generation profile 和默认禁用的 logit adjustment。
- 写出后立即运行完整 Stage-0 数据门禁。
- 产物标记为 `stage0_smoke_not_for_training`。

#### `tests/test_stage0.py` 与 `tests/__init__.py`

新增 10 项回归测试，覆盖：

1. 水平、垂直俯冲和向东飞行的姿态—速度关系。
2. 四元数与欧拉角旋转矩阵一致性。
3. SI 终端状态只进行一次 m→cm 适配。
4. 旋转长方体 OBB 消除膨胀 AABB 假命中。
5. `armor.csv` 厚度覆盖实际进入内部部件侵彻链路。
6. 采样 AABB 包含所有解析后的部件几何。
7. 规则事件到显式序数概率的转换与保序。
8. Phase 1 样本 ID、根谱系、预分配 split 和坐标版本。
9. 多次蒙特卡洛标签及谱系元数据透传。
10. Parquet/profile 原子写出、SHA-256、校验器通过，以及跨 split 根谱系拒绝。

#### `pyproject.toml`

- 声明 Python `>=3.10,<3.14`。
- 统一声明仿真、数据、训练、ONNX、UI 和测试依赖。
- 要求 `pyarrow>=21,<23`，解决历史数据由较新 Arrow 写出而当前环境版本过低的问题。
- 增加 pytest 配置。

#### `README.md`

- 记录 Stage-0 坐标、单位、schema、谱系和模型特征合同。
- 给出环境安装、单元测试、冒烟数据、数据校验和生产数据重建命令。
- 明确旧数据/旧模型不可与 Stage-0 合同混用。

### 3.2 修改文件

#### `sim_engine.py`

- `rotation_matrix_zyx` 统一调用 FRD body-to-target 旋转实现。
- `EncounterCondition.from_speed_and_attitude` 改用弹体 `+X_B` 前向，俯仰角现在正确改变速度方向。
- 增加 `EncounterCondition.from_terminal_state`，接收统一数字孪生终端状态。
- 破片轴向位置改为 X，破片环改到 Y-Z 平面，Taylor 角统一从 `+X_B` 弹轴度量。
- 长方体由 AABB 相交改为带旋转矩阵的 OBB slab 相交。
- 旋转圆柱按旋转后的轴向计算精确包络和射线相交。
- 新增统一 `ray_geometry` 分派。
- `armor.csv` 支持可选厚度列；没有厚度列时保留上装甲 15 mm、下装甲 10 mm 默认值。
- 同名装甲 CSV 厚度覆盖车辆模型的等效厚度，并通过 `HitRecord.self_thickness_mm` 进入侵彻计算。
- 新增爆点到部件的装甲视线厚度计算，冲击波优先使用几何遮蔽。
- 修复侵彻概率合成：只有 `margin>=1.0` 的真实穿透参与毁伤概率。
- 新增显式 `ordinal_probability_dict/vector`；K/M/F 的一级超越概率用事件概率 OR，C 使用嵌套语义；二级概率始终不高于一级。
- 综合评分改为归一化的期望序数严重度。
- 引擎输出同时保留旧规则事件向量和新序数概率字段，便于审计但避免语义混用。

#### `generate_dataset.py`

- 新增 `stage0_lineage_v1` 数据/profile schema 和坐标版本字段。
- 默认每个终端状态执行 3 次蒙特卡洛毁伤仿真；输出均值、标准差和实际重复次数。
- 仿真种子由全局种子、不可变 `sample_id` 和重复编号稳定派生，不再依赖批内索引。
- 修正采样器的 yaw/pitch 与速度方向公式，和 FRD/NED 坐标约定一致。
- 车辆 AABB 改为汇总全部解析后 OBB、旋转圆柱和拉伸多边形几何。
- 为每个初始样本生成唯一 `sample_id/root_seed_id`，并在主动爬行中保存 `parent_id/crawl_stage`。
- 根样本在爬行前通过稳定哈希分配 train/val/test；后代继承根 split。
- Phase 2 只能从 train 根谱系选择种子，val/test 保持独立参考分布。
- 弹型裁剪时优先保留 val/test 根样本，避免参考集被主动富化策略挤掉。
- 输出显式 `K/M/F/C_ge1_prob` 与 `ge2_prob`，等级直接由这些序数概率确定。
- `los_distance/impact_cosine/target_*` 仅保留为生成诊断元数据。
- profile 记录 schema、坐标版本、split 计数、谱系字段、蒙特卡洛次数和产物摘要。
- 写出前拒绝缺字段、重复 ID、混合坐标/schema、混合重复次数和跨 split 根谱系。
- Parquet 使用“进程专属临时文件→回读关键列→原子替换”；profile 同样原子替换。
- 写出后记录行数、字节数和 SHA-256。
- Logit adjustment 先验改由序数超越硬标签统计；schema 更新为 `ordinal_exceedance_v2`，在未和阈值联合校准前显式 `enabled=false`。

#### `nn_dataset.py`

- 训练入口要求 Stage-0 profile/dataset/frame schema；旧数据在读 Parquet 前被拒绝。
- 训练前校验产物大小和 SHA-256，读入后校验 profile 行数、唯一 sample ID 和蒙特卡洛次数。
- 模型输入缩减为 13 个可部署观测量：位置 3、速度 3、姿态正余弦 6、速度模 1。
- 删除训练侧对瞄准点派生特征的补算，避免它们重新进入特征链路。
- 软标签只允许从显式 `*_ge1_prob/*_ge2_prob` 读取，不再把规则事件概率误解释为序数概率。
- 数据切分只接受预分配 `split_role` 和 `root_seed_id`，禁止回退到空间桶近似切分。
- split manifest 标记为 `preassigned_root_seed_v1`，复用前校验数据签名。
- 验证三组根谱系两两无交集。

#### `nn_model.py`

- 默认输入维度从 15 改为 13。
- 注释明确瞄准点派生量禁止进入模型；训练/导出仍按活动特征数显式构造网络。

#### `nn_train.py`

- 更新网络构造说明，取消固定“15D”合同，使用 Stage-0 活动特征列。

#### `nn_eval_export.py`

- 模型输入维度改为 `len(FEATURE_COLUMNS)`。
- 删除误导性的 `FEATURES_15D_ORDER/FEATURES_13D_ORDER`，使用活动特征顺序。
- ONNX 输入名由 `encounter_features_15d` 改为 `encounter_features`。
- 部署配置写出真实 `feature_count` 和 `features_active_order`。
- Logit adjustment 只有在 schema 正确且 `enabled=true` 时生效，否则强制使用零偏移并输出说明。
- 部署配置增加 `logit_adjustment_enabled`，避免推理侧猜测是否启用。

#### `run .bat`

- 删除其他用户目录下 Python 3.9 的硬编码路径，改为使用当前激活环境的 PATH。
- 依赖安装改为 `pip install -e ".[ui,test]"`。
- 测试入口改为 Stage-0 自动回归测试。
- 增加 Stage-0 端到端冒烟数据菜单项。

#### `test_v2.py`

- 将末尾 emoji 成功标记改为 ASCII `[OK]`，避免 Windows 非 UTF-8 控制台编码错误。

### 3.3 新增验证产物

目录 `output/stage0_smoke/`：

- `damage_dataset.parquet`：24 行真实引擎冒烟数据。
- `generation_profile.json`：Stage-0 schema、谱系/分割统计和 SHA-256。
- `logit_adjustment.json`：序数先验统计，明确禁用。

这些文件仅用于端到端验收，不用于模型训练或指标报告。

## 4. 验证结果

### 4.1 静态编译

命令：

```powershell
python -m py_compile coordinate_frames.py sim_engine.py generate_dataset.py validate_stage0_dataset.py stage0_smoke.py nn_dataset.py nn_model.py nn_train.py nn_eval_export.py
```

结果：通过，无语法错误。

### 4.2 自动回归测试

命令：

```powershell
python -m unittest discover -s tests -v
python -m pytest -q
```

结果：`10/10` 通过，pytest 为 `10 passed`。

### 4.3 原项目场景集成测试

命令：

```powershell
python -X utf8 test_v2.py
```

结果：通过，末尾输出 `[OK] 全部测试完成`。代表场景“中型前端起爆、垂直俯冲、dz=200 cm、100 m/s”得到：

- 150 枚破片；
- 194 次命中；
- 31 次穿透；
- 38/70 个部件毁伤；
- 综合评分 0.4869。

这些数值用于锁定本次代码状态，不表示实装武器试验标定结果。

### 4.4 性能基准

命令：

```powershell
python benchmark_sim_engine.py --runs 20 --warmup 3 --seed 42
```

结果：

- 单次仿真中位数：86.014 ms；
- 平均值：83.557 ms；
- 标准差：9.535 ms；
- 最小/最大：66.376/99.730 ms。

该基准包含新的 OBB 和冲击波几何遮蔽计算，排除模型加载时间。

### 4.5 真实引擎端到端数据验证

命令：

```powershell
python stage0_smoke.py --rows 24 --mc-replicates 2 --output-dir output/stage0_smoke
python validate_stage0_dataset.py output/stage0_smoke/damage_dataset.parquet
```

结果：`PASS`。

| 指标 | 结果 |
|---|---:|
| 行/列 | 24 / 73 |
| 根谱系 | 24 |
| train/val/test | 17 / 3 / 4 |
| 跨 split 根谱系 | 0 |
| 序数概率保序违规 | 0 |
| 每状态蒙特卡洛重复 | 2 |
| SHA-256 | `a50ddcb0c24ce36ba79999d07c9389d0ed40c92a77f3d42838170f924f9124ee` |

### 4.6 旧数据拒绝验证

命令：

```powershell
python validate_stage0_dataset.py output/damage_dataset.parquet
```

结果：按预期返回 `FAIL` 和非零退出码，原因是旧 profile 为 `v5_per_munition_topoff_small_c1_m_balance`，不是 `stage0_lineage_v1`。门禁在读取旧 Parquet 表体之前生效。

## 5. 兼容性与迁移要求

- 旧 15 维模型权重不能加载到新的 13 维输入网络。
- 旧 scaler、阈值、logit adjustment、ONNX 和部署配置不能和新模型混用。
- 旧 30 万行数据没有不可变谱系、预分配 split、新坐标版本和显式序数标签，因此不能“就地补列”冒充 Stage-0 数据。
- 正确迁移顺序是：重建生产数据 → 数据门禁 → 重新训练 → 在同一推理空间校准阈值/可选 logit adjustment → 重新评估 → 重新导出 ONNX。
- `TerminalEncounterState` 是后续末制导/飞控数字孪生向毁伤引擎交付终端状态的唯一推荐接口。

## 6. 尚未执行的计算型工作

以下工作没有在本次阶段 0 修改中伪装为已完成：

1. 未用新物理合同重新生成生产级 300,000 行数据集。
2. 未在新数据上重新训练神经网络。
3. 未重新联合校准阈值和 logit adjustment。
4. 未重新导出或验证 Stage-0 ONNX 部署包。

原因是这些步骤需要生产级长时间计算，并且当前解释器为 Python 3.13.5，环境中未安装 PyTorch，PyArrow 为 19.0.0，而新可复现配置要求 PyArrow 21+。本次已通过无需训练框架的真实引擎端到端冒烟数据证明生成、写出和门禁链路可运行。执行生产重算前应先按 `pyproject.toml` 建立隔离环境，并将新产物写到独立目录，经门禁通过后再切换正式路径。

## 7. 阶段 0 验收状态

| 项目 | 状态 |
|---|---|
| 坐标/单位/姿态合同 | 已完成并测试 |
| 弹轴与采样方向统一 | 已完成并测试 |
| 几何、装甲和冲击波遮蔽 | 已完成并测试 |
| 序数概率语义 | 已完成并测试 |
| 谱系与无泄漏 split | 已完成并测试 |
| 数据原子写出与完整性门禁 | 已完成并测试 |
| 可观测特征和部署 schema | 已完成静态验证 |
| 小规模真实引擎端到端链路 | 已完成并通过 |
| 生产级数据重建 | 待算力环境执行 |
| 新模型训练/校准/ONNX | 依赖生产级新数据，尚未执行 |
