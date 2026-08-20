# Stage-0 v2 数据集可用性增强修改记录

> 后续 299999 行生产门禁失败所触发的二次修复（精确配额、ESS、失败产物合同、起爆方向/破片锥面感知 HUNT 与真实可达性探测）记录在 `DATASET_GENERATOR_REPAIR_RECORD.md`。本文保留第一轮 v2 升级记录。

## 1. 修改目标

本轮修改针对 `DATASET_VALIDATION_REPORT.md` 中对新生成 30 万行数据集的审计结论，重点解决以下问题：

1. 主动采样后代集中在少数 root，行数充足但独立信息量不足。
2. 固定 3 次蒙特卡洛对边界样本和高方差样本不够稳定。
3. 旧实现先平均 K1/K2 等原始规则事件，再做非线性 OR，存在聚合偏差。
4. 爬行子样本继承父样本的混合 `loss_weight`，但子样本的采样提议已发生变化。
5. 生成器类别平衡、训练重采样、`pos_weight`、focal loss 等多种机制叠加，可能造成极端梯度权重。
6. Parquet 只回读少数关键列，无法发现其他列的物理编码兼容问题。
7. 固定的验证/测试稀有格数量门槛会诱导把主动采样数据注入 holdout，破坏自然分布评估。
8. 原数据无法区分破片和冲击波对最终毁伤标签的贡献。

本轮没有修改冲击波、侵彻、装药等毁伤物理参数。当前物理模型是否符合实测规律，仍需依靠外部试验数据校准。

## 2. 数据合同升级

- 数据 schema：`stage0_lineage_v1` → `stage0_lineage_v2`。
- 坐标合同保持：`stage0_ned_frd_v1`。
- 旧 v1 数据、旧模型权重和旧部署包不能直接进入 v2 训练流程。
- v2 新增自适应 MC、机制分解、采样权重分量、家族规模和生成环境 provenance。

## 3. 主动采样与家族多样性

修改文件：`generate_dataset.py`

### 3.1 root 级限流

- 最终每个 `root_seed_id` 最多保留 64 行。
- 每个爬行阶段、每个 root 最多生成 8 个候选。
- root 超额裁剪时优先保留初始 root 行，再以稳定随机种子抽取后代。
- 新增 `family_size` 和 `family_weight`；大家族按 `sqrt(8 / family_size)` 降权，最大为 1。

### 3.2 root 均衡爬行

- 边界种子先按 `root_seed_id` 去重，再做标准化特征空间去重。
- 爬行候选按独立 root 轮转产生，不再对所有种子行无约束有放回抽样。
- 多阶段爬行保留 lineage 和预分配 split，禁止同一家族跨 split。

### 3.3 新 root 发现

- 当某个 top-off 任务的训练种子 root 少于目标数时，先使用对应的 `K1/K2/M/F/C_HUNT` 层发现独立新 root。
- 新 root 的 `split_role` 仍由稳定哈希预分配；只有 train root 可用于主动 top-off。
- val/test 不注入主动采样后代，继续表示自然 root 分布。
- 对审计中坍缩明显的弹型×任务增加了仅影响采样提议的 HUNT 窄带覆盖；没有改变毁伤计算公式。

### 3.4 生产可用性门禁

当最终数据不少于 50,000 行时，写出前按“弹型×任务×序数头”检查适用训练格：

- 独立正例 root 不少于 16；
- 有效正例 root 数不少于 8；
- 单个最大 root 的正例占比不高于 25%；
- 配置的结构零假设若被正例反驳，会报告配置矛盾。

门禁失败时不覆盖正式 Parquet。小规模 smoke 数据只报告这些指标，不执行生产拒绝。

## 4. 蒙特卡洛标签修正

修改文件：`generate_dataset.py`、`sim_engine.py`

### 4.1 自适应重复次数

- 最少 3 次，最多 9 次。
- 3 次后若任一序数概率距 0.5 不超过 0.15，或重复间标准差不小于 0.20，则继续采样到最多 9 次。
- 每行保存：
  - `label_mc_min_replicates`
  - `label_mc_max_replicates`
  - `label_mc_replicates`（实际执行次数）
- profile 保存实际次数直方图，不再要求全表重复次数一致。

### 4.2 正确的序数聚合

- 每次仿真先计算 `P(L>=1)` 与 `P(L>=2)`。
- 对每次仿真的序数概率直接求均值和标准差。
- 不再对已平均的 K1/K2、M1/M2、F1/F2 规则事件二次执行非线性 OR。
- 新增 8 个 `*_ge1/2_prob_std` 序数不确定性字段；原始规则事件均值和标准差继续保留作诊断。

### 4.3 机制分解

毁伤引擎在同一次部件仿真结果上额外计算：

- `damage_tree_fragment`：仅破片毁伤概率；
- `damage_tree_shockwave`：仅冲击波毁伤概率。

数据表新增 16 个 `fragment_*_ge1/2_prob`、`shock_*_ge1/2_prob` 字段，以及两个机制综合分数。该分解不额外生成破片场，计算开销很小。

## 5. 权重体系修正

修改文件：`generate_dataset.py`、`nn_dataset.py`

### 5.1 生成端拆分记录

原先不透明的 `loss_weight` 拆分为：

- `aoa_accept_prob`
- `aoa_ipw`，上限 20
- `physics_weight`
- `active_sampling_weight`
- `family_weight`
- `class_balance_weight`
- `loss_weight_raw`
- `loss_weight`

爬行子样本重新计算 AoA 和物理权重，不再继承父样本的混合权重。当前无法严格估计主动提议相对自然分布的密度比，因此 `active_sampling_weight` 明确保持 1，而不是伪造校正量。

最终权重按 split 均值归一化，并裁剪到 `[0.05, 20]`。生成器侧 class-balanced 权重默认关闭。

### 5.2 训练端避免重复放大

- 默认保留四弹型等量 batch。
- 弹型内部稀有类重采样默认关闭。
- adaptive train loss balance 默认关闭。
- `pos_weight` 遇到某弹型某头零正例时返回 1，不再把不可观测正类错误地推到上限。
- focal loss、per-munition `pos_weight` 等现有训练机制仍可通过消融配置比较。

## 6. 验证集、测试集与结构零

修改文件：`nn_dataset.py`、`validate_stage0_dataset.py`

- 删除旧的 Small×C1、Small×M 在 val/test 中必须达到固定数量的硬编码门槛。
- 生产硬门禁改为训练集独立 root 多样性。
- val/test 保持自然、独立 root 分布；适用格无正例时只给出警告，建议另建独立 challenge set，不允许用训练后代污染 holdout。
- `ORDINAL_APPLICABILITY` 是根据当前 30 万行审计结果形成的结构零假设，不是经试验确认的物理事实；获得更高保真证据后应更新该配置。

## 7. Parquet、profile 与先验语义

修改文件：`generate_dataset.py`、`validate_stage0_dataset.py`、`audit_stage0_dataset.py`、`nn_eval_export.py`、`pyproject.toml`

### 7.1 写出与回读

- 固定 Parquet row group 上限为 50,000 行。
- 临时文件写出后，使用 PyArrow 遍历所有 row group 和所有列，而不是只读取 5 个关键列。
- 全列回读成功后才原子替换正式文件。
- profile 保存 PyArrow 版本、Python/NumPy/Pandas 版本、平台、配置 SHA-256、row group 数、文件 SHA-256。
- 读取时优先 PyArrow；遇到版本解码问题时使用 fastparquet 独立回退。
- 项目依赖新增 `fastparquet>=2024.11,<2027`；PyArrow 仍固定在 `>=21,<23`，减少写读版本漂移。

### 7.2 Logit adjustment

- 把历史原始规则事件先验转换为显式序数先验：`K_ge1_prob` … `C_ge2_prob`。
- 评估导出端同步读取新键名。
- adjustment 继续保持 `enabled=false`；只有在同一 shifted-logit 概率空间联合校准阈值后才允许启用。

## 8. 修改文件清单

- `generate_dataset.py`
- `sim_engine.py`
- `validate_stage0_dataset.py`
- `audit_stage0_dataset.py`
- `nn_dataset.py`
- `nn_eval_export.py`
- `stage0_smoke.py`
- `tests/test_stage0.py`
- `pyproject.toml`
- `README.md`
- `DATASET_USABILITY_MODIFICATION_RECORD.md`（本文档）

## 9. 验证结果

### 9.1 静态编译

```powershell
python -m py_compile generate_dataset.py sim_engine.py validate_stage0_dataset.py audit_stage0_dataset.py nn_dataset.py nn_eval_export.py stage0_smoke.py
```

结果：通过。

### 9.2 单元测试

```powershell
python -m pytest -q
```

结果：`12 passed`。覆盖坐标合同、旋转几何、装甲覆盖、序数概率、lineage、机制字段、root family 限流、每阶段 root 子代限流、原子写出与哈希验证。

### 9.3 原物理回归

```powershell
python test_v2.py
```

结果：退出码 0，原有物理回归通过。

### 9.4 真实引擎端到端 smoke

```powershell
python stage0_smoke.py --rows 24 --mc-replicates 2 --output-dir output/stage0_smoke_v2
```

结果：通过。

- 24 行、108 列；
- 24 个独立 root；
- train/val/test = 18/4/2；
- 跨 split root = 0；
- 序数保序违规 = 0；
- Parquet 全列回读、文件大小和 SHA-256 校验通过。

smoke 使用 `min=max=2` 的固定重复次数以节省测试时间；生产配置仍为自适应 3–9 次。

### 9.5 深度审计

```powershell
python audit_stage0_dataset.py output/stage0_smoke_v2/damage_dataset.parquet --output output/stage0_smoke_v2/audit.json
```

结果：通过；profile 行数、文件大小、SHA-256、split 计数和 MC 直方图全部匹配。

## 10. 生产数据重新生成与验收

本轮没有覆盖现有 `output/damage_dataset.parquet`。原 30 万行文件仍是 v1 审计对象，不能直接用于 v2 训练。

安装或同步依赖后执行：

```powershell
python -m pip install -e ".[test]"
python generate_dataset.py
python validate_stage0_dataset.py output/damage_dataset.parquet
python audit_stage0_dataset.py output/damage_dataset.parquet --output output/stage0_dataset_audit_v2.json
```

只有生成 profile 中 `usability_gate.passed=true` 且验证器返回 `PASS` 后，才建议启动 `nn_train.py`。

## 11. 尚未由本轮解决的问题

1. HUNT 采样窄带是根据现有仿真审计形成的提议分布优化，不等价于物理标定。
2. `ORDINAL_APPLICABILITY` 仍需领域专家或高保真/实测数据确认。
3. 冲击波遮挡、材料阈值和零命中时 M/F 高毁伤现象没有被任意改写；应单独用实验或高保真模型校准。
4. 自适应 MC 会提高边界样本的生成成本，生产运行时间将高于固定 3 次方案。
5. 自然 val/test 中极稀有格可能仍为空；应建立与训练 root 完全隔离的定向 challenge set，用于能力边界评估，而不是替代自然测试集。
