# Stage-0 数据集生成器生产失败修复记录

日期：2026-07-21
适用数据合同：`stage0_lineage_v2`

## 1. 本轮输入与结论

生产运行曾得到 299999 行候选，并在写出门禁处失败：

- Small 的 `C>=1` 只有 8 个独立正例 root、有效 root 2.64，最大 root 占比 58.3%；
- Med-LM 的 `K>=2` 没有独立正例 root；
- `loss_weight` 达到 `[0.05, 20]` 两端，ESS 仅为 `100769/299999=33.6%`；
- 随后验证的 `output/damage_dataset.parquet` 实际仍是旧 `stage0_lineage_v1` 文件，并非门禁拒绝的 v2 候选；
- 门禁前写出的 `logit_adjustment.json` 来自被拒绝候选，与旧 Parquet 不属于同一数据版本。

因此，该次运行没有产生可训练的 v2 数据集，旧 Parquet、旧 profile 和新 logit 文件不能混合使用。

## 2. 根因分析

### 2.1 稀有正例发现不足

旧 fresh-root 逻辑只进行一次约 512 个候选的探测，且按原始候选数而不是物理过滤后的实际仿真数消耗预算。稀有任务常常只剩几十个 train 候选进入毁伤引擎，不能稳定发现 16 个独立正例 root。

### 2.2 HUNT 采样与破片物理错配

`sim_engine.generate_fragment_field` 中：

- FRONT 起爆战斗部的主破片沿机体 `-X_B` 方向形成锥面；
- REAR 起爆战斗部的主破片沿 `+X_B` 方向形成锥面；
- 破片位于约 26–35° 的主锥面上，不在锥轴上。

旧 HUNT 却让所有弹型的机体/速度 `+X_B` 直接指向关键组件，同时把关键组件放在锥轴上。对于 Small、Med-LM 和 Heavy 等 FRONT 起爆弹，这会把主破片锥指向目标外侧；即使方向翻转，锥轴直接穿过组件也会让破片环绕开组件。

### 2.3 K 类靶池与毁伤树不一致

毁伤树中 K1 仅由油箱 `id=3` 触发，K2 仅由弹药架 `id=46` 触发。旧 K2 靶池还包含供弹装置和辅助弹药等不触发 K2 的组件，浪费了大部分定向预算。

### 2.4 候选几何过滤效率过低

车辆 AABB 的横向范围约为 `x=[-165,165] cm`。旧稀有探测仍在半径球体内采样，很多 `r=170–250 cm` 的点实际位于车辆 AABB 内，随后被物理过滤。实测 2048 个候选只剩 128–180 个 train 样本进入仿真。

### 2.5 配额、权重和产物提交缺少硬合同

- 固定轮数回填仍可能少 1 行；
- Med-LM top-off 曾把预算分给结构零 C2，而没有稳定补 K2；
- 只有全局每 root 64 行上限，没有“弹型×序数头”的正例家族上限；
- 权重只做硬裁剪，没有 ESS 下限；
- logit 元数据在正式数据门禁前写出；
- 失败日志只预览少数原因，没有持久化完整拒绝报告。

## 3. 代码修改

### 3.1 稀有 root 循环发现

修改 `generate_dataset.py`：

- 新增 `discover_fresh_target_roots`，按任务循环生成相互独立的 train root；
- 同时满足边界种子 root 和严格正例 root 目标后才提前停止；
- 默认每批 1024、每任务最多 8192 个原始候选、最多 8 轮；
- profile 保存每任务轮数、请求数、实际仿真数、前后 root 数和预算耗尽状态；
- fresh 正例优先每 root 保留 1 行，再由多阶段爬行扩展边界。

### 3.2 起爆方向感知和锥面感知 HUNT

- 新增 `HUNT_AXIS_SIGN_BY_MUNITION`：FRONT 弹为 `-1`，REAR 弹为 `+1`；
- 新增按弹型配置的有效主破片锥角：Small 35°、Med-LM 34°、Med-RD 26°、Heavy 30°；
- HUNT 先按起爆方向确定锥轴，再旋转有效 Taylor 锥角，使关键组件落在主破片锥面上；
- HUNT 姿态只保留 ±5° 小攻角扰动，普通 Phase-1 层仍保留原 ±25° 广域覆盖；
- 新增诊断列 `fragment_aim_sign`、`cone_aim_angle_deg`；
- Phase-2 爬行按 `fragment_aim_sign` 判断正确方向，不再误删 FRONT 弹的后向瞄准后代。

这些修改只改变主动采样提案，不改变破片生成、装甲侵彻或毁伤树判据。

### 3.3 稀有探测外壳采样

- fresh-root 探测在车辆左右侧 AABB 外 16–80 cm 处生成起爆点；
- 沿目标前后和高度方向分别保留 ±180 cm、±120 cm 切向覆盖；
- 保证 `z>=0`、不进入 AABB 且总半径不超过 500 cm；
- 该外壳提案仅用于 Phase-2 fresh-root 探测，普通 Phase-1 的径向 LHS 分布不变；
- 新增诊断列 `sampling_geometry=fresh_lateral_shell`。

### 3.4 靶点和 top-off 配置

- `K1_CRITICAL_COMPONENT_IDS=[3]`；
- `K2_CRITICAL_COMPONENT_IDS=[46]`；
- Med-LM top-off 在结构零假设修订后更新为：
  `C2 20% / K2 15% / F2 20% / M2 10% / M1-only 15% / K1 20%`；
- 启动仿真前执行配置一致性检查：弹型比例、top-off 比例、结构零冲突、HUNT 方向/锥角、外壳范围和 ESS 目标不合法时立即失败。

### 3.5 正例家族限流和可用性门禁

- 全局每 root 仍最多 64 行；
- 每个“弹型×序数头”每 root 最多保留 8 个正例，包含其他任务意外带入的正例；
- 生产门禁检查所有适用序数头：
  - train 正例行数不少于 128；
  - 正例 root 不少于 16；
  - 负例 root 不少于 16；
  - 有效正例 root 不少于 8；
  - 最大正例 root 占比不超过 25%；
- 另对实际等级 L0/L1/L2 检查独立 root、有效 root 和最大家族占比；
- 结构零单元一旦观察到正例，作为配置矛盾拒绝，而不是静默忽略。

### 3.6 精确行数与弹型配额

- 最终缺口改为最多 10 轮独立 Phase-1 回填；
- 每轮至少请求 256 个候选，不能因单轮为空提前放弃；
- 写出前硬断言总行数与各弹型配额；
- 默认生产目标必须是 300000 行，四弹型各 75000 行，不再允许 299999 行进入写出阶段。

### 3.7 权重稳定性

- 保留 `[0.05,20]` 裁剪；
- 若 ESS 比例低于 50%，对原始权重执行幂温度缩放，并二分求取尽量接近 1 的温度指数；
- profile 保存温度指数、ESS 比例、触底行数和触顶行数；
- 生产门禁要求 `ESS/N >= 50%`；
- 生成器类别平衡保持关闭，训练侧只保留一套类别平衡机制。

### 3.8 正式产物和失败诊断

- 先完成数据/profile 门禁与 Parquet 全列回读，再写 `logit_adjustment.json`；
- logit 的 `pi_train` 只使用 train split，不再使用全表；
- logit 写入正式数据 SHA-256，验证器检查其与 profile/Parquet 是否一致；
- 门禁失败时不写候选 Parquet，不覆盖正式数据，并原子写出 `output/generation_profile.rejected.json`，保存全部失败原因；
- `audit_stage0_dataset.py` 新增 `contract_status` 和 artifact identity，旧 v1 文件会显示 `LEGACY_OR_SCHEMA_MISMATCH`，避免把 `AUDIT_COMPLETE` 误解为 v2 可训练。

### 3.9 数值稳定性

`sim_engine.py` 的毁伤树比例规则对 sigmoid 指数自变量裁剪到 `[-60,60]`，消除极端样本的 `exp overflow` 警告，不改变饱和区间的概率含义。

## 4. 新增工具

新增 `stage0_reachability_probe.py`：

- 默认探测 `Small/C>=1`、`Med-LM/K>=2` 和 `Med-LM/C>=2`；
- 使用与生产一致的独立 fresh root、物理过滤和 3–9 次自适应蒙特卡洛；
- 不写、不修改正式训练数据；
- 输出严格正例 root、有效 root、最大 root 占比、最高/99 分位概率和候选预算状态。

## 5. 验证结果

### 5.1 静态编译与自动化测试

```powershell
python -m py_compile sim_engine.py generate_dataset.py stage0_reachability_probe.py validate_stage0_dataset.py audit_stage0_dataset.py stage0_smoke.py
python -m pytest -q
```

结果：`23 passed`。覆盖配置冲突、精确配额、全门禁缩小管线、Med-LM/C2
适用性和定向配额、C2 聚类靶点、任务正例家族限流、循环 root 发现、
外壳几何、起爆方向和锥角、ESS 温度缩放、logit 哈希绑定、拒绝 profile
持久化等。

### 5.2 原毁伤物理回归

```powershell
python test_v2.py
```

结果：退出码 0，原有集成回归全部通过。

### 5.3 真实引擎稀有标签可达性

```powershell
python stage0_reachability_probe.py --max-candidates 1024 --batch-size 512 --required-roots 16
```

首轮 512 个原始候选即同时通过：

| 单元 | 实际仿真 train root | 严格正例 root | 有效正例 root | 最大 root 占比 | 最大概率 |
|---|---:|---:|---:|---:|---:|
| Small / C>=1 | 403 | 76 | 76.0 | 1.32% | 1.0000 |
| Med-LM / K>=2 | 410 | 54 | 54.0 | 1.85% | 0.9999 |

报告：`output/stage0_reachability_probe.json`。

### 5.4 真实引擎端到端 smoke

```powershell
python stage0_smoke.py --rows 24 --mc-replicates 2 --output-dir output/stage0_smoke_v5
python validate_stage0_dataset.py output/stage0_smoke_v5/damage_dataset.parquet
python audit_stage0_dataset.py output/stage0_smoke_v5/damage_dataset.parquet --output output/stage0_smoke_v5/audit.json
```

结果：

- 24 行、111 列、24 个独立 root；
- 跨 split root 为 0，序数保序违规为 0；
- profile/Parquet 行数、大小和 SHA-256 一致；
- logit SHA-256 与数据一致；
- 验证器 `PASS`，审计合同状态 `CURRENT_V2`。

smoke 小于 50000 行，所以只报告生产可用性指标而不执行 128 行/16 root 的生产拒绝门禁。

## 6. 重新生成和验收

本次修改没有运行完整 30 万行生产生成，也没有覆盖当前 `output/damage_dataset.parquet`。该路径若仍指向 v1 文件，在新数据成功提交前不能用于训练。

建议执行：

```powershell
python stage0_reachability_probe.py
python generate_dataset.py
python validate_stage0_dataset.py output/damage_dataset.parquet
python audit_stage0_dataset.py output/damage_dataset.parquet --output output/stage0_dataset_audit_v2.json
```

完整数据可训练的必要条件：

1. 生成器正常结束，最终总数 300000，四弹型各 75000；
2. `generation_profile.json` 中 `profile_schema=stage0_lineage_v2`；
3. `usability_gate.enforced=true` 且 `passed=true`；
4. `validate_stage0_dataset.py` 返回 `PASS`；
5. 审计 `contract_status=CURRENT_V2`，profile 的行数、大小、SHA-256、split 和 MC 直方图全部匹配；
6. logit adjustment 状态为 `sha256_match`。

若生成仍被拒绝，应查看 `output/generation_profile.rejected.json` 的完整 `rejection.reasons` 和 `phase2_root_discovery`，不要继续校验旧 Parquet，也不要启动训练。

## 7. 修改文件清单

- `generate_dataset.py`
- `sim_engine.py`
- `validate_stage0_dataset.py`
- `audit_stage0_dataset.py`
- `stage0_smoke.py`
- `stage0_reachability_probe.py`（新增）
- `tests/test_stage0.py`
- `README.md`
- `DATASET_GENERATOR_REPAIR_RECORD.md`（本文档）

## 8. 边界说明

本轮已证明两个原失败单元在当前真实毁伤引擎中可达，并证明生成器能够形成满足独立 root 门槛的候选。最终 30 万行是否通过全部弹型×任务×等级门禁，仍以重新运行生成器得到的正式 profile 为准。HUNT 优化是主动采样提案，不是对真实战场先验分布或毁伤参数的实验标定；物理可信度仍需高保真模型或实测数据校准。

## 9. Med-LM/C2 结构零假设修订（2026-07-22）

锥面感知 HUNT 修复后的 30 万行候选只触发了一项门禁失败：Med-LM 的
`C>=2` 原配置为结构零，但 train split 实际观察到 29 个正例。拒绝 profile
证明这 29 行分别来自 29 个独立 root，有效 root 数为 29.0，最大 root 占比
仅 3.45%；全表共有 33 个 Med-LM/C2 正例。因此这不是单一家族复制或偶发
爬行坍缩，而是当前毁伤引擎中的可达状态，旧结构零假设已被推翻。

对应修订：

- `ORDINAL_APPLICABILITY[1]["C"]` 改为 `[True, True]`；
- Med-LM Phase-2 新增 20% `C2_prob` 定向补样，并从已经充足的 F2/M2
  预算中让出比例；
- 保留 train 正例至少 128 行、正例 root 至少 16、有效 root 至少 8、
  最大 root 占比不超过 25% 的原门禁，不通过删样或降低阈值规避问题；
- 可达性工具新增 `med_lm_c2` 目标，生产前可独立验证该单元。

`C2` 要求至少 60% 乘员（当前目标为 10 人中的 6 人）毁伤，单人靶点不适合
该联合事件。为此新增专用 `C2_HUNT`：对每个乘员构造其最近 6 人的聚类质心，
再加入全体乘员质心，共得到 7 个稳定去重靶点；其余起爆方向、Taylor 锥面、
fresh-root 外壳、3–9 次自适应蒙特卡洛和严格标签判定均保持不变。

修订前先按通用单人 `C_HUNT` 完整使用 8192 个候选进行对照探测，结果只有
15 个严格正例 root，距离门槛还差 1 个，因此没有降低门槛或扩大预算，而是
按 C2 联合事件的定义修正靶点提案。该对照也说明新增层不是为了绕过门禁，
而是修复主动采样几何与标签语义的错配。

真实验收执行：

```powershell
python stage0_reachability_probe.py --targets med_lm_c2 --output output/stage0_reachability_probe_med_lm_c2.json
```

默认候选预算为 8192。探测在请求 6656 个候选后提前停止，实际仿真 5331 个
train root，得到 19 个严格正例 root，超过 16-root 门槛；有效正例 root 为
19.0，最大正例 root 占比为 5.26%，候选预算未耗尽，最终状态为 `PASS`。
这验证的是当前毁伤引擎中的物理可达性和独立 root 多样性，不改变全量生产中
128 个 train 正例行、16 个正例 root、8 个有效 root 和 25% 最大占比的门禁。

## 10. 30 万行正式数据二次审计修订（2026-07-25）

2026-07-24 正式生成结果满足 300000 行、四弹型各 75000、v2 schema、无跨
split root、可用性门禁、Parquet/profile/logit SHA-256 一致等原合同。进一步
直接读取 Parquet 后发现三个不会造成文件损坏、但会影响训练或评估解释的问题：

1. 全表权重 ESS 恰好为 50.000%，但 train split ESS 只有 49.743%；
2. `K2_GLOBAL_TARGET_RATIO=3%` 的旧命名暗示最终比例目标，实际 K2 为
   21567/300000=7.19%；代码实际上只把 3% 用作 Phase-2 停止继续补样的阈值；
3. 自然留出集中 Med-LM/C2 只有 val=3、test=4 个正例，不足以稳定计算分弹型
   Recall、AUPRC 或校准指标。

对应代码修改：

- 权重温度搜索改为直接优化 train split ESS，生产目标设为
  `MIN_WEIGHT_ESS_RATIO + WEIGHT_ESS_TARGET_MARGIN = 50.5%`；profile、
  验证器和审计器均增加分 split ESS，并以 train ESS≥50%作为硬门禁；
- 3% 配置改名为 `K2_PHASE2_STOP_RATIO`，新增
  `K2_FINAL_MAX_RATIO=8%`。最终回填会过滤导致越过上限的 K2 候选，profile
  记录停止阈值、最终上限、实际行数和实际比例，验证器重新计算并核对；
- 新增 `build_stage0_c2_challenge.py`。它使用独立随机种子命名空间和
  `C2_HUNT`，为 Med-LM、Med-RD、Heavy 分别构建一行一 root 的严格正例与
  难负例，并可验证与正式数据 root 零重叠；
- challenge profile 明确标记
  `root_independent_rare_event_discrimination_not_calibration`。它不改变自然
  val/test，不得用于估计部署先验、选择生产阈值或做概率校准；
- `stage0_smoke.py` 在写 profile 前也执行正式权重合成，确保小规模真实引擎
  测试能覆盖新的 ESS 元数据合同；小于生产门禁规模时只报告 K2 比例，不强制
  8% 上限。

验证结果：

```powershell
python -m py_compile generate_dataset.py validate_stage0_dataset.py audit_stage0_dataset.py stage0_smoke.py build_stage0_c2_challenge.py
python -m pytest -q
python stage0_smoke.py --rows 24 --mc-replicates 2 --output-dir output/stage0_smoke_post_audit
python audit_stage0_dataset.py output/stage0_smoke_post_audit/damage_dataset.parquet --output output/stage0_smoke_post_audit/audit.json
```

- 自动化测试：`26 passed`；
- 真实引擎 smoke：24 行、24 个独立 root、跨 split root=0、train ESS=71.6%、
  验证器 `PASS`；
- smoke 审计：`CURRENT_V2`，profile/Parquet 行数、大小、SHA-256、split 和
  MC 直方图全部匹配；
- 未自动重跑或覆盖 30 万行正式数据。2026-07-24 的正式 Parquet 仍保留，
  但其旧权重在当前 train-only ESS 门禁下不再视为新合同合格产物。
