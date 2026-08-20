# 巡飞弹毁伤评估项目对话迁移与工作交接

> 更新时间：2026-08-18（Asia/Shanghai）
> 项目目录：`<PROJECT_ROOT>`
> 用途：将当前超长对话迁移到新对话。新对话开始后，应先完整阅读本文与根目录 `AGENT.md`，再检查实时文件状态；不要仅依据聊天摘要直接启动长任务。

## 1. 项目目标

本项目当前包含两条相互衔接的主线：

1. 构建轻量化巡飞弹毁伤评估仿真引擎，通过主动采样生成具有物理含义、谱系可追溯、划分无泄漏的数据集。
2. 使用该数据集训练快速毁伤评估代理神经网络，并以严格、封存的验证/测试流程检验模型。

后续长期目标是在现有毁伤模型基础上继续建模末制导与飞控，构建数字孪生环境，并使用强化学习实现巡飞弹智能控制。当前尚未进入该阶段；眼下工作的完成条件仍是：新数据合同通过，并让 A40 候选模型依次通过严格 validation promotion 和一次性封存 test。

## 2. 当前最重要结论

### 2.1 数据集已经成功生成，禁止无故重生成

最新 30 万行数据集已经于 2026-08-18 17:02 左右完成生成。生成器正常退出，旧生成器 PID `148604` 与守门流水线 PID `110484` 均已结束。

权威产物：

- `output\damage_dataset.parquet`
  - 300000 行，132 列
  - 文件大小：235971820 bytes
  - SHA-256：`a62684a42aa9c950877becebcbdf1fefabfb529bad3142710629cbb6b87a9d12`
- `output\generation_profile.json`
- `output\logit_adjustment.json`
- `output\component_supervision.parquet`
  - 300000 行，103 列
  - SHA-256：`f1df9e3e0df24a8181f0c8df340e9a355a708f4d40b5040d5d3c7297fe138481`
  - 样本顺序 SHA-256：`956ed9dd37657b2d31bed5e066d45e770349f6fd6989efba5cb41a739b75141c`
  - 只可作为训练辅助监督，严禁作为部署模型输入。

四种弹型均为 75000 行：Small、Med-LM、Med-RD、Heavy。

### 2.2 数据合同本身已通过

`output\post_generation_logs\validate_dataset.stdout.log` 的最终结果为 `PASS`：

- `dataset_schema=stage0_lineage_v2`
- `frame_convention=stage0_ned_frd_v1`
- 300000 行、132 列、294402 个 root family
- train/val/test：240786 / 29473 / 29741
- 跨划分 root family：0
- 序数单调性违规：0
- 最大每 root 行数：10
- Parquet SHA-256 校验通过
- usability gate 通过
- logit adjustment 与 Parquet SHA-256 一致
- component supervision 与主表身份绑定一致
- train 权重 ESS 比例：`0.5050000000003421`，高于 0.50 下限
- K2 正例 21327 行，占 7.109%，低于 8% 上限
- exact-level evidence：通过；所有适用单元满足 train 至少 128 行/16 root，val/test 至少 100 行/16 root

`output\post_generation_logs\audit_dataset.stdout.log` 的最终结果为：

- `status=AUDIT_COMPLETE`
- `contract_status=CURRENT_V2`
- schema 匹配、SHA 匹配、行数匹配、划分计数匹配、MC 重复次数匹配
- exact-level evidence `PASS`，`contract_ready=true`
- component supervision `CURRENT_V1`

审计中的 `cells_below_100=6` 不是数据缺口。它们全部对应 Small 弹型的 K2、C2 在 train/val/test 中的结构零单元；当前适用性合同明确规定这些单元不适用，所以不应补样，也不应据此判定失败。

### 2.3 `post_generation_pipeline_state.json` 的 FAILED 是集成代码误判

`output\post_generation_pipeline_state.json` 当前为：

- 生成器退出码：0
- validation 子进程退出码：0
- audit 子进程退出码：0
- 失败阶段：`audit_dataset`
- 失败原因：`contract_status=None`

这不是数据失败。根因是 `run_post_generation_pipeline.py` 读取完整审计文件后，错误地要求顶层存在 `contract_status`：

```python
audit_payload = _load_json(audit_path)
if audit_payload.get("contract_status") != "CURRENT_V2":
    ...
```

但 `audit_stage0_dataset.py` 当前把完整审计主体写入文件，并把 `contract_status` 只放在控制台摘要中；完整文件的关键字段位于 `statistics` 内。因此流水线把已经通过的 `CURRENT_V2` 审计误判为 `None`。

这是下一个对话应首先修复的唯一已知代码阻塞点。建议：

1. 给流水线增加兼容性解析函数：优先读取顶层 `contract_status`；旧格式则结合 `statistics.artifact_identity.current_schema_match` 与 `statistics.exact_level_evidence.contract_ready` 推导合同状态。
2. 同时可让 `audit_stage0_dataset.py` 在输出文件顶层写入 `status` 和 `contract_status`，保留现有 `statistics` 内容，形成显式稳定接口。
3. 添加回归测试，必须覆盖当前“完整审计 JSON 无顶层 contract_status，但 statistics 表明 CURRENT_V2”的实际负载。
4. 不得通过跳过审计、放宽门禁或硬编码成功来修复。

### 2.4 A40 尚未开始训练

目录 `abli_exp\results\A40_independent_mechanism_component_proxies\seed42` 当前不存在。这意味着：

- A40 没有在最新数据集上完成训练；
- validation promotion 尚未执行；
- 封存 test 尚未打开；
- 不能声称 94%/90% 指标已达到。

自动化任务 `a40` 当前为 `PAUSED`。必须保持暂停，除非用户在新对话中明确要求恢复或启动训练。

## 3. 数据集部分的主要演进

### 3.1 Stage-0 物理与数据合同

已经完成的核心修复包括：

- 统一并显式记录 NED/FRD 坐标约定，合同名为 `stage0_ned_frd_v1`。
- 修复侵彻概率的组合逻辑：只有真实侵彻余量 `margin >= 1` 才能组成有效侵彻结果。
- 引入 `stage0_lineage_v2`：包含 sample/root/split 谱系、root family 隔离、确定性划分及产物身份绑定。
- validation/audit 均验证行数、列数、SHA-256、划分计数、MC 重复次数与序数标签单调性。
- 禁止同一 root family 跨 train/val/test，避免由爬行或衍生样本造成泄漏。

### 3.2 主动采样、MC 与权重

- 标签 MC 采用 8–64 次自适应重复及 antithetic pairs。
- 最新数据中：全部头都提前收敛的样本约 61.07%，达到最大重复数的样本约 39.13%。K_ge1 与 C_ge1 是相对最难稳定的头，后续训练分析应关注标签不确定性，但不能把不确定性辅助量当部署输入。
- 生成器侧 class-balanced 权重已经禁用，只在训练侧保留一套类别平衡机制，避免重复校正。
- 最终权重范围 `[0.05, 20.0]`，tempering alpha 约 `0.960831059`。
- train ESS 为 50.5%，合同下限仍为 50%，不得下调以换取通过。

### 3.3 可用性与精确等级证据门禁

对每个适用的“弹型 × 任务 × 精确等级”单元执行严格证据约束：

- train 正例至少 128 行、16 个独立 root；
- val/test 各至少 100 行、16 个独立 root；
- 有效 root 数和最大 root 占比受限；
- 单 root 最多保留 8 个正例；
- 最终裁剪优先保留 Phase-2 train 难例及证据样本；
- 不能删除真实难例来改善指标。

最新数据的所有适用单元已通过该门禁。

### 3.4 Med-LM/C2 从“结构零”改为“已证实可达”

早期一次拒绝 profile 发现 Med-LM/C2 有 29 个正样本，且来自 29 个独立 root：

- 正例行数：29
- 独立 root：29
- 有效 root：29
- 最大 root 占比：3.45%

后续严格复验中，32 个 root 里有 31 个概率仍大于 0.5，29 个 root 的置信区间下界大于 0.5，均值约 0.775，最大值约 0.9745。因此旧“结构零”假设被物理引擎中的独立证据推翻。

已实施：

- `ORDINAL_APPLICABILITY[1]["C"]` 改为 `[True, True]`；
- Med-LM Phase-2 配额包含 `C2_prob=0.20`；
- `stage0_reachability_probe.py` 新增 `med_lm_c2` 目标；
- 使用独立种子、fresh-root 与 `C_HUNT` 路径补充 C2；
- 保持全部严格门禁，不删除 29 个真实正例。

最新生成中 Med-LM/C2 的总量补样过程从 264 行/152 root，经多轮补样达到 384 行/164 root；Med-RD/C2 也达到 395 行/188 root。最终不存在未满足的 L2 供给缺口。

### 3.5 生成失败与修复经验

发生过的关键失败：

1. Med-LM/C2 被配置为结构零，却观察到 29 个独立正例。正确处理是修正适用性，而不是删除数据。
2. 改为适用后，C2 在 val/test 的精确等级证据不足。加入了跨三划分的总 L2 供给预门禁（目标至少 344 行/48 root）。
3. 第一版补样仍停在 Med-LM/C2 337 行/164 root，原因是单轮请求 256 太小且重试限制过紧。随后把最小 top-off 请求提高到 1024、最大轮数提高到 32，并允许 3 轮无增长再失败。
4. 重平衡器曾因某个不可满足单元使用 `break` 并过度锁定，导致后续可满足单元得不到处理。已改为单元级跳过，不能让局部不可达阻断整个证据修复。
5. Python 3.9 不支持运行时求值的 `str | None`，曾在类定义阶段报 `TypeError`。已改为 `Optional[str]` 并导入 `Optional`。后续代码继续保持 Python 3.9 兼容，不使用无 future annotations 保护的 PEP 604 类型写法。

### 3.6 Phase-1 检查点与恢复

`output\stage0_phase1_checkpoint_v1` 下存在身份绑定的 Phase-1 检查点：150 个 part、150000 行，已经完整。兼容性文件为 `phase1_checkpoint_compatibility.json`。

旧生成器 SHA：`0c53...45e0`（文档中的缩写）
当前生成器 SHA：`c1e35831859d0e43aa3b973c6d653dd118e59dbec38820b4f16aab19bfc9bc82`

兼容性判断依据是：变化仅发生在 Phase-1 之后的供给补样、重平衡或诊断流程；物理、输入与 Phase-1 配置保持一致。虽然检查点对未来异常恢复有价值，但当前数据已完整生成，不应再恢复或重跑。

## 4. 神经网络部分的既有结论

### 4.1 旧模型未达到最终目标

历史 A0/seed42 测试结果中：

- K/C 三分类 accuracy 约 91.22% / 87.98%；
- 11 个有支持的 L1 单元低于旧 85% 门槛；
- Small/K L1 recall 约 28.37%；
- Small/K0 假阳性约 0.466%，全局 C0 假阳性约 2.196%，安全约束通过。

A13 baseline 的平均 accuracy 约 93.13%，仍有 8 个严格失败单元。多种子 A13/A19 验证表明，弱点并非 seed42 的偶然波动：Small/K 约 26.64%–28%，Med-RD/M 约 53.2%–53.69%，Med-RD/C 约 55.99%。

### 4.2 阈值调整不是主要解法

在现有可部署分数上，94% accuracy / 90% exact recall 同时可行的只有约 6/16 个任务单元；ExtraTrees 的上限也相近。

- Small/K：在 K0 FP ≤0.5% 约束下，damage-entry recall 上限约 26.64%–28.7%，与 L1 recall 几乎重合，说明不是简单阈值偏移。
- Med-RD/M：damage-entry recall 已约 96.2%–96.7%，但 exact L1 只有约 53%，错误主要是 L1 被判为 L2，属于序数等级分离问题。

因此不得通过在 test 上调阈值、降低验收线或掩盖难例来宣称达标。

### 4.3 机制诊断与 A40 方向

反事实机制诊断结果：

- 真实 fragment + 预测 shock：15/16 单元可行；
- 预测 fragment + 真实 shock：5/16；
- 真实 component MC + tree：16/16；
- 预测 component + tree：6/16。

主瓶颈因此位于“末端状态 → 破片/关键部件毁伤”映射，而非冲击波或单纯阈值。

A40 配置：`abli_exp\configs\A40_independent_mechanism_component_proxies.json`

关键设计：

- `base_input_dim=296`；
- 从 13 维末端状态、弹型常量和公开车辆几何构造可部署、装甲感知的 component/fragment proxy；
- fragment 与 shock 使用独立 encoder/skip；
- branch 权重 `[3, 1]`；
- fragment boundary focus=2、band=0.15；
- hard mechanism classification 权重 0.5；
- 使用 soft labels；不使用 label uncertainty 输入；
- 当前 `use_component_supervision=false`；
- 不使用 generator row weights，也不叠加额外 positive weights；
- 目标感知校准：每适用单元 accuracy ≥94%，各 exact class recall ≥90%；
- epochs=60，最早 selection epoch=40，patience=12。

### 4.4 模型选择和封存测试规则

已修复并应继续遵守：

- checkpoint selection 按严格目标/安全是否满足、最差 margin、总 gap，再到原 selection score 进行字典序选择；
- 阈值搜索要求 L0/L1/L2 的所有适用对角 recall 均 ≥90%，不能只检查 L1；
- 严格比较已经从错误的 `<=` 修为 `<`，恰好 94%/90% 应通过；
- validation promotion 绑定模型、阈值、scaler 与数据集的 SHA-256、大小和 candidate ID；
- validation 未通过时不得读取 test 标签；
- promotion 通过后，test 只能揭封一次，并独立验证相同目标。

## 5. 严格最终验收条件

只有以下条件全部真实满足，任务才可标记完成：

1. 数据：`CURRENT_V2`，所有身份 SHA 匹配，无跨 split root 泄漏，适用单元证据门禁通过。
2. Validation：所有适用“弹型 × 任务”三分类 accuracy ≥94%。
3. Validation：所有适用 exact L0/L1/L2 recall 均 ≥90%，尤其不得忽略 L1。
4. 安全：Small/K0 FP ≤0.5%，全局 C0 FP ≤2.5%。
5. validation promotion 返回 PASS，且身份清单完整。
6. 在 promotion 之后进行一次性封存 test；test 独立满足同样的 94%/90% 与安全约束。

明确禁止：

- 降低 94%/90% 或安全阈值；
- 删除、屏蔽或重标真实难例来改善成绩；
- 将已证实可达单元重新标为结构零；
- 使用 test 选择模型、阈值或超参数；
- validation 未通过就执行 test；
- 把 component supervision、标签不确定性、未来信息或其他训练期辅助量作为部署输入；
- 只凭总体平均指标宣布通过。

## 6. 下一对话的建议执行顺序

### 步骤 A：先修复流水线的审计 JSON 契约

检查：

- `run_post_generation_pipeline.py`
- `audit_stage0_dataset.py`
- `tests\test_post_generation_pipeline.py`

实现稳定解析并添加真实负载回归测试。不要重跑生成器。

建议验证命令（PowerShell）：

```powershell
python -m py_compile run_post_generation_pipeline.py audit_stage0_dataset.py
python -m pytest -q tests/test_post_generation_pipeline.py
python -m pytest -q
python validate_stage0_dataset.py output/damage_dataset.parquet
python audit_stage0_dataset.py output/damage_dataset.parquet --output output/stage0_dataset_audit_post_generation.json
```

注意：最近一次已记录的完整测试结果是 141 项通过，但这是本次文档整理之前的历史记录，不能替代修复后的重新测试。

### 步骤 B：修复后进行 validation-only A40

只有用户明确授权启动训练后才执行：

```powershell
python abli_exp/run_ablations.py --configs A40_independent_mechanism_component_proxies --seeds 42 --train-only --fail-fast
python abli_exp/promote_strict_validation_goal.py --run-dir '<PROJECT_ROOT>\abli_exp\results\A40_independent_mechanism_component_proxies\seed42' --candidate A40_independent_mechanism_component_proxies
```

训练开始后，每 20 分钟最多检查一次；若日志没有有意义的里程碑，不要向用户重复输出相同状态。

### 步骤 C：仅在 promotion PASS 后揭封 test

```powershell
python abli_exp/run_ablations.py --configs A40_independent_mechanism_component_proxies --seeds 42 --eval-only --allow-test-evaluation --fail-fast
python abli_exp/validate_strict_performance_goal.py --run-dir '<PROJECT_ROOT>\abli_exp\results\A40_independent_mechanism_component_proxies\seed42'
```

若 validation promotion 失败，应停在 validation 阶段，分析分单元 confusion、exact recall、机制误差和安全 margin；不能读取 test 来指导修改。

## 7. 重要文件索引

### 数据与仿真

- `generate_dataset.py`：主动采样、补样、权重、最终裁剪、profile 写出。
- `sim_engine.py`：轻量化毁伤物理。
- `validate_stage0_dataset.py`：Stage-0 严格合同验证。
- `audit_stage0_dataset.py`：数据审计与统计报告。
- `run_post_generation_pipeline.py`：生成后验证/审计/训练守门流水线；当前存在 audit JSON 解析 bug。
- `stage0_reachability_probe.py`：生产前可达性探测，包含 `med_lm_c2`。
- `revalidate_stage0_roots.py`：对候选 root 做高精度复验。
- `tests\test_stage0.py`
- `tests\test_post_generation_pipeline.py`

### 神经网络

- `nn_feature_engineering.py`
- `nn_dataset.py`
- `nn_model.py`
- `nn_train.py`
- `nn_eval_export.py`
- `tests\test_nn_pipeline.py`
- `abli_exp\configs\A40_independent_mechanism_component_proxies.json`
- `abli_exp\run_ablations.py`
- `abli_exp\promote_strict_validation_goal.py`
- `abli_exp\validate_strict_performance_goal.py`

### 现有记录

- `NN_DATA_CONTINUATION_RECORD_20260731.md`
- `NN_SINGLE_SEED_FOLLOWUP_MODIFICATION_RECORD.md`
- `DATASET_GENERATOR_REPAIR_RECORD.md`
- 本文：`CONVERSATION_HANDOFF_20260818.md`

### 当前证据产物

- `output\generation_profile.json`
- `output\post_generation_pipeline_state.json`
- `output\post_generation_logs\generator.stdout.log`
- `output\post_generation_logs\generator.stderr.log`
- `output\post_generation_logs\validate_dataset.stdout.log`
- `output\post_generation_logs\audit_dataset.stdout.log`
- `output\stage0_dataset_audit_post_generation.json`

## 8. 长任务、进程与自动化约束

用户已明确要求避免高频检查，以免快速消耗额度：

- 生成或训练期间最多每 20 分钟检查一次；
- 只记录进度百分比、阶段切换、checkpoint、完成、失败等有意义里程碑；
- 不做分钟级忙轮询；
- 用户说“暂停”后，立即停止主动工作和自动化；
- 当前自动化 `a40` 已暂停，不得擅自恢复；
- 当前没有仍需等待的生成进程。

此外，`run_post_generation_pipeline.py --wait-pid 148604` 不能原样重跑，因为该 PID 已不存在。修复代码后应增加/使用基于现有已绑定产物的 resume/post-generation-only 路径，或手工按“验证 → 审计 → validation-only 训练”执行，不能因此重生成 30 万行数据。

## 9. 高价值经验归纳

1. **合同失败与数据失败必须分离。** 子程序返回 0 且原始摘要明确 `CURRENT_V2` 时，先检查集成层 schema，而不是重新生成数十小时数据。
2. **独立物理证据优先于先验假设。** 29 个独立 Med-LM/C2 root 足以触发结构零假设复核，不能为通过门禁删除证据。
3. **精确等级任务需要按 L0/L1/L2 分别验收。** 总体 accuracy 和 damage-entry recall 会掩盖 L1→L2 错误。
4. **root 独立性比单纯行数重要。** 爬行样本可增加行数，但不能用少数 root 重复堆出虚假的统计支持。
5. **类别平衡只能保留一套主机制。** 生成器重采样、row weight、positive weight、focal loss 叠加会显著扭曲先验与 ESS。
6. **训练辅助监督与部署输入要严格分界。** component sidecar 可以用于辅助 loss，但不得形成推理时不可获得的输入。
7. **封存 test 是一次性裁判，不是调参集。** 所有模型选择、阈值和校准必须在 validation 完成。
8. **长任务应以状态文件和日志为证据。** PID 只是运行态线索，最终应以退出码、产物 mtime/SHA、合同验证和阶段状态共同判断。
9. **保持 Python 3.9 兼容。** 类型注解要使用 `Optional[T]` 或启用可靠的 postponed evaluation，避免模块导入阶段崩溃。
10. **修复必须有实际负载回归测试。** 单元测试不能只构造理想化顶层字段，应覆盖工具实际写出的嵌套 JSON。

## 10. 新对话启动模板

可将下面文字作为新对话首条消息：

> 请先完整阅读 `<PROJECT_ROOT>\AGENT.md` 和 `<PROJECT_ROOT>\CONVERSATION_HANDOFF_20260818.md`，并以当前磁盘文件重新核对关键状态。数据集已经生成且 validate/audit 均真实通过，目前 `post_generation_pipeline_state.json` 的 FAILED 是 `run_post_generation_pipeline.py` 对审计 JSON schema 的误判。请先修复该集成 bug、补回归测试并运行相关测试；不要重生成数据，不要打开 test，也不要恢复 a40 自动化。完成并汇报后，等待我明确授权再启动 A40 validation-only 训练。长任务最多每 20 分钟检查一次。
