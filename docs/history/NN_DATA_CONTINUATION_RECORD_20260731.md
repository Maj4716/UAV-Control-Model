# 神经网络与数据合同续改记录（2026-07-31）

## 1. 当前结论

本轮修改没有把现有模型标记为通过。旧数据集
`output/damage_dataset.parquet` 的 schema、谱系隔离、SHA-256、ESS 和 K2
安全上限均有效，但它不满足最终性能目标所需的分弹型证据合同：

- Med-LM/C2：val 3 行/3 root，test 4 行/4 root；
- Med-RD/C2：val 26 行/26 root，test 27 行/27 root；
- Heavy/C2：val 48 行/48 root，test 61 行/61 root。

正式门槛是每个适用“弹型×任务×精确等级”：

- train 至少 128 行、16 个独立 root；
- val 至少 100 行、16 个独立 root；
- test 至少 100 行、16 个独立 root。

因此旧数据现在被审计为 `CURRENT_V2_EVIDENCE_GAP`，不能用来证明所有
分弹型各级准确率达到 94%。

旧 A0/seed42 单种子模型也未通过：

- test K/C 三分类准确率分别为 91.22% 和 87.98%；
- 11 个具有足够支持量的 L1 单元低于旧 85% recall 门槛；
- Small/K L1 recall 仅 28.37%；
- Small/K0 FP=0.466% 和全局 C0 FP=2.196%，两项安全约束通过。

## 2. 数据生成合同修改

修改文件：`generate_dataset.py`

1. 新增生产级、独立的评估证据启用阈值
   `EVALUATION_SUPPORT_GATE_MIN_ROWS=50000`。缩小版单元测试不会误触发
   100 行门槛，正式 30 万行生产必定启用。
2. 对所有适用精确等级增加 train 行数门禁
   `MIN_TRAIN_EXACT_LEVEL_ROWS=128`，不再只检查 root 数。
3. 写入 `training_exact_level_support` 和
   `evaluation_exact_level_support` profile。
4. 在最终裁剪后按整个 `root_seed_id` 家族执行确定性 split 重分配：
   - 不拆分 root；
   - 不把 train 的任一适用精确等级降到 128 行/16 root 以下；
   - val/test 分别补到 100 行/16 root；
   - 无法补足时保持严格失败，不放宽门槛。
5. C2 fresh-root 预算和目标保持扩大后的 32768 候选、32 轮和 128 个严格
   正例 root，使 Med-LM/Med-RD/Heavy C2 有足够来源供三个 split 使用。

## 3. MC 标签方差修改

修改文件：`generate_dataset.py`、`relabel_stage0_high_mc.py`、
`validate_stage0_dataset.py`

1. 自适应 MC 保持 8–64 次、标准误目标 0.02。
2. 正式启用成对反向扰动 `LABEL_MC_ANTITHETIC=true`：
   - 同一 pair 使用相同 RNG seed；
   - spread sign 依次为 `+1/-1`；
   - 自适应提前停止只在完整 pair 后检查。
3. 采用该设置的依据是 validation-only r32/r64 对照：
   - 硬等级一致率：98.34% → 98.54%；
   - 概率 MAE：0.00427 → 0.00372。
4. relabel manifest 现在绑定 antithetic 配置；旧的独立随机 shard 不允许与
   新 shard 混合恢复。
5. 64 次 MC 数据的验证合同要求 profile 明确记录
   `antithetic_pairs=true`。

## 4. 验证器和审计器修改

修改文件：`validate_stage0_dataset.py`、`audit_stage0_dataset.py`

- 验证器不再只相信 `usability_gate.passed`，而是从 Parquet 重新计算
  train/val/test 的行数和独立 root 数；
- 验证器核对 profile 中每个 val/test 单元的统计与 Parquet 完全一致；
- 验证器独立复核结构零和 `*_level` 与序数概率阈值的一致性；
- 审计器新增 `exact_level_evidence`，只对适用等级计入合同缺口；
- 当前旧数据的 12 个原始 `<100` 格子中，6 个是结构零或非目标统计，
  真正的适用证据缺口是上述 6 个 C2 val/test 单元。

## 5. 单种子 test 封闭修改

修改文件：`abli_exp/run_ablations.py`、`nn_eval_export.py`

- 默认 ablation 运行在训练和 validation 阶段结束，test 保持封闭；
- 只有显式使用 `--allow-test-evaluation` 才会尝试 test；
- 运行器和评估脚本都会验证 promotion report：
  - 位于当前 run directory 内；
  - `status=PASS`；
  - `test_metrics_read=false`；
  - 若报告包含 candidate，必须与 experiment ID 一致；
- promotion 文件路径和 SHA-256 写入 `evaluation_status.json`；
- `run_status.json` 新增：
  - `performance_gate_passed`；
  - `test_metrics_read`；
  - `promotion_report`；
  - `evaluation_status`。

`status=COMPLETE` 以后仍表示 test/导出流程完整执行，是否通过性能验收必须
读取 `performance_gate_passed`，避免再次混淆“执行完成”和“性能通过”。

## 6. 为什么不能只继续调阈值

validation-only 可分性诊断表明：

- 当前神经网络的 94%/90% 阈值组合只在 16 个弹型×任务单元中的 6 个可行；
- 使用同样可部署输入的 ExtraTrees 上限与神经网络接近；
- 使用真实 component MC 目标传播毁伤树时，16/16 单元均存在满足目标的
  阈值组合；
- 早期分析式/单机理 proxy 最低仅 2/16 可行；A35 网络预测 component
  毁伤树为 6/16，仍显著低于真实 component MC 的 16/16。

这说明主要瓶颈是终端状态到关键部件毁伤概率的可观测映射，而非网络宽度或
阈值搜索。高 MC、C2 证据补齐和更低方差标签是下一轮训练的必要前提，但
仍需继续提升可部署部件物理特征/部件概率分支，才能证明最终 94%/90% 目标。

## 7. 重新生产与验收

不要覆盖旧数据进行半成品续写。使用当前生成器完整重建后依次执行：

```powershell
python generate_dataset.py
python validate_stage0_dataset.py output/damage_dataset.parquet
python audit_stage0_dataset.py output/damage_dataset.parquet --output output/stage0_dataset_audit_v2.json
```

新数据必须同时满足：

- 300000 行、四弹型各 75000；
- `usability_gate.passed=true`；
- `exact_level_evidence.passed=true`；
- `contract_status=CURRENT_V2`；
- train MC 最小 8、最大 64，`antithetic_pairs=true`；
- 无跨 split root；
- Parquet、profile、component sidecar、logit adjustment 的 SHA-256 全部匹配。

在这些条件满足之前，不应再次启动正式单种子训练；否则 C2 指标仍无充分
证据，且训练结果不能完成最终目标验收。

## 8. 破片瓶颈的反事实定位（2026-08-06）

在不读取 test 标签的 validation-only 诊断中，对 A29 的破片、冲击机理概率
分别进行“预测值/真值”互换，得到以下 94% 准确率、90% 对角召回率可行单元
数量：

- 真值破片 + 预测冲击，通过 15/16 个“弹型×任务”单元；
- 预测破片 + 真值冲击，仅通过 5/16 个单元；
- 真实 component MC 毁伤树，通过 16/16 个单元；
- 网络预测 component 毁伤树，仅通过 6/16 个单元。

这些反事实结果将主要误差定位到“终端状态 → 破片/关键部件毁伤概率”映射，
而不是冲击波分支、序数阈值或最终 OR 组合。因而本轮没有继续单纯扩大网络或
反复搜索阈值，而是对破片可观测特征、机理编码器和边界监督作定向修改。

对应诊断文件：

- `abli_exp/results/A29_target_fragment_predicted_shock_or_feasibility.json`；
- `abli_exp/results/A29_predicted_fragment_target_shock_or_feasibility.json`；
- `abli_exp/results/A35_independent_component_tree_fusion/seed42/output/validation/threshold_feasibility_target_component_tree.json`；
- `abli_exp/results/A35_independent_component_tree_fusion/seed42/output/validation/threshold_feasibility_predicted_component_tree.json`。

## 9. A40 神经网络定向修改

修改文件：`nn_feature_engineering.py`、`nn_dataset.py`、`nn_model.py`、
`nn_train.py`、`abli_exp/ablation_config.py`、
`abli_exp/configs/A40_independent_mechanism_component_proxies.json`。

1. 增加可部署的 armor-aware component fragment proxy：
   - 破片到达速度及入射几何；
   - 部件遮蔽/装甲厚度与 Thor 侵彻裕度；
   - 暴露脆弱面积比例；
   - 多破片期望命中后的存活/毁伤解析量。
2. 所有新增特征仅由 13 维终端状态、弹型常数和车辆公开几何计算，不读取
   命中、侵彻结果、组件毁伤标签、root/split 或主动采样元数据。
3. `mechanism_encoder_mode=independent` 时，破片与冲击使用独立编码器和
   独立 skip path，避免冲击分支的易学信号覆盖破片表示；旧配置仍默认
   `shared`，保持向后兼容。
4. A40 固定使用破片/冲击概率 OR，同时设置：
   - `mechanism_branch_weights=[3.0, 1.0]`；
   - 破片边界聚焦权重 2.0、带宽 0.15；
   - hard mechanism classification 权重 0.5；
   - 不重复使用生成器行权重或额外正类权重。
5. 保留 soft MC 标签监督，并把模型容量和额外损失投入已由反事实诊断确认的
   破片 0.5 毁伤边界。

以上是“增加可观测信息并使优化目标对准瓶颈”的修改，尚不能代替正式 A40
验证和封存 test 的实测结论。

## 10. 严格目标选模与 test 封存修复

修改文件：`nn_train.py`、`abli_exp/promote_strict_validation_goal.py`、
`abli_exp/validate_multiseed_ensemble.py`、`run_post_generation_pipeline.py`。

1. 旧 `selection_score` 虽计算最差单元准确率，但没有把 94% 准确率缺口纳入
   惩罚；历史告警还沿用 85% L1 recall。现在候选评价显式计算：
   - 每个有证据单元的三分类准确率下限 94%；
   - 每个适用精确等级的对角 recall 下限 90%；
   - L1 recall 下限 90%；
   - Small/K0 与全局 C0 安全约束。
2. epoch、top-k checkpoint、model soup 和最终候选统一使用目标感知字典序：
   先比较完整严格门禁是否通过，再比较最差目标裕量、总缺口和原选择分数。
   这避免平均准确率较高但个别弹型/等级严重失效的 checkpoint 被选中。
   后续审计又发现 raw-best 保存与 early stopping 仍只比较标量
   `selection_score`，现也已统一到相同字典序；Small/K0 与全局 C0 安全上限
   作为硬可行门槛进入该键。安全范围内不会为了继续压低误报而牺牲
   94%/90% 性能裕量。排序定向 unittest 2/2 通过。
3. validation promotion 必须同时满足：
   - `stage0_nn_validation_selection_v2`；
   - `split=validation` 且 `test_labels_used=false`；
   - 历史/安全门禁通过；
   - 完整 94%/90% 目标门禁通过；
   - 数据、模型、阈值三个 SHA-256 身份有效；
   - test 产物尚不存在。
4. promotion 失败时 test 保持封存；不会为了获得一次 test 数字而绕过
   validation 失败。
5. `validate_strict_performance_goal.py` 的最小值比较已由错误的 `<=` 失败
   修正为 `<` 失败，并把合同元数据改为
   `greater_than_or_equal_to_for_minima`。因此恰好 94.0% 准确率或 90.0%
   recall 按“不得低于”要求通过；94%/90% 以下仍失败。新增回归测试构造
   Med-LM/K 恰好 94.0% 准确率、90.0% L1 recall，3 个相关 unittest 均通过。
6. `promote_strict_validation_goal.py` 不再只检查 SHA-256 字符串长度。test
   解封前会重新读取 run/model manifest，核对 candidate experiment ID，重新计算
   `best_model.pth`、阈值、两个 scaler 和当前 Parquet 的实际大小与 SHA-256，
   并要求 validation report 的数据/模型/阈值身份与实物完全一致。新增篡改
   `best_model.pth` 的负向测试，promotion/身份绑定相关 unittest 3/3 通过。
7. 修复 test 解封命令链的两个必现/潜在错误：
   - `run_ablations.py` 现在把已选中的 `--promotion-report` 显式传给要求该
     参数的 `nn_eval_export.py`；旧代码会在 validation PASS 后因缺少必需参数
     立即退出；
   - promotion 必须带有与当前 experiment ID 完全相同的 candidate，匿名或
     其他实验的 PASS 不再被选择或授权。
   同时修复 recalibration 分支在赋值前使用 `promotion_path` 的顺序问题。
   解封命令、candidate 绑定和篡改拒绝相关 unittest 4/4 通过。
8. 阈值校准也改为与最终目标一致。旧联合搜索只硬约束精确 L1 recall，再按
   `0.8*accuracy + 0.2*head-F1` 选择，可能主动选中 L2 recall 不足 90% 的
   组合；同时当前 `train_model()` 曾遗漏从 calibration 配置赋值两个变量，
   A40 会在首轮校准时 `NameError`，且训练期/最终重评会使用不同默认门槛。
   现已：
   - 显式解析并校验 `minimum_exact_class1_recall` 与 accuracy-drop；
   - A40 启用 `goal_aware_cell_search`；
   - 网格中若存在同时满足单元 accuracy≥94%、L0/L1/L2 recall≥90% 及安全
     上限的组合，优先按最差裕量选择；不存在时才回退原安全/L1/F1策略，
     不伪造可行性；
   - epoch 校准与最终 checkpoint 重评使用完全相同的参数，参数写入阈值合同。
   构造反例中旧策略选择 L2 recall=85% 的阈值，新策略选择 accuracy=98.33%、
   三个等级 recall=[100%,90%,90%]；新旧兼容测试 4/4 通过。

## 11. Med-LM/C2 物理可达性与生产采样修复

Med-LM/C2 曾因旧结构零假设和稀疏样本阻断生产。为区分真实可达状态与 MC
偶然值，新增 `revalidate_stage0_roots.py`，对每个 root 只取一行并以固定 64
次成对反向 MC 复核：

- 选取 32 个独立 root；
- 31/32 在 0.5 阈值上仍为正例；
- 29/32 的 95% 正态区间下界仍高于 0.5；
- 重算均值 0.7750，最高 0.9745。

因此 Med-LM/C2 是当前毁伤物理中的真实可达状态，不应删除或声明结构零。
训练模型的 `DEFAULT_ORDINAL_APPLICABILITY` 与严格 test 验收器已同步包含
Med-LM/C2；新增跨模块回归测试锁定该合同，同时确认 Small/K2、Small/C2
仍为结构零，避免生成/训练/验收矩阵以后再次漂移。
生产 proposal 新增 crew-corridor 定向 fresh root 几何，但不改变毁伤物理、
标签阈值或验收门槛。正式生产前探测在 8192 候选预算内实际仅请求 896 个
候选、正式确认 185 行，即获得 16 个严格独立正例 root：

- 有效 root=16；
- 最大单 root 占比=6.25%；
- 最高正式 C2 概率=0.8433；
- `candidate_budget_exhausted=false`。

对应证据：

- `output/med_lm_c2_root_revalidation.json`；
- `output/stage0_reachability_probe_med_lm_c2_final.json`。

`sim_engine.py` 还增加了精确 shockwave probability cache；同一行的后续 MC
重复使用确定性的冲击波概率，只重算随机破片过程。该缓存不近似冲击波物理，
用于降低 8–64 次标签重算成本。

## 12. 当前正式运行与未完成验收

截至 2026-08-06，本轮 300000 行完整重建已启动。为了防止运行时代码与产物
漂移，在生成进程结束前冻结 `generate_dataset.py` 和 `sim_engine.py`。

`run_post_generation_pipeline.py` 已启动守门流水线，生成结束后自动执行：

1. `validate_stage0_dataset.py`；
2. `audit_stage0_dataset.py` 并要求 `contract_status=CURRENT_V2`；
3. 训练 A40/seed42（只使用 train/validation）；
4. 严格 validation promotion；
5. 只有 promotion PASS 才读取一次封存 test；
6. `validate_strict_performance_goal.py` 复核 test 的完整分弹型 94%/90% 目标。

正式生成等待期间已做只读启动预检：A40 配置名可解析到实际 JSON，
`experiment_id` 和预期 `seed42` 输出目录完全一致；活动特征共 296 个且名称
唯一，与 `base_input_dim=296` 的显式 direct-path 合同一致。A40 配置、独立机理
编码器、目标感知 checkpoint 排序、validation promotion 四项定向 unittest
均通过，当前 A40 输出目录不存在，不会触发防覆盖拒绝。

完成上述选模、校准、promotion 与解封链修复后，使用正式 PyTorch 环境执行
完整 `tests.test_nn_pipeline`：96 项全部通过（65.608 秒）。这证明现有 NN
合同与历史兼容性未被新目标策略破坏；它仍不代替新数据上的 A40 实测指标。

当前仍是未完成状态。以下证据缺一不可，不能根据代码实现、旧数据、平均准确率
或 validation-only 反事实上限提前宣称达标：

- 新 Parquet/profile/sidecar/logit 的合同和 SHA-256 全部通过；
- 所有适用 val/test 精确等级均达到 100 行、16 root；
- A40 validation 的所有适用分弹型单元准确率均不低于 94%；
- A40 validation 的所有适用 L1 及其他精确等级 recall 均不低于 90%；
- 安全约束通过；
- promotion 后的封存 test 独立达到同一 94%/90% 目标。

若 A40 validation 未通过，流水线将停在 `VALIDATION_GOAL_NOT_MET` 且不读取
test；若 test 未通过，将保留真实失败结果并继续针对失败单元优化，而不会降低
门槛或把单种子偶然结果包装为完成。

## 13. 2026-08-16 中断诊断与 Phase-1 可恢复检查点

2026-08-06 的正式生成未正常完成：生成器日志停在 Phase-1 的
`105000/150000`，标准错误日志为空，日志尾部位于完整批次边界；生成器与守门
流水线进程随后均已不存在。正式 `damage_dataset.parquet` 仍是 2026-07-25 的
旧产物，A40 输出目录也尚未创建。综合这些证据，本次中断更符合外部会话或系统
终止，而不是 Python 异常。旧实现只在全部生成完成后原子写出 Parquet，因此
这 105000 行内存结果没有可验证的落盘分片，不能安全复用。

为避免长时间生产再次因外部中断全部返工，`generate_dataset.py` 新增了
`stage0_phase1_checkpoint_v1`：

- 检查点身份绑定 Phase-1 输入内容、完整生成配置，以及生成器、仿真引擎、
  坐标系、部件监督、车辆模型和装甲配置的 SHA-256；任何输入或实现变化都会
  自动进入新的身份目录，不会误用旧结果；
- 每 1000 个输入任务写出一个带原始任务序号的原子 Parquet 分片；恢复时严格
  检查序号范围、分片内重复和跨分片重叠；
- 恢复后的结果仍按原始输入序号重排，保证并行完成顺序不改变后续数据语义；
- 普通异常退出前会刷新已完成缓冲区，外部强制终止最多损失当前未刷新的小批；
- 该机制只缓存确定的仿真调用结果，不修改物理、采样分布、随机种子、标签、
  权重、配额、裁剪或数据 schema。

新增两项回归测试分别覆盖检查点身份/原子分片往返，以及“已有 0、2 号结果时
只补算 1、3 号并恢复原顺序”的行为。恢复前验证结果如下：

- `py_compile`：通过；
- Phase-1 检查点定向测试：2/2 通过；
- 完整 `tests.test_stage0`：通过；
- 完整 `tests.test_nn_pipeline`：96/96 通过。

由于旧运行没有这些分片，本轮 Phase-1 必须从零重新开始；此后的同一身份运行
可以从已验证检查点续算。最终 94% 分弹型准确率与 90% 各适用等级召回率仍须由
新数据上的 A40 validation 和一次性封存 test 实测证明，当前不提前宣称完成。
