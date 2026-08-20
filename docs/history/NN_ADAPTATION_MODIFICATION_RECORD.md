# Stage-0 v2 代理神经网络适配修改记录

日期：2026-07-25

> 后续修订：本文件记录的是首次 Stage-0 v2 适配。2026-07-26 的性能改进已将
> 默认标签置信度加权关闭、增加显式三分类分布损失，并将阈值合同升级为
> `v7_monotone_fpr_constrained`。当前行为以
> `NN_PERFORMANCE_IMPROVEMENT_RECORD.md` 为准。

## 1. 适配对象

- 数据集：`output/damage_dataset.parquet`
- 数据 schema：`stage0_lineage_v2`
- 坐标合同：`stage0_ned_frd_v1`
- 行数：300000
- 数据 SHA-256：`5646b592d6e61da410b14c532929c064297c50a1671b1b4c89cd6716c8fef4da`
- 模型输入：13个终端时刻可观测特征

历史 `output/models`、`output/deploy` 和既有消融结果来自旧数据/旧模型合同，其中旧权重为15维输入，缩放器还包含 `los_distance`、`impact_cosine`。这些产物未被修改或伪装升级；新评估器会明确拒绝它们，必须重新训练。

## 2. 数据与标签

修改文件：`nn_dataset.py`

1. 永久禁止生成阶段瞄准点元数据 `los_distance`、`impact_cosine` 进入代理模型。
2. 消融通过 `drop_features` 真正改变输入维度；A1现在删除 `norm_velocity`，从13维变为12维。
3. 软标签改为完整 MC 均值：
   - 保留硬负例中低于0.5但非零的仿真概率；
   - 不再执行 `where(hard_label==1, probability, 0)` 的单边截断；
   - 强制数值范围 `[0,1]` 和 `P(L>=2)<=P(L>=1)`。
4. 利用 `*_prob_std/sqrt(label_mc_replicates)` 计算标签标准误，并生成 `[0.25,1]` 内的标签可信度。目标概率不被改变，只降低高不确定性标签的损失贡献。
5. K/M/C任务倍率不再由代码硬编码覆盖，默认四弹型全部为1；仅允许实验配置显式改变。
6. 默认关闭额外 `pos_weight`、自适应loss平衡和自适应重采样，避免与数据集 `loss_weight` 重复放大。
7. 默认训练使用普通 `shuffle=True`，每个物理样本每轮最多出现一次。
8. 可选四弹型等量 batch 采样器改为加权无放回排列。旧实现 `replace=True` 每轮仅覆盖约63%的独立训练行；新实现不重复样本，只丢弃不足一个完整 batch 的尾部。
9. DataLoader批次增加：
   - `label_confidence`
   - `sample_id`
   - `root_seed_id`
10. DataLoader返回 `stage0_nn_data_v1` 数据合同，包含数据哈希、特征顺序、适用性矩阵和切分规模。

## 3. 模型

修改文件：`nn_model.py`

1. 四任务输出头改为单调序数参数化：
   - `logit_ge1 = raw1`
   - `logit_ge2 = logit_ge1 - softplus(raw_gap)`
   - 因而模型天然满足 `P(L>=2)<=P(L>=1)`，无需依赖训练惩罚或推理后修补。
2. 将数据集适用性写入持久化模型 buffer。
3. Small/K2与Small/C2结构零在模型图内输出关闭值，并在最终输出再次保证单调。
4. 结构零同时适用于分弹型专家头和共享头消融。
5. K级联概率在送入M/F/C分支前执行停止梯度。M/F/C仍可使用K预测作为条件信息，但不能借由自身损失反向改写K业务输出。

## 4. 损失与模型选择

修改文件：`nn_train.py`

1. BCE直接使用完整软目标，是对概率预测适用的 proper scoring rule。
2. 如启用 `pos_weight`，按软目标连续插值正类权重，不再按硬阈值决定软样本的正负权重。
3. 如启用 focal，`p_t` 使用软目标且保留梯度，不再基于硬标签并 `no_grad`。
4. 标签可信度逐头进入BCE。
5. 可信基线默认关闭：
   - focal
   - 冗余序数惩罚
   - class-1 margin
   - cell class-1 alpha
   - `pos_weight`
   - 重采样
6. 保留数据集 `loss_weight`，用于修正主动采样/接受分布，不再叠加隐式弹型任务倍率。
7. 模型选择删除“相对历史最佳模型下降”的路径依赖惩罚。任一epoch和最终候选均使用相同的固定绝对指标与门槛，候选排序不再依赖出现顺序。
8. 稀疏L2单元不再使用任意固定0.90阈值：
   - L1支持充分时单独校准L1；
   - L2支持不足时继承全局L2阈值；
   - 结构零L2阈值固定为1且由模型同时屏蔽；
   - 所有16个“任务×弹型”单元均记录样本数、正例数、适用性和校准模式。
9. 当时的阈值 schema 更新为 `v6_monotone_cellwise`；当前已由 v7 合同替代。
10. 修正日志中log-var安全范围的显示，使其与K下限-2、M/F/C下限-1、统一上限2.5一致。

## 5. 产物合同、评估与部署

新增/修改文件：`nn_artifacts.py`、`nn_eval_export.py`

1. 训练结束后写出 `output/models/model_manifest.json`，封存：
   - Parquet SHA-256和schema；
   - 特征名称及顺序；
   - 序数适用性；
   - 模型完整结构参数；
   - 随机种子与训练损失配置；
   - 权重、阈值、scaler pkl/json的大小和SHA-256。
2. 评估采用失败关闭策略：
   - 缺模型、阈值、scaler或manifest时直接失败；
   - 任一哈希或数据SHA不一致时直接失败；
   - 不再加载随机初始化模型；
   - 不再把缺失阈值回退为0.5；
   - 不再在评估时重新拟合scaler；
   - 不再兼容旧15维权重或旧阈值schema。
3. `logit_adjustment.json` 必须与当前Parquet SHA一致。只有明确标记联合校准启用时才应用shift；当前生成器的禁用状态会得到尊重。
4. 测试集逐样本输出增加 `sample_id/root_seed_id`、8个序数概率和8个MC均值目标。
5. 每个序数头增加：
   - Brier（硬标签与MC均值口径）
   - NLL/交叉熵
   - 10-bin ECE
   - AUPRC
6. 每任务等级准确率增加按 `root_seed_id` 聚类的500次bootstrap 95%置信区间，避免把同root衍生行错误当成完全独立样本。
7. 可通过 `--challenge-data` 对独立root的C2 challenge做额外评估。评估器校验challenge哈希、用途和源数据SHA，并明确禁止其参与阈值校准或先验估计。
8. ONNX导出后强制执行：
   - ONNX checker
   - ONNX Runtime动态批次1/4/7与PyTorch数值一致性
   - 最大绝对误差门槛 `1e-5`
9. 部署包缺任一必需文件即失败；部署配置记录全部散件哈希、数据SHA、适用性和ONNX校验结果。

## 6. 配置与消融

修改文件：`abli_exp/configs/*`、`abli_exp/ablation_config.py`

- A0：可信基线。
- A1：真实删除 `norm_velocity`。
- A2：硬标签替代完整MC均值。
- A3：启用无放回四弹型等曝光采样。
- A4：启用per-munition正类权重。
- A5：共享头。
- A6：关闭physics skip。
- A7：关闭K级联。
- A8：浅M分支。
- A9：启用focal。
- A10：启用class-1 margin。
- A11/A12：只改变阈值策略。

原A1、A3、A9、A10与旧基线存在实际无变化的问题，现已改为单变量、可解释的实验。

`pyproject.toml` 的Python范围改为 `>=3.9,<3.14`，与当前PyTorch环境Python 3.9.23及代码的兼容情况一致。

## 7. 验证结果

1. 编译：
   - `nn_model.py`
   - `nn_dataset.py`
   - `nn_artifacts.py`
   - `nn_train.py`
   - `nn_eval_export.py`
   - 结果：通过。
2. 原有项目测试：26 passed。
3. 新增神经网络合同测试：6 passed，包括：
   - 负硬标签的非零软概率得以保留；
   - 标签可信度范围；
   - 真实特征消融和泄漏特征拒绝；
   - 专家/共享模型单调性和结构零；
   - 采样器单轮无重复；
   - 软目标损失反向传播；
   - 产物哈希篡改拒绝。
4. 真实30万行数据加载：
   - Train/Val/Test = 240962/29562/29476；
   - 输入13维；
   - 批次11项；
   - 完整软目标和可信度形状均为 `(B,4,2)`。
5. 真实数据单batch前向/反向：
   - 输出 `(256,4,2)`；
   - loss有限；
   - 缺失梯度参数数为0；
   - 单调性通过。
6. ONNX动态批次测试：
   - 输出 `(7,4,2)`；
   - 最大绝对误差约 `6.33e-8`；
   - 单调性通过。
7. 仿真引擎集成测试 `test_v2.py`：通过。
8. 现有旧 `output/models` 缺少新manifest，被验证器按预期拒绝。

## 8. 后续正式训练与验收

当前代码修改完成，但没有自动启动耗时较长的正式训练。必须重新训练，不能复用旧权重：

```powershell
python nn_train.py --data output/damage_dataset.parquet
python nn_eval_export.py --data output/damage_dataset.parquet
```

建议至少运行随机种子42、43、44。论文/报告应给出均值、标准差，并同时报告：

- 每任务3-class accuracy和macro-F1；
- 每弹型×任务混淆矩阵；
- 8个序数头AUPRC、Brier、NLL、ECE；
- 稀有单元正例支持数和阈值模式；
- C2 challenge结果（若已构建）；
- 3个种子的均值与标准差；
- ONNX parity与完整产物SHA合同。

只有新训练产生 `model_manifest.json`，随后评估、ONNX和部署全部通过，才能将模型标记为可用。
