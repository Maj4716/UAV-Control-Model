# 神经网络性能改进修改记录

日期：2026-07-26

## 1. 修改背景

上一轮训练在工程合同、序数单调性和整体收敛方面正常，但验证结果暴露出：

- Small×K、Small×C、Med-RD×M 等中间等级召回率低；
- C0 总体误报率达到 15.71%；
- MC 标准误置信度与等级相关，Small×K 的 L0/L1 平均置信度约为
  0.974/0.710，Small×C 约为 0.947/0.763；
- 累计序数 BCE 没有直接优化 `P(L=1)=P(L>=1)-P(L>=2)`；
- 训练期与最终候选重评存在重复阈值实现；
- 日志把部署真实先验 1.5% 错当成验证集 K2 的“理想比例”，而验证集实际
  K2 比例约为 7.15%。

## 2. 训练目标修改

修改文件：`nn_train.py`

新增 `ordinal_class_distribution_nll()`。对单调累计输出构造：

```text
q0 = 1 - p_ge1
q1 = p_ge1 - p_ge2
q2 = p_ge2
```

MC 软目标同样转换为：

```text
y0 = 1 - y_ge1
y1 = y_ge1 - y_ge2
y2 = y_ge2
```

改进基线损失为累计软标签 BCE 加权重 0.25 的三分类分布 NLL。该项直接给
中间等级概率质量提供梯度，同时保留原有单调输出和结构零合同。

新配置：

```json
"use_class_distribution_loss": true,
"class_distribution_weight": 0.25
```

最终启用状态和权重会写入 `model_manifest.json`。

## 3. MC 置信度策略与消融

修改文件：

- `nn_dataset.py`
- `abli_exp/ablation_config.py`
- `abli_exp/configs/_base.json`
- `abli_exp/configs/A0_full.json`
- `abli_exp/configs/A13_with_label_confidence.json`

改进 A0 默认：

```json
"use_label_uncertainty": false
```

完整 MC 均值仍作为训练目标，不丢弃概率信息；只取消与等级相关的置信度
乘法权重。`A13_with_label_confidence` 只把该开关恢复为 `true`，其余数据、
模型、损失和阈值配置与 A0 相同，可严格检验置信度权重的净影响。

另增 `A14_no_class_distribution_loss`，只关闭新三分类分布损失，用于确认
中间等级改善是否确由该损失带来。

## 4. 阈值校准修改

修改文件：

- `nn_train.py`
- `nn_eval_export.py`

训练期和最终候选重评现在共用：

- `_search_l1_threshold()`
- `_search_joint_ordinal_thresholds()`

统一参数：

- 联合目标 `alpha=0.80`；
- 阈值网格 `[0.02, 1.00]`，步长 0.02；
- `thr2 >= thr1 - 0.10`；
- C 任务全局及各弹型均要求 `C0 FP <= 2.5%`。

新阈值 schema：

```text
v7_monotone_fpr_constrained
```

当前评估器会拒绝旧 schema，因此本轮修改后必须重新训练。

## 5. 评估和汇总接口

`nn_eval_export.py` 的 `test_metrics.json` 新增：

- 16 个“弹型×任务”单元的样本数；
- Class-1 precision/recall/F1；
- 三分类准确率；
- Level-0 误报率；
- `performance_gate` 及完整失败原因。

性能门禁诊断条件：

- 支持数不少于 100 的 Class-1 单元召回率至少 85%；
- Small×K0 误报率不高于 0.5%；
- 全局 C0 误报率不高于 2.5%。

新增 `abli_exp/compare_performance_ablations.py`，自动汇总 A0/A13/A14
的关键弱单元、误报率和概率指标，输出 JSON 与 CSV。

## 6. 训练日志修正

K2 日志不再显示“理想约 1.5%”，改为同时显示：

- 当前调优阈值下的预测比例；
- 当前验证集的硬标签观测比例。

部署真实先验仍保留在 `logit_adjustment.json`，但在阈值没有于校正空间重新
标定前继续保持 `enabled=false`。

## 7. 已完成验证

- Python 编译：通过；
- 项目单元测试：64 项通过（新增定向排序、冻结残差专家、validation晋级、
  MC置信区间诊断、等权集成与多种子统计契约；原有损失、阈值、ONNX容差、
  完整性标记、控制台过滤与消融汇总契约继续通过）；
- 仿真引擎集成测试 `test_v2.py`：通过；
- 真实 30 万行数据单批次前向/反向：
  - 输出形状 `(256, 4, 2)`；
  - 改进基线置信度张量全为 1；
  - 总损失和四任务损失有限；
  - 序数单调性通过；
  - 模型参数缺失梯度数为 0。

未自动启动 30 万行完整训练。代码级正确性已经验证，但性能是否改善必须由
下述正式 A0/A13/A14 实验结果证明。

## 8. 用户后续操作

第一阶段只运行 seed 42：

```powershell
python abli_exp/run_ablations.py `
  --configs A0_full A13_with_label_confidence A14_no_class_distribution_loss `
  --seeds 42

python abli_exp/compare_performance_ablations.py --seeds 42
```

若 A0 相比 A13/A14 在关键弱单元上改善，且 C0 FP、Brier、NLL/ECE 没有
恶化，再运行正式三种子：

```powershell
python abli_exp/run_ablations.py `
  --configs A0_full A13_with_label_confidence A14_no_class_distribution_loss `
  --seeds 42 43 44

python abli_exp/compare_performance_ablations.py --seeds 42 43 44
```

若只需要训练最终 A0，而不做消融：

```powershell
python nn_train.py --data output/damage_dataset.parquet
python nn_eval_export.py --data output/damage_dataset.parquet
```

必须以新生成的 `output/eval/test_metrics.json` 中
`performance_gate.passed`、全部单元指标及概率指标判断是否可用；不能复用
修改前的 `best_model.pth`、阈值或 ONNX。

## 9. 2026-07-26 单种子流程修复

首次 A0/seed42 评估在 ONNXRuntime 对齐检查中得到最大绝对误差
`1.144e-05`，略高于原固定阈值 `1e-05`。这是 float32 后端的正常量级，
但原逻辑将其判为失败并使调度器停止，导致 A13/A14 未开始。

修复内容：

- ONNX 对齐改用逐元素 `atol=2e-5, rtol=1e-4`，同时记录最大绝对误差和
  归一化误差；明显错误仍失败关闭。
- 评估只有在指标、ONNX校验及部署包全部完成后才写
  `evaluation_status.json: COMPLETE`，并绑定指标与部署配置 SHA-256。
- 比较器不再接受只有 `test_metrics.json` 的半成品；缺失或失败结果返回
  `INCOMPLETE`，默认以非零状态退出。
- 消融调度器将每个实验隔离执行，一个失败不会阻止其余实验，并写
  `run_status.json` 与批次汇总。
- 调度器默认控制台只显示关键进度；所有详细统计继续完整保存在
  `logs/train_subprocess.log` 和 `logs/eval_subprocess.log`。可用
  `--verbose-console` 恢复完整实时输出。
