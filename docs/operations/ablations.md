# 消融实验框架

本目录集中存放消融实验的配置、运行脚本和结果。A0 是改进基线；其余配置每次只启用或关闭一个真实生效的变量。

## 目录约定

```text
src/loitering_munition_damage_twin/experiments/
  configs/                 # 每个消融实验一个 JSON 配置
  lmdt-ablate          # 一键式 Python 调度脚本
  compare_performance_ablations.py  # 汇总关键性能指标和实验差值
  analyze_single_seed_predictions.py # 配对root bootstrap与单元可分性分析
  results/
    A0_full/
      seed42/
        config_resolved.json
        run_manifest.json
        logs/
        output/
          models/
          runs/damage_model/
          eval/
```

## 常用命令

快速检查框架是否能跑通：

```bash
lmdt-ablate --configs A0_full --seeds 42 --smoke-test
```

冒烟测试会训练较少 epoch，并在该实验目录下保存临时 `best_model.pth` 与 `best_thresholds.json`，用于继续验证评估与导出链路。该结果只用于检查流程，不应写入论文表格。

运行全部核心消融组，单个随机种子：

```bash
lmdt-ablate --seeds 42
```

运行全部核心消融组，三个随机种子：

```bash
lmdt-ablate --seeds 42 43 44
```

只运行部分实验：

```bash
lmdt-ablate --configs A0_full A3_balanced_sampler A7_no_k_cascade --seeds 42
```

## 输出文件

每个 `seed` 目录下会独立保存：

- `config_resolved.json`：合并默认值后的完整配置。
- `run_manifest.json`：实验编号、随机种子、路径等运行元数据。
- `run_status.json`：该实验当前处于运行、完成或失败状态；失败时记录阶段和错误。
- `logs/train_subprocess.log` 与 `logs/eval_subprocess.log`：调度脚本捕获的完整控制台输出。
- `output/models/best_model.pth`：当前实验的最优模型。
- `output/models/best_thresholds.json`：训练阶段保存的阈值与选择指标。
- `output/models/minmax_scaler.pkl/json`：当前实验对应的特征缩放器。
- `output/models/model_manifest.json`：数据、权重、阈值、缩放器和模型结构的哈希合同。
- `output/runs/damage_model/*.png`：训练过程图、混淆矩阵、最终指标摘要等。
- `output/eval/test_metrics.json`：测试集机器可读指标。
- `output/eval/evaluation_status.json`：只有评估、ONNX校验和部署打包全部完成后
  才写为 `COMPLETE`，并绑定指标文件哈希。
- `output/eval/predictions.csv`：含样本/root标识、等级、概率和MC均值目标的逐样本结果。

调度器默认只在控制台显示每轮 epoch、选择分数、最终指标、性能门禁和失败摘要，
完整诊断仍全部写入上述日志。需要查看旧式完整实时输出时增加
`--verbose-console`。单个实验失败后，默认会继续执行其余实验，并在
`results/ablation_run_summary.json` 汇总；需要首错即停时增加 `--fail-fast`。

## 校准类消融

`A11_global_thresholds` 与 `A12_fixed_0_5_thresholds` 默认复用 `A0_full` 同一随机种子的训练产物，只改变评估阶段的阈值策略。这样可以把“模型能力差异”和“阈值校准差异”分开。运行这两个实验前，请先完成对应 seed 的 `A0_full`。

## 本轮性能改进消融

- `A0_full`：关闭类别相关的 MC 置信度乘子，启用权重 0.25 的显式三分类
  分布 NLL。
- `A13_with_label_confidence`：只重新启用历史 MC 置信度乘子，其余配置与
  A0 完全相同。
- `A14_no_class_distribution_loss`：只关闭显式三分类分布 NLL，其余配置与
  A0 完全相同。

先用一个种子判断方向：

```powershell
lmdt-ablate `
  --configs A0_full A13_with_label_confidence A14_no_class_distribution_loss `
  --seeds 42

python -m loitering_munition_damage_twin.experiments.compare_performance_ablations --seeds 42
```

若某次评估失败但训练模型已经生成，可恢复而不重复训练：

```powershell
lmdt-ablate `
  --configs A0_full A13_with_label_confidence A14_no_class_distribution_loss `
  --seeds 42 --skip-existing
```

比较器只接受带有 `COMPLETE` 标记且 SHA-256 匹配的结果。请求的实验不完整时，
汇总状态为 `INCOMPLETE` 并返回非零状态，不再把部分结果误报为 `OK`；仅在明确
需要临时查看部分结果时使用 `--allow-incomplete`。

比较器生成：

- `output/experiments/performance_ablation_summary.json`
- `output/experiments/performance_ablation_summary.csv`

重点查看 Small×K/C、Med-RD×M、Heavy×M、三种 K 单元的 Class-1 recall，
以及 `global_c0_fp`、Brier、MC 均值交叉熵和 ECE。方向成立后再运行
`--seeds 42 43 44`，不要用单个随机种子作为最终结论。

## 单种子结果后的候选基线

seed42表明完整MC置信度总体优于全局关闭置信度，三分类分布损失也具有稳定
正收益。下一轮使用2×2因子设计：

- `A13_with_label_confidence`：完整置信度、普通三分类损失；
- `A15_selective_confidence`：只绕过Small×K置信度；
- `A16_weak_cell_middle_loss`：完整置信度、弱单元精确L1增强；
- `A17_selective_confidence_weak_cell_loss`：同时启用两项，为候选基线。

配置矩阵的行顺序均为K/M/F/C，列顺序均为
Small/Med-LM/Med-RD/Heavy。详细依据、矩阵和验收条件见仓库根目录
`NN_SINGLE_SEED_FOLLOWUP_MODIFICATION_RECORD.md`。

## A22–A26与三种子可信度复验

- `A22_targeted_ranking`：混合批次低FPR/条件排序，validation晋级失败；
- `A23_frozen_cell_residual_adapters`：冻结A13，只训练零初始化单元残差专家；
- `A24_hard_boundary_residual_adapters`：冻结残差专家直接学习硬等级边界；
- `A25_equal_weight_seed_ensemble`：固定seed42/43/44各1/3的概率集成；
- `A26_nominal_softmax_heads`：显式学习L0/L1/L2三类概率，再转换为单调
  累积概率。

A22–A24均未超过A19 validation基线，且没有读取测试集。A25在validation上
改善全部预注册排序指标，但严格失败项从最佳成员8项增至9项，因此同样停止在
validation。A26平均准确率下降0.4355个百分点、失败项为9且8个弱单元排序
目标平均下降0.00947，也在validation停止且没有生成测试指标。

三种子A19正式统计：

```powershell
python -m loitering_munition_damage_twin.experiments.summarize_multiseed_reliability
```

结果为平均准确率93.0811%、样本标准差0.0877个百分点；Small×K0和全局C0
安全门禁3/3通过，严格精确L1召回门禁0/3通过。输出写入
`results/a19_multiseed_reliability.json/.csv`。

生成最终单种子决策（会校验所有被拒候选均未产生测试指标）：

```powershell
python -m loitering_munition_damage_twin.experiments.summarize_single_seed_decision
```

输出为`results/single_seed_final_decision.json`。当前结论是A19相对A0平均
准确率提高0.2146个百分点、失败项11→8，但严格部署仍为`REJECT`。

逐样本可信度分析示例：

```powershell
python -m loitering_munition_damage_twin.experiments.analyze_single_seed_predictions `
  --experiments A13_with_label_confidence `
                A15_selective_confidence `
                A16_weak_cell_middle_loss `
                A17_selective_confidence_weak_cell_loss `
  --reference A13_with_label_confidence `
  --seed 42
```

该脚本要求各实验样本、root和标签严格对齐，并报告配对root-cluster
bootstrap区间。它评估同一seed测试集的不确定性，不能代替seed42/43/44的
训练方差。

## seed42候选结论与阈值再校准

A15、A16、A17已完成正式训练，但相对A13的平均准确率分别显著下降
0.180、0.265、0.128个百分点，门禁失败数分别为11、11、12，均已否定。
A13仍是默认平衡基线。

阈值搜索已升级为`v8_exact_l1_floor_constrained`，使验证集校准目标与最终
精确L1召回门禁一致，同时保持误报约束优先。校准类实验复用A13权重：

- `A18_exact_class1_floor_calibration`：无准确率预算的召回优先工作点；
- `A19_bounded_class1_floor_calibration`：每单元0.5个百分点准确率预算；
- `A20_pareto_class1_calibration`：87%召回目标、1个百分点预算；
- `A21_pareto_class1_calibration_1p5`：最终1.5个百分点预算探测。

A18将失败单元从8降至5，但平均准确率下降0.606个百分点，只能作为明确标注
的召回优先模式。A19–A21保持A13准确率，但没有减少失败单元。为防止污染
seed42 test，A21后停止继续调阈值预算。

再校准由`recalibrate_checkpoint.py`完成，只使用validation；模型权重必须与
来源实验SHA-256一致，阈值更新后重新封存manifest。详细证据和下一阶段路径见
`NN_SINGLE_SEED_FOLLOWUP_MODIFICATION_RECORD.md`。
