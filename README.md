# 巡飞弹毁伤评估数字孪生

本项目包含轻量毁伤仿真、Stage‑0 数据合同、代理神经网络、消融实验与可视化工具。代码采用标准 `src` 包结构，面向 Python 3.9。

## 当前状态

- Stage‑0 正式数据已生成并验收：300000 行、132 列，合同状态 `CURRENT_V2`。
- 主数据 SHA‑256：`a62684a42aa9c950877becebcbdf1fefabfb529bad3142710629cbb6b87a9d12`。
- component supervision SHA‑256：`f1df9e3e0df24a8181f0c8df340e9a355a708f4d40b5040d5d3c7297fe138481`。
- A40/seed42 尚未训练，validation promotion 尚未执行，held‑out test 仍封存。
- 大型数据、旧权重和逐样本预测不进入 Git；完整历史快照位于相邻本地归档目录。

## 目录结构

```text
src/loitering_munition_damage_twin/
  simulation/       仿真物理、坐标系、车辆与装甲资源
  stage0/           数据生成、验证、审计和谱系工具
  surrogate/        特征、模型、训练和评估
  experiments/      消融、诊断、promotion 和严格门禁
  visualization/    数据与车辆可视化
configs/ablations/  消融实验配置
reports/legacy_ablations/  轻量历史结果元数据
docs/               操作、验证、历史与答辩资料
notebooks/          后续飞控探索
tests/              自动化合同测试
output/             本地数据与运行产物，Git 忽略
```

## 安装

在项目指定环境中安装：

```powershell
python -m pip install -e . --no-deps
```

如需在新环境安装全部依赖：

```powershell
python -m pip install -e ".[ui,test]"
```

## 常用命令

```powershell
# 自动化测试
python -m unittest discover -s tests -v

# 只读验证现有 Stage-0 数据
lmdt-stage0-validate output/damage_dataset.parquet
lmdt-stage0-audit output/damage_dataset.parquet --output output/stage0_dataset_audit_post_generation.json

# 可视化
python -m streamlit run src/loitering_munition_damage_twin/visualization/app.py

# 查看实验入口，不会启动训练
lmdt-ablate --help
```

Stage‑0 生成入口带有额外确认参数，避免误覆盖昂贵产物：

```powershell
lmdt-stage0-generate --confirm-generation
```

只有在明确决定重新生成数据时才可执行该命令。正常验证、整理或训练准备不需要重新生成。

## A40 与封存测试

A40 只能先进行 validation-only 训练：

```powershell
lmdt-ablate --configs A40_independent_mechanism_component_proxies --seeds 42 --train-only --fail-fast
lmdt-promote --run-dir output/experiments/A40_independent_mechanism_component_proxies/seed42 --candidate A40_independent_mechanism_component_proxies
```

promotion `PASS` 只是技术前置条件，不代表允许打开 test。一次性 test 揭封必须在当次操作前获得明确授权，详细规则见 [AGENTS.md](AGENTS.md)。

## 文档与归档

- 当前操作约束：[AGENTS.md](AGENTS.md)
- 历史交接：[docs/history/CONVERSATION_HANDOFF_20260818.md](docs/history/CONVERSATION_HANDOFF_20260818.md)
- 数据验证记录：[docs/validation/DATASET_VALIDATION_REPORT.md](docs/validation/DATASET_VALIDATION_REPORT.md)
- 消融说明：[docs/operations/ablations.md](docs/operations/ablations.md)
- 完整重构前快照：相邻目录 `..\代码_archive_20260820\workspace_snapshot`

## Git 原则

`.gitignore` 已排除 Parquet、权重、ONNX、预测明细、日志、缓存和本机配置。提交前始终检查：

```powershell
git diff --cached --check
git status
```
