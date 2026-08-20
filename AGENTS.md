# 项目执行约束

## 权威状态

- 项目根目录：当前仓库根目录。
- Python：使用项目已激活的 Python 3.9 环境。
- 当前交接文档：`docs\history\CONVERSATION_HANDOFF_20260818.md`。
- 当前源码位于 `src\loitering_munition_damage_twin`，消融配置位于 `configs\ablations`。
- 运行产物位于 `output`；实验运行目录为 `output\experiments`。

## Stage-0 合同

- 不因流水线状态文件的单个 `FAILED` 字段推断数据失败。
- 当前正式数据应为 `stage0_lineage_v2` / `stage0_ned_frd_v1`、300000 行、四弹型各 75000。
- `output\damage_dataset.parquet` SHA‑256 必须为 `a62684a42aa9c950877becebcbdf1fefabfb529bad3142710629cbb6b87a9d12`。
- `output\component_supervision.parquet` SHA‑256 必须为 `f1df9e3e0df24a8181f0c8df340e9a355a708f4d40b5040d5d3c7297fe138481`。
- component supervision 只能作为训练辅助监督，不得成为部署输入。
- Small/K2 与 Small/C2 是结构零；Med‑LM/C2 已证实可达，必须保持适用。
- 未获明确授权，不得重新生成或恢复 Stage‑0。

只读验证命令：

```powershell
lmdt-stage0-validate output/damage_dataset.parquet
lmdt-stage0-audit output/damage_dataset.parquet --output output/stage0_dataset_audit_post_generation.json
```

## A40 与 test sealing

- 固定候选：`A40_independent_mechanism_component_proxies`，seed 42。
- 配置：`configs\ablations\A40_independent_mechanism_component_proxies.json`。
- 运行目录：`output\experiments\A40_independent_mechanism_component_proxies\seed42`。
- 未获明确授权，不得启动 A40、恢复训练、启动监控或自动化。
- 训练必须使用 `--train-only`；validation promotion 必须绑定当前候选、数据、模型、阈值、scaler 和 manifest。
- 每个适用单元 accuracy ≥94%，每个适用 exact L0/L1/L2 recall ≥90%；Small/K0 FP ≤0.5%，全局 C0 FP ≤2.5%。
- promotion 未 PASS 时必须停止，不得读取 test。
- promotion PASS 也不自动授权 test；必须在揭封前获得当次明确授权。
- test 输出存在后视为已消费，严禁用于调参或重复刷分。

Validation-only 命令：

```powershell
lmdt-ablate --configs A40_independent_mechanism_component_proxies --seeds 42 --train-only --fail-fast
lmdt-promote --run-dir output/experiments/A40_independent_mechanism_component_proxies/seed42 --candidate A40_independent_mechanism_component_proxies
```

## 修改与验证

- 修改前先检查实际文件、日志、状态、SHA 和测试，不依赖旧聊天结论。
- 使用补丁修改源码，保护无关文件。
- 合同、身份绑定、promotion、test sealing 或阈值边界修改必须补回归测试。
- 先运行定向测试，再运行完整测试：

```powershell
python -m unittest tests.test_post_generation_pipeline -v
python -m unittest discover -s tests -v
```

- 长任务获准后最多每 20 分钟检查一次，只报告阶段变化、checkpoint、完成或失败。
- 不降低合同门槛、不删除真实难例、不伪造成功状态、不以 test 指导选择。
