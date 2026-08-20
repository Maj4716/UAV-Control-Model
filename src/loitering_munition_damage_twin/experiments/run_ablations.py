from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from loitering_munition_damage_twin.experiments.ablation_config import (
    load_ablation_config,
    resolve_output_dir,
    write_resolved_config,
)
from loitering_munition_damage_twin.paths import (
    ABLATION_CONFIG_ROOT,
    EXPERIMENT_OUTPUT_ROOT,
    PROJECT_ROOT,
)


CONFIG_DIR = ABLATION_CONFIG_ROOT
REPO_ROOT = PROJECT_ROOT
DEFAULT_CONFIGS = [
    "A0_full",
    "A1_no_physics_features",
    "A2_hard_labels",
    "A3_balanced_sampler",
    "A4_per_munition_pos_weight",
    "A5_shared_head",
    "A6_no_physics_skip",
    "A7_no_k_cascade",
    "A8_shallow_m_branch",
    "A9_focal_loss",
    "A10_ordinal_margin",
    "A11_global_thresholds",
    "A12_fixed_0_5_thresholds",
    "A13_with_label_confidence",
    "A14_no_class_distribution_loss",
    "A15_selective_confidence",
    "A16_weak_cell_middle_loss",
    "A17_selective_confidence_weak_cell_loss",
    "A18_exact_class1_floor_calibration",
    "A19_bounded_class1_floor_calibration",
    "A22_targeted_ranking",
    "A23_frozen_cell_residual_adapters",
    "A24_hard_boundary_residual_adapters",
    "A26_nominal_softmax_heads",
    "A27_mechanism_decomposition",
    "A28_terminal_physics_features",
    "A29_mechanism_with_terminal_physics",
]

SUMMARY_CONSOLE_PREFIXES = (
    "[Training] Booting",
    "[Training][Ablation] experiment_id=",
    "[Training] Sealed",
    "[Dataset] [V2 Gate]",
    "[Dataset] 切分完成",
    "Epoch ",
    "    -> Selection",
    "    [*] Saved best_model.pth",
    "[EARLY STOP]",
    "[FINAL-CANDIDATES] Selected",
    "  Training Finished!",
    "  Final model variant:",
    "  Best epoch (selection_score):",
    "  Best selection score:",
    "  3-class Acc:",
    "  Class-1 diagonal recall",
    "    K=",
    "[EVAL] 正在加载数据集",
    "[EVAL] 成功读入合同校验后的最优权重",
    "[EVAL] Cell-level guardrail check",
    "    Performance gate:",
    "      - ",
    "[EVAL] Machine-readable metrics written",
    "[EVAL] Pipeline completion marker:",
    "[EXPORT] ONNX checker + Runtime parity",
    "[DEPLOY] version=",
    "[RECAL]",
    "[VALIDATION]",
    "RuntimeError:",
)


def _config_path(name: str) -> Path:
    path = Path(name)
    if path.suffix != ".json":
        path = path.with_suffix(".json")
    if not path.is_absolute():
        path = CONFIG_DIR / path
    return path


def _should_echo_summary_line(line: str) -> bool:
    return line.startswith(SUMMARY_CONSOLE_PREFIXES)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    os.replace(temporary_path, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(cmd: list[str], log_path: Path,
         verbose_console: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[RUN]", " ".join(cmd))
    print(f"[RUN] 完整输出日志: {log_path.resolve()}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    tail: list[str] = []
    echoed_gate_failures = 0
    total_gate_failures = 0
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            tail.append(line.rstrip())
            if len(tail) > 12:
                tail.pop(0)
            if line.startswith("      - "):
                total_gate_failures += 1
                if not verbose_console and echoed_gate_failures >= 3:
                    continue
                echoed_gate_failures += 1
            if verbose_console or _should_echo_summary_line(line):
                print(line, end="")
        rc = proc.wait()
    if not verbose_console and total_gate_failures > echoed_gate_failures:
        print(
            f"[RUN] 另有 {total_gate_failures - echoed_gate_failures} 项"
            f"门禁失败；完整清单见 {log_path.resolve()}")
    if rc != 0:
        print(f"[RUN][FAILED] exit={rc}; 完整诊断见 {log_path.resolve()}")
        if not verbose_console:
            print("[RUN][FAILED] 日志末尾:")
            for line in tail:
                print(f"  {line}")
        raise subprocess.CalledProcessError(rc, cmd)


def _write_run_manifest(run_dir: Path, config: dict, seed: int,
                        config_path: Path, smoke_test: bool) -> None:
    package_root = REPO_ROOT / "src" / "loitering_munition_damage_twin"
    source_files = (
        package_root / "surrogate" / "training.py",
        package_root / "surrogate" / "dataset.py",
        package_root / "surrogate" / "features.py",
        package_root / "surrogate" / "model.py",
        package_root / "surrogate" / "artifacts.py",
        package_root / "stage0" / "component_supervision.py",
        package_root / "surrogate" / "evaluation.py",
        package_root / "experiments" / "ablation_config.py",
        Path(__file__).resolve(),
    )
    payload = {
        "schema": "stage0_ablation_run_manifest_v2",
        "experiment_id": config.get("experiment_id"),
        "description": config.get("description"),
        "seed": seed,
        "smoke_test": smoke_test,
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "run_dir": str(run_dir),
        "runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "source_sha256": {
            str(path.relative_to(REPO_ROOT)): _sha256_file(path)
            for path in source_files
        },
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _passed_validation_promotion(
        run_dir: Path,
        candidate: str | None = None) -> Path | None:
    """Return a test-blind PASS report that authorizes test evaluation."""
    validation_dir = run_dir / "output" / "validation"
    candidates = sorted({
        *validation_dir.glob("*promotion*.json"),
        *validation_dir.glob("validation_promotion.json"),
        *run_dir.glob("*promotion*.json"),
        *run_dir.glob("validation_promotion.json"),
    })
    for path in candidates:
        try:
            with path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, ValueError):
            continue
        if (
            payload.get("status") == "PASS"
            and payload.get("test_metrics_read") is False
            and (
                candidate is None
                or payload.get("candidate") == str(candidate)
            )
        ):
            return path
    return None


def _build_evaluation_command(
        config_path: Path,
        run_dir: Path,
        seed: int,
        promotion_path: Path,
        data: str | None = None) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "loitering_munition_damage_twin.surrogate.evaluation",
        "--ablation-config",
        str(config_path),
        "--output-dir",
        str(run_dir),
        "--seed",
        str(seed),
        "--promotion-report",
        str(promotion_path),
    ]
    if data:
        command.extend(["--data", data])
    return command


def _copy_reused_training_artifacts(config: dict, run_dir: Path, seed: int) -> bool:
    execution = config.get("execution", {})
    reuse_from = execution.get("reuse_train_from")
    if not reuse_from:
        return False

    source_cfg = load_ablation_config(str(_config_path(reuse_from)))
    source_run = Path(resolve_output_dir(source_cfg, seed, repo_root=str(REPO_ROOT)))
    source_models = source_run / "output" / "models"
    target_models = run_dir / "output" / "models"
    if not source_models.exists():
        raise FileNotFoundError(
            f"{config.get('experiment_id')} reuses training artifacts from "
            f"{reuse_from}, but source models are missing: {source_models}. "
            f"Run the source experiment first."
        )
    target_models.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_models, target_models, dirs_exist_ok=True)
    with (run_dir / "reused_training_from.txt").open("w", encoding="utf-8") as f:
        f.write(str(source_run))
        f.write("\n")
    print(f"[REUSE] {config.get('experiment_id')} copied training artifacts from {source_run}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run configured ablation experiments under output/experiments."
    )
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS,
                        help="Config names without .json, or explicit JSON paths.")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42],
                        help="Random seeds to run for each config.")
    parser.add_argument("--data", default=None,
                        help="Optional dataset path override.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Pass --smoke_test to nn_train.py for a quick wiring check.")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--allow-test-evaluation", action="store_true",
        help=(
            "Permit test evaluation only after the run directory contains "
            "a validation-promotion PASS report with "
            "test_metrics_read=false."))
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip training when best_model.pth already exists.")
    parser.add_argument(
        "--verbose-console", action="store_true",
        help="Echo every subprocess line. By default only key progress is shown "
             "while complete output is retained in each run's logs directory.")
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="Stop at the first failed run. The default records the failure and "
             "continues with the remaining requested experiments.")
    args = parser.parse_args()

    if args.train_only and args.eval_only:
        raise ValueError("--train-only and --eval-only cannot be used together.")
    if args.eval_only and not args.allow_test_evaluation:
        raise ValueError(
            "--eval-only reads the sealed test split and therefore requires "
            "--allow-test-evaluation plus a test-blind PASS promotion report.")

    run_results = []
    for config_name in args.configs:
        config_path = _config_path(config_name)
        config = load_ablation_config(str(config_path))
        for seed in args.seeds:
            run_dir = Path(resolve_output_dir(config, seed, repo_root=str(REPO_ROOT)))
            experiment_id = str(config.get("experiment_id") or config_name)
            status_path = run_dir / "run_status.json"
            phase = "prepare"
            _write_json(status_path, {
                "schema": "stage0_ablation_run_status_v1",
                "status": "RUNNING",
                "experiment_id": experiment_id,
                "seed": int(seed),
                "phase": phase,
            })
            try:
                _write_run_manifest(
                    run_dir, config, seed, config_path, args.smoke_test)
                write_resolved_config(
                    config,
                    str(run_dir / "config_resolved.json"),
                    extra={"seed": seed, "output_dir": str(run_dir)},
                )

                train_cmd = [
                    sys.executable,
                    "-m",
                    "loitering_munition_damage_twin.surrogate.training",
                    "--ablation-config",
                    str(config_path),
                    "--output-dir",
                    str(run_dir),
                    "--seed",
                    str(seed),
                ]
                if args.data:
                    train_cmd.extend(["--data", args.data])
                if args.smoke_test:
                    train_cmd.append("--smoke_test")

                model_path = (
                    run_dir / "output" / "models" / "best_model.pth")
                reused_training = _copy_reused_training_artifacts(
                    config, run_dir, seed)
                if not args.eval_only:
                    phase = "train"
                    if reused_training:
                        print(
                            "[SKIP] Training skipped for reuse-only "
                            f"ablation: {experiment_id}")
                    elif args.skip_existing and model_path.exists():
                        print(f"[SKIP] Existing model found: {model_path}")
                    else:
                        _run(
                            train_cmd,
                            run_dir / "logs" / "train_subprocess.log",
                            verbose_console=args.verbose_console,
                        )

                execution = config.get("execution", {})
                recalibrate_thresholds = bool(
                    execution.get("recalibrate_thresholds", False))
                evaluate_test = bool(args.allow_test_evaluation)
                promotion_path = None
                if recalibrate_thresholds or evaluate_test:
                    promotion_path = _passed_validation_promotion(
                        run_dir, experiment_id)
                    if promotion_path is None:
                        raise RuntimeError(
                            "Validation-gated operation refused: no "
                            "test-blind PASS promotion bound to candidate "
                            f"{experiment_id!r} exists under "
                            f"{run_dir / 'output' / 'validation'}.")

                if recalibrate_thresholds:
                    phase = "recalibrate"
                    configured_data = (
                        args.data
                        or config.get("paths", {}).get("data")
                        or str(REPO_ROOT / "output" / "damage_dataset.parquet")
                    )
                    configured_data_path = Path(configured_data)
                    if not configured_data_path.is_absolute():
                        configured_data_path = REPO_ROOT / configured_data_path
                    recalibrate_cmd = [
                        sys.executable,
                        "-m",
                        (
                            "loitering_munition_damage_twin.experiments."
                            "recalibrate_checkpoint"
                        ),
                        "--run-dir",
                        str(run_dir),
                        "--ablation-config",
                        str(config_path),
                        "--data",
                        str(configured_data_path),
                        "--seed",
                        str(seed),
                        "--promotion-report",
                        str(promotion_path),
                    ]
                    _run(
                        recalibrate_cmd,
                        run_dir / "logs" / "recalibrate_subprocess.log",
                        verbose_console=args.verbose_console,
                    )

                if evaluate_test:
                    print(
                        "[TEST AUTH] validation promotion PASS: "
                        f"{promotion_path}")
                    phase = "evaluate"
                    eval_cmd = _build_evaluation_command(
                        config_path,
                        run_dir,
                        seed,
                        promotion_path,
                        data=args.data,
                    )
                    _run(
                        eval_cmd,
                        run_dir / "logs" / "eval_subprocess.log",
                        verbose_console=args.verbose_console,
                    )

                final_status = (
                    "COMPLETE" if evaluate_test
                    else "VALIDATION_COMPLETE")
                performance_gate_passed = None
                evaluation_status = None
                if evaluate_test:
                    evaluation_status_path = (
                        run_dir / "output" / "eval"
                        / "evaluation_status.json")
                    with evaluation_status_path.open(
                            "r", encoding="utf-8") as stream:
                        evaluation_status = json.load(stream)
                    if evaluation_status.get("status") != "COMPLETE":
                        raise RuntimeError(
                            "Test subprocess returned without a COMPLETE "
                            "evaluation_status.json.")
                    performance_gate_passed = bool(
                        evaluation_status.get(
                            "performance_gate_passed", False))
                result = {
                    "schema": "stage0_ablation_run_status_v1",
                    "status": final_status,
                    "experiment_id": experiment_id,
                    "seed": int(seed),
                    "phase": (
                        "test_complete"
                        if evaluate_test
                        else "validation_complete"),
                    "run_dir": str(run_dir),
                    "performance_gate_passed":
                        performance_gate_passed,
                    "test_metrics_read": evaluate_test,
                }
                if evaluate_test:
                    result["promotion_report"] = str(
                        promotion_path)
                    result["evaluation_status"] = str(
                        evaluation_status_path)
                _write_json(status_path, result)
                run_results.append(result)
                print(
                    f"[RUN][{final_status}] "
                    f"{experiment_id}/seed{seed}"
                    + (
                        " | performance_gate="
                        + (
                            "PASS"
                            if performance_gate_passed
                            else "FAIL"
                        )
                        if evaluate_test else
                        " | test=SEALED"
                    ))
            except Exception as exc:
                result = {
                    "schema": "stage0_ablation_run_status_v1",
                    "status": "FAILED",
                    "experiment_id": experiment_id,
                    "seed": int(seed),
                    "phase": phase,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "run_dir": str(run_dir),
                }
                _write_json(status_path, result)
                run_results.append(result)
                print(
                    f"[RUN][FAILED] {experiment_id}/seed{seed} "
                    f"phase={phase}: {exc}")
                if args.fail_fast:
                    raise

    failures = [
        result for result in run_results
        if result["status"] == "FAILED"
    ]
    summary = {
        "schema": "stage0_ablation_batch_status_v1",
        "status": "COMPLETE" if not failures else "COMPLETED_WITH_FAILURES",
        "requested_runs": len(run_results),
        "completed_runs": sum(
            result["status"] in {
                "COMPLETE", "TRAIN_COMPLETE", "VALIDATION_COMPLETE"
            }
            for result in run_results),
        "failed_runs": len(failures),
        "runs": run_results,
    }
    summary_path = EXPERIMENT_OUTPUT_ROOT / "ablation_run_summary.json"
    _write_json(summary_path, summary)
    print(json.dumps({
        "status": summary["status"],
        "requested_runs": summary["requested_runs"],
        "completed_runs": summary["completed_runs"],
        "failed_runs": summary["failed_runs"],
        "summary": str(summary_path),
    }, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
