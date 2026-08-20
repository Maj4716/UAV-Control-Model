from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


TASKS = ("K", "M", "F", "C")
MUNITION_NAMES = ("Small", "Med-LM", "Med-RD", "Heavy")
REPORT_SCHEMA = "stage0_nn_strict_performance_goal_v1"
ORDINAL_APPLICABILITY = (
    ((True, False), (True, True), (True, True), (True, False)),
    ((True, True), (True, True), (True, True), (True, True)),
    ((True, True), (True, True), (True, True), (True, True)),
    ((True, True), (True, True), (True, True), (True, True)),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    os.replace(temporary_path, path)


def _wilson_interval(
        successes: int,
        total: int,
        z_score: float = 1.959963984540054,
) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    probability = successes / total
    z2 = z_score * z_score
    denominator = 1.0 + z2 / total
    center = (probability + z2 / (2.0 * total)) / denominator
    radius = (
        z_score
        * math.sqrt(
            probability * (1.0 - probability) / total
            + z2 / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _rate_record(successes: int, total: int) -> dict:
    lower, upper = _wilson_interval(successes, total)
    return {
        "correct": int(successes),
        "support": int(total),
        "percent": (
            100.0 * float(successes) / float(total)
            if total else None
        ),
        "wilson_95_percent": [
            100.0 * lower if total else None,
            100.0 * upper if total else None,
        ],
    }


def _require_columns(frame: pd.DataFrame) -> None:
    required = {"sample_id", "root_seed_id", "munition_id"}
    for task in TASKS:
        required.update({f"true_{task}", f"pred_{task}"})
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "Predictions artifact is missing required columns: "
            + ", ".join(missing)
        )
    if frame.empty:
        raise ValueError("Predictions artifact contains no rows.")
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("Predictions artifact contains duplicate sample_id.")
    munition_ids = frame["munition_id"].to_numpy()
    if not np.isin(munition_ids, np.arange(len(MUNITION_NAMES))).all():
        raise ValueError("munition_id must be one of 0,1,2,3.")
    for task in TASKS:
        true_values = frame[f"true_{task}"].to_numpy()
        predicted_values = frame[f"pred_{task}"].to_numpy()
        if not np.isin(true_values, (0, 1, 2)).all():
            raise ValueError(f"true_{task} contains a value outside 0/1/2.")
        if not np.isin(predicted_values, (0, 1, 2)).all():
            raise ValueError(f"pred_{task} contains a value outside 0/1/2.")


def _load_and_verify_run(run_dir: Path) -> tuple[pd.DataFrame, dict, dict]:
    evaluation_dir = run_dir / "output" / "eval"
    predictions_path = evaluation_dir / "predictions.csv"
    metrics_path = evaluation_dir / "test_metrics.json"
    status_path = evaluation_dir / "evaluation_status.json"
    run_status_path = run_dir / "run_status.json"
    for path in (
        predictions_path,
        metrics_path,
        status_path,
        run_status_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required evaluation artifact missing: {path}")

    with status_path.open("r", encoding="utf-8") as stream:
        evaluation_status = json.load(stream)
    with run_status_path.open("r", encoding="utf-8") as stream:
        run_status = json.load(stream)
    with metrics_path.open("r", encoding="utf-8") as stream:
        metrics = json.load(stream)

    if evaluation_status.get("status") != "COMPLETE":
        raise ValueError("evaluation_status.json is not COMPLETE.")
    if run_status.get("status") != "COMPLETE":
        raise ValueError("run_status.json is not COMPLETE.")
    expected_metrics_hash = evaluation_status.get("metrics_sha256")
    actual_metrics_hash = _sha256_file(metrics_path)
    if expected_metrics_hash != actual_metrics_hash:
        raise ValueError(
            "test_metrics.json SHA-256 does not match evaluation_status.json.")

    frame = pd.read_csv(predictions_path)
    _require_columns(frame)
    identity = {
        "run_dir": str(run_dir.resolve()),
        "experiment_id": evaluation_status.get("experiment_id"),
        "seed": evaluation_status.get("seed"),
        "rows": int(len(frame)),
        "unique_root_families": int(
            frame["root_seed_id"].astype(str).nunique()),
        "predictions_sha256": _sha256_file(predictions_path),
        "test_metrics_sha256": actual_metrics_hash,
        "dataset_sha256": metrics.get("dataset_sha256"),
        "model_sha256": metrics.get("model_sha256"),
    }
    return frame, metrics, identity


def evaluate_strict_goal(
        frame: pd.DataFrame,
        *,
        minimum_global_class_recall_percent: float = 90.0,
        minimum_munition_task_accuracy_percent: float = 94.0,
        minimum_munition_task_l1_recall_percent: float = 90.0,
        maximum_small_k0_false_positive_percent: float = 0.5,
        maximum_global_c0_false_positive_percent: float = 2.5,
        minimum_global_class_support: int = 50,
        minimum_munition_l1_support: int = 100,
) -> dict:
    _require_columns(frame)
    criteria = {
        "comparison": "greater_than_or_equal_to_for_minima",
        "minimum_global_class_recall_percent":
            float(minimum_global_class_recall_percent),
        "minimum_munition_task_accuracy_percent":
            float(minimum_munition_task_accuracy_percent),
        "minimum_munition_task_l1_recall_percent":
            float(minimum_munition_task_l1_recall_percent),
        "minimum_applicable_cell_diagonal_recall_percent":
            float(minimum_global_class_recall_percent),
        "maximum_small_k0_false_positive_percent":
            float(maximum_small_k0_false_positive_percent),
        "maximum_global_c0_false_positive_percent":
            float(maximum_global_c0_false_positive_percent),
        "minimum_global_class_support":
            int(minimum_global_class_support),
        "minimum_munition_l1_support":
            int(minimum_munition_l1_support),
    }
    failures: list[str] = []
    global_class_recall = {}
    for task in TASKS:
        true_values = frame[f"true_{task}"].to_numpy()
        predicted_values = frame[f"pred_{task}"].to_numpy()
        task_records = {}
        for level in (0, 1, 2):
            mask = true_values == level
            support = int(mask.sum())
            correct = int((predicted_values[mask] == level).sum())
            record = _rate_record(correct, support)
            task_records[f"L{level}"] = record
            if support < minimum_global_class_support:
                failures.append(
                    f"global {task}/L{level} support {support} "
                    f"< {minimum_global_class_support}"
                )
            elif record["percent"] < minimum_global_class_recall_percent:
                failures.append(
                    f"global {task}/L{level} recall "
                    f"{record['percent']:.2f}% < "
                    f"{minimum_global_class_recall_percent:.2f}% "
                    f"(n={support})"
                )
        global_class_recall[task] = task_records

    munition_task_metrics = {}
    for munition_id, munition_name in enumerate(MUNITION_NAMES):
        munition_mask = frame["munition_id"].to_numpy() == munition_id
        munition_records = {}
        for task in TASKS:
            task_id = TASKS.index(task)
            true_values = frame.loc[munition_mask, f"true_{task}"].to_numpy()
            predicted_values = frame.loc[
                munition_mask, f"pred_{task}"].to_numpy()
            accuracy = _rate_record(
                int((predicted_values == true_values).sum()),
                int(len(true_values)),
            )
            l1_mask = true_values == 1
            l1_recall = _rate_record(
                int((predicted_values[l1_mask] == 1).sum()),
                int(l1_mask.sum()),
            )
            class_diagonal = {}
            for level in (0, 1, 2):
                applicable = (
                    True if level == 0
                    else bool(
                        ORDINAL_APPLICABILITY[
                            munition_id][task_id][level - 1])
                )
                level_mask = true_values == level
                diagonal = _rate_record(
                    int((
                        predicted_values[level_mask] == level
                    ).sum()),
                    int(level_mask.sum()),
                )
                diagonal["applicable"] = applicable
                diagonal["status"] = "NOT_APPLICABLE"
                if applicable:
                    minimum_recall = (
                        max(
                            minimum_global_class_recall_percent,
                            minimum_munition_task_l1_recall_percent,
                        )
                        if level == 1
                        else minimum_global_class_recall_percent
                    )
                    if (
                        diagonal["support"]
                        < minimum_munition_l1_support
                    ):
                        diagonal["status"] = "INSUFFICIENT_EVIDENCE"
                        failures.append(
                            f"{munition_name}/{task}/L{level} support "
                            f"{diagonal['support']} < "
                            f"{minimum_munition_l1_support}"
                        )
                    elif (
                        diagonal["percent"]
                        < minimum_recall
                    ):
                        diagonal["status"] = "FAIL"
                        failures.append(
                            f"{munition_name}/{task}/L{level} "
                            f"diagonal recall "
                            f"{diagonal['percent']:.2f}% < "
                            f"{minimum_recall:.2f}% "
                            f"(n={diagonal['support']})"
                        )
                    else:
                        diagonal["status"] = "PASS"
                class_diagonal[f"L{level}"] = diagonal
            munition_records[task] = {
                "three_class_accuracy": accuracy,
                "class1_recall": l1_recall,
                "class_diagonal_recall": class_diagonal,
            }
            if (
                accuracy["percent"]
                < minimum_munition_task_accuracy_percent
            ):
                failures.append(
                    f"{munition_name}/{task} accuracy "
                    f"{accuracy['percent']:.2f}% < "
                    f"{minimum_munition_task_accuracy_percent:.2f}%"
                )
        munition_task_metrics[munition_name] = munition_records

    small_mask = frame["munition_id"].to_numpy() == 0
    small_k0_mask = (
        small_mask & (frame["true_K"].to_numpy() == 0))
    small_k0_false_positive = _rate_record(
        int((
            frame.loc[small_k0_mask, "pred_K"].to_numpy() > 0
        ).sum()),
        int(small_k0_mask.sum()),
    )
    global_c0_mask = frame["true_C"].to_numpy() == 0
    global_c0_false_positive = _rate_record(
        int((
            frame.loc[global_c0_mask, "pred_C"].to_numpy() > 0
        ).sum()),
        int(global_c0_mask.sum()),
    )
    if (
        small_k0_false_positive["percent"]
        > maximum_small_k0_false_positive_percent
    ):
        failures.append(
            "Small/K L0 false-positive rate "
            f"{small_k0_false_positive['percent']:.3f}% > "
            f"{maximum_small_k0_false_positive_percent:.3f}%"
        )
    if (
        global_c0_false_positive["percent"]
        > maximum_global_c0_false_positive_percent
    ):
        failures.append(
            "global C L0 false-positive rate "
            f"{global_c0_false_positive['percent']:.3f}% > "
            f"{maximum_global_c0_false_positive_percent:.3f}%"
        )

    return {
        "schema": REPORT_SCHEMA,
        "status": "PASS" if not failures else "FAIL",
        "criteria": criteria,
        "global_confusion_diagonal_recall": global_class_recall,
        "munition_task_metrics": munition_task_metrics,
        "safety": {
            "small_k0_false_positive":
                small_k0_false_positive,
            "global_c0_false_positive":
                global_c0_false_positive,
        },
        "failure_count": len(failures),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the final held-out test predictions against the "
            "pre-registered strict surrogate-performance goal."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    frame, _, identity = _load_and_verify_run(run_dir)
    report = evaluate_strict_goal(frame)
    report["artifact_identity"] = identity
    output_path = (
        Path(args.output).resolve()
        if args.output
        else run_dir / "output" / "eval"
        / "strict_performance_goal.json"
    )
    _write_json_atomic(output_path, report)
    print(json.dumps({
        "status": report["status"],
        "failure_count": report["failure_count"],
        "small_k0_fp_percent": report["safety"][
            "small_k0_false_positive"]["percent"],
        "global_c0_fp_percent": report["safety"][
            "global_c0_false_positive"]["percent"],
        "output": str(output_path),
    }, indent=2, ensure_ascii=False))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
