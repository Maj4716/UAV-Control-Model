from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXPERIMENTS = [
    "A0_full",
    "A13_with_label_confidence",
    "A14_no_class_distribution_loss",
]
KEY_CELLS = [
    ("Small", "K"),
    ("Small", "M"),
    ("Small", "C"),
    ("Med-LM", "K"),
    ("Med-LM", "C"),
    ("Med-RD", "K"),
    ("Med-RD", "M"),
    ("Med-RD", "C"),
    ("Heavy", "K"),
    ("Heavy", "M"),
    ("Heavy", "C"),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_completed_result(run_dir: Path) -> tuple[Path | None, str | None]:
    metrics_path = run_dir / "output" / "eval" / "test_metrics.json"
    status_path = run_dir / "output" / "eval" / "evaluation_status.json"
    if not metrics_path.is_file():
        return None, f"missing metrics: {metrics_path}"
    if not status_path.is_file():
        return None, (
            f"missing completion marker: {status_path}; the evaluation may "
            "have stopped after writing partial metrics")
    try:
        with status_path.open("r", encoding="utf-8") as stream:
            status = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid completion marker {status_path}: {exc}"
    if status.get("status") != "COMPLETE":
        return None, (
            f"evaluation status is {status.get('status')!r}: "
            f"{status.get('error', status_path)}")
    expected_sha256 = status.get("metrics_sha256")
    if not isinstance(expected_sha256, str):
        return None, f"completion marker lacks metrics_sha256: {status_path}"
    observed_sha256 = _sha256(metrics_path)
    if observed_sha256 != expected_sha256:
        return None, (
            f"metrics hash mismatch: expected {expected_sha256}, "
            f"observed {observed_sha256}")
    return metrics_path, None


def _finite_mean(values: list[float | None]) -> float | None:
    finite = [
        float(value) for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return mean(finite) if finite else None


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    performance_gate = metrics.get("performance_gate", {})
    row: dict[str, float] = {
        "avg_accuracy": float(metrics["avg_acc"]),
        "small_k0_fp": float(metrics["small_k1"]["k0_fp"]),
        "global_c0_fp": float(metrics["small_c1"]["c0_fp"]),
        "small_c0_fp": float(metrics["small_c1"]["small_c0_fp"]),
        "performance_gate_passed": float(bool(
            performance_gate.get("passed", False))),
        "performance_gate_failure_count": float(len(
            performance_gate.get("failures", []))),
    }
    for task, value in metrics["overall_acc"].items():
        row[f"accuracy_{task}"] = float(value)

    cell_metrics = metrics.get("cell_metrics")
    if not isinstance(cell_metrics, dict):
        raise RuntimeError(
            "test_metrics.json lacks cell_metrics. Retrain and evaluate with "
            "the current neural pipeline before comparing ablations.")
    for munition, task in KEY_CELLS:
        cell = cell_metrics[munition][task]
        prefix = f"{munition}_{task}".replace("-", "_")
        row[f"{prefix}_class1_recall"] = float(cell["class1_recall"])
        row[f"{prefix}_class1_f1"] = float(cell["class1_f1"])
        row[f"{prefix}_accuracy"] = float(cell["three_class_accuracy"])

    probability_metrics = metrics.get("probability_metrics", {})
    for metric_name in (
            "brier_mc_mean", "cross_entropy_mc_mean",
            "ece_mc_mean_10bin", "average_precision"):
        aggregate = _finite_mean([
            head.get(metric_name)
            for head in probability_metrics.values()
            if isinstance(head, dict)
        ])
        if aggregate is not None:
            row[f"mean_{metric_name}"] = float(aggregate)
    return row


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_experiment: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_experiment.setdefault(row["experiment_id"], []).append(row)
    output = {}
    for experiment_id, experiment_rows in by_experiment.items():
        metric_names = sorted({
            key
            for row in experiment_rows
            for key, value in row.items()
            if key not in {"experiment_id", "seed"}
            and isinstance(value, (int, float))
        })
        output[experiment_id] = {
            "runs": len(experiment_rows),
            "seeds": sorted(int(row["seed"]) for row in experiment_rows),
            "metrics": {
                metric: {
                    "mean": mean(float(row[metric]) for row in experiment_rows),
                    "std": (
                        pstdev(float(row[metric]) for row in experiment_rows)
                        if len(experiment_rows) > 1 else 0.0
                    ),
                }
                for metric in metric_names
                if all(metric in row for row in experiment_rows)
            },
        }
    return output


def _delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    left_metrics = left["metrics"]
    right_metrics = right["metrics"]
    shared = sorted(set(left_metrics) & set(right_metrics))
    return {
        metric: float(left_metrics[metric]["mean"])
        - float(right_metrics[metric]["mean"])
        for metric in shared
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the improved neural baseline with the confidence and "
            "class-distribution-loss ablations."
        )
    )
    parser.add_argument(
        "--experiments", nargs="+", default=DEFAULT_EXPERIMENTS)
    parser.add_argument(
        "--reference", default="A0_full",
        help="Experiment used as the left side of generic delta tables.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--result-root", default="output/experiments")
    parser.add_argument(
        "--output",
        default="output/experiments/performance_ablation_summary.json")
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help="Write a partial summary and exit successfully even when requested "
             "runs are missing or incomplete.")
    args = parser.parse_args()

    result_root = Path(args.result_root)
    if not result_root.is_absolute():
        result_root = REPO_ROOT / result_root
    rows: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    for experiment_id in args.experiments:
        for seed in args.seeds:
            run_dir = result_root / experiment_id / f"seed{seed}"
            metrics_path, reason = _validate_completed_result(run_dir)
            if metrics_path is None:
                incomplete.append({
                    "experiment_id": experiment_id,
                    "seed": int(seed),
                    "reason": str(reason),
                })
                continue
            with metrics_path.open("r", encoding="utf-8") as stream:
                metrics = json.load(stream)
            row: dict[str, Any] = {
                "experiment_id": experiment_id,
                "seed": int(seed),
            }
            row.update(_flatten_metrics(metrics))
            rows.append(row)

    if incomplete:
        print("[COMPARE] Incomplete requested runs:")
        for item in incomplete:
            print(
                f"  - {item['experiment_id']}/seed{item['seed']}: "
                f"{item['reason']}")
    if not rows:
        print("[COMPARE] No completed current ablation evaluations were found.")

    aggregates = _aggregate(rows)
    comparisons = {}
    if args.reference in aggregates:
        for experiment_id in sorted(aggregates):
            if experiment_id == args.reference:
                continue
            comparisons[
                f"{args.reference}_minus_{experiment_id}"
            ] = _delta(
                aggregates[args.reference],
                aggregates[experiment_id],
            )
    if (
        "A0_full" in aggregates
        and "A13_with_label_confidence" in aggregates
    ):
        comparisons["A0_minus_A13_confidence"] = _delta(
            aggregates["A0_full"],
            aggregates["A13_with_label_confidence"],
        )
    if (
        "A0_full" in aggregates
        and "A14_no_class_distribution_loss" in aggregates
    ):
        comparisons["A0_minus_A14_no_class_distribution"] = _delta(
            aggregates["A0_full"],
            aggregates["A14_no_class_distribution_loss"],
        )

    payload = {
        "schema": "stage0_nn_performance_ablation_v1",
        "status": "COMPLETE" if not incomplete else "INCOMPLETE",
        "reference": args.reference,
        "experiments_requested": list(args.experiments),
        "seeds_requested": [int(seed) for seed in args.seeds],
        "incomplete_results": incomplete,
        "runs": rows,
        "aggregates": aggregates,
        "comparisons": comparisons,
        "interpretation": {
            "higher_is_better": [
                "accuracy_*", "*_class1_recall", "*_class1_f1",
                "*_accuracy", "mean_average_precision",
                "performance_gate_passed",
            ],
            "lower_is_better": [
                "*_fp", "mean_brier_mc_mean",
                "mean_cross_entropy_mc_mean", "mean_ece_mc_mean_10bin",
                "performance_gate_failure_count",
            ],
        },
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)

    csv_path = output_path.with_suffix(".csv")
    fieldnames = sorted({
        key for row in rows for key in row
        if key not in {"experiment_id", "seed"}
    })
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["experiment_id", "seed", *fieldnames])
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({
        "status": payload["status"],
        "runs": len(rows),
        "incomplete": len(incomplete),
        "json": str(output_path),
        "csv": str(csv_path),
    }, indent=2, ensure_ascii=False))
    if incomplete and not args.allow_incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
