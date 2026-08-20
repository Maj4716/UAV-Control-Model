from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

REPO_ROOT = Path(__file__).resolve().parents[3]

from loitering_munition_damage_twin.surrogate.artifacts import sha256_file


def _summary_stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty vector.")
    count = int(array.size)
    mean = float(array.mean())
    sample_std = (
        float(array.std(ddof=1)) if count > 1 else 0.0)
    if count > 1:
        critical = float(student_t.ppf(0.975, count - 1))
        half_width = critical * sample_std / math.sqrt(count)
    else:
        critical = None
        half_width = 0.0
    return {
        "n": count,
        "mean": mean,
        "sample_standard_deviation": sample_std,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean_ci95_low": mean - half_width,
        "mean_ci95_high": mean + half_width,
        "student_t_critical": critical,
        "interpretation": (
            "CI reflects between-seed initialization variability; "
            "it is not a test-population sampling interval."
        ),
    }


def _flatten_key_metrics(metrics: dict) -> dict[str, float]:
    flattened = {
        "average_3class_accuracy_percent": float(metrics["avg_acc"]),
        "small_k0_false_positive_percent": float(
            metrics["small_k1"]["k0_fp"]),
        "global_c0_false_positive_percent": float(
            metrics["small_c1"]["c0_fp"]),
        "small_c0_false_positive_percent": float(
            metrics["small_c1"]["small_c0_fp"]),
        "performance_gate_failure_count": float(len(
            metrics["performance_gate"]["failures"])),
        "mean_brier_mc_mean": float(np.mean([
            value["brier_mc_mean"]
            for value in metrics["probability_metrics"].values()
        ])),
        "mean_ece_mc_mean_10bin": float(np.mean([
            value["ece_mc_mean_10bin"]
            for value in metrics["probability_metrics"].values()
        ])),
        "mean_average_precision": float(np.mean([
            value["average_precision"]
            for value in metrics["probability_metrics"].values()
        ])),
    }
    for munition, task in (
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
    ):
        key = f"{munition}/{task}_class1_recall_percent"
        flattened[key] = float(
            metrics["cell_metrics"][munition][task][
                "class1_recall"])
    return flattened


def summarize(
        result_root: Path,
        experiment_id: str,
        seeds: list[int]) -> dict:
    runs = []
    dataset_hashes = set()
    threshold_schemas = set()
    scaler_hashes = set()
    model_hashes = set()
    for seed in seeds:
        run_dir = result_root / experiment_id / f"seed{seed}"
        status_path = run_dir / "run_status.json"
        evaluation_status_path = (
            run_dir / "output" / "eval" / "evaluation_status.json")
        metrics_path = (
            run_dir / "output" / "eval" / "test_metrics.json")
        manifest_path = (
            run_dir / "output" / "models" / "model_manifest.json")
        for path in (
                status_path, evaluation_status_path,
                metrics_path, manifest_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing required seed artifact: {path}")
        with status_path.open("r", encoding="utf-8") as stream:
            run_status = json.load(stream)
        with evaluation_status_path.open(
                "r", encoding="utf-8") as stream:
            evaluation_status = json.load(stream)
        with metrics_path.open("r", encoding="utf-8") as stream:
            metrics = json.load(stream)
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        if (
            run_status.get("status") != "COMPLETE"
            or evaluation_status.get("status") != "COMPLETE"
        ):
            raise RuntimeError(
                f"Seed {seed} is not a completed evaluation.")
        observed_metrics_hash = sha256_file(str(metrics_path))
        if observed_metrics_hash != evaluation_status.get(
                "metrics_sha256"):
            raise RuntimeError(
                f"Seed {seed} metrics hash mismatch.")
        sealed_model_hash = (
            manifest["artifacts"]["best_model.pth"]["sha256"])
        observed_model_hash = sha256_file(str(
            run_dir / "output" / "models" / "best_model.pth"))
        if sealed_model_hash != observed_model_hash:
            raise RuntimeError(
                f"Seed {seed} model hash mismatch.")
        dataset_hashes.add(str(metrics["dataset_sha256"]))
        threshold_schemas.add(str(metrics["threshold_schema"]))
        scaler_hashes.add(str(
            manifest["artifacts"]["minmax_scaler.pkl"]["sha256"]))
        model_hashes.add(observed_model_hash)
        runs.append({
            "seed": int(seed),
            "performance_gate_passed": bool(
                metrics["performance_gate"]["passed"]),
            "small_k0_safety_passed": (
                float(metrics["small_k1"]["k0_fp"]) <= 0.5),
            "global_c0_safety_passed": (
                float(metrics["small_c1"]["c0_fp"]) <= 2.5),
            "metrics_sha256": observed_metrics_hash,
            "model_sha256": observed_model_hash,
            "metrics": _flatten_key_metrics(metrics),
        })
    if len(dataset_hashes) != 1:
        raise RuntimeError("Seeds do not use one dataset SHA-256.")
    if threshold_schemas != {"v8_exact_l1_floor_constrained"}:
        raise RuntimeError(
            "Seeds do not share the current v8 threshold schema.")
    if len(scaler_hashes) != 1:
        raise RuntimeError("Seeds do not use one scaler SHA-256.")
    if len(model_hashes) != len(seeds):
        raise RuntimeError(
            "Random seeds did not produce distinct model artifacts.")

    metric_names = tuple(runs[0]["metrics"])
    statistics = {
        name: _summary_stats([
            run["metrics"][name] for run in runs])
        for name in metric_names
    }
    gate_passes = sum(
        run["performance_gate_passed"] for run in runs)
    small_k_safety_passes = sum(
        run["small_k0_safety_passed"] for run in runs)
    global_c_safety_passes = sum(
        run["global_c0_safety_passed"] for run in runs)
    return {
        "schema": "stage0_nn_multiseed_reliability_v1",
        "status": "COMPLETE",
        "experiment_id": experiment_id,
        "seeds": [int(seed) for seed in seeds],
        "artifact_contract": {
            "dataset_sha256": next(iter(dataset_hashes)),
            "threshold_schema": next(iter(threshold_schemas)),
            "scaler_sha256": next(iter(scaler_hashes)),
            "distinct_model_sha256_count": len(model_hashes),
            "all_evaluation_hashes_verified": True,
        },
        "acceptance": {
            "strict_performance_gate_pass_count": gate_passes,
            "strict_performance_gate_total": len(runs),
            "strict_multiseed_validation_passed": (
                gate_passes == len(runs)),
            "small_k0_safety_pass_count": small_k_safety_passes,
            "global_c0_safety_pass_count": global_c_safety_passes,
            "safety_caps_stable_across_all_seeds": (
                small_k_safety_passes == len(runs)
                and global_c_safety_passes == len(runs)),
        },
        "statistics": statistics,
        "runs": runs,
        "conclusion": (
            "REJECT_STRICT_DEPLOYMENT"
            if gate_passes < len(runs)
            else "PASS_STRICT_DEPLOYMENT"
        ),
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    os.replace(temporary_path, path)


def _write_csv(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "metric", "n", "mean", "sample_standard_deviation",
                "minimum", "maximum", "mean_ci95_low",
                "mean_ci95_high"),
        )
        writer.writeheader()
        for metric, values in payload["statistics"].items():
            writer.writerow({
                "metric": metric,
                **{
                    key: values[key]
                    for key in writer.fieldnames
                    if key != "metric"
                },
            })


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and summarize a completed multi-seed neural evaluation."
        )
    )
    parser.add_argument(
        "--result-root",
        default=str(REPO_ROOT / "abli_exp" / "results"))
    parser.add_argument(
        "--experiment",
        default="A19_bounded_class1_floor_calibration")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument(
        "--output",
        default=str(
            REPO_ROOT / "abli_exp" / "results"
            / "a19_multiseed_reliability.json"))
    args = parser.parse_args()
    result = summarize(
        Path(args.result_root),
        str(args.experiment),
        list(args.seeds),
    )
    output_path = Path(args.output).resolve()
    _write_json_atomic(output_path, result)
    csv_path = output_path.with_suffix(".csv")
    _write_csv(csv_path, result)
    accuracy = result["statistics"][
        "average_3class_accuracy_percent"]
    print(json.dumps({
        "status": result["status"],
        "conclusion": result["conclusion"],
        "average_accuracy_mean_percent": accuracy["mean"],
        "average_accuracy_sample_std_percent": accuracy[
            "sample_standard_deviation"],
        "average_accuracy_mean_ci95": [
            accuracy["mean_ci95_low"],
            accuracy["mean_ci95_high"],
        ],
        "strict_gate_pass_count": result["acceptance"][
            "strict_performance_gate_pass_count"],
        "safety_caps_stable": result["acceptance"][
            "safety_caps_stable_across_all_seeds"],
        "json": str(output_path),
        "csv": str(csv_path),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
