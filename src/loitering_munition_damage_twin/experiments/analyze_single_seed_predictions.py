from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from loitering_munition_damage_twin.experiments.compare_performance_ablations import (
    _validate_completed_result,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
TASKS = ("K", "M", "F", "C")
MUNITIONS = ("Small", "Med-LM", "Med-RD", "Heavy")
DEFAULT_EXPERIMENTS = (
    "A0_full",
    "A13_with_label_confidence",
    "A14_no_class_distribution_loss",
)


def _safe_auc(target: np.ndarray, score: np.ndarray) -> float | None:
    if np.unique(target).size < 2:
        return None
    return float(roc_auc_score(target, score))


def _safe_partial_auc(
        target: np.ndarray,
        score: np.ndarray,
        maximum_false_positive_rate: float) -> float | None:
    if np.unique(target).size < 2:
        return None
    return float(roc_auc_score(
        target,
        score,
        max_fpr=maximum_false_positive_rate,
    ))


def _safe_average_precision(
        target: np.ndarray, score: np.ndarray) -> float | None:
    if int(np.sum(target == 1)) == 0:
        return None
    return float(average_precision_score(target, score))


def _max_entry_recall_under_l0_fp(
        level: np.ndarray, probability_ge1: np.ndarray,
        maximum_fp_rate: float) -> dict[str, float | int | None]:
    negative = level == 0
    positive = level >= 1
    if not np.any(negative) or not np.any(positive):
        return {
            "maximum_l0_false_positive_rate": maximum_fp_rate,
            "threshold": None,
            "observed_l0_false_positive_rate": None,
            "damage_entry_recall": None,
        }
    candidates = np.unique(np.concatenate((
        probability_ge1,
        np.asarray([0.0, 1.0], dtype=np.float64),
    )))
    best: tuple[float, float, float] | None = None
    for threshold in candidates:
        predicted = probability_ge1 >= threshold
        fp_rate = float(np.mean(predicted[negative]))
        if fp_rate > maximum_fp_rate + 1e-12:
            continue
        recall = float(np.mean(predicted[positive]))
        candidate = (recall, -fp_rate, float(threshold))
        if best is None or candidate > best:
            best = candidate
    if best is None:
        return {
            "maximum_l0_false_positive_rate": maximum_fp_rate,
            "threshold": None,
            "observed_l0_false_positive_rate": None,
            "damage_entry_recall": None,
        }
    return {
        "maximum_l0_false_positive_rate": maximum_fp_rate,
        "threshold": float(best[2]),
        "observed_l0_false_positive_rate": float(-best[1]),
        "damage_entry_recall": float(best[0]),
    }


def _cell_metrics(frame: pd.DataFrame, task: str,
                  munition_id: int) -> dict[str, Any]:
    cell = frame.loc[frame["munition_id"] == munition_id]
    level = cell[f"true_{task}"].to_numpy(dtype=np.int64)
    prediction = cell[f"pred_{task}"].to_numpy(dtype=np.int64)
    probability_ge1 = cell[f"prob_{task}1"].to_numpy(dtype=np.float64)
    probability_ge2 = cell[f"prob_{task}2"].to_numpy(dtype=np.float64)
    class1_score = np.clip(
        probability_ge1 - probability_ge2, 0.0, 1.0)
    true_class1 = level == 1
    predicted_class1 = prediction == 1
    class1_support = int(np.sum(true_class1))
    class1_to_level0 = int(np.sum(true_class1 & (prediction == 0)))
    class1_to_level2 = int(np.sum(true_class1 & (prediction == 2)))
    tp = int(np.sum(true_class1 & predicted_class1))
    fp = int(np.sum((~true_class1) & predicted_class1))
    fn = int(np.sum(true_class1 & (~predicted_class1)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    l0_cap = 0.005 if task == "K" and munition_id == 0 else 0.025
    return {
        "samples": int(len(cell)),
        "class1_support": class1_support,
        "class1_precision": float(precision),
        "class1_recall": float(recall),
        "class1_f1": float(f1),
        "class1_error_direction": {
            "predicted_as_level0_count": class1_to_level0,
            "predicted_as_level0_rate": (
                class1_to_level0 / max(class1_support, 1)),
            "predicted_as_level2_count": class1_to_level2,
            "predicted_as_level2_rate": (
                class1_to_level2 / max(class1_support, 1)),
        },
        "three_class_accuracy": float(np.mean(prediction == level)),
        "class1_mass_average_precision": _safe_average_precision(
            true_class1.astype(np.int64), class1_score),
        "l0_vs_damage_auc": _safe_auc(
            (level >= 1).astype(np.int64), probability_ge1),
        "l0_vs_damage_standardized_partial_auc": _safe_partial_auc(
            (level >= 1).astype(np.int64),
            probability_ge1,
            l0_cap,
        ),
        "partial_auc_maximum_false_positive_rate": l0_cap,
        "le1_vs_l2_auc": _safe_auc(
            (level >= 2).astype(np.int64), probability_ge2),
        "entry_recall_feasibility": _max_entry_recall_under_l0_fp(
            level, probability_ge1, l0_cap),
    }


def _cluster_bootstrap_mean_delta(
        difference: np.ndarray,
        roots: np.ndarray,
        repetitions: int,
        rng: np.random.Generator) -> dict[str, float | int]:
    root_codes, unique_roots = pd.factorize(roots, sort=True)
    root_count = len(unique_roots)
    root_sums = np.bincount(
        root_codes, weights=difference.astype(np.float64),
        minlength=root_count)
    root_rows = np.bincount(root_codes, minlength=root_count)
    estimates = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        sampled = rng.integers(0, root_count, size=root_count)
        estimates[repetition] = (
            np.sum(root_sums[sampled])
            / max(float(np.sum(root_rows[sampled])), 1.0)
        )
    estimate = float(np.mean(difference))
    return {
        "estimate_percentage_points": estimate * 100.0,
        "ci95_low_percentage_points": (
            float(np.quantile(estimates, 0.025)) * 100.0),
        "ci95_high_percentage_points": (
            float(np.quantile(estimates, 0.975)) * 100.0),
        "bootstrap_repetitions": int(repetitions),
        "root_clusters": int(root_count),
    }


def _paired_differences(
        reference: pd.DataFrame,
        candidate: pd.DataFrame,
        repetitions: int,
        seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    reference_correct = np.column_stack([
        reference[f"pred_{task}"].to_numpy()
        == reference[f"true_{task}"].to_numpy()
        for task in TASKS
    ]).mean(axis=1)
    candidate_correct = np.column_stack([
        candidate[f"pred_{task}"].to_numpy()
        == candidate[f"true_{task}"].to_numpy()
        for task in TASKS
    ]).mean(axis=1)
    roots = reference["root_seed_id"].astype(str).to_numpy()
    output: dict[str, Any] = {
        "average_accuracy": _cluster_bootstrap_mean_delta(
            candidate_correct - reference_correct,
            roots,
            repetitions,
            rng,
        ),
        "class1_recall_by_cell": {},
    }
    for munition_id, munition in enumerate(MUNITIONS):
        output["class1_recall_by_cell"][munition] = {}
        for task in TASKS:
            mask = (
                (reference["munition_id"].to_numpy() == munition_id)
                & (reference[f"true_{task}"].to_numpy() == 1)
            )
            reference_hit = (
                reference.loc[mask, f"pred_{task}"].to_numpy() == 1)
            candidate_hit = (
                candidate.loc[mask, f"pred_{task}"].to_numpy() == 1)
            output["class1_recall_by_cell"][munition][task] = (
                _cluster_bootstrap_mean_delta(
                    candidate_hit.astype(np.float64)
                    - reference_hit.astype(np.float64),
                    roots[mask],
                    repetitions,
                    rng,
                )
            )
    return output


def _load_predictions(result_root: Path, experiment: str,
                      seed: int) -> pd.DataFrame:
    run_dir = result_root / experiment / f"seed{seed}"
    metrics_path, reason = _validate_completed_result(run_dir)
    if metrics_path is None:
        raise RuntimeError(
            f"{experiment}/seed{seed} is incomplete: {reason}")
    predictions_path = (
        run_dir / "output" / "eval" / "predictions.csv")
    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"Missing predictions for {experiment}/seed{seed}: "
            f"{predictions_path}")
    return pd.read_csv(predictions_path)


def _assert_aligned(reference: pd.DataFrame,
                    candidate: pd.DataFrame,
                    experiment: str) -> None:
    columns = [
        "sample_id", "root_seed_id", "munition_id",
        *(f"true_{task}" for task in TASKS),
    ]
    if len(reference) != len(candidate):
        raise RuntimeError(
            f"{experiment} has {len(candidate)} predictions; "
            f"reference has {len(reference)}.")
    for column in columns:
        if not np.array_equal(
                reference[column].to_numpy(),
                candidate[column].to_numpy()):
            raise RuntimeError(
                f"Prediction alignment mismatch in {column} for "
                f"{experiment}. Paired comparison is invalid.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze aligned single-seed predictions with cell diagnostics "
            "and paired root-cluster bootstrap confidence intervals."))
    parser.add_argument(
        "--experiments", nargs="+", default=list(DEFAULT_EXPERIMENTS))
    parser.add_argument("--reference", default="A0_full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--result-root", default="output/experiments")
    parser.add_argument(
        "--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument(
        "--output",
        default="output/experiments/single_seed_prediction_analysis.json")
    args = parser.parse_args()
    if args.bootstrap_repetitions < 100:
        raise ValueError("bootstrap-repetitions must be at least 100.")
    if args.reference not in args.experiments:
        raise ValueError("reference must be included in experiments.")

    result_root = Path(args.result_root)
    if not result_root.is_absolute():
        result_root = REPO_ROOT / result_root
    frames = {
        experiment: _load_predictions(
            result_root, experiment, args.seed)
        for experiment in args.experiments
    }
    reference = frames[args.reference]
    for experiment, frame in frames.items():
        _assert_aligned(reference, frame, experiment)

    cell_diagnostics = {}
    for experiment, frame in frames.items():
        cell_diagnostics[experiment] = {
            munition: {
                task: _cell_metrics(frame, task, munition_id)
                for task in TASKS
            }
            for munition_id, munition in enumerate(MUNITIONS)
        }
    paired = {
        experiment: _paired_differences(
            reference,
            frame,
            args.bootstrap_repetitions,
            seed=20260726 + index,
        )
        for index, (experiment, frame) in enumerate(frames.items())
        if experiment != args.reference
    }
    payload = {
        "schema": "stage0_nn_single_seed_analysis_v1",
        "status": "COMPLETE",
        "seed": int(args.seed),
        "reference": args.reference,
        "experiments": list(args.experiments),
        "aligned_samples": int(len(reference)),
        "aligned_root_families": int(
            reference["root_seed_id"].nunique()),
        "cell_diagnostics": cell_diagnostics,
        "paired_candidate_minus_reference": paired,
        "interpretation": (
            "Bootstrap intervals quantify test-sample/root uncertainty for "
            "this seed only; they do not replace multi-seed training.")
    }
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    temporary_path.replace(output_path)
    print(json.dumps({
        "status": payload["status"],
        "seed": payload["seed"],
        "experiments": len(frames),
        "aligned_samples": payload["aligned_samples"],
        "aligned_root_families": payload["aligned_root_families"],
        "output": str(output_path),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
