from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from loitering_munition_damage_twin.experiments.validate_a23_promotion import (
    _load_report,
    _objective_values,
)


REPORT_SCHEMA = "stage0_nn_a34_validation_promotion_v1"


def evaluate_a34_promotion(
        baseline: dict,
        candidate: dict) -> dict:
    """Pre-registered, test-blind promotion rule for component supervision."""
    criteria = {
        "maximum_average_accuracy_drop_percentage_points": 0.10,
        "maximum_small_k0_false_positive_percent": 0.50,
        "maximum_global_c0_false_positive_percent": 2.50,
        "minimum_improved_weak_objectives": 3,
        "minimum_objective_improvement": 0.002,
        "maximum_objective_degradation": 0.003,
        "minimum_mean_objective_delta": 0.0005,
        "historical_gate_failure_count_may_increase": False,
        "minimum_goal_metric_failure_reduction": 1,
        "goal_evidence_failure_count_may_increase": False,
    }
    baseline_values = _objective_values(baseline)
    candidate_values = _objective_values(candidate)
    objective_deltas = {
        name: candidate_values[name] - baseline_values[name]
        for name in baseline_values
    }
    improved = [
        name for name, delta in objective_deltas.items()
        if delta >= criteria["minimum_objective_improvement"]
    ]
    degraded = [
        name for name, delta in objective_deltas.items()
        if delta < -criteria["maximum_objective_degradation"]
    ]
    mean_delta = sum(objective_deltas.values()) / len(objective_deltas)
    accuracy_delta = (
        float(candidate["average_3class_accuracy_percent"])
        - float(baseline["average_3class_accuracy_percent"])
    )
    baseline_historical_failures = int(
        baseline["performance_gate"]["failure_count"])
    candidate_historical_failures = int(
        candidate["performance_gate"]["failure_count"])
    baseline_goal = baseline.get("goal_performance_gate")
    candidate_goal = candidate.get("goal_performance_gate")
    if not isinstance(baseline_goal, dict):
        raise ValueError(
            "Baseline lacks the strict goal_performance_gate.")
    if not isinstance(candidate_goal, dict):
        raise ValueError(
            "Candidate lacks the strict goal_performance_gate.")
    baseline_goal_metric_failures = int(
        baseline_goal["metric_failure_count"])
    candidate_goal_metric_failures = int(
        candidate_goal["metric_failure_count"])
    goal_metric_failure_reduction = (
        baseline_goal_metric_failures
        - candidate_goal_metric_failures
    )
    baseline_evidence_failures = int(
        baseline_goal["evidence_failure_count"])
    candidate_evidence_failures = int(
        candidate_goal["evidence_failure_count"])
    small_k0_fp = float(
        candidate["small_k0_false_positive_percent"])
    global_c0_fp = float(
        candidate["global_c0_false_positive_percent"])

    failures = []
    if accuracy_delta < -criteria[
            "maximum_average_accuracy_drop_percentage_points"]:
        failures.append(
            "validation average accuracy degraded by more than 0.10pp")
    if small_k0_fp > criteria[
            "maximum_small_k0_false_positive_percent"]:
        failures.append("Small/K level-0 false-positive cap exceeded")
    if global_c0_fp > criteria[
            "maximum_global_c0_false_positive_percent"]:
        failures.append("global C level-0 false-positive cap exceeded")
    if candidate_historical_failures > baseline_historical_failures:
        failures.append(
            "historical validation gate failure count increased")
    if (
        goal_metric_failure_reduction
        < criteria["minimum_goal_metric_failure_reduction"]
    ):
        failures.append(
            "strict goal metric failure count did not decrease")
    if candidate_evidence_failures > baseline_evidence_failures:
        failures.append(
            "strict goal evidence failure count increased")
    if len(improved) < criteria["minimum_improved_weak_objectives"]:
        failures.append(
            "fewer than three weak-cell ranking objectives improved by 0.002")
    if degraded:
        failures.append(
            "at least one weak-cell ranking objective degraded by over 0.003")
    if mean_delta < criteria["minimum_mean_objective_delta"]:
        failures.append(
            "mean weak-cell ranking improvement is below 0.0005")

    return {
        "schema": REPORT_SCHEMA,
        "candidate": "A34_component_physics_auxiliary",
        "baseline": "A31_dual_target_with_terminal_physics",
        "status": "PASS" if not failures else "FAIL",
        "test_metrics_read": False,
        "criteria": criteria,
        "metrics": {
            "average_accuracy_delta_percentage_points": accuracy_delta,
            "baseline_historical_gate_failures":
                baseline_historical_failures,
            "candidate_historical_gate_failures":
                candidate_historical_failures,
            "baseline_goal_metric_failures":
                baseline_goal_metric_failures,
            "candidate_goal_metric_failures":
                candidate_goal_metric_failures,
            "goal_metric_failure_reduction":
                goal_metric_failure_reduction,
            "baseline_goal_evidence_failures":
                baseline_evidence_failures,
            "candidate_goal_evidence_failures":
                candidate_evidence_failures,
            "small_k0_false_positive_percent": small_k0_fp,
            "global_c0_false_positive_percent": global_c0_fp,
            "objective_deltas": objective_deltas,
            "mean_objective_delta": mean_delta,
            "improved_objectives": improved,
            "degraded_objectives": degraded,
        },
        "failures": failures,
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    os.replace(temporary_path, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-registered validation-only A34 promotion gate. "
            "The candidate test split must still be sealed."
        )
    )
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--candidate-run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    candidate_run_dir = Path(args.candidate_run_dir).resolve()
    forbidden_test_artifacts = (
        candidate_run_dir / "output" / "eval" / "test_metrics.json",
        candidate_run_dir / "output" / "eval" / "predictions.csv",
    )
    present = [
        str(path) for path in forbidden_test_artifacts
        if path.exists()
    ]
    if present:
        raise RuntimeError(
            "A34 test split is no longer sealed; promotion audit refused: "
            + ", ".join(present)
        )

    baseline = _load_report(args.baseline_report)
    candidate = _load_report(args.candidate_report)
    result = evaluate_a34_promotion(baseline, candidate)
    output_path = Path(args.output).resolve()
    _write_json_atomic(output_path, result)
    print(json.dumps({
        "status": result["status"],
        "goal_metric_failure_reduction": result["metrics"][
            "goal_metric_failure_reduction"],
        "improved_objectives": result["metrics"][
            "improved_objectives"],
        "mean_objective_delta": result["metrics"][
            "mean_objective_delta"],
        "failures": result["failures"],
        "output": str(output_path),
    }, indent=2, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
