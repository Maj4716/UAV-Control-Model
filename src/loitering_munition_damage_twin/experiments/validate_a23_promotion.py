from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


WEAK_ENTRY_CELLS = (
    "Small/K",
    "Med-LM/K",
    "Med-RD/K",
    "Heavy/K",
    "Small/C",
    "Med-LM/C",
    "Med-RD/C",
)
CONDITIONAL_CELL = "Med-RD/M_L1_vs_L2"


def _selection_report(payload: dict) -> dict:
    nested = payload.get("selection_validation_report")
    return nested if isinstance(nested, dict) else payload


def _load_report(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as stream:
        report = _selection_report(json.load(stream))
    if report.get("schema") not in {
        "stage0_nn_validation_selection_v1",
        "stage0_nn_validation_selection_v2",
    }:
        raise ValueError(
            f"{path} does not contain a current validation report.")
    if (
        report.get("split") != "validation"
        or report.get("test_labels_used") is not False
    ):
        raise ValueError(
            f"{path} is not an explicitly test-blind validation report.")
    return report


def _objective_values(report: dict) -> dict[str, float]:
    diagnostics = report.get("targeted_probability_diagnostics", {})
    values = {}
    for cell in WEAK_ENTRY_CELLS:
        value = diagnostics.get(cell, {}).get(
            "entry_standardized_partial_auc")
        if value is None:
            raise ValueError(f"Missing validation partial AUC for {cell}.")
        values[cell] = float(value)
    conditional_value = diagnostics.get(
        CONDITIONAL_CELL, {}).get("conditional_auc")
    if conditional_value is None:
        raise ValueError(
            f"Missing validation conditional AUC for {CONDITIONAL_CELL}.")
    values[CONDITIONAL_CELL] = float(conditional_value)
    return values


def _evaluate_promotion(
        baseline: dict,
        candidate: dict,
        candidate_id: str = "A23_frozen_cell_residual_adapters") -> dict:
    criteria = {
        "maximum_average_accuracy_drop_percentage_points": 0.10,
        "maximum_small_k0_false_positive_percent": 0.50,
        "maximum_global_c0_false_positive_percent": 2.50,
        "minimum_improved_weak_objectives": 3,
        "minimum_objective_improvement": 0.002,
        "maximum_objective_degradation": 0.003,
        "minimum_mean_objective_delta": 0.0005,
        "gate_failure_count_may_increase": False,
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
    baseline_failures = int(
        baseline["performance_gate"]["failure_count"])
    candidate_failures = int(
        candidate["performance_gate"]["failure_count"])
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
    if candidate_failures > baseline_failures:
        failures.append(
            "strict validation gate failure count increased")
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
        "schema": "stage0_nn_validation_promotion_v2",
        "candidate": str(candidate_id),
        "status": "PASS" if not failures else "FAIL",
        "test_metrics_read": False,
        "criteria": criteria,
        "metrics": {
            "average_accuracy_delta_percentage_points": accuracy_delta,
            "baseline_gate_failures": baseline_failures,
            "candidate_gate_failures": candidate_failures,
            "small_k0_false_positive_percent": small_k0_fp,
            "global_c0_false_positive_percent": global_c0_fp,
            "objective_deltas": objective_deltas,
            "mean_objective_delta": mean_delta,
            "improved_objectives": improved,
            "degraded_objectives": degraded,
        },
        "failures": failures,
    }


def _write_json_atomic(path: str, payload: dict) -> None:
    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    os.replace(temporary_path, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-registered, validation-only A23 promotion gate. "
            "No test metrics are read."
        )
    )
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument(
        "--candidate-id",
        default="A23_frozen_cell_residual_adapters")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = _load_report(args.baseline_report)
    candidate = _load_report(args.candidate_report)
    result = _evaluate_promotion(
        baseline, candidate, candidate_id=args.candidate_id)
    _write_json_atomic(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "improved_objectives": result["metrics"][
            "improved_objectives"],
        "mean_objective_delta": result["metrics"][
            "mean_objective_delta"],
        "failures": result["failures"],
        "output": str(Path(args.output).resolve()),
    }, indent=2, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
