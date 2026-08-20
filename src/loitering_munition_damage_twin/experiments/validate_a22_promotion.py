from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
TARGETED_ENTRY_CELLS = ("Small/C", "Med-LM/C", "Med-RD/C")


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def _diagnostics(report: dict) -> dict:
    diagnostics = report.get("targeted_probability_diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError(
            "validation report lacks targeted_probability_diagnostics")
    return diagnostics


def _mean_c_partial_auc(report: dict) -> float:
    diagnostics = _diagnostics(report)
    values = [
        diagnostics[cell]["entry_standardized_partial_auc"]
        for cell in TARGETED_ENTRY_CELLS
    ]
    if any(value is None for value in values):
        raise ValueError("C partial-AUC is undefined in a required cell")
    return sum(float(value) for value in values) / len(values)


def _evaluate_promotion(
        baseline: dict,
        candidate: dict) -> dict[str, Any]:
    """Apply the pre-registered validation-only A22 promotion contract."""
    baseline_diag = _diagnostics(baseline)
    candidate_diag = _diagnostics(candidate)
    metrics = {
        "average_accuracy_delta_percentage_points": (
            float(candidate["average_3class_accuracy_percent"])
            - float(baseline["average_3class_accuracy_percent"])
        ),
        "small_k_partial_auc_delta": (
            float(candidate_diag["Small/K"][
                "entry_standardized_partial_auc"])
            - float(baseline_diag["Small/K"][
                "entry_standardized_partial_auc"])
        ),
        "mean_c_partial_auc_delta": (
            _mean_c_partial_auc(candidate)
            - _mean_c_partial_auc(baseline)
        ),
        "med_rd_m_conditional_auc_delta": (
            float(candidate_diag["Med-RD/M_L1_vs_L2"][
                "conditional_auc"])
            - float(baseline_diag["Med-RD/M_L1_vs_L2"][
                "conditional_auc"])
        ),
        "baseline_gate_failures": int(
            baseline["performance_gate"]["failure_count"]),
        "candidate_gate_failures": int(
            candidate["performance_gate"]["failure_count"]),
        "small_k0_false_positive_percent": float(
            candidate["small_k0_false_positive_percent"]),
        "global_c0_false_positive_percent": float(
            candidate["global_c0_false_positive_percent"]),
    }
    failures = []
    if metrics["average_accuracy_delta_percentage_points"] < -0.10:
        failures.append(
            "validation average accuracy degraded by more than 0.10pp")
    if metrics["small_k0_false_positive_percent"] > 0.5:
        failures.append("Small/K0 false-positive constraint failed")
    if metrics["global_c0_false_positive_percent"] > 2.5:
        failures.append("global C0 false-positive constraint failed")
    if (
        metrics["candidate_gate_failures"]
        > metrics["baseline_gate_failures"]
    ):
        failures.append("strict validation gate failure count increased")

    ranking_deltas = (
        metrics["small_k_partial_auc_delta"],
        metrics["mean_c_partial_auc_delta"],
        metrics["med_rd_m_conditional_auc_delta"],
    )
    improved_objectives = sum(
        delta >= 0.002 for delta in ranking_deltas)
    if improved_objectives < 2:
        failures.append(
            "fewer than two targeted ranking objectives improved by 0.002")
    if min(ranking_deltas) < -0.003:
        failures.append(
            "at least one targeted ranking objective degraded by over 0.003")

    return {
        "schema": "stage0_nn_a22_validation_promotion_v1",
        "status": "PASS" if not failures else "FAIL",
        "test_metrics_read": False,
        "criteria": {
            "maximum_average_accuracy_drop_percentage_points": 0.10,
            "maximum_small_k0_false_positive_percent": 0.5,
            "maximum_global_c0_false_positive_percent": 2.5,
            "minimum_improved_ranking_objectives": 2,
            "minimum_objective_improvement": 0.002,
            "maximum_objective_degradation": 0.003,
            "gate_failure_count_may_increase": False,
        },
        "metrics": metrics,
        "improved_ranking_objectives": improved_objectives,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Decide whether A22 may advance from validation to test. "
            "This command does not read test artifacts."))
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument(
        "--output",
        default=(
            "output/experiments/A22_targeted_ranking/seed42/"
            "validation_promotion.json"))
    args = parser.parse_args()
    with open(args.baseline_report, "r", encoding="utf-8") as stream:
        baseline_payload = json.load(stream)
    baseline = baseline_payload.get(
        "selection_validation_report", baseline_payload)
    with open(args.candidate_report, "r", encoding="utf-8") as stream:
        candidate_payload = json.load(stream)
    candidate = candidate_payload.get(
        "selection_validation_report", candidate_payload)
    decision = _evaluate_promotion(baseline, candidate)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    _write_json_atomic(output_path, decision)
    print(json.dumps({
        "status": decision["status"],
        "improved_ranking_objectives": (
            decision["improved_ranking_objectives"]),
        "failures": decision["failures"],
        "output": str(output_path),
    }, indent=2, ensure_ascii=False))
    if decision["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
