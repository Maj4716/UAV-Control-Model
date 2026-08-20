from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from loitering_munition_damage_twin.experiments.validate_a23_promotion import (
    _load_report,
)
from loitering_munition_damage_twin.experiments.validate_a34_promotion import (
    evaluate_a34_promotion,
)


REPORT_SCHEMA = "stage0_nn_a37_validation_promotion_v1"
CANDIDATE_ID = "A37_component_tree_teacher_residual"
BASELINE_ID = "A35_independent_component_tree_fusion"


def evaluate_a37_promotion(
        baseline: dict, candidate: dict) -> dict:
    """Apply the frozen broad-gain rule to the A37 validation candidate."""
    result = evaluate_a34_promotion(baseline, candidate)
    result["schema"] = REPORT_SCHEMA
    result["candidate"] = CANDIDATE_ID
    result["baseline"] = BASELINE_ID
    return result


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    os.replace(temporary_path, path)


def _assert_test_split_is_sealed(candidate_run_dir: Path) -> None:
    forbidden_test_artifacts = (
        candidate_run_dir / "output" / "eval" / "test_metrics.json",
        candidate_run_dir / "output" / "eval" / "predictions.csv",
    )
    present = [
        str(path) for path in forbidden_test_artifacts if path.exists()
    ]
    if present:
        raise RuntimeError(
            "A37 test split is no longer sealed; promotion audit refused: "
            + ", ".join(present)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-registered, test-blind A37 validation promotion gate. "
            "The target-tree candidate is compared with the last promoted "
            "A35 baseline using the already frozen broad-gain criteria."
        )
    )
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--candidate-run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    candidate_run_dir = Path(args.candidate_run_dir).resolve()
    _assert_test_split_is_sealed(candidate_run_dir)
    result = evaluate_a37_promotion(
        _load_report(args.baseline_report),
        _load_report(args.candidate_report),
    )
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
