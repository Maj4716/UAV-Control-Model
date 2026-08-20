from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DEFAULT_CANDIDATES = (
    "A22_targeted_ranking",
    "A23_frozen_cell_residual_adapters",
    "A24_hard_boundary_residual_adapters",
    "A26_nominal_softmax_heads",
)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    os.replace(temporary_path, path)


def _test_metrics(results_root: Path, experiment: str, seed: int) -> dict:
    path = (
        results_root / experiment / f"seed{seed}"
        / "output" / "eval" / "test_metrics.json"
    )
    payload = _load_json(path)
    if payload.get("performance_gate", {}).get("passed") not in (
            True, False):
        raise ValueError(f"Missing performance gate in {path}.")
    return payload


def _promotion_summary(
        results_root: Path, experiment: str, seed: int) -> dict:
    run_root = results_root / experiment / f"seed{seed}"
    promotion_path = run_root / "validation_promotion.json"
    payload = _load_json(promotion_path)
    test_path = run_root / "output" / "eval" / "test_metrics.json"
    if payload.get("test_metrics_read") is not False:
        raise ValueError(
            f"{promotion_path} is not explicitly test-blind.")
    if test_path.exists():
        raise ValueError(
            f"Rejected candidate unexpectedly has test metrics: {test_path}")
    return {
        "validation_promotion_status": payload.get("status"),
        "test_metrics_read": False,
        "test_metrics_file_exists": False,
        "failures": list(payload.get("failures", [])),
        "promotion_report": str(promotion_path.resolve()),
    }


def build_summary(results_root: Path, seed: int) -> dict:
    baseline = _test_metrics(results_root, "A0_full", seed)
    confidence = _test_metrics(
        results_root, "A13_with_label_confidence", seed)
    selected = _test_metrics(
        results_root, "A19_bounded_class1_floor_calibration", seed)
    reliability_path = results_root / "a19_multiseed_reliability.json"
    reliability = _load_json(reliability_path)

    candidate_promotions = {
        experiment: _promotion_summary(
            results_root, experiment, seed)
        for experiment in DEFAULT_CANDIDATES
    }
    all_rejected = all(
        item["validation_promotion_status"] == "FAIL"
        for item in candidate_promotions.values()
    )
    all_test_blind = all(
        item["test_metrics_read"] is False
        and item["test_metrics_file_exists"] is False
        for item in candidate_promotions.values()
    )

    baseline_accuracy = float(baseline["avg_acc"])
    confidence_accuracy = float(confidence["avg_acc"])
    selected_accuracy = float(selected["avg_acc"])
    baseline_failure_count = len(
        baseline["performance_gate"]["failures"])
    selected_failure_count = len(
        selected["performance_gate"]["failures"])
    reliability_acceptance = reliability["acceptance"]

    return {
        "schema": "stage0_nn_single_seed_final_decision_v1",
        "status": "COMPLETE",
        "seed": int(seed),
        "decision": {
            "research_reference_model": (
                "A19_bounded_class1_floor_calibration"),
            "strict_deployment_status": "REJECT",
            "reason": (
                "Safety false-positive caps are stable, but the exact-L1 "
                "recall gate fails in every evaluated seed."),
            "no_rejected_candidate_test_access": bool(all_test_blind),
        },
        "single_seed_improvement": {
            "original_baseline": "A0_full",
            "improved_training_candidate": "A13_with_label_confidence",
            "current_reference": (
                "A19_bounded_class1_floor_calibration"),
            "a0_average_accuracy_percent": baseline_accuracy,
            "a13_average_accuracy_percent": confidence_accuracy,
            "a19_average_accuracy_percent": selected_accuracy,
            "a13_minus_a0_accuracy_percentage_points": (
                confidence_accuracy - baseline_accuracy),
            "a19_minus_a0_accuracy_percentage_points": (
                selected_accuracy - baseline_accuracy),
            "a0_gate_failure_count": baseline_failure_count,
            "a19_gate_failure_count": selected_failure_count,
            "gate_failure_count_reduction": (
                baseline_failure_count - selected_failure_count),
            "a19_strict_gate_passed": bool(
                selected["performance_gate"]["passed"]),
            "a19_small_k0_false_positive_percent": float(
                selected["small_k1"]["k0_fp"]),
            "a19_global_c0_false_positive_percent": float(
                selected["small_c1"]["c0_fp"]),
        },
        "post_reference_candidates": candidate_promotions,
        "post_reference_candidates_all_rejected": bool(all_rejected),
        "multiseed_reliability": {
            "report": str(reliability_path.resolve()),
            "status": reliability.get("status"),
            "seeds": list(reliability.get("seeds", [])),
            "strict_gate_pass_count": int(
                reliability_acceptance[
                    "strict_performance_gate_pass_count"]),
            "strict_gate_total": int(
                reliability_acceptance[
                    "strict_performance_gate_total"]),
            "safety_caps_stable_across_all_seeds": bool(
                reliability_acceptance[
                    "safety_caps_stable_across_all_seeds"]),
            "average_accuracy_mean_percent": float(
                reliability["statistics"][
                    "average_3class_accuracy_percent"]["mean"]),
            "average_accuracy_seed_std_percent": float(
                reliability["statistics"][
                    "average_3class_accuracy_percent"][
                        "sample_standard_deviation"]),
        },
        "artifact_identity": {
            "dataset_sha256": selected["dataset_sha256"],
            "selected_model_sha256": selected["model_sha256"],
            "all_multiseed_evaluation_hashes_verified": bool(
                reliability["artifact_contract"][
                    "all_evaluation_hashes_verified"]),
        },
        "limitations": [
            (
                "The fixed validation and test splits have been used by "
                "multiple historical experiments; new candidates are "
                "exploratory rather than untouched external validation."),
            (
                "The three-seed interval quantifies initialization "
                "variability only, not population or simulator uncertainty."),
            (
                "A19 is suitable as the current research reference, not as a "
                "strictly accepted deployment surrogate."),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create the hash-bound final decision for the single-seed "
            "improvement cycle without reading rejected-candidate tests."))
    parser.add_argument(
        "--results-root", default="output/experiments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="output/experiments/single_seed_final_decision.json")
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    payload = build_summary(results_root, args.seed)
    output_path = Path(args.output).resolve()
    _write_json_atomic(output_path, payload)
    print(json.dumps({
        "status": payload["status"],
        "strict_deployment_status": payload["decision"][
            "strict_deployment_status"],
        "a19_minus_a0_accuracy_percentage_points": payload[
            "single_seed_improvement"][
                "a19_minus_a0_accuracy_percentage_points"],
        "gate_failure_count_reduction": payload[
            "single_seed_improvement"][
                "gate_failure_count_reduction"],
        "rejected_candidates_test_blind": payload["decision"][
            "no_rejected_candidate_test_access"],
        "output": str(output_path),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
