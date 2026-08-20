from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]

from loitering_munition_damage_twin.surrogate.artifacts import (
    data_contracts_match,
    load_and_verify_manifest,
    sha256_file,
)
from loitering_munition_damage_twin.surrogate.dataset import get_dataloaders, get_feature_columns
from loitering_munition_damage_twin.surrogate.model import DamageAssessmentMTL
from loitering_munition_damage_twin.surrogate.training import (
    FocalUncertaintyOrdinalLoss,
    _evaluate_selection_snapshot,
    _threshold_json_from_metrics,
    _validation_report_from_metrics,
)
from loitering_munition_damage_twin.experiments.ablation_config import load_ablation_config
from loitering_munition_damage_twin.experiments.validate_a23_promotion import (
    CONDITIONAL_CELL,
    WEAK_ENTRY_CELLS,
)


INFERENCE_CONTRACT_KEYS = (
    "dataset_sha256",
    "dataset_schema",
    "frame_convention",
    "feature_names",
    "ordinal_applicability",
)


def inference_contracts_match(left: dict, right: dict) -> bool:
    """Compare deployable inputs while allowing training-only targets to vary."""
    return all(
        left.get(key) == right.get(key)
        for key in INFERENCE_CONTRACT_KEYS
    )


class EqualWeightProbabilityEnsemble(nn.Module):
    def __init__(self, members: list[nn.Module], weights: list[float]):
        super().__init__()
        if not members or len(members) != len(weights):
            raise ValueError("members and weights must have equal nonzero size.")
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        if (
            not torch.isfinite(weight_tensor).all()
            or (weight_tensor <= 0.0).any()
        ):
            raise ValueError("Ensemble weights must be finite and positive.")
        weight_tensor /= weight_tensor.sum()
        self.members = nn.ModuleList(members)
        self.register_buffer(
            "weights", weight_tensor, persistent=True)

    def forward(self, x, munition_id):
        member_probabilities = torch.stack([
            torch.sigmoid(member(x, munition_id))
            for member in self.members
        ], dim=0)
        probabilities = torch.sum(
            member_probabilities
            * self.weights.view(-1, 1, 1, 1),
            dim=0,
        )
        probabilities = torch.clamp(
            probabilities, min=1e-7, max=1.0 - 1e-7)
        return torch.logit(probabilities)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    os.replace(temporary_path, path)


def _canonical_hash_payload(payload: dict, path: Path) -> str:
    _write_json_atomic(path, payload)
    return sha256_file(str(path))


def _diagnostic_objectives(report: dict) -> dict[str, float]:
    diagnostics = report["targeted_probability_diagnostics"]
    objectives = {
        cell: float(
            diagnostics[cell][
                "entry_standardized_partial_auc"])
        for cell in WEAK_ENTRY_CELLS
    }
    objectives[CONDITIONAL_CELL] = float(
        diagnostics[CONDITIONAL_CELL]["conditional_auc"])
    return objectives


def _evaluate_promotion(
        member_reports: list[dict],
        candidate_report: dict,
        criteria: dict) -> dict:
    member_accuracy = [
        float(report["average_3class_accuracy_percent"])
        for report in member_reports
    ]
    accuracy_delta = (
        float(candidate_report["average_3class_accuracy_percent"])
        - sum(member_accuracy) / len(member_accuracy)
    )
    member_failures = [
        int(report["performance_gate"]["failure_count"])
        for report in member_reports
    ]
    candidate_failures = int(
        candidate_report["performance_gate"]["failure_count"])
    member_objectives = [
        _diagnostic_objectives(report)
        for report in member_reports
    ]
    objective_names = tuple(member_objectives[0])
    member_mean_objectives = {
        name: sum(values[name] for values in member_objectives)
        / len(member_objectives)
        for name in objective_names
    }
    candidate_objectives = _diagnostic_objectives(candidate_report)
    deltas = {
        name: candidate_objectives[name] - member_mean_objectives[name]
        for name in objective_names
    }
    minimum_improvement = float(
        criteria["minimum_objective_improvement_vs_member_mean"])
    maximum_degradation = float(
        criteria["maximum_objective_degradation_vs_member_mean"])
    improved = [
        name for name, delta in deltas.items()
        if delta >= minimum_improvement
    ]
    degraded = [
        name for name, delta in deltas.items()
        if delta < -maximum_degradation
    ]
    mean_delta = sum(deltas.values()) / len(deltas)
    failures = []
    if (
        bool(criteria.get("require_strict_goal_pass", False))
        and candidate_report.get(
            "goal_performance_gate", {}).get("passed") is not True
    ):
        failures.append(
            "ensemble does not pass the strict 94%/90% validation goal")
    if accuracy_delta < -float(criteria[
            "maximum_accuracy_drop_vs_member_mean_percentage_points"]):
        failures.append(
            "validation accuracy is below the member mean budget")
    if float(candidate_report[
            "small_k0_false_positive_percent"]) > float(
                criteria[
                    "maximum_small_k0_false_positive_percent"]):
        failures.append("Small/K false-positive cap exceeded")
    if float(candidate_report[
            "global_c0_false_positive_percent"]) > float(
                criteria[
                    "maximum_global_c0_false_positive_percent"]):
        failures.append("global C false-positive cap exceeded")
    if candidate_failures > min(member_failures):
        failures.append(
            "strict validation failures exceed the best member")
    if len(improved) < int(
            criteria["minimum_improved_weak_objectives"]):
        failures.append(
            "too few weak-cell objectives improved over member mean")
    if degraded:
        failures.append(
            "at least one weak-cell objective degraded over budget")
    if mean_delta < float(
            criteria["minimum_mean_objective_delta"]):
        failures.append(
            "mean weak-cell objective gain is below the threshold")
    return {
        "schema": "stage0_nn_multiseed_ensemble_promotion_v1",
        "status": "PASS" if not failures else "FAIL",
        "split": "validation",
        "test_metrics_read": False,
        "criteria": criteria,
        "metrics": {
            "member_accuracy_percent": member_accuracy,
            "candidate_accuracy_percent": float(
                candidate_report[
                    "average_3class_accuracy_percent"]),
            "accuracy_delta_vs_member_mean_percentage_points": (
                accuracy_delta),
            "member_gate_failure_counts": member_failures,
            "candidate_gate_failure_count": candidate_failures,
            "objective_deltas_vs_member_mean": deltas,
            "mean_objective_delta": mean_delta,
            "improved_objectives": improved,
            "degraded_objectives": degraded,
        },
        "failures": failures,
    }


def validate_ensemble(config_path: Path, output_dir: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as stream:
        ensemble_config = json.load(stream)
    members_config = ensemble_config.get("members", [])
    if len(members_config) < 2:
        raise ValueError("Ensemble requires at least two members.")
    weights = [float(member["weight"]) for member in members_config]
    if abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError("Pre-registered ensemble weights must sum to one.")

    data_path = REPO_ROOT / "output" / "damage_dataset.parquet"
    profile_path = data_path.parent / "generation_profile.json"
    with profile_path.open("r", encoding="utf-8") as stream:
        profile = json.load(stream)
    dataset_sha256 = str(profile["artifact"]["sha256"])
    training_config = load_ablation_config(
        str(ensemble_config.get(
            "input_config",
            "A19_bounded_class1_floor_calibration",
        )))
    feature_columns = get_feature_columns(training_config)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    loaded_members = []
    member_records = []
    member_reports = []
    scaler_hashes = set()
    member_contracts = []
    for member in members_config:
        experiment_id = str(member["experiment_id"])
        seed = int(member["seed"])
        run_dir = (
            REPO_ROOT / "abli_exp" / "results"
            / experiment_id / f"seed{seed}"
        )
        model_dir = run_dir / "output" / "models"
        manifest = load_and_verify_manifest(
            str(model_dir),
            dataset_sha256=dataset_sha256,
            feature_names=feature_columns,
        )
        model_path = model_dir / "best_model.pth"
        model = DamageAssessmentMTL(
            **manifest["model_config"]).to(device)
        model.load_state_dict(torch.load(
            model_path, map_location=device, weights_only=True))
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        loaded_members.append(model)
        scaler_hashes.add(
            manifest["artifacts"]["minmax_scaler.pkl"]["sha256"])
        member_contracts.append(manifest["data_contract"])
        report_source = str(member.get(
            "validation_report_source", "recalibration"))
        if report_source == "selection":
            report_path = (
                run_dir / "output" / "validation"
                / "selection_metrics.json"
            )
        elif report_source == "recalibration":
            report_path = run_dir / "recalibration_report.json"
        else:
            raise ValueError(
                "validation_report_source must be "
                "'selection' or 'recalibration'.")
        with report_path.open("r", encoding="utf-8") as stream:
            raw_report = json.load(stream)
        member_report = (
            raw_report
            if report_source == "selection"
            else raw_report["selection_validation_report"]
        )
        if (
            member_report.get("split") != "validation"
            or member_report.get("test_labels_used") is not False
        ):
            raise RuntimeError(
                f"Member seed {seed} has no test-blind validation report.")
        member_reports.append(member_report)
        member_records.append({
            "experiment_id": experiment_id,
            "seed": seed,
            "weight": float(member["weight"]),
            "model_path": str(model_path),
            "model_sha256": sha256_file(str(model_path)),
            "manifest_path": str(model_dir / "model_manifest.json"),
            "manifest_sha256": sha256_file(
                str(model_dir / "model_manifest.json")),
            "validation_report_path": str(report_path),
            "validation_report_sha256": sha256_file(str(report_path)),
        })
    if len(scaler_hashes) != 1:
        raise RuntimeError("Ensemble members do not share one scaler.")
    if any(
        not inference_contracts_match(
            member_contracts[0], contract)
        for contract in member_contracts[1:]
    ):
        raise RuntimeError(
            "Ensemble member inference contracts are not identical.")

    first_model_dir = Path(
        member_records[0]["model_path"]).parent
    with (first_model_dir / "minmax_scaler.pkl").open("rb") as stream:
        fitted_scaler = pickle.load(stream)
    (_, val_loader, _, _, _, runtime_contract) = get_dataloaders(
        str(data_path),
        batch_size=256,
        persist_scaler=False,
        ablation_config=training_config,
        scaler_override=fitted_scaler,
        load_test_split=False,
    )
    if not inference_contracts_match(
            runtime_contract, member_contracts[0]):
        raise RuntimeError(
            "Runtime inference contract differs from ensemble members.")

    ensemble = EqualWeightProbabilityEnsemble(
        loaded_members, weights).to(device)
    ensemble.eval()
    criterion = FocalUncertaintyOrdinalLoss(
        gamma=0.0,
        penalty_weight=0.0,
        class_distribution_weight=0.0,
        pos_weight=torch.ones(4, 4, 2, device=device),
    ).to(device)
    calibration = ensemble_config["calibration"]
    maximum_accuracy_drop_raw = calibration.get(
        "maximum_class1_floor_accuracy_drop")
    maximum_accuracy_drop = (
        None if maximum_accuracy_drop_raw is None
        else float(maximum_accuracy_drop_raw)
    )
    metrics = _evaluate_selection_snapshot(
        ensemble,
        criterion,
        val_loader,
        device,
        use_amp=(device.type == "cuda"),
        epoch_label=0,
        minimum_exact_l1_recall=float(
            calibration["minimum_exact_class1_recall"]),
        maximum_class1_floor_accuracy_drop=maximum_accuracy_drop,
    )

    output_dir = output_dir.resolve()
    model_dir = output_dir / "output" / "models"
    validation_dir = output_dir / "output" / "validation"
    members_path = model_dir / "ensemble_members.json"
    members_payload = {
        "schema": "stage0_nn_probability_ensemble_v1",
        "experiment_id": ensemble_config["experiment_id"],
        "aggregation": "weighted_arithmetic_mean_of_probabilities",
        "members": member_records,
        "dataset_sha256": dataset_sha256,
        "scaler_sha256": next(iter(scaler_hashes)),
        "test_labels_used": False,
    }
    ensemble_identity_sha256 = _canonical_hash_payload(
        members_payload, members_path)
    thresholds = _threshold_json_from_metrics(
        metrics,
        model_variant="equal_weight_probability_ensemble",
        raw_best_epoch=0,
        soup_epochs=[],
    )
    thresholds["_ensemble"] = {
        "members_sha256": ensemble_identity_sha256,
        "member_model_sha256": [
            member["model_sha256"] for member in member_records],
        "weights": weights,
        "validation_only": True,
    }
    threshold_path = model_dir / "best_thresholds.json"
    _write_json_atomic(threshold_path, thresholds)
    validation_report = _validation_report_from_metrics(
        metrics,
        model_variant="equal_weight_probability_ensemble",
        dataset_sha256=dataset_sha256,
        model_sha256=ensemble_identity_sha256,
        threshold_sha256=sha256_file(str(threshold_path)),
    )
    validation_report_path = (
        validation_dir / "selection_metrics.json")
    _write_json_atomic(validation_report_path, validation_report)
    promotion = _evaluate_promotion(
        member_reports,
        validation_report,
        ensemble_config["promotion"],
    )
    promotion["candidate"] = str(
        ensemble_config["experiment_id"])
    promotion_path = output_dir / "validation_promotion.json"
    _write_json_atomic(promotion_path, promotion)
    print(json.dumps({
        "status": promotion["status"],
        "validation_accuracy_percent": validation_report[
            "average_3class_accuracy_percent"],
        "validation_gate_failures": validation_report[
            "performance_gate"]["failure_count"],
        "improved_objectives": promotion["metrics"][
            "improved_objectives"],
        "mean_objective_delta": promotion["metrics"][
            "mean_objective_delta"],
        "test_metrics_read": False,
        "output": str(promotion_path),
    }, indent=2, ensure_ascii=False))
    return promotion


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate and gate a fixed multi-seed probability ensemble "
            "using validation data only."
        )
    )
    parser.add_argument(
        "--config",
        default=str(
            REPO_ROOT / "abli_exp" / "configs"
            / "A25_equal_weight_seed_ensemble.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            REPO_ROOT / "abli_exp" / "results"
            / "A25_equal_weight_seed_ensemble"
            / "seeds42_43_44"),
    )
    args = parser.parse_args()
    result = validate_ensemble(
        Path(args.config), Path(args.output_dir))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
