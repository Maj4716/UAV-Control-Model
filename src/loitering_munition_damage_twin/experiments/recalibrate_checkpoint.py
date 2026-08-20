from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]

from loitering_munition_damage_twin.surrogate.artifacts import (
    data_contracts_match,
    load_and_verify_manifest,
    sha256_file,
    write_model_manifest,
)
from loitering_munition_damage_twin.surrogate.dataset import get_dataloaders, get_feature_columns
from loitering_munition_damage_twin.surrogate.model import DamageAssessmentMTL
from loitering_munition_damage_twin.surrogate.training import (
    CLASS1_FLOOR_MAX_ACCURACY_DROP,
    CLASS1_FLOOR_RECALL,
    CURRENT_THRESHOLD_SCHEMA,
    FocalUncertaintyOrdinalLoss,
    MUN_NAMES,
    TASK_NAMES,
    _evaluate_selection_snapshot,
    _threshold_json_from_metrics,
    _validation_report_from_metrics,
)

from loitering_munition_damage_twin.experiments.ablation_config import (
    load_ablation_config,
)

def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def recalibrate_checkpoint(
        run_dir: Path,
        data_path: Path,
        config: dict,
        seed: int) -> Path:
    """Recalibrate a sealed checkpoint on validation data only.

    The model and scaler remain byte-identical.  Only the threshold artifact
    and its manifest entry are renewed, with explicit provenance linking the
    recalibrated artifact to the source model/threshold hashes.
    """
    run_dir = run_dir.resolve()
    data_path = data_path.resolve()
    model_dir = run_dir / "output" / "models"
    model_path = model_dir / "best_model.pth"
    threshold_path = model_dir / "best_thresholds.json"
    scaler_path = model_dir / "minmax_scaler.pkl"
    if not model_path.is_file() or not threshold_path.is_file():
        raise FileNotFoundError(
            f"Recalibration requires copied sealed model artifacts: {model_dir}")

    profile_path = data_path.parent / "generation_profile.json"
    with profile_path.open("r", encoding="utf-8") as stream:
        profile = json.load(stream)
    dataset_sha256 = str(profile.get("artifact", {}).get("sha256", ""))
    feature_columns = get_feature_columns(config)
    source_manifest = load_and_verify_manifest(
        str(model_dir),
        dataset_sha256=dataset_sha256,
        feature_names=feature_columns,
    )
    source_model_sha256 = sha256_file(str(model_path))
    source_threshold_sha256 = sha256_file(str(threshold_path))
    with threshold_path.open("r", encoding="utf-8") as stream:
        source_thresholds = json.load(stream)

    with scaler_path.open("rb") as stream:
        fitted_scaler = pickle.load(stream)
    (_, val_loader, _, _, _, data_contract) = get_dataloaders(
        str(data_path),
        batch_size=256,
        persist_scaler=False,
        ablation_config=config,
        scaler_override=fitted_scaler,
        load_test_split=False,
    )
    if not data_contracts_match(
            data_contract, source_manifest["data_contract"]):
        raise RuntimeError(
            "Recalibration data contract differs from the sealed model contract.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DamageAssessmentMTL(**source_manifest["model_config"]).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True),
        strict=True,
    )
    criterion = FocalUncertaintyOrdinalLoss(
        gamma=0.0,
        penalty_weight=0.0,
        class_distribution_weight=0.0,
        pos_weight=torch.ones(4, 4, 2, device=device),
    ).to(device)
    source_epoch = int(source_thresholds.get("_best_epoch", -1))
    calibration_config = config.get("calibration", {})
    minimum_exact_l1_recall = float(
        calibration_config.get(
            "minimum_exact_class1_recall",
            CLASS1_FLOOR_RECALL / 100.0))
    maximum_accuracy_drop = calibration_config.get(
        "maximum_class1_floor_accuracy_drop",
        CLASS1_FLOOR_MAX_ACCURACY_DROP,
    )
    if maximum_accuracy_drop is not None:
        maximum_accuracy_drop = float(maximum_accuracy_drop)
    metrics = _evaluate_selection_snapshot(
        model,
        criterion,
        val_loader,
        device,
        use_amp=(device.type == "cuda"),
        epoch_label=source_epoch,
        minimum_exact_l1_recall=minimum_exact_l1_recall,
        maximum_class1_floor_accuracy_drop=maximum_accuracy_drop,
    )

    thresholds = _threshold_json_from_metrics(
        metrics,
        model_variant="posthoc_exact_l1_floor_recalibration",
        raw_best_epoch=int(source_thresholds.get(
            "_raw_best_epoch", source_epoch)),
        soup_epochs=list(source_thresholds.get("_soup_epochs", [])),
    )
    thresholds["_recalibration"] = {
        "validation_only": True,
        "source_threshold_schema": source_thresholds.get("_schema"),
        "source_model_sha256": source_model_sha256,
        "source_threshold_sha256": source_threshold_sha256,
        "minimum_exact_class1_recall": minimum_exact_l1_recall,
        "maximum_class1_floor_accuracy_drop": maximum_accuracy_drop,
        "fallback_policy": (
            "retain the best false-positive-safe threshold when the recall "
            "floor is unattainable"),
    }
    _write_json_atomic(threshold_path, thresholds)

    training_config = dict(source_manifest.get("training_config", {}))
    training_config["posthoc_recalibration"] = {
        "enabled": True,
        "threshold_schema": CURRENT_THRESHOLD_SCHEMA,
        "validation_only": True,
        "source_model_sha256": source_model_sha256,
        "source_threshold_sha256": source_threshold_sha256,
        "minimum_exact_class1_recall": minimum_exact_l1_recall,
        "maximum_class1_floor_accuracy_drop": maximum_accuracy_drop,
    }
    manifest_path = write_model_manifest(
        str(model_dir),
        data_contract=data_contract,
        model_config=source_manifest["model_config"],
        training_config=training_config,
        seed=seed,
    )

    recall_matrix = metrics["cls1_recall_per_mun"].numpy()
    support_matrix = metrics["cls1_count_per_mun"].numpy()
    validation_cells = {
        munition: {
            task: {
                "class1_support": int(support_matrix[task_id, munition_id]),
                "class1_recall_percent": float(
                    recall_matrix[task_id, munition_id]),
            }
            for task_id, task in enumerate(TASK_NAMES)
        }
        for munition_id, munition in enumerate(MUN_NAMES)
    }
    report = {
        "schema": "stage0_nn_recalibration_v1",
        "status": "COMPLETE",
        "experiment_id": config.get("experiment_id"),
        "seed": int(seed),
        "device": str(device),
        "validation_rows": int(sum(metrics["samples_per_munition"]).item()),
        "source_model_sha256": source_model_sha256,
        "source_threshold_sha256": source_threshold_sha256,
        "new_threshold_sha256": sha256_file(str(threshold_path)),
        "threshold_schema": thresholds["_schema"],
        "minimum_exact_class1_recall": minimum_exact_l1_recall,
        "maximum_class1_floor_accuracy_drop": maximum_accuracy_drop,
        "model_weights_unchanged": (
            sha256_file(str(model_path)) == source_model_sha256),
        "validation_average_3class_accuracy": (
            float(metrics["acc3_mean"]) * 100.0),
        "validation_class1_recall_by_cell": validation_cells,
        "validation_targeted_probability_diagnostics": metrics.get(
            "targeted_probability_diagnostics", {}),
        "validation_low_class1_cells": metrics["low_cls1_cells"],
        "validation_small_k0_false_positive_percent": (
            float(metrics["small_k0_fp_rate"]) * 100.0),
        "validation_global_c0_false_positive_percent": (
            float(metrics["c0_fp_rate"]) * 100.0),
        "model_manifest": str(manifest_path),
        "selection_validation_report": (
            _validation_report_from_metrics(
                metrics,
                model_variant=(
                    "posthoc_exact_l1_floor_recalibration"),
                dataset_sha256=dataset_sha256,
                model_sha256=source_model_sha256,
                threshold_sha256=sha256_file(
                    str(threshold_path)),
            )
        ),
    }
    report_path = run_dir / "recalibration_report.json"
    _write_json_atomic(report_path, report)
    print(
        f"[RECAL] COMPLETE schema={thresholds['_schema']} "
        f"val_acc={report['validation_average_3class_accuracy']:.2f}% "
        f"SmallK0_FP="
        f"{report['validation_small_k0_false_positive_percent']:.3f}% "
        f"C0_FP={report['validation_global_c0_false_positive_percent']:.3f}%"
    )
    print(f"[RECAL] Report: {report_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recalibrate copied model thresholds on validation data without "
            "changing checkpoint weights."))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--ablation-config", required=True)
    parser.add_argument(
        "--data", default=str(REPO_ROOT / "output" / "damage_dataset.parquet"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = load_ablation_config(args.ablation_config)
    recalibrate_checkpoint(
        Path(args.run_dir),
        Path(args.data),
        config,
        args.seed,
    )


if __name__ == "__main__":
    main()
