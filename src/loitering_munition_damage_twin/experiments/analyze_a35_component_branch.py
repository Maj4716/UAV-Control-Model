from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as arrow_dataset
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


REPO_ROOT = Path(__file__).resolve().parents[3]

from loitering_munition_damage_twin.surrogate.dataset import FEATURE_COLUMNS
from loitering_munition_damage_twin.surrogate.features import (
    COMPONENT_PROXY_FEATURE_COLUMNS,
    augment_terminal_physics_features,
)
from loitering_munition_damage_twin.surrogate.model import (
    DamageAssessmentMTL,
    component_probabilities_to_ordinal,
)


MUNITION_NAMES = ("Small", "Med-LM", "Med-RD", "Heavy")
TASK_NAMES = ("K", "M", "F", "C")
ENTRY_CELLS = (
    ("Small/K", 0, 0, 0.005),
    ("Small/M", 0, 1, 0.025),
    ("Med-LM/K", 1, 0, 0.025),
    ("Med-RD/K", 2, 0, 0.025),
    ("Med-RD/M", 2, 1, 0.025),
    ("Heavy/K", 3, 0, 0.025),
    ("Small/C", 0, 3, 0.025),
    ("Med-LM/C", 1, 3, 0.025),
    ("Med-RD/C", 2, 3, 0.025),
    ("Heavy/C", 3, 3, 0.025),
)
CONDITIONAL_CELLS = (
    ("Med-LM/K_L1_vs_L2", 1, 0),
    ("Med-LM/F_L1_vs_L2", 1, 2),
    ("Med-RD/K_L1_vs_L2", 2, 0),
    ("Med-RD/M_L1_vs_L2", 2, 1),
    ("Med-RD/F_L1_vs_L2", 2, 2),
    ("Heavy/K_L1_vs_L2", 3, 0),
    ("Heavy/F_L1_vs_L2", 3, 2),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def _recall_at_fpr_cap(
        target: np.ndarray,
        score: np.ndarray,
        maximum_fpr: float,
) -> dict[str, float]:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        target, score)
    feasible = np.flatnonzero(
        false_positive_rate <= maximum_fpr + 1e-12)
    if feasible.size == 0:
        return {
            "maximum_recall": 0.0,
            "observed_false_positive_rate": 0.0,
            "threshold": float("inf"),
        }
    best = int(feasible[np.argmax(true_positive_rate[feasible])])
    return {
        "maximum_recall": float(true_positive_rate[best]),
        "observed_false_positive_rate": float(false_positive_rate[best]),
        "threshold": float(thresholds[best]),
    }


def _binary_ranking_metrics(
        target: np.ndarray,
        score: np.ndarray,
        maximum_fpr: float,
) -> dict:
    target = np.asarray(target, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    if np.unique(target).size != 2:
        return {
            "status": "ONE_CLASS",
            "support": int(target.sum()),
            "rows": int(target.size),
        }
    return {
        "status": "OK",
        "rows": int(target.size),
        "support": int(target.sum()),
        "full_auc": float(roc_auc_score(target, score)),
        "standardized_partial_auc": float(roc_auc_score(
            target, score, max_fpr=maximum_fpr)),
        "average_precision": float(
            average_precision_score(target, score)),
        "maximum_false_positive_rate": float(maximum_fpr),
        "recall_at_fpr_cap": _recall_at_fpr_cap(
            target, score, maximum_fpr),
    }


def _probability_metrics(
        target: np.ndarray,
        predicted: np.ndarray,
) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    residual = predicted - target
    return {
        "rows": int(target.shape[0]),
        "elements": int(target.size),
        "target_mean": float(target.mean()),
        "predicted_mean": float(predicted.mean()),
        "mean_bias": float(residual.mean()),
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.square(residual).mean())),
        "brier": float(np.square(residual).mean()),
    }


def _load_validation_frame(
        dataset_path: Path,
        feature_names: list[str],
        armor_aware_fragment_proxies: bool = False,
) -> pd.DataFrame:
    raw_columns = list(dict.fromkeys(
        list(FEATURE_COLUMNS)
        + [
            "munition_id",
            "sample_id",
            "split_role",
            "K_level",
            "M_level",
            "F_level",
            "C_level",
        ]
    ))
    dataset = arrow_dataset.dataset(
        str(dataset_path), format="parquet")
    frame = dataset.to_table(
        columns=raw_columns,
        filter=arrow_dataset.field("split_role") == "val",
    ).to_pandas()
    if frame.empty or not frame["split_role"].eq("val").all():
        raise RuntimeError(
            "Validation-only Parquet predicate was not enforced.")
    include_component_proxies = any(
        name in COMPONENT_PROXY_FEATURE_COLUMNS
        for name in feature_names
    )
    frame = augment_terminal_physics_features(
        frame,
        copy=False,
        include_component_proxies=include_component_proxies,
        armor_aware_fragment_proxies=(
            bool(armor_aware_fragment_proxies)),
    )
    missing = sorted(set(feature_names) - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"Model feature contract cannot be reconstructed: {missing}")
    return frame


def _load_validation_components(
        component_path: Path,
        sample_ids: np.ndarray,
        target_columns: list[str],
) -> np.ndarray:
    sidecar = arrow_dataset.dataset(
        str(component_path), format="parquet")
    identifier_values = [str(value) for value in sample_ids.tolist()]
    table = sidecar.to_table(
        columns=["sample_id"] + target_columns,
        filter=arrow_dataset.field("sample_id").isin(
            identifier_values),
    )
    frame = table.to_pandas()
    if len(frame) != len(sample_ids):
        raise RuntimeError(
            "Validation component join is incomplete: "
            f"expected={len(sample_ids)}, observed={len(frame)}")
    if frame["sample_id"].duplicated().any():
        raise RuntimeError(
            "Component sidecar contains duplicate validation sample_id.")
    frame = frame.set_index("sample_id").reindex(
        identifier_values)
    if frame.index.hasnans or frame[target_columns].isna().any().any():
        raise RuntimeError(
            "Validation component join contains missing rows.")
    values = frame[target_columns].to_numpy(
        dtype=np.float32)
    if values.shape[1] % 2:
        raise RuntimeError(
            "Component target column count must be even.")
    component_count = values.shape[1] // 2
    return values.reshape(
        len(values), 2, component_count)


def _predict(
        model: DamageAssessmentMTL,
        features: np.ndarray,
        munition_ids: np.ndarray,
        batch_size: int,
        device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    direct_probabilities = []
    fused_probabilities = []
    component_probabilities = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), int(batch_size)):
            stop = min(start + int(batch_size), len(features))
            batch_features = torch.as_tensor(
                features[start:stop],
                dtype=torch.float32,
                device=device,
            )
            batch_munitions = torch.as_tensor(
                munition_ids[start:stop],
                dtype=torch.long,
                device=device,
            )
            direct_logits, _, _ = model.forward_with_mechanisms(
                batch_features, batch_munitions)
            fused_logits, component_logits = (
                model.forward_with_components(
                    batch_features, batch_munitions))
            direct_probabilities.append(
                torch.sigmoid(direct_logits).cpu().numpy())
            fused_probabilities.append(
                torch.sigmoid(fused_logits).cpu().numpy())
            component_probabilities.append(
                torch.sigmoid(component_logits).cpu().numpy())
    return (
        np.concatenate(direct_probabilities, axis=0),
        np.concatenate(fused_probabilities, axis=0),
        np.concatenate(component_probabilities, axis=0),
    )


def _component_tree(
        mechanism_probabilities: np.ndarray,
        batch_size: int = 4096,
) -> np.ndarray:
    outputs = []
    with torch.inference_mode():
        for start in range(
                0, len(mechanism_probabilities), int(batch_size)):
            values = torch.as_tensor(
                mechanism_probabilities[
                    start:start + int(batch_size)],
                dtype=torch.float32,
            )
            combined = (
                1.0
                - (1.0 - values[:, 0])
                * (1.0 - values[:, 1])
            )
            outputs.append(
                component_probabilities_to_ordinal(
                    combined).numpy())
    return np.concatenate(outputs, axis=0)


def _entry_analysis(
        levels: np.ndarray,
        munition_ids: np.ndarray,
        probability_sets: dict[str, np.ndarray],
) -> dict:
    cells = {}
    for (
        name,
        munition_index,
        task_index,
        maximum_fpr,
    ) in ENTRY_CELLS:
        mask = munition_ids == munition_index
        target = (
            levels[mask, task_index] >= 1
        ).astype(np.int64)
        metrics = {}
        for probability_name, probabilities in (
                probability_sets.items()):
            metrics[probability_name] = (
                _binary_ranking_metrics(
                    target,
                    probabilities[mask, task_index, 0],
                    maximum_fpr,
                )
            )
        direct_score = probability_sets[
            "direct_A31"][mask, task_index, 0]
        tree_score = probability_sets[
            "predicted_component_tree"][
                mask, task_index, 0]
        grid_candidates = []
        for alpha in np.linspace(0.0, 1.0, 41):
            blended_score = (
                (1.0 - alpha) * direct_score
                + alpha * tree_score
            )
            candidate = _binary_ranking_metrics(
                target, blended_score, maximum_fpr)
            candidate["alpha"] = float(alpha)
            grid_candidates.append(candidate)
        valid_candidates = [
            candidate for candidate in grid_candidates
            if candidate["status"] == "OK"
        ]
        best_partial_auc = max(
            valid_candidates,
            key=lambda candidate: (
                candidate["standardized_partial_auc"],
                candidate["recall_at_fpr_cap"][
                    "maximum_recall"],
                -candidate["alpha"],
            ),
        )
        best_recall = max(
            valid_candidates,
            key=lambda candidate: (
                candidate["recall_at_fpr_cap"][
                    "maximum_recall"],
                candidate["standardized_partial_auc"],
                -candidate["alpha"],
            ),
        )
        metrics["validation_blend_grid"] = {
            "step": 0.025,
            "selection_role": (
                "diagnostic only; alpha is not applied to A35"),
            "best_standardized_partial_auc": (
                best_partial_auc),
            "best_recall_at_fpr_cap": best_recall,
        }
        cells[name] = metrics
    return cells


def _conditional_analysis(
        levels: np.ndarray,
        munition_ids: np.ndarray,
        probability_sets: dict[str, np.ndarray],
) -> dict:
    cells = {}
    for (
        name,
        munition_index,
        task_index,
    ) in CONDITIONAL_CELLS:
        mask = (
            (munition_ids == munition_index)
            & (levels[:, task_index] >= 1)
        )
        target = (
            levels[mask, task_index] >= 2
        ).astype(np.int64)
        metrics = {}
        for probability_name, probabilities in (
                probability_sets.items()):
            score = probabilities[
                mask, task_index, 1]
            if np.unique(target).size != 2:
                metrics[probability_name] = {
                    "status": "ONE_CLASS",
                    "rows": int(target.size),
                    "support": int(target.sum()),
                }
            else:
                metrics[probability_name] = {
                    "status": "OK",
                    "rows": int(target.size),
                    "level2_support": int(target.sum()),
                    "level1_support": int(
                        target.size - target.sum()),
                    "conditional_auc": float(
                        roc_auc_score(target, score)),
                    "average_precision": float(
                        average_precision_score(
                            target, score)),
                }
        direct_score = probability_sets[
            "direct_A31"][mask, task_index, 1]
        tree_score = probability_sets[
            "predicted_component_tree"][
                mask, task_index, 1]
        grid = []
        if np.unique(target).size == 2:
            for alpha in np.linspace(0.0, 1.0, 41):
                score = (
                    (1.0 - alpha) * direct_score
                    + alpha * tree_score
                )
                grid.append({
                    "alpha": float(alpha),
                    "conditional_auc": float(
                        roc_auc_score(target, score)),
                })
            metrics["validation_blend_grid"] = {
                "step": 0.025,
                "selection_role": (
                    "diagnostic only; alpha is not applied to A35"),
                "best": max(
                    grid,
                    key=lambda candidate: (
                        candidate["conditional_auc"],
                        -candidate["alpha"],
                    ),
                ),
            }
        cells[name] = metrics
    return cells


def _component_detail(
        targets: np.ndarray,
        predictions: np.ndarray,
        munition_ids: np.ndarray,
        component_ids: list[int],
) -> dict:
    relevant_ids = (3, 46, *range(58, 68))
    component_index = {
        int(component_id): index
        for index, component_id in enumerate(component_ids)
    }
    detail = {}
    for munition_index, munition_name in enumerate(
            MUNITION_NAMES):
        mask = munition_ids == munition_index
        per_component = {}
        for component_id in relevant_ids:
            index = component_index[int(component_id)]
            per_mechanism = {}
            for mechanism_index, mechanism_name in enumerate(
                    ("fragment", "shock")):
                target = targets[
                    mask, mechanism_index, index]
                predicted = predictions[
                    mask, mechanism_index, index]
                metrics = _probability_metrics(
                    target, predicted)
                binary_target = (target >= 0.5).astype(
                    np.int64)
                metrics["target_ge_0p5_support"] = int(
                    binary_target.sum())
                if np.unique(binary_target).size == 2:
                    metrics["target_ge_0p5_auc"] = float(
                        roc_auc_score(
                            binary_target, predicted))
                per_mechanism[mechanism_name] = metrics
            per_component[str(component_id)] = (
                per_mechanism)
        detail[munition_name] = per_component
    return detail


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only diagnosis of the A35 independent component "
            "branch. Test rows are excluded at the Arrow scan predicate."
        )
    )
    parser.add_argument(
        "--run-dir",
        default=(
            "output/experiments/"
            "A35_independent_component_tree_fusion/seed42"),
    )
    parser.add_argument(
        "--output",
        default=(
            "output/experiments/"
            "A35_independent_component_tree_fusion/seed42/"
            "output/validation/component_branch_analysis.json"),
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    model_dir = run_dir / "output" / "models"
    manifest_path = model_dir / "model_manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != "stage0_nn_artifact_v1":
        raise RuntimeError(
            "Unsupported model manifest schema.")

    artifacts = manifest["artifacts"]
    checkpoint_path = model_dir / "best_model.pth"
    scaler_path = model_dir / "minmax_scaler.pkl"
    for path, key in (
            (checkpoint_path, "best_model.pth"),
            (scaler_path, "minmax_scaler.pkl")):
        expected = artifacts[key]["sha256"]
        observed = _sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"Artifact SHA-256 mismatch for {path}: "
                f"expected={expected}, observed={observed}")

    data_contract = manifest["data_contract"]
    dataset_path = Path(
        data_contract["dataset_path"]).resolve()
    if _sha256(dataset_path) != (
            data_contract["dataset_sha256"]):
        raise RuntimeError(
            "Dataset SHA-256 does not match model manifest.")
    component_contract = data_contract[
        "component_supervision_contract"]
    component_path = Path(
        component_contract["path"]).resolve()
    if _sha256(component_path) != component_contract["sha256"]:
        raise RuntimeError(
            "Component sidecar SHA-256 does not match model manifest.")

    feature_names = list(data_contract["feature_names"])
    validation = _load_validation_frame(
        dataset_path, feature_names)
    sample_ids = validation["sample_id"].astype(
        str).to_numpy()
    component_targets = _load_validation_components(
        component_path,
        sample_ids,
        list(component_contract["target_columns"]),
    )
    with scaler_path.open("rb") as stream:
        scaler = pickle.load(stream)
    features = scaler.transform(
        validation[feature_names].to_numpy(
            dtype=np.float32)
    ).astype(np.float32)
    munition_ids = validation[
        "munition_id"].to_numpy(dtype=np.int64)
    levels = validation[
        [f"{task}_level" for task in TASK_NAMES]
    ].to_numpy(dtype=np.int64)

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = DamageAssessmentMTL(
        **manifest["model_config"]).to(device)
    try:
        state = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        state = torch.load(
            checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=True)
    direct, fused, component_predictions = _predict(
        model,
        features,
        munition_ids,
        args.batch_size,
        device,
    )
    predicted_tree = _component_tree(
        component_predictions)
    target_tree = _component_tree(
        component_targets)

    probability_sets = {
        "direct_A31": direct,
        "fused_A35": fused,
        "predicted_component_tree": predicted_tree,
        "target_component_tree_upper_bound": target_tree,
    }
    component_metrics_by_munition = {}
    for munition_index, munition_name in enumerate(
            MUNITION_NAMES):
        mask = munition_ids == munition_index
        component_metrics_by_munition[munition_name] = (
            _probability_metrics(
                component_targets[mask],
                component_predictions[mask],
            )
        )

    payload = {
        "schema": "stage0_a35_component_branch_analysis_v1",
        "status": "COMPLETE",
        "split": "validation",
        "test_labels_used": False,
        "validation_scan_predicate": "split_role == 'val'",
        "validation_rows": int(len(validation)),
        "run_dir": str(run_dir),
        "model_checkpoint_sha256": _sha256(
            checkpoint_path),
        "scaler_sha256": _sha256(scaler_path),
        "dataset_sha256": _sha256(dataset_path),
        "component_sidecar_sha256": _sha256(
            component_path),
        "device": str(device),
        "component_probability_metrics": {
            "overall": _probability_metrics(
                component_targets,
                component_predictions,
            ),
            "by_munition": component_metrics_by_munition,
        },
        "entry_cell_ranking": _entry_analysis(
            levels, munition_ids, probability_sets),
        "conditional_cell_ranking": _conditional_analysis(
            levels, munition_ids, probability_sets),
        "relevant_component_detail": _component_detail(
            component_targets,
            component_predictions,
            munition_ids,
            list(component_contract["component_ids"]),
        ),
    }
    output_path = Path(args.output).resolve()
    _write_json_atomic(output_path, payload)
    print(json.dumps({
        "status": payload["status"],
        "split": payload["split"],
        "test_labels_used": payload[
            "test_labels_used"],
        "validation_rows": payload[
            "validation_rows"],
        "component_probability_metrics": payload[
            "component_probability_metrics"],
        "entry_cell_ranking": payload[
            "entry_cell_ranking"],
        "output": str(output_path),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
