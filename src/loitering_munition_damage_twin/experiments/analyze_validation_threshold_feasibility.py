from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pyarrow.dataset as arrow_dataset
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]

from loitering_munition_damage_twin.surrogate.model import DamageAssessmentMTL, DEFAULT_ORDINAL_APPLICABILITY
from loitering_munition_damage_twin.surrogate.features import (
    COMPONENT_PROXY_FEATURE_COLUMNS,
    augment_terminal_physics_features,
)

from loitering_munition_damage_twin.experiments.analyze_a35_component_branch import (
    MUNITION_NAMES,
    TASK_NAMES,
    _component_tree,
    _load_json,
    _load_validation_components,
    _load_validation_frame,
    _predict,
    _sha256,
    _write_json_atomic,
)


REPORT_SCHEMA = "stage0_validation_threshold_feasibility_v1"


def _terminal_rule_proxy_probabilities(validation) -> np.ndarray:
    """Combine deployable fragment/shock rule proxies into ordinal risks."""
    probabilities = np.empty(
        (len(validation), len(TASK_NAMES), 2),
        dtype=np.float64,
    )
    for task_index, task_name in enumerate(TASK_NAMES):
        for level_index, level in enumerate((1, 2)):
            fragment = validation[
                f"phys_fragment_{task_name}_ge{level}_rule_proxy"
            ].to_numpy(dtype=np.float64)
            shock = validation[
                f"phys_shock_{task_name}_ge{level}_rule_proxy"
            ].to_numpy(dtype=np.float64)
            probabilities[:, task_index, level_index] = (
                1.0 - (1.0 - fragment) * (1.0 - shock)
            )
    probabilities[..., 1] = np.minimum(
        probabilities[..., 1], probabilities[..., 0])
    return np.clip(probabilities, 0.0, 1.0)


def _terminal_combined_rule_proxy_probabilities(
        validation) -> np.ndarray:
    """Read the component-union-first analytic damage-tree proxy."""
    probabilities = np.empty(
        (len(validation), len(TASK_NAMES), 2),
        dtype=np.float64,
    )
    for task_index, task_name in enumerate(TASK_NAMES):
        for level_index, level in enumerate((1, 2)):
            probabilities[:, task_index, level_index] = validation[
                f"phys_combined_{task_name}_ge{level}_rule_proxy"
            ].to_numpy(dtype=np.float64)
    probabilities[..., 1] = np.minimum(
        probabilities[..., 1], probabilities[..., 0])
    return np.clip(probabilities, 0.0, 1.0)


def _predict_probabilities(
        model: DamageAssessmentMTL,
        features: np.ndarray,
        munition_ids: np.ndarray,
        batch_size: int,
        device: torch.device) -> np.ndarray:
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), int(batch_size)):
            stop = min(start + int(batch_size), len(features))
            values = torch.as_tensor(
                features[start:stop],
                dtype=torch.float32,
                device=device,
            )
            munitions = torch.as_tensor(
                munition_ids[start:stop],
                dtype=torch.long,
                device=device,
            )
            outputs.append(
                torch.sigmoid(model(values, munitions).float())
                .cpu().numpy()
            )
    return np.concatenate(outputs, axis=0)


def _predict_mechanism_probabilities(
        model: DamageAssessmentMTL,
        features: np.ndarray,
        munition_ids: np.ndarray,
        batch_size: int,
        device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return predicted combined, fragment and shock ordinal risks.

    The diagnostic deliberately calls ``forward_with_mechanisms`` instead
    of the deployment ``forward`` method so the two latent physical experts
    can be audited independently on validation data.  Recomputing the OR
    from the returned expert logits also makes this helper valid for both
    fixed-OR decomposition and auxiliary-head experiments.
    """
    combined_batches = []
    fragment_batches = []
    shock_batches = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), int(batch_size)):
            stop = min(start + int(batch_size), len(features))
            values = torch.as_tensor(
                features[start:stop],
                dtype=torch.float32,
                device=device,
            )
            munitions = torch.as_tensor(
                munition_ids[start:stop],
                dtype=torch.long,
                device=device,
            )
            _, fragment_logits, shock_logits = (
                model.forward_with_mechanisms(values, munitions)
            )
            if fragment_logits is None or shock_logits is None:
                raise RuntimeError(
                    "Selected model does not expose mechanism outputs.")
            fragment = torch.sigmoid(fragment_logits.float())
            shock = torch.sigmoid(shock_logits.float())
            combined = (
                1.0 - (1.0 - fragment) * (1.0 - shock)
            ).clamp_(0.0, 1.0)
            combined_batches.append(combined.cpu().numpy())
            fragment_batches.append(fragment.cpu().numpy())
            shock_batches.append(shock.cpu().numpy())
    return (
        np.concatenate(combined_batches, axis=0),
        np.concatenate(fragment_batches, axis=0),
        np.concatenate(shock_batches, axis=0),
    )


def _load_validation_mechanism_targets(
        dataset_path: Path,
        sample_ids: np.ndarray,
) -> np.ndarray:
    """Load fragment/shock MC means with an explicit validation predicate."""
    target_columns = [
        f"{mechanism}_{task}_ge{level}_prob"
        for mechanism in ("fragment", "shock")
        for task in TASK_NAMES
        for level in (1, 2)
    ]
    dataset = arrow_dataset.dataset(
        str(dataset_path), format="parquet")
    frame = dataset.to_table(
        columns=["sample_id", *target_columns],
        filter=arrow_dataset.field("split_role") == "val",
    ).to_pandas()
    identifiers = [str(value) for value in sample_ids.tolist()]
    if len(frame) != len(identifiers):
        raise RuntimeError(
            "Validation mechanism target read is incomplete: "
            f"expected={len(identifiers)}, observed={len(frame)}")
    frame["sample_id"] = frame["sample_id"].astype(str)
    if frame["sample_id"].duplicated().any():
        raise RuntimeError(
            "Validation mechanism targets contain duplicate sample_id.")
    frame = frame.set_index("sample_id").reindex(identifiers)
    if frame.index.hasnans or frame[target_columns].isna().any().any():
        raise RuntimeError(
            "Validation mechanism target join contains missing rows.")
    values = frame[target_columns].to_numpy(dtype=np.float32)
    return values.reshape(len(frame), 2, len(TASK_NAMES), 2)


def _mechanism_fit_metrics(
        predictions: np.ndarray,
        targets: np.ndarray,
        munition_ids: np.ndarray,
) -> dict:
    """Quantify expert regression without hiding sparse positives in zeros."""
    if predictions.shape != targets.shape or predictions.shape[1:] != (2, 4, 2):
        raise ValueError(
            "Mechanism predictions/targets must both have shape (N,2,4,2).")

    def metrics(predicted: np.ndarray, target: np.ndarray) -> dict:
        predicted = np.asarray(predicted, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        residual = predicted - target
        nonzero = target > 0.0
        hard_positive = target >= 0.5
        predicted_positive = predicted >= 0.5
        return {
            "rows": int(target.size),
            "target_mean": float(target.mean()),
            "predicted_mean": float(predicted.mean()),
            "mean_bias": float(residual.mean()),
            "mae": float(np.abs(residual).mean()),
            "rmse": float(np.sqrt(np.square(residual).mean())),
            "nonzero_target_count": int(nonzero.sum()),
            "nonzero_target_mae": (
                float(np.abs(residual[nonzero]).mean())
                if np.any(nonzero) else None
            ),
            "hard_positive_count": int(hard_positive.sum()),
            "hard_positive_recall_at_0p5": (
                float(predicted_positive[hard_positive].mean())
                if np.any(hard_positive) else None
            ),
        }

    report = {"aggregate": {}, "cells": {}}
    for mechanism_index, mechanism in enumerate(("fragment", "shock")):
        report["aggregate"][mechanism] = metrics(
            predictions[:, mechanism_index],
            targets[:, mechanism_index],
        )
        report["cells"][mechanism] = {}
        for munition_index, munition in enumerate(MUNITION_NAMES):
            mask = munition_ids == munition_index
            report["cells"][mechanism][munition] = {}
            for task_index, task in enumerate(TASK_NAMES):
                report["cells"][mechanism][munition][task] = {
                    f"ge{level_index + 1}": metrics(
                        predictions[
                            mask, mechanism_index,
                            task_index, level_index],
                        targets[
                            mask, mechanism_index,
                            task_index, level_index],
                    )
                    for level_index in range(2)
                }
    return report


def _cell_threshold_matrices(
        p1: np.ndarray,
        p2: np.ndarray,
        true_level: np.ndarray,
        threshold_grid: np.ndarray) -> dict[str, np.ndarray]:
    """Return all GxG ordinal confusion metrics without Python row loops."""
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.minimum(
        np.asarray(p2, dtype=np.float64), p1)
    true_level = np.asarray(true_level, dtype=np.int64)
    threshold_grid = np.asarray(
        threshold_grid, dtype=np.float64)
    size = len(threshold_grid)
    supports = np.bincount(
        true_level, minlength=3).astype(np.int64)
    recalls = np.full((3, size, size), np.nan, dtype=np.float64)
    diagonal_counts = np.zeros(
        (3, size, size), dtype=np.int64)

    for class_index in range(3):
        mask = true_level == class_index
        if not np.any(mask):
            continue
        pass1 = (
            p1[mask, None] >= threshold_grid[None, :]
        ).astype(np.float64)
        pass2 = (
            p2[mask, None] >= threshold_grid[None, :]
        ).astype(np.float64)
        if class_index == 0:
            count_by_threshold1 = (
                (~pass1.astype(bool)).sum(axis=0)
            )
            counts = np.repeat(
                count_by_threshold1[:, None],
                size,
                axis=1,
            )
        elif class_index == 1:
            counts = np.rint(
                pass1.T @ (1.0 - pass2)
            ).astype(np.int64)
        else:
            counts = np.rint(
                pass1.T @ pass2
            ).astype(np.int64)
        diagonal_counts[class_index] = counts
        recalls[class_index] = counts / float(mask.sum())

    rows = max(int(supports.sum()), 1)
    accuracy = diagonal_counts.sum(axis=0) / float(rows)
    l0_false_positive_count = (
        int(supports[0]) - diagonal_counts[0])
    if int(supports[0]) > 0:
        l0_false_positive_rate = (
            l0_false_positive_count / float(supports[0]))
    else:
        l0_false_positive_rate = np.zeros(
            (size, size), dtype=np.float64)
    return {
        "supports": supports,
        "recalls": recalls,
        "diagonal_counts": diagonal_counts,
        "accuracy": accuracy,
        "l0_false_positive_count": l0_false_positive_count,
        "l0_false_positive_rate": l0_false_positive_rate,
    }


def evaluate_cell_threshold_feasibility(
        p1: np.ndarray,
        p2: np.ndarray,
        true_level: np.ndarray,
        threshold_grid: np.ndarray,
        applicable_levels: tuple[bool, bool],
        minimum_accuracy: float = 0.94,
        minimum_recall: float = 0.90,
        minimum_support: int = 100,
        maximum_l0_false_positive_rate: float | None = None,
        minimum_threshold2_slack: float = 0.10) -> dict:
    """Find a goal-feasible threshold pair or the closest safe trade-off."""
    matrices = _cell_threshold_matrices(
        p1, p2, true_level, threshold_grid)
    supports = matrices["supports"]
    recalls = matrices["recalls"]
    accuracy = matrices["accuracy"]
    l0_fp_rate = matrices["l0_false_positive_rate"]
    grid = np.asarray(threshold_grid, dtype=np.float64)

    threshold_order_valid = (
        grid[None, :] >= (
            grid[:, None] - float(minimum_threshold2_slack))
    )
    safety_valid = threshold_order_valid.copy()
    if maximum_l0_false_positive_rate is not None:
        safety_valid &= (
            l0_fp_rate
            <= float(maximum_l0_false_positive_rate) + 1e-12
        )

    applicable_classes = [0, 1]
    if bool(applicable_levels[1]):
        applicable_classes.append(2)
    metric_classes = [
        class_index
        for class_index in applicable_classes
        if int(supports[class_index]) >= int(minimum_support)
    ]
    evidence_failures = [
        {
            "class": int(class_index),
            "support": int(supports[class_index]),
            "minimum_support": int(minimum_support),
        }
        for class_index in applicable_classes
        if int(supports[class_index]) < int(minimum_support)
    ]

    goal_valid = (
        safety_valid
        & (accuracy >= float(minimum_accuracy) - 1e-12)
    )
    for class_index in metric_classes:
        goal_valid &= (
            recalls[class_index]
            >= float(minimum_recall) - 1e-12
        )

    safe_indices = np.argwhere(safety_valid)
    if safe_indices.size == 0:
        raise RuntimeError(
            "Threshold grid contains no safety-feasible pair.")

    def candidate(index1: int, index2: int) -> dict:
        return {
            "threshold_ge1": float(grid[index1]),
            "threshold_ge2": float(grid[index2]),
            "three_class_accuracy_percent": float(
                accuracy[index1, index2] * 100.0),
            "class_diagonal_recall_percent": [
                (
                    None
                    if not np.isfinite(
                        recalls[class_index, index1, index2])
                    else float(
                        recalls[
                            class_index, index1, index2] * 100.0)
                )
                for class_index in range(3)
            ],
            "l0_false_positive_count": int(
                matrices["l0_false_positive_count"][
                    index1, index2]),
            "l0_false_positive_percent": float(
                l0_fp_rate[index1, index2] * 100.0),
        }

    minimum_recall_matrix = np.ones_like(accuracy)
    if metric_classes:
        minimum_recall_matrix = np.min(
            recalls[metric_classes], axis=0)
    metric_pass_count = (
        (accuracy >= float(minimum_accuracy)).astype(np.int64)
    )
    for class_index in metric_classes:
        metric_pass_count += (
            recalls[class_index] >= float(minimum_recall)
        ).astype(np.int64)

    goal_indices = np.argwhere(goal_valid)
    if goal_indices.size:
        selected_goal = max(
            goal_indices.tolist(),
            key=lambda index: (
                -int(matrices["l0_false_positive_count"][
                    index[0], index[1]]),
                float(minimum_recall_matrix[
                    index[0], index[1]]),
                float(accuracy[index[0], index[1]]),
                float(grid[index[0]] + grid[index[1]]),
            ),
        )
        goal_candidate = candidate(*selected_goal)
    else:
        goal_candidate = None

    selected_tradeoff = max(
        safe_indices.tolist(),
        key=lambda index: (
            int(metric_pass_count[index[0], index[1]]),
            float(minimum_recall_matrix[
                index[0], index[1]]),
            float(accuracy[index[0], index[1]]),
            -float(l0_fp_rate[index[0], index[1]]),
        ),
    )
    return {
        "metric_goal_feasible": bool(goal_candidate is not None),
        "evidence_sufficient": not evidence_failures,
        "class_support": supports.astype(int).tolist(),
        "applicable_classes": applicable_classes,
        "metric_classes": metric_classes,
        "evidence_failures": evidence_failures,
        "goal_candidate": goal_candidate,
        "best_safe_tradeoff": candidate(*selected_tradeoff),
    }


def _verify_model_artifacts(
        run_dir: Path) -> tuple[dict, Path, Path]:
    model_dir = run_dir / "output" / "models"
    manifest_path = model_dir / "model_manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != "stage0_nn_artifact_v1":
        raise RuntimeError("Unsupported model manifest schema.")
    checkpoint_path = model_dir / "best_model.pth"
    scaler_path = model_dir / "minmax_scaler.pkl"
    for path, key in (
            (checkpoint_path, "best_model.pth"),
            (scaler_path, "minmax_scaler.pkl")):
        expected = manifest["artifacts"][key]["sha256"]
        observed = _sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"Artifact SHA-256 mismatch for {path}: "
                f"expected={expected}, observed={observed}")
    dataset_path = Path(
        manifest["data_contract"]["dataset_path"]).resolve()
    if _sha256(dataset_path) != (
            manifest["data_contract"]["dataset_sha256"]):
        raise RuntimeError(
            "Dataset SHA-256 does not match model manifest.")
    return manifest, checkpoint_path, scaler_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only exhaustive threshold feasibility audit for the "
            "94% cell-accuracy, 90% class-recall business goal."))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold-step", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--probability-source",
        choices=(
            "deployed",
            "direct",
            "predicted_component_tree",
            "target_component_tree",
            "terminal_rule_proxy",
            "terminal_combined_rule_proxy",
            "terminal_component_proxy_tree",
            "predicted_mechanism_fragment",
            "predicted_mechanism_shock",
            "predicted_mechanism_or",
            "predicted_fragment_target_shock_or",
            "target_fragment_predicted_shock_or",
        ),
        default="deployed",
        help=(
            "Use the deployed model output, its frozen direct path, the "
            "predicted component tree, the validation-only target "
            "component-tree ceiling, predicted physical mechanism experts, "
            "or deterministic deployable terminal-physics rule proxies."),
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if not 0.0 < args.threshold_step <= 0.25:
        raise ValueError("threshold-step must be in (0,0.25].")

    run_dir = Path(args.run_dir).resolve()
    manifest, checkpoint_path, scaler_path = (
        _verify_model_artifacts(run_dir))
    feature_names = list(
        manifest["data_contract"]["feature_names"])
    dataset_path = Path(
        manifest["data_contract"]["dataset_path"]).resolve()
    terminal_contract = manifest["data_contract"].get(
        "terminal_physics_contract") or {}
    armor_aware_fragment_proxies = any(
        bool(extension.get("armor_aware_fragment_proxies", False))
        for extension in terminal_contract.get("extensions", [])
        if isinstance(extension, dict)
    )
    validation = _load_validation_frame(
        dataset_path,
        feature_names,
        armor_aware_fragment_proxies=(
            armor_aware_fragment_proxies),
    )
    with scaler_path.open("rb") as stream:
        scaler = pickle.load(stream)
    features = scaler.transform(
        validation[feature_names].to_numpy(
            dtype=np.float32)).astype(np.float32)
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
    model = None
    if args.probability_source not in {
            "terminal_rule_proxy",
            "terminal_combined_rule_proxy",
            "terminal_component_proxy_tree"}:
        model = DamageAssessmentMTL(
            **manifest["model_config"]).to(device)
        try:
            state = torch.load(
                checkpoint_path, map_location=device,
                weights_only=True)
        except TypeError:
            state = torch.load(
                checkpoint_path, map_location=device)
        model.load_state_dict(state, strict=True)
    mechanism_fit = None
    if args.probability_source == "terminal_component_proxy_tree":
        validation = augment_terminal_physics_features(
            validation,
            copy=False,
            include_component_proxies=True,
            armor_aware_fragment_proxies=(
                armor_aware_fragment_proxies),
        )
        component_proxy = validation[
            COMPONENT_PROXY_FEATURE_COLUMNS
        ].to_numpy(dtype=np.float32).reshape(
            len(validation), 2, -1)
        probabilities = _component_tree(component_proxy)
    elif args.probability_source == "terminal_combined_rule_proxy":
        validation = augment_terminal_physics_features(
            validation,
            copy=False,
            include_combined_rule_proxies=True,
        )
        probabilities = (
            _terminal_combined_rule_proxy_probabilities(
                validation)
        )
    elif args.probability_source == "terminal_rule_proxy":
        probabilities = _terminal_rule_proxy_probabilities(
            validation)
    elif args.probability_source == "deployed":
        probabilities = _predict_probabilities(
            model, features, munition_ids,
            args.batch_size, device)
    elif (
        args.probability_source.startswith("predicted_mechanism_")
        or args.probability_source in {
            "predicted_fragment_target_shock_or",
            "target_fragment_predicted_shock_or",
        }
    ):
        if not bool(
                manifest["model_config"].get(
                    "use_mechanism_decomposition", False)
                or manifest["model_config"].get(
                    "use_mechanism_auxiliary_heads", False)):
            raise RuntimeError(
                f"{args.probability_source} requires mechanism outputs.")
        mechanism_or, fragment, shock = (
            _predict_mechanism_probabilities(
                model,
                features,
                munition_ids,
                args.batch_size,
                device,
            )
        )
        mechanism_targets = _load_validation_mechanism_targets(
            dataset_path,
            validation["sample_id"].astype(str).to_numpy(),
        )
        target_fragment = mechanism_targets[:, 0]
        target_shock = mechanism_targets[:, 1]
        predicted_mechanisms = np.stack(
            (fragment, shock), axis=1)
        mechanism_fit = _mechanism_fit_metrics(
            predicted_mechanisms,
            mechanism_targets,
            munition_ids,
        )
        probabilities = {
            "predicted_mechanism_fragment": fragment,
            "predicted_mechanism_shock": shock,
            "predicted_mechanism_or": mechanism_or,
            "predicted_fragment_target_shock_or": (
                1.0 - (1.0 - fragment) * (1.0 - target_shock)
            ),
            "target_fragment_predicted_shock_or": (
                1.0 - (1.0 - target_fragment) * (1.0 - shock)
            ),
        }[args.probability_source]
    else:
        if not bool(
                manifest["model_config"].get(
                    "use_component_auxiliary_heads", False)):
            raise RuntimeError(
                f"{args.probability_source} requires component outputs.")
        direct, _, component_predictions = _predict(
            model,
            features,
            munition_ids,
            args.batch_size,
            device,
        )
        if args.probability_source == "direct":
            probabilities = direct
        elif args.probability_source == "predicted_component_tree":
            probabilities = _component_tree(
                component_predictions)
        else:
            component_contract = manifest["data_contract"].get(
                "component_supervision_contract")
            if not isinstance(component_contract, dict):
                raise RuntimeError(
                    "Model contract lacks component supervision metadata.")
            component_path = Path(
                component_contract["path"]).resolve()
            if _sha256(component_path) != (
                    component_contract["sha256"]):
                raise RuntimeError(
                    "Component sidecar SHA-256 mismatch.")
            component_targets = _load_validation_components(
                component_path,
                validation["sample_id"].astype(str).to_numpy(),
                list(component_contract["target_columns"]),
            )
            probabilities = _component_tree(
                component_targets)
    threshold_grid = np.arange(
        float(args.threshold_step),
        1.0 + 0.5 * float(args.threshold_step),
        float(args.threshold_step),
        dtype=np.float64,
    )

    cells = {}
    all_metric_feasible = True
    all_evidence_sufficient = True
    c_goal_minimum_false_positives = []
    c_l0_support = 0
    for munition_index, munition_name in enumerate(
            MUNITION_NAMES):
        mask = munition_ids == munition_index
        cells[munition_name] = {}
        for task_index, task_name in enumerate(TASK_NAMES):
            special_cap = (
                0.005
                if munition_index == 0 and task_index == 0
                else None
            )
            result = evaluate_cell_threshold_feasibility(
                probabilities[mask, task_index, 0],
                probabilities[mask, task_index, 1],
                levels[mask, task_index],
                threshold_grid,
                tuple(
                    DEFAULT_ORDINAL_APPLICABILITY[
                        munition_index][task_index]),
                maximum_l0_false_positive_rate=special_cap,
            )
            cells[munition_name][task_name] = result
            all_metric_feasible &= bool(
                result["metric_goal_feasible"])
            all_evidence_sufficient &= bool(
                result["evidence_sufficient"])
            if task_index == 3:
                c_l0_support += int(
                    result["class_support"][0])
                if result["goal_candidate"] is not None:
                    c_goal_minimum_false_positives.append(
                        int(result["goal_candidate"][
                            "l0_false_positive_count"]))

    global_c0_budget = int(math.floor(
        0.025 * c_l0_support + 1e-12))
    c_metric_cells_feasible = (
        len(c_goal_minimum_false_positives)
        == len(MUNITION_NAMES)
    )
    c_minimum_false_positives = (
        int(sum(c_goal_minimum_false_positives))
        if c_metric_cells_feasible else None
    )
    global_c0_feasible = bool(
        c_metric_cells_feasible
        and c_minimum_false_positives <= global_c0_budget
    )
    all_metric_feasible &= global_c0_feasible

    payload = {
        "schema": REPORT_SCHEMA,
        "status": "COMPLETE",
        "split": "validation",
        "test_labels_used": False,
        "validation_scan_predicate": "split_role == 'val'",
        "run_dir": str(run_dir),
        "probability_source": str(
            args.probability_source),
        "dataset_sha256": _sha256(dataset_path),
        "model_sha256": _sha256(checkpoint_path),
        "validation_rows": int(len(validation)),
        "requirements": {
            "minimum_cell_3class_accuracy_percent": 94.0,
            "minimum_applicable_class_diagonal_recall_percent": 90.0,
            "minimum_class_support": 100,
            "small_k0_max_false_positive_percent": 0.5,
            "global_c0_max_false_positive_percent": 2.5,
        },
        "threshold_grid": {
            "minimum": float(threshold_grid[0]),
            "maximum": float(threshold_grid[-1]),
            "step": float(args.threshold_step),
            "pairs_per_cell": int(
                len(threshold_grid) ** 2),
        },
        "metric_goal_threshold_feasible": bool(
            all_metric_feasible),
        "evidence_sufficient": bool(
            all_evidence_sufficient),
        "global_c0_joint_feasibility": {
            "l0_support": int(c_l0_support),
            "maximum_false_positive_count": int(
                global_c0_budget),
            "minimum_false_positive_count_for_local_goals": (
                c_minimum_false_positives),
            "feasible": global_c0_feasible,
        },
        "mechanism_fit_metrics": mechanism_fit,
        "cells": cells,
    }
    output_path = Path(args.output).resolve()
    _write_json_atomic(output_path, payload)
    print(json.dumps({
        "status": payload["status"],
        "probability_source": payload[
            "probability_source"],
        "metric_goal_threshold_feasible": payload[
            "metric_goal_threshold_feasible"],
        "evidence_sufficient": payload[
            "evidence_sufficient"],
        "global_c0_joint_feasibility": payload[
            "global_c0_joint_feasibility"],
        "infeasible_cells": [
            f"{munition}/{task}"
            for munition, tasks in cells.items()
            for task, result in tasks.items()
            if not result["metric_goal_feasible"]
        ],
        "evidence_insufficient_cells": [
            f"{munition}/{task}"
            for munition, tasks in cells.items()
            for task, result in tasks.items()
            if not result["evidence_sufficient"]
        ],
        "output": str(output_path),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
