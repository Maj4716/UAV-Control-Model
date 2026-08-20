from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]

from loitering_munition_damage_twin.stage0.generation import _init_worker, _process_single_encounter
from loitering_munition_damage_twin.surrogate.artifacts import sha256_file
from loitering_munition_damage_twin.simulation.engine import load_armor_plates, load_vehicle_model


MUNITION_NAMES = ("Small", "Med-LM", "Med-RD", "Heavy")
TASK_NAMES = ("K", "M", "F", "C")

# The audit is intentionally validation-only.  It covers every historically
# weak exact-L1 cell and the adjacent competing classes.
AUDIT_STRATA = (
    (0, "K", (0, 1)),
    (0, "C", (0, 1)),
    (1, "K", (0, 1, 2)),
    (1, "C", (0, 1, 2)),
    (2, "K", (0, 1, 2)),
    (2, "M", (0, 1, 2)),
    (2, "C", (0, 1, 2)),
    (3, "K", (0, 1, 2)),
)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def _select_rows(
        frame: pd.DataFrame,
        rows_per_class: int,
        seed: int) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    selected_indices: set[int] = set()
    requested = {}
    for munition_id, task_name, levels in AUDIT_STRATA:
        for level in levels:
            mask = (
                frame["m_id"].eq(munition_id)
                & frame[f"{task_name}_level"].eq(level)
            )
            candidates = frame.index[mask].to_numpy(dtype=np.int64)
            count = min(int(rows_per_class), len(candidates))
            if count:
                chosen = rng.choice(
                    candidates, size=count, replace=False)
                selected_indices.update(int(value) for value in chosen)
            key = f"{MUNITION_NAMES[munition_id]}/{task_name}/L{level}"
            requested[key] = {
                "available": int(len(candidates)),
                "requested": int(rows_per_class),
                "selected_before_deduplication": int(count),
            }
    selected = frame.loc[sorted(selected_indices)].copy()
    return selected, requested


def _comparison_summary(
        original: pd.DataFrame,
        repeated: pd.DataFrame) -> dict:
    repeated_by_id = repeated.set_index("audit_original_sample_id")
    original_id_column = (
        "audit_original_sample_id"
        if "audit_original_sample_id" in original.columns
        else "sample_id"
    )
    original_by_id = original.set_index(original_id_column)
    if set(repeated_by_id.index) != set(original_by_id.index):
        raise RuntimeError(
            "Repeated-label output does not match selected sample IDs.")

    result = {}
    for munition_id, task_name, levels in AUDIT_STRATA:
        munition_name = MUNITION_NAMES[munition_id]
        for level in levels:
            original_mask = (
                original_by_id["m_id"].eq(munition_id)
                & original_by_id[f"{task_name}_level"].eq(level)
            )
            ids = original_by_id.index[original_mask]
            key = f"{munition_name}/{task_name}/L{level}"
            if len(ids) == 0:
                result[key] = {"support": 0}
                continue
            old_level = original_by_id.loc[
                ids, f"{task_name}_level"].to_numpy(dtype=np.int64)
            new_level = repeated_by_id.loc[
                ids, f"{task_name}_level"].to_numpy(dtype=np.int64)
            old_ge1 = original_by_id.loc[
                ids, f"{task_name}_ge1_prob"].to_numpy(dtype=np.float64)
            new_ge1 = repeated_by_id.loc[
                ids, f"{task_name}_ge1_prob"].to_numpy(dtype=np.float64)
            old_ge2 = original_by_id.loc[
                ids, f"{task_name}_ge2_prob"].to_numpy(dtype=np.float64)
            new_ge2 = repeated_by_id.loc[
                ids, f"{task_name}_ge2_prob"].to_numpy(dtype=np.float64)
            transitions = np.zeros((3, 3), dtype=np.int64)
            for before, after in zip(old_level, new_level):
                transitions[int(before), int(after)] += 1
            result[key] = {
                "support": int(len(ids)),
                "exact_level_agreement_percent": float(
                    np.mean(old_level == new_level) * 100.0),
                "transition_counts_rows_old_columns_new": (
                    transitions.tolist()),
                "mean_absolute_ge1_probability_change": float(
                    np.mean(np.abs(old_ge1 - new_ge1))),
                "mean_absolute_ge2_probability_change": float(
                    np.mean(np.abs(old_ge2 - new_ge2))),
                "new_level_histogram": np.bincount(
                    new_level, minlength=3).astype(int).tolist(),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate a validation-only stratified sample with independent "
            "higher-replicate Monte Carlo labels."))
    parser.add_argument(
        "--data", default="output/damage_dataset.parquet")
    parser.add_argument("--rows-per-class", type=int, default=20)
    parser.add_argument("--replicates", type=int, default=32)
    parser.add_argument("--workers", type=int, default=max(
        1, min(12, os.cpu_count() or 1)))
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--output",
        default="output/experiments/label_reproducibility_audit.json")
    args = parser.parse_args()
    if args.rows_per_class <= 0 or args.replicates < 2:
        raise ValueError(
            "rows-per-class must be positive and replicates at least 2.")

    dataset_path = Path(args.data).resolve()
    required = [
        "x", "y", "z", "vx", "vy", "vz",
        "pitch", "roll", "yaw", "m_id",
        "sample_id", "root_seed_id", "split_role",
        "K_level", "M_level", "F_level", "C_level",
    ] + [
        f"{task}_ge{level}_prob"
        for task in TASK_NAMES for level in (1, 2)
    ]
    frame = pd.read_parquet(
        dataset_path, columns=required, engine="pyarrow")
    validation = frame.loc[frame["split_role"].eq("val")].copy()
    selected, requested = _select_rows(
        validation, args.rows_per_class, args.seed)
    if selected.empty:
        raise RuntimeError("No validation rows were selected.")

    selected["audit_original_sample_id"] = selected["sample_id"].astype(str)
    selected["sample_id"] = (
        "label-reproducibility-v1|"
        + selected["audit_original_sample_id"])
    selected["label_mc_min_replicates"] = int(args.replicates)
    selected["label_mc_max_replicates"] = int(args.replicates)

    components = load_vehicle_model()
    plates = load_armor_plates()
    records = selected.to_dict("records")
    with ProcessPoolExecutor(
        max_workers=int(args.workers),
        initializer=_init_worker,
        initargs=(components, plates),
    ) as pool:
        repeated_rows = list(pool.map(
            _process_single_encounter,
            enumerate(records),
            chunksize=1,
        ))
    repeated = pd.DataFrame(repeated_rows)
    repeated["audit_original_sample_id"] = [
        record["audit_original_sample_id"] for record in records
    ]

    strata = _comparison_summary(selected, repeated)
    l1_agreements = [
        item["exact_level_agreement_percent"]
        for key, item in strata.items()
        if key.endswith("/L1") and item.get("support", 0) > 0
    ]
    result = {
        "schema": "stage0_label_reproducibility_audit_v1",
        "status": "COMPLETE",
        "split": "validation",
        "test_labels_used": False,
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(str(dataset_path)),
        "selection_seed": int(args.seed),
        "independent_seed_namespace": "label-reproducibility-v1",
        "original_replicate_range": [
            3, 9
        ],
        "audit_replicates": int(args.replicates),
        "unique_rows": int(len(selected)),
        "requested_strata": requested,
        "strata": strata,
        "summary": {
            "weak_l1_strata": int(len(l1_agreements)),
            "mean_weak_l1_exact_level_agreement_percent": float(
                np.mean(l1_agreements)),
            "minimum_weak_l1_exact_level_agreement_percent": float(
                np.min(l1_agreements)),
            "maximum_weak_l1_exact_level_agreement_percent": float(
                np.max(l1_agreements)),
        },
    }
    output_path = Path(args.output).resolve()
    _write_json_atomic(output_path, result)
    print(json.dumps({
        "status": result["status"],
        "unique_rows": result["unique_rows"],
        "audit_replicates": result["audit_replicates"],
        "summary": result["summary"],
        "output": str(output_path),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
