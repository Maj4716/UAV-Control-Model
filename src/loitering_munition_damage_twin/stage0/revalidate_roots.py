"""High-MC, read-only revalidation of selected Stage-0 root families.

This diagnostic intentionally does not rewrite the source Parquet.  It selects
at most one representative row per independent root family, replays the same
immutable sample lineage with a fixed Monte-Carlo count, and writes a compact
JSON report.  The primary use case is distinguishing a genuinely reachable
rare ordinal cell from a low-replicate false positive before a costly rebuild.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
import os
from pathlib import Path
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from loitering_munition_damage_twin.stage0.component_supervision import sha256_file
from loitering_munition_damage_twin.stage0.generation import CONFIG, _init_worker, _process_single_encounter
from loitering_munition_damage_twin.simulation.engine import load_armor_plates, load_vehicle_model


REPORT_SCHEMA = "stage0_root_revalidation_v1"
DEFAULT_INPUT = "output/damage_dataset.parquet"
DEFAULT_OUTPUT = "output/med_lm_c2_root_revalidation.json"
TASK_INDEX = {"K": 0, "M": 1, "F": 2, "C": 3}


def _atomic_json_dump(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(str(temporary), str(path))


def _select_independent_roots(
    path: Path,
    munition_id: int,
    score_column: str,
    root_count: int,
) -> pd.DataFrame:
    schema_names = set(pq.read_schema(path).names)
    required = {
        "sample_id", "root_seed_id", "m_id", "x", "y", "z",
        "vx", "vy", "vz", "pitch", "roll", "yaw", score_column,
    }
    missing = sorted(required - schema_names)
    if missing:
        raise RuntimeError(f"Source dataset is missing columns: {missing}")

    # Read only the requested munition partition.  PyArrow can push this filter
    # down when row-group statistics are available; no source state is changed.
    table = pq.read_table(
        path,
        filters=[("m_id", "=", int(munition_id))],
    )
    frame = table.to_pandas()
    frame[score_column] = pd.to_numeric(
        frame[score_column], errors="coerce")
    frame = frame[np.isfinite(frame[score_column].to_numpy(dtype=float))]
    if frame.empty:
        raise RuntimeError(
            f"No finite {score_column} rows for m_id={munition_id}.")
    frame["root_seed_id"] = frame["root_seed_id"].astype(str)
    frame["sample_id"] = frame["sample_id"].astype(str)
    selected = (
        frame.sort_values(
            [score_column, "sample_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        .drop_duplicates("root_seed_id", keep="first")
        .head(int(root_count))
        .copy()
    )
    if len(selected) < int(root_count):
        raise RuntimeError(
            f"Requested {root_count} roots, only {len(selected)} available.")
    return selected.reset_index(drop=True)


def _prepare_tasks(
    selected: pd.DataFrame,
    fixed_replicates: int,
) -> List[tuple[int, Dict[str, Any]]]:
    tasks: List[tuple[int, Dict[str, Any]]] = []
    for index, row in enumerate(selected.to_dict("records")):
        # Supplying equal adaptive bounds forces every selected sample to use
        # exactly this many replicates while retaining antithetic pairing.
        row["label_mc_min_replicates"] = int(fixed_replicates)
        row["label_mc_max_replicates"] = int(fixed_replicates)
        tasks.append((index, row))
    return tasks


def _normal_interval(
    mean: float,
    replicate_std: float,
    replicates: int,
    z_value: float,
) -> tuple[float, float, float]:
    standard_error = float(replicate_std) / math.sqrt(int(replicates))
    return (
        max(0.0, float(mean) - float(z_value) * standard_error),
        min(1.0, float(mean) + float(z_value) * standard_error),
        standard_error,
    )


def run(args: argparse.Namespace) -> int:
    started = time.time()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if int(args.replicates) <= 0:
        raise ValueError("--replicates must be positive")
    if bool(CONFIG.get("LABEL_MC_ANTITHETIC", False)) and (
        int(args.replicates) % 2 != 0
    ):
        raise ValueError(
            "--replicates must be even when antithetic MC is enabled")

    task = str(args.task).upper()
    if task not in TASK_INDEX:
        raise ValueError(f"Unsupported task: {task}")
    level = int(args.level)
    if level not in (1, 2):
        raise ValueError("--level must be 1 or 2")
    score_column = args.score_column or f"{task}_ge{level}_prob"
    std_column = f"{task}_ge{level}_prob_std"

    selected = _select_independent_roots(
        input_path,
        int(args.munition_id),
        score_column,
        int(args.roots),
    )
    original_by_sample = selected.set_index("sample_id", drop=False)
    workers = (
        max(1, multiprocessing.cpu_count() - 1)
        if int(args.workers) <= 0 else int(args.workers)
    )
    print(
        f"[REVALIDATE] START | m_id={int(args.munition_id)} | "
        f"target={task}>={level} | roots={len(selected)} | "
        f"fixed_MC={int(args.replicates)} | workers={workers}",
        flush=True,
    )
    components = load_vehicle_model()
    plates = load_armor_plates()
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(components, plates),
    ) as executor:
        replay_rows = list(executor.map(
            _process_single_encounter,
            _prepare_tasks(selected, int(args.replicates)),
            chunksize=1,
        ))

    replay = pd.DataFrame(replay_rows)
    if set(replay["sample_id"].astype(str)) != set(
        selected["sample_id"].astype(str)
    ):
        raise RuntimeError("Replay sample lineage differs from selection.")

    threshold = float(args.threshold)
    details: List[Dict[str, Any]] = []
    for row in replay.to_dict("records"):
        sample_id = str(row["sample_id"])
        source = original_by_sample.loc[sample_id]
        replay_mean = float(row[score_column])
        low, high, standard_error = _normal_interval(
            replay_mean,
            float(row[std_column]),
            int(row["label_mc_replicates"]),
            float(args.z_value),
        )
        details.append({
            "sample_id": sample_id,
            "root_seed_id": str(source["root_seed_id"]),
            "split_role": str(source.get("split_role", "unknown")),
            "original_probability": float(source[score_column]),
            "original_mc_replicates": int(
                source.get("label_mc_replicates", 0)),
            "revalidated_probability": replay_mean,
            "revalidated_replicate_std": float(row[std_column]),
            "revalidated_standard_error": standard_error,
            "normal_interval_low": low,
            "normal_interval_high": high,
            "positive_at_threshold": bool(replay_mean >= threshold),
            "interval_above_threshold": bool(low >= threshold),
            "fragment_probability": float(
                row[f"fragment_{task}_ge{level}_prob"]),
            "shock_probability": float(
                row[f"shock_{task}_ge{level}_prob"]),
            "label_mc_all_resolved": bool(row["label_mc_all_resolved"]),
        })
    details.sort(
        key=lambda item: item["original_probability"], reverse=True)

    original_scores = np.asarray([
        item["original_probability"] for item in details], dtype=float)
    replay_scores = np.asarray([
        item["revalidated_probability"] for item in details], dtype=float)
    positive_count = int(np.sum(replay_scores >= threshold))
    robust_count = int(np.sum([
        item["interval_above_threshold"] for item in details]))
    report: Dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "COMPLETE",
        "source": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
        },
        "selection": {
            "munition_id": int(args.munition_id),
            "target": f"{task}>={level}",
            "score_column": score_column,
            "independent_root_policy": "highest_score_row_per_root",
            "requested_roots": int(args.roots),
            "selected_roots": int(len(details)),
        },
        "revalidation": {
            "fixed_replicates": int(args.replicates),
            "antithetic": bool(CONFIG.get("LABEL_MC_ANTITHETIC", False)),
            "threshold": threshold,
            "normal_interval_z": float(args.z_value),
            "positive_roots": positive_count,
            "interval_confirmed_positive_roots": robust_count,
            "all_mc_resolved_roots": int(sum(
                item["label_mc_all_resolved"] for item in details)),
            "original_probability_min": float(original_scores.min()),
            "original_probability_max": float(original_scores.max()),
            "revalidated_probability_min": float(replay_scores.min()),
            "revalidated_probability_max": float(replay_scores.max()),
            "revalidated_probability_mean": float(replay_scores.mean()),
            "pearson_original_vs_revalidated": (
                float(np.corrcoef(original_scores, replay_scores)[0, 1])
                if len(details) >= 2
                and np.std(original_scores) > 0
                and np.std(replay_scores) > 0
                else None
            ),
        },
        "elapsed_seconds": float(time.time() - started),
        "roots": details,
    }
    _atomic_json_dump(report, output_path)
    print(
        f"[REVALIDATE] COMPLETE | positive={positive_count}/{len(details)} | "
        f"interval_confirmed={robust_count}/{len(details)} | "
        f"max={replay_scores.max():.4f} | "
        f"elapsed={time.time() - started:.1f}s | output={output_path}",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Revalidate the highest-scoring independent Stage-0 roots with "
            "a fixed high Monte-Carlo count without changing the dataset."))
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--munition-id", type=int, default=1)
    parser.add_argument("--task", choices=list(TASK_INDEX), default="C")
    parser.add_argument("--level", type=int, choices=(1, 2), default=2)
    parser.add_argument("--score-column", default=None)
    parser.add_argument("--roots", type=int, default=32)
    parser.add_argument("--replicates", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--z-value", type=float, default=1.96)
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if int(args.roots) <= 0:
        parser.error("--roots must be positive")
    if int(args.replicates) <= 0:
        parser.error("--replicates must be positive")
    if not 0.0 < float(args.threshold) < 1.0:
        parser.error("--threshold must be in (0, 1)")
    if float(args.z_value) <= 0.0:
        parser.error("--z-value must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
