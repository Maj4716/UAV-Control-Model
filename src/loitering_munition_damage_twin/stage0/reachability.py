"""Probe rare Stage-0 label reachability with independent physical-simulation roots.

This utility never writes or modifies the production dataset.  It exercises the
same targeted fresh-root generator used by ``generate_dataset.py`` and records
whether rare ordinal cells are physically reachable under the current model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from loitering_munition_damage_twin.stage0.generation import (
    CONFIG,
    PhysicsAwareSampler,
    _target_score,
    _target_seed_mask,
    _target_valid_mask,
    _validate_generation_config,
)


DEFAULT_TARGETS: Dict[str, Tuple[int, str]] = {
    "small_c1": (0, "C1_only"),
    "med_lm_k2": (1, "K2_prob"),
    "med_lm_c2": (1, "C2_prob"),
}


def _effective_root_count(counts: pd.Series) -> float:
    if counts.empty:
        return 0.0
    values = counts.to_numpy(dtype=float)
    return float(values.sum() ** 2 / np.square(values).sum())


def _root_statistics(df: pd.DataFrame, target_col: str) -> dict:
    if df.empty:
        return {
            "rows_simulated": 0,
            "seed_rows": 0,
            "strict_positive_rows": 0,
            "seed_roots": 0,
            "strict_positive_roots": 0,
            "effective_positive_roots": 0.0,
            "largest_positive_root_share": None,
            "score_max": None,
            "score_p99": None,
            "highest_scoring_roots": [],
        }

    seed_mask = _target_seed_mask(
        df, target_col, CONFIG["SEED_PROB_RELAX"], CONFIG["VALID_PROB_STRICT"])
    strict_mask = _target_valid_mask(df, target_col, CONFIG["VALID_PROB_STRICT"])
    root_col = df["root_seed_id"].astype(str)
    positive_counts = root_col[strict_mask].value_counts()
    score = _target_score(df, target_col).astype(float)
    ranked = df.assign(_probe_score=score).sort_values(
        "_probe_score", ascending=False).drop_duplicates(
        "root_seed_id", keep="first").head(10)
    detail_columns = [
        "sample_id", "root_seed_id", "x", "y", "z",
        "vx", "vy", "vz", "target_x", "target_y", "target_z",
        "pitch", "roll", "yaw", "sampling_geometry",
        "label_mc_replicates", "_probe_score",
    ]
    highest_scoring_roots = []
    for row in ranked[
        [name for name in detail_columns if name in ranked.columns]
    ].to_dict("records"):
        highest_scoring_roots.append({
            name: (
                value.item() if isinstance(value, np.generic) else value)
            for name, value in row.items()
        })
    return {
        "rows_simulated": int(len(df)),
        "seed_rows": int(seed_mask.sum()),
        "strict_positive_rows": int(strict_mask.sum()),
        "seed_roots": int(root_col[seed_mask].nunique()),
        "strict_positive_roots": int(root_col[strict_mask].nunique()),
        "effective_positive_roots": float(_effective_root_count(positive_counts)),
        "largest_positive_root_share": (
            float(positive_counts.max() / positive_counts.sum())
            if not positive_counts.empty else None
        ),
        "score_max": float(score.max()),
        "score_p99": float(np.quantile(score, 0.99)),
        "highest_scoring_roots": highest_scoring_roots,
    }


def _write_json_atomic(path: str, payload: dict) -> None:
    absolute = os.path.abspath(path)
    os.makedirs(os.path.dirname(absolute), exist_ok=True)
    temporary = absolute + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, absolute)


def _two_stage_discovery(
        sampler: PhysicsAwareSampler,
        munition_id: int,
        target_col: str,
        max_candidates: int,
        batch_size: int,
        required_roots: int,
        screen_replicates: int,
        screen_threshold: float,
        progress_path: str) -> tuple[pd.DataFrame, dict]:
    """Cheaply screen roots, then count only production-MC confirmations."""
    target_layer = (
        "K2_HUNT" if target_col == "K2_prob" else
        "K1_HUNT" if target_col.startswith("K") else
        "M_HUNT" if target_col.startswith("M") else
        "F_HUNT" if target_col.startswith("F") else
        "C2_HUNT" if target_col.startswith("C2") else
        "C_HUNT"
    )
    confirmed_batches = []
    requested_total = 0
    screened_total = 0
    confirmation_total = 0
    rounds = 0
    strict_roots = 0
    screening_rounds = []

    while requested_total < max_candidates and strict_roots < required_roots:
        request_n = min(batch_size, max_candidates - requested_total)
        if request_n <= 0:
            break
        rounds += 1
        requested_total += request_n
        print(
            f"  [ProbeScreen][{rounds}] m_id={munition_id} {target_col}: "
            f"strict roots={strict_roots}/{required_roots}; "
            f"screen {request_n} roots with {screen_replicates} MC.",
            flush=True,
        )
        fresh_inputs = sampler._generate_lhs_batch(
            request_n,
            target_layer,
            force_m_id=munition_id,
            exterior_lateral_shell=True,
        )
        fresh_inputs = sampler._apply_phase1_filters_and_weights(
            fresh_inputs)
        if fresh_inputs.empty:
            continue
        fresh_inputs = fresh_inputs[
            fresh_inputs["split_role"].astype(str) == "train"
        ].copy()
        if fresh_inputs.empty:
            continue
        fresh_inputs["sampling_phase"] = "reachability_probe"

        screen_inputs = fresh_inputs.drop(
            columns=[
                "label_mc_min_replicates",
                "label_mc_max_replicates",
            ],
            errors="ignore",
        ).copy()
        screen_inputs["label_mc_replicates"] = int(screen_replicates)
        screened = sampler.run_simulation_batch(screen_inputs)
        screened_total += int(len(screened))
        if screened.empty:
            continue

        scores = np.asarray(
            _target_score(screened, target_col), dtype=float)
        missing = max(required_roots - strict_roots, 1)
        confirmation_cap = min(
            len(screened), max(16, 3 * missing))
        relaxed_indices = np.flatnonzero(
            scores >= float(screen_threshold))
        ranked_indices = np.argsort(
            -scores, kind="stable")[:confirmation_cap]
        selected_indices = np.unique(np.concatenate((
            relaxed_indices,
            ranked_indices,
        )))
        if len(selected_indices) > confirmation_cap:
            selected_indices = selected_indices[
                np.argsort(
                    -scores[selected_indices],
                    kind="stable",
                )[:confirmation_cap]
            ]
        selected_ids = set(
            screened.iloc[selected_indices]["sample_id"].astype(str))
        selected_scores = scores[selected_indices]
        screening_rounds.append({
            "round": int(rounds),
            "screened_rows": int(len(screened)),
            "screen_score_max": float(np.max(scores)),
            "screen_score_p99": float(np.quantile(scores, 0.99)),
            "selected_rows": int(len(selected_indices)),
            "selected_screen_score_min": float(np.min(selected_scores)),
            "selected_screen_score_max": float(np.max(selected_scores)),
        })
        confirmation_inputs = fresh_inputs[
            fresh_inputs["sample_id"].astype(str).isin(selected_ids)
        ].copy()
        confirmation_inputs["label_mc_min_replicates"] = int(
            CONFIG["LABEL_MC_MIN_REPLICATES"])
        confirmation_inputs["label_mc_max_replicates"] = int(
            CONFIG["LABEL_MC_MAX_REPLICATES"])
        confirmation_inputs = confirmation_inputs.drop(
            columns=["label_mc_replicates"], errors="ignore")
        print(
            f"  [ProbeConfirm][{rounds}] formally re-evaluate "
            f"{len(confirmation_inputs)} selected roots with adaptive "
            f"{CONFIG['LABEL_MC_MIN_REPLICATES']}-"
            f"{CONFIG['LABEL_MC_MAX_REPLICATES']} MC.",
            flush=True,
        )
        confirmed = sampler.run_simulation_batch(confirmation_inputs)
        confirmation_total += int(len(confirmed))
        if not confirmed.empty:
            confirmed_batches.append(confirmed)
        combined = (
            pd.concat(confirmed_batches, ignore_index=True)
            if confirmed_batches else pd.DataFrame()
        )
        stats = _root_statistics(combined, target_col)
        strict_roots = int(stats["strict_positive_roots"])
        _write_json_atomic(progress_path, {
            "status": "RUNNING",
            "munition_id": int(munition_id),
            "target_column": target_col,
            "rounds": int(rounds),
            "raw_candidates_requested": int(requested_total),
            "screen_rows_simulated": int(screened_total),
            "formal_confirmation_rows": int(confirmation_total),
            "strict_positive_roots": int(strict_roots),
            "required_strict_positive_roots": int(required_roots),
        })

    combined = (
        pd.concat(confirmed_batches, ignore_index=True)
        if confirmed_batches else pd.DataFrame()
    )
    discovery = {
        "mode": "two_stage_screen_then_production_mc_confirmation",
        "target_layer": target_layer,
        "split_scope": "train",
        "rounds": int(rounds),
        "raw_candidates_requested": int(requested_total),
        "screen_rows_simulated": int(screened_total),
        "formal_confirmation_rows": int(confirmation_total),
        "screen_replicates": int(screen_replicates),
        "screen_threshold": float(screen_threshold),
        "screening_rounds": screening_rounds,
        "strict_roots_after": int(strict_roots),
        "desired_strict_roots": int(required_roots),
        "candidate_budget_exhausted": bool(
            requested_total >= max_candidates
            and strict_roots < required_roots),
    }
    _write_json_atomic(progress_path, {
        "status": (
            "COMPLETE" if strict_roots >= required_roots
            else "BUDGET_EXHAUSTED"),
        **discovery,
    })
    return combined, discovery


def run_probe(target_names: list[str], max_candidates: int,
              batch_size: int, required_roots: int, output_path: str,
              screen_replicates: int = 2,
              screen_threshold: float = 0.10) -> dict:
    unknown = sorted(set(target_names) - set(DEFAULT_TARGETS))
    if unknown:
        raise ValueError(f"未知目标 {unknown}；可选值为 {sorted(DEFAULT_TARGETS)}")
    if min(max_candidates, batch_size, required_roots) <= 0:
        raise ValueError("候选数、批大小和目标 root 数必须均为正数。")
    if screen_replicates <= 0:
        raise ValueError("screen_replicates 必须为正数。")
    if not 0.0 <= screen_threshold < 0.5:
        raise ValueError("screen_threshold 必须位于 [0,0.5)。")

    # The probe process may use a smaller budget than production, but otherwise
    # keeps the production targeting, filters and adaptive Monte Carlo settings.
    # C2 production discovery has a separate, larger budget.  The original
    # probe only overrode the generic keys, so ``--max-candidates 8192`` still
    # silently ran the 32768-candidate C2 path.  Override both namespaces and
    # use small batches so reaching the required 16 roots stops promptly.
    CONFIG["FRESH_ROOT_MAX_CANDIDATES_PER_TASK"] = int(max_candidates)
    CONFIG["C2_FRESH_ROOT_MAX_CANDIDATES"] = int(max_candidates)
    CONFIG["FRESH_ROOT_BATCH_SIZE"] = int(min(batch_size, max_candidates))
    max_rounds = int(math.ceil(max_candidates / batch_size))
    CONFIG["FRESH_ROOT_MAX_ROUNDS"] = max_rounds
    CONFIG["C2_FRESH_ROOT_MAX_ROUNDS"] = max_rounds
    _validate_generation_config()

    np.random.seed(CONFIG["RANDOM_SEED"])
    sampler = PhysicsAwareSampler()
    report = {
        "status": "PASS",
        "purpose": "rare_label_physical_reachability_probe_not_for_training",
        "dataset_schema": CONFIG["DATASET_SCHEMA"],
        "settings": {
            "max_candidates_per_target": int(max_candidates),
            "batch_size": int(batch_size),
            "required_strict_positive_roots": int(required_roots),
            "label_mc_min_replicates": int(CONFIG["LABEL_MC_MIN_REPLICATES"]),
            "label_mc_max_replicates": int(CONFIG["LABEL_MC_MAX_REPLICATES"]),
            "valid_probability_threshold": float(CONFIG["VALID_PROB_STRICT"]),
            "two_stage_screening": True,
            "screen_replicates": int(screen_replicates),
            "screen_threshold": float(screen_threshold),
        },
        "targets": {},
        "failures": [],
    }

    for name in target_names:
        munition_id, target_col = DEFAULT_TARGETS[name]
        fresh, discovery = _two_stage_discovery(
            sampler,
            munition_id=munition_id,
            target_col=target_col,
            max_candidates=int(max_candidates),
            batch_size=int(batch_size),
            required_roots=int(required_roots),
            screen_replicates=int(screen_replicates),
            screen_threshold=float(screen_threshold),
            progress_path=(
                os.path.abspath(output_path)
                + f".{name}.progress.json"),
        )
        stats = _root_statistics(fresh, target_col)
        reached = stats["strict_positive_roots"] >= required_roots
        report["targets"][name] = {
            "munition_id": int(munition_id),
            "target_column": target_col,
            "required_roots_reached": bool(reached),
            "discovery": discovery,
            "observed": stats,
        }
        if not reached:
            report["status"] = "INCONCLUSIVE_OR_FAIL"
            report["failures"].append(
                f"{name}: strict positive roots={stats['strict_positive_roots']} "
                f"< {required_roots} within {max_candidates} candidates"
            )

    output_path = os.path.abspath(output_path)
    _write_json_atomic(output_path, report)
    report["output"] = output_path
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用真实毁伤引擎探测稀有标签的独立正例 root 可达性。")
    parser.add_argument(
        "--targets", nargs="+", default=list(DEFAULT_TARGETS),
        choices=sorted(DEFAULT_TARGETS), help="要探测的失败单元。")
    parser.add_argument(
        "--max-candidates", type=int,
        default=int(CONFIG["FRESH_ROOT_MAX_CANDIDATES_PER_TASK"]),
        help=("每个目标最多探测的独立候选数，默认与生产配置一致："
              f"{CONFIG['FRESH_ROOT_MAX_CANDIDATES_PER_TASK']}。"))
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="每轮候选数，默认 128；达到所需严格 root 后立即停止。")
    parser.add_argument(
        "--required-roots", type=int, default=16,
        help="判定可达所需的严格正例 root 数，默认 16。")
    parser.add_argument(
        "--screen-replicates", type=int, default=2,
        help="宽松筛选的固定反向配对 MC 次数，默认 2。")
    parser.add_argument(
        "--screen-threshold", type=float, default=0.10,
        help="进入正式复核的宽松概率阈值，默认 0.10。")
    parser.add_argument(
        "--output", default="output/stage0_reachability_probe.json",
        help="JSON 报告路径。")
    args = parser.parse_args()
    report = run_probe(
        args.targets, args.max_candidates, args.batch_size,
        args.required_roots, args.output,
        screen_replicates=args.screen_replicates,
        screen_threshold=args.screen_threshold)
    # Keep the console suitable for long production runs.  Full screening and
    # geometry diagnostics remain available in the atomic JSON artifact.
    console_targets = {
        name: {
            "required_roots_reached": bool(
                values["required_roots_reached"]),
            "strict_positive_roots": int(
                values["observed"]["strict_positive_roots"]),
            "required_roots": int(
                values["discovery"]["desired_strict_roots"]),
            "raw_candidates_requested": int(
                values["discovery"]["raw_candidates_requested"]),
        }
        for name, values in report["targets"].items()
    }
    print(json.dumps({
        "status": report["status"],
        "targets": console_targets,
        "output": report["output"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
