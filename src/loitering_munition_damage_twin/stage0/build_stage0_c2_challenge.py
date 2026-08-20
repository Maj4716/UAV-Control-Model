"""Build a root-independent C2 challenge set without contaminating holdouts.

The production val/test splits intentionally preserve their natural sampling
prevalence.  That leaves too few Med-LM C2 positives for stable per-munition
evaluation.  This tool creates a separate, explicitly biased challenge set:

* one row per independent root;
* equal strict-positive and hard-negative counts for Med-LM/Med-RD/Heavy;
* a random-seed namespace disjoint from the production generator;
* optional root-overlap verification against the official training Parquet.

The artifact is for rare-event discrimination tests only.  Its balanced class
prevalence must not be used for deployment-prior estimation or calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from loitering_munition_damage_twin.stage0.generation import CONFIG, PhysicsAwareSampler


CHALLENGE_SCHEMA = "stage0_c2_challenge_v1"
APPLICABLE_MUNITIONS = (1, 2, 3)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_path(output_path: Path) -> Path:
    return output_path.with_suffix(".profile.json")


def select_root_independent_c2_rows(
    candidates: pd.DataFrame,
    munition_id: int,
    positive_roots: int,
    negative_roots: int,
    valid_threshold: float,
    seed: int,
) -> pd.DataFrame:
    """Select one strict positive or hard negative row per independent root."""
    required = {
        "munition_id", "root_seed_id", "sample_id", "C_ge2_prob",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise RuntimeError(f"C2 challenge 候选缺少字段: {missing}")

    scoped = candidates[
        candidates["munition_id"].astype(int) == int(munition_id)
    ].copy()
    scoped["root_seed_id"] = scoped["root_seed_id"].astype(str)
    scoped = scoped.sort_values(
        ["C_ge2_prob", "sample_id"], ascending=[False, True])

    positives = scoped[
        scoped["C_ge2_prob"].to_numpy(dtype=float) >= valid_threshold
    ].drop_duplicates("root_seed_id", keep="first")
    if len(positives) < positive_roots:
        raise RuntimeError(
            f"m_id={munition_id} C2 challenge 严格正例 root="
            f"{len(positives)} < {positive_roots}")
    positives = positives.sample(
        n=positive_roots, replace=False, random_state=seed)
    positives["challenge_target"] = 1

    negatives = scoped[
        scoped["C_ge2_prob"].to_numpy(dtype=float) < valid_threshold
    ].drop_duplicates("root_seed_id", keep="first")
    negatives = negatives[
        ~negatives["root_seed_id"].isin(positives["root_seed_id"])
    ]
    if len(negatives) < negative_roots:
        raise RuntimeError(
            f"m_id={munition_id} C2 challenge 负例 root="
            f"{len(negatives)} < {negative_roots}")

    # Half of the negatives sit closest to the strict threshold; the other
    # half is sampled from the remaining support so the challenge is not only
    # an infinitesimal boundary test.
    hard_count = min((negative_roots + 1) // 2, len(negatives))
    hard = negatives.nlargest(hard_count, "C_ge2_prob")
    remaining = negatives[
        ~negatives["root_seed_id"].isin(hard["root_seed_id"])
    ]
    broad_count = negative_roots - len(hard)
    broad = remaining.sample(
        n=broad_count, replace=False, random_state=seed + 1
    ) if broad_count else remaining.iloc[0:0]
    negatives = pd.concat([hard, broad], ignore_index=True)
    negatives["challenge_target"] = 0

    selected = pd.concat([positives, negatives], ignore_index=True)
    selected["challenge_schema"] = CHALLENGE_SCHEMA
    selected["challenge_task"] = "C2"
    selected["challenge_munition_id"] = int(munition_id)
    selected["split_role"] = "test"
    selected["loss_weight"] = 1.0
    selected["active_sampling_weight"] = 1.0
    selected["family_weight"] = 1.0
    selected["class_balance_weight"] = 1.0
    if selected["root_seed_id"].duplicated().any():
        raise RuntimeError("C2 challenge 选择后出现重复 root。")
    return selected.sample(
        frac=1.0, random_state=seed + 2).reset_index(drop=True)


def _read_source_identity(source_dataset: Path | None) -> tuple[set[str], dict[str, Any]]:
    if source_dataset is None:
        return set(), {"path": None, "sha256": None, "rows": None}
    if not source_dataset.is_file():
        raise FileNotFoundError(f"正式数据集不存在: {source_dataset}")
    table = pq.read_table(source_dataset, columns=["root_seed_id"])
    roots = set(table.column("root_seed_id").to_pylist())
    profile_path = source_dataset.with_name("generation_profile.json")
    profile = {}
    if profile_path.is_file():
        with profile_path.open("r", encoding="utf-8") as stream:
            profile = json.load(stream)
    source_sha256 = profile.get("artifact", {}).get("sha256")
    if not source_sha256:
        source_sha256 = _sha256(source_dataset)
    identity = {
        "path": str(source_dataset.resolve()),
        "sha256": source_sha256,
        "rows": int(table.num_rows),
    }
    return roots, identity


def write_c2_challenge(
    frame: pd.DataFrame,
    output_path: str | os.PathLike[str],
    build_metadata: dict[str, Any],
    source_roots: set[str] | None = None,
) -> Path:
    """Atomically write and verify the standalone challenge artifact."""
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    source_roots = source_roots or set()

    if frame.empty:
        raise RuntimeError("C2 challenge 为空。")
    if set(frame["challenge_schema"].astype(str).unique()) != {CHALLENGE_SCHEMA}:
        raise RuntimeError("C2 challenge schema 混杂。")
    if frame["sample_id"].astype(str).duplicated().any():
        raise RuntimeError("C2 challenge sample_id 不唯一。")
    if frame["root_seed_id"].astype(str).duplicated().any():
        raise RuntimeError("C2 challenge 必须严格每 root 一行。")
    overlap = set(frame["root_seed_id"].astype(str)) & source_roots
    if overlap:
        raise RuntimeError(
            f"C2 challenge 与正式数据集存在 {len(overlap)} 个重叠 root。")

    temporary = output.with_name(
        f"{output.name}.{os.getpid()}.challenge.tmp")
    try:
        frame.to_parquet(temporary, engine="pyarrow", index=False)
        parquet = pq.ParquetFile(temporary)
        try:
            if parquet.metadata.num_rows != len(frame):
                raise RuntimeError("C2 challenge Parquet 回读行数不一致。")
            if parquet.schema_arrow.names != list(frame.columns):
                raise RuntimeError("C2 challenge Parquet 回读列不一致。")
        finally:
            parquet.close()
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    counts = {}
    for munition_id in APPLICABLE_MUNITIONS:
        scoped = frame[
            frame["challenge_munition_id"].astype(int) == munition_id]
        counts[str(munition_id)] = {
            "rows": int(len(scoped)),
            "positive_rows": int((scoped["challenge_target"] == 1).sum()),
            "negative_rows": int((scoped["challenge_target"] == 0).sum()),
            "root_families": int(scoped["root_seed_id"].nunique()),
        }

    profile = {
        "profile_schema": CHALLENGE_SCHEMA,
        "dataset_schema": str(frame["dataset_schema"].iloc[0]),
        "purpose": "root_independent_rare_event_discrimination_not_calibration",
        "selection_bias": (
            "balanced strict positives plus boundary-enriched negatives; "
            "do not estimate deployment prevalence from this artifact"
        ),
        "task": "C2",
        "valid_threshold": float(build_metadata["valid_threshold"]),
        "per_munition": counts,
        "root_overlap_with_source": 0,
        "source_dataset": build_metadata.get("source_dataset", {}),
        "discovery": build_metadata.get("discovery", {}),
        "artifact": {
            "path": output.name,
            "rows": int(len(frame)),
            "size_bytes": int(output.stat().st_size),
            "sha256": _sha256(output),
        },
    }
    profile_file = _profile_path(output)
    profile_temporary = profile_file.with_name(
        f"{profile_file.name}.{os.getpid()}.challenge.tmp")
    with profile_temporary.open("w", encoding="utf-8") as stream:
        json.dump(profile, stream, indent=2, ensure_ascii=False)
    os.replace(profile_temporary, profile_file)
    return profile_file


def validate_c2_challenge(
    output_path: str | os.PathLike[str],
    source_dataset: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    output = Path(output_path).resolve()
    profile_file = _profile_path(output)
    if not output.is_file() or not profile_file.is_file():
        raise FileNotFoundError(
            f"缺少 challenge Parquet/profile: {output}, {profile_file}")
    with profile_file.open("r", encoding="utf-8") as stream:
        profile = json.load(stream)
    if profile.get("profile_schema") != CHALLENGE_SCHEMA:
        raise RuntimeError("C2 challenge profile schema 不匹配。")
    frame = pd.read_parquet(output)
    if len(frame) != int(profile["artifact"]["rows"]):
        raise RuntimeError("C2 challenge 行数与 profile 不一致。")
    if output.stat().st_size != int(profile["artifact"]["size_bytes"]):
        raise RuntimeError("C2 challenge 文件大小与 profile 不一致。")
    if _sha256(output) != profile["artifact"]["sha256"]:
        raise RuntimeError("C2 challenge SHA-256 不匹配。")
    if frame["root_seed_id"].astype(str).duplicated().any():
        raise RuntimeError("C2 challenge 存在重复 root。")
    if not (
        (frame["challenge_target"].to_numpy(dtype=int) == 1) ==
        (frame["C_ge2_prob"].to_numpy(dtype=float) >=
         float(profile["valid_threshold"]))
    ).all():
        raise RuntimeError("C2 challenge target 与 C_ge2_prob 不一致。")

    source_roots, _ = _read_source_identity(
        Path(source_dataset).resolve() if source_dataset else None)
    overlap = len(set(frame["root_seed_id"].astype(str)) & source_roots)
    if overlap:
        raise RuntimeError(
            f"C2 challenge 与正式数据集存在 {overlap} 个重叠 root。")
    return {
        "status": "PASS",
        "dataset": str(output),
        "profile": str(profile_file),
        "rows": int(len(frame)),
        "root_families": int(frame["root_seed_id"].nunique()),
        "root_overlap_with_source": overlap,
        "per_munition": profile["per_munition"],
        "sha256_verified": True,
    }


def build_c2_challenge(
    output_path: str | os.PathLike[str],
    positive_roots: int = 32,
    negative_roots: int = 32,
    max_candidates: int = 32768,
    batch_size: int = 1024,
    base_seed: int = 104729,
    source_dataset: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    if min(positive_roots, negative_roots, max_candidates, batch_size) <= 0:
        raise ValueError("challenge 数量和候选预算必须为正数。")
    source_path = (
        Path(source_dataset).resolve() if source_dataset is not None else None)
    source_roots, source_identity = _read_source_identity(source_path)
    valid_threshold = float(CONFIG["VALID_PROB_STRICT"])
    original_config = {
        key: CONFIG[key]
        for key in (
            "RANDOM_SEED",
            "FRESH_ROOT_BATCH_SIZE",
            "FRESH_ROOT_MAX_CANDIDATES_PER_TASK",
            "FRESH_ROOT_MAX_ROUNDS",
        )
    }
    selected_blocks = []
    discovery = {}
    try:
        for munition_id in APPLICABLE_MUNITIONS:
            challenge_seed = int(base_seed + munition_id * 1009)
            CONFIG["RANDOM_SEED"] = challenge_seed
            CONFIG["FRESH_ROOT_BATCH_SIZE"] = int(batch_size)
            CONFIG["FRESH_ROOT_MAX_CANDIDATES_PER_TASK"] = int(max_candidates)
            CONFIG["FRESH_ROOT_MAX_ROUNDS"] = int(
                math.ceil(max_candidates / batch_size))
            np.random.seed(challenge_seed)
            sampler = PhysicsAwareSampler()
            fresh, stats = sampler.discover_fresh_target_roots(
                existing_pool=pd.DataFrame(),
                munition_id=munition_id,
                target_col="C2_prob",
                seed_th=float(CONFIG["SEED_PROB_RELAX"]),
                valid_th=valid_threshold,
                desired_seed_roots=positive_roots,
                desired_strict_roots=positive_roots,
                required_split_role=None,
            )
            block = select_root_independent_c2_rows(
                fresh,
                munition_id=munition_id,
                positive_roots=positive_roots,
                negative_roots=negative_roots,
                valid_threshold=valid_threshold,
                seed=challenge_seed,
            )
            selected_blocks.append(block)
            discovery[str(munition_id)] = stats
    finally:
        CONFIG.update(original_config)

    challenge = pd.concat(selected_blocks, ignore_index=True)
    metadata = {
        "valid_threshold": valid_threshold,
        "source_dataset": source_identity,
        "discovery": discovery,
    }
    profile_file = write_c2_challenge(
        challenge, output_path, metadata, source_roots=source_roots)
    report = validate_c2_challenge(output_path, source_dataset=source_dataset)
    report["profile"] = str(profile_file)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="构建或校验与正式训练root隔离的C2稀有事件challenge集。")
    parser.add_argument(
        "--output",
        default="output/challenges/stage0_c2_challenge.parquet")
    parser.add_argument(
        "--source-dataset", default="output/damage_dataset.parquet",
        help="用于检查root不重叠的正式数据集；传空字符串可跳过。")
    parser.add_argument("--positive-roots", type=int, default=32)
    parser.add_argument("--negative-roots", type=int, default=32)
    parser.add_argument("--max-candidates", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=104729)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    source = args.source_dataset or None
    if args.validate_only:
        report = validate_c2_challenge(args.output, source_dataset=source)
    else:
        report = build_c2_challenge(
            output_path=args.output,
            positive_roots=args.positive_roots,
            negative_roots=args.negative_roots,
            max_candidates=args.max_candidates,
            batch_size=args.batch_size,
            base_seed=args.seed,
            source_dataset=source,
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
