from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from loitering_munition_damage_twin.stage0.component_supervision import (
    COMPONENT_SUPERVISION_FILENAME,
    COMPONENT_SUPERVISION_PROFILE_FILENAME,
    COMPONENT_TARGET_COLUMNS,
    build_component_supervision_profile,
    component_supervision_source_hashes,
    sha256_file,
    sha256_text_sequence,
)
from loitering_munition_damage_twin.stage0.generation import (
    CONFIG,
    _init_worker,
    _process_single_encounter,
)
from loitering_munition_damage_twin.simulation.engine import load_armor_plates, load_vehicle_model
from loitering_munition_damage_twin.paths import PROJECT_ROOT


REPO_ROOT = PROJECT_ROOT
ORDINAL_COLUMNS = tuple(
    f"{task}_ge{level}_prob"
    for task in ("K", "M", "F", "C")
    for level in (1, 2)
)
REPLAY_INPUT_COLUMNS = (
    "x", "y", "z", "vx", "vy", "vz",
    "pitch", "roll", "yaw", "m_id",
    "sample_id", "label_mc_replicates",
) + ORDINAL_COLUMNS


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def _replay_one(args) -> dict:
    row_index, row = args
    expected = np.asarray(
        [row[column] for column in ORDINAL_COLUMNS],
        dtype=np.float64,
    )
    # Supplying only label_mc_replicates deliberately selects the legacy
    # fixed-count path.  It replays exactly the same first N deterministic
    # replicates recorded by the official adaptive run.
    result = _process_single_encounter((row_index, row))
    observed = np.asarray(
        [result[column] for column in ORDINAL_COLUMNS],
        dtype=np.float64,
    )
    if not np.allclose(
            observed, expected, rtol=1e-7, atol=1e-7):
        maximum_error = float(np.max(np.abs(
            observed - expected)))
        raise RuntimeError(
            "Stage-0 label replay mismatch for "
            f"sample_id={row['sample_id']}: "
            f"max_abs_error={maximum_error:.3e}. "
            "The simulator/RNG contract differs from the dataset; "
            "component targets must not be attached.")
    return {
        "row_index": int(row_index),
        "sample_id": str(row["sample_id"]),
        **{
            column: np.float32(result[column])
            for column in COMPONENT_TARGET_COLUMNS
        },
    }


def _load_base_contract(dataset_path: Path) -> tuple[dict, str]:
    profile_path = dataset_path.with_name("generation_profile.json")
    if not profile_path.is_file():
        raise FileNotFoundError(
            f"Missing generation profile: {profile_path}")
    with profile_path.open("r", encoding="utf-8") as stream:
        profile = json.load(stream)
    artifact = profile.get("artifact", {})
    expected_hash = str(artifact.get("sha256", ""))
    observed_hash = sha256_file(dataset_path)
    if expected_hash != observed_hash:
        raise RuntimeError(
            "Stage-0 Parquet SHA-256 differs from "
            "generation_profile.json.")
    if int(artifact.get("rows", -1)) <= 0:
        raise RuntimeError(
            "generation_profile.json lacks a valid row count.")
    if profile.get("usability_gate", {}).get("passed") is not True:
        raise RuntimeError(
            "Component supervision may only be built for a dataset that "
            "passed the Stage-0 usability gate.")
    return profile, observed_hash


def _part_is_valid(
        part_path: Path,
        expected_sample_ids: list[str]) -> bool:
    if not part_path.is_file():
        return False
    try:
        table = pq.read_table(
            part_path, columns=["sample_id", "row_index"])
    except Exception:
        return False
    if table.num_rows != len(expected_sample_ids):
        return False
    observed = [
        str(value)
        for value in table.column("sample_id").to_pylist()
    ]
    if observed != expected_sample_ids:
        return False
    row_index = np.asarray(
        table.column("row_index").to_numpy(), dtype=np.int64)
    return bool(
        len(row_index) == 0
        or np.all(np.diff(row_index) == 1)
    )


def _write_part(part_path: Path, records: list[dict]) -> None:
    frame = pd.DataFrame.from_records(records)
    ordered = ["row_index", "sample_id", *COMPONENT_TARGET_COLUMNS]
    frame = frame[ordered].sort_values(
        "row_index", kind="stable").reset_index(drop=True)
    for column in COMPONENT_TARGET_COLUMNS:
        frame[column] = frame[column].astype(np.float32)
    temporary = Path(str(part_path) + ".tmp")
    frame.to_parquet(
        temporary, engine="pyarrow", index=False,
        row_group_size=max(1, len(frame)),
    )
    check = pq.ParquetFile(temporary)
    try:
        if check.schema_arrow.names != ordered:
            raise RuntimeError(
                f"Part schema readback failed: {part_path}")
        if check.metadata.num_rows != len(frame):
            raise RuntimeError(
                f"Part row-count readback failed: {part_path}")
    finally:
        check.close()
    os.replace(temporary, part_path)


def _assemble_parts(
        part_paths: list[Path],
        output_path: Path) -> tuple[int, int]:
    temporary = Path(str(output_path) + ".tmp")
    writer = None
    total_rows = 0
    try:
        for part_path in part_paths:
            table = pq.read_table(part_path).drop(["row_index"])
            table = table.select(
                ["sample_id", *COMPONENT_TARGET_COLUMNS])
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary, table.schema,
                    compression="snappy")
            writer.write_table(table)
            total_rows += int(table.num_rows)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("No component supervision parts were built.")
    check = pq.ParquetFile(temporary)
    try:
        if check.schema_arrow.names != [
                "sample_id", *COMPONENT_TARGET_COLUMNS]:
            raise RuntimeError(
                "Final component supervision schema readback failed.")
        if int(check.metadata.num_rows) != total_rows:
            raise RuntimeError(
                "Final component supervision row-count readback failed.")
        row_groups = int(check.num_row_groups)
    finally:
        check.close()
    os.replace(temporary, output_path)
    return total_rows, row_groups


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the immutable Stage-0 Monte-Carlo lineage and build "
            "training-only component fragment/shock targets without "
            "regenerating active-sampling geometry."
        ))
    parser.add_argument(
        "dataset", nargs="?",
        default="output/damage_dataset.parquet")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--workers", type=int,
        default=max(1, multiprocessing.cpu_count() - 1))
    parser.add_argument(
        "--rows-per-part", type=int, default=2048)
    parser.add_argument(
        "--map-chunksize", type=int, default=8)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Build a non-production prefix for smoke validation.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = (REPO_ROOT / dataset_path).resolve()
    profile, dataset_sha256 = _load_base_contract(dataset_path)
    base_rows = int(profile["artifact"]["rows"])

    output_path = (
        Path(args.output).resolve()
        if args.output
        else dataset_path.with_name(
            COMPONENT_SUPERVISION_FILENAME
            if args.limit is None
            else "component_supervision_smoke.parquet")
    )
    if args.output:
        profile_path = output_path.with_name(
            output_path.stem + "_profile.json")
    else:
        profile_path = output_path.with_name(
            COMPONENT_SUPERVISION_PROFILE_FILENAME
            if args.limit is None
            else "component_supervision_smoke_profile.json")
    part_root = output_path.with_suffix(
        output_path.suffix + ".parts")
    part_root.mkdir(parents=True, exist_ok=True)

    requested_rows = (
        base_rows
        if args.limit is None
        else min(base_rows, max(1, int(args.limit)))
    )
    rows_per_part = max(1, int(args.rows_per_part))
    build_contract = {
        "schema": "stage0_component_supervision_replay_build_v1",
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "base_rows": base_rows,
        "requested_rows": requested_rows,
        "rows_per_part": rows_per_part,
        "random_seed": int(CONFIG["RANDOM_SEED"]),
        "target_columns": list(COMPONENT_TARGET_COLUMNS),
        "source_sha256": component_supervision_source_hashes(),
    }
    manifest_path = part_root / "build_manifest.json"
    if manifest_path.is_file():
        with manifest_path.open(
                "r", encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing != build_contract:
            raise RuntimeError(
                "Existing replay parts belong to a different contract. "
                f"Choose a new --output path or explicitly remove "
                f"{part_root}.")
    else:
        _atomic_json(manifest_path, build_contract)

    frame = pd.read_parquet(
        dataset_path, engine="pyarrow",
        columns=list(REPLAY_INPUT_COLUMNS))
    if len(frame) != base_rows:
        raise RuntimeError(
            "Stage-0 row count changed while reading replay inputs.")
    frame = frame.iloc[:requested_rows].copy()
    if frame["sample_id"].astype(str).duplicated().any():
        raise RuntimeError(
            "Stage-0 sample_id is not unique.")
    if (
        not np.isfinite(
            frame[list(ORDINAL_COLUMNS)].to_numpy(
                dtype=np.float64)).all()
    ):
        raise RuntimeError(
            "Stage-0 ordinal labels contain non-finite values.")

    part_count = int(math.ceil(requested_rows / rows_per_part))
    part_paths = [
        part_root / f"part-{index:06d}.parquet"
        for index in range(part_count)
    ]
    components = load_vehicle_model()
    plates = load_armor_plates()
    started = time.time()
    completed_rows = 0

    with ProcessPoolExecutor(
            max_workers=max(1, int(args.workers)),
            initializer=_init_worker,
            initargs=(components, plates)) as executor:
        for part_index, part_path in enumerate(part_paths):
            start = part_index * rows_per_part
            stop = min(requested_rows, start + rows_per_part)
            sample_ids = frame.iloc[start:stop][
                "sample_id"].astype(str).tolist()
            if _part_is_valid(part_path, sample_ids):
                completed_rows += stop - start
                continue
            tasks = []
            for row_index, values in zip(
                    range(start, stop),
                    frame.iloc[start:stop].itertuples(
                        index=False, name=None)):
                row = dict(zip(REPLAY_INPUT_COLUMNS, values))
                row["sample_id"] = str(row["sample_id"])
                row["label_mc_replicates"] = int(
                    row["label_mc_replicates"])
                tasks.append((row_index, row))
            records = list(executor.map(
                _replay_one, tasks,
                chunksize=max(1, int(args.map_chunksize))))
            _write_part(part_path, records)
            completed_rows += stop - start
            elapsed = max(time.time() - started, 1e-6)
            rate = completed_rows / elapsed
            remaining = (
                (requested_rows - completed_rows) / rate
                if rate > 0 else float("nan")
            )
            print(
                "[COMPONENT] "
                f"{completed_rows}/{requested_rows} "
                f"({100.0 * completed_rows / requested_rows:.1f}%) "
                f"rate={rate:.1f} rows/s "
                f"eta={remaining / 60.0:.1f} min")

    total_rows, row_groups = _assemble_parts(
        part_paths, output_path)
    sample_id_hash = sha256_text_sequence(
        frame["sample_id"].astype(str))
    sidecar_sample_ids = pd.read_parquet(
        output_path, engine="pyarrow",
        columns=["sample_id"])["sample_id"].astype(str)
    if sha256_text_sequence(
            sidecar_sample_ids) != sample_id_hash:
        raise RuntimeError(
            "Final sidecar sample_id order differs from Stage-0.")
    sidecar_profile = build_component_supervision_profile(
        base_dataset_path=str(dataset_path),
        base_dataset_sha256=dataset_sha256,
        base_dataset_rows=base_rows,
        base_dataset_schema=str(profile["dataset_schema"]),
        frame_convention=str(profile["frame_convention"]),
        sidecar_path=str(output_path),
        sidecar_rows=total_rows,
        sidecar_size_bytes=output_path.stat().st_size,
        sidecar_sha256=sha256_file(output_path),
        sample_id_order_sha256=sample_id_hash,
        parquet_row_groups=row_groups,
        pyarrow_version=pa.__version__,
        label_replay_verified=True,
    )
    sidecar_profile["build"] = {
        "status": (
            "COMPLETE" if total_rows == base_rows
            else "SMOKE_PREFIX_ONLY"
        ),
        "workers": max(1, int(args.workers)),
        "rows_per_part": rows_per_part,
        "elapsed_seconds": float(time.time() - started),
    }
    _atomic_json(profile_path, sidecar_profile)
    print(json.dumps({
        "status": sidecar_profile["build"]["status"],
        "rows": total_rows,
        "output": str(output_path),
        "profile": str(profile_path),
        "sha256": sidecar_profile["artifact"]["sha256"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
