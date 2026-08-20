from __future__ import annotations

"""Resumable high-MC relabelling for an existing Stage-0 data set.

The sampler coordinates, lineage and split assignments are immutable.  Only
simulator-derived labels/diagnostics are replayed.  Results are written to a
new directory and pass the same writer/usability gates as a full generation.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from loitering_munition_damage_twin.stage0.component_supervision import (
    COMPONENT_TARGET_COLUMNS,
    sha256_file,
)
from loitering_munition_damage_twin.stage0.generation import (
    CONFIG,
    _build_generation_profile,
    _emit_logit_adjustment,
    _finalize_sample_weights,
    _init_worker,
    _process_single_encounter,
    _validate_generation_config,
    _write_dataset_with_profile,
)
from loitering_munition_damage_twin.simulation.engine import load_armor_plates, load_vehicle_model


MANIFEST_SCHEMA = "stage0_high_mc_relabel_v1"
DEFAULT_INPUT = "output/damage_dataset.parquet"
DEFAULT_OUTPUT = "output/high_mc_stage0/damage_dataset.parquet"
DEFAULT_WORK_DIR = "output/high_mc_stage0_work"
REQUIRED_INPUT_COLUMNS = {
    "sample_id", "root_seed_id", "split_role", "dataset_schema",
    "x", "y", "z", "vx", "vy", "vz", "pitch", "roll", "yaw", "m_id",
}
REQUIRED_REPLAY_COLUMNS = {
    "sample_id", "label_mc_replicates",
    "label_mc_min_replicates", "label_mc_max_replicates",
    "label_mc_all_resolved", "label_mc_max_reached",
    "label_mc_max_standard_error",
}
REQUIRED_REPLAY_COLUMNS.update({
    f"{task}_ge{level}_mc_resolved"
    for task in ("K", "M", "F", "C") for level in (1, 2)
})
REQUIRED_REPLAY_COLUMNS.update({
    f"{task}_ge{level}_mc_standard_error"
    for task in ("K", "M", "F", "C") for level in (1, 2)
})
REQUIRED_REPLAY_COLUMNS.update(COMPONENT_TARGET_COLUMNS)


def _atomic_json_dump(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(
        f"{path.name}.{os.getpid()}.tmp")
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(str(temp_path), str(path))


def _configuration_snapshot() -> Dict[str, Any]:
    keys = (
        "RANDOM_SEED",
        "LABEL_MC_MIN_REPLICATES",
        "LABEL_MC_MAX_REPLICATES",
        "LABEL_MC_CONFIDENCE_Z",
        "LABEL_MC_STANDARD_ERROR_TARGET",
        "LABEL_MC_DECISION_MARGIN",
        "LABEL_MC_ANTITHETIC",
        "VALID_PROB_STRICT",
        "DATASET_SCHEMA",
        "FRAME_CONVENTION_VERSION",
    )
    return {key: CONFIG[key] for key in keys}


def _configuration_sha256(snapshot: Dict[str, Any]) -> str:
    encoded = json.dumps(
        snapshot, sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_source_profile(
        dataset_path: Path) -> Tuple[Dict[str, Any], Path]:
    profile_path = dataset_path.with_name("generation_profile.json")
    if not profile_path.is_file():
        raise RuntimeError(
            f"Missing source generation profile: {profile_path}")
    with open(profile_path, "r", encoding="utf-8") as handle:
        profile = json.load(handle)
    artifact = profile.get("artifact", {})
    expected_sha = str(artifact.get("sha256", ""))
    observed_sha = sha256_file(dataset_path)
    if not expected_sha or expected_sha != observed_sha:
        raise RuntimeError(
            "Source Parquet SHA-256 does not match generation_profile.json.")
    if profile.get("profile_schema") != CONFIG["GENERATION_PROFILE_SCHEMA"]:
        raise RuntimeError(
            "Source profile is not the current Stage-0 lineage contract.")
    return profile, profile_path


def _new_manifest(
        source_path: Path,
        source_sha256: str,
        source_rows: int,
        shard_size: int,
        output_path: Path,
) -> Dict[str, Any]:
    snapshot = _configuration_snapshot()
    expected_shards = (
        (int(source_rows) + int(shard_size) - 1) // int(shard_size)
    )
    return {
        "schema": MANIFEST_SCHEMA,
        "status": "IN_PROGRESS",
        "source": {
            "path": str(source_path),
            "sha256": source_sha256,
            "rows": int(source_rows),
        },
        "output_path": str(output_path),
        "shard_size": int(shard_size),
        "expected_shards": int(expected_shards),
        "configuration": snapshot,
        "configuration_sha256": _configuration_sha256(snapshot),
        "completed_shards": {},
        "started_unix_time": float(time.time()),
    }


def _validate_manifest(
        manifest: Dict[str, Any],
        source_path: Path,
        source_sha256: str,
        source_rows: int,
        shard_size: int,
        output_path: Path,
) -> None:
    failures: List[str] = []
    if manifest.get("schema") != MANIFEST_SCHEMA:
        failures.append("manifest schema")
    source = manifest.get("source", {})
    if str(source.get("path")) != str(source_path):
        failures.append("source path")
    if str(source.get("sha256")) != source_sha256:
        failures.append("source sha256")
    if int(source.get("rows", -1)) != int(source_rows):
        failures.append("source rows")
    if int(manifest.get("shard_size", -1)) != int(shard_size):
        failures.append("shard size")
    if str(manifest.get("output_path")) != str(output_path):
        failures.append("output path")
    snapshot = _configuration_snapshot()
    if manifest.get("configuration") != snapshot:
        failures.append("MC configuration")
    if (
        str(manifest.get("configuration_sha256"))
        != _configuration_sha256(snapshot)
    ):
        failures.append("MC configuration sha256")
    if failures:
        raise RuntimeError(
            "Existing relabel manifest is incompatible: "
            + ", ".join(failures))


def _shard_bounds(
        row_count: int, shard_size: int,
) -> Iterable[Tuple[int, int, int]]:
    shard_index = 0
    for start in range(0, int(row_count), int(shard_size)):
        stop = min(start + int(shard_size), int(row_count))
        yield shard_index, start, stop
        shard_index += 1


def _shard_path(shards_dir: Path, shard_index: int) -> Path:
    return shards_dir / f"part-{int(shard_index):06d}.parquet"


def _prepare_records(
        source_slice: pd.DataFrame,
        global_start: int,
) -> List[Tuple[int, Dict[str, Any]]]:
    records = source_slice.to_dict("records")
    minimum = int(CONFIG["LABEL_MC_MIN_REPLICATES"])
    maximum = int(CONFIG["LABEL_MC_MAX_REPLICATES"])
    tasks: List[Tuple[int, Dict[str, Any]]] = []
    for offset, row in enumerate(records):
        row["label_mc_min_replicates"] = minimum
        row["label_mc_max_replicates"] = maximum
        tasks.append((int(global_start) + offset, row))
    return tasks


def _validate_replay_frame(
        frame: pd.DataFrame,
        expected_sample_ids: pd.Series,
) -> None:
    missing = sorted(REQUIRED_REPLAY_COLUMNS - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"Relabel shard is missing replay columns: {missing}")
    observed = frame["sample_id"].astype(str).reset_index(drop=True)
    expected = expected_sample_ids.astype(str).reset_index(drop=True)
    if not observed.equals(expected):
        raise RuntimeError(
            "Relabel shard sample_id order differs from the source slice.")
    if frame["sample_id"].duplicated().any():
        raise RuntimeError("Relabel shard contains duplicate sample_id values.")
    actual = frame["label_mc_replicates"].astype(int)
    minimum = frame["label_mc_min_replicates"].astype(int)
    maximum = frame["label_mc_max_replicates"].astype(int)
    if (
        (actual < minimum).any()
        or (actual > maximum).any()
        or (minimum != int(CONFIG["LABEL_MC_MIN_REPLICATES"])).any()
        or (maximum != int(CONFIG["LABEL_MC_MAX_REPLICATES"])).any()
    ):
        raise RuntimeError("Relabel shard violates the adaptive MC bounds.")


def _load_completed_shard(
        path: Path,
        metadata: Dict[str, Any],
        expected_sample_ids: pd.Series,
) -> pd.DataFrame:
    if not path.is_file():
        raise RuntimeError(f"Completed shard is missing: {path}")
    if str(metadata.get("sha256", "")) != sha256_file(path):
        raise RuntimeError(f"Completed shard SHA-256 mismatch: {path}")
    frame = pd.read_parquet(path, engine="pyarrow")
    if int(metadata.get("rows", -1)) != len(frame):
        raise RuntimeError(f"Completed shard row count mismatch: {path}")
    _validate_replay_frame(frame, expected_sample_ids)
    return frame


def _phase2_map(
        profile: Dict[str, Any],
        field: str,
) -> Dict[int, Dict[str, Any]]:
    raw = profile.get(field, {})
    return {
        int(munition_id): dict(values)
        for munition_id, values in raw.items()
    }


def _attach_rebuilt_profile(
        frame: pd.DataFrame,
        source_profile: Dict[str, Any],
        source_sha256: str,
        manifest: Dict[str, Any],
) -> pd.DataFrame:
    valid_threshold = float(
        source_profile.get(
            "valid_prob_strict", CONFIG["VALID_PROB_STRICT"]))
    frame = _finalize_sample_weights(frame, valid_threshold)
    final_quota = {
        int(munition_id): int(values["target_quota"])
        for munition_id, values
        in source_profile.get("per_munition", {}).items()
    }
    phase2_task_counts = {
        int(munition_id): {
            str(task): int(count)
            for task, count in values.get(
                "phase2_task_additions", {}).items()
        }
        for munition_id, values
        in source_profile.get("per_munition", {}).items()
    }
    phase1_kept_counts = {
        int(munition_id): int(count)
        for munition_id, count
        in source_profile.get("phase1_kept_counts", {}).items()
    }
    rebuilt = _build_generation_profile(
        frame,
        final_quota=final_quota,
        phase1_kept_counts=phase1_kept_counts,
        phase2_task_counts=phase2_task_counts,
        seed_th=float(source_profile.get(
            "seed_prob_relax", CONFIG["SEED_PROB_RELAX"])),
        valid_th=valid_threshold,
        target_total=int(source_profile.get("target_total", len(frame))),
        phase1_ratio=float(source_profile.get(
            "phase1_ratio", CONFIG["PHASE1_RATIO"])),
        phase2_discovery_stats=_phase2_map(
            source_profile, "phase2_root_discovery"),
        phase2_cell_cap_stats=_phase2_map(
            source_profile, "phase2_ordinal_cell_cap_removals"),
    )
    rebuilt["relabel"] = {
        "schema": MANIFEST_SCHEMA,
        "mode": "existing_terminal_states_high_mc_replay",
        "source_dataset_sha256": source_sha256,
        "source_rows": int(len(frame)),
        "configuration_sha256": manifest["configuration_sha256"],
        "sampling_coordinates_lineage_and_splits_preserved": True,
        "simulator_labels_recomputed": True,
    }
    frame.attrs["focal_loss_gamma"] = CONFIG["FOCAL_LOSS_GAMMA"]
    frame.attrs["valid_prob_strict"] = valid_threshold
    frame.attrs["generation_profile"] = rebuilt
    return frame


def _finalize(
        source_frame: pd.DataFrame,
        source_profile: Dict[str, Any],
        source_sha256: str,
        manifest: Dict[str, Any],
        shards_dir: Path,
        output_path: Path,
) -> Path:
    completed = manifest.get("completed_shards", {})
    replay_frames: List[pd.DataFrame] = []
    for shard_index, start, stop in _shard_bounds(
            len(source_frame), int(manifest["shard_size"])):
        metadata = completed.get(str(shard_index))
        if metadata is None:
            raise RuntimeError(
                f"Cannot finalize: shard {shard_index} is incomplete.")
        frame = _load_completed_shard(
            _shard_path(shards_dir, shard_index),
            metadata,
            source_frame.iloc[start:stop]["sample_id"],
        )
        replay_frames.append(frame)
    replay = pd.concat(replay_frames, ignore_index=True)
    _validate_replay_frame(replay, source_frame["sample_id"])

    merged = source_frame.copy()
    for column in replay.columns:
        merged[column] = replay[column].to_numpy()
    merged = _attach_rebuilt_profile(
        merged, source_profile, source_sha256, manifest)

    profile_path = Path(_write_dataset_with_profile(
        merged, str(output_path)))
    with open(profile_path, "r", encoding="utf-8") as handle:
        written_profile = json.load(handle)
    _emit_logit_adjustment(
        merged,
        float(merged.attrs["valid_prob_strict"]),
        CONFIG["PHYSICAL_PRIOR"],
        str(output_path.with_name("logit_adjustment.json")),
        dataset_sha256=written_profile["artifact"]["sha256"],
    )
    manifest["status"] = "COMPLETE"
    manifest["completed_unix_time"] = float(time.time())
    manifest["output"] = {
        "path": str(output_path),
        "sha256": written_profile["artifact"]["sha256"],
        "rows": int(len(merged)),
        "profile_path": str(profile_path),
    }
    return profile_path


def run_relabel(args: argparse.Namespace) -> int:
    _validate_generation_config()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    work_dir = Path(args.work_dir).resolve()
    manifest_path = work_dir / "manifest.json"
    shards_dir = work_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    if input_path == output_path:
        raise RuntimeError(
            "High-MC replay must not overwrite its source dataset.")
    source_profile, _ = _load_source_profile(input_path)
    source_sha256 = str(source_profile["artifact"]["sha256"])
    source_frame = pd.read_parquet(input_path, engine="pyarrow")
    missing_input = sorted(
        REQUIRED_INPUT_COLUMNS - set(source_frame.columns))
    if missing_input:
        raise RuntimeError(
            f"Source dataset is missing replay inputs: {missing_input}")
    if source_frame["sample_id"].duplicated().any():
        raise RuntimeError("Source dataset contains duplicate sample_id values.")

    if manifest_path.is_file():
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        _validate_manifest(
            manifest, input_path, source_sha256, len(source_frame),
            int(args.shard_size), output_path)
    else:
        manifest = _new_manifest(
            input_path, source_sha256, len(source_frame),
            int(args.shard_size), output_path)
        _atomic_json_dump(manifest, manifest_path)

    if manifest.get("status") == "COMPLETE":
        print(
            f"[RELABEL] COMPLETE | rows={len(source_frame)} | "
            f"output={output_path}")
        return 0

    completed_this_run = 0
    pending = []
    for shard_index, start, stop in _shard_bounds(
            len(source_frame), int(args.shard_size)):
        metadata = manifest["completed_shards"].get(str(shard_index))
        if metadata is not None:
            _load_completed_shard(
                _shard_path(shards_dir, shard_index),
                metadata,
                source_frame.iloc[start:stop]["sample_id"],
            )
        else:
            pending.append((shard_index, start, stop))

    if pending and not args.finalize_only:
        workers = (
            max(1, multiprocessing.cpu_count() - 1)
            if int(args.workers) <= 0 else int(args.workers)
        )
        components = load_vehicle_model()
        plates = load_armor_plates()
        print(
            f"[RELABEL] START | rows={len(source_frame)} | "
            f"pending_shards={len(pending)} | workers={workers} | "
            f"MC={CONFIG['LABEL_MC_MIN_REPLICATES']}-"
            f"{CONFIG['LABEL_MC_MAX_REPLICATES']}")
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(components, plates),
        ) as executor:
            for shard_index, start, stop in pending:
                shard_started = time.time()
                tasks = _prepare_records(
                    source_frame.iloc[start:stop], start)
                results = list(executor.map(
                    _process_single_encounter, tasks, chunksize=1))
                replay = pd.DataFrame(results)
                _validate_replay_frame(
                    replay,
                    source_frame.iloc[start:stop]["sample_id"],
                )
                shard_path = _shard_path(shards_dir, shard_index)
                temp_path = shard_path.with_name(
                    f"{shard_path.name}.{os.getpid()}.tmp")
                replay.to_parquet(
                    temp_path, engine="pyarrow", index=False)
                os.replace(str(temp_path), str(shard_path))
                manifest["completed_shards"][str(shard_index)] = {
                    "rows": int(len(replay)),
                    "start": int(start),
                    "stop": int(stop),
                    "sha256": sha256_file(shard_path),
                    "elapsed_seconds": float(
                        time.time() - shard_started),
                    "all_resolved_rows": int(
                        replay["label_mc_all_resolved"].astype(bool).sum()),
                    "maximum_reached_rows": int(
                        replay["label_mc_max_reached"].astype(bool).sum()),
                }
                _atomic_json_dump(manifest, manifest_path)
                completed_this_run += 1
                total_complete = len(manifest["completed_shards"])
                print(
                    f"[RELABEL] shard={shard_index + 1}/"
                    f"{manifest['expected_shards']} | rows={start}:{stop} | "
                    f"complete={total_complete}/"
                    f"{manifest['expected_shards']} | "
                    f"elapsed={time.time() - shard_started:.1f}s")
                if (
                    args.max_shards is not None
                    and completed_this_run >= int(args.max_shards)
                ):
                    break

    if (
        len(manifest["completed_shards"])
        != int(manifest["expected_shards"])
    ):
        print(
            f"[RELABEL] PAUSED | complete_shards="
            f"{len(manifest['completed_shards'])}/"
            f"{manifest['expected_shards']} | manifest={manifest_path}")
        return 0

    profile_path = _finalize(
        source_frame, source_profile, source_sha256,
        manifest, shards_dir, output_path)
    _atomic_json_dump(manifest, manifest_path)
    print(
        f"[RELABEL] COMPLETE | rows={len(source_frame)} | "
        f"profile={profile_path}")
    return 0


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay an existing Stage-0 dataset with the configured "
            "adaptive high-MC label contract."))
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    parser.add_argument(
        "--shard-size", type=int, default=512,
        help="Rows per resumable shard (default: 512).")
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Worker count; <=0 uses CPU count minus one.")
    parser.add_argument(
        "--max-shards", type=int, default=None,
        help="Process at most this many new shards, then pause safely.")
    parser.add_argument(
        "--finalize-only", action="store_true",
        help="Do not simulate; finalize only when every shard exists.")
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    if int(args.shard_size) <= 0:
        parser.error("--shard-size must be positive")
    if args.max_shards is not None and int(args.max_shards) <= 0:
        parser.error("--max-shards must be positive")
    return run_relabel(args)


if __name__ == "__main__":
    raise SystemExit(main())
