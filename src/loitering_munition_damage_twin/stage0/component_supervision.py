from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from loitering_munition_damage_twin.simulation import engine as engine_module
from loitering_munition_damage_twin.simulation.engine import (
    bundled_resource_path,
)


COMPONENT_SUPERVISION_SCHEMA = "stage0_component_supervision_v1"
COMPONENT_SUPERVISION_FILENAME = "component_supervision.parquet"
COMPONENT_SUPERVISION_PROFILE_FILENAME = (
    "component_supervision_profile.json"
)
COMPONENT_SUPERVISION_MECHANISMS = ("fragment", "shock")

# This is the exact union of component IDs consumed by
# sim_engine.build_damage_tree_rules().  Keeping the list here gives the
# generator, replay tool, loader and neural model one stable column order.
DAMAGE_RULE_COMPONENT_GROUPS = {
    "k_fuel": (3,),
    "k_ammo": (46,),
    "m_wheels": tuple(range(6, 18)),
    "m_idlers": (30, 31),
    "m_power": (1, 2, 3),
    "m_drives": (4, 5),
    "m_tracks": tuple(range(32, 40)),
    "m_driver": (58,),
    "f_secondary": (47, 48, 50, 51, 52, 53, 54),
    "f_sights": (41, 42),
    "f_fire_control": (43, 44),
    "f_main_supply": (45, 49),
    "f_command": (59, 60),
    "c_crew": tuple(range(58, 68)),
}
CRITICAL_COMPONENT_IDS = tuple(sorted({
    int(component_id)
    for component_ids in DAMAGE_RULE_COMPONENT_GROUPS.values()
    for component_id in component_ids
}))


def component_target_column(
        mechanism: str, component_id: int) -> str:
    mechanism = str(mechanism)
    component_id = int(component_id)
    if mechanism not in COMPONENT_SUPERVISION_MECHANISMS:
        raise ValueError(
            f"Unsupported component mechanism: {mechanism!r}")
    if component_id not in CRITICAL_COMPONENT_IDS:
        raise ValueError(
            f"Component {component_id} is not used by the damage tree.")
    return (
        f"component_{component_id:03d}_"
        f"{mechanism}_damage_prob"
    )


COMPONENT_TARGET_COLUMNS = tuple(
    component_target_column(mechanism, component_id)
    for mechanism in COMPONENT_SUPERVISION_MECHANISMS
    for component_id in CRITICAL_COMPONENT_IDS
)


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text_sequence(values: Iterable[str]) -> str:
    """Hash an ordered string sequence without ambiguous concatenation."""
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def component_supervision_source_hashes(
        source_root: str | os.PathLike[str] | None = None) -> dict:
    if source_root is not None:
        legacy_root = Path(source_root).resolve()
        legacy_paths = {
            name: legacy_root / name
            for name in ("sim_engine.py", "vehicle_model.json", "armor.csv")
        }
        if all(path.is_file() for path in legacy_paths.values()):
            return {
                name: sha256_file(path)
                for name, path in legacy_paths.items()
            }

    source_paths = {
        "sim_engine.py": Path(engine_module.__file__).resolve(),
        "vehicle_model.json": Path(
            bundled_resource_path("vehicle_model.json")
        ),
        "armor.csv": Path(bundled_resource_path("armor.csv")),
    }
    return {
        name: sha256_file(path) if path.is_file() else None
        for name, path in source_paths.items()
    }


def extract_component_mc_means(
        replicate_results: Sequence[object]) -> np.ndarray:
    """Return fragment/shock MC means in the immutable (2, C) order."""
    if not replicate_results:
        raise ValueError(
            "At least one simulator replicate is required.")
    samples = np.empty(
        (
            len(replicate_results),
            len(COMPONENT_SUPERVISION_MECHANISMS),
            len(CRITICAL_COMPONENT_IDS),
        ),
        dtype=np.float64,
    )
    for replicate_index, result in enumerate(replicate_results):
        by_id = {
            int(component.component_id): component
            for component in result.component_results
        }
        missing = [
            component_id
            for component_id in CRITICAL_COMPONENT_IDS
            if component_id not in by_id
        ]
        if missing:
            raise RuntimeError(
                "Simulator result is missing damage-tree components: "
                f"{missing}")
        for component_index, component_id in enumerate(
                CRITICAL_COMPONENT_IDS):
            component = by_id[component_id]
            samples[replicate_index, 0, component_index] = float(
                component.fragment_damage_prob)
            samples[replicate_index, 1, component_index] = float(
                component.shockwave_damage_prob)
    if (
        not np.isfinite(samples).all()
        or np.any(samples < 0.0)
        or np.any(samples > 1.0)
    ):
        raise RuntimeError(
            "Component supervision contains non-finite or out-of-range "
            "simulator probabilities.")
    return samples.mean(axis=0).astype(np.float32)


def component_means_to_columns(
        component_means: np.ndarray) -> dict[str, np.float32]:
    values = np.asarray(component_means, dtype=np.float32)
    expected_shape = (
        len(COMPONENT_SUPERVISION_MECHANISMS),
        len(CRITICAL_COMPONENT_IDS),
    )
    if values.shape != expected_shape:
        raise ValueError(
            f"component_means must have shape {expected_shape}, "
            f"got {values.shape}.")
    return {
        component_target_column(mechanism, component_id):
            np.float32(values[mechanism_index, component_index])
        for mechanism_index, mechanism in enumerate(
            COMPONENT_SUPERVISION_MECHANISMS)
        for component_index, component_id in enumerate(
            CRITICAL_COMPONENT_IDS)
    }


def build_component_supervision_profile(
        *,
        base_dataset_path: str,
        base_dataset_sha256: str,
        base_dataset_rows: int,
        base_dataset_schema: str,
        frame_convention: str,
        sidecar_path: str,
        sidecar_rows: int,
        sidecar_size_bytes: int,
        sidecar_sha256: str,
        sample_id_order_sha256: str,
        parquet_row_groups: int,
        pyarrow_version: str,
        label_replay_verified: bool,
) -> dict:
    if len(str(base_dataset_sha256)) != 64:
        raise ValueError("base_dataset_sha256 must be a SHA-256 digest.")
    if len(str(sidecar_sha256)) != 64:
        raise ValueError("sidecar_sha256 must be a SHA-256 digest.")
    return {
        "schema": COMPONENT_SUPERVISION_SCHEMA,
        "base_dataset": {
            "path": os.path.basename(base_dataset_path),
            "sha256": str(base_dataset_sha256),
            "rows": int(base_dataset_rows),
            "dataset_schema": str(base_dataset_schema),
            "frame_convention": str(frame_convention),
            "sample_id_order_sha256": str(sample_id_order_sha256),
        },
        "target_contract": {
            "component_ids": list(CRITICAL_COMPONENT_IDS),
            "mechanisms": list(COMPONENT_SUPERVISION_MECHANISMS),
            "target_columns": list(COMPONENT_TARGET_COLUMNS),
            "target_count": len(COMPONENT_TARGET_COLUMNS),
            "aggregation": (
                "mean_of_component_damage_probabilities_over_the_same_"
                "replicates_and_rng_lineage_as_stage0_labels"
            ),
            "role": "training_only_auxiliary_targets",
            "model_input_allowed": False,
            "label_replay_verified": bool(label_replay_verified),
            "source_sha256": component_supervision_source_hashes(),
        },
        "artifact": {
            "path": os.path.basename(sidecar_path),
            "rows": int(sidecar_rows),
            "columns": 1 + len(COMPONENT_TARGET_COLUMNS),
            "size_bytes": int(sidecar_size_bytes),
            "sha256": str(sidecar_sha256),
            "sample_id_order_sha256": str(
                sample_id_order_sha256),
            "parquet_row_groups": int(parquet_row_groups),
            "all_columns_readback_verified": True,
            "pyarrow_version": str(pyarrow_version),
        },
    }


def validate_component_supervision_profile(
        profile: dict,
        *,
        base_dataset_path: str,
        base_dataset_sha256: str,
        base_dataset_rows: int,
        base_dataset_schema: str,
        frame_convention: str,
) -> None:
    """Validate the semantic binding before the sidecar is opened."""
    if profile.get("schema") != COMPONENT_SUPERVISION_SCHEMA:
        raise RuntimeError(
            "Unsupported component supervision schema: "
            f"{profile.get('schema')!r}")
    base = profile.get("base_dataset", {})
    expected = {
        "sha256": str(base_dataset_sha256),
        "rows": int(base_dataset_rows),
        "dataset_schema": str(base_dataset_schema),
        "frame_convention": str(frame_convention),
    }
    observed = {
        "sha256": str(base.get("sha256", "")),
        "rows": int(base.get("rows", -1)),
        "dataset_schema": str(base.get("dataset_schema", "")),
        "frame_convention": str(base.get("frame_convention", "")),
    }
    if observed != expected:
        raise RuntimeError(
            "Component supervision is bound to a different Stage-0 "
            f"dataset: expected={expected}, observed={observed}")
    if base.get("path") != os.path.basename(base_dataset_path):
        raise RuntimeError(
            "Component supervision base dataset filename differs from "
            "the requested artifact.")
    contract = profile.get("target_contract", {})
    if contract.get("component_ids") != list(CRITICAL_COMPONENT_IDS):
        raise RuntimeError(
            "Component supervision ID order differs from the neural "
            "contract.")
    if contract.get("mechanisms") != list(
            COMPONENT_SUPERVISION_MECHANISMS):
        raise RuntimeError(
            "Component supervision mechanism order differs from the "
            "neural contract.")
    if contract.get("target_columns") != list(
            COMPONENT_TARGET_COLUMNS):
        raise RuntimeError(
            "Component supervision target column order differs from the "
            "neural contract.")
    if contract.get("model_input_allowed") is not False:
        raise RuntimeError(
            "Component supervision must be explicitly marked as "
            "training-only and forbidden as model input.")
    if contract.get("label_replay_verified") is not True:
        raise RuntimeError(
            "Component supervision lacks exact Stage-0 label replay "
            "verification.")
