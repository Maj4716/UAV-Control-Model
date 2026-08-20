from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from typing import Iterable


MANIFEST_SCHEMA = "stage0_nn_artifact_v1"
REQUIRED_ARTIFACTS = (
    "best_model.pth",
    "best_thresholds.json",
    "minmax_scaler.pkl",
    "minmax_scaler.json",
)


def canonicalize_data_contract(contract: dict) -> dict:
    """Return the semantic form used for artifact compatibility checks.

    Runs sealed before per-cell confidence strength was introduced have no
    corresponding key; their behavior is exactly the all-ones matrix.  No
    other missing or changed contract field is normalized.
    """
    canonical = dict(contract)
    canonical.setdefault(
        "label_confidence_strength_by_task_munition",
        [[1.0] * 4 for _ in range(4)],
    )
    canonical.setdefault("terminal_physics_contract", None)
    canonical.setdefault("mechanism_supervision_enabled", False)
    canonical.setdefault("mechanism_target_schema", None)
    canonical.setdefault("component_supervision_enabled", False)
    canonical.setdefault("component_supervision_contract", None)
    canonical.setdefault("component_positive_weight", None)
    return canonical


def data_contracts_match(left: dict, right: dict) -> bool:
    return canonicalize_data_contract(left) == canonicalize_data_contract(right)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def write_model_manifest(model_dir: str, data_contract: dict,
                         model_config: dict, training_config: dict,
                         seed: int) -> str:
    model_dir = os.path.abspath(model_dir)
    artifacts = {}
    for name in REQUIRED_ARTIFACTS:
        path = os.path.join(model_dir, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Cannot seal model artifacts; missing required file: {path}")
        artifacts[name] = {
            "sha256": sha256_file(path),
            "size_bytes": int(os.path.getsize(path)),
        }

    payload = {
        "schema": MANIFEST_SCHEMA,
        "created_unix": int(time.time()),
        "python_version": platform.python_version(),
        "seed": int(seed),
        "data_contract": data_contract,
        "model_config": model_config,
        "training_config": training_config,
        "artifacts": artifacts,
    }
    manifest_path = os.path.join(model_dir, "model_manifest.json")
    _atomic_write_json(manifest_path, payload)
    return manifest_path


def load_and_verify_manifest(model_dir: str, dataset_sha256: str | None = None,
                             feature_names: Iterable[str] | None = None) -> dict:
    model_dir = os.path.abspath(model_dir)
    manifest_path = os.path.join(model_dir, "model_manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            "Missing model_manifest.json. Legacy or partially generated model "
            "artifacts are rejected; retrain with the current pipeline.")
    with open(manifest_path, "r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise RuntimeError(
            f"Unsupported model manifest schema: {manifest.get('schema')!r}")

    for name in REQUIRED_ARTIFACTS:
        expected = manifest.get("artifacts", {}).get(name, {})
        path = os.path.join(model_dir, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Manifest artifact is missing: {path}")
        if int(expected.get("size_bytes", -1)) != os.path.getsize(path):
            raise RuntimeError(f"Artifact size mismatch: {name}")
        if expected.get("sha256") != sha256_file(path):
            raise RuntimeError(f"Artifact SHA-256 mismatch: {name}")

    contract = manifest.get("data_contract", {})
    if dataset_sha256 is not None and (
            contract.get("dataset_sha256") != dataset_sha256):
        raise RuntimeError(
            "Model and Parquet SHA-256 contracts differ; refusing evaluation.")
    if feature_names is not None and (
            contract.get("feature_names") != list(feature_names)):
        raise RuntimeError(
            "Model/scaler feature order differs from the requested pipeline.")
    return manifest
