"""Gate-controlled continuation after a long Stage-0 generation process.

The coordinator waits for an already-running generator PID.  It never treats
process exit as success: current dataset validation and audit contracts must
pass before training starts.  Test evaluation remains sealed unless the A40
validation report passes the complete strict 94%/90% goal and safety gate.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List

import psutil

from loitering_munition_damage_twin.paths import (
    EXPERIMENT_OUTPUT_ROOT,
    OUTPUT_ROOT,
    PROJECT_ROOT,
)


SCHEMA = "stage0_post_generation_pipeline_v1"
REPO_ROOT = PROJECT_ROOT
DEFAULT_STATE = OUTPUT_ROOT / "post_generation_pipeline_state.json"
EXPERIMENT_ID = "A40_independent_mechanism_component_proxies"
CONFIG_NAME = "A40_independent_mechanism_component_proxies"


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    os.replace(str(temporary), str(path))


def _update_state(
        state_path: Path,
        state: Dict[str, Any],
        status: str,
        **updates: Any) -> None:
    state["status"] = str(status)
    state["updated_unix_time"] = float(time.time())
    state.update(updates)
    _write_json_atomic(state_path, state)


def _run_step(
        name: str,
        command: List[str],
        state_path: Path,
        state: Dict[str, Any]) -> subprocess.CompletedProcess:
    logs_dir = state_path.parent / "post_generation_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / f"{name}.stdout.log"
    stderr_path = logs_dir / f"{name}.stderr.log"
    _update_state(
        state_path, state, f"RUNNING_{name.upper()}",
        active_command=command,
        active_stdout=str(stdout_path),
        active_stderr=str(stderr_path),
    )
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, \
            stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    record = {
        "name": name,
        "command": command,
        "returncode": int(completed.returncode),
        "elapsed_seconds": float(time.time() - started),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    state.setdefault("steps", []).append(record)
    _write_json_atomic(state_path, state)
    return completed


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def _audit_contract_status(payload: Dict[str, Any]) -> str:
    """Read current and legacy audit payloads without weakening the gate."""
    recognized = {
        "CURRENT_V2",
        "CURRENT_V2_EVIDENCE_GAP",
        "LEGACY_OR_SCHEMA_MISMATCH",
    }
    top_level = payload.get("contract_status")
    if top_level is not None:
        if top_level not in recognized:
            raise RuntimeError(
                f"Unrecognized audit contract_status: {top_level!r}")
        return str(top_level)

    statistics = payload.get("statistics")
    if not isinstance(statistics, dict):
        raise RuntimeError(
            "Audit payload has neither contract_status nor statistics.")
    identity = statistics.get("artifact_identity")
    evidence = statistics.get("exact_level_evidence")
    if not isinstance(identity, dict) or not isinstance(evidence, dict):
        raise RuntimeError(
            "Audit statistics lack artifact identity or exact-level evidence.")
    schema_match = identity.get("current_schema_match")
    contract_ready = evidence.get("contract_ready")
    if not isinstance(schema_match, bool) or not isinstance(
            contract_ready, bool):
        raise RuntimeError(
            "Audit compatibility fields must be explicit booleans.")
    if not schema_match:
        return "LEGACY_OR_SCHEMA_MISMATCH"
    return (
        "CURRENT_V2"
        if contract_ready
        else "CURRENT_V2_EVIDENCE_GAP"
    )


def _require_success(
        completed: subprocess.CompletedProcess,
        name: str,
        state_path: Path,
        state: Dict[str, Any]) -> None:
    if completed.returncode != 0:
        _update_state(
            state_path, state, "FAILED",
            failure_stage=name,
            failure_reason=f"returncode={completed.returncode}",
        )
        raise RuntimeError(
            f"Post-generation step {name} failed with "
            f"returncode={completed.returncode}.")


def _require_fresh_generation_artifacts(
        dataset_path: Path,
        profile_path: Path,
        generator_create_time: float,
        generator_exit_code: int) -> Dict[str, Any]:
    """Prove that this generator, rather than a stale artifact, succeeded."""
    if int(generator_exit_code) != 0:
        raise RuntimeError(
            "Dataset generator failed with exit code "
            f"{int(generator_exit_code)}.")
    not_before = float(generator_create_time) - 2.0
    artifacts = {}
    for name, path in (
        ("dataset", dataset_path),
        ("profile", profile_path),
    ):
        if not path.is_file():
            raise RuntimeError(
                f"Dataset generator did not write required {name}: {path}")
        stat = path.stat()
        if float(stat.st_mtime) < not_before:
            raise RuntimeError(
                f"Refusing stale {name} artifact written at "
                f"{stat.st_mtime:.6f}; generator started at "
                f"{generator_create_time:.6f}.")
        artifacts[name] = {
            "path": str(path),
            "size_bytes": int(stat.st_size),
            "mtime_unix": float(stat.st_mtime),
        }
    return artifacts


def run(args: argparse.Namespace) -> int:
    state_path = Path(args.state).resolve()
    run_dir = (
        EXPERIMENT_OUTPUT_ROOT
        / EXPERIMENT_ID / f"seed{int(args.seed)}"
    )
    state: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "STARTING",
        "generator_pid": int(args.wait_pid),
        "seed": int(args.seed),
        "experiment_id": EXPERIMENT_ID,
        "run_dir": str(run_dir),
        "started_unix_time": float(time.time()),
        "steps": [],
    }
    _write_json_atomic(state_path, state)

    generator_exit_code = None
    try:
        generator = psutil.Process(int(args.wait_pid))
        state["generator_create_time"] = float(generator.create_time())
        _update_state(state_path, state, "WAITING_FOR_GENERATION")
        generator_exit_code = int(generator.wait())
        _update_state(
            state_path, state, "GENERATOR_EXITED",
            generator_exit_code=generator_exit_code,
        )
    except psutil.NoSuchProcess:
        _update_state(
            state_path, state, "FAILED",
            failure_stage="wait_for_generation",
            failure_reason=(
                "generator PID was unavailable; exit status and artifact "
                "freshness cannot be bound to this run"),
        )
        raise RuntimeError(
            "Generator process is unavailable; refusing to validate a "
            "possibly stale dataset.")

    python = str(Path(sys.executable).resolve())
    dataset_path = OUTPUT_ROOT / "damage_dataset.parquet"
    profile_path = OUTPUT_ROOT / "generation_profile.json"
    try:
        fresh_artifacts = _require_fresh_generation_artifacts(
            dataset_path,
            profile_path,
            float(state["generator_create_time"]),
            int(generator_exit_code),
        )
    except RuntimeError as exc:
        _update_state(
            state_path, state, "FAILED",
            failure_stage="generation_artifact_binding",
            failure_reason=str(exc),
        )
        raise
    _update_state(
        state_path, state, "GENERATION_ARTIFACTS_BOUND",
        generation_artifacts=fresh_artifacts,
    )
    audit_path = OUTPUT_ROOT / "stage0_dataset_audit_post_generation.json"
    validation = _run_step(
        "validate_dataset",
        [
            python, "-m",
            "loitering_munition_damage_twin.stage0.validation",
            str(dataset_path),
        ],
        state_path, state,
    )
    _require_success(
        validation, "validate_dataset", state_path, state)

    audit = _run_step(
        "audit_dataset",
        [
            python, "-m", "loitering_munition_damage_twin.stage0.audit",
            str(dataset_path),
            "--output", str(audit_path),
        ],
        state_path, state,
    )
    _require_success(audit, "audit_dataset", state_path, state)
    audit_payload = _load_json(audit_path)
    contract_status = _audit_contract_status(audit_payload)
    if contract_status != "CURRENT_V2":
        _update_state(
            state_path, state, "FAILED",
            failure_stage="audit_dataset",
            failure_reason=(
                "contract_status="
                f"{contract_status!r}"),
        )
        raise RuntimeError("Dataset audit is not CURRENT_V2.")

    if run_dir.exists():
        _update_state(
            state_path, state, "FAILED",
            failure_stage="train_a40",
            failure_reason=(
                "A40 seed run directory already exists; refusing to "
                "overwrite an experiment artifact"),
        )
        raise RuntimeError(
            f"A40 run directory already exists: {run_dir}")

    training = _run_step(
        "train_a40",
        [
            python, "-m",
            "loitering_munition_damage_twin.experiments.run_ablations",
            "--configs", CONFIG_NAME,
            "--seeds", str(int(args.seed)),
            "--train-only", "--fail-fast",
        ],
        state_path, state,
    )
    _require_success(training, "train_a40", state_path, state)

    promotion = _run_step(
        "promote_a40_validation",
        [
            python, "-m",
            "loitering_munition_damage_twin.experiments.promote_strict_validation_goal",
            "--run-dir", str(run_dir),
            "--candidate", EXPERIMENT_ID,
        ],
        state_path, state,
    )
    if promotion.returncode != 0:
        _update_state(
            state_path, state, "VALIDATION_GOAL_NOT_MET",
            failure_stage="promote_a40_validation",
            failure_reason=(
                "A40 remains test-sealed because strict validation "
                "promotion did not pass"),
        )
        return 2

    if not args.allow_test_evaluation:
        _update_state(
            state_path, state, "VALIDATION_PROMOTED_TEST_SEALED",
            completed_unix_time=float(time.time()),
            test_evaluation_authorized=False,
        )
        return 0

    evaluation = _run_step(
        "evaluate_a40_test",
        [
            python, "-m",
            "loitering_munition_damage_twin.experiments.run_ablations",
            "--configs", CONFIG_NAME,
            "--seeds", str(int(args.seed)),
            "--eval-only", "--allow-test-evaluation", "--fail-fast",
        ],
        state_path, state,
    )
    _require_success(
        evaluation, "evaluate_a40_test", state_path, state)

    strict_test = _run_step(
        "validate_a40_strict_test_goal",
        [
            python, "-m",
            "loitering_munition_damage_twin.experiments.validate_strict_performance_goal",
            "--run-dir", str(run_dir),
        ],
        state_path, state,
    )
    if strict_test.returncode != 0:
        _update_state(
            state_path, state, "TEST_GOAL_NOT_MET",
            failure_stage="validate_a40_strict_test_goal",
            failure_reason=(
                "sealed test evaluation completed but strict goal failed"),
        )
        return 3

    _update_state(
        state_path, state, "COMPLETE",
        completed_unix_time=float(time.time()),
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for Stage-0 generation, validate it, train A40, and open "
            "the test split only after strict validation promotion."))
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--state", default=str(DEFAULT_STATE))
    parser.add_argument(
        "--allow-test-evaluation",
        action="store_true",
        help=(
            "After a validation promotion PASS, explicitly authorize the "
            "one-time held-out test reveal. Omit to keep test sealed."
        ),
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
