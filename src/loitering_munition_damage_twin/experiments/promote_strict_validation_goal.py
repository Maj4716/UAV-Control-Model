from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict


REPORT_SCHEMA = "stage0_nn_strict_validation_promotion_v1"
VALIDATION_SCHEMA = "stage0_nn_validation_selection_v2"
MODEL_MANIFEST_SCHEMA = "stage0_nn_artifact_v1"
REQUIRED_MODEL_ARTIFACTS = (
    "best_model.pth",
    "best_thresholds.json",
    "minmax_scaler.pkl",
    "minmax_scaler.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    os.replace(str(temporary), str(path))


def _load_json_object(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def _verify_run_artifacts(
        run_dir: Path,
        validation_report: Dict[str, Any],
        candidate: str) -> Dict[str, Any]:
    """Recompute the exact artifact identity before test unsealing."""
    failures = []
    run_dir = run_dir.resolve()
    run_manifest_path = run_dir / "run_manifest.json"
    model_dir = run_dir / "output" / "models"
    model_manifest_path = model_dir / "model_manifest.json"
    verification: Dict[str, Any] = {
        "run_manifest_path": str(run_manifest_path),
        "model_manifest_path": str(model_manifest_path),
        "candidate": str(candidate),
        "artifacts": {},
    }

    run_manifest: Dict[str, Any] = {}
    try:
        run_manifest = _load_json_object(run_manifest_path)
    except (OSError, ValueError, RuntimeError) as exc:
        failures.append(f"cannot read run manifest: {exc}")
    run_experiment_id = run_manifest.get("experiment_id")
    verification["run_experiment_id"] = run_experiment_id
    if run_experiment_id != str(candidate):
        failures.append(
            "run manifest experiment_id does not match candidate: "
            f"{run_experiment_id!r} != {str(candidate)!r}")

    model_manifest: Dict[str, Any] = {}
    try:
        model_manifest = _load_json_object(model_manifest_path)
    except (OSError, ValueError, RuntimeError) as exc:
        failures.append(f"cannot read model manifest: {exc}")
    verification["model_manifest_schema"] = model_manifest.get("schema")
    if model_manifest.get("schema") != MODEL_MANIFEST_SCHEMA:
        failures.append(
            "model manifest schema mismatch: "
            f"{model_manifest.get('schema')!r} != {MODEL_MANIFEST_SCHEMA!r}")

    manifest_artifacts = model_manifest.get("artifacts", {})
    actual_hashes: Dict[str, str] = {}
    for name in REQUIRED_MODEL_ARTIFACTS:
        path = model_dir / name
        expected = (
            manifest_artifacts.get(name, {})
            if isinstance(manifest_artifacts, dict) else {}
        )
        record: Dict[str, Any] = {
            "path": str(path),
            "exists": path.is_file(),
            "expected_sha256": expected.get("sha256"),
            "expected_size_bytes": expected.get("size_bytes"),
        }
        if not path.is_file():
            failures.append(f"missing sealed model artifact: {name}")
        else:
            try:
                record["actual_size_bytes"] = int(path.stat().st_size)
                record["actual_sha256"] = _sha256(path)
                actual_hashes[name] = str(record["actual_sha256"])
                if record["expected_size_bytes"] != record["actual_size_bytes"]:
                    failures.append(f"model artifact size mismatch: {name}")
                if record["expected_sha256"] != record["actual_sha256"]:
                    failures.append(f"model artifact SHA-256 mismatch: {name}")
            except OSError as exc:
                failures.append(f"cannot hash model artifact {name}: {exc}")
        verification["artifacts"][name] = record

    data_contract = model_manifest.get("data_contract", {})
    if not isinstance(data_contract, dict):
        data_contract = {}
        failures.append("model manifest data_contract is not an object")
    dataset_value = data_contract.get("dataset_path")
    dataset_path = Path(str(dataset_value)) if dataset_value else None
    if dataset_path is not None and not dataset_path.is_absolute():
        dataset_path = (run_dir / dataset_path).resolve()
    verification["dataset"] = {
        "path": str(dataset_path) if dataset_path is not None else None,
        "expected_sha256": data_contract.get("dataset_sha256"),
    }
    dataset_hash = None
    if dataset_path is None or not dataset_path.is_file():
        failures.append(
            f"sealed dataset path is missing or invalid: {dataset_path}")
    else:
        try:
            dataset_hash = _sha256(dataset_path)
            verification["dataset"]["actual_sha256"] = dataset_hash
            verification["dataset"]["actual_size_bytes"] = int(
                dataset_path.stat().st_size)
            if data_contract.get("dataset_sha256") != dataset_hash:
                failures.append("dataset SHA-256 differs from model data contract")
        except OSError as exc:
            failures.append(f"cannot hash sealed dataset: {exc}")

    identity = validation_report.get("artifact_identity", {})
    expected_identity = {
        "dataset_sha256": dataset_hash,
        "model_sha256": actual_hashes.get("best_model.pth"),
        "threshold_sha256": actual_hashes.get("best_thresholds.json"),
    }
    verification["validation_report_identity"] = identity
    verification["recomputed_identity"] = expected_identity
    for key, actual_value in expected_identity.items():
        if actual_value is None:
            failures.append(f"cannot recompute artifact identity: {key}")
        elif not isinstance(identity, dict) or identity.get(key) != actual_value:
            failures.append(
                f"validation report artifact identity mismatch: {key}")

    verification["status"] = "PASS" if not failures else "FAIL"
    verification["failure_count"] = len(failures)
    verification["failures"] = failures
    return verification


def _assert_test_is_sealed(run_dir: Path) -> None:
    forbidden = (
        run_dir / "output" / "eval" / "test_metrics.json",
        run_dir / "output" / "eval" / "predictions.csv",
        run_dir / "output" / "eval" / "strict_performance_goal.json",
    )
    present = [str(path) for path in forbidden if path.exists()]
    if present:
        raise RuntimeError(
            "Test split is already unsealed; validation promotion refused: "
            + ", ".join(present))


def evaluate_validation_promotion(
        validation_report: Dict[str, Any],
        candidate: str,
        report_path: Path) -> Dict[str, Any]:
    failures = []
    if validation_report.get("schema") != VALIDATION_SCHEMA:
        failures.append(
            f"validation schema must be {VALIDATION_SCHEMA}")
    if validation_report.get("split") != "validation":
        failures.append("report split is not validation")
    if validation_report.get("test_labels_used") is not False:
        failures.append("validation report does not seal test labels")

    performance_gate = validation_report.get("performance_gate", {})
    goal_gate = validation_report.get("goal_performance_gate", {})
    if performance_gate.get("passed") is not True:
        failures.append(
            "validation safety/historical performance gate did not pass")
    if goal_gate.get("passed") is not True:
        status = str(goal_gate.get("status", "MISSING"))
        metric_failures = int(goal_gate.get("metric_failure_count", -1))
        evidence_failures = int(goal_gate.get("evidence_failure_count", -1))
        failures.append(
            "strict validation goal did not pass: "
            f"status={status}, metric_failures={metric_failures}, "
            f"evidence_failures={evidence_failures}")

    identity = validation_report.get("artifact_identity", {})
    for key in ("dataset_sha256", "model_sha256", "threshold_sha256"):
        value = identity.get(key)
        if not isinstance(value, str) or len(value) != 64:
            failures.append(f"missing or invalid artifact identity: {key}")

    return {
        "schema": REPORT_SCHEMA,
        "status": "PASS" if not failures else "FAIL",
        "candidate": str(candidate),
        "split": "validation",
        "test_metrics_read": False,
        "criteria": {
            "performance_gate_must_pass": True,
            "strict_goal_gate_must_pass": True,
            "full_artifact_identity_required": True,
            "recomputed_run_artifacts_required": True,
            "test_split_must_be_sealed": True,
        },
        "validation": {
            "report_path": str(report_path),
            "report_sha256": _sha256(report_path),
            "average_3class_accuracy_percent": validation_report.get(
                "average_3class_accuracy_percent"),
            "performance_gate": performance_gate,
            "goal_performance_gate": goal_gate,
            "artifact_identity": identity,
        },
        "failure_count": len(failures),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a test-blind promotion credential only when the complete "
            "strict 94%/90% validation goal and safety gates pass."))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--validation-report", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    _assert_test_is_sealed(run_dir)
    report_path = (
        Path(args.validation_report).resolve()
        if args.validation_report
        else run_dir / "output" / "validation" / "selection_metrics.json"
    )
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    try:
        report_path.relative_to(run_dir)
    except ValueError as exc:
        raise RuntimeError(
            "Validation report must be inside the candidate run directory.") from exc
    validation_report = _load_json_object(report_path)

    promotion = evaluate_validation_promotion(
        validation_report, str(args.candidate), report_path)
    artifact_verification = _verify_run_artifacts(
        run_dir, validation_report, str(args.candidate))
    promotion["artifact_verification"] = artifact_verification
    promotion["failures"].extend(artifact_verification["failures"])
    promotion["failure_count"] = len(promotion["failures"])
    promotion["status"] = (
        "PASS" if promotion["failure_count"] == 0 else "FAIL")
    output_path = (
        Path(args.output).resolve()
        if args.output
        else run_dir / "output" / "validation"
        / "strict_validation_promotion.json"
    )
    try:
        output_path.relative_to(run_dir)
    except ValueError as exc:
        raise RuntimeError(
            "Promotion report must be written inside the run directory.") from exc
    _write_json_atomic(output_path, promotion)
    print(json.dumps({
        "status": promotion["status"],
        "failure_count": promotion["failure_count"],
        "test_metrics_read": False,
        "output": str(output_path),
    }, ensure_ascii=False, indent=2))
    return 0 if promotion["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
