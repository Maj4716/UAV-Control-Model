# -*- coding: utf-8 -*-
import os
from pathlib import Path
import tempfile
import unittest

from loitering_munition_damage_twin.stage0.post_generation import (
    _audit_contract_status,
    _require_fresh_generation_artifacts,
)


class PostGenerationPipelineTests(unittest.TestCase):
    def test_nested_current_v2_audit_payload_is_accepted(self):
        payload = {
            "rows": 300000,
            "statistics": {
                "artifact_identity": {"current_schema_match": True},
                "exact_level_evidence": {"contract_ready": True},
            },
        }
        self.assertEqual(_audit_contract_status(payload), "CURRENT_V2")

    def test_nested_evidence_gap_is_not_promoted(self):
        payload = {
            "statistics": {
                "artifact_identity": {"current_schema_match": True},
                "exact_level_evidence": {"contract_ready": False},
            },
        }
        self.assertEqual(
            _audit_contract_status(payload),
            "CURRENT_V2_EVIDENCE_GAP",
        )

    def test_malformed_nested_audit_payload_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "exact-level evidence"):
            _audit_contract_status({
                "statistics": {
                    "artifact_identity": {"current_schema_match": True},
                },
            })

    def test_fresh_artifacts_are_bound_to_successful_generator(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir) / "damage_dataset.parquet"
            profile = Path(temp_dir) / "generation_profile.json"
            dataset.write_bytes(b"parquet")
            profile.write_text("{}", encoding="utf-8")
            started = min(dataset.stat().st_mtime, profile.stat().st_mtime)
            report = _require_fresh_generation_artifacts(
                dataset, profile, started, 0)
        self.assertEqual(report["dataset"]["size_bytes"], 7)
        self.assertEqual(report["profile"]["size_bytes"], 2)

    def test_stale_dataset_is_rejected_even_when_old_contract_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir) / "damage_dataset.parquet"
            profile = Path(temp_dir) / "generation_profile.json"
            dataset.write_bytes(b"old-parquet")
            profile.write_text("{}", encoding="utf-8")
            old_time = 1_700_000_000.0
            os.utime(dataset, (old_time, old_time))
            os.utime(profile, (old_time, old_time))
            with self.assertRaisesRegex(RuntimeError, "stale dataset"):
                _require_fresh_generation_artifacts(
                    dataset, profile, old_time + 60.0, 0)

    def test_nonzero_generator_exit_is_rejected_before_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir) / "damage_dataset.parquet"
            profile = Path(temp_dir) / "generation_profile.json"
            dataset.write_bytes(b"old-parquet")
            profile.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "exit code 1"):
                _require_fresh_generation_artifacts(
                    dataset, profile, 0.0, 1)


if __name__ == "__main__":
    unittest.main()
