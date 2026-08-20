import copy
import json
import hashlib
import inspect
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch

    from loitering_munition_damage_twin.surrogate.artifacts import (
        data_contracts_match,
        load_and_verify_manifest,
        write_model_manifest,
    )
    from loitering_munition_damage_twin.surrogate.dataset import (
        BalancedMunitionBatchSampler,
        _apply_label_confidence_strength,
        _build_mechanism_targets,
        _build_ordinal_targets,
        _load_component_targets,
        _validate_dataset_usability,
        get_dataloaders,
        get_feature_columns,
    )
    from loitering_munition_damage_twin.stage0.component_supervision import (
        COMPONENT_SUPERVISION_PROFILE_FILENAME,
        COMPONENT_TARGET_COLUMNS,
        CRITICAL_COMPONENT_IDS,
        build_component_supervision_profile,
        sha256_file,
        sha256_text_sequence,
    )
    from loitering_munition_damage_twin.surrogate.features import (
        COMBINED_TASK_RULE_PROXY_FEATURE_COLUMNS,
        COMPONENT_PROXY_FEATURE_COLUMNS,
        TERMINAL_PHYSICS_FEATURE_COLUMNS,
        TERMINAL_PHYSICS_FEATURE_VERSION,
        augment_terminal_physics_features,
        terminal_physics_contract_metadata,
    )
    from loitering_munition_damage_twin.surrogate.model import (
        DEFAULT_ORDINAL_APPLICABILITY,
        DamageAssessmentMTL,
        component_probabilities_to_ordinal,
    )
    from loitering_munition_damage_twin.surrogate.evaluation import (
        _authorize_test_evaluation,
        _load_thresholds,
        _onnx_parity_stats,
    )
    from loitering_munition_damage_twin.surrogate.training import (
        FocalUncertaintyOrdinalLoss,
        _confidence_resolved_diagnostics,
        _validation_report_from_metrics,
        hard_negative_pairwise_ranking_loss,
        component_auxiliary_loss,
        mechanism_auxiliary_loss,
        _goal_candidate_sort_key,
        _insert_topk_candidate,
        _minimum_supported_diagonal_recall,
        _minimum_supported_class1_recall,
        _search_l1_threshold,
        _search_joint_ordinal_thresholds,
        ordinal_class_distribution_nll,
    )
    from loitering_munition_damage_twin.experiments.ablation_config import load_ablation_config
    from loitering_munition_damage_twin.experiments.compare_performance_ablations import (
        _aggregate,
        _delta,
        _flatten_metrics,
        _validate_completed_result,
    )
    from loitering_munition_damage_twin.experiments.run_ablations import (
        _build_evaluation_command,
        _passed_validation_promotion,
        _should_echo_summary_line,
        _write_run_manifest,
    )
    from loitering_munition_damage_twin.experiments.promote_strict_validation_goal import (
        evaluate_validation_promotion,
        _verify_run_artifacts,
    )
    from loitering_munition_damage_twin.experiments.validate_a22_promotion import _evaluate_promotion
    from loitering_munition_damage_twin.experiments.validate_a23_promotion import (
        _evaluate_promotion as _evaluate_a23_promotion,
    )
    from loitering_munition_damage_twin.experiments.validate_a34_promotion import (
        evaluate_a34_promotion,
    )
    from loitering_munition_damage_twin.experiments.validate_a36_promotion import (
        evaluate_a36_promotion,
    )
    from loitering_munition_damage_twin.experiments.validate_a37_promotion import (
        evaluate_a37_promotion,
    )
    from loitering_munition_damage_twin.experiments.validate_a38_promotion import (
        evaluate_a38_promotion,
    )
    from loitering_munition_damage_twin.experiments.validate_multiseed_ensemble import (
        EqualWeightProbabilityEnsemble,
        _evaluate_promotion as _evaluate_ensemble_promotion,
        inference_contracts_match,
    )
    from loitering_munition_damage_twin.experiments.validate_strict_performance_goal import (
        ORDINAL_APPLICABILITY as STRICT_GOAL_ORDINAL_APPLICABILITY,
        evaluate_strict_goal,
    )
    from loitering_munition_damage_twin.experiments.summarize_multiseed_reliability import _summary_stats
    from loitering_munition_damage_twin.experiments.analyze_single_seed_predictions import (
        _cluster_bootstrap_mean_delta,
        _safe_partial_auc,
    )
    from loitering_munition_damage_twin.experiments.analyze_validation_threshold_feasibility import (
        evaluate_cell_threshold_feasibility,
        _terminal_combined_rule_proxy_probabilities,
        _terminal_rule_proxy_probabilities,
    )
    from loitering_munition_damage_twin.simulation.engine import DamageTreeEvaluator

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch environment is not active")
class NeuralPipelineContractTests(unittest.TestCase):
    def test_validation_only_loader_contract_can_keep_test_labels_sealed(self):
        parameter = inspect.signature(get_dataloaders).parameters[
            "load_test_split"]
        self.assertIs(parameter.default, True)

        profile = {
            "usability_gate": {"enforced": True, "passed": True},
            "ordinal_applicability": {},
        }
        targets = np.ones((4, 4, 2), dtype=np.float32)
        munitions = np.arange(4, dtype=np.int64)
        captured = io.StringIO()
        with redirect_stdout(captured):
            _validate_dataset_usability(
                profile, targets, munitions,
                y_test=None, mun_test=None)
        self.assertNotIn("Test m_id=", captured.getvalue())

    def test_soft_targets_retain_probability_below_hard_threshold(self):
        frame = pd.DataFrame({
            "label_mc_replicates": [9, 9],
            **{
                f"{task}_level": [0, 2]
                for task in ("K", "M", "F", "C")
            },
            **{
                f"{task}_ge{level}_prob": [0.30, 0.80 if level == 1 else 0.70]
                for task in ("K", "M", "F", "C")
                for level in (1, 2)
            },
            **{
                f"{task}_ge{level}_prob_std": [0.15, 0.03]
                for task in ("K", "M", "F", "C")
                for level in (1, 2)
            },
        })
        hard, soft, confidence = _build_ordinal_targets(
            frame, use_label_uncertainty=True)
        self.assertEqual(float(hard[0, 0, 0]), 0.0)
        self.assertAlmostEqual(float(soft[0, 0, 0]), 0.30, places=6)
        self.assertTrue(np.all(soft[:, :, 1] <= soft[:, :, 0]))
        self.assertTrue(np.all((confidence >= 0.25) & (confidence <= 1.0)))
        self.assertLess(float(confidence[0, 0, 0]),
                        float(confidence[1, 0, 0]))

    def test_unresolved_mc_head_retains_soft_target_at_confidence_floor(self):
        frame = pd.DataFrame({
            "label_mc_replicates": [64, 64],
            **{
                f"{task}_level": [1, 1]
                for task in ("K", "M", "F", "C")
            },
            **{
                f"{task}_ge{level}_prob": [0.51, 0.51]
                for task in ("K", "M", "F", "C")
                for level in (1, 2)
            },
            **{
                f"{task}_ge{level}_prob_std": [0.08, 0.08]
                for task in ("K", "M", "F", "C")
                for level in (1, 2)
            },
            **{
                f"{task}_ge{level}_mc_standard_error": [0.01, 0.01]
                for task in ("K", "M", "F", "C")
                for level in (1, 2)
            },
            **{
                f"{task}_ge{level}_mc_resolved": [False, True]
                for task in ("K", "M", "F", "C")
                for level in (1, 2)
            },
        })
        _, soft, confidence = _build_ordinal_targets(
            frame,
            use_label_uncertainty=True,
            uncertainty_scale=0.10,
            confidence_floor=0.25,
        )
        self.assertAlmostEqual(
            float(soft[0, 0, 0]), 0.51, places=6)
        self.assertAlmostEqual(
            float(confidence[0, 0, 0]), 0.25, places=6)
        self.assertGreater(
            float(confidence[1, 0, 0]), 0.90)

    def test_feature_ablation_changes_input_and_leakage_is_rejected(self):
        baseline = get_feature_columns()
        reduced = get_feature_columns(
            {"data": {"drop_features": ["norm_velocity"]}})
        self.assertEqual(len(baseline), 13)
        self.assertEqual(len(reduced), 12)
        with self.assertRaises(ValueError):
            get_feature_columns(
                {"data": {"extra_features": ["impact_cosine"]}})
        physics = get_feature_columns(
            {"data": {"use_terminal_physics_features": True}})
        self.assertEqual(
            physics,
            baseline + list(TERMINAL_PHYSICS_FEATURE_COLUMNS))
        self.assertEqual(len(physics), 194)
        component_extended = get_feature_columns({
            "data": {
                "use_terminal_physics_features": True,
                "use_component_proxy_features": True,
            },
        })
        self.assertEqual(
            component_extended,
            physics + list(COMPONENT_PROXY_FEATURE_COLUMNS))
        self.assertEqual(len(component_extended), 296)
        with self.assertRaises(ValueError):
            get_feature_columns({
                "data": {
                    "use_component_proxy_features": True,
                },
            })
        with self.assertRaises(ValueError):
            get_feature_columns({
                "data": {
                    "use_terminal_physics_features": True,
                    "use_armor_aware_fragment_proxies": True,
                },
            })

    def test_terminal_physics_v2_is_finite_deterministic_and_deployable(self):
        frame = pd.DataFrame({
            "x_cm": [100.0, -75.0, 40.0, 10.0],
            "y_cm": [20.0, 80.0, -30.0, 15.0],
            "z_cm": [120.0, 90.0, 70.0, 200.0],
            "vx_ms": [10.0, -20.0, 5.0, 0.0],
            "vy_ms": [80.0, 70.0, -90.0, 60.0],
            "vz_ms": [-15.0, -10.0, -20.0, -30.0],
            "sin_yaw": [0.0, 0.5, -0.5, 1.0],
            "cos_yaw": [1.0, np.sqrt(0.75), np.sqrt(0.75), 0.0],
            "sin_pitch": [0.0, -0.5, -0.25, -1.0],
            "cos_pitch": [1.0, np.sqrt(0.75), np.sqrt(0.9375), 0.0],
            "sin_roll": [0.0, 0.2, -0.2, 0.0],
            "cos_roll": [1.0, np.sqrt(0.96), np.sqrt(0.96), 1.0],
            "m_id": [0, 1, 2, 3],
        })
        first = augment_terminal_physics_features(frame)
        second = augment_terminal_physics_features(frame)
        values = first[
            TERMINAL_PHYSICS_FEATURE_COLUMNS].to_numpy()
        self.assertEqual(
            TERMINAL_PHYSICS_FEATURE_VERSION,
            "terminal_geometry_v2")
        self.assertEqual(values.shape, (4, 181))
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_allclose(
            values,
            second[TERMINAL_PHYSICS_FEATURE_COLUMNS].to_numpy(),
            rtol=0.0,
            atol=0.0,
        )
        contract = terminal_physics_contract_metadata()
        self.assertEqual(contract["derived_feature_count"], 181)
        self.assertEqual(contract["extensions"], [])
        extended_contract = terminal_physics_contract_metadata(
            include_component_proxies=True)
        self.assertEqual(
            extended_contract["derived_feature_count"], 283)
        self.assertEqual(
            extended_contract["extensions"][0]["feature_count"], 102)
        armor_contract = terminal_physics_contract_metadata(
            include_component_proxies=True,
            armor_aware_fragment_proxies=True,
        )
        self.assertTrue(
            armor_contract["extensions"][0][
                "armor_aware_fragment_proxies"])
        self.assertIn(
            "armor_aware_fragment",
            armor_contract["extensions"][0]["name"])
        with self.assertRaises(ValueError):
            terminal_physics_contract_metadata(
                armor_aware_fragment_proxies=True)
        self.assertNotIn("target_x", first.columns)
        self.assertNotIn("total_hits", first.columns)
        combined = augment_terminal_physics_features(
            frame,
            include_combined_rule_proxies=True,
            include_component_proxies=True,
        )
        combined_values = combined[
            COMBINED_TASK_RULE_PROXY_FEATURE_COLUMNS
        ].to_numpy()
        self.assertEqual(combined_values.shape, (4, 8))
        self.assertTrue(np.isfinite(combined_values).all())
        combined_probabilities = (
            _terminal_combined_rule_proxy_probabilities(
                combined)
        )
        self.assertEqual(
            combined_probabilities.shape, (4, 4, 2))
        self.assertTrue(np.all(
            combined_probabilities[..., 1]
            <= combined_probabilities[..., 0]))
        component_proxy_values = combined[
            COMPONENT_PROXY_FEATURE_COLUMNS].to_numpy()
        self.assertEqual(component_proxy_values.shape, (4, 102))
        self.assertTrue(np.isfinite(
            component_proxy_values).all())
        self.assertTrue(np.all(
            (component_proxy_values >= 0.0)
            & (component_proxy_values <= 1.0)))
        armor_aware = augment_terminal_physics_features(
            frame,
            include_component_proxies=True,
            armor_aware_fragment_proxies=True,
        )
        fragment_proxy_columns = [
            name for name in COMPONENT_PROXY_FEATURE_COLUMNS
            if "_fragment_" in name
        ]
        armor_values = armor_aware[
            fragment_proxy_columns].to_numpy()
        legacy_values = combined[
            fragment_proxy_columns].to_numpy()
        self.assertTrue(np.isfinite(armor_values).all())
        self.assertTrue(np.all(
            (armor_values >= 0.0) & (armor_values <= 1.0)))
        self.assertFalse(np.array_equal(
            armor_values, legacy_values))
        with self.assertRaises(ValueError):
            augment_terminal_physics_features(
                frame,
                armor_aware_fragment_proxies=True,
            )

    def test_mechanism_targets_are_bounded_monotone_mc_means(self):
        frame = pd.DataFrame({
            f"{mechanism}_{task}_ge{level}_prob": [
                0.75 if level == 1 else 0.25,
                0.60 if level == 1 else 0.10,
            ]
            for mechanism in ("fragment", "shock")
            for task in ("K", "M", "F", "C")
            for level in (1, 2)
        })
        targets = _build_mechanism_targets(frame)
        self.assertEqual(targets.shape, (2, 2, 4, 2))
        self.assertTrue(np.all(targets[..., 1] <= targets[..., 0]))
        broken = frame.copy()
        broken["fragment_K_ge2_prob"] = [0.9, 0.9]
        with self.assertRaises(RuntimeError):
            _build_mechanism_targets(broken)

    def test_component_supervision_contract_is_dense_and_not_an_input(self):
        self.assertEqual(len(CRITICAL_COMPONENT_IDS), 51)
        self.assertEqual(len(COMPONENT_TARGET_COLUMNS), 102)
        self.assertEqual(len(set(COMPONENT_TARGET_COLUMNS)), 102)
        active_features = get_feature_columns(
            load_ablation_config(
                "A34_component_physics_auxiliary"))
        self.assertTrue(
            set(active_features).isdisjoint(
                COMPONENT_TARGET_COLUMNS))

    def test_component_sidecar_loader_verifies_hash_and_sample_order(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = os.path.join(
                temp_dir, "damage_dataset.parquet")
            sample_ids = np.asarray(
                ["sample-a", "sample-b", "sample-c"])
            base_frame = pd.DataFrame({
                "sample_id": sample_ids,
            })
            base_frame.to_parquet(
                base_path, engine="pyarrow", index=False)
            sidecar_path = os.path.join(
                temp_dir, "component_supervision.parquet")
            sidecar = pd.DataFrame({
                "sample_id": sample_ids,
                **{
                    column: np.asarray(
                        [0.1, 0.2, 0.3],
                        dtype=np.float32)
                    for column in COMPONENT_TARGET_COLUMNS
                },
            })
            sidecar.to_parquet(
                sidecar_path, engine="pyarrow", index=False)
            sample_hash = sha256_text_sequence(sample_ids)
            parquet_file = pq.ParquetFile(sidecar_path)
            try:
                row_groups = parquet_file.num_row_groups
            finally:
                parquet_file.close()
            profile = build_component_supervision_profile(
                base_dataset_path=base_path,
                base_dataset_sha256=sha256_file(base_path),
                base_dataset_rows=len(base_frame),
                base_dataset_schema="stage0_lineage_v2",
                frame_convention="stage0_ned_frd_v1",
                sidecar_path=sidecar_path,
                sidecar_rows=len(sidecar),
                sidecar_size_bytes=os.path.getsize(sidecar_path),
                sidecar_sha256=sha256_file(sidecar_path),
                sample_id_order_sha256=sample_hash,
                parquet_row_groups=row_groups,
                pyarrow_version=pa.__version__,
                label_replay_verified=True,
            )
            profile_path = os.path.join(
                temp_dir,
                COMPONENT_SUPERVISION_PROFILE_FILENAME)
            with open(
                    profile_path, "w",
                    encoding="utf-8") as stream:
                json.dump(profile, stream)
            generation_profile = {
                "artifact": {
                    "sha256": sha256_file(base_path),
                },
            }
            targets, contract = _load_component_targets(
                base_frame,
                base_path,
                generation_profile,
                {},
            )
            self.assertEqual(tuple(targets.shape), (3, 2, 51))
            self.assertFalse(contract["model_input_allowed"])
            self.assertEqual(
                contract["sha256"], sha256_file(sidecar_path))

    def test_model_is_monotone_and_masks_structural_zeros(self):
        features = torch.randn(8, 13)
        munitions = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        for experts in (True, False):
            for parameterization in (
                    "cumulative_logits", "nominal_softmax"):
                model = DamageAssessmentMTL(
                    in_dim=13,
                    use_munition_experts=experts,
                    ordinal_parameterization=parameterization,
                )
                model.eval()
                with torch.no_grad():
                    logits = model(features, munitions)
                probabilities = torch.sigmoid(logits)
                self.assertTrue(torch.all(
                    probabilities[:, :, 1] <= probabilities[:, :, 0]))
                small = munitions == 0
                self.assertTrue(torch.all(logits[small, 0, 1] <= -29.0))
                self.assertTrue(torch.all(logits[small, 3, 1] <= -29.0))

    def test_mechanism_model_uses_exact_probability_or_fusion(self):
        model = DamageAssessmentMTL(
            in_dim=13,
            use_mechanism_decomposition=True,
        )
        model.eval()
        features = torch.randn(12, 13)
        munitions = torch.tensor([0, 1, 2, 3] * 3)
        with torch.no_grad():
            combined, fragment, shock = (
                model.forward_with_mechanisms(features, munitions))
            public = model(features, munitions)
        self.assertTrue(torch.equal(combined, public))
        applicability = model.ordinal_applicability[munitions]
        expected_probability = (
            1.0
            - (1.0 - torch.sigmoid(fragment.float()))
            * (1.0 - torch.sigmoid(shock.float()))
        )
        self.assertTrue(torch.allclose(
            torch.sigmoid(combined)[applicability],
            expected_probability[applicability],
            atol=2e-6,
            rtol=2e-6,
        ))
        self.assertTrue(torch.all(
            combined[..., 1] <= combined[..., 0]))
        self.assertTrue(torch.all(combined[~applicability] <= -29.0))
        with self.assertRaises(ValueError):
            DamageAssessmentMTL(
                in_dim=13,
                use_mechanism_decomposition=True,
                residual_adapter_cells=[[0, 0, 0]],
            )

    def test_auxiliary_mechanism_heads_preserve_direct_combined_path(self):
        torch.manual_seed(20260729)
        model = DamageAssessmentMTL(
            in_dim=13,
            use_mechanism_auxiliary_heads=True,
        )
        model.eval()
        features = torch.randn(12, 13)
        munitions = torch.tensor([0, 1, 2, 3] * 3)
        with torch.no_grad():
            combined, fragment, shock = (
                model.forward_with_mechanisms(features, munitions))
            public = model(features, munitions)
        self.assertTrue(model.has_mechanism_outputs)
        self.assertFalse(model.use_mechanism_decomposition)
        self.assertTrue(torch.equal(combined, public))
        self.assertEqual(tuple(fragment.shape), (12, 4, 2))
        self.assertEqual(tuple(shock.shape), (12, 4, 2))
        self.assertTrue(torch.all(combined[..., 1] <= combined[..., 0]))
        applicability = model.ordinal_applicability[munitions]
        forced_or = (
            1.0
            - (1.0 - torch.sigmoid(fragment.float()))
            * (1.0 - torch.sigmoid(shock.float()))
        )
        # The auxiliary-only contract keeps an independently parameterized
        # direct combined head instead of silently applying fixed OR fusion.
        self.assertGreater(float(torch.max(torch.abs(
            torch.sigmoid(combined)[applicability]
            - forced_or[applicability]
        ))), 1e-5)
        with self.assertRaises(ValueError):
            DamageAssessmentMTL(
                in_dim=13,
                use_mechanism_decomposition=True,
                use_mechanism_auxiliary_heads=True,
            )

    def test_component_auxiliary_head_preserves_direct_deployment_path(self):
        torch.manual_seed(20260729)
        model = DamageAssessmentMTL(
            in_dim=13,
            use_component_auxiliary_heads=True,
        )
        model.eval()
        features = torch.randn(12, 13)
        munitions = torch.tensor([0, 1, 2, 3] * 3)
        with torch.no_grad():
            combined, component_logits = (
                model.forward_with_components(
                    features, munitions))
            public = model(features, munitions)
        self.assertTrue(model.has_component_outputs)
        self.assertTrue(torch.equal(combined, public))
        self.assertEqual(
            tuple(component_logits.shape), (12, 2, 51))
        with self.assertRaises(ValueError):
            DamageAssessmentMTL(
                in_dim=13,
                use_mechanism_auxiliary_heads=True,
                use_component_auxiliary_heads=True,
            )

    def test_component_auxiliary_head_does_not_change_direct_initialization(self):
        model_kwargs = {
            "in_dim": 13,
            "num_munitions": 4,
            "munition_emb_dim": 16,
            "use_munition_embedding": True,
            "use_munition_experts": True,
            "use_physics_skip": True,
            "use_k_cascade": True,
            "deep_m_branch": True,
        }
        torch.manual_seed(314159)
        baseline = DamageAssessmentMTL(
            **model_kwargs,
            use_component_auxiliary_heads=False,
        )
        torch.manual_seed(314159)
        candidate = DamageAssessmentMTL(
            **model_kwargs,
            use_component_auxiliary_heads=True,
        )
        baseline_state = baseline.state_dict()
        candidate_state = candidate.state_dict()
        for name, value in baseline_state.items():
            self.assertIn(name, candidate_state)
            self.assertTrue(
                torch.equal(value, candidate_state[name]),
                msg=f"Direct-path initialization differs at {name}",
            )

    def test_independent_component_tree_fusion_preserves_frozen_direct_path(self):
        fusion = [
            [[0.15, 0.0], [0.0, 0.0], [0.0, 0.0], [0.50, 0.0]],
            [[0.45, 0.0], [0.0, 0.0], [0.0, 0.0], [0.60, 0.0]],
            [[0.35, 0.0], [0.0, 0.25], [0.0, 0.0], [0.50, 0.0]],
            [[0.50, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ]
        torch.manual_seed(271828)
        model = DamageAssessmentMTL(
            in_dim=13,
            use_component_auxiliary_heads=True,
            component_branch_mode="independent_experts",
            component_tree_fusion_alpha=fusion,
        )
        features = torch.randn(16, 13)
        munitions = torch.tensor([0, 1, 2, 3] * 4)
        model.eval()
        with torch.no_grad():
            direct, _, _ = model.forward_with_mechanisms(
                features, munitions)
            fused, component_logits = model.forward_with_components(
                features, munitions)
            public = model(features, munitions)
        self.assertTrue(torch.equal(fused, public))
        self.assertEqual(tuple(component_logits.shape), (16, 2, 51))
        self.assertTrue(torch.all(
            torch.sigmoid(fused)[..., 1]
            <= torch.sigmoid(fused)[..., 0]))
        # F has zero fusion alpha in every munition and must remain bitwise
        # identical to the frozen direct surrogate.
        self.assertTrue(torch.equal(fused[:, 2], direct[:, 2]))
        self.assertGreater(float(torch.max(torch.abs(
            torch.sigmoid(fused[:, 0, 0])
            - torch.sigmoid(direct[:, 0, 0])
        ))), 1e-6)

        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.independent_component_branch.parameters():
            parameter.requires_grad_(True)
        model.train()
        fused, component_logits = model.forward_with_components(
            features, munitions)
        (fused.square().mean()
         + component_logits.square().mean()).backward()
        direct_gradients = [
            parameter.grad for name, parameter in model.named_parameters()
            if not name.startswith("independent_component_branch.")
        ]
        component_gradients = [
            parameter.grad for name, parameter in model.named_parameters()
            if name.startswith("independent_component_branch.")
        ]
        self.assertTrue(all(value is None for value in direct_gradients))
        self.assertTrue(any(
            value is not None and torch.isfinite(value).all()
            for value in component_gradients))

    def test_component_feature_extension_isolated_from_direct_path(self):
        torch.manual_seed(271829)
        baseline = DamageAssessmentMTL(
            in_dim=13,
            use_component_auxiliary_heads=True,
            component_branch_mode="independent_experts",
        )
        extended = DamageAssessmentMTL(
            in_dim=15,
            base_input_dim=13,
            use_component_auxiliary_heads=True,
            component_branch_mode="independent_experts",
        )
        source_state = baseline.state_dict()
        target_state = extended.state_dict()
        input_key = (
            "independent_component_branch.encoder.0.weight")
        for name, value in source_state.items():
            if name != input_key:
                target_state[name].copy_(value)
        target_state[input_key].zero_()
        target_state[input_key][:, :13].copy_(
            source_state[input_key][:, :13])
        target_state[input_key][:, 15:].copy_(
            source_state[input_key][:, 13:])
        extended.load_state_dict(target_state)
        baseline.eval()
        extended.eval()
        base_features = torch.randn(12, 13)
        tail_features = torch.randn(12, 2)
        munitions = torch.tensor([0, 1, 2, 3] * 3)
        with torch.no_grad():
            baseline_direct, baseline_components = (
                baseline.forward_with_components(
                    base_features, munitions))
            extended_direct, extended_components = (
                extended.forward_with_components(
                    torch.cat((base_features, tail_features), dim=1),
                    munitions))
        self.assertTrue(torch.equal(
            baseline_direct, extended_direct))
        self.assertTrue(torch.equal(
            baseline_components, extended_components))
        with self.assertRaises(ValueError):
            DamageAssessmentMTL(in_dim=13, base_input_dim=14)

    def test_explicit_direct_feature_extension_reaches_task_logits(self):
        torch.manual_seed(161803)
        model = DamageAssessmentMTL(
            in_dim=15,
            base_input_dim=15,
        )
        model.eval()
        first = torch.randn(8, 15)
        second = first.clone()
        second[:, 13:] += 2.0
        munitions = torch.tensor([0, 1, 2, 3] * 2)
        with torch.no_grad():
            first_logits = model(first, munitions)
            second_logits = model(second, munitions)
        self.assertFalse(torch.equal(first_logits, second_logits))

    def test_differentiable_component_tree_matches_simulator_rules(self):
        rng = np.random.default_rng(20260729)
        probabilities = rng.uniform(
            0.05, 0.95, size=(7, 51)).astype(np.float32)
        observed = component_probabilities_to_ordinal(
            torch.from_numpy(probabilities)).detach().numpy()
        evaluator = DamageTreeEvaluator()
        expected = []
        for row in probabilities:
            component_map = {
                int(component_id): float(row[index])
                for index, component_id in enumerate(
                    CRITICAL_COMPONENT_IDS)
            }
            expected.append(
                evaluator.evaluate(
                    component_map).ordinal_probability_vector)
        np.testing.assert_allclose(
            observed, np.asarray(expected).reshape(-1, 4, 2),
            rtol=2e-6, atol=2e-6)

    def test_mechanism_auxiliary_loss_rewards_matching_probabilities(self):
        target = torch.tensor(
            [[[[0.8, 0.2]] * 4, [[0.7, 0.1]] * 4],
             [[[0.6, 0.3]] * 4, [[0.9, 0.4]] * 4]],
            dtype=torch.float32,
        )
        matching = torch.logit(target.clamp(1e-5, 1.0 - 1e-5))
        reversed_logits = -matching
        applicability = torch.ones(2, 4, 2, dtype=torch.bool)
        weights = torch.ones(2)
        matching_loss = mechanism_auxiliary_loss(
            matching[:, 0], matching[:, 1], target,
            weights, applicability)
        reversed_loss = mechanism_auxiliary_loss(
            reversed_logits[:, 0], reversed_logits[:, 1], target,
            weights, applicability)
        self.assertLess(float(matching_loss), float(reversed_loss))

    def test_mechanism_loss_can_prioritize_fragment_boundary(self):
        target = torch.full((4, 2, 4, 2), 0.5)
        matching = torch.logit(target.clamp(1e-5, 1.0 - 1e-5))
        wrong = torch.full_like(matching, -4.0)
        fragment_correct = mechanism_auxiliary_loss(
            matching[:, 0], wrong[:, 1], target,
            torch.ones(4), torch.ones(4, 4, 2, dtype=torch.bool),
            class_distribution_weight=0.0,
            branch_weights=torch.tensor([3.0, 1.0]),
            boundary_focus_weight=2.0,
            hard_classification_weight=0.5,
            use_dataset_row_weights=False,
        )
        shock_correct = mechanism_auxiliary_loss(
            wrong[:, 0], matching[:, 1], target,
            torch.ones(4), torch.ones(4, 4, 2, dtype=torch.bool),
            class_distribution_weight=0.0,
            branch_weights=torch.tensor([3.0, 1.0]),
            boundary_focus_weight=2.0,
            hard_classification_weight=0.5,
            use_dataset_row_weights=False,
        )
        self.assertLess(float(fragment_correct), float(shock_correct))

    def test_mechanism_loss_default_is_backward_compatible(self):
        torch.manual_seed(271828)
        target = torch.rand(5, 2, 4, 2)
        target[..., 1] = torch.minimum(
            target[..., 1], target[..., 0])
        logits = torch.randn(5, 2, 4, 2)
        applicability = torch.ones(5, 4, 2, dtype=torch.bool)
        row_weights = torch.linspace(0.2, 2.0, 5)
        legacy_default = mechanism_auxiliary_loss(
            logits[:, 0], logits[:, 1], target,
            row_weights, applicability)
        explicit_default = mechanism_auxiliary_loss(
            logits[:, 0], logits[:, 1], target,
            row_weights, applicability,
            branch_weights=torch.ones(2),
            boundary_focus_weight=0.0,
            hard_classification_weight=0.0,
            use_dataset_row_weights=True,
        )
        self.assertTrue(torch.equal(legacy_default, explicit_default))

    def test_independent_mechanism_encoders_are_disjoint(self):
        torch.manual_seed(314159)
        model = DamageAssessmentMTL(
            in_dim=13,
            use_mechanism_decomposition=True,
            mechanism_encoder_mode="independent",
        )
        fragment_parameters = {
            id(parameter)
            for module in (
                model.fragment_shared_1,
                model.fragment_shared_2,
                model.fragment_shared_3,
                model.fragment_physics_skip,
            )
            for parameter in module.parameters()
        }
        shock_parameters = {
            id(parameter)
            for module in (
                model.shock_shared_1,
                model.shock_shared_2,
                model.shock_shared_3,
                model.shock_physics_skip,
            )
            for parameter in module.parameters()
        }
        self.assertTrue(fragment_parameters.isdisjoint(shock_parameters))
        self.assertFalse(any(
            parameter.requires_grad
            for module in (
                model.shared_1, model.shared_2,
                model.shared_3, model.physics_skip,
            )
            for parameter in module.parameters()
        ))
        features = torch.randn(8, 13)
        munitions = torch.tensor([0, 1, 2, 3] * 2)
        combined, fragment, shock = model.forward_with_mechanisms(
            features, munitions)
        self.assertEqual(tuple(combined.shape), (8, 4, 2))
        expected = 1.0 - (
            1.0 - torch.sigmoid(fragment)
        ) * (
            1.0 - torch.sigmoid(shock)
        )
        self.assertTrue(torch.allclose(
            torch.sigmoid(combined), expected,
            atol=2e-6, rtol=2e-6))
        with self.assertRaises(ValueError):
            DamageAssessmentMTL(
                in_dim=13,
                mechanism_encoder_mode="independent",
            )

    def test_component_auxiliary_loss_rewards_matching_probabilities(self):
        torch.manual_seed(20260729)
        target = torch.rand(8, 2, 51) * 0.8 + 0.1
        matching = torch.logit(target)
        reversed_logits = -matching
        weights = torch.ones(8)
        positive_weight = torch.ones(2, 51)
        ordinal_target = torch.rand(8, 4, 2)
        ordinal_target[..., 1] = torch.minimum(
            ordinal_target[..., 1],
            ordinal_target[..., 0])
        applicability = torch.ones(8, 4, 2, dtype=torch.bool)
        matching_loss = component_auxiliary_loss(
            matching, target, weights, positive_weight,
            ordinal_target, applicability,
            rule_consistency_weight=0.0,
            distribution_weight=0.1)
        reversed_loss = component_auxiliary_loss(
            reversed_logits, target, weights, positive_weight,
            ordinal_target, applicability,
            rule_consistency_weight=0.0,
            distribution_weight=0.1)
        self.assertLess(
            float(matching_loss), float(reversed_loss))

    def test_component_target_tree_teacher_rewards_matching_deployed_logits(self):
        torch.manual_seed(20260731)
        component_target = torch.rand(
            8, 2, len(CRITICAL_COMPONENT_IDS)) * 0.8 + 0.1
        component_logits = torch.logit(component_target)
        combined_target = (
            1.0
            - (1.0 - component_target[:, 0])
            * (1.0 - component_target[:, 1])
        )
        target_tree = component_probabilities_to_ordinal(
            combined_target)
        matching_deployed = torch.logit(
            target_tree.clamp(1e-5, 1.0 - 1e-5))
        reversed_deployed = -matching_deployed
        common = {
            "component_logits": component_logits,
            "component_targets": component_target,
            "row_weights": torch.ones(8),
            "positive_weight": torch.ones(
                2, len(CRITICAL_COMPONENT_IDS)),
            "ordinal_targets": target_tree,
            "applicability": torch.ones(
                8, 4, 2, dtype=torch.bool),
            "target_tree_teacher_weight": 1.0,
            "rule_consistency_weight": 0.0,
            "distribution_weight": 0.0,
        }
        matching_loss = component_auxiliary_loss(
            deployed_logits=matching_deployed, **common)
        reversed_loss = component_auxiliary_loss(
            deployed_logits=reversed_deployed, **common)
        self.assertLess(
            float(matching_loss), float(reversed_loss))

    def test_component_rule_ranking_rewards_low_fpr_ordering(self):
        component_count = len(CRITICAL_COMPONENT_IDS)
        fuel_index = list(CRITICAL_COMPONENT_IDS).index(3)
        good_logits = torch.full(
            (4, 2, component_count), -10.0)
        good_logits[:, 0, fuel_index] = torch.tensor(
            [3.0, 1.0, -1.0, -3.0])
        bad_logits = good_logits.clone()
        bad_logits[:, 0, fuel_index] = torch.tensor(
            [-3.0, -1.0, 1.0, 3.0])
        component_target = torch.full_like(
            good_logits, 0.5)
        ordinal_target = torch.zeros(4, 4, 2)
        ordinal_target[:2, 0, 0] = 1.0
        applicability = torch.ones(
            4, 4, 2, dtype=torch.bool)
        entry_weight = torch.zeros(4, 4)
        entry_weight[0, 0] = 1.0
        common = {
            "component_targets": component_target,
            "row_weights": torch.ones(4),
            "positive_weight": torch.ones(
                2, component_count),
            "ordinal_targets": ordinal_target,
            "applicability": applicability,
            "rule_consistency_weight": 0.0,
            "distribution_weight": 0.0,
            "munition_ids": torch.zeros(
                4, dtype=torch.long),
            "rule_entry_ranking_weight": entry_weight,
            "ranking_margin": 0.5,
            "hard_negative_fraction": 0.5,
        }
        good_loss = component_auxiliary_loss(
            good_logits, **common)
        bad_loss = component_auxiliary_loss(
            bad_logits, **common)
        self.assertLess(float(good_loss), float(bad_loss))

    def test_nominal_softmax_projection_is_a_proper_ordinal_simplex(self):
        model = DamageAssessmentMTL(
            in_dim=13,
            ordinal_parameterization="nominal_softmax",
        )
        model.eval()
        features = torch.randn(32, 13)
        munitions = torch.tensor([1, 2, 3, 1] * 8)
        with torch.no_grad():
            cumulative = torch.sigmoid(model(features, munitions))
        class_probabilities = torch.stack((
            1.0 - cumulative[..., 0],
            cumulative[..., 0] - cumulative[..., 1],
            cumulative[..., 1],
        ), dim=-1)
        self.assertTrue(torch.all(class_probabilities >= -1e-7))
        self.assertTrue(torch.allclose(
            class_probabilities.sum(dim=-1),
            torch.ones_like(class_probabilities[..., 0]),
            atol=1e-6,
            rtol=1e-6,
        ))
        with self.assertRaises(ValueError):
            DamageAssessmentMTL(
                in_dim=13,
                ordinal_parameterization="not_a_parameterization",
            )

    def test_zero_initialized_cell_adapter_reproduces_base_model(self):
        torch.manual_seed(19)
        base = DamageAssessmentMTL(in_dim=13)
        adapted = DamageAssessmentMTL(
            in_dim=13,
            residual_adapter_cells=[
                [0, 0, 0],
                [2, 1, 1],
            ],
        )
        incompatible = adapted.load_state_dict(
            base.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(all(
            key.startswith((
                "residual_feature_expansion.",
                "residual_adapters.",
                "residual_adapter_munitions",
                "residual_adapter_basis",
            ))
            for key in incompatible.missing_keys
        ))
        base.eval()
        adapted.eval()
        features = torch.rand(12, 13)
        munitions = torch.tensor([0, 1, 2, 3] * 3)
        with torch.no_grad():
            base_logits = base(features, munitions)
            adapted_logits = adapted(features, munitions)
        self.assertTrue(torch.equal(base_logits, adapted_logits))

    def test_cell_adapter_update_is_confined_to_configured_cell(self):
        torch.manual_seed(23)
        model = DamageAssessmentMTL(
            in_dim=13,
            residual_adapter_cells=[[0, 0, 0]],
        )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.residual_adapters.parameters():
            parameter.requires_grad_(True)
        model.eval()
        model.residual_adapters.train()
        features = torch.rand(8, 13)
        munitions = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        with torch.no_grad():
            before = model(features, munitions).clone()
        optimizer = torch.optim.SGD(
            model.residual_adapters.parameters(), lr=0.05)
        optimizer.zero_grad()
        logits = model(features, munitions)
        loss = -logits[munitions == 0, 0, 0].mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            after = model(features, munitions)
        changed = torch.abs(after - before) > 1e-8
        expected = torch.zeros_like(changed)
        expected[munitions == 0, 0, 0] = True
        self.assertTrue(torch.equal(changed, expected))
        self.assertTrue(torch.all(
            after[:, :, 1] <= after[:, :, 0]))

    def test_balanced_sampler_has_no_duplicate_rows_per_epoch(self):
        munition_ids = np.repeat(np.arange(4), 10)
        weights = np.ones(40)
        sampler = BalancedMunitionBatchSampler(
            munition_ids, weights, batch_size=8, random_state=7)
        sampled = [index for batch in sampler for index in batch]
        self.assertEqual(len(sampled), len(set(sampled)))
        sampled_munitions = munition_ids[np.asarray(sampled)]
        self.assertEqual(
            np.bincount(sampled_munitions, minlength=4).tolist(),
            [10, 10, 10, 10],
        )

    def test_loss_accepts_soft_targets_and_confidence(self):
        logits = torch.randn(12, 4, 2, requires_grad=True)
        hard = torch.randint(0, 2, (12, 4, 2)).float()
        hard[:, :, 1] *= hard[:, :, 0]
        soft = hard * 0.8 + (1.0 - hard) * 0.2
        confidence = torch.full_like(soft, 0.7)
        criterion = FocalUncertaintyOrdinalLoss(
            gamma=0.0, penalty_weight=0.0,
            class1_margin_weight=0.0)
        loss, _ = criterion(
            logits, hard, torch.ones(12),
            m_ids=torch.arange(12) % 4,
            targets_soft=soft,
            target_confidence=confidence,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_class_distribution_loss_directly_rewards_middle_class_mass(self):
        target_class1 = torch.tensor([[1.0, 0.0]])
        good_middle_mass = torch.tensor(
            [[2.1972246, -2.1972246]], requires_grad=True)
        bad_middle_mass = torch.tensor(
            [[2.1972246, 1.3862944]], requires_grad=True)
        good_loss = ordinal_class_distribution_nll(
            good_middle_mass, target_class1)
        bad_loss = ordinal_class_distribution_nll(
            bad_middle_mass, target_class1)
        self.assertLess(float(good_loss.item()), float(bad_loss.item()))
        good_loss.mean().backward()
        self.assertTrue(torch.isfinite(good_middle_mass.grad).all())

    def test_hard_level_auxiliary_resolves_soft_middle_class_ambiguity(self):
        hard_class1 = torch.zeros(1, 4, 2)
        hard_class1[:, 0, 0] = 1.0
        # A physically plausible soft score can imply only 0.2 middle-class
        # mass even though thresholding it produces the deterministic L1
        # label.  The auxiliary must distinguish those two semantics.
        soft = hard_class1.clone()
        soft[:, 0] = torch.tensor([0.6, 0.4])
        weights = torch.ones(1)
        munition_ids = torch.zeros(1, dtype=torch.long)
        good_middle = torch.zeros(1, 4, 2)
        good_middle[:, 0] = torch.tensor([2.1972246, -2.1972246])
        bad_middle = torch.zeros(1, 4, 2)
        bad_middle[:, 0] = torch.tensor([2.1972246, 1.3862944])

        base = FocalUncertaintyOrdinalLoss(
            gamma=0.0,
            penalty_weight=0.0,
            class1_margin_weight=0.0,
            class_distribution_weight=0.0,
            hard_level_classification_weight=0.0,
        )
        dual = FocalUncertaintyOrdinalLoss(
            gamma=0.0,
            penalty_weight=0.0,
            class1_margin_weight=0.0,
            class_distribution_weight=0.0,
            hard_level_classification_weight=1.0,
        )
        base_good = base(
            good_middle, hard_class1, weights,
            m_ids=munition_ids, targets_soft=soft)[0]
        base_bad = base(
            bad_middle, hard_class1, weights,
            m_ids=munition_ids, targets_soft=soft)[0]
        dual_good = dual(
            good_middle, hard_class1, weights,
            m_ids=munition_ids, targets_soft=soft)[0]
        dual_bad = dual(
            bad_middle, hard_class1, weights,
            m_ids=munition_ids, targets_soft=soft)[0]
        self.assertLess(
            float((dual_good - base_good).item()),
            float((dual_bad - base_bad).item()),
        )

    def test_improved_baseline_and_confidence_ablation_are_single_variable(self):
        baseline = load_ablation_config("A0_full")
        confidence = load_ablation_config("A13_with_label_confidence")
        no_class_distribution = load_ablation_config(
            "A14_no_class_distribution_loss")

        self.assertFalse(baseline["data"]["use_label_uncertainty"])
        self.assertTrue(confidence["data"]["use_label_uncertainty"])
        self.assertTrue(baseline["loss"]["use_class_distribution_loss"])
        self.assertFalse(
            no_class_distribution["loss"]["use_class_distribution_loss"])

        baseline_data = dict(baseline["data"])
        confidence_data = dict(confidence["data"])
        baseline_data.pop("use_label_uncertainty")
        confidence_data.pop("use_label_uncertainty")
        self.assertEqual(baseline_data, confidence_data)
        self.assertEqual(baseline["model"], confidence["model"])
        self.assertEqual(baseline["loss"], confidence["loss"])
        self.assertEqual(baseline["calibration"], confidence["calibration"])

    def test_selective_confidence_and_weak_cell_factorial_configs(self):
        full_confidence = load_ablation_config(
            "A13_with_label_confidence")
        selective = load_ablation_config(
            "A15_selective_confidence")
        weak_middle = load_ablation_config(
            "A16_weak_cell_middle_loss")
        combined = load_ablation_config(
            "A17_selective_confidence_weak_cell_loss")

        full_strength = full_confidence["data"][
            "label_confidence_strength_by_task_munition"]
        selective_strength = selective["data"][
            "label_confidence_strength_by_task_munition"]
        self.assertEqual(full_strength[0][0], 1.0)
        self.assertEqual(selective_strength[0][0], 0.0)
        self.assertEqual(selective_strength, combined["data"][
            "label_confidence_strength_by_task_munition"])
        self.assertEqual(
            weak_middle["loss"][
                "middle_class_distribution_multiplier"],
            combined["loss"][
                "middle_class_distribution_multiplier"],
        )
        self.assertTrue(all(
            value == 1.0
            for row in selective["loss"][
                "middle_class_distribution_multiplier"]
            for value in row
        ))
        self.assertTrue(all(
            value == 1.0
            for row in weak_middle["data"][
                "label_confidence_strength_by_task_munition"]
            for value in row
        ))

    def test_a18_reuses_a13_weights_and_recalibrates_validation_only(self):
        config = load_ablation_config(
            "A18_exact_class1_floor_calibration")
        self.assertTrue(config["data"]["use_label_uncertainty"])
        self.assertEqual(
            config["execution"]["reuse_train_from"],
            "A13_with_label_confidence",
        )
        self.assertTrue(config["execution"]["recalibrate_thresholds"])
        self.assertEqual(
            config["calibration"]["minimum_exact_class1_recall"], 0.85)
        self.assertIsNone(
            config["calibration"][
                "maximum_class1_floor_accuracy_drop"])

        bounded = load_ablation_config(
            "A19_bounded_class1_floor_calibration")
        self.assertEqual(
            bounded["execution"]["reuse_train_from"],
            "A13_with_label_confidence",
        )
        self.assertEqual(
            bounded["calibration"][
                "maximum_class1_floor_accuracy_drop"],
            0.005,
        )

    def test_legacy_confidence_contract_matches_only_all_ones_default(self):
        legacy = {"dataset_sha256": "abc"}
        current_default = {
            "dataset_sha256": "abc",
            "label_confidence_strength_by_task_munition": [
                [1.0] * 4 for _ in range(4)
            ],
        }
        selective = {
            **current_default,
            "label_confidence_strength_by_task_munition": [
                [0.0, 1.0, 1.0, 1.0],
                *[[1.0] * 4 for _ in range(3)],
            ],
        }
        self.assertTrue(data_contracts_match(legacy, current_default))
        self.assertFalse(data_contracts_match(legacy, selective))

    def test_mechanism_experiment_configs_are_factorially_isolated(self):
        mechanism = load_ablation_config(
            "A27_mechanism_decomposition")
        physics = load_ablation_config(
            "A28_terminal_physics_features")
        combined = load_ablation_config(
            "A29_mechanism_with_terminal_physics")
        self.assertTrue(
            mechanism["model"]["use_mechanism_decomposition"])
        self.assertTrue(
            mechanism["data"]["use_mechanism_supervision"])
        self.assertFalse(
            mechanism["data"]["use_terminal_physics_features"])
        self.assertTrue(
            physics["data"]["use_terminal_physics_features"])
        self.assertFalse(
            physics["model"]["use_mechanism_decomposition"])
        self.assertTrue(
            combined["model"]["use_mechanism_decomposition"])
        self.assertTrue(
            combined["data"]["use_mechanism_supervision"])
        self.assertTrue(
            combined["data"]["use_terminal_physics_features"])

    def test_auxiliary_mechanism_config_keeps_direct_combined_head(self):
        config = load_ablation_config(
            "A32_auxiliary_mechanism_terminal_physics")
        self.assertTrue(
            config["model"]["use_mechanism_auxiliary_heads"])
        self.assertFalse(
            config["model"]["use_mechanism_decomposition"])
        self.assertTrue(
            config["data"]["use_mechanism_supervision"])
        self.assertTrue(
            config["data"]["use_terminal_physics_features"])
        self.assertEqual(
            config["loss"]["hard_level_classification_weight"], 0.5)
        self.assertEqual(
            config["loss"]["mechanism_auxiliary_weight"], 0.2)

    def test_component_auxiliary_config_is_factorially_isolated(self):
        config = load_ablation_config(
            "A34_component_physics_auxiliary")
        self.assertTrue(
            config["data"]["use_component_supervision"])
        self.assertFalse(
            config["data"]["use_mechanism_supervision"])
        self.assertTrue(
            config["model"]["use_component_auxiliary_heads"])
        self.assertFalse(
            config["model"]["use_mechanism_auxiliary_heads"])
        self.assertFalse(
            config["model"]["use_mechanism_decomposition"])
        self.assertTrue(
            config["data"]["use_terminal_physics_features"])
        self.assertEqual(
            config["loss"]["hard_level_classification_weight"],
            0.5)

    def test_a35_uses_frozen_independent_proper_component_fusion(self):
        config = load_ablation_config(
            "A35_independent_component_tree_fusion")
        self.assertTrue(
            config["data"]["use_component_supervision"])
        self.assertTrue(
            config["model"]["use_component_auxiliary_heads"])
        self.assertEqual(
            config["model"]["component_branch_mode"],
            "independent_experts")
        self.assertFalse(
            config["loss"]["component_use_positive_weight"])
        self.assertTrue(config["training"]["freeze_base_model"])
        self.assertTrue(
            config["training"]["freeze_criterion_uncertainty"])
        self.assertIn(
            "A31_dual_target_with_terminal_physics",
            config["training"]["initial_checkpoint"])
        alpha = np.asarray(
            config["model"]["component_tree_fusion_alpha"],
            dtype=np.float32)
        self.assertEqual(alpha.shape, (4, 4, 2))
        self.assertTrue(np.all((alpha >= 0.0) & (alpha <= 1.0)))
        self.assertGreater(int(np.count_nonzero(alpha)), 0)
        # Structural Small/K2 and Small/C2 cells remain immutable zero.
        self.assertEqual(float(alpha[0, 0, 1]), 0.0)
        self.assertEqual(float(alpha[0, 3, 1]), 0.0)

    def test_a36_continues_a35_with_component_rule_ranking(self):
        config = load_ablation_config(
            "A36_component_rule_low_fpr_continuation")
        self.assertIn(
            "A35_independent_component_tree_fusion",
            config["training"]["initial_checkpoint"])
        self.assertTrue(config["training"]["freeze_base_model"])
        self.assertFalse(
            config["loss"]["component_use_positive_weight"])
        entry = np.asarray(
            config["loss"][
                "component_rule_entry_ranking_weight"],
            dtype=np.float32,
        )
        conditional = np.asarray(
            config["loss"][
                "component_rule_conditional_l1_l2_ranking_weight"],
            dtype=np.float32,
        )
        self.assertEqual(entry.shape, (4, 4))
        self.assertEqual(conditional.shape, (4, 4))
        self.assertGreater(float(entry[0, 0]), 0.0)
        self.assertGreater(float(entry[3, 2]), 0.0)
        self.assertGreater(float(conditional[1, 2]), 0.0)
        alpha = np.asarray(
            config["model"]["component_tree_fusion_alpha"],
            dtype=np.float32,
        )
        self.assertEqual(alpha.shape, (4, 4, 2))
        self.assertEqual(float(alpha[0, 0, 1]), 0.0)
        self.assertEqual(float(alpha[0, 3, 1]), 0.0)
        self.assertGreater(float(alpha[3, 3, 0]), 0.0)

    def test_a37_uses_target_tree_teacher_and_no_structural_zero_adapter(self):
        config = load_ablation_config(
            "A37_component_tree_teacher_residual")
        self.assertIn(
            "A35_independent_component_tree_fusion",
            config["training"]["initial_checkpoint"])
        self.assertTrue(config["training"]["freeze_base_model"])
        self.assertGreater(
            float(config["loss"][
                "component_target_tree_teacher_weight"]),
            0.0,
        )
        adapter_cells = {
            tuple(int(value) for value in cell)
            for cell in config["model"]["residual_adapter_cells"]
        }
        self.assertIn((0, 0, 0), adapter_cells)
        self.assertIn((2, 1, 1), adapter_cells)
        self.assertIn((3, 3, 1), adapter_cells)
        self.assertNotIn((0, 0, 1), adapter_cells)
        self.assertNotIn((0, 3, 1), adapter_cells)
        self.assertEqual(
            float(config["calibration"][
                "minimum_exact_class1_recall"]),
            0.90,
        )

    def test_a38_isolates_component_proxy_extension_from_a35_direct_path(self):
        config = load_ablation_config(
            "A38_component_proxy_feature_extension")
        self.assertTrue(
            config["data"]["use_component_proxy_features"])
        self.assertTrue(
            config["data"]["use_terminal_physics_features"])
        self.assertEqual(
            int(config["model"]["base_input_dim"]), 194)
        self.assertEqual(
            config["model"]["component_branch_mode"],
            "independent_experts")
        self.assertTrue(config["training"]["freeze_base_model"])
        self.assertIn(
            "A35_independent_component_tree_fusion",
            config["training"]["initial_checkpoint"])
        self.assertFalse(
            config["loss"]["component_use_positive_weight"])
        self.assertEqual(
            float(config["calibration"][
                "minimum_exact_class1_recall"]),
            0.90,
        )

    def test_a39_is_from_scratch_direct_component_proxy_ablation(self):
        config = load_ablation_config(
            "A39_direct_component_proxy_features")
        self.assertTrue(
            config["data"]["use_component_proxy_features"])
        self.assertTrue(
            config["model"]["allow_component_proxy_direct_path"])
        self.assertEqual(
            int(config["model"]["base_input_dim"]), 296)
        self.assertFalse(
            config["model"]["use_component_auxiliary_heads"])
        self.assertFalse(config["training"]["freeze_base_model"])
        self.assertNotIn(
            "initial_checkpoint", config["training"])

    def test_a40_targets_diagnosed_fragment_bottleneck(self):
        config = load_ablation_config(
            "A40_independent_mechanism_component_proxies")
        self.assertTrue(
            config["data"]["use_component_proxy_features"])
        self.assertTrue(
            config["data"]["use_armor_aware_fragment_proxies"])
        self.assertTrue(
            config["data"]["use_mechanism_supervision"])
        self.assertTrue(
            config["model"]["use_mechanism_decomposition"])
        self.assertEqual(
            config["model"]["mechanism_encoder_mode"],
            "independent")
        self.assertEqual(
            config["model"]["base_input_dim"], 296)
        self.assertTrue(
            config["model"]["allow_component_proxy_direct_path"])
        self.assertGreater(
            config["loss"]["mechanism_branch_weights"][0],
            config["loss"]["mechanism_branch_weights"][1])
        self.assertGreater(
            config["loss"]["mechanism_boundary_focus_weight"], 0.0)
        self.assertGreater(
            config["loss"]["mechanism_hard_classification_weight"], 0.0)
        self.assertFalse(
            config["loss"]["mechanism_use_dataset_row_weights"])
        self.assertTrue(
            config["calibration"]["goal_aware_cell_search"])
        self.assertEqual(
            config["calibration"]["minimum_cell_accuracy"], 0.94)
        self.assertEqual(
            config["calibration"][
                "minimum_class_diagonal_recall"], 0.90)

    def test_run_manifest_hashes_component_supervision_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            config_path = (
                Path(__file__).resolve().parents[1]
                / "configs" / "ablations"
                / "A34_component_physics_auxiliary.json"
            )
            _write_run_manifest(
                run_dir,
                {"experiment_id": "manifest_contract_test"},
                seed=42,
                config_path=config_path,
                smoke_test=True,
            )
            manifest = json.loads(
                (run_dir / "run_manifest.json").read_text(
                    encoding="utf-8"))
            source_hashes = manifest["source_sha256"]
            self.assertIn(
                "src/loitering_munition_damage_twin/stage0/"
                "component_supervision.py",
                {name.replace("\\", "/") for name in source_hashes},
            )
            self.assertIn(
                "src/loitering_munition_damage_twin/surrogate/artifacts.py",
                {name.replace("\\", "/") for name in source_hashes},
            )
            self.assertIn(
                "src/loitering_munition_damage_twin/experiments/"
                "ablation_config.py",
                {
                    name.replace("\\", "/")
                    for name in source_hashes
                },
            )

    def test_test_evaluation_requires_test_blind_promotion_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            validation_dir = run_dir / "output" / "validation"
            validation_dir.mkdir(parents=True)
            report_path = (
                validation_dir / "validation_promotion.json")
            report_path.write_text(json.dumps({
                "status": "FAIL",
                "test_metrics_read": False,
            }), encoding="utf-8")
            self.assertIsNone(
                _passed_validation_promotion(run_dir))
            report_path.write_text(json.dumps({
                "status": "PASS",
                "test_metrics_read": True,
            }), encoding="utf-8")
            self.assertIsNone(
                _passed_validation_promotion(run_dir))
            report_path.write_text(json.dumps({
                "status": "PASS",
                "test_metrics_read": False,
            }), encoding="utf-8")
            self.assertEqual(
                _passed_validation_promotion(run_dir),
                report_path,
            )
            self.assertIsNone(
                _passed_validation_promotion(run_dir, "A_candidate"))
            report_path.write_text(json.dumps({
                "candidate": "A_candidate",
                "status": "PASS",
                "test_metrics_read": False,
            }), encoding="utf-8")
            self.assertEqual(
                _passed_validation_promotion(run_dir, "A_candidate"),
                report_path,
            )

    def test_ablation_eval_command_forwards_promotion_report(self):
        command = _build_evaluation_command(
            Path("config.json"),
            Path("run"),
            42,
            Path("strict_validation_promotion.json"),
            data="dataset.parquet",
        )
        promotion_index = command.index("--promotion-report")
        self.assertEqual(
            command[promotion_index + 1],
            "strict_validation_promotion.json",
        )
        self.assertIn("--data", command)

    def test_eval_export_authorization_is_bound_to_run_and_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "candidate"
            run_dir.mkdir()
            report_path = run_dir / "validation_promotion.json"
            report_path.write_text(json.dumps({
                "schema": "synthetic_promotion_v1",
                "candidate": "A_candidate",
                "status": "PASS",
                "test_metrics_read": False,
            }), encoding="utf-8")
            authorization = _authorize_test_evaluation(
                str(report_path), str(run_dir), "A_candidate")
            self.assertEqual(
                authorization["promotion_candidate"],
                "A_candidate")
            self.assertEqual(
                len(authorization[
                    "promotion_report_sha256"]), 64)

            with self.assertRaisesRegex(
                    RuntimeError, "does not match"):
                _authorize_test_evaluation(
                    str(report_path), str(run_dir), "A_other")

            report_path.write_text(json.dumps({
                "schema": "synthetic_promotion_v1",
                "status": "PASS",
                "test_metrics_read": False,
            }), encoding="utf-8")
            with self.assertRaisesRegex(
                    RuntimeError, "does not match"):
                _authorize_test_evaluation(
                    str(report_path), str(run_dir), "A_candidate")

            outside = Path(directory) / "outside.json"
            outside.write_text(report_path.read_text(
                encoding="utf-8"), encoding="utf-8")
            with self.assertRaisesRegex(
                    RuntimeError, "inside"):
                _authorize_test_evaluation(
                    str(outside), str(run_dir), "A_candidate")

    @staticmethod
    def _strict_goal_predictions() -> pd.DataFrame:
        rows = []
        for munition_id in range(4):
            for row_in_munition in range(300):
                level = row_in_munition % 3
                row = {
                    "sample_id":
                        f"strict-{munition_id}-{row_in_munition}",
                    "root_seed_id":
                        f"strict-root-{munition_id}-{row_in_munition}",
                    "munition_id": munition_id,
                }
                for task in ("K", "M", "F", "C"):
                    row[f"true_{task}"] = level
                    row[f"pred_{task}"] = level
                rows.append(row)
        return pd.DataFrame(rows)

    def test_strict_goal_validator_passes_every_preregistered_requirement(self):
        report = evaluate_strict_goal(
            self._strict_goal_predictions())
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["failure_count"], 0)

    def test_strict_goal_applicability_matches_model_and_includes_med_lm_c2(self):
        self.assertEqual(
            STRICT_GOAL_ORDINAL_APPLICABILITY,
            DEFAULT_ORDINAL_APPLICABILITY,
        )
        self.assertTrue(
            STRICT_GOAL_ORDINAL_APPLICABILITY[1][3][1])
        self.assertFalse(
            STRICT_GOAL_ORDINAL_APPLICABILITY[0][0][1])
        self.assertFalse(
            STRICT_GOAL_ORDINAL_APPLICABILITY[0][3][1])

    def test_strict_goal_validator_rejects_one_sub94_munition_task(self):
        predictions = self._strict_goal_predictions()
        affected = (
            (predictions["munition_id"] == 2)
            & (predictions["true_F"] == 0)
        )
        affected_indices = predictions.index[affected][:19]
        predictions.loc[affected_indices, "pred_F"] = 1
        report = evaluate_strict_goal(predictions)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any(
            "Med-RD/F accuracy" in failure
            for failure in report["failures"]
        ))

    def test_strict_goal_validator_accepts_exact_inclusive_minima(self):
        predictions = self._strict_goal_predictions()
        med_lm_k_l1 = predictions.index[
            (predictions["munition_id"] == 1)
            & (predictions["true_K"] == 1)
        ][:10]
        med_lm_k_l2 = predictions.index[
            (predictions["munition_id"] == 1)
            & (predictions["true_K"] == 2)
        ][:8]
        predictions.loc[med_lm_k_l1, "pred_K"] = 0
        predictions.loc[med_lm_k_l2, "pred_K"] = 1

        report = evaluate_strict_goal(predictions)

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(
            report["criteria"]["comparison"],
            "greater_than_or_equal_to_for_minima",
        )
        med_lm_k = report["munition_task_metrics"]["Med-LM"]["K"]
        self.assertEqual(
            med_lm_k["three_class_accuracy"]["percent"], 94.0)
        self.assertEqual(
            med_lm_k["class_diagonal_recall"]["L1"]["percent"],
            90.0,
        )

    def test_dual_target_configs_are_factorially_isolated(self):
        baseline = load_ablation_config(
            "A13_with_label_confidence")
        hard_level = load_ablation_config(
            "A30_dual_target_hard_level")
        combined = load_ablation_config(
            "A31_dual_target_with_terminal_physics")
        self.assertEqual(
            hard_level["loss"]["hard_level_classification_weight"],
            0.5,
        )
        self.assertFalse(
            hard_level["data"]["use_terminal_physics_features"])
        self.assertTrue(
            combined["data"]["use_terminal_physics_features"])
        self.assertEqual(
            combined["loss"]["hard_level_classification_weight"],
            hard_level["loss"]["hard_level_classification_weight"],
        )
        baseline_without_new_term = dict(baseline["loss"])
        hard_level_without_new_term = dict(hard_level["loss"])
        baseline_without_new_term.pop(
            "hard_level_classification_weight")
        hard_level_without_new_term.pop(
            "hard_level_classification_weight")
        self.assertEqual(
            baseline_without_new_term, hard_level_without_new_term)

    def test_selective_confidence_bypasses_only_configured_cell(self):
        confidence = np.full((2, 4, 2), 0.4, dtype=np.float32)
        munition_ids = np.asarray([0, 1], dtype=np.int64)
        strength = np.ones((4, 4), dtype=np.float32)
        strength[0, 0] = 0.0
        adjusted = _apply_label_confidence_strength(
            confidence, munition_ids, strength)
        self.assertTrue(np.allclose(adjusted[0, 0], 1.0))
        self.assertTrue(np.allclose(adjusted[1, 0], 0.4))
        self.assertTrue(np.allclose(adjusted[:, 1:], 0.4))

    def test_middle_class_multiplier_only_changes_targeted_middle_rows(self):
        logits = torch.zeros(2, 4, 2)
        targets = torch.zeros_like(logits)
        targets[0, 0] = torch.tensor([1.0, 0.0])
        targets[1, 0] = torch.tensor([0.0, 0.0])
        weights = torch.ones(2)
        munition_ids = torch.zeros(2, dtype=torch.long)
        base = FocalUncertaintyOrdinalLoss(
            gamma=0.0,
            penalty_weight=0.0,
            class_distribution_weight=0.25,
        )
        multiplier = torch.ones(4, 4)
        multiplier[0, 0] = 2.0
        targeted = FocalUncertaintyOrdinalLoss(
            gamma=0.0,
            penalty_weight=0.0,
            class_distribution_weight=0.25,
            middle_class_distribution_multiplier=multiplier,
        )
        targeted.log_vars.data.copy_(base.log_vars.data)
        _, base_tasks = base(
            logits, targets, weights, m_ids=munition_ids)
        _, targeted_tasks = targeted(
            logits, targets, weights, m_ids=munition_ids)
        self.assertGreater(
            float(targeted_tasks[0].detach()),
            float(base_tasks[0].detach()))
        self.assertTrue(torch.allclose(
            targeted_tasks[1:], base_tasks[1:]))

    def test_hard_negative_ranking_rewards_separated_scores(self):
        masks_positive = torch.tensor([True, True, False, False])
        masks_negative = ~masks_positive
        good = torch.tensor(
            [2.0, 1.5, -0.5, -1.0], requires_grad=True)
        bad = torch.tensor(
            [-0.5, -1.0, 2.0, 1.5], requires_grad=True)
        good_loss = hard_negative_pairwise_ranking_loss(
            good,
            masks_positive,
            masks_negative,
            margin=0.5,
            hard_negative_fraction=1.0,
        )
        bad_loss = hard_negative_pairwise_ranking_loss(
            bad,
            masks_positive,
            masks_negative,
            margin=0.5,
            hard_negative_fraction=1.0,
        )
        self.assertLess(
            float(good_loss.detach()), float(bad_loss.detach()))
        good_loss.backward()
        self.assertTrue(torch.isfinite(good.grad).all())
        self.assertTrue(torch.all(good.grad[:2] < 0.0))
        self.assertTrue(torch.all(good.grad[2:] > 0.0))

    def test_targeted_entry_ranking_changes_only_target_task(self):
        logits = torch.zeros(4, 4, 2)
        targets = torch.zeros_like(logits)
        targets[:2, 0, 0] = 1.0
        weights = torch.ones(4)
        munition_ids = torch.zeros(4, dtype=torch.long)
        base = FocalUncertaintyOrdinalLoss(
            gamma=0.0,
            penalty_weight=0.0,
            class_distribution_weight=0.25,
        )
        entry_weights = torch.zeros(4, 4)
        entry_weights[0, 0] = 0.05
        targeted = FocalUncertaintyOrdinalLoss(
            gamma=0.0,
            penalty_weight=0.0,
            class_distribution_weight=0.25,
            entry_ranking_weight=entry_weights,
        )
        targeted.log_vars.data.copy_(base.log_vars.data)
        _, base_tasks = base(
            logits, targets, weights, m_ids=munition_ids)
        _, targeted_tasks = targeted(
            logits, targets, weights, m_ids=munition_ids)
        self.assertGreater(
            float(targeted_tasks[0].detach()),
            float(base_tasks[0].detach()))
        self.assertTrue(torch.allclose(
            targeted_tasks[1:], base_tasks[1:]))

    def test_targeted_conditional_ranking_changes_only_med_rd_m(self):
        logits = torch.zeros(4, 4, 2)
        targets = torch.zeros_like(logits)
        targets[:, 1, 0] = 1.0
        targets[:2, 1, 1] = 1.0
        weights = torch.ones(4)
        munition_ids = torch.full((4,), 2, dtype=torch.long)
        base = FocalUncertaintyOrdinalLoss(
            gamma=0.0,
            penalty_weight=0.0,
            class_distribution_weight=0.25,
        )
        conditional_weights = torch.zeros(4, 4)
        conditional_weights[1, 2] = 0.05
        targeted = FocalUncertaintyOrdinalLoss(
            gamma=0.0,
            penalty_weight=0.0,
            class_distribution_weight=0.25,
            conditional_l1_l2_ranking_weight=conditional_weights,
        )
        targeted.log_vars.data.copy_(base.log_vars.data)
        _, base_tasks = base(
            logits, targets, weights, m_ids=munition_ids)
        _, targeted_tasks = targeted(
            logits, targets, weights, m_ids=munition_ids)
        self.assertGreater(
            float(targeted_tasks[1].detach()),
            float(base_tasks[1].detach()))
        self.assertTrue(torch.allclose(
            targeted_tasks[[0, 2, 3]], base_tasks[[0, 2, 3]]))

    def test_a22_targeted_ranking_config_is_sparse_and_preregistered(self):
        config = load_ablation_config("A22_targeted_ranking")
        entry = np.asarray(
            config["loss"]["entry_ranking_weight"])
        conditional = np.asarray(
            config["loss"][
                "conditional_l1_l2_ranking_weight"])
        self.assertEqual(set(map(tuple, np.argwhere(entry > 0.0))), {
            (0, 0), (3, 0), (3, 1), (3, 2)})
        self.assertEqual(
            set(map(tuple, np.argwhere(conditional > 0.0))),
            {(1, 2)},
        )
        self.assertTrue(config["data"]["use_label_uncertainty"])
        self.assertEqual(config["loss"]["hard_negative_fraction"], 0.1)

    def test_a23_adapter_config_is_frozen_and_preregistered(self):
        config = load_ablation_config(
            "A23_frozen_cell_residual_adapters")
        self.assertTrue(config["training"]["freeze_base_model"])
        self.assertTrue(
            config["training"]["freeze_criterion_uncertainty"])
        self.assertTrue(
            config["training"]["include_initial_candidate"])
        self.assertIn(
            "seed{seed}",
            config["training"]["initial_checkpoint"],
        )
        cells = {
            tuple(cell)
            for cell in config["model"]["residual_adapter_cells"]
        }
        self.assertEqual(cells, {
            (0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0),
            (0, 3, 0), (1, 3, 0), (2, 3, 0),
            (2, 1, 1),
        })
        self.assertEqual(
            config["model"]["residual_adapter_frequencies"],
            [1.0, 2.0, 4.0, 8.0],
        )

    def test_a24_is_final_frozen_adapter_candidate(self):
        config = load_ablation_config(
            "A24_hard_boundary_residual_adapters")
        self.assertFalse(config["data"]["use_soft_labels"])
        self.assertFalse(config["data"]["use_label_uncertainty"])
        self.assertTrue(config["training"]["freeze_base_model"])
        self.assertTrue(
            config["training"]["include_initial_candidate"])
        self.assertIn(
            "冻结残差适配器家族的最后一个预注册候选",
            config["ablation"]["note"],
        )
        multipliers = np.asarray(
            config["loss"][
                "middle_class_distribution_multiplier"])
        self.assertLessEqual(float(multipliers.max()), 1.5)

    def test_a26_nominal_head_config_changes_only_parameterization(self):
        baseline = load_ablation_config(
            "A13_with_label_confidence")
        candidate = load_ablation_config(
            "A26_nominal_softmax_heads")
        self.assertEqual(
            baseline["model"]["ordinal_parameterization"],
            "cumulative_logits",
        )
        self.assertEqual(
            candidate["model"]["ordinal_parameterization"],
            "nominal_softmax",
        )
        for section in ("data", "loss", "training"):
            self.assertEqual(candidate[section], baseline[section])
        baseline_model = dict(baseline["model"])
        candidate_model = dict(candidate["model"])
        baseline_model.pop("ordinal_parameterization")
        candidate_model.pop("ordinal_parameterization")
        self.assertEqual(candidate_model, baseline_model)
        self.assertIn(
            "禁止读取A26测试集",
            candidate["ablation"]["note"],
        )

    def test_probability_ensemble_uses_fixed_normalized_weights(self):
        class ConstantLogit(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = float(value)

            def forward(self, x, munition_id):
                return torch.full(
                    (len(x), 4, 2), self.value,
                    dtype=x.dtype, device=x.device)

        ensemble = EqualWeightProbabilityEnsemble(
            [ConstantLogit(-2.0), ConstantLogit(2.0)],
            [1.0, 1.0],
        )
        logits = ensemble(
            torch.zeros(3, 13),
            torch.zeros(3, dtype=torch.long),
        )
        self.assertTrue(torch.allclose(
            torch.sigmoid(logits),
            torch.full_like(logits, 0.5),
            atol=1e-7,
        ))
        self.assertTrue(torch.allclose(
            ensemble.weights,
            torch.tensor([0.5, 0.5]),
        ))

    def test_heterogeneous_ensemble_matches_inference_contract_only(self):
        base = {
            "dataset_sha256": "dataset",
            "dataset_schema": "stage0_lineage_v2",
            "frame_convention": "stage0_ned_frd_v1",
            "feature_names": ["x", "physics"],
            "ordinal_applicability": [[[True, False]]],
            "mechanism_supervision_enabled": False,
        }
        auxiliary = {
            **base,
            "mechanism_supervision_enabled": True,
            "mechanism_target_names": ["fragment_K_ge1_prob"],
        }
        self.assertTrue(
            inference_contracts_match(base, auxiliary))
        changed_input = {
            **auxiliary,
            "feature_names": ["x", "different"],
        }
        self.assertFalse(
            inference_contracts_match(base, changed_input))

        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs" / "ablations"
            / "A33_equal_weight_direct_head_ensemble.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(
            config["input_config"],
            "A31_dual_target_with_terminal_physics",
        )
        self.assertEqual(len(config["members"]), 3)
        self.assertAlmostEqual(
            sum(float(member["weight"])
                for member in config["members"]),
            1.0,
        )
        self.assertTrue(all(
            member["validation_report_source"] == "selection"
            for member in config["members"]
        ))

    def test_ensemble_promotion_requires_broad_gain(self):
        entry_names = (
            "Small/K", "Med-LM/K", "Med-RD/K", "Heavy/K",
            "Small/C", "Med-LM/C", "Med-RD/C",
        )

        def report(delta=0.0, failures=8):
            diagnostics = {
                name: {
                    "entry_standardized_partial_auc": 0.7 + delta}
                for name in entry_names
            }
            diagnostics["Med-RD/M_L1_vs_L2"] = {
                "conditional_auc": 0.9 + delta}
            return {
                "average_3class_accuracy_percent": 93.0,
                "small_k0_false_positive_percent": 0.4,
                "global_c0_false_positive_percent": 2.0,
                "performance_gate": {"failure_count": failures},
                "targeted_probability_diagnostics": diagnostics,
            }

        criteria = {
            "maximum_accuracy_drop_vs_member_mean_percentage_points": 0.02,
            "maximum_small_k0_false_positive_percent": 0.5,
            "maximum_global_c0_false_positive_percent": 2.5,
            "gate_failure_count_may_exceed_best_member": False,
            "minimum_improved_weak_objectives": 4,
            "minimum_objective_improvement_vs_member_mean": 0.001,
            "maximum_objective_degradation_vs_member_mean": 0.002,
            "minimum_mean_objective_delta": 0.001,
        }
        members = [report(), report()]
        candidate = report(delta=0.002)
        decision = _evaluate_ensemble_promotion(
            members, candidate, criteria)
        self.assertEqual(decision["status"], "PASS")
        candidate = report(delta=0.0, failures=9)
        decision = _evaluate_ensemble_promotion(
            members, candidate, criteria)
        self.assertEqual(decision["status"], "FAIL")

    def test_multiseed_summary_uses_sample_standard_deviation(self):
        summary = _summary_stats([92.0, 93.0, 94.0])
        self.assertEqual(summary["n"], 3)
        self.assertAlmostEqual(summary["mean"], 93.0)
        self.assertAlmostEqual(
            summary["sample_standard_deviation"], 1.0)
        self.assertLess(summary["mean_ci95_low"], 93.0)
        self.assertGreater(summary["mean_ci95_high"], 93.0)
        self.assertIn(
            "between-seed", summary["interpretation"])

    def test_validation_report_is_split_explicit_and_hash_bound(self):
        metrics = {
            "cls1_recall_per_mun": torch.full((4, 4), 90.0),
            "cls1_count_per_mun": torch.full(
                (4, 4), 120, dtype=torch.long),
            "acc_per_munition": torch.full((4, 4), 95.0),
            "samples_per_munition": torch.tensor(
                [10, 11, 12, 13]),
            "small_k0_fp_rate": 0.004,
            "c0_fp_rate": 0.02,
            "epoch": 3,
            "selection_score": 0.5,
            "acc3_mean": 0.95,
            "acc_per_task": [94.0, 95.0, 96.0, 95.0],
        }
        report = _validation_report_from_metrics(
            metrics,
            model_variant="raw_best",
            dataset_sha256="data",
            model_sha256="model",
            threshold_sha256="threshold",
        )
        self.assertEqual(report["split"], "validation")
        self.assertFalse(report["test_labels_used"])
        self.assertTrue(report["performance_gate"]["passed"])
        self.assertFalse(report["goal_performance_gate"]["passed"])
        self.assertEqual(
            report["goal_performance_gate"]["status"],
            "INSUFFICIENT_EVIDENCE",
        )
        self.assertEqual(
            report["artifact_identity"]["model_sha256"], "model")

    def test_goal_gate_requires_full_confusion_diagonals_and_support(self):
        # Four munitions, 4 tasks and exactly 100 examples per applicable
        # class.  Structural Small/K2 and Small/C2 are intentionally absent.
        true_rows = []
        predicted_rows = []
        munition_rows = []
        for munition_id in range(4):
            for level in range(3):
                for _ in range(100):
                    row = [level, level, level, level]
                    if munition_id == 0 and level == 2:
                        row[0] = 1
                        row[3] = 1
                    true_rows.append(row)
                    predicted_rows.append(list(row))
                    munition_rows.append(munition_id)
        true_levels = torch.tensor(true_rows, dtype=torch.long)
        predicted_levels = torch.tensor(
            predicted_rows, dtype=torch.long)
        munition_ids = torch.tensor(
            munition_rows, dtype=torch.long)
        metrics = {
            "cls1_recall_per_mun": torch.full((4, 4), 100.0),
            "cls1_count_per_mun": torch.full(
                (4, 4), 100, dtype=torch.long),
            "acc_per_munition": torch.full((4, 4), 100.0),
            "samples_per_munition": torch.tensor(
                [300, 300, 300, 300]),
            "small_k0_fp_rate": 0.0,
            "c0_fp_rate": 0.0,
            "epoch": 1,
            "selection_score": 1.0,
            "acc3_mean": 1.0,
            "acc_per_task": [100.0] * 4,
            "pred_level": predicted_levels,
            "true_level": true_levels,
            "munition_ids": munition_ids,
        }
        report = _validation_report_from_metrics(
            metrics,
            model_variant="raw_best",
            dataset_sha256="data",
            model_sha256="model",
            threshold_sha256="threshold",
        )
        self.assertTrue(report["goal_performance_gate"]["passed"])
        self.assertEqual(
            report["goal_performance_gate"]["status"], "PASS")
        self.assertEqual(
            report["goal_performance_gate"][
                "cell_confusion_metrics"]["Small"]["K"][
                    "class_status"][2],
            "NOT_APPLICABLE",
        )

        predicted_levels[:100, 0] = 1
        metrics["pred_level"] = predicted_levels
        metrics["acc_per_munition"][0, 0] = 91.0
        failed = _validation_report_from_metrics(
            metrics,
            model_variant="raw_best",
            dataset_sha256="data",
            model_sha256="model",
            threshold_sha256="threshold",
        )
        self.assertFalse(failed["goal_performance_gate"]["passed"])
        self.assertEqual(
            failed["goal_performance_gate"]["status"], "FAIL")
        self.assertGreater(
            failed["goal_performance_gate"]["metric_failure_count"], 0)

    def test_confidence_resolved_diagnostics_excludes_mc_boundary(self):
        probabilities = torch.tensor([
            [[0.80, 0.20], [0.5, 0.1], [0.5, 0.1], [0.5, 0.1]],
            [[0.52, 0.48], [0.5, 0.1], [0.5, 0.1], [0.5, 0.1]],
        ])
        soft_targets = probabilities.clone()
        confidence = torch.ones_like(probabilities)
        confidence[1, 0] = 0.5
        true_levels = torch.tensor([
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ])
        predicted_levels = torch.tensor([
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        munitions = torch.zeros(2, dtype=torch.long)
        report = _confidence_resolved_diagnostics(
            probabilities,
            soft_targets,
            confidence,
            true_levels,
            predicted_levels,
            munitions,
        )
        small_k_l1 = report["cells"]["Small"]["K"]["L1"]
        self.assertEqual(small_k_l1["full_support"], 2)
        self.assertEqual(small_k_l1["resolved_support"], 1)
        self.assertEqual(
            small_k_l1["resolved_recall_percent"], 100.0)

    def test_partial_auc_is_defined_for_two_class_cell(self):
        target = np.asarray([0, 0, 1, 1], dtype=np.int64)
        score = np.asarray([0.1, 0.2, 0.8, 0.9])
        value = _safe_partial_auc(target, score, 0.5)
        self.assertIsNotNone(value)
        self.assertAlmostEqual(value, 1.0)

    def test_a22_promotion_requires_two_validation_ranking_gains(self):
        def report(small_k, control, conditional):
            return {
                "average_3class_accuracy_percent": 93.0,
                "small_k0_false_positive_percent": 0.4,
                "global_c0_false_positive_percent": 2.0,
                "performance_gate": {"failure_count": 8},
                "targeted_probability_diagnostics": {
                    "Small/K": {
                        "entry_standardized_partial_auc": small_k},
                    "Small/C": {
                        "entry_standardized_partial_auc": control},
                    "Med-LM/C": {
                        "entry_standardized_partial_auc": control},
                    "Med-RD/C": {
                        "entry_standardized_partial_auc": control},
                    "Med-RD/M_L1_vs_L2": {
                        "conditional_auc": conditional},
                },
            }
        baseline = report(0.60, 0.65, 0.70)
        candidate = report(0.603, 0.653, 0.700)
        decision = _evaluate_promotion(baseline, candidate)
        self.assertEqual(decision["status"], "PASS")
        candidate = report(0.603, 0.650, 0.700)
        decision = _evaluate_promotion(baseline, candidate)
        self.assertEqual(decision["status"], "FAIL")

    def test_a23_promotion_requires_broad_weak_cell_gain(self):
        names = (
            "Small/K", "Med-LM/K", "Med-RD/K", "Heavy/K",
            "Small/C", "Med-LM/C", "Med-RD/C",
        )

        def report(delta_by_name=None):
            delta_by_name = delta_by_name or {}
            diagnostics = {
                name: {
                    "entry_standardized_partial_auc":
                        0.70 + delta_by_name.get(name, 0.0)
                }
                for name in names
            }
            diagnostics["Med-RD/M_L1_vs_L2"] = {
                "conditional_auc": (
                    0.90
                    + delta_by_name.get(
                        "Med-RD/M_L1_vs_L2", 0.0))
            }
            return {
                "average_3class_accuracy_percent": 93.0,
                "small_k0_false_positive_percent": 0.4,
                "global_c0_false_positive_percent": 2.0,
                "performance_gate": {"failure_count": 8},
                "targeted_probability_diagnostics": diagnostics,
            }

        baseline = report()
        candidate = report({
            "Small/K": 0.003,
            "Med-LM/K": 0.003,
            "Small/C": 0.003,
            "Med-LM/C": 0.001,
        })
        decision = _evaluate_a23_promotion(baseline, candidate)
        self.assertEqual(decision["status"], "PASS")
        candidate = report({
            "Small/K": 0.003,
            "Med-LM/K": 0.003,
            "Small/C": 0.003,
            "Med-RD/C": -0.004,
        })
        decision = _evaluate_a23_promotion(baseline, candidate)
        self.assertEqual(decision["status"], "FAIL")

    def test_a34_promotion_requires_goal_failure_reduction_and_ranking_gain(self):
        names = (
            "Small/K", "Med-LM/K", "Med-RD/K", "Heavy/K",
            "Small/C", "Med-LM/C", "Med-RD/C",
        )

        def report(delta=0.0, goal_failures=26):
            diagnostics = {
                name: {
                    "entry_standardized_partial_auc": 0.70 + delta
                }
                for name in names
            }
            diagnostics["Med-RD/M_L1_vs_L2"] = {
                "conditional_auc": 0.90 + delta}
            return {
                "average_3class_accuracy_percent": 93.0,
                "small_k0_false_positive_percent": 0.4,
                "global_c0_false_positive_percent": 2.0,
                "performance_gate": {"failure_count": 8},
                "goal_performance_gate": {
                    "metric_failure_count": goal_failures,
                    "evidence_failure_count": 3,
                },
                "targeted_probability_diagnostics": diagnostics,
            }

        baseline = report()
        candidate = report(delta=0.003, goal_failures=25)
        decision = evaluate_a34_promotion(baseline, candidate)
        self.assertEqual(decision["status"], "PASS")
        candidate = report(delta=0.003, goal_failures=26)
        decision = evaluate_a34_promotion(baseline, candidate)
        self.assertEqual(decision["status"], "FAIL")

    def test_a36_promotion_uses_a35_baseline_and_frozen_broad_gain_rule(self):
        names = (
            "Small/K", "Med-LM/K", "Med-RD/K", "Heavy/K",
            "Small/C", "Med-LM/C", "Med-RD/C",
        )

        def report(delta=0.0, goal_failures=25):
            diagnostics = {
                name: {
                    "entry_standardized_partial_auc": 0.70 + delta
                }
                for name in names
            }
            diagnostics["Med-RD/M_L1_vs_L2"] = {
                "conditional_auc": 0.90 + delta}
            return {
                "average_3class_accuracy_percent": 94.0,
                "small_k0_false_positive_percent": 0.4,
                "global_c0_false_positive_percent": 2.0,
                "performance_gate": {"failure_count": 8},
                "goal_performance_gate": {
                    "metric_failure_count": goal_failures,
                    "evidence_failure_count": 3,
                },
                "targeted_probability_diagnostics": diagnostics,
            }

        decision = evaluate_a36_promotion(
            report(), report(delta=0.003, goal_failures=24))
        self.assertEqual(decision["status"], "PASS")
        self.assertEqual(
            decision["baseline"],
            "A35_independent_component_tree_fusion")
        self.assertEqual(
            decision["candidate"],
            "A36_component_rule_low_fpr_continuation")
        decision = evaluate_a36_promotion(
            report(), report(delta=0.003, goal_failures=25))
        self.assertEqual(decision["status"], "FAIL")

    def test_a37_promotion_uses_a35_baseline_and_frozen_broad_gain_rule(self):
        names = (
            "Small/K", "Med-LM/K", "Med-RD/K", "Heavy/K",
            "Small/C", "Med-LM/C", "Med-RD/C",
        )

        def report(delta=0.0, goal_failures=25):
            diagnostics = {
                name: {
                    "entry_standardized_partial_auc": 0.70 + delta
                }
                for name in names
            }
            diagnostics["Med-RD/M_L1_vs_L2"] = {
                "conditional_auc": 0.90 + delta}
            return {
                "average_3class_accuracy_percent": 94.0,
                "small_k0_false_positive_percent": 0.4,
                "global_c0_false_positive_percent": 2.0,
                "performance_gate": {"failure_count": 8},
                "goal_performance_gate": {
                    "metric_failure_count": goal_failures,
                    "evidence_failure_count": 3,
                },
                "targeted_probability_diagnostics": diagnostics,
            }

        decision = evaluate_a37_promotion(
            report(), report(delta=0.003, goal_failures=24))
        self.assertEqual(decision["status"], "PASS")
        self.assertEqual(
            decision["baseline"],
            "A35_independent_component_tree_fusion")
        self.assertEqual(
            decision["candidate"],
            "A37_component_tree_teacher_residual")
        decision = evaluate_a37_promotion(
            report(), report(delta=0.003, goal_failures=25))
        self.assertEqual(decision["status"], "FAIL")

    def test_a38_promotion_uses_a35_baseline_and_frozen_broad_gain_rule(self):
        names = (
            "Small/K", "Med-LM/K", "Med-RD/K", "Heavy/K",
            "Small/C", "Med-LM/C", "Med-RD/C",
        )

        def report(delta=0.0, goal_failures=25):
            diagnostics = {
                name: {
                    "entry_standardized_partial_auc": 0.70 + delta
                }
                for name in names
            }
            diagnostics["Med-RD/M_L1_vs_L2"] = {
                "conditional_auc": 0.90 + delta}
            return {
                "average_3class_accuracy_percent": 94.0,
                "small_k0_false_positive_percent": 0.4,
                "global_c0_false_positive_percent": 2.0,
                "performance_gate": {"failure_count": 8},
                "goal_performance_gate": {
                    "metric_failure_count": goal_failures,
                    "evidence_failure_count": 3,
                },
                "targeted_probability_diagnostics": diagnostics,
            }

        decision = evaluate_a38_promotion(
            report(), report(delta=0.003, goal_failures=24))
        self.assertEqual(decision["status"], "PASS")
        self.assertEqual(
            decision["baseline"],
            "A35_independent_component_tree_fusion")
        self.assertEqual(
            decision["candidate"],
            "A38_component_proxy_feature_extension")
        decision = evaluate_a38_promotion(
            report(), report(delta=0.003, goal_failures=25))
        self.assertEqual(decision["status"], "FAIL")

    def test_goal_candidate_sort_prioritizes_worst_cell_contract(self):
        average_better_but_failing = {
            "selection_score": 0.95,
            "min_cell_acc_3class": 0.93,
            "min_supported_class1_recall": 0.95,
        }
        goal_passing = {
            "selection_score": 0.80,
            "min_cell_acc_3class": 0.94,
            "min_supported_class1_recall": 0.90,
        }
        self.assertGreater(
            _goal_candidate_sort_key(goal_passing),
            _goal_candidate_sort_key(average_better_but_failing),
        )
        selected = _insert_topk_candidate(
            [average_better_but_failing], goal_passing, k=1)
        self.assertIs(selected[0], goal_passing)

    def test_goal_candidate_sort_includes_safety_and_not_scalar_score(self):
        unsafe_high_score = {
            "selection_score": 0.99,
            "min_cell_acc_3class": 0.96,
            "min_supported_class1_recall": 0.93,
            "min_supported_class_diagonal_recall": 0.93,
            "small_k0_fp_rate": 0.006,
            "c0_fp_rate": 0.020,
        }
        safe_lower_score = {
            "selection_score": 0.80,
            "min_cell_acc_3class": 0.94,
            "min_supported_class1_recall": 0.90,
            "min_supported_class_diagonal_recall": 0.90,
            "small_k0_fp_rate": 0.005,
            "c0_fp_rate": 0.025,
        }
        self.assertGreater(
            _goal_candidate_sort_key(safe_lower_score),
            _goal_candidate_sort_key(unsafe_high_score),
        )

    def test_strict_validation_promotion_requires_both_goal_and_safety(self):
        base = {
            "schema": "stage0_nn_validation_selection_v2",
            "split": "validation",
            "test_labels_used": False,
            "average_3class_accuracy_percent": 96.0,
            "performance_gate": {"passed": True, "failure_count": 0},
            "goal_performance_gate": {
                "passed": True,
                "status": "PASS",
                "metric_failure_count": 0,
                "evidence_failure_count": 0,
            },
            "artifact_identity": {
                "dataset_sha256": "a" * 64,
                "model_sha256": "b" * 64,
                "threshold_sha256": "c" * 64,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "selection_metrics.json"
            report_path.write_text("{}", encoding="utf-8")
            passed = evaluate_validation_promotion(
                base, "A40", report_path)
            self.assertEqual(passed["status"], "PASS")
            failed_safety = copy.deepcopy(base)
            failed_safety["performance_gate"]["passed"] = False
            failed = evaluate_validation_promotion(
                failed_safety, "A40", report_path)
            self.assertEqual(failed["status"], "FAIL")
            failed_goal = copy.deepcopy(base)
            failed_goal["goal_performance_gate"]["passed"] = False
            failed = evaluate_validation_promotion(
                failed_goal, "A40", report_path)
            self.assertEqual(failed["status"], "FAIL")

    def test_strict_validation_promotion_recomputes_artifact_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            model_dir = run_dir / "output" / "models"
            model_dir.mkdir(parents=True)
            dataset_path = Path(directory) / "dataset.parquet"
            dataset_path.write_bytes(b"sealed-dataset")
            artifact_names = (
                "best_model.pth",
                "best_thresholds.json",
                "minmax_scaler.pkl",
                "minmax_scaler.json",
            )
            artifacts = {}
            for index, name in enumerate(artifact_names):
                path = model_dir / name
                path.write_bytes(f"artifact-{index}".encode("ascii"))
                payload = path.read_bytes()
                artifacts[name] = {
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
            (run_dir / "run_manifest.json").write_text(
                json.dumps({"experiment_id": "A40"}),
                encoding="utf-8",
            )
            (model_dir / "model_manifest.json").write_text(
                json.dumps({
                    "schema": "stage0_nn_artifact_v1",
                    "data_contract": {
                        "dataset_path": str(dataset_path),
                        "dataset_sha256": hashlib.sha256(
                            dataset_path.read_bytes()).hexdigest(),
                    },
                    "artifacts": artifacts,
                }),
                encoding="utf-8",
            )
            validation_report = {
                "artifact_identity": {
                    "dataset_sha256": hashlib.sha256(
                        dataset_path.read_bytes()).hexdigest(),
                    "model_sha256": artifacts[
                        "best_model.pth"]["sha256"],
                    "threshold_sha256": artifacts[
                        "best_thresholds.json"]["sha256"],
                },
            }

            passed = _verify_run_artifacts(
                run_dir, validation_report, "A40")
            self.assertEqual(passed["status"], "PASS")
            (model_dir / "best_model.pth").write_bytes(b"tampered")
            failed = _verify_run_artifacts(
                run_dir, validation_report, "A40")
            self.assertEqual(failed["status"], "FAIL")
            self.assertTrue(any(
                "best_model.pth" in failure
                for failure in failed["failures"]
            ))

    def test_minimum_supported_class1_recall_ignores_unsupported_cells(self):
        recalls = torch.tensor([
            [99.0, 91.0],
            [5.0, 95.0],
        ])
        counts = torch.tensor([
            [100, 100],
            [99, 101],
        ])
        self.assertAlmostEqual(
            _minimum_supported_class1_recall(
                recalls, counts, minimum_support=100),
            0.91,
        )

    def test_minimum_supported_diagonal_recall_covers_each_level_cell(self):
        true_levels = torch.ones(100, 4, dtype=torch.long)
        predicted_levels = true_levels.clone()
        predicted_levels[:20, 0] = 0
        munition_ids = torch.zeros(100, dtype=torch.long)
        self.assertAlmostEqual(
            _minimum_supported_diagonal_recall(
                predicted_levels,
                true_levels,
                munition_ids,
                minimum_support=100,
            ),
            0.80,
        )

    def test_l1_threshold_search_enforces_false_positive_cap(self):
        probabilities = torch.tensor([
            0.10, 0.20, 0.30, 0.40,
            0.15, 0.25, 0.35, 0.45,
        ])
        targets = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        grid = [i / 100.0 for i in range(5, 101, 5)]
        threshold, _, _, _, reported_fpr = _search_l1_threshold(
            probabilities,
            targets,
            grid,
            max_fp_rate=0.25,
            recall_weight=0.7,
        )
        predictions = probabilities >= threshold
        observed_fpr = float(predictions[:4].float().mean().item())
        self.assertLessEqual(observed_fpr, 0.25)
        self.assertAlmostEqual(observed_fpr, reported_fpr, places=7)

    def test_goal_threshold_feasibility_finds_separated_ordinal_classes(self):
        p1 = np.asarray([
            0.01, 0.02, 0.03, 0.04,
            0.70, 0.75, 0.80, 0.85,
            0.95, 0.96, 0.97, 0.98,
        ])
        p2 = np.asarray([
            0.001, 0.002, 0.003, 0.004,
            0.05, 0.06, 0.07, 0.08,
            0.90, 0.91, 0.92, 0.93,
        ])
        levels = np.repeat(np.arange(3), 4)
        decision = evaluate_cell_threshold_feasibility(
            p1,
            p2,
            levels,
            np.arange(0.05, 1.01, 0.05),
            (True, True),
            minimum_support=4,
        )
        self.assertTrue(decision["metric_goal_feasible"])
        self.assertTrue(decision["evidence_sufficient"])
        self.assertGreaterEqual(
            decision["goal_candidate"][
                "three_class_accuracy_percent"],
            94.0,
        )
        self.assertTrue(all(
            value >= 90.0
            for value in decision["goal_candidate"][
                "class_diagonal_recall_percent"]
        ))

    def test_terminal_rule_proxy_union_is_monotone_and_bounded(self):
        frame = pd.DataFrame({
            f"phys_{mechanism}_{task}_ge{level}_rule_proxy": (
                [0.2, 0.9]
                if mechanism == "fragment" and level == 1
                else [0.3, 0.4]
                if mechanism == "shock" and level == 1
                else [0.5, 0.8]
                if mechanism == "fragment"
                else [0.4, 0.7]
            )
            for mechanism in ("fragment", "shock")
            for task in ("K", "M", "F", "C")
            for level in (1, 2)
        })
        probabilities = _terminal_rule_proxy_probabilities(frame)
        self.assertEqual(probabilities.shape, (2, 4, 2))
        self.assertTrue(np.all(
            probabilities[..., 1] <= probabilities[..., 0]))
        self.assertTrue(np.all(
            (probabilities >= 0.0) & (probabilities <= 1.0)))
        self.assertAlmostEqual(
            float(probabilities[0, 0, 0]), 0.44, places=7)

    def test_goal_threshold_feasibility_preserves_small_k_safety_cap(self):
        p1 = np.asarray([
            *np.linspace(0.01, 0.20, 200),
            *np.linspace(0.21, 0.25, 20),
        ])
        p2 = np.zeros_like(p1)
        levels = np.asarray([0] * 200 + [1] * 20)
        decision = evaluate_cell_threshold_feasibility(
            p1,
            p2,
            levels,
            np.arange(0.01, 1.01, 0.01),
            (True, False),
            minimum_support=20,
            maximum_l0_false_positive_rate=0.005,
        )
        self.assertTrue(decision["metric_goal_feasible"])
        self.assertLessEqual(
            decision["goal_candidate"][
                "l0_false_positive_percent"],
            0.5,
        )

    def test_l1_threshold_prefers_recall_floor_when_safety_feasible(self):
        probabilities = torch.tensor([
            0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38,
            0.35, 0.36, 0.90, 0.95,
        ])
        targets = torch.tensor([0] * 8 + [1] * 4)
        grid = [i / 100.0 for i in range(5, 101, 5)]
        threshold, _, _, recall, reported_fpr = _search_l1_threshold(
            probabilities,
            targets,
            grid,
            max_fp_rate=0.50,
            recall_weight=0.0,
            minimum_recall=0.85,
            maximum_accuracy_drop=None,
        )
        self.assertGreaterEqual(recall, 0.85)
        self.assertLessEqual(reported_fpr, 0.50)
        self.assertLessEqual(threshold, 0.35 + 1e-7)

    def test_joint_threshold_prefers_exact_class1_recall_floor(self):
        true_level = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)
        target1 = (true_level >= 1).float()
        target2 = (true_level >= 2).float()
        probability1 = torch.tensor([
            0.10, 0.20, 0.30, 0.40,
            0.55, 0.60, 0.65, 0.70,
            0.90, 0.92, 0.94, 0.96,
        ])
        probability2 = torch.tensor([
            0.05, 0.05, 0.10, 0.10,
            0.45, 0.50, 0.55, 0.60,
            0.82, 0.86, 0.90, 0.94,
        ])
        grid = [i / 100.0 for i in range(5, 101, 5)]
        threshold1, threshold2, *_ = _search_joint_ordinal_thresholds(
            probability1,
            probability2,
            target1,
            target2,
            grid,
            max_l0_fp_rate=0.25,
            minimum_exact_l1_recall=0.85,
            maximum_accuracy_drop=None,
        )
        pass1 = probability1 >= threshold1
        pass2 = pass1 & (probability2 >= threshold2)
        prediction = pass1.long() + pass2.long()
        exact_recall = float(
            (prediction[true_level == 1] == 1).float().mean().item())
        l0_fp = float(
            (prediction[true_level == 0] > 0).float().mean().item())
        self.assertGreaterEqual(exact_recall, 0.85)
        self.assertLessEqual(l0_fp, 0.25)

    def test_goal_aware_joint_threshold_prefers_full_94_90_feasibility(self):
        true_level = torch.tensor(
            [0] * 1000 + [1] * 100 + [2] * 100)
        target1 = (true_level >= 1).float()
        target2 = (true_level >= 2).float()
        probability1 = torch.tensor(
            [0.10] * 1000 + [0.80] * 100 + [0.90] * 100)
        probability2 = torch.tensor(
            [0.05] * 1000
            + [0.65] * 10 + [0.40] * 90
            + [0.80] * 85 + [0.65] * 5 + [0.40] * 10
        )
        grid = [0.50, 0.60, 0.70, 0.90]

        legacy_t1, legacy_t2, *_ = _search_joint_ordinal_thresholds(
            probability1, probability2, target1, target2, grid,
            minimum_exact_l1_recall=0.85,
            maximum_accuracy_drop=None,
        )
        goal_t1, goal_t2, *_ = _search_joint_ordinal_thresholds(
            probability1, probability2, target1, target2, grid,
            minimum_exact_l1_recall=0.90,
            maximum_accuracy_drop=None,
            minimum_three_class_accuracy=0.94,
            minimum_class_diagonal_recall=0.90,
        )

        def metrics(threshold1, threshold2):
            pass1 = probability1 >= threshold1
            pass2 = pass1 & (probability2 >= threshold2)
            predicted = pass1.long() + pass2.long()
            accuracy = float((predicted == true_level).float().mean())
            recalls = [
                float((predicted[true_level == level] == level).float().mean())
                for level in (0, 1, 2)
            ]
            return accuracy, recalls

        legacy_accuracy, legacy_recalls = metrics(
            legacy_t1, legacy_t2)
        goal_accuracy, goal_recalls = metrics(goal_t1, goal_t2)
        self.assertLess(legacy_recalls[2], 0.90)
        self.assertGreaterEqual(goal_accuracy, 0.94)
        self.assertTrue(all(
            recall >= 0.90 - 1e-7
            for recall in goal_recalls
        ))

    def test_export_accepts_only_current_constrained_threshold_schema(self):
        heads = ("K1", "K2", "M1", "M2", "F1", "F2", "C1", "C2")
        payload = {
            "_schema": "v7_monotone_fpr_constrained",
            **{head: 0.5 for head in heads},
            "per_munition": {
                head: {str(munition_id): 0.5 for munition_id in range(4)}
                for head in heads
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            threshold_path = os.path.join(directory, "best_thresholds.json")
            with open(threshold_path, "w", encoding="utf-8") as stream:
                json.dump(payload, stream)
            default, per_munition, schema = _load_thresholds(threshold_path)
            self.assertEqual(schema, "v7_monotone_fpr_constrained")
            self.assertEqual(default["C1"], 0.5)
            self.assertEqual(per_munition["C1"][1], 0.5)

            payload["_schema"] = "v8_exact_l1_floor_constrained"
            with open(threshold_path, "w", encoding="utf-8") as stream:
                json.dump(payload, stream)
            _, _, schema = _load_thresholds(threshold_path)
            self.assertEqual(schema, "v8_exact_l1_floor_constrained")

            payload["_schema"] = "v6_monotone_cellwise"
            with open(threshold_path, "w", encoding="utf-8") as stream:
                json.dump(payload, stream)
            with self.assertRaises(RuntimeError):
                _load_thresholds(threshold_path)

    def test_onnx_parity_uses_float32_absolute_and_relative_tolerance(self):
        reference = np.zeros((1, 4, 2), dtype=np.float32)
        expected_float32_drift = np.full_like(reference, 1.144e-5)
        excessive_drift = np.full_like(reference, 1e-3)
        self.assertTrue(
            _onnx_parity_stats(
                reference, expected_float32_drift)["passed"])
        self.assertFalse(
            _onnx_parity_stats(reference, excessive_drift)["passed"])

    def test_ablation_comparison_requires_hashed_completion_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = os.path.join(directory, "A0_full", "seed42")
            eval_dir = os.path.join(run_dir, "output", "eval")
            os.makedirs(eval_dir)
            metrics_path = os.path.join(eval_dir, "test_metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as stream:
                json.dump({"avg_acc": 90.0}, stream)

            completed_path, reason = _validate_completed_result(
                Path(run_dir))
            self.assertIsNone(completed_path)
            self.assertIn("completion marker", str(reason))

            digest = hashlib.sha256()
            with open(metrics_path, "rb") as stream:
                digest.update(stream.read())
            with open(
                    os.path.join(eval_dir, "evaluation_status.json"),
                    "w", encoding="utf-8") as stream:
                json.dump({
                    "status": "COMPLETE",
                    "metrics_sha256": digest.hexdigest(),
                }, stream)
            completed_path, reason = _validate_completed_result(
                Path(run_dir))
            self.assertEqual(str(completed_path), metrics_path)
            self.assertIsNone(reason)

    def test_ablation_console_filter_keeps_progress_not_diagnostics(self):
        self.assertTrue(_should_echo_summary_line(
            "Epoch 003/45 | Train Loss: 0.2\n"))
        self.assertTrue(_should_echo_summary_line(
            "    -> Selection | score=0.5\n"))
        self.assertFalse(_should_echo_summary_line(
            "[Dataset] [Adaptive] class-1 task counts: {...}\n"))

    def test_paired_root_bootstrap_reports_percentage_point_delta(self):
        result = _cluster_bootstrap_mean_delta(
            np.asarray([0.1, 0.1, 0.1], dtype=np.float64),
            np.asarray(["r1", "r2", "r3"]),
            repetitions=100,
            rng=np.random.default_rng(42),
        )
        self.assertAlmostEqual(
            result["estimate_percentage_points"], 10.0)
        self.assertAlmostEqual(
            result["ci95_low_percentage_points"], 10.0)
        self.assertAlmostEqual(
            result["ci95_high_percentage_points"], 10.0)

    def test_performance_ablation_summary_flattens_current_eval_schema(self):
        cell_metrics = {
            munition: {
                task: {
                    "class1_recall": 80.0,
                    "class1_f1": 75.0,
                    "three_class_accuracy": 90.0,
                }
                for task in ("K", "M", "F", "C")
            }
            for munition in ("Small", "Med-LM", "Med-RD", "Heavy")
        }
        metrics = {
            "avg_acc": 91.0,
            "overall_acc": {
                "K": 88.0, "M": 93.0, "F": 95.0, "C": 88.0},
            "small_k1": {"k0_fp": 0.4},
            "small_c1": {"c0_fp": 2.0, "small_c0_fp": 1.0},
            "cell_metrics": cell_metrics,
            "probability_metrics": {
                "K1": {
                    "brier_mc_mean": 0.10,
                    "cross_entropy_mc_mean": 0.30,
                    "ece_mc_mean_10bin": 0.02,
                    "average_precision": 0.70,
                },
                "K2": {
                    "brier_mc_mean": 0.20,
                    "cross_entropy_mc_mean": 0.40,
                    "ece_mc_mean_10bin": 0.04,
                    "average_precision": 0.60,
                },
            },
        }
        flattened = _flatten_metrics(metrics)
        self.assertAlmostEqual(flattened["mean_brier_mc_mean"], 0.15)
        rows = [
            {"experiment_id": "A0_full", "seed": 42, **flattened},
            {"experiment_id": "A0_full", "seed": 43, **flattened},
            {
                "experiment_id": "A13_with_label_confidence",
                "seed": 42,
                **{**flattened, "avg_accuracy": 90.0},
            },
        ]
        aggregate = _aggregate(rows)
        comparison = _delta(
            aggregate["A0_full"],
            aggregate["A13_with_label_confidence"],
        )
        self.assertAlmostEqual(comparison["avg_accuracy"], 1.0)

    def test_manifest_rejects_changed_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in (
                    "best_model.pth", "best_thresholds.json",
                    "minmax_scaler.pkl", "minmax_scaler.json"):
                with open(os.path.join(directory, name), "wb") as stream:
                    stream.write(name.encode("utf-8"))
            contract = {
                "dataset_sha256": "a" * 64,
                "feature_names": ["x"],
            }
            write_model_manifest(
                directory, contract, {"in_dim": 1}, {}, seed=42)
            load_and_verify_manifest(
                directory, dataset_sha256="a" * 64,
                feature_names=["x"])
            with open(os.path.join(directory, "best_model.pth"), "ab") as stream:
                stream.write(b"changed")
            with self.assertRaises(RuntimeError):
                load_and_verify_manifest(directory)


if __name__ == "__main__":
    unittest.main()
