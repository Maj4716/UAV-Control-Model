# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import Future
from unittest.mock import patch

import numpy as np
import pandas as pd

from loitering_munition_damage_twin.stage0.component_supervision import (
    COMPONENT_SUPERVISION_FILENAME,
    COMPONENT_SUPERVISION_PROFILE_FILENAME,
    COMPONENT_TARGET_COLUMNS,
)
from loitering_munition_damage_twin.simulation.coordinate_frames import (
    FRAME_CONVENTION_VERSION,
    TerminalEncounterState,
    body_to_ned_matrix,
    quaternion_from_euler,
    quaternion_to_body_to_ned_matrix,
)
from loitering_munition_damage_twin.stage0.generation import (
    CONFIG,
    PhysicsAwareSampler,
    build_dataset_pipeline,
    _init_worker,
    _emit_logit_adjustment,
    _finalize_sample_weights,
    _process_single_encounter,
    _mc_estimate_is_resolved,
    _mc_resolution_diagnostics,
    _rebalance_evaluation_level_support,
    _minimum_total_exact_level_support,
    _level2_total_support_deficits,
    _level2_support_topoff_budget,
    _cap_root_families,
    _cap_all_ordinal_positive_families,
    _take_target_rows_with_capacity,
    _validate_generation_config,
    _write_dataset_with_profile,
    _stable_uint32,
    _label_mc_rng_pair,
    _phase1_checkpoint_identity,
    _prepare_phase1_checkpoint,
    _load_simulation_checkpoint,
    _write_simulation_checkpoint_part,
)
from loitering_munition_damage_twin.simulation.engine import (
    ArmorPlate,
    BoxGeometry,
    DamageEngine,
    DamageTreeResult,
    EncounterCondition,
    FragmentState,
    Warhead,
    create_small_loitering_munition,
    detect_all_hits,
    load_armor_plates,
    load_vehicle_model,
    parse_component_geometry,
    ray_aabb,
    ray_box,
)
from loitering_munition_damage_twin.stage0.validation import (
    Stage0ValidationError,
    _validate_exact_level_evidence,
    validate_stage0_dataset,
)
from loitering_munition_damage_twin.stage0.reachability import DEFAULT_TARGETS
from loitering_munition_damage_twin.stage0.build_stage0_c2_challenge import (
    CHALLENGE_SCHEMA,
    build_c2_challenge,
    select_root_independent_c2_rows,
    validate_c2_challenge,
    write_c2_challenge,
)
from loitering_munition_damage_twin.stage0.relabel_high_mc import (
    MANIFEST_SCHEMA,
    REQUIRED_REPLAY_COLUMNS,
    _configuration_sha256,
    _configuration_snapshot,
    _new_manifest,
    _prepare_records,
    _shard_bounds,
    _validate_manifest,
    _validate_replay_frame,
)


class CoordinateConventionTests(unittest.TestCase):
    def test_pitch_and_yaw_change_forward_velocity(self):
        level = EncounterCondition.from_speed_and_attitude(pitch_deg=0.0, speed=100.0)
        dive = EncounterCondition.from_speed_and_attitude(pitch_deg=-90.0, speed=100.0)
        east = EncounterCondition.from_speed_and_attitude(yaw_deg=90.0, pitch_deg=0.0, speed=100.0)
        np.testing.assert_allclose(level.velocity_vector_ms, [0.0, 100.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(dive.velocity_vector_ms, [0.0, 0.0, -100.0], atol=1e-9)
        np.testing.assert_allclose(east.velocity_vector_ms, [100.0, 0.0, 0.0], atol=1e-9)

    def test_quaternion_matches_euler_rotation(self):
        yaw, pitch, roll = np.radians([33.0, -27.0, 14.0])
        q = quaternion_from_euler(yaw, pitch, roll)
        np.testing.assert_allclose(
            quaternion_to_body_to_ned_matrix(q),
            body_to_ned_matrix(yaw, pitch, roll),
            atol=1e-12,
        )

    def test_terminal_state_is_si_and_adapts_once(self):
        state = TerminalEncounterState(
            position_t_m=np.array([1.0, 2.0, 3.0]),
            velocity_t_mps=np.array([0.0, 0.0, -80.0]),
            attitude_bn_wxyz=quaternion_from_euler(0.0, -np.pi / 2.0, 0.0),
            munition_id=1,
        )
        legacy = EncounterCondition.from_terminal_state(state)
        np.testing.assert_allclose([legacy.dx, legacy.dy, legacy.dz], [100.0, 200.0, 300.0])
        np.testing.assert_allclose(legacy.velocity_vector_ms, [0.0, 0.0, -80.0])
        self.assertAlmostEqual(legacy.pitch_deg, -90.0, places=8)


class GeometryAndArmorTests(unittest.TestCase):
    def test_rotated_box_uses_obb_not_expanded_aabb(self):
        angle = np.radians(45.0)
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        half = np.array([2.0, 0.5, 0.5])
        extent = np.abs(rotation) @ half
        box = BoxGeometry(
            center=np.zeros(3), half_extents=half,
            aabb_min=-extent, aabb_max=extent, rotation=rotation,
        )
        origin = np.array([1.6, 1.6, 2.0])
        direction = np.array([0.0, 0.0, -1.0])
        self.assertTrue(ray_aabb(origin, direction, box.aabb_min, box.aabb_max)[0])
        self.assertFalse(ray_box(origin, direction, box)[0])

    @staticmethod
    def _box_component(component_id, name, z, thickness):
        return {
            "id": component_id,
            "name": name,
            "geometry": {
                "shape": "长方体",
                "dimensions": {"length_or_radius": 10.0, "width": 10.0, "height": 1.0},
                "position": {"x": 0.0, "y": 0.0, "z": z},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
            },
            "material": {
                "equivalent_thickness": thickness,
                "vulnerable_area_ratio": 1.0,
                "overpressure_threshold": 0.3,
            },
        }

    def test_armor_csv_override_reaches_internal_hit(self):
        components = [
            self._box_component(69, "上装甲", 0.0, 30.0),
            self._box_component(1, "内部部件", -5.0, 5.0),
        ]
        fragment = FragmentState(
            frag_id=0, mass_g=8.0,
            origin_T=np.array([0.0, 0.0, 10.0]),
            direction_T=np.array([0.0, 0.0, -1.0]),
            speed_initial=1000.0, gurney_speed=1000.0,
            taylor_angle_deg=0.0,
        )
        plate = ArmorPlate(
            name="上装甲", thickness_mm=15.0,
            aabb_min=np.array([-5.0, -5.0, -0.5]),
            aabb_max=np.array([5.0, 5.0, 0.5]),
        )
        with_override = detect_all_hits([fragment], components, Warhead(), [plate])
        without_override = detect_all_hits([fragment], components, Warhead(), [])
        internal_with = next(hit for hit in with_override if hit.component_id == 1)
        internal_without = next(hit for hit in without_override if hit.component_id == 1)
        self.assertEqual(internal_with.armor_traversed_mm, 15.0)
        self.assertEqual(internal_without.armor_traversed_mm, 30.0)

    def test_sampler_aabb_contains_every_parsed_component(self):
        sampler = PhysicsAwareSampler()
        for component in sampler.components:
            _, geometry = parse_component_geometry(component)
            self.assertTrue(np.all(geometry.aabb_min >= sampler.min_aabb - 1e-9))
            self.assertTrue(np.all(geometry.aabb_max <= sampler.max_aabb + 1e-9))


class ProbabilityAndLineageTests(unittest.TestCase):
    @staticmethod
    def _fake_simulation_batch(sampler, inputs):
        if inputs.empty:
            return pd.DataFrame()
        frame = inputs.copy().reset_index(drop=True)
        frame["munition_id"] = frame["m_id"].astype(int)
        aliases = {
            "x_cm": "x", "y_cm": "y", "z_cm": "z",
            "vx_ms": "vx", "vy_ms": "vy", "vz_ms": "vz",
        }
        for target, source in aliases.items():
            frame[target] = frame[source].astype(float)
        frame["norm_velocity"] = np.sqrt(
            frame["vx"] ** 2 + frame["vy"] ** 2 + frame["vz"] ** 2)
        for angle in ("yaw", "pitch", "roll"):
            frame[f"sin_{angle}"] = np.sin(np.radians(frame[angle]))
            frame[f"cos_{angle}"] = np.cos(np.radians(frame[angle]))

        for task in "KMFC":
            levels = []
            p1_values = []
            p2_values = []
            for sample_id, munition_id in zip(frame["sample_id"], frame["m_id"]):
                level = _stable_uint32(f"{sample_id}|{task}") % 3
                if level == 2 and not CONFIG["ORDINAL_APPLICABILITY"][int(munition_id)][task][1]:
                    level = 1
                if level == 0:
                    p1, p2 = 0.1, 0.1
                elif level == 1:
                    p1, p2 = 0.8, 0.1
                elif task == "C":
                    p1, p2 = 0.8, 0.8
                else:
                    p1, p2 = 0.1, 0.8
                levels.append(int(level))
                p1_values.append(p1)
                p2_values.append(p2)
            p1 = np.asarray(p1_values, dtype=float)
            p2 = np.asarray(p2_values, dtype=float)
            ge1 = p1 if task == "C" else 1.0 - (1.0 - p1) * (1.0 - p2)
            ge2 = np.minimum(p2, ge1)
            frame[f"{task}1_prob"] = p1
            frame[f"{task}2_prob"] = p2
            frame[f"{task}_ge1_prob"] = ge1
            frame[f"{task}_ge2_prob"] = ge2
            frame[f"{task}_level"] = levels
            for level_index in (1, 2):
                frame[f"{task}{level_index}_prob_std"] = 0.0
                frame[f"{task}_ge{level_index}_prob_std"] = 0.0
                for mechanism in ("fragment", "shock"):
                    frame[f"{mechanism}_{task}_ge{level_index}_prob"] = (
                        frame[f"{task}_ge{level_index}_prob"] * 0.5)
        frame["label_mc_replicates"] = frame[
            "label_mc_min_replicates"].astype(int)
        frame["overall_score"] = 0.5
        frame["fragment_overall_score"] = 0.25
        frame["shock_overall_score"] = 0.25
        frame["total_hits"] = 1.0
        frame["total_penetrations"] = 0.0
        frame["K_task_weight"] = 1.0
        frame["C_task_weight"] = 1.0
        existing_component_columns = [
            column for column in COMPONENT_TARGET_COLUMNS
            if column in frame.columns
        ]
        if existing_component_columns:
            frame = frame.drop(
                columns=existing_component_columns)
        frame = pd.concat([
            frame,
            pd.DataFrame({
                column: np.full(
                    len(frame), 0.1, dtype=np.float32)
                for column in COMPONENT_TARGET_COLUMNS
            }, index=frame.index),
        ], axis=1)
        return frame

    def test_generation_config_rejects_structural_zero_topoff(self):
        _validate_generation_config()
        bad_plan = dict(CONFIG["PHASE2_TOP_OFF_PLAN"])
        bad_plan[0] = {"C2_prob": 1.0}
        with patch.dict(CONFIG, {"PHASE2_TOP_OFF_PLAN": bad_plan}):
            with self.assertRaisesRegex(RuntimeError, "结构零"):
                _validate_generation_config()

    def test_med_lm_c2_is_applicable_targeted_and_probeable(self):
        _validate_generation_config()
        self.assertTrue(CONFIG["ORDINAL_APPLICABILITY"][1]["C"][1])
        med_lm_plan = CONFIG["PHASE2_TOP_OFF_PLAN"][1]
        self.assertAlmostEqual(sum(med_lm_plan.values()), 1.0)
        self.assertAlmostEqual(med_lm_plan["C2_prob"], 0.20)
        self.assertGreater(med_lm_plan["C2_prob"], 0.0)
        self.assertEqual(DEFAULT_TARGETS["med_lm_c2"], (1, "C2_prob"))
        self.assertEqual(CONFIG["C2_CLUSTER_SIZE"], 6)

    def test_rule_events_are_converted_to_explicit_ordinal_probabilities(self):
        result = DamageTreeResult(K1_prob=0.4, K2_prob=0.4, C1_prob=0.7, C2_prob=0.2)
        ordinal = result.ordinal_probability_dict
        self.assertAlmostEqual(ordinal["K_ge1_prob"], 0.64)
        self.assertAlmostEqual(ordinal["K_ge2_prob"], 0.4)
        self.assertAlmostEqual(ordinal["C_ge1_prob"], 0.7)
        self.assertAlmostEqual(ordinal["C_ge2_prob"], 0.2)
        self.assertLessEqual(result.overall_score, 1.0)

    def test_phase1_samples_have_immutable_lineage_and_split(self):
        np.random.seed(CONFIG["RANDOM_SEED"])
        sampler = PhysicsAwareSampler()
        frame = sampler._generate_lhs_batch(256, "M_F", force_m_id=0)
        self.assertEqual(frame["sample_id"].nunique(), len(frame))
        self.assertTrue((frame["sample_id"] == frame["root_seed_id"]).all())
        self.assertEqual(set(frame["split_role"].unique()), {"train", "val", "test"})
        self.assertEqual(set(frame["frame_version"].unique()), {FRAME_CONVENTION_VERSION})

    def test_small_simulation_batch_preserves_lineage_and_mc_metadata(self):
        components = load_vehicle_model()
        plates = load_armor_plates()
        _init_worker(components, plates)
        row = {
            "x": 0.0, "y": 0.0, "z": 250.0,
            "vx": 0.0, "vy": 0.0, "vz": -180.0,
            "pitch": -90.0, "roll": 0.0, "yaw": 0.0,
            "target_x": 0.0, "target_y": 0.0, "target_z": 100.0,
            "m_id": 0, "sample_id": "unit-root-1", "root_seed_id": "unit-root-1",
            "parent_id": "", "crawl_stage": 0, "split_role": "test",
            "frame_version": FRAME_CONVENTION_VERSION,
            "dataset_schema": CONFIG["DATASET_SCHEMA"],
            "label_mc_replicates": 2,
        }
        result = _process_single_encounter((0, row))
        self.assertEqual(result["sample_id"], "unit-root-1")
        self.assertEqual(result["root_seed_id"], "unit-root-1")
        self.assertEqual(result["split_role"], "test")
        self.assertEqual(result["label_mc_replicates"], 2)
        for task in "KMFC":
            self.assertLessEqual(result[f"{task}_ge2_prob"], result[f"{task}_ge1_prob"])
            self.assertIn(f"fragment_{task}_ge1_prob", result)
            self.assertIn(f"shock_{task}_ge1_prob", result)

    def test_shockwave_cache_is_damage_equivalent(self):
        components = load_vehicle_model()
        plates = load_armor_plates()
        engine = DamageEngine(armor_plates=plates)
        projectile = create_small_loitering_munition()
        encounter = EncounterCondition(
            dx=20.0, dy=-35.0, dz=180.0,
            vx=5.0, vy=15.0, vz=-170.0,
            pitch_deg=-82.0, roll_deg=3.0, yaw_deg=12.0,
        )
        baseline = engine.evaluate(
            projectile, encounter, components,
            rng_seed=20260806,
        )
        cache = {
            int(component.component_id): float(
                component.shockwave_damage_prob)
            for component in baseline.component_results
        }
        cached = engine.evaluate(
            projectile, encounter, components,
            rng_seed=20260806,
            shockwave_probability_cache=cache,
        )
        np.testing.assert_allclose(
            cached.damage_tree.ordinal_probability_vector,
            baseline.damage_tree.ordinal_probability_vector,
            rtol=0.0, atol=0.0,
        )
        for expected, observed in zip(
                baseline.component_results,
                cached.component_results):
            self.assertEqual(
                expected.fragment_damage_prob,
                observed.fragment_damage_prob)
            self.assertEqual(
                expected.shockwave_damage_prob,
                observed.shockwave_damage_prob)
            self.assertEqual(
                expected.combined_damage_prob,
                observed.combined_damage_prob)
        with self.assertRaises(ValueError):
            engine.evaluate(
                projectile, encounter, components,
                rng_seed=20260806,
                shockwave_probability_cache={0: 0.5},
            )

    def test_adaptive_mc_requires_precision_and_decision_margin(self):
        self.assertGreaterEqual(
            int(CONFIG["LABEL_MC_MIN_REPLICATES"]), 8)
        self.assertGreaterEqual(
            int(CONFIG["LABEL_MC_MAX_REPLICATES"]), 64)
        self.assertLessEqual(
            float(CONFIG["LABEL_MC_STANDARD_ERROR_TARGET"]), 0.02)
        stable_low = np.full(
            (int(CONFIG["LABEL_MC_MIN_REPLICATES"]), 8),
            0.10,
        )
        self.assertTrue(
            _mc_estimate_is_resolved(stable_low))
        stable_diagnostics = _mc_resolution_diagnostics(
            stable_low)
        self.assertTrue(
            stable_diagnostics["all_resolved"])
        self.assertTrue(np.all(
            stable_diagnostics["resolved_mask"]))
        boundary = np.tile(
            np.asarray([0.49, 0.51], dtype=np.float64),
            (int(CONFIG["LABEL_MC_MIN_REPLICATES"]), 4),
        )
        self.assertFalse(
            _mc_estimate_is_resolved(boundary))
        self.assertFalse(
            _mc_resolution_diagnostics(
                boundary)["all_resolved"])
        noisy = stable_low.copy()
        noisy[:, 0] = np.resize(
            np.asarray([0.0, 1.0], dtype=np.float64),
            len(noisy),
        )
        self.assertFalse(
            _mc_estimate_is_resolved(noisy))

    def test_label_mc_uses_antithetic_seed_pairs(self):
        self.assertTrue(CONFIG["LABEL_MC_ANTITHETIC"])
        seed0, sign0 = _label_mc_rng_pair(
            "sample-a", 0)
        seed1, sign1 = _label_mc_rng_pair(
            "sample-a", 1)
        seed2, sign2 = _label_mc_rng_pair(
            "sample-a", 2)
        self.assertEqual(seed0, seed1)
        self.assertEqual((sign0, sign1), (1.0, -1.0))
        self.assertNotEqual(seed0, seed2)
        self.assertEqual(sign2, 1.0)

    def test_root_family_cap_limits_correlated_descendants(self):
        frame = pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(10)],
            "root_seed_id": ["root-a"] * 8 + ["root-b"] * 2,
            "parent_id": [""] + ["s0"] * 7 + ["", "s8"],
            "crawl_stage": [0] + [1] * 7 + [0, 1],
        })
        capped, stats = _cap_root_families(frame, 3, CONFIG["RANDOM_SEED"])
        self.assertEqual(int(capped["root_seed_id"].value_counts().max()), 3)
        self.assertEqual(len(capped), 5)
        self.assertEqual(stats["rows_removed"], 5)
        self.assertIn("s0", set(capped["sample_id"]))

    def test_crawler_limits_children_per_root_per_stage(self):
        np.random.seed(CONFIG["RANDOM_SEED"])
        sampler = PhysicsAwareSampler()
        seeds = sampler._generate_lhs_batch(3, "M_HUNT", force_m_id=0)
        crawled = sampler.crawl_boundary_stage_2(seeds, num_needed=100)
        if not crawled.empty:
            per_root = crawled["root_seed_id"].value_counts()
            self.assertLessEqual(
                int(per_root.max()),
                int(CONFIG["MAX_CHILDREN_PER_ROOT_PER_STAGE"]),
            )
            self.assertTrue((crawled["sampling_phase"] == "phase2_crawl").all())

    def test_hunt_targets_follow_damage_tree_and_detonation_direction(self):
        np.random.seed(CONFIG["RANDOM_SEED"])
        sampler = PhysicsAwareSampler()
        components = {int(c["id"]): c for c in sampler.components}
        k1_position = components[3]["geometry"]["position"]
        k2_position = components[46]["geometry"]["position"]
        np.testing.assert_allclose(
            sampler.target_centers["K1"][0],
            [k1_position["x"], k1_position["y"], k1_position["z"]],
        )
        np.testing.assert_allclose(
            sampler.target_centers["K2"][0],
            [k2_position["x"], k2_position["y"], k2_position["z"]],
        )
        self.assertEqual(len(sampler.target_centers["K1"]), 1)
        self.assertEqual(len(sampler.target_centers["K2"]), 1)
        self.assertGreaterEqual(len(sampler.target_centers["C2"]), 2)
        self.assertEqual(sampler.target_centers["C2"].shape[1], 3)

        front = sampler._generate_lhs_batch(64, "K2_HUNT", force_m_id=1)
        rear = sampler._generate_lhs_batch(64, "C_HUNT", force_m_id=2)
        c2_front = sampler._generate_lhs_batch(64, "C2_HUNT", force_m_id=1)
        for frame, expected_sign in (
            (front, -1.0), (rear, 1.0), (c2_front, -1.0)
        ):
            los = frame[["target_x", "target_y", "target_z"]].to_numpy() - frame[
                ["x", "y", "z"]].to_numpy()
            velocity = frame[["vx", "vy", "vz"]].to_numpy()
            dot = np.sum(los * velocity, axis=1)
            self.assertTrue((frame["fragment_aim_sign"] == expected_sign).all())
            self.assertTrue((expected_sign * dot > 0).all())
            los_unit = los / np.linalg.norm(los, axis=1, keepdims=True)
            velocity_unit = velocity / np.linalg.norm(
                velocity, axis=1, keepdims=True)
            observed_angle = np.degrees(np.arccos(np.clip(np.sum(
                los_unit * (expected_sign * velocity_unit), axis=1), -1.0, 1.0)))
            expected_angle = float(frame["cone_aim_angle_deg"].iloc[0])
            self.assertLess(abs(float(np.median(observed_angle)) - expected_angle), 3.0)

    def test_target_positive_cap_is_separate_from_global_family_cap(self):
        existing = pd.DataFrame({
            "root_seed_id": ["a"] * 3,
            "C1_prob": [0.8] * 3,
            "C2_prob": [0.0] * 3,
        })
        candidates = pd.DataFrame({
            "root_seed_id": ["a"] * 10 + ["b"] * 10,
            "C1_prob": [0.8] * 20,
            "C2_prob": [0.0] * 20,
        })
        kept = _take_target_rows_with_capacity(
            existing, candidates, "C1_only", 0.5,
            max_rows_per_root=64,
            max_positive_rows_per_root=4,
            seed=CONFIG["RANDOM_SEED"],
        )
        counts = kept["root_seed_id"].value_counts().to_dict()
        self.assertEqual(counts, {"b": 4, "a": 1})

    def test_incidental_ordinal_positives_are_also_family_capped(self):
        frame = pd.DataFrame({
            "root_seed_id": ["a"] * 10 + ["b"] * 3,
            "munition_id": [0] * 13,
            "crawl_stage": [1] * 13,
        })
        for task in "KMFC":
            frame[f"{task}_ge1_prob"] = 0.8
            frame[f"{task}_ge2_prob"] = 0.1
        capped, stats = _cap_all_ordinal_positive_families(
            frame, 0.5, max_rows_per_root_per_cell=4,
            seed=CONFIG["RANDOM_SEED"],
        )
        positives = capped[capped["C_ge1_prob"] >= 0.5]
        self.assertLessEqual(
            int(positives["root_seed_id"].value_counts().max()), 4)
        self.assertIn("m_id=0:K>=1", stats)

    def test_fresh_root_discovery_repeats_until_strict_root_target(self):
        sampler = PhysicsAwareSampler()
        state = {"counter": 0}

        def fake_generate(n_samples, layer_type, force_m_id=None, **kwargs):
            start = state["counter"]
            state["counter"] += n_samples
            roots = [f"fresh-{i}" for i in range(start, start + n_samples)]
            return pd.DataFrame({
                "root_seed_id": roots,
                "sample_id": roots,
                "split_role": ["train"] * n_samples,
            })

        def fake_simulate(inputs):
            result = inputs.copy()
            result["K2_prob"] = 0.3
            result.loc[result.index[:2], "K2_prob"] = 0.7
            return result

        sampler._generate_lhs_batch = fake_generate
        sampler._apply_phase1_filters_and_weights = lambda frame: frame
        sampler.run_simulation_batch = fake_simulate
        existing = pd.DataFrame(columns=[
            "root_seed_id", "sample_id", "split_role", "K2_prob"])
        with patch.dict(CONFIG, {
            "FRESH_ROOT_BATCH_SIZE": 4,
            "FRESH_ROOT_MAX_CANDIDATES_PER_TASK": 12,
            "FRESH_ROOT_MAX_ROUNDS": 3,
            "FRESH_ROOT_CANDIDATE_MULTIPLIER": 1,
        }):
            fresh, stats = sampler.discover_fresh_target_roots(
                existing, 1, "K2_prob", 0.25, 0.5,
                desired_seed_roots=6,
                desired_strict_roots=5,
            )
        self.assertEqual(stats["rounds"], 3)
        self.assertGreaterEqual(stats["strict_roots_after"], 5)
        self.assertGreaterEqual(stats["seed_roots_after"], 6)
        self.assertEqual(fresh["root_seed_id"].nunique(), len(fresh))

    def test_c2_fresh_discovery_uses_cluster_hunt_layer(self):
        sampler = PhysicsAwareSampler()
        sampler._generate_lhs_batch = lambda n, layer, force_m_id=None, **kwargs: pd.DataFrame({
            "root_seed_id": [f"r{i}" for i in range(n)],
            "sample_id": [f"r{i}" for i in range(n)],
            "split_role": ["train"] * n,
            "layer_type": [layer] * n,
        })
        sampler._apply_phase1_filters_and_weights = lambda frame: frame

        def fake_simulate(frame):
            result = frame.copy()
            result["C2_prob"] = 0.8
            return result

        sampler.run_simulation_batch = fake_simulate
        with patch.dict(CONFIG, {
            "FRESH_ROOT_BATCH_SIZE": 4,
            "FRESH_ROOT_MAX_CANDIDATES_PER_TASK": 4,
            "FRESH_ROOT_MAX_ROUNDS": 1,
        }):
            fresh, stats = sampler.discover_fresh_target_roots(
                pd.DataFrame(), 1, "C2_prob", 0.25, 0.5,
                desired_seed_roots=2, desired_strict_roots=2,
            )
        self.assertEqual(stats["target_layer"], "C2_HUNT")
        self.assertTrue((fresh["layer_type"] == "C2_HUNT").all())

    def test_fresh_lateral_shell_is_outside_vehicle_aabb(self):
        np.random.seed(CONFIG["RANDOM_SEED"])
        sampler = PhysicsAwareSampler()
        frame = sampler._generate_lhs_batch(
            512, "K2_HUNT", force_m_id=1,
            exterior_lateral_shell=True,
        )
        clearance = float(CONFIG["FRESH_ROOT_LATERAL_CLEARANCE_RANGE_CM"][0])
        outside_left = frame["x"] <= float(sampler.min_aabb[0]) - clearance
        outside_right = frame["x"] >= float(sampler.max_aabb[0]) + clearance
        self.assertTrue((outside_left | outside_right).all())
        self.assertTrue((frame["z"] >= 0.0).all())
        radius = np.sqrt(frame["x"]**2 + frame["y"]**2 + frame["z"]**2)
        self.assertLessEqual(float(radius.max()), CONFIG["RADIUS_MAX_CM"] + 1e-6)
        self.assertTrue((frame["sampling_geometry"] == "fresh_lateral_shell").all())

    def test_c2_fresh_shell_uses_crew_access_corridor(self):
        np.random.seed(CONFIG["RANDOM_SEED"])
        sampler = PhysicsAwareSampler()
        frame = sampler._generate_lhs_batch(
            1024, "C2_HUNT", force_m_id=1,
            exterior_lateral_shell=True,
        )
        self.assertTrue((
            frame["sampling_geometry"] == "fresh_c2_crew_corridor"
        ).all())
        clearance = float(
            CONFIG["FRESH_ROOT_LATERAL_CLEARANCE_RANGE_CM"][0])
        outside_left = (
            frame["x"] <= float(sampler.min_aabb[0]) - clearance)
        outside_right = (
            frame["x"] >= float(sampler.max_aabb[0]) + clearance)
        self.assertTrue((outside_left | outside_right).all())
        observed_right_share = float(outside_right.mean())
        self.assertAlmostEqual(
            observed_right_share,
            float(CONFIG["C2_FRESH_RIGHT_SIDE_PROB"]),
            delta=0.02,
        )
        # Target jitter is small relative to the gap between the maximum-y
        # cluster group and all remaining cluster centroids.
        maximum_cluster_y = float(
            np.max(sampler.target_centers["C2"][:, 1]))
        other_cluster_y = sampler.target_centers["C2"][:, 1]
        other_cluster_y = other_cluster_y[
            other_cluster_y < maximum_cluster_y - 1e-6]
        separator = 0.5 * (
            maximum_cluster_y + float(np.max(other_cluster_y)))
        observed_preferred_share = float(
            (frame["target_y"] > separator).mean())
        uniform_preferred_share = float(np.mean(np.isclose(
            sampler.target_centers["C2"][:, 1],
            maximum_cluster_y,
            rtol=0.0,
            atol=1e-6,
        )))
        expected_preferred_share = (
            float(CONFIG["C2_FRESH_MAX_Y_CLUSTER_PROB"])
            + (1.0 - float(CONFIG["C2_FRESH_MAX_Y_CLUSTER_PROB"]))
            * uniform_preferred_share
        )
        self.assertAlmostEqual(
            observed_preferred_share,
            expected_preferred_share,
            delta=0.04,
        )

        crew_y = sampler.target_centers["C2"][:, 1]
        margin_low, margin_high = CONFIG[
            "C2_FRESH_CREW_Y_CORRIDOR_MARGIN_CM"]
        expected_low = max(
            float(sampler.min_aabb[1]),
            float(np.min(crew_y)) - float(margin_low),
        )
        expected_high = min(
            float(sampler.max_aabb[1]),
            float(np.median(crew_y)) + float(margin_high),
        )
        self.assertGreaterEqual(float(frame["y"].min()), expected_low - 1e-6)
        self.assertLessEqual(float(frame["y"].max()), expected_high + 1e-6)
        self.assertTrue((frame["z"] >= 0.0).all())
        self.assertTrue((frame["z"] <= float(sampler.max_aabb[2])).all())
        radius = np.sqrt(frame["x"]**2 + frame["y"]**2 + frame["z"]**2)
        self.assertLessEqual(
            float(radius.max()), float(CONFIG["RADIUS_MAX_CM"]) + 1e-6)

    def test_weight_tempering_meets_ess_floor(self):
        n_rows = 1000
        frame = pd.DataFrame({
            "root_seed_id": [f"r{i}" for i in range(n_rows)],
            "split_role": ["train"] * 800 + ["val"] * 100 + ["test"] * 100,
            "aoa_accept_prob": [1.0] * n_rows,
            # Only train is strongly skewed.  An all-table ESS objective can
            # pass while the actual gradient split remains below the floor.
            "aoa_ipw": [1.0] * 700 + [20.0] * 100 + [1.0] * 200,
            "physics_weight": [1.0] * n_rows,
            "active_sampling_weight": [1.0] * n_rows,
            "class_balance_weight": [1.0] * n_rows,
        })
        weighted = _finalize_sample_weights(frame, 0.5)
        train_weights = weighted.loc[
            weighted["split_role"] == "train", "loss_weight"]
        train_ess = train_weights.sum() ** 2 / np.square(train_weights).sum()
        target = (
            CONFIG["MIN_WEIGHT_ESS_RATIO"] +
            CONFIG["WEIGHT_ESS_TARGET_MARGIN"]
        )
        self.assertGreaterEqual(
            train_ess / len(train_weights), target - 1e-6)
        self.assertGreaterEqual(
            weighted.attrs["weight_train_ess_ratio"], target - 1e-6)
        self.assertLess(weighted.attrs["weight_tempering_alpha"], 1.0)

    def test_k2_ratio_contract_has_distinct_stop_and_final_ceiling(self):
        _validate_generation_config()
        self.assertLess(
            CONFIG["K2_PHASE2_STOP_RATIO"],
            CONFIG["K2_FINAL_MAX_RATIO"],
        )
        with patch.dict(CONFIG, {
            "K2_PHASE2_STOP_RATIO": 0.10,
            "K2_FINAL_MAX_RATIO": 0.05,
        }):
            with self.assertRaisesRegex(RuntimeError, "K2 比例合同"):
                _validate_generation_config()

    def test_c2_challenge_selection_and_atomic_profile(self):
        rows = 24
        candidates = pd.DataFrame({
            "munition_id": [1] * rows,
            "root_seed_id": [f"challenge-root-{i}" for i in range(rows)],
            "sample_id": [f"challenge-sample-{i}" for i in range(rows)],
            "C_ge2_prob": [0.9] * 8 + list(np.linspace(0.49, 0.01, 16)),
            "dataset_schema": [CONFIG["DATASET_SCHEMA"]] * rows,
            "split_role": ["train"] * rows,
        })
        selected = select_root_independent_c2_rows(
            candidates,
            munition_id=1,
            positive_roots=4,
            negative_roots=4,
            valid_threshold=0.5,
            seed=123,
        )
        self.assertEqual(len(selected), 8)
        self.assertEqual(selected["root_seed_id"].nunique(), 8)
        self.assertEqual(set(selected["challenge_schema"]), {CHALLENGE_SCHEMA})
        self.assertEqual(
            selected["challenge_target"].value_counts().to_dict(),
            {1: 4, 0: 4},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output = os.path.join(temp_dir, "c2_challenge.parquet")
            profile_path = write_c2_challenge(
                selected,
                output,
                build_metadata={
                    "valid_threshold": 0.5,
                    "source_dataset": {
                        "path": None, "sha256": None, "rows": None,
                    },
                    "discovery": {"1": {"strict_roots_after": 4}},
                },
            )
            self.assertTrue(os.path.exists(profile_path))
            report = validate_c2_challenge(output)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["root_families"], 8)
            with self.assertRaisesRegex(RuntimeError, "重叠 root"):
                write_c2_challenge(
                    selected,
                    os.path.join(temp_dir, "overlap.parquet"),
                    build_metadata={
                        "valid_threshold": 0.5,
                        "source_dataset": {},
                        "discovery": {},
                    },
                    source_roots={str(selected["root_seed_id"].iloc[0])},
                )

    def test_c2_challenge_builder_uses_disjoint_seed_namespace(self):
        original_seed = CONFIG["RANDOM_SEED"]

        def fake_discovery(
            sampler, existing_pool, munition_id, target_col,
            seed_th, valid_th, desired_seed_roots,
            desired_strict_roots, required_split_role="train",
        ):
            rows = 12
            prefix = f"challenge-{CONFIG['RANDOM_SEED']}-{munition_id}"
            frame = pd.DataFrame({
                "munition_id": [munition_id] * rows,
                "root_seed_id": [f"{prefix}-r{i}" for i in range(rows)],
                "sample_id": [f"{prefix}-s{i}" for i in range(rows)],
                "C_ge2_prob": [0.8] * 6 + [0.4] * 6,
                "dataset_schema": [CONFIG["DATASET_SCHEMA"]] * rows,
                "split_role": ["train"] * rows,
            })
            return frame, {
                "target_layer": "C2_HUNT",
                "split_scope": "all_independent_roots",
                "strict_roots_after": 6,
            }

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            PhysicsAwareSampler,
            "discover_fresh_target_roots",
            new=fake_discovery,
        ):
            output = os.path.join(temp_dir, "built_challenge.parquet")
            report = build_c2_challenge(
                output_path=output,
                positive_roots=2,
                negative_roots=2,
                max_candidates=16,
                batch_size=8,
                base_seed=9001,
                source_dataset=None,
            )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["rows"], 12)
        self.assertEqual(CONFIG["RANDOM_SEED"], original_seed)

    def test_logit_adjustment_uses_train_only_and_binds_dataset_hash(self):
        frame = pd.DataFrame({"split_role": ["train", "train", "test"]})
        for task in "KMFC":
            frame[f"{task}_ge1_prob"] = [0.8, 0.2, 0.8]
            frame[f"{task}_ge2_prob"] = [0.2, 0.2, 0.8]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "logit_adjustment.json")
            _emit_logit_adjustment(
                frame, 0.5, CONFIG["PHYSICAL_PRIOR"], path,
                dataset_sha256="abc123",
            )
            with open(path, "r", encoding="utf-8") as handle:
                adjustment = json.load(handle)
        self.assertAlmostEqual(adjustment["K_ge1_prob"]["pi_train"], 0.5)
        self.assertEqual(adjustment["__meta__"]["n_train_samples"], 2)
        self.assertEqual(adjustment["__meta__"]["dataset_sha256"], "abc123")

    def test_phase1_checkpoint_identity_and_atomic_part_roundtrip(self):
        frame = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "root_seed_id": ["r0", "r1", "r2"],
        })
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(CONFIG, {
            "PHASE1_CHECKPOINT_ENABLED": True,
            "PHASE1_CHECKPOINT_DIR": temp_dir,
            "PHASE1_CHECKPOINT_INTERVAL": 2,
        }):
            identity = _phase1_checkpoint_identity(frame)
            checkpoint_dir = _prepare_phase1_checkpoint(identity)
            _write_simulation_checkpoint_part(checkpoint_dir, [
                (2, {"sample_id": "s2", "value": 2.0}),
                (0, {"sample_id": "s0", "value": 0.0}),
            ])
            restored = _load_simulation_checkpoint(checkpoint_dir, 3)
            changed = frame.copy()
            changed.loc[0, "x"] = 9.0
            changed_identity = _phase1_checkpoint_identity(changed)

        self.assertEqual(sorted(restored), [0, 2])
        self.assertEqual(restored[0]["sample_id"], "s0")
        self.assertEqual(restored[2]["value"], 2.0)
        self.assertNotEqual(
            identity["signature"], changed_identity["signature"])

    def test_complete_checkpoint_allows_only_declared_source_migration(self):
        frame = pd.DataFrame({
            "x": [1.0, 2.0],
            "root_seed_id": ["r0", "r1"],
        })
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(CONFIG, {
            "PHASE1_CHECKPOINT_ENABLED": True,
            "PHASE1_CHECKPOINT_DIR": temp_dir,
        }):
            current = _phase1_checkpoint_identity(frame)
            stored = json.loads(json.dumps(current))
            stored["file_sha256"]["generate_dataset.py"] = "a" * 64
            stored["signature"] = "b" * 64
            stored_dir = os.path.join(temp_dir, stored["signature"])
            os.makedirs(os.path.join(stored_dir, "parts"))
            with open(
                    os.path.join(stored_dir, "manifest.json"),
                    "w", encoding="utf-8") as stream:
                json.dump({
                    "schema": "stage0_phase1_checkpoint_v1",
                    "identity": stored,
                    "complete": True,
                    "completed_rows": len(frame),
                }, stream)
            migration = [{
                "from_generator_sha256": "a" * 64,
                "to_generator_sha256": current[
                    "file_sha256"]["generate_dataset.py"],
                "reason": "test post-processing-only migration",
            }]
            with patch(
                "loitering_munition_damage_twin.stage0.generation."
                "_load_phase1_checkpoint_compatibility",
                return_value=migration,
            ):
                recovered_dir = _prepare_phase1_checkpoint(current)

            changed = json.loads(json.dumps(current))
            changed["input_sha256"] = "c" * 64
            changed["signature"] = "d" * 64
            with patch(
                "loitering_munition_damage_twin.stage0.generation."
                "_load_phase1_checkpoint_compatibility",
                return_value=migration,
            ):
                rejected_dir = _prepare_phase1_checkpoint(changed)

        self.assertEqual(recovered_dir, stored_dir)
        self.assertNotEqual(rejected_dir, stored_dir)

    def test_phase1_simulation_resumes_only_missing_task_indices(self):
        class ImmediateExecutor:
            submitted = []

            def __init__(
                    self, max_workers, initializer=None, initargs=()):
                self.max_workers = max_workers
                if initializer is not None:
                    initializer(*initargs)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def submit(self, function, task):
                self.submitted.append(int(task[0]))
                future = Future()
                try:
                    future.set_result(function(task))
                except BaseException as exc:  # pragma: no cover - safety
                    future.set_exception(exc)
                return future

        frame = pd.DataFrame({
            "input": [10, 11, 12, 13],
        })
        frame.attrs["simulation_checkpoint_namespace"] = "phase1"
        sampler = object.__new__(PhysicsAwareSampler)
        sampler.components = []
        sampler.plates = []

        def fake_worker(task):
            return {
                "sample_id": f"sample-{int(task[0])}",
                "value": int(task[0]),
            }

        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(CONFIG, {
            "PHASE1_CHECKPOINT_ENABLED": True,
            "PHASE1_CHECKPOINT_DIR": temp_dir,
            "PHASE1_CHECKPOINT_INTERVAL": 1,
        }), patch(
            "loitering_munition_damage_twin.stage0.generation."
            "ProcessPoolExecutor", ImmediateExecutor,
        ), patch(
            "loitering_munition_damage_twin.stage0.generation."
            "_process_single_encounter", fake_worker,
        ):
            identity = _phase1_checkpoint_identity(frame)
            checkpoint_dir = _prepare_phase1_checkpoint(identity)
            _write_simulation_checkpoint_part(checkpoint_dir, [
                (0, fake_worker((0, {}))),
                (2, fake_worker((2, {}))),
            ])
            result = sampler.run_simulation_batch(frame)
            with open(
                    os.path.join(checkpoint_dir, "manifest.json"),
                    "r", encoding="utf-8") as stream:
                manifest = json.load(stream)

        self.assertEqual(ImmediateExecutor.submitted, [1, 3])
        self.assertEqual(result["value"].tolist(), [0, 1, 2, 3])
        self.assertTrue(manifest["complete"])
        self.assertEqual(manifest["completed_rows"], 4)

    def test_small_controlled_pipeline_meets_exact_quota_and_full_gate(self):
        overrides = {
            "USABILITY_GATE_MIN_ROWS": 0,
            "MIN_TRAIN_POSITIVE_ROWS": 2,
            "MIN_TRAIN_POSITIVE_ROOTS": 2,
            "MIN_TRAIN_NEGATIVE_ROOTS": 2,
            "MIN_TRAIN_LEVEL_ROOTS": 2,
            "MIN_TRAIN_EXACT_LEVEL_ROWS": 2,
            "MIN_EFFECTIVE_POSITIVE_ROOTS": 1.5,
            "MAX_DOMINANT_ROOT_SHARE": 0.75,
            "MIN_BOUNDARY_SEED_ROOTS": 4,
            "TARGET_STRICT_POSITIVE_ROOTS": 4,
            "FRESH_ROOT_BATCH_SIZE": 8,
            "FRESH_ROOT_MAX_CANDIDATES_PER_TASK": 16,
            "FRESH_ROOT_MAX_ROUNDS": 2,
            "CRAWL_N_STAGES": 1,
            "CRAWL_TOPK_PER_STAGE": 4,
            "MAX_CHILDREN_PER_ROOT_PER_STAGE": 2,
            "MAX_POSITIVE_ROWS_PER_ROOT_PER_CELL": 2,
            "FINAL_TOPUP_MIN_BATCH": 8,
            # The synthetic simulator intentionally emits uniform random
            # levels; its K2 share is not a physical distribution test.
            "K2_FINAL_MAX_RATIO": 0.99,
        }
        with patch.dict(CONFIG, overrides), patch.object(
            PhysicsAwareSampler,
            "run_simulation_batch",
            new=self._fake_simulation_batch,
        ):
            np.random.seed(CONFIG["RANDOM_SEED"])
            final = build_dataset_pipeline(target_total=160, phase1_ratio=0.5)
        self.assertEqual(len(final), 160)
        self.assertEqual(
            final["munition_id"].value_counts().sort_index().to_dict(),
            {0: 40, 1: 40, 2: 40, 3: 40},
        )
        self.assertTrue(final.attrs["generation_profile"]["usability_gate"]["passed"])
        med_lm_c2 = final.attrs["generation_profile"][
            "positive_family_diversity_train"]["1"]["C"]["2"]
        self.assertTrue(med_lm_c2["applicable"])
        self.assertGreaterEqual(
            med_lm_c2["positive"]["rows"],
            overrides["MIN_TRAIN_POSITIVE_ROWS"],
        )
        med_lm_train = final[
            (final["munition_id"] == 1) & (final["split_role"] == "train")]
        self.assertGreaterEqual(
            int((med_lm_train["C_level"] == 2).sum()),
            overrides["MIN_TRAIN_LEVEL_ROOTS"],
        )
        self.assertGreaterEqual(
            final.attrs["weight_train_ess_ratio"],
            CONFIG["MIN_WEIGHT_ESS_RATIO"] +
            CONFIG["WEIGHT_ESS_TARGET_MARGIN"],
        )
        k2_contract = final.attrs["generation_profile"]["k2_ratio_contract"]
        self.assertLessEqual(
            k2_contract["observed_final_ratio"],
            k2_contract["final_max_ratio"],
        )

    def test_atomic_dataset_writer_records_hash_and_rejects_cross_split_family(self):
        frame = pd.DataFrame({
            "sample_id": ["a", "b", "c"],
            "root_seed_id": ["a", "b", "c"],
            "parent_id": ["", "", ""],
            "crawl_stage": [0, 0, 0],
            "split_role": ["train", "val", "test"],
            "frame_version": [FRAME_CONVENTION_VERSION] * 3,
            "dataset_schema": [CONFIG["DATASET_SCHEMA"]] * 3,
            "label_mc_replicates": [1] * 3,
            "label_mc_min_replicates": [1] * 3,
            "label_mc_max_replicates": [1] * 3,
            "x_cm": [0.0] * 3, "y_cm": [0.0] * 3, "z_cm": [100.0] * 3,
            "vx_ms": [0.0] * 3, "vy_ms": [0.0] * 3, "vz_ms": [-100.0] * 3,
            "sin_yaw": [0.0] * 3, "cos_yaw": [1.0] * 3,
            "sin_pitch": [-1.0] * 3, "cos_pitch": [0.0] * 3,
            "sin_roll": [0.0] * 3, "cos_roll": [1.0] * 3,
            "norm_velocity": [100.0] * 3, "munition_id": [0, 1, 2],
            "loss_weight": [1.0] * 3,
            "aoa_accept_prob": [1.0] * 3,
            "aoa_ipw": [1.0] * 3,
            "physics_weight": [1.0] * 3,
            "active_sampling_weight": [1.0] * 3,
            "family_weight": [1.0] * 3,
            "class_balance_weight": [1.0] * 3,
        })
        for task in "KMFC":
            frame[f"{task}_ge1_prob"] = 0.4
            frame[f"{task}_ge2_prob"] = 0.2
            frame[f"{task}_ge1_prob_std"] = 0.0
            frame[f"{task}_ge2_prob_std"] = 0.0
            for mechanism in ("fragment", "shock"):
                frame[f"{mechanism}_{task}_ge1_prob"] = 0.2
                frame[f"{mechanism}_{task}_ge2_prob"] = 0.1
        frame = pd.concat([
            frame,
            pd.DataFrame({
                column: np.full(
                    len(frame), 0.1, dtype=np.float32)
                for column in COMPONENT_TARGET_COLUMNS
            }, index=frame.index),
        ], axis=1)
        frame.attrs["generation_profile"] = {
            "profile_schema": CONFIG["GENERATION_PROFILE_SCHEMA"],
            "dataset_schema": CONFIG["DATASET_SCHEMA"],
            "frame_convention": FRAME_CONVENTION_VERSION,
            "phase2_mode": "per_munition_topoff",
            "label_mc": {"replicate_histogram": {"1": 3}},
            "family_distribution": {
                "maximum_rows_per_root_configured": CONFIG["MAX_ROWS_PER_ROOT"],
            },
            "weighting": {
                "minimum_effective_sample_size_ratio":
                    CONFIG["MIN_WEIGHT_ESS_RATIO"],
            },
            "k2_ratio_contract": {
                "enforced": False,
                "phase2_stop_ratio": CONFIG["K2_PHASE2_STOP_RATIO"],
                "final_max_ratio": CONFIG["K2_FINAL_MAX_RATIO"],
                "observed_positive_rows": 0,
                "observed_final_ratio": 0.0,
            },
            "usability_gate": {"enforced": False, "passed": False, "failures": []},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = os.path.join(temp_dir, "tiny.parquet")
            profile_path = _write_dataset_with_profile(frame, dataset_path)
            self.assertTrue(os.path.exists(dataset_path))
            with open(profile_path, "r", encoding="utf-8") as handle:
                profile = json.load(handle)
            self.assertEqual(profile["artifact"]["rows"], 3)
            self.assertEqual(len(profile["artifact"]["sha256"]), 64)
            self.assertTrue(os.path.exists(os.path.join(
                temp_dir, COMPONENT_SUPERVISION_FILENAME)))
            self.assertTrue(os.path.exists(os.path.join(
                temp_dir,
                COMPONENT_SUPERVISION_PROFILE_FILENAME)))
            self.assertEqual(
                profile["component_supervision"]["schema"],
                "stage0_component_supervision_v1")
            report = validate_stage0_dataset(dataset_path)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["cross_split_root_families"], 0)

            invalid = frame.copy()
            invalid.loc[1, "root_seed_id"] = "a"
            with self.assertRaisesRegex(RuntimeError, "root_seed_id"):
                _write_dataset_with_profile(invalid, os.path.join(temp_dir, "invalid.parquet"))

            rejected = frame.copy()
            rejected.attrs["generation_profile"] = dict(frame.attrs["generation_profile"])
            rejected.attrs["generation_profile"]["usability_gate"] = {
                "enforced": True,
                "passed": False,
                "failures": ["synthetic diversity failure"],
            }
            rejected_path = os.path.join(temp_dir, "rejected.parquet")
            with self.assertRaisesRegex(RuntimeError, "完整诊断"):
                _write_dataset_with_profile(rejected, rejected_path)
            self.assertFalse(os.path.exists(rejected_path))
            rejected_profile_path = os.path.join(
                temp_dir, "generation_profile.rejected.json")
            self.assertTrue(os.path.exists(rejected_profile_path))
            with open(rejected_profile_path, "r", encoding="utf-8") as handle:
                rejected_profile = json.load(handle)
            self.assertEqual(
                rejected_profile["rejection"]["status"],
                "REJECTED_NOT_FOR_TRAINING",
            )


class HighMcRelabelContractTests(unittest.TestCase):
    def test_shards_cover_rows_exactly_once(self):
        self.assertEqual(
            list(_shard_bounds(10, 4)),
            [(0, 0, 4), (1, 4, 8), (2, 8, 10)],
        )

    def test_prepare_records_forces_current_adaptive_bounds(self):
        source = pd.DataFrame({
            "sample_id": ["a", "b"],
            "label_mc_min_replicates": [3, 3],
            "label_mc_max_replicates": [9, 9],
        })
        tasks = _prepare_records(source, 17)
        self.assertEqual([task[0] for task in tasks], [17, 18])
        self.assertTrue(all(
            task[1]["label_mc_min_replicates"]
            == CONFIG["LABEL_MC_MIN_REPLICATES"]
            for task in tasks
        ))
        self.assertTrue(all(
            task[1]["label_mc_max_replicates"]
            == CONFIG["LABEL_MC_MAX_REPLICATES"]
            for task in tasks
        ))

    def test_manifest_is_bound_to_source_and_mc_configuration(self):
        source = Path("source.parquet").resolve()
        output = Path("output.parquet").resolve()
        manifest = _new_manifest(
            source, "a" * 64, 10, 4, output)
        self.assertEqual(manifest["schema"], MANIFEST_SCHEMA)
        self.assertEqual(
            manifest["configuration_sha256"],
            _configuration_sha256(_configuration_snapshot()),
        )
        _validate_manifest(
            manifest, source, "a" * 64, 10, 4, output)
        with self.assertRaisesRegex(RuntimeError, "source sha256"):
            _validate_manifest(
                manifest, source, "b" * 64, 10, 4, output)

    def test_replay_frame_contract_checks_order_and_mc_bounds(self):
        expected = pd.Series(["a", "b"])
        payload = {
            column: np.zeros(2, dtype=np.float32)
            for column in REQUIRED_REPLAY_COLUMNS
        }
        payload["sample_id"] = ["a", "b"]
        payload["label_mc_replicates"] = [
            CONFIG["LABEL_MC_MIN_REPLICATES"],
            CONFIG["LABEL_MC_MAX_REPLICATES"],
        ]
        payload["label_mc_min_replicates"] = [
            CONFIG["LABEL_MC_MIN_REPLICATES"],
        ] * 2
        payload["label_mc_max_replicates"] = [
            CONFIG["LABEL_MC_MAX_REPLICATES"],
        ] * 2
        replay = pd.DataFrame(payload)
        _validate_replay_frame(replay, expected)
        reversed_replay = replay.iloc[::-1].reset_index(drop=True)
        with self.assertRaisesRegex(RuntimeError, "sample_id order"):
            _validate_replay_frame(reversed_replay, expected)


class EvaluationSupportRebalanceTests(unittest.TestCase):
    def test_total_level2_supply_includes_split_and_root_move_buffers(self):
        applicability = {
            munition_id: {
                task: [False, False]
                for task in ("K", "M", "F", "C")
            }
            for munition_id in range(4)
        }
        applicability[1]["C"] = [False, True]
        rows = [{
            "root_seed_id": f"c2-{index}",
            "munition_id": 1,
            "C_level": 2,
            "K_level": 0,
            "M_level": 0,
            "F_level": 0,
        } for index in range(11)]
        frame = pd.DataFrame(rows)
        with patch.dict(CONFIG, {
            "MIN_TRAIN_EXACT_LEVEL_ROWS": 2,
            "MIN_EVAL_EXACT_LEVEL_ROWS": 3,
            "MIN_TRAIN_LEVEL_ROOTS": 2,
            "MIN_EVAL_EXACT_LEVEL_ROOTS": 1,
            "MAX_POSITIVE_ROWS_PER_ROOT_PER_CELL": 2,
            "ORDINAL_APPLICABILITY": applicability,
        }):
            minimum = _minimum_total_exact_level_support()
            deficits = _level2_total_support_deficits(frame)
        self.assertEqual(minimum, (12, 4))
        self.assertEqual(len(deficits), 1)
        self.assertEqual(deficits[0]["row_deficit"], 1)
        self.assertEqual(deficits[0]["root_deficit"], 0)

    def test_level2_topoff_uses_c2_scale_and_retry_budget(self):
        with patch.dict(CONFIG, {
            "FINAL_TOPUP_MIN_BATCH": 256,
            "FRESH_ROOT_BATCH_SIZE": 1024,
            "FINAL_TOPUP_MAX_ROUNDS": 10,
            "C2_FRESH_ROOT_MAX_ROUNDS": 32,
            "MAX_POSITIVE_ROWS_PER_ROOT_PER_CELL": 8,
        }):
            request_rows, maximum_rounds = (
                _level2_support_topoff_budget(7, 0))
            scaled_rows, _ = _level2_support_topoff_budget(200, 0)
        self.assertEqual(request_rows, 1024)
        self.assertEqual(scaled_rows, 1600)
        self.assertEqual(maximum_rounds, 32)

    def test_exhausted_cell_does_not_block_later_solvable_cell(self):
        rows = []
        for munition_id in range(4):
            for split_role in ("train", "val", "test"):
                rows.append({
                    "root_seed_id": f"base-{munition_id}-{split_role}",
                    "split_role": split_role,
                    "munition_id": munition_id,
                    "K_level": 0,
                    "M_level": 0,
                    "F_level": 0,
                    "C_level": 0,
                })
        rows.append({
            "root_seed_id": "impossible-med-lm",
            "split_role": "train",
            "munition_id": 1,
            "K_level": 0,
            "M_level": 0,
            "F_level": 0,
            "C_level": 2,
        })
        rows.extend([{
            "root_seed_id": f"solvable-med-rd-{index}",
            "split_role": "train",
            "munition_id": 2,
            "K_level": 0,
            "M_level": 0,
            "F_level": 0,
            "C_level": 2,
        } for index in range(3)])
        applicability = {
            munition_id: {
                task: [False, False]
                for task in ("K", "M", "F", "C")
            }
            for munition_id in range(4)
        }
        applicability[1]["C"] = [False, True]
        applicability[2]["C"] = [False, True]
        with patch.dict(CONFIG, {
            "EVALUATION_SUPPORT_GATE_MIN_ROWS": 0,
            "MIN_EVAL_EXACT_LEVEL_ROWS": 1,
            "MIN_EVAL_EXACT_LEVEL_ROOTS": 1,
            "MIN_TRAIN_EXACT_LEVEL_ROWS": 1,
            "MIN_TRAIN_LEVEL_ROOTS": 1,
            "ORDINAL_APPLICABILITY": applicability,
        }):
            balanced, report = _rebalance_evaluation_level_support(
                pd.DataFrame(rows))
        self.assertFalse(report["passed"])
        for split_role in ("val", "test"):
            med_rd = balanced[
                (balanced["munition_id"] == 2)
                & (balanced["split_role"] == split_role)
                & (balanced["C_level"] == 2)
            ]
            self.assertEqual(len(med_rd), 1)

    def test_rebalance_moves_whole_roots_to_both_evaluation_splits(self):
        rows = []
        for munition_id in range(4):
            for split_role in ("train", "val", "test"):
                rows.append({
                    "root_seed_id":
                        f"base-{munition_id}-{split_role}",
                    "split_role": split_role,
                    "munition_id": munition_id,
                    "K_level": 0,
                    "M_level": 0,
                    "F_level": 0,
                    "C_level": 0,
                })
        rows.extend([
            {
                "root_seed_id": f"c2-{index}",
                "split_role": "train",
                "munition_id": 1,
                "K_level": 0,
                "M_level": 0,
                "F_level": 0,
                "C_level": 2,
            }
            for index in range(4)
        ])
        frame = pd.DataFrame(rows)
        applicability = {
            munition_id: {
                task: [False, False]
                for task in ("K", "M", "F", "C")
            }
            for munition_id in range(4)
        }
        applicability[1]["C"] = [False, True]
        with patch.dict(CONFIG, {
            "EVALUATION_SUPPORT_GATE_MIN_ROWS": 0,
            "MIN_EVAL_EXACT_LEVEL_ROWS": 1,
            "MIN_EVAL_EXACT_LEVEL_ROOTS": 1,
            "MIN_TRAIN_EXACT_LEVEL_ROWS": 1,
            "MIN_TRAIN_LEVEL_ROOTS": 1,
            "ORDINAL_APPLICABILITY": applicability,
        }):
            balanced, report = (
                _rebalance_evaluation_level_support(frame))
        self.assertTrue(report["passed"])
        for split_role in ("val", "test"):
            cell = balanced[
                (balanced["munition_id"] == 1)
                & (balanced["split_role"] == split_role)
                & (balanced["C_level"] == 2)
            ]
            self.assertGreaterEqual(len(cell), 1)
        self.assertEqual(
            int(balanced.groupby(
                "root_seed_id")["split_role"].nunique().max()),
            1,
        )

    def test_validator_recomputes_production_evidence_from_rows(self):
        rows = []
        split_sizes = {
            "train": 128,
            "val": 100,
            "test": 100,
        }
        for munition_id in range(4):
            for split_role, count in split_sizes.items():
                for index in range(count):
                    rows.append({
                        "root_seed_id":
                            f"{munition_id}-{split_role}-{index}",
                        "split_role": split_role,
                        "munition_id": munition_id,
                        **{
                            f"{task}_ge1_prob": 0.0
                            for task in "KMFC"
                        },
                        **{
                            f"{task}_ge2_prob": 0.0
                            for task in "KMFC"
                        },
                    })
        frame = pd.DataFrame(rows)
        applicability = {
            str(munition_id): {
                task: [False, False]
                for task in "KMFC"
            }
            for munition_id in range(4)
        }
        reported_cells = {}
        for munition_id in range(4):
            reported_cells[str(munition_id)] = {}
            for task in "KMFC":
                reported_cells[str(munition_id)][task] = {
                    "0": {
                        split_role: {
                            "rows": count,
                            "root_families": count,
                        }
                        for split_role, count in (
                            ("val", 100), ("test", 100)
                        )
                    }
                }
        profile = {
            "usability_gate": {
                "enforced": True,
                "passed": True,
            },
            "ordinal_applicability": applicability,
            "training_exact_level_support": {
                "minimum_rows": 128,
                "minimum_root_families": 16,
            },
            "evaluation_exact_level_support": {
                "enforced": True,
                "minimum_rows": 100,
                "minimum_root_families": 16,
                "cells": reported_cells,
            },
        }
        report = _validate_exact_level_evidence(
            frame, profile)
        self.assertTrue(report["passed"])
        self.assertEqual(report["checked_cells"], 32)

        profile["evaluation_exact_level_support"][
            "cells"]["1"]["C"]["0"]["val"]["rows"] = 99
        with self.assertRaisesRegex(
                Stage0ValidationError, "证据不一致"):
            _validate_exact_level_evidence(frame, profile)


if __name__ == "__main__":
    unittest.main()
