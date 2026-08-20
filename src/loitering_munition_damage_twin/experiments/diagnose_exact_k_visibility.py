from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as arrow_dataset
from scipy.special import ndtr
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, roc_curve

REPO_ROOT = Path(__file__).resolve().parents[3]

from loitering_munition_damage_twin.surrogate.artifacts import sha256_file
from loitering_munition_damage_twin.surrogate.dataset import FEATURE_COLUMNS
from loitering_munition_damage_twin.surrogate.features import (
    TERMINAL_PHYSICS_FEATURE_COLUMNS,
    _body_axes_target,
    _component_radius_cm,
    _munition_physics_contract,
    augment_terminal_physics_features,
)
from loitering_munition_damage_twin.simulation.engine import (
    FragmentRetardation,
    INTERNAL_IDS,
    ThorPenetrationModel,
    create_small_loitering_munition,
    load_armor_plates,
    load_vehicle_model,
    parse_component_geometry,
    ray_geometry,
)


K_COMPONENT_IDS = (3, 46)


def _recall_at_fpr(
        target: np.ndarray,
        score: np.ndarray,
        maximum_fpr: float,
) -> dict[str, float]:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        target, score)
    valid = np.flatnonzero(false_positive_rate <= maximum_fpr + 1e-12)
    selected = int(valid[np.argmax(true_positive_rate[valid])])
    return {
        "recall": float(true_positive_rate[selected]),
        "false_positive_rate": float(false_positive_rate[selected]),
        "threshold": float(thresholds[selected]),
    }


def _support_radius(geometry, shape: str, direction: np.ndarray) -> np.ndarray:
    """Return the component support radius along target-frame directions."""
    if shape == "box":
        local = direction @ geometry.rotation
        return np.sum(
            np.abs(local) * geometry.half_extents[None, :], axis=1)
    if shape == "cylinder":
        axial = np.abs(direction @ geometry.axis)
        radial = np.sqrt(np.maximum(1.0 - axial ** 2, 0.0))
        return geometry.half_height * axial + geometry.radius * radial
    half = 0.5 * (geometry.aabb_max - geometry.aabb_min)
    return np.sum(np.abs(direction) * half[None, :], axis=1)


def _circle_overlap_fraction(
        target_radius: np.ndarray,
        blocker_radius: np.ndarray,
        separation: np.ndarray) -> np.ndarray:
    """Fraction of a projected target disk covered by a blocker disk."""
    target_radius = np.maximum(
        np.asarray(target_radius, dtype=np.float64), 1e-10)
    blocker_radius = np.maximum(
        np.asarray(blocker_radius, dtype=np.float64), 1e-10)
    separation = np.maximum(
        np.asarray(separation, dtype=np.float64), 0.0)
    output = np.zeros_like(separation)

    target_inside = (
        (blocker_radius >= target_radius)
        & (separation <= blocker_radius - target_radius)
    )
    blocker_inside = (
        (target_radius > blocker_radius)
        & (separation <= target_radius - blocker_radius)
    )
    output[target_inside] = 1.0
    output[blocker_inside] = np.square(
        blocker_radius[blocker_inside]
        / target_radius[blocker_inside]
    )

    partial = (
        ~target_inside
        & ~blocker_inside
        & (separation < target_radius + blocker_radius)
    )
    if np.any(partial):
        r_target = target_radius[partial]
        r_blocker = blocker_radius[partial]
        distance = np.maximum(separation[partial], 1e-10)
        target_angle = np.arccos(np.clip(
            (
                np.square(distance)
                + np.square(r_target)
                - np.square(r_blocker)
            ) / (2.0 * distance * r_target),
            -1.0,
            1.0,
        ))
        blocker_angle = np.arccos(np.clip(
            (
                np.square(distance)
                + np.square(r_blocker)
                - np.square(r_target)
            ) / (2.0 * distance * r_blocker),
            -1.0,
            1.0,
        ))
        radical = np.maximum(
            (
                -distance + r_target + r_blocker
            ) * (
                distance + r_target - r_blocker
            ) * (
                distance - r_target + r_blocker
            ) * (
                distance + r_target + r_blocker
            ),
            0.0,
        )
        intersection = (
            np.square(r_target) * target_angle
            + np.square(r_blocker) * blocker_angle
            - 0.5 * np.sqrt(radical)
        )
        output[partial] = (
            intersection / (
                np.pi * np.square(r_target)
            )
        )
    return np.clip(output, 0.0, 1.0)


def _expected_nominal_hits(
        required_polar: np.ndarray,
        required_azimuth: np.ndarray,
        polar_half_width: np.ndarray,
        azimuth_half_width: np.ndarray,
        ring_angles: np.ndarray,
        fragments_per_ring: int,
        sigma: float,
) -> np.ndarray:
    expected = np.zeros_like(required_polar, dtype=np.float64)
    nominal_azimuths = (
        2.0 * np.pi
        * np.arange(fragments_per_ring, dtype=np.float64)
        / fragments_per_ring
    )
    for ring_angle in ring_angles:
        polar_delta = required_polar - ring_angle
        polar_probability = (
            ndtr((polar_delta + polar_half_width) / sigma)
            - ndtr((polar_delta - polar_half_width) / sigma)
        )
        for nominal_azimuth in nominal_azimuths:
            azimuth_delta = np.remainder(
                required_azimuth - nominal_azimuth + np.pi,
                2.0 * np.pi,
            ) - np.pi
            azimuth_probability = (
                ndtr((azimuth_delta + azimuth_half_width) / sigma)
                - ndtr((azimuth_delta - azimuth_half_width) / sigma)
            )
            expected += polar_probability * azimuth_probability
    return expected


def _centerline_visibility(
        origins: np.ndarray,
        target_component: dict,
        parsed_components: list[tuple],
        armor_thickness_by_name: dict[str, float],
) -> dict[str, np.ndarray]:
    """Trace the exact centre ray using the same geometry order as the engine.

    This is deterministic deployable geometry.  It does not consume a hit,
    penetration, damage, sampling or label column from the dataset.
    """
    count = len(origins)
    target_id = int(target_component["id"])
    target_position = target_component["geometry"]["position"]
    target_center = np.asarray([
        float(target_position.get(axis) or 0.0)
        for axis in ("x", "y", "z")
    ], dtype=np.float64)
    target_shape, target_geometry = parse_component_geometry(target_component)
    visible = np.zeros(count, dtype=np.float64)
    armor_mm = np.zeros(count, dtype=np.float64)
    armor_count = np.zeros(count, dtype=np.float64)
    blocker_gap_cm = np.zeros(count, dtype=np.float64)
    incidence_cosine = np.zeros(count, dtype=np.float64)
    hit_distance_cm = np.zeros(count, dtype=np.float64)

    for row_index, origin in enumerate(origins):
        ray = target_center - origin
        centre_distance = float(np.linalg.norm(ray))
        if centre_distance < 1e-9:
            continue
        direction = ray / centre_distance
        hits = []
        for cid, name, shape, geometry, component in parsed_components:
            hit, distance, normal = ray_geometry(
                origin, direction, shape, geometry)
            if hit and 1e-8 < distance < centre_distance + 1e-6:
                hits.append((
                    float(distance),
                    int(cid),
                    name,
                    normal,
                    component,
                    "装甲" in name,
                ))
        hits.sort(key=lambda value: value[0])

        traversed = 0.0
        traversed_count = 0
        first_nonarmor_distance = None
        first_nonarmor_id = None
        target_hit = None
        for hit in hits:
            distance, cid, name, normal, component, is_armor = hit
            if is_armor:
                default = (
                    component.get("material", {}).get(
                        "equivalent_thickness", 30.0)
                    or 30.0
                )
                traversed += float(
                    armor_thickness_by_name.get(name, default))
                traversed_count += 1
                continue
            if first_nonarmor_id is None:
                first_nonarmor_id = cid
                first_nonarmor_distance = distance
            if cid == target_id:
                target_hit = hit
                break

        if target_hit is None:
            # A ray through the component centre must intersect supported
            # geometry.  Retain an explicit invalid value for diagnostics.
            blocker_gap_cm[row_index] = -centre_distance
            continue
        target_distance = float(target_hit[0])
        target_normal = np.asarray(target_hit[3], dtype=np.float64)
        visible[row_index] = float(first_nonarmor_id == target_id)
        armor_mm[row_index] = traversed
        armor_count[row_index] = float(traversed_count)
        hit_distance_cm[row_index] = target_distance
        incidence_cosine[row_index] = float(np.clip(
            abs(np.dot(direction, target_normal)), 0.0, 1.0))
        if first_nonarmor_id == target_id:
            blocker_gap_cm[row_index] = 0.0
        else:
            blocker_gap_cm[row_index] = float(
                first_nonarmor_distance - target_distance)

    return {
        "centerline_visible": visible,
        "centerline_armor_mm": armor_mm,
        "centerline_armor_count": armor_count,
        "centerline_blocker_gap_cm": blocker_gap_cm,
        "centerline_incidence_cosine": incidence_cosine,
        "centerline_hit_distance_cm": hit_distance_cm,
    }


def exact_k_visibility_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add exact K-component visibility and penetration expectation features."""
    if not np.all(frame["munition_id"].to_numpy(dtype=np.int64) == 0):
        raise ValueError("This diagnostic is restricted to Small munition rows.")

    components = load_vehicle_model()
    component_by_id = {
        int(component["id"]): component for component in components
    }
    parsed_components = []
    nonarmor_components = []
    for component in components:
        shape, geometry = parse_component_geometry(component)
        entry = (
            int(component["id"]),
            component["name"],
            shape,
            geometry,
            component,
        )
        parsed_components.append(entry)
        if "装甲" not in component["name"]:
            nonarmor_components.append(entry)

    armor_values: dict[str, list[float]] = {}
    for plate in load_armor_plates():
        armor_values.setdefault(plate.name, []).append(
            float(plate.thickness_mm))
    armor_thickness_by_name = {
        name: float(np.mean(values))
        for name, values in armor_values.items()
    }

    origins = frame[
        ["x_cm", "y_cm", "z_cm"]].to_numpy(dtype=np.float64)
    velocity = frame[
        ["vx_ms", "vy_ms", "vz_ms"]].to_numpy(dtype=np.float64)
    forward, right, down = _body_axes_target(frame)
    velocity_body = np.column_stack((
        np.einsum("ij,ij->i", velocity, forward),
        np.einsum("ij,ij->i", velocity, right),
        np.einsum("ij,ij->i", velocity, down),
    ))

    contract = _munition_physics_contract()
    gurney_speed = float(contract["gurney_speed_mps"][0])
    axial_sign = float(contract["axial_sign"][0])
    ring_angles = np.asarray(
        contract["ring_angles_rad"][0], dtype=np.float64)
    projectile = create_small_loitering_munition()
    bed = projectile.warhead.fragment_bed
    fragments_per_ring = int(bed.fragments_per_ring)
    sigma = np.radians(float(bed.spread_sigma_deg))
    cross_section = FragmentRetardation.cross_section(bed.single_mass_g)
    retardation = FragmentRetardation.alpha(
        bed.single_mass_g, bed.drag_coefficient, cross_section)

    feature_values: dict[str, np.ndarray] = {}
    damaging_probabilities = []
    for component_id in K_COMPONENT_IDS:
        component = component_by_id[component_id]
        shape, geometry = parse_component_geometry(component)
        center = np.asarray([
            float(component["geometry"]["position"].get(axis) or 0.0)
            for axis in ("x", "y", "z")
        ])
        relative = center[None, :] - origins
        distance_cm = np.linalg.norm(relative, axis=1)
        target_direction = (
            relative / np.maximum(distance_cm[:, None], 1e-8))

        direction_body = np.column_stack((
            np.einsum("ij,ij->i", target_direction, forward),
            np.einsum("ij,ij->i", target_direction, right),
            np.einsum("ij,ij->i", target_direction, down),
        ))
        required_velocity = (
            gurney_speed * direction_body - velocity_body)
        required_direction = required_velocity / np.maximum(
            np.linalg.norm(required_velocity, axis=1, keepdims=True),
            1e-8,
        )
        required_polar = np.arccos(np.clip(
            required_direction[:, 0] * axial_sign, -1.0, 1.0))
        required_azimuth = np.arctan2(
            required_direction[:, 2], required_direction[:, 1])

        fragment_axis = forward * axial_sign
        axial_cosine = np.einsum(
            "ij,ij->i", target_direction, fragment_axis)
        radial = (
            target_direction
            - axial_cosine[:, None] * fragment_axis
        )
        radial /= np.maximum(
            np.linalg.norm(radial, axis=1, keepdims=True), 1e-8)
        polar_tangent = (
            -np.sqrt(np.maximum(
                1.0 - axial_cosine ** 2, 0.0))[:, None]
            * fragment_axis
            + axial_cosine[:, None] * radial
        )
        azimuth_tangent = np.cross(fragment_axis, radial)
        azimuth_tangent /= np.maximum(
            np.linalg.norm(
                azimuth_tangent, axis=1, keepdims=True),
            1e-8,
        )
        polar_extent = _support_radius(
            geometry, shape, polar_tangent)
        azimuth_extent = _support_radius(
            geometry, shape, azimuth_tangent)
        polar_half_width = np.arctan2(
            polar_extent, np.maximum(distance_cm, 1e-8))
        azimuth_half_width = (
            np.arctan2(
                azimuth_extent, np.maximum(distance_cm, 1e-8))
            / np.maximum(np.sin(required_polar), 0.05)
        )
        expected_hits = _expected_nominal_hits(
            required_polar,
            required_azimuth,
            polar_half_width,
            azimuth_half_width,
            ring_angles,
            fragments_per_ring,
            sigma,
        )

        visibility = _centerline_visibility(
            origins,
            component,
            parsed_components,
            armor_thickness_by_name,
        )
        incidence_cosine = visibility[
            "centerline_incidence_cosine"]
        incidence_angle = np.arccos(
            np.clip(incidence_cosine, 0.1, 1.0))
        initial_speed = np.linalg.norm(
            gurney_speed * required_direction + velocity_body,
            axis=1,
        )
        arrival_speed = initial_speed * np.exp(
            -retardation
            * visibility["centerline_hit_distance_cm"] / 100.0
        )
        own_thickness = float(
            component.get("material", {}).get(
                "equivalent_thickness", 10.0)
            or 10.0
        )
        traversed_armor = visibility["centerline_armor_mm"].copy()
        if component_id in INTERNAL_IDS:
            traversed_armor = np.where(
                traversed_armor > 0.0, traversed_armor, 30.0)
        total_armor = traversed_armor + own_thickness
        v50 = np.asarray([
            ThorPenetrationModel.v50(
                armor,
                bed.single_mass_g,
                angle,
            )
            for armor, angle in zip(total_armor, incidence_angle)
        ])
        penetration_margin = arrival_speed / np.maximum(v50, 1e-8)
        vulnerable_ratio = float(
            component.get("material", {}).get(
                "vulnerable_area_ratio", 1.0)
            or 1.0
        )
        single_hit_damage = np.where(
            penetration_margin >= 1.0,
            1.0 / (
                1.0 + np.exp(np.clip(
                    -8.0 * (penetration_margin - 1.0),
                    -60.0,
                    60.0,
                ))
            ) * vulnerable_ratio,
            0.0,
        )
        centreline_damaging_intensity = (
            expected_hits
            * visibility["centerline_visible"]
            * single_hit_damage
        )
        centreline_damaging_probability = (
            1.0 - np.exp(-centreline_damaging_intensity))

        prefix = f"exact_k_component_{component_id}_"
        feature_values[prefix + "expected_hits"] = expected_hits
        feature_values[prefix + "hit_probability"] = (
            1.0 - np.exp(-expected_hits))
        feature_values[prefix + "polar_half_width_rad"] = polar_half_width
        feature_values[prefix + "azimuth_half_width_rad"] = (
            azimuth_half_width)
        feature_values[prefix + "arrival_speed_ms"] = arrival_speed
        feature_values[prefix + "penetration_margin"] = penetration_margin
        feature_values[prefix + "single_hit_damage"] = single_hit_damage
        feature_values[prefix + "centreline_damaging_probability"] = (
            centreline_damaging_probability)
        for name, values in visibility.items():
            feature_values[prefix + name] = values

        # A smooth angular-clearance proxy supplements the exact centre ray.
        target_radius = _component_radius_cm(component)
        closer_clearances = []
        closer_coverage_fractions = []
        target_half_angle = np.arctan2(
            target_radius, np.maximum(distance_cm, 1e-8))
        for other_id, _, _, _, other in nonarmor_components:
            if other_id == component_id:
                continue
            other_position = other["geometry"]["position"]
            other_center = np.asarray([
                float(other_position.get(axis) or 0.0)
                for axis in ("x", "y", "z")
            ])
            other_relative = other_center[None, :] - origins
            other_distance = np.linalg.norm(other_relative, axis=1)
            other_direction = (
                other_relative
                / np.maximum(other_distance[:, None], 1e-8)
            )
            separation = np.arccos(np.clip(np.einsum(
                "ij,ij->i", target_direction, other_direction
            ), -1.0, 1.0))
            other_radius = _component_radius_cm(other)
            other_half_angle = np.arctan2(
                other_radius, np.maximum(other_distance, 1e-8))
            clearance = (
                separation - other_half_angle - target_half_angle)
            closer = other_distance < distance_cm
            closer_clearances.append(np.where(
                closer,
                clearance,
                np.inf,
            ))
            closer_coverage_fractions.append(np.where(
                closer,
                _circle_overlap_fraction(
                    target_half_angle,
                    other_half_angle,
                    separation,
                ),
                0.0,
            ))
        minimum_clearance = np.min(
            np.stack(closer_clearances, axis=1), axis=1)
        minimum_clearance = np.where(
            np.isfinite(minimum_clearance),
            minimum_clearance,
            np.pi,
        )
        feature_values[
            prefix + "minimum_nonarmor_angular_clearance_rad"
        ] = minimum_clearance
        coverage_matrix = np.stack(
            closer_coverage_fractions, axis=1)
        maximum_coverage = coverage_matrix.max(axis=1)
        summed_coverage = np.clip(
            coverage_matrix.sum(axis=1), 0.0, 1.0)
        # Exact projected-shape unions are expensive.  The product is a
        # deterministic smooth approximation that retains partial exposure
        # when a centre ray is blocked.
        soft_visible_fraction = np.prod(
            1.0 - coverage_matrix, axis=1)
        feature_values[
            prefix + "maximum_projected_blocker_coverage"
        ] = maximum_coverage
        feature_values[
            prefix + "summed_projected_blocker_coverage"
        ] = summed_coverage
        feature_values[
            prefix + "soft_visible_fraction"
        ] = soft_visible_fraction

        penetrative_intensity = (
            expected_hits
            * soft_visible_fraction
            * single_hit_damage
        )
        penetrative_damage_probability = (
            1.0 - np.exp(-penetrative_intensity))
        expected_visible_hits = (
            expected_hits * soft_visible_fraction)
        nonpenetrating_damage_probability = np.clip(
            0.02 * expected_visible_hits * vulnerable_ratio,
            0.0,
            1.0,
        )
        approximate_damage_probability = (
            penetrative_damage_probability
            + (1.0 - penetrative_damage_probability)
            * nonpenetrating_damage_probability
        )
        feature_values[prefix + "expected_visible_hits"] = (
            expected_visible_hits)
        feature_values[
            prefix + "penetrative_damage_probability"
        ] = penetrative_damage_probability
        feature_values[
            prefix + "nonpenetrating_damage_probability"
        ] = nonpenetrating_damage_probability
        feature_values[
            prefix + "approximate_fragment_damage_probability"
        ] = approximate_damage_probability
        damaging_probabilities.append(
            approximate_damage_probability)

    probability_matrix = np.column_stack(damaging_probabilities)
    feature_values["exact_k_ge1_damage_rule_proxy"] = (
        0.5 * probability_matrix.max(axis=1)
        + 0.5 * (
            1.0 - np.prod(1.0 - probability_matrix, axis=1)
        )
    )
    output = pd.DataFrame(feature_values, index=frame.index)
    if not np.isfinite(output.to_numpy(dtype=np.float64)).all():
        raise RuntimeError(
            "Exact K visibility feature derivation produced non-finite values.")
    return output


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only exact visibility/penetration probe for Small/K. "
            "Test rows are excluded at the Parquet scan boundary."))
    parser.add_argument(
        "--data", default="output/damage_dataset.parquet")
    parser.add_argument("--estimators", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--output",
        default=(
            "output/experiments/"
            "exact_k_visibility_diagnostic.json"))
    args = parser.parse_args()

    dataset_path = Path(args.data).resolve()
    required = list(dict.fromkeys(
        list(FEATURE_COLUMNS)
        + ["munition_id", "split_role", "loss_weight", "K_level"]
    ))
    dataset = arrow_dataset.dataset(str(dataset_path), format="parquet")
    frame = dataset.to_table(
        columns=required,
        filter=(
            (arrow_dataset.field("split_role") != "test")
            & (arrow_dataset.field("munition_id") == 0)
        ),
    ).to_pandas()
    train = frame["split_role"].eq("train").to_numpy()
    validation = frame["split_role"].eq("val").to_numpy()
    target = (
        frame["K_level"].to_numpy(dtype=np.int64) >= 1
    ).astype(np.int64)

    started = time.perf_counter()
    frame = augment_terminal_physics_features(frame, copy=False)
    exact_features = exact_k_visibility_features(frame)
    derivation_seconds = time.perf_counter() - started
    frame = pd.concat((frame, exact_features), axis=1)

    feature_sets = {
        "terminal_geometry_v2": (
            list(FEATURE_COLUMNS)
            + list(TERMINAL_PHYSICS_FEATURE_COLUMNS)
        ),
        "terminal_geometry_v2_plus_exact_k_visibility": (
            list(FEATURE_COLUMNS)
            + list(TERMINAL_PHYSICS_FEATURE_COLUMNS)
            + list(exact_features.columns)
        ),
    }
    results = {}
    for name, columns in feature_sets.items():
        model = ExtraTreesClassifier(
            n_estimators=int(args.estimators),
            criterion="entropy",
            max_features=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=int(args.seed),
        )
        weights = np.clip(
            frame.loc[
                train, "loss_weight"
            ].to_numpy(dtype=np.float64),
            0.05,
            20.0,
        )
        model.fit(
            frame.loc[
                train, columns
            ].to_numpy(dtype=np.float32),
            target[train],
            sample_weight=weights,
        )
        score = model.predict_proba(
            frame.loc[
                validation, columns
            ].to_numpy(dtype=np.float32)
        )[:, 1]
        results[name] = {
            "feature_count": int(len(columns)),
            "full_auc": float(
                roc_auc_score(target[validation], score)),
            "standardized_partial_auc_at_fpr_0p005": float(
                roc_auc_score(
                    target[validation],
                    score,
                    max_fpr=0.005,
                )),
            "recall_at_fpr_0p005": _recall_at_fpr(
                target[validation], score, 0.005),
        }

    payload = {
        "schema": "stage0_exact_k_visibility_probe_v1",
        "status": "COMPLETE",
        "split": "validation",
        "test_labels_used": False,
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(str(dataset_path)),
        "train_rows": int(train.sum()),
        "validation_rows": int(validation.sum()),
        "validation_positive_rows": int(
            target[validation].sum()),
        "exact_feature_count": int(len(exact_features.columns)),
        "feature_derivation_seconds": float(derivation_seconds),
        "feature_rows_per_second": float(
            len(frame) / max(derivation_seconds, 1e-9)),
        "results": results,
    }
    _write_json_atomic(Path(args.output).resolve(), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
