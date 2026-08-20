from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as arrow_dataset
from scipy.special import ndtr
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, roc_curve

REPO_ROOT = Path(__file__).resolve().parents[3]

from loitering_munition_damage_twin.surrogate.dataset import FEATURE_COLUMNS
from loitering_munition_damage_twin.surrogate.features import (
    TERMINAL_PHYSICS_FEATURE_COLUMNS,
    _body_axes_target,
    _munition_physics_contract,
    augment_terminal_physics_features,
)
from loitering_munition_damage_twin.simulation.engine import (
    FragmentRetardation,
    ThorPenetrationModel,
    load_vehicle_model,
    parse_component_geometry,
)


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


def _small_k_geometry_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Approximate the expected penetrative-hit probability for the fuel tank.

    This is a validation-only prototype.  It adds the physical terms omitted
    by terminal_geometry_v2: the oriented box footprint, every Taylor ring and
    azimuth ray, fragment retardation, incidence angle and the Thor
    penetration threshold.  No hit, damage, label or sampling column is read.
    """
    components = {
        int(component["id"]): component
        for component in load_vehicle_model()
    }
    fuel = components[3]
    shape, geometry = parse_component_geometry(fuel)
    if shape != "box":
        raise RuntimeError("Fuel component 3 must be an oriented box.")

    position = frame[
        ["x_cm", "y_cm", "z_cm"]].to_numpy(dtype=np.float64)
    velocity = frame[
        ["vx_ms", "vy_ms", "vz_ms"]].to_numpy(dtype=np.float64)
    forward, right, down = _body_axes_target(frame)
    relative = geometry.center[None, :] - position
    distance_cm = np.linalg.norm(relative, axis=1)
    target_direction = relative / np.maximum(distance_cm[:, None], 1e-8)

    contract = _munition_physics_contract()
    # This diagnostic is deliberately limited to the Small munition.
    gurney_speed = float(contract["gurney_speed_mps"][0])
    axial_sign = float(contract["axial_sign"][0])
    ring_angles = np.asarray(
        contract["ring_angles_rad"][0], dtype=np.float64)
    projectile = __import__("sim_engine").create_small_loitering_munition()
    fragment_bed = projectile.warhead.fragment_bed
    fragments_per_ring = int(fragment_bed.fragments_per_ring)
    sigma = np.radians(float(fragment_bed.spread_sigma_deg))

    direction_body = np.column_stack((
        np.einsum("ij,ij->i", target_direction, forward),
        np.einsum("ij,ij->i", target_direction, right),
        np.einsum("ij,ij->i", target_direction, down),
    ))
    velocity_body = np.column_stack((
        np.einsum("ij,ij->i", velocity, forward),
        np.einsum("ij,ij->i", velocity, right),
        np.einsum("ij,ij->i", velocity, down),
    ))
    required_velocity = (
        gurney_speed * direction_body - velocity_body)
    required_direction = required_velocity / np.maximum(
        np.linalg.norm(required_velocity, axis=1, keepdims=True), 1e-8)
    required_polar = np.arccos(np.clip(
        required_direction[:, 0] * axial_sign, -1.0, 1.0))
    required_azimuth = np.arctan2(
        required_direction[:, 2], required_direction[:, 1])

    fragment_axis = forward * axial_sign
    axial_cosine = np.einsum(
        "ij,ij->i", target_direction, fragment_axis)
    radial = target_direction - axial_cosine[:, None] * fragment_axis
    radial /= np.maximum(np.linalg.norm(
        radial, axis=1, keepdims=True), 1e-8)
    polar_tangent = (
        -np.sqrt(np.maximum(1.0 - axial_cosine ** 2, 0.0))[:, None]
        * fragment_axis
        + axial_cosine[:, None] * radial
    )
    azimuth_tangent = np.cross(fragment_axis, radial)
    azimuth_tangent /= np.maximum(np.linalg.norm(
        azimuth_tangent, axis=1, keepdims=True), 1e-8)

    # Support function of an oriented box along the two local tangent axes.
    polar_local = polar_tangent @ geometry.rotation
    azimuth_local = azimuth_tangent @ geometry.rotation
    polar_extent_cm = np.sum(
        np.abs(polar_local) * geometry.half_extents[None, :], axis=1)
    azimuth_extent_cm = np.sum(
        np.abs(azimuth_local) * geometry.half_extents[None, :], axis=1)
    polar_half_width = np.arctan2(
        polar_extent_cm, np.maximum(distance_cm, 1e-8))
    azimuth_arc_half_width = np.arctan2(
        azimuth_extent_cm, np.maximum(distance_cm, 1e-8))
    azimuth_half_width = (
        azimuth_arc_half_width
        / np.maximum(np.sin(required_polar), 0.05)
    )

    expected_hits = np.zeros(len(frame), dtype=np.float64)
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
            expected_hits += polar_probability * azimuth_probability

    # Approximate the centre-line incidence normal using the first face met by
    # a ray passing through the box centre.
    local_direction = target_direction @ geometry.rotation
    face_index = np.argmax(
        np.abs(local_direction)
        / np.maximum(geometry.half_extents[None, :], 1e-8),
        axis=1,
    )
    incidence_cosine = np.take_along_axis(
        np.abs(local_direction), face_index[:, None], axis=1
    ).reshape(-1)
    incidence_angle = np.arccos(np.clip(incidence_cosine, 0.1, 1.0))

    initial_speed = np.linalg.norm(
        gurney_speed * required_direction + velocity_body, axis=1)
    cross_section = FragmentRetardation.cross_section(
        fragment_bed.single_mass_g)
    retardation = FragmentRetardation.alpha(
        fragment_bed.single_mass_g,
        fragment_bed.drag_coefficient,
        cross_section,
    )
    arrival_speed = initial_speed * np.exp(
        -retardation * distance_cm / 100.0)
    # Component 3 is internal.  The engine supplies a 30 mm fallback when no
    # explicit hull was crossed, then adds the component's own thickness.
    own_thickness = float(
        fuel["material"]["equivalent_thickness"])
    armor_thickness = 30.0 + own_thickness
    v50 = np.asarray([
        ThorPenetrationModel.v50(
            armor_thickness,
            fragment_bed.single_mass_g,
            angle,
        )
        for angle in incidence_angle
    ])
    penetration_margin = arrival_speed / np.maximum(v50, 1e-8)
    single_hit_damage = np.where(
        penetration_margin >= 1.0,
        1.0 / (1.0 + np.exp(np.clip(
            -8.0 * (penetration_margin - 1.0), -60.0, 60.0))),
        0.0,
    )
    damaging_hit_intensity = expected_hits * single_hit_damage

    return pd.DataFrame({
        "fuel_expected_fragment_hits": expected_hits,
        "fuel_at_least_one_hit_probability": (
            1.0 - np.exp(-expected_hits)),
        "fuel_incidence_cosine": incidence_cosine,
        "fuel_arrival_speed_ms": arrival_speed,
        "fuel_thor_penetration_margin": penetration_margin,
        "fuel_single_hit_damage_probability": single_hit_damage,
        "fuel_expected_damaging_hit_intensity": damaging_hit_intensity,
        "fuel_at_least_one_damaging_hit_probability": (
            1.0 - np.exp(-damaging_hit_intensity)),
        "fuel_projected_polar_half_width_rad": polar_half_width,
        "fuel_projected_azimuth_half_width_rad": azimuth_half_width,
    }, index=frame.index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="output/damage_dataset.parquet")
    parser.add_argument("--estimators", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=(
            "output/experiments/"
            "small_k_fragment_geometry_diagnostic.json"),
    )
    args = parser.parse_args()

    required = (
        list(FEATURE_COLUMNS)
        + ["munition_id", "split_role", "loss_weight", "K_level"]
    )
    dataset = arrow_dataset.dataset(args.data, format="parquet")
    frame = dataset.to_table(
        columns=required,
        filter=(
            (arrow_dataset.field("split_role") != "test")
            & (arrow_dataset.field("munition_id") == 0)
        ),
    ).to_pandas()
    frame = augment_terminal_physics_features(frame, copy=False)
    geometry = _small_k_geometry_features(frame)
    frame = pd.concat((frame, geometry), axis=1)

    train = frame["split_role"].to_numpy() == "train"
    validation = frame["split_role"].to_numpy() == "val"
    target = (frame["K_level"].to_numpy(dtype=np.int64) >= 1).astype(
        np.int64)
    feature_sets = {
        "terminal_geometry_v2": (
            list(FEATURE_COLUMNS)
            + list(TERMINAL_PHYSICS_FEATURE_COLUMNS)
        ),
        "terminal_geometry_v2_plus_fuel_penetration": (
            list(FEATURE_COLUMNS)
            + list(TERMINAL_PHYSICS_FEATURE_COLUMNS)
            + list(geometry.columns)
        ),
    }
    results = {}
    for name, columns in feature_sets.items():
        model = ExtraTreesClassifier(
            n_estimators=args.estimators,
            criterion="entropy",
            max_features=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=args.seed,
        )
        sample_weight = np.clip(
            frame.loc[train, "loss_weight"].to_numpy(dtype=np.float64),
            0.05,
            20.0,
        )
        model.fit(
            frame.loc[train, columns].to_numpy(dtype=np.float32),
            target[train],
            sample_weight=sample_weight,
        )
        score = model.predict_proba(
            frame.loc[validation, columns].to_numpy(dtype=np.float32)
        )[:, 1]
        results[name] = {
            "feature_count": len(columns),
            "full_auc": float(roc_auc_score(target[validation], score)),
            "standardized_partial_auc_at_fpr_0p005": float(
                roc_auc_score(
                    target[validation], score, max_fpr=0.005)),
            "recall_at_fpr_0p005": _recall_at_fpr(
                target[validation], score, 0.005),
        }

    payload = {
        "schema": "stage0_small_k_fragment_geometry_probe_v1",
        "status": "COMPLETE",
        "split": "validation",
        "test_labels_used": False,
        "train_rows": int(train.sum()),
        "validation_rows": int(validation.sum()),
        "validation_positive_rows": int(target[validation].sum()),
        "results": results,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
