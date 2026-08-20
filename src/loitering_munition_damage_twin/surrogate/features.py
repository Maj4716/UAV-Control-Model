from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from loitering_munition_damage_twin.stage0.component_supervision import (
    CRITICAL_COMPONENT_IDS,
    DAMAGE_RULE_COMPONENT_GROUPS,
)
from loitering_munition_damage_twin.simulation.engine import (
    DetonationPoint,
    FragmentRetardation,
    GurneyModel,
    ShockwaveModel,
    TaylorAngleModel,
    ThorPenetrationModel,
    bundled_resource_path,
    create_heavy_loitering_munition,
    create_medium_loitering_munition,
    create_medium_rear_det,
    create_small_loitering_munition,
)


TERMINAL_PHYSICS_FEATURE_VERSION = "terminal_geometry_v2"

GLOBAL_TERMINAL_PHYSICS_FEATURE_COLUMNS = (
    "phys_position_radius_m",
    "phys_position_horizontal_radius_m",
    "phys_velocity_unit_x",
    "phys_velocity_unit_y",
    "phys_velocity_unit_z",
    "phys_velocity_body_forward_cos",
    "phys_velocity_body_right_cos",
    "phys_velocity_body_down_cos",
    "phys_center_closing_cos",
    "phys_time_to_center_closest_s",
    "phys_center_miss_distance_m",
)

GROUP_TERMINAL_PHYSICS_STATISTICS = (
    "min_distance_m",
    "mean_distance_m",
    "max_shock_damage_proxy",
    "mean_shock_damage_proxy",
    "max_fragment_exposure_proxy",
    "mean_fragment_exposure_proxy",
    "min_fragment_cone_residual_rad",
    "min_fragment_azimuth_residual_rad",
    "max_fragment_grid_alignment",
    "max_fragment_damage_proxy",
    "mean_fragment_damage_proxy",
)

TASK_RULE_PROXY_FEATURE_COLUMNS = [
    f"phys_{mechanism}_{task}_ge{level}_rule_proxy"
    for mechanism in ("fragment", "shock")
    for task in ("K", "M", "F", "C")
    for level in (1, 2)
]

COMBINED_TASK_RULE_PROXY_FEATURE_COLUMNS = [
    f"phys_combined_{task}_ge{level}_rule_proxy"
    for task in ("K", "M", "F", "C")
    for level in (1, 2)
]

COMPONENT_PROXY_FEATURE_COLUMNS = [
    f"phys_component_{component_id:03d}_{mechanism}_damage_proxy"
    for mechanism in ("fragment", "shock")
    for component_id in CRITICAL_COMPONENT_IDS
]

TERMINAL_PHYSICS_FEATURE_COLUMNS = (
    list(
    GLOBAL_TERMINAL_PHYSICS_FEATURE_COLUMNS
) + [
    f"phys_{group}_{statistic}"
    for group in DAMAGE_RULE_COMPONENT_GROUPS
    for statistic in GROUP_TERMINAL_PHYSICS_STATISTICS
] + TASK_RULE_PROXY_FEATURE_COLUMNS
)

REQUIRED_TERMINAL_COLUMNS = (
    "x_cm", "y_cm", "z_cm",
    "vx_ms", "vy_ms", "vz_ms",
    "sin_yaw", "cos_yaw",
    "sin_pitch", "cos_pitch",
    "sin_roll", "cos_roll",
)


def terminal_physics_contract_metadata(
        include_component_proxies: bool = False,
        armor_aware_fragment_proxies: bool = False) -> dict:
    """Return the immutable inputs required to reproduce derived features."""
    if armor_aware_fragment_proxies and not include_component_proxies:
        raise ValueError(
            "armor_aware_fragment_proxies requires component proxies.")
    model_path = Path(bundled_resource_path("vehicle_model.json"))
    digest = hashlib.sha256()
    with model_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    derived_feature_names = list(TERMINAL_PHYSICS_FEATURE_COLUMNS)
    extensions = []
    if include_component_proxies:
        derived_feature_names.extend(COMPONENT_PROXY_FEATURE_COLUMNS)
        extensions.append({
            "name": (
                "component_damage_proxies_v2_armor_aware_fragment"
                if armor_aware_fragment_proxies
                else "component_damage_proxies_v1"
            ),
            "feature_names": list(COMPONENT_PROXY_FEATURE_COLUMNS),
            "feature_count": len(COMPONENT_PROXY_FEATURE_COLUMNS),
            "armor_aware_fragment_proxies": bool(
                armor_aware_fragment_proxies),
        })
    return {
        "derivation_version": TERMINAL_PHYSICS_FEATURE_VERSION,
        "vehicle_model_sha256": digest.hexdigest(),
        "derived_feature_names": derived_feature_names,
        "derived_feature_count": len(derived_feature_names),
        "extensions": extensions,
    }


def _component_radius_cm(component: dict) -> float:
    dimensions = component.get("geometry", {}).get("dimensions", {})
    length_or_radius = float(
        dimensions.get("length_or_radius") or 0.0)
    height = float(dimensions.get("height") or 0.0)
    width = dimensions.get("width")
    if width is None:
        return max(length_or_radius, 0.5 * height, 1.0)
    width = float(width or 0.0)
    return max(
        0.5 * np.sqrt(
            length_or_radius ** 2 + height ** 2 + width ** 2),
        1.0,
    )


@lru_cache(maxsize=1)
def _component_geometry_contract() -> dict:
    model_path = Path(bundled_resource_path("vehicle_model.json"))
    with model_path.open("r", encoding="utf-8") as stream:
        components = json.load(stream)["components"]
    by_id = {int(component["id"]): component for component in components}
    contract = {}
    for group_name, component_ids in DAMAGE_RULE_COMPONENT_GROUPS.items():
        missing = [value for value in component_ids if value not in by_id]
        if missing:
            raise RuntimeError(
                f"Vehicle model is missing {group_name} components: {missing}")
        group_components = [by_id[value] for value in component_ids]
        centers = []
        radii = []
        thresholds = []
        exposure_modes = []
        fallback_shielding = []
        equivalent_thickness = []
        vulnerable_area_ratio = []
        for component in group_components:
            position = component["geometry"]["position"]
            centers.append((
                float(position.get("x") or 0.0),
                float(position.get("y") or 0.0),
                float(position.get("z") or 0.0),
            ))
            radii.append(_component_radius_cm(component))
            threshold = (
                ShockwaveModel.armor_threshold(component)
                or component.get("material", {}).get(
                    "overpressure_threshold")
                or 0.3
            )
            thresholds.append(float(threshold))
            component_id = int(component["id"])
            if component_id in ShockwaveModel.EXPOSED_IDS:
                exposure_modes.append(2)
            elif component_id in ShockwaveModel.SEMI_EXPOSED_IDS:
                exposure_modes.append(1)
            else:
                exposure_modes.append(0)
            fallback_shielding.append(float(
                ShockwaveModel.SHIELDING.get(component_id, 15.0)))
            material = component.get("material", {})
            equivalent_thickness.append(float(
                material.get("equivalent_thickness", 10.0) or 10.0))
            vulnerable_area_ratio.append(float(
                material.get("vulnerable_area_ratio", 1.0) or 1.0))
        contract[group_name] = {
            "component_ids": tuple(
                int(component["id"])
                for component in group_components),
            "centers_cm": np.asarray(centers, dtype=np.float64),
            "radii_cm": np.asarray(radii, dtype=np.float64),
            "thresholds_mpa": np.asarray(
                thresholds, dtype=np.float64),
            "exposure_modes": np.asarray(
                exposure_modes, dtype=np.int64),
            "fallback_shielding_mm": np.asarray(
                fallback_shielding, dtype=np.float64),
            "equivalent_thickness_mm": np.asarray(
                equivalent_thickness, dtype=np.float64),
            "vulnerable_area_ratio": np.asarray(
                vulnerable_area_ratio, dtype=np.float64),
        }
    return contract


@lru_cache(maxsize=1)
def _munition_physics_contract() -> dict:
    projectiles = (
        create_small_loitering_munition(),
        create_medium_loitering_munition(),
        create_medium_rear_det(),
        create_heavy_loitering_munition(),
    )
    tnt_mass = []
    fragment_count = []
    fragments_per_ring = []
    fragment_mass_g = []
    fragment_drag_coefficient = []
    axial_sign = []
    cone_center_rad = []
    cone_sigma_rad = []
    gurney_speed_mps = []
    ring_angles_rad = []
    for projectile in projectiles:
        warhead = projectile.warhead
        bed = warhead.fragment_bed
        gurney_speed = GurneyModel.cylinder_velocity(
            warhead.gurney_energy_mps,
            warhead.metal_to_charge_ratio,
        )
        base_angle = TaylorAngleModel.base_angle_rad(
            warhead.detonation_velocity_mps,
            gurney_speed,
        )
        ring_angles = np.asarray([
            TaylorAngleModel.angle_at_position(
                base_angle, position, warhead.detonation_point)
            for position in np.linspace(
                0.0, 1.0, int(bed.num_rings))
        ])
        tnt_mass.append(float(warhead.tnt_equivalent_mass_kg))
        fragment_count.append(float(bed.total_count))
        fragments_per_ring.append(float(bed.fragments_per_ring))
        fragment_mass_g.append(float(bed.single_mass_g))
        fragment_drag_coefficient.append(float(bed.drag_coefficient))
        axial_sign.append(
            1.0 if warhead.detonation_point == DetonationPoint.REAR
            else -1.0)
        cone_center_rad.append(float(ring_angles.mean()))
        # Combine axial ring variation and random angular spread.  A small
        # lower bound avoids an unrealistically needle-like analytic proxy.
        cone_sigma_rad.append(float(max(
            ring_angles.std(),
            np.radians(float(bed.spread_sigma_deg)),
            np.radians(3.0),
        )))
        gurney_speed_mps.append(float(gurney_speed))
        ring_angles_rad.append(ring_angles.astype(np.float64))
    return {
        "tnt_mass_kg": np.asarray(tnt_mass, dtype=np.float64),
        "fragment_count": np.asarray(
            fragment_count, dtype=np.float64),
        "fragments_per_ring": np.asarray(
            fragments_per_ring, dtype=np.float64),
        "fragment_mass_g": np.asarray(
            fragment_mass_g, dtype=np.float64),
        "fragment_drag_coefficient": np.asarray(
            fragment_drag_coefficient, dtype=np.float64),
        "axial_sign": np.asarray(axial_sign, dtype=np.float64),
        "cone_center_rad": np.asarray(
            cone_center_rad, dtype=np.float64),
        "cone_sigma_rad": np.asarray(
            cone_sigma_rad, dtype=np.float64),
        "gurney_speed_mps": np.asarray(
            gurney_speed_mps, dtype=np.float64),
        "ring_angles_rad": tuple(ring_angles_rad),
    }


def _body_axes_target(frame: pd.DataFrame) -> tuple[np.ndarray, ...]:
    sy = frame["sin_yaw"].to_numpy(dtype=np.float64)
    cy = frame["cos_yaw"].to_numpy(dtype=np.float64)
    sp = frame["sin_pitch"].to_numpy(dtype=np.float64)
    cp = frame["cos_pitch"].to_numpy(dtype=np.float64)
    sr = frame["sin_roll"].to_numpy(dtype=np.float64)
    cr = frame["cos_roll"].to_numpy(dtype=np.float64)

    forward = np.column_stack((sy * cp, cy * cp, sp))
    right = np.column_stack((
        sy * sp * sr + cy * cr,
        cy * sp * sr - sy * cr,
        -cp * sr,
    ))
    down = np.column_stack((
        sy * sp * cr - cy * sr,
        cy * sp * cr + sy * sr,
        -cp * cr,
    ))
    return forward, right, down


def _transmitted_pressure(
        incident: np.ndarray, shielding_mm: np.ndarray) -> np.ndarray:
    beta = np.where(
        incident < 0.5, 0.25,
        np.where(incident < 2.0, 0.18, 0.12),
    )
    return incident * np.exp(-beta * shielding_mm.reshape(1, -1))


def _shock_damage_proxy(
        distance_m: np.ndarray,
        tnt_mass_kg: np.ndarray,
        group_contract: dict) -> np.ndarray:
    scaled_distance = (
        np.maximum(distance_m, 0.01)
        / np.cbrt(tnt_mass_kg).reshape(-1, 1)
    )
    incident = (
        0.084 / scaled_distance
        + 0.27 / np.square(scaled_distance)
        + 0.705 / np.power(scaled_distance, 3)
    )
    modes = group_contract["exposure_modes"].reshape(1, -1)
    reflected = (
        incident
        * (2.0 + np.minimum(incident, 1.0) * 6.0)
        * np.cos(np.radians(45.0))
    )
    semi_exposed = _transmitted_pressure(
        incident, np.full(incident.shape[1], 5.0))
    internal = _transmitted_pressure(
        incident, group_contract["fallback_shielding_mm"])
    effective = np.where(
        modes == 2, reflected,
        np.where(modes == 1, semi_exposed, internal),
    )
    ratio = (
        effective
        / group_contract["thresholds_mpa"].reshape(1, -1)
    )
    logistic_argument = np.clip(6.0 * (ratio - 1.0), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logistic_argument))


def _probabilistic_or(values: np.ndarray, axis: int = 1) -> np.ndarray:
    """Mirror DamageTreeEvaluator._p_or with rho=0.5."""
    return (
        0.5 * np.max(values, axis=axis)
        + 0.5 * (1.0 - np.prod(1.0 - values, axis=axis))
    )


def _probabilistic_ratio(
        values: np.ndarray, ratio_threshold: float) -> np.ndarray:
    """Vectorized counterpart of DamageTreeEvaluator._p_ratio."""
    count = values.shape[1]
    threshold_count = ratio_threshold * count
    mean = values.sum(axis=1)
    variance = (values * (1.0 - values)).sum(axis=1)
    deterministic = variance < 1e-6
    safe_variance = np.maximum(variance, 1e-6)
    z_value = (
        mean - threshold_count + 0.5
    ) / np.sqrt(safe_variance)
    probability = 1.0 / (
        1.0 + np.exp(-np.clip(1.7 * z_value, -60.0, 60.0))
    )
    return np.where(
        deterministic,
        (mean >= threshold_count).astype(np.float64),
        probability,
    )


def _damage_tree_rule_proxies(
        groups: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Propagate component proxies through the authoritative damage tree.

    This is a feature derivation, not a replacement label generator.  It
    mirrors the rule topology while the neural network learns the residual
    error caused by armor occlusion, exact component geometry and stochastic
    fragment spread.
    """
    p_or = _probabilistic_or

    k1 = groups["k_fuel"][:, 0]
    k2 = groups["k_ammo"][:, 0]
    k_ge1 = p_or(np.column_stack((k1, k2)))

    m1 = p_or(np.column_stack((
        _probabilistic_ratio(groups["m_wheels"], 0.2),
        p_or(groups["m_idlers"]),
    )))
    m2 = p_or(np.column_stack((
        p_or(groups["m_power"]),
        p_or(groups["m_drives"]),
        _probabilistic_ratio(groups["m_tracks"], 0.25),
        groups["m_driver"][:, 0],
    )))
    m_ge1 = p_or(np.column_stack((m1, m2)))

    sights = groups["f_sights"]
    fire_control = groups["f_fire_control"]
    command = groups["f_command"]
    f1 = p_or(np.column_stack((
        p_or(groups["f_secondary"]),
        sights[:, 0] * (1.0 - sights[:, 1]),
        (
            fire_control[:, 0] * (1.0 - fire_control[:, 1])
            + fire_control[:, 1] * (1.0 - fire_control[:, 0])
        ),
        command[:, 1] * (1.0 - command[:, 0]),
    )))
    f2 = p_or(np.column_stack((
        np.prod(groups["f_main_supply"], axis=1),
        np.prod(sights, axis=1),
        np.prod(fire_control, axis=1),
        np.prod(command, axis=1),
    )))
    f_ge1 = p_or(np.column_stack((f1, f2)))

    c1 = _probabilistic_ratio(groups["c_crew"], 0.2)
    c2 = _probabilistic_ratio(groups["c_crew"], 0.6)
    return {
        "K_ge1": np.clip(k_ge1, 0.0, 1.0),
        "K_ge2": np.clip(np.minimum(k2, k_ge1), 0.0, 1.0),
        "M_ge1": np.clip(m_ge1, 0.0, 1.0),
        "M_ge2": np.clip(np.minimum(m2, m_ge1), 0.0, 1.0),
        "F_ge1": np.clip(f_ge1, 0.0, 1.0),
        "F_ge2": np.clip(np.minimum(f2, f_ge1), 0.0, 1.0),
        "C_ge1": np.clip(c1, 0.0, 1.0),
        "C_ge2": np.clip(np.minimum(c2, c1), 0.0, 1.0),
    }


def augment_terminal_physics_features(
        frame: pd.DataFrame,
        copy: bool = True,
        include_combined_rule_proxies: bool = False,
        include_component_proxies: bool = False,
        armor_aware_fragment_proxies: bool = False) -> pd.DataFrame:
    """Add deployable analytic geometry features to a terminal-state frame.

    The function is deliberately deterministic and rejects any missing base
    input.  It never reads target aim points, sampling phase, simulated hits,
    penetrations, component damage or labels.
    """
    if armor_aware_fragment_proxies and not include_component_proxies:
        raise ValueError(
            "armor_aware_fragment_proxies requires component proxies.")
    missing = [
        column for column in REQUIRED_TERMINAL_COLUMNS
        if column not in frame.columns
    ]
    munition_column = (
        "munition_id" if "munition_id" in frame.columns
        else "m_id" if "m_id" in frame.columns else None
    )
    if missing or munition_column is None:
        raise ValueError(
            "Terminal physics feature input is incomplete: "
            f"missing={missing}, munition_column={munition_column!r}.")
    output = frame.copy() if copy else frame
    derived: dict[str, np.ndarray] = {}
    munition_id = output[munition_column].to_numpy(dtype=np.int64)
    if np.any((munition_id < 0) | (munition_id > 3)):
        raise ValueError("munition_id must be in [0,3].")

    position_m = output[
        ["x_cm", "y_cm", "z_cm"]].to_numpy(dtype=np.float64) / 100.0
    velocity = output[
        ["vx_ms", "vy_ms", "vz_ms"]].to_numpy(dtype=np.float64)
    speed = np.linalg.norm(velocity, axis=1)
    safe_speed = np.maximum(speed, 1e-8)
    velocity_unit = velocity / safe_speed[:, None]
    position_radius = np.linalg.norm(position_m, axis=1)
    safe_radius = np.maximum(position_radius, 1e-8)
    position_unit = position_m / safe_radius[:, None]
    forward, right, down = _body_axes_target(output)

    derived["phys_position_radius_m"] = position_radius
    derived["phys_position_horizontal_radius_m"] = np.linalg.norm(
        position_m[:, :2], axis=1)
    derived["phys_velocity_unit_x"] = velocity_unit[:, 0]
    derived["phys_velocity_unit_y"] = velocity_unit[:, 1]
    derived["phys_velocity_unit_z"] = velocity_unit[:, 2]
    derived["phys_velocity_body_forward_cos"] = np.einsum(
        "ij,ij->i", velocity_unit, forward)
    derived["phys_velocity_body_right_cos"] = np.einsum(
        "ij,ij->i", velocity_unit, right)
    derived["phys_velocity_body_down_cos"] = np.einsum(
        "ij,ij->i", velocity_unit, down)
    derived["phys_center_closing_cos"] = np.einsum(
        "ij,ij->i", velocity_unit, -position_unit)
    time_to_closest = -np.einsum(
        "ij,ij->i", position_m, velocity) / np.square(safe_speed)
    closest_vector = position_m + time_to_closest[:, None] * velocity
    derived["phys_time_to_center_closest_s"] = time_to_closest
    derived["phys_center_miss_distance_m"] = np.linalg.norm(
        closest_vector, axis=1)

    component_contract = _component_geometry_contract()
    munition_contract = _munition_physics_contract()
    tnt_mass = munition_contract["tnt_mass_kg"][munition_id]
    fragment_count_scale = (
        munition_contract["fragment_count"][munition_id] / 150.0)
    fragment_axis = (
        forward
        * munition_contract["axial_sign"][munition_id, None]
    )
    cone_center = munition_contract["cone_center_rad"][munition_id]
    cone_sigma = munition_contract["cone_sigma_rad"][munition_id]
    gurney_speed = (
        munition_contract["gurney_speed_mps"][munition_id])
    fragments_per_ring = (
        munition_contract["fragments_per_ring"][munition_id])
    fragment_mass_g = (
        munition_contract["fragment_mass_g"][munition_id])
    fragment_drag_coefficient = (
        munition_contract["fragment_drag_coefficient"][munition_id])
    fragment_cross_section_cm2 = np.asarray([
        FragmentRetardation.cross_section(value)
        for value in fragment_mass_g
    ], dtype=np.float64)
    fragment_retardation = (
        FragmentRetardation.RHO
        * fragment_drag_coefficient
        * (fragment_cross_section_cm2 / 1e4)
        / (2.0 * np.maximum(fragment_mass_g / 1000.0, 1e-12))
    )
    position_cm = position_m * 100.0
    velocity_body = np.column_stack((
        np.einsum("ij,ij->i", velocity, forward),
        np.einsum("ij,ij->i", velocity, right),
        np.einsum("ij,ij->i", velocity, down),
    ))
    mechanism_group_proxies = {
        "fragment": {},
        "shock": {},
    }
    component_proxies = {
        "fragment": {},
        "fragment_armor_aware": {},
        "shock": {},
    }

    for group_name, group_contract in component_contract.items():
        relative_cm = (
            group_contract["centers_cm"][None, :, :]
            - position_cm[:, None, :]
        )
        distance_cm = np.linalg.norm(relative_cm, axis=2)
        safe_distance_cm = np.maximum(distance_cm, 1e-6)
        direction = relative_cm / safe_distance_cm[:, :, None]
        axial_cosine = np.einsum(
            "nkj,nj->nk", direction, fragment_axis)
        polar_angle = np.arccos(np.clip(axial_cosine, -1.0, 1.0))
        angular_radius = np.arctan2(
            group_contract["radii_cm"].reshape(1, -1),
            safe_distance_cm,
        )
        effective_sigma = (
            cone_sigma[:, None] + angular_radius)
        angular_delta = np.abs(
            polar_angle - cone_center[:, None])
        cone_alignment = np.exp(
            -0.5 * np.square(
                angular_delta / np.maximum(effective_sigma, 1e-6)))
        solid_angle_proxy = np.square(
            group_contract["radii_cm"].reshape(1, -1)
            / safe_distance_cm)
        fragment_proxy = np.clip(
            fragment_count_scale[:, None]
            * cone_alignment
            * solid_angle_proxy,
            0.0,
            10.0,
        )

        # The continuous cone proxy above loses the phase of a sparse,
        # uniform-ring fragment bed.  That omission is especially damaging
        # for the 60-fragment Small/K case.  Transform the required component
        # ray into body coordinates, compensate approximately for projectile
        # velocity composition, then measure its distance from the nearest
        # nominal Taylor ring and azimuth ray.
        direction_body = np.stack((
            np.einsum("nkj,nj->nk", direction, forward),
            np.einsum("nkj,nj->nk", direction, right),
            np.einsum("nkj,nj->nk", direction, down),
        ), axis=2)
        required_fragment_body = (
            gurney_speed[:, None, None] * direction_body
            - velocity_body[:, None, :]
        )
        required_norm = np.linalg.norm(
            required_fragment_body, axis=2)
        required_direction_body = (
            required_fragment_body
            / np.maximum(required_norm[:, :, None], 1e-8)
        )
        required_axial_cosine = (
            required_direction_body[:, :, 0]
            * munition_contract["axial_sign"][munition_id, None]
        )
        required_polar_angle = np.arccos(
            np.clip(required_axial_cosine, -1.0, 1.0))
        required_azimuth = np.arctan2(
            required_direction_body[:, :, 2],
            required_direction_body[:, :, 1],
        )
        polar_residual = np.full_like(
            required_polar_angle, np.inf)
        for current_munition, ring_angles in enumerate(
                munition_contract["ring_angles_rad"]):
            mask = munition_id == current_munition
            if not np.any(mask):
                continue
            polar_residual[mask] = np.min(
                np.abs(
                    required_polar_angle[mask, :, None]
                    - ring_angles[None, None, :]
                ),
                axis=2,
            )
        azimuth_period = (
            2.0 * np.pi / fragments_per_ring[:, None])
        azimuth_residual = np.abs(
            np.remainder(
                required_azimuth + 0.5 * azimuth_period,
                azimuth_period,
            ) - 0.5 * azimuth_period
        )
        combined_angular_residual = np.sqrt(
            np.square(polar_residual)
            + np.square(
                np.sin(required_polar_angle) * azimuth_residual)
        )
        angular_miss = np.maximum(
            combined_angular_residual - angular_radius, 0.0)
        grid_alignment = np.exp(
            -0.5 * np.square(
                angular_miss
                / np.maximum(cone_sigma[:, None], 1e-6)
            )
        )
        fragment_grid_exposure = np.clip(
            fragment_count_scale[:, None]
            * grid_alignment
            * solid_angle_proxy,
            0.0,
            10.0,
        )
        fragment_damage_proxy = (
            1.0 - np.exp(-fragment_grid_exposure))
        # Match the simulator's missing penetration chain more closely.  The
        # legacy proxy above only estimates whether a nominal ray intersects
        # a component.  It treats a lightly protected wheel and an internal
        # ammunition rack identically, even though arrival velocity, LOS
        # shielding, self armor and vulnerable area determine whether a hit
        # causes damage.  This deployable proxy uses only terminal state,
        # immutable munition constants and public vehicle geometry/materials.
        initial_velocity_body = (
            gurney_speed[:, None, None] * required_direction_body
            + velocity_body[:, None, :]
        )
        initial_speed = np.linalg.norm(initial_velocity_body, axis=2)
        distance_m = distance_cm / 100.0
        arrival_speed = (
            initial_speed
            * np.exp(-fragment_retardation[:, None] * distance_m)
        )
        exposure_mode = group_contract[
            "exposure_modes"].reshape(1, -1)
        shielding = group_contract[
            "fallback_shielding_mm"].reshape(1, -1)
        traversal_armor = np.where(
            exposure_mode == 2,
            0.0,
            np.where(exposure_mode == 1, 0.35 * shielding, shielding),
        )
        total_armor = (
            group_contract["equivalent_thickness_mm"].reshape(1, -1)
            + traversal_armor
        )
        v50 = (
            ThorPenetrationModel.K
            * np.power(np.maximum(total_armor, 1e-6),
                       ThorPenetrationModel.ALPHA)
            * np.power(fragment_mass_g[:, None],
                       ThorPenetrationModel.BETA)
        )
        penetration_margin = arrival_speed / np.maximum(v50, 1e-8)
        penetration_probability = 1.0 / (
            1.0
            + np.exp(np.clip(
                -8.0 * (penetration_margin - 1.0), -60.0, 60.0))
        )
        vulnerable_ratio = group_contract[
            "vulnerable_area_ratio"].reshape(1, -1)
        single_hit_damage = np.where(
            penetration_margin >= 1.0,
            penetration_probability * vulnerable_ratio,
            0.02 * vulnerable_ratio,
        )
        fragment_armor_aware_damage_proxy = (
            1.0
            - np.exp(-fragment_grid_exposure * single_hit_damage)
        )
        shock_proxy = _shock_damage_proxy(
            distance_m, tnt_mass, group_contract)
        mechanism_group_proxies["fragment"][group_name] = (
            fragment_damage_proxy)
        mechanism_group_proxies["shock"][group_name] = shock_proxy
        for component_index, component_id in enumerate(
                group_contract["component_ids"]):
            component_proxies["fragment"][int(component_id)] = (
                fragment_damage_proxy[:, component_index])
            component_proxies[
                "fragment_armor_aware"][int(component_id)] = (
                    fragment_armor_aware_damage_proxy[:, component_index])
            component_proxies["shock"][int(component_id)] = (
                shock_proxy[:, component_index])

        prefix = f"phys_{group_name}_"
        derived[prefix + "min_distance_m"] = distance_m.min(axis=1)
        derived[prefix + "mean_distance_m"] = distance_m.mean(axis=1)
        derived[prefix + "max_shock_damage_proxy"] = shock_proxy.max(axis=1)
        derived[prefix + "mean_shock_damage_proxy"] = shock_proxy.mean(axis=1)
        derived[prefix + "max_fragment_exposure_proxy"] = (
            fragment_proxy.max(axis=1))
        derived[prefix + "mean_fragment_exposure_proxy"] = (
            fragment_proxy.mean(axis=1))
        derived[prefix + "min_fragment_cone_residual_rad"] = (
            polar_residual.min(axis=1))
        derived[prefix + "min_fragment_azimuth_residual_rad"] = (
            azimuth_residual.min(axis=1))
        derived[prefix + "max_fragment_grid_alignment"] = (
            grid_alignment.max(axis=1))
        derived[prefix + "max_fragment_damage_proxy"] = (
            fragment_damage_proxy.max(axis=1))
        derived[prefix + "mean_fragment_damage_proxy"] = (
            fragment_damage_proxy.mean(axis=1))

    for mechanism, group_proxies in mechanism_group_proxies.items():
        rule_proxies = _damage_tree_rule_proxies(group_proxies)
        for task in ("K", "M", "F", "C"):
            for level in (1, 2):
                derived[
                    f"phys_{mechanism}_{task}_ge{level}_rule_proxy"
                ] = rule_proxies[f"{task}_ge{level}"]

    if include_combined_rule_proxies:
        combined_group_proxies = {
            group_name: (
                1.0
                - (1.0 - mechanism_group_proxies["fragment"][group_name])
                * (1.0 - mechanism_group_proxies["shock"][group_name])
            )
            for group_name in DAMAGE_RULE_COMPONENT_GROUPS
        }
        combined_rule_proxies = _damage_tree_rule_proxies(
            combined_group_proxies)
        for task in ("K", "M", "F", "C"):
            for level in (1, 2):
                derived[
                    f"phys_combined_{task}_ge{level}_rule_proxy"
                ] = combined_rule_proxies[f"{task}_ge{level}"]

    if include_component_proxies:
        for mechanism in ("fragment", "shock"):
            missing_component_ids = sorted(
                set(CRITICAL_COMPONENT_IDS)
                - set(component_proxies[mechanism])
            )
            if missing_component_ids:
                raise RuntimeError(
                    "Per-component proxy derivation omitted IDs: "
                    f"{missing_component_ids}")
            for component_id in CRITICAL_COMPONENT_IDS:
                proxy_key = (
                    "fragment_armor_aware"
                    if (
                        mechanism == "fragment"
                        and armor_aware_fragment_proxies
                    ) else mechanism
                )
                derived[
                    f"phys_component_{component_id:03d}_"
                    f"{mechanism}_damage_proxy"
                ] = component_proxies[proxy_key][component_id]

    feature_frame = pd.DataFrame(derived, index=output.index)
    requested_feature_columns = list(
        TERMINAL_PHYSICS_FEATURE_COLUMNS)
    if include_combined_rule_proxies:
        requested_feature_columns.extend(
            COMBINED_TASK_RULE_PROXY_FEATURE_COLUMNS)
    if include_component_proxies:
        requested_feature_columns.extend(
            COMPONENT_PROXY_FEATURE_COLUMNS)
    missing_features = [
        name for name in requested_feature_columns
        if name not in feature_frame.columns
    ]
    if missing_features:
        raise RuntimeError(
            "Terminal physics feature derivation omitted columns: "
            f"{missing_features}")
    existing_derived = [
        name for name in requested_feature_columns
        if name in output.columns
    ]
    if existing_derived:
        output = output.drop(columns=existing_derived)
    # Append all derived columns as one block.  Repeated DataFrame insertion
    # fragments memory badly on the 300k-row production dataset.
    output = pd.concat(
        [output, feature_frame[requested_feature_columns]],
        axis=1,
        copy=False,
    )
    values = output[requested_feature_columns].to_numpy(
        dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(
            "Terminal physics feature derivation produced non-finite values.")
    return output
