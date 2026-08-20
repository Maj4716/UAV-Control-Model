# -*- coding: utf-8 -*-
"""Shared frame, attitude, and SI-boundary definitions.

Stage-0 convention
------------------
Target/world frame ``T`` keeps the vehicle-model axes used by the existing JSON:

* ``+X_T``: right/east
* ``+Y_T``: forward/north
* ``+Z_T``: up

Navigation frame ``N`` is NED and body frame ``B`` is FRD:

* ``N`` = north/east/down
* ``B`` = forward/right/down

Euler angles follow the aerospace ZYX convention.  Yaw is clockwise from
``+Y_T`` (north) towards ``+X_T`` (east); positive pitch is nose-up; positive
roll is right-wing-down.  Quaternions are stored scalar-first ``[w, x, y, z]``
and rotate vectors from body to NED.

The damage geometry remains centimetre-based because the source vehicle model
is in centimetres.  Every external/digital-twin boundary in this module is SI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


FRAME_CONVENTION_VERSION = "stage0_ned_frd_v1"
CM_PER_M = 100.0


def target_to_ned(vector_t: np.ndarray) -> np.ndarray:
    """Convert a target-frame vector ``[east, north, up]`` to NED."""
    v = np.asarray(vector_t, dtype=float)
    return np.stack((v[..., 1], v[..., 0], -v[..., 2]), axis=-1)


def ned_to_target(vector_ned: np.ndarray) -> np.ndarray:
    """Convert a NED vector to target frame ``[east, north, up]``."""
    v = np.asarray(vector_ned, dtype=float)
    return np.stack((v[..., 1], v[..., 0], -v[..., 2]), axis=-1)


def body_to_ned_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Return the FRD-body to NED rotation matrix for ZYX Euler angles."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=float)


def body_to_target_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Return the FRD-body to target/world rotation matrix."""
    r_bn = body_to_ned_matrix(yaw, pitch, roll)
    # T=[east,north,up], NED=[north,east,down].
    return np.vstack((r_bn[1], r_bn[0], -r_bn[2]))


def normalize_quaternion(quaternion_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion_wxyz, dtype=float)
    if q.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {q.shape}")
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("quaternion norm must be non-zero")
    return q / norm


def quaternion_from_euler(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Create scalar-first body-to-NED quaternion from ZYX Euler angles."""
    hy, hp, hr = yaw * 0.5, pitch * 0.5, roll * 0.5
    cy, sy = np.cos(hy), np.sin(hy)
    cp, sp = np.cos(hp), np.sin(hp)
    cr, sr = np.cos(hr), np.sin(hr)
    return normalize_quaternion(np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]))


def quaternion_to_body_to_ned_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Convert scalar-first body-to-NED quaternion to a rotation matrix."""
    w, x, y, z = normalize_quaternion(quaternion_wxyz)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


@dataclass(frozen=True)
class TerminalEncounterState:
    """SI-only terminal state passed from flight dynamics to damage assessment.

    ``position_t_m`` and ``velocity_t_mps`` describe the munition in target
    coordinates. ``target_*`` permits a moving target without changing the
    damage engine's target-fixed internal geometry.
    """

    position_t_m: np.ndarray
    velocity_t_mps: np.ndarray
    attitude_bn_wxyz: np.ndarray
    angular_rate_body_rps: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target_position_t_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target_velocity_t_mps: np.ndarray = field(default_factory=lambda: np.zeros(3))
    munition_id: int = 0
    detonation_delay_s: float = 0.0
    frame_version: str = FRAME_CONVENTION_VERSION

    def __post_init__(self) -> None:
        for name in (
            "position_t_m", "velocity_t_mps", "angular_rate_body_rps",
            "target_position_t_m", "target_velocity_t_mps",
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite shape-(3,) vector")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "attitude_bn_wxyz", normalize_quaternion(self.attitude_bn_wxyz))
        if not 0 <= int(self.munition_id) <= 3:
            raise ValueError("munition_id must be in [0, 3]")
        if not np.isfinite(self.detonation_delay_s) or self.detonation_delay_s < 0:
            raise ValueError("detonation_delay_s must be finite and non-negative")
        if self.frame_version != FRAME_CONVENTION_VERSION:
            raise ValueError(
                f"unsupported frame_version={self.frame_version!r}; "
                f"expected {FRAME_CONVENTION_VERSION!r}")

    @property
    def relative_position_t_m(self) -> np.ndarray:
        return self.position_t_m - self.target_position_t_m

    @property
    def relative_velocity_t_mps(self) -> np.ndarray:
        return self.velocity_t_mps - self.target_velocity_t_mps

    @property
    def body_to_target_rotation(self) -> np.ndarray:
        r_bn = quaternion_to_body_to_ned_matrix(self.attitude_bn_wxyz)
        return np.vstack((r_bn[1], r_bn[0], -r_bn[2]))

    def to_damage_engine_inputs(self) -> Dict[str, Any]:
        """Return explicit legacy-engine fields with unit conversion at one boundary."""
        rel_pos_cm = self.relative_position_t_m * CM_PER_M
        rel_vel = self.relative_velocity_t_mps
        return {
            "position_cm": rel_pos_cm,
            "velocity_t_mps": rel_vel,
            "body_to_target_rotation": self.body_to_target_rotation,
            "munition_id": int(self.munition_id),
            "detonation_delay_s": float(self.detonation_delay_s),
            "frame_version": self.frame_version,
        }
