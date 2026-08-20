# -*- coding: utf-8 -*-
"""
巡飞弹对装甲车毁伤评估物理仿真器 (合并版 v2)
============================================

审计修正清单:
  Fix1: Taylor角基准 — arctan(2Vg/Vd) 从弹轴度量, sin→径向 cos→轴向
  Fix2: 装甲壳体碰撞 — 破片穿过车体装甲后扣减存速再判内部部件
  Fix3: 爆炸点一致 — dx/dy/dz = 起爆中心, 去掉warhead_position偏移
  Fix4: 破片总数 — 最后一环补足余数
  Fix5: de Marre → Thor方程 — V50=K*t^α*m^β*(secθ)^γ
  Fix6: 着靶角死代码清理
  Fix7: 多次穿透概率逐个合成(存储每次margin)
"""

import numpy as np
import json
import csv
import os
from importlib import resources
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any
from enum import Enum
from collections import defaultdict

from loitering_munition_damage_twin.simulation.coordinate_frames import (
    FRAME_CONVENTION_VERSION,
    TerminalEncounterState,
    body_to_target_matrix,
)


def bundled_resource_path(name: str) -> str:
    """Return the installed path for a bundled simulation resource."""
    resource = resources.files(
        "loitering_munition_damage_twin.simulation.resources"
    ).joinpath(name)
    return str(resource)


# ============================================================================
#  枚举
# ============================================================================

class DetonationPoint(Enum):
    FRONT = "前端起爆"
    REAR  = "后端起爆"
    CENTER = "中心起爆"

class FragmentArrangement(Enum):
    UNIFORM_RING = "均匀环排"

class FragmentShape(Enum):
    CUBE = "立方体"
    SPHERE = "球形"
    CYLINDER = "圆柱体"
    IRREGULAR = "不规则"

class DamageLevel(Enum):
    NONE = "未毁伤"
    LIGHT = "轻度毁伤"
    MODERATE = "中度毁伤"
    SEVERE = "重度毁伤"
    CATASTROPHIC = "灾难性毁伤"

class DamageCategory(Enum):
    K = "灾难性"
    M = "机动"
    F = "火力"
    C = "乘员"


# ============================================================================
#  坐标系
# ============================================================================

def rotation_matrix_zyx(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """FRD机体系到目标系的 ZYX 旋转矩阵，参数为弧度。

    目标系为 ``[东/右, 北/前, 天/上]``；yaw=0 时机头 ``+X_B`` 指向
    ``+Y_T``，pitch=-90° 时机头垂直向下。名称保留以兼容旧调用方。
    """
    return body_to_target_matrix(yaw, pitch, roll)

def _euler_rotation_matrix(rx, ry, rz):
    """XYZ欧拉角 (弧度) → 3×3旋转矩阵, 用于部件旋转"""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rz @ Ry @ Rx


@dataclass
class EncounterCondition:
    """
    弹目交汇条件（兼容层，位置仍以 cm 表示）

    dx/dy/dz: 起爆中心在T系中相对目标几何中心的偏移 (cm)
    yaw/pitch/roll: FRD机体系相对NED导航系姿态 (deg), pitch=-90°=垂直俯冲
    vx/vy/vz: 三维速度矢量 (m/s), 在T系中
    """
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 200.0
    yaw_deg: float = 0.0
    pitch_deg: float = -90.0
    roll_deg: float = 0.0
    vx: float = 0.0
    vy: float = 100.0
    vz: float = 0.0
    weight: float = 1.0

    @property
    def velocity(self) -> float:
        """速度标量 (m/s)"""
        return float(np.sqrt(self.vx**2 + self.vy**2 + self.vz**2))

    @property
    def position_cm(self) -> np.ndarray:
        return np.array([self.dx, self.dy, self.dz])
    @property
    def yaw_rad(self): return np.radians(self.yaw_deg)
    @property
    def pitch_rad(self): return np.radians(self.pitch_deg)
    @property
    def roll_rad(self): return np.radians(self.roll_deg)

    @property
    def velocity_vector_ms(self) -> np.ndarray:
        """三维速度矢量 (m/s), 在T系中"""
        return np.array([self.vx, self.vy, self.vz])

    def to_feature_vector(self) -> np.ndarray:
        """12维特征向量: [dx, dy, dz, vx, vy, vz, sin_yaw, cos_yaw, sin_pitch, cos_pitch, sin_roll, cos_roll]"""
        return np.array([self.dx, self.dy, self.dz,
                         self.vx, self.vy, self.vz,
                         np.sin(self.yaw_rad), np.cos(self.yaw_rad),
                         np.sin(self.pitch_rad), np.cos(self.pitch_rad),
                         np.sin(self.roll_rad), np.cos(self.roll_rad)])

    @classmethod
    def from_speed_and_attitude(
        cls,
        dx: float = 0.0, dy: float = 0.0, dz: float = 200.0,
        yaw_deg: float = 0.0, pitch_deg: float = -90.0, roll_deg: float = 0.0,
        speed: float = 100.0,
    ):
        """从标量速度+姿态构造零攻角速度（弹轴为 FRD ``+X_B``）。"""
        R = rotation_matrix_zyx(np.radians(yaw_deg), np.radians(pitch_deg), np.radians(roll_deg))
        v = R @ np.array([1.0, 0.0, 0.0]) * speed
        return cls(dx=dx, dy=dy, dz=dz,
                   yaw_deg=yaw_deg, pitch_deg=pitch_deg, roll_deg=roll_deg,
                   vx=float(v[0]), vy=float(v[1]), vz=float(v[2]))

    @classmethod
    def from_terminal_state(cls, state: TerminalEncounterState):
        """在唯一 SI→cm 边界把数字孪生终端状态适配为毁伤引擎输入。"""
        data = state.to_damage_engine_inputs()
        rotation = data["body_to_target_rotation"]
        # 从旋转矩阵的机头列恢复本项目 yaw/pitch；roll 用稳定的 FRD/NED 公式。
        r_bn = np.vstack((rotation[1], rotation[0], -rotation[2]))
        pitch = np.arcsin(np.clip(-r_bn[2, 0], -1.0, 1.0))
        yaw = np.arctan2(r_bn[1, 0], r_bn[0, 0])
        roll = np.arctan2(r_bn[2, 1], r_bn[2, 2])
        pos = data["position_cm"]
        vel = data["velocity_t_mps"]
        return cls(
            dx=float(pos[0]), dy=float(pos[1]), dz=float(pos[2]),
            yaw_deg=float(np.degrees(yaw)),
            pitch_deg=float(np.degrees(pitch)),
            roll_deg=float(np.degrees(roll)),
            vx=float(vel[0]), vy=float(vel[1]), vz=float(vel[2]),
        )


def transform_b2t(pts: np.ndarray, enc: EncounterCondition) -> np.ndarray:
    """点 B系→T系 (N,3)"""
    R = rotation_matrix_zyx(enc.yaw_rad, enc.pitch_rad, enc.roll_rad)
    return (R @ pts.T).T + enc.position_cm

def transform_dir_b2t(dirs: np.ndarray, enc: EncounterCondition) -> np.ndarray:
    """方向 B系→T系 (N,3), 仅旋转"""
    R = rotation_matrix_zyx(enc.yaw_rad, enc.pitch_rad, enc.roll_rad)
    return (R @ dirs.T).T


# ============================================================================
#  战斗部参数
# ============================================================================

@dataclass
class FragmentBed: # 破片床几何参数
    total_count: int = 100
    single_mass_g: float = 8.0
    shape: FragmentShape = FragmentShape.CUBE
    arrangement: FragmentArrangement = FragmentArrangement.UNIFORM_RING
    num_rings: int = 10
    per_ring: Optional[int] = None
    axial_coverage: float = 0.85
    spread_sigma_deg: float = 2.0

    @property
    def total_mass_kg(self): return self.total_count * self.single_mass_g / 1000.0
    @property
    def fragments_per_ring(self):
        if self.per_ring is not None: return self.per_ring
        return max(1, self.total_count // max(1, self.num_rings))
    @property
    def drag_coefficient(self):
        return {FragmentShape.SPHERE:0.47, FragmentShape.CUBE:1.05,
                FragmentShape.CYLINDER:0.82, FragmentShape.IRREGULAR:0.9}.get(self.shape, 0.9)

@dataclass
class Warhead:
    length_cm: float = 20.0
    diameter_cm: float = 12.0
    wall_thickness_cm: float = 0.3
    charge_mass_kg: float = 2.0
    tnt_equivalent: float = 1.0
    gurney_energy_mps: float = 2440.0
    detonation_velocity_mps: float = 6900.0
    detonation_point: DetonationPoint = DetonationPoint.FRONT
    fragment_bed: FragmentBed = field(default_factory=FragmentBed)

    @property
    def tnt_equivalent_mass_kg(self): return self.charge_mass_kg * self.tnt_equivalent
    @property
    def radius_cm(self): return self.diameter_cm / 2.0
    @property
    def _case_mass_kg(self):
        ro = self.radius_cm / 100; ri = (self.radius_cm - self.wall_thickness_cm) / 100
        return np.pi * (ro**2 - ri**2) * (self.length_cm / 100) * 7850
    @property
    def metal_to_charge_ratio(self):
        m = self.fragment_bed.total_mass_kg + self._case_mass_kg
        return m / self.charge_mass_kg if m > 0 and self.charge_mass_kg > 0 else 1.0

@dataclass
class Projectile:
    name: str = "通用巡飞弹战斗部"
    total_length_cm: float = 40.0
    total_diameter_cm: float = 15.0
    warhead: Warhead = field(default_factory=Warhead)

    def summary(self):
        w = self.warhead; fb = w.fragment_bed
        return (f"=== {self.name} ===\n"
                f"装药{w.charge_mass_kg}kg, Gurney{w.gurney_energy_mps}m/s, "
                f"{w.detonation_point.value}\n"
                f"破片{fb.total_count}枚×{fb.single_mass_g}g, M/C={w.metal_to_charge_ratio:.2f}\n")


# ---- 预设弹药库 ----

def create_small_loitering_munition():
    return Projectile(name="小型巡飞弹战斗部", total_length_cm=30, total_diameter_cm=10,
        warhead=Warhead(length_cm=12, diameter_cm=9, wall_thickness_cm=0.25,
            charge_mass_kg=1.0, gurney_energy_mps=2440, detonation_velocity_mps=6900,
            detonation_point=DetonationPoint.FRONT,
            fragment_bed=FragmentBed(total_count=60, single_mass_g=5.0, num_rings=6)))

def create_medium_loitering_munition():
    return Projectile(name="中型巡飞弹战斗部", total_length_cm=45, total_diameter_cm=15,
        warhead=Warhead(length_cm=20, diameter_cm=13, wall_thickness_cm=0.3,
            charge_mass_kg=3.0, gurney_energy_mps=2440, detonation_velocity_mps=6900,
            detonation_point=DetonationPoint.FRONT,
            fragment_bed=FragmentBed(total_count=150, single_mass_g=8.0, num_rings=12)))

def create_medium_rear_det():
    """中型巡飞弹 (后端起爆 — 适合俯冲攻顶)"""
    return Projectile(name="中型巡飞弹(后端起爆)", total_length_cm=45, total_diameter_cm=15,
        warhead=Warhead(length_cm=20, diameter_cm=13, wall_thickness_cm=0.3,
            charge_mass_kg=3.0, gurney_energy_mps=2440, detonation_velocity_mps=6900,
            detonation_point=DetonationPoint.REAR,
            fragment_bed=FragmentBed(total_count=150, single_mass_g=8.0, num_rings=12)))

def create_heavy_loitering_munition():
    return Projectile(name="大型巡飞弹战斗部", total_length_cm=60, total_diameter_cm=20,
        warhead=Warhead(length_cm=28, diameter_cm=18, wall_thickness_cm=0.4,
            charge_mass_kg=5.0, gurney_energy_mps=2700, detonation_velocity_mps=7900,
            detonation_point=DetonationPoint.FRONT,
            fragment_bed=FragmentBed(total_count=300, single_mass_g=10.0, num_rings=18)))


# ============================================================================
#  Gurney + Taylor + 气动衰减
# ============================================================================

class GurneyModel:
    @staticmethod
    def cylinder_velocity(gurney_e: float, mc_ratio: float) -> float:
        """圆柱壳体Gurney速度, mc_ratio=M/C"""
        return gurney_e / np.sqrt(mc_ratio + 0.5) if mc_ratio > 0 else gurney_e

class TaylorAngleModel:
    """
    [Fix1/Stage0] Taylor飞散角 — 从 FRD 弹轴(+X_B)度量

    经典Taylor角 δ = arctan(Vd/(2Vg)) 是从径向方向度量的。
    本代码返回的是从弹轴度量: θ = arctan(2Vg/Vd) = π/2 - δ。
    典型值: Vd=6900, Vg≈1975 → θ≈29.8° (破片在弹轴两侧约30°锥面飞散)。
    """
    @staticmethod
    def base_angle_rad(det_vel: float, gurney_vel: float) -> float:
        if gurney_vel <= 0 or det_vel <= 0:
            return np.pi / 4
        # [Fix1] 从弹轴度量: arctan(2Vg/Vd)
        ratio = 2.0 * gurney_vel / det_vel
        return np.arctan(np.clip(ratio, 0.05, 10.0))

    @staticmethod
    def angle_at_position(base_rad: float, norm_axial: float, det_pt: DetonationPoint) -> float:
        """轴向修正: 靠近起爆端飞散角增大"""
        mc = np.radians(10.0)
        if det_pt == DetonationPoint.FRONT:
            corr = mc * norm_axial
        elif det_pt == DetonationPoint.REAR:
            corr = mc * (1.0 - norm_axial)
        else:
            corr = mc * abs(2.0 * norm_axial - 1.0)
        return base_rad + corr


class FragmentRetardation:
    """V(r) = V₀·exp(-αr), α = ρ·Cd·A/(2m)"""
    RHO = 1.225

    @staticmethod
    def alpha(mass_g, cd, cs_cm2):
        m = mass_g / 1000; a = cs_cm2 / 1e4
        return FragmentRetardation.RHO * cd * a / (2*m) if m > 0 else 1.0

    @staticmethod
    def v_at_dist(v0, dist_m, alp):
        return v0 * np.exp(-alp * dist_m)

    @staticmethod
    def cross_section(mass_g):
        """等效迎风面积 (cm²)"""
        side = (mass_g / 1000 / 7850) ** (1/3)
        return side**2 * 1e4


# ============================================================================
#  破片飞散场生成
# ============================================================================

@dataclass
class FragmentState:
    frag_id: int
    mass_g: float
    origin_T: np.ndarray
    direction_T: np.ndarray
    speed_initial: float
    gurney_speed: float
    taylor_angle_deg: float
    ring_index: int = 0
    azimuth_deg: float = 0.0


def generate_fragment_field(
    projectile: Projectile,
    encounter: EncounterCondition,
    add_random_spread: bool = True,
    rng_seed: int = None,
    spread_sign: float = 1.0,
) -> List[FragmentState]:
    """
    在T系中生成全部破片状态

    [Stage0] FRD ``+X_B`` 为唯一弹轴；破片环位于 Y-Z 平面。
    [Fix3] 不加warhead_position偏移
    [Fix4] 最后一环补足余数
    """
    if float(spread_sign) not in (-1.0, 1.0):
        raise ValueError("spread_sign must be either -1 or +1.")
    rng = np.random.default_rng(rng_seed)
    w = projectile.warhead
    fb = w.fragment_bed

    v_gurney = GurneyModel.cylinder_velocity(w.gurney_energy_mps, w.metal_to_charge_ratio)
    base_taylor = TaylorAngleModel.base_angle_rad(w.detonation_velocity_mps, v_gurney)

    half_len = w.length_cm / 2.0
    x_start = -half_len * fb.axial_coverage
    x_end   =  half_len * fb.axial_coverage
    radius  = w.radius_cm

    # [Fix4] 补足破片
    n_rings = fb.num_rings
    base_per = fb.fragments_per_ring
    ring_counts = [base_per] * n_rings
    rem = fb.total_count - base_per * n_rings
    if rem > 0:
        ring_counts[-1] += rem

    # 将T系速度矢量变换到B系, 用于破片速度合成
    R_enc = rotation_matrix_zyx(encounter.yaw_rad, encounter.pitch_rad, encounter.roll_rad)
    v_proj_body = R_enc.T @ encounter.velocity_vector_ms

    origins_list, dirs_list = [], []
    taylor_list, ring_list, az_list = [], [], []
    fid = 0

    for i_ring in range(n_rings):
        norm_axial = i_ring / max(1, n_rings - 1) if n_rings > 1 else 0.5
        xpos = x_start + norm_axial * (x_end - x_start)
        taylor = TaylorAngleModel.angle_at_position(
            base_taylor, norm_axial, w.detonation_point)

        n_this = ring_counts[i_ring]
        for j in range(n_this):
            if fid >= fb.total_count:
                break
            az = 2 * np.pi * j / n_this
            if add_random_spread:
                spread_sigma = np.radians(fb.spread_sigma_deg)
                t_act = (
                    taylor
                    + float(spread_sign)
                    * rng.normal(0, spread_sigma)
                )
                a_act = (
                    az
                    + float(spread_sign)
                    * rng.normal(0, spread_sigma)
                )
            else:
                t_act, a_act = taylor, az

            origin = np.array([xpos, radius*np.cos(a_act), radius*np.sin(a_act)])
            radial = np.array([0.0, np.cos(a_act), np.sin(a_act)])

            # 轴向方向: 前端起爆→破片向后飞(-X), 后端起爆→向前飞(+X)
            if w.detonation_point == DetonationPoint.FRONT:
                ax_sign = -1.0
            elif w.detonation_point == DetonationPoint.REAR:
                ax_sign = 1.0
            else:
                ax_sign = 1.0 if ny >= 0.5 else -1.0

            # [Fix1] θ从弹轴度量: sin(θ)=径向, cos(θ)=轴向
            fd = np.sin(t_act) * radial + np.cos(t_act) * ax_sign * np.array([1,0,0])
            fd /= np.linalg.norm(fd)

            origins_list.append(origin)
            dirs_list.append(fd)
            taylor_list.append(np.degrees(t_act))
            ring_list.append(i_ring)
            az_list.append(np.degrees(a_act))
            fid += 1

    origins_b = np.array(origins_list)
    dirs_b = np.array(dirs_list)

    # 矢量合成
    v_abs = dirs_b * v_gurney + v_proj_body
    speeds = np.linalg.norm(v_abs, axis=1)
    dirs_abs = v_abs / speeds[:, np.newaxis]

    # [Fix3] 不加warhead_position偏移, position_cm就是起爆中心
    origins_T = transform_b2t(origins_b, encounter)
    dirs_T = transform_dir_b2t(dirs_abs, encounter)

    frags = []
    for i in range(len(origins_list)):
        frags.append(FragmentState(
            frag_id=i, mass_g=fb.single_mass_g,
            origin_T=origins_T[i], direction_T=dirs_T[i],
            speed_initial=float(speeds[i]), gurney_speed=v_gurney,
            taylor_angle_deg=float(taylor_list[i]),
            ring_index=ring_list[i], azimuth_deg=float(az_list[i])))
    return frags


def _arrival_speed(frag: FragmentState, hit_pt: np.ndarray, fb: FragmentBed) -> float:
    """破片到达某点的存速"""
    dist_m = np.linalg.norm(hit_pt - frag.origin_T) / 100
    cs = FragmentRetardation.cross_section(fb.single_mass_g)
    alp = FragmentRetardation.alpha(fb.single_mass_g, fb.drag_coefficient, cs)
    return FragmentRetardation.v_at_dist(frag.speed_initial, dist_m, alp)


# ============================================================================
#  碰撞检测几何体
# ============================================================================

@dataclass
class HitRecord:
    fragment_id: int
    component_id: int
    component_name: str
    hit_point: np.ndarray
    arrival_speed: float         # m/s, 含装甲衰减
    impact_angle_deg: float      # 0=正入射, 90=掠射
    distance_cm: float
    surface_normal: np.ndarray
    armor_traversed_mm: float = 0.0   # [Fix2] 途中穿过的装甲板总厚度
    self_thickness_mm: Optional[float] = None  # 装甲本体厚度覆盖（来自 armor.csv）

@dataclass
class BoxGeometry:
    center: np.ndarray
    half_extents: np.ndarray
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))

@dataclass
class CylinderGeometry:
    center: np.ndarray
    radius: float
    half_height: float
    axis: np.ndarray
    aabb_min: np.ndarray
    aabb_max: np.ndarray

@dataclass
class ExtrudedPolygonGeometry:
    """拉伸多边形: 在Y-Z平面定义多边形顶点，沿X轴拉伸"""
    start_x: float
    end_x: float
    vertices_yz: np.ndarray  # (N, 2)
    aabb_min: np.ndarray
    aabb_max: np.ndarray



# ---- 装甲壳体 [Fix2] ----

@dataclass
class ArmorPlate:
    """装甲板: 从 armor.csv 解析的侧板几何与厚度覆盖。"""
    name: str
    thickness_mm: float
    aabb_min: np.ndarray
    aabb_max: np.ndarray


def load_armor_plates(filepath: str = None) -> List[ArmorPlate]:
    """
    [Fix2] 加载装甲板几何, 参与碰撞检测
    上装甲较厚(15mm), 下装甲较薄(10mm)
    """
    if filepath is None:
        filepath = bundled_resource_path("armor.csv")
    if not filepath or not os.path.exists(filepath):
        return []

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))

    groups = defaultdict(list)
    for r in rows:
        key = (r['对象'].strip(), float(r['平面X']))
        groups[key].append(np.array([float(r['X坐标']), float(r['Y坐标']), float(r['Z坐标'])]))

    plates = []
    for (name, _px), verts in groups.items():
        pts = np.array(verts)
        source_rows = [r for r in rows
                       if r['对象'].strip() == name and float(r['平面X']) == _px]
        explicit = None
        for key in ("厚度mm", "厚度_mm", "厚度", "thickness_mm"):
            if source_rows and source_rows[0].get(key) not in (None, ""):
                explicit = float(source_rows[0][key])
                break
        thickness = explicit if explicit is not None else (15.0 if '上' in name else 10.0)
        # 装甲板是薄板, X方向给±2cm厚度
        mn = pts.min(axis=0) - np.array([2.0, 0.0, 0.0])
        mx = pts.max(axis=0) + np.array([2.0, 0.0, 0.0])
        plates.append(ArmorPlate(name=name, thickness_mm=thickness, aabb_min=mn, aabb_max=mx))
    return plates


# ---- 部件几何解析 ----

def parse_component_geometry(comp: Dict) -> Tuple[str, Any]:
    geom = comp['geometry']
    shape = geom['shape']
    dims = geom['dimensions']
    pos = geom['position']
    rot = geom.get('rotation', {'x':0,'y':0,'z':0})

    center = np.array([pos.get('x',0) or 0, pos.get('y',0) or 0, pos.get('z',0) or 0], dtype=float)

    if shape == "长方体":
        hx = (dims.get('length_or_radius',50) or 50) / 2
        hy = (dims.get('width',50) or 50) / 2
        hz = (dims.get('height',50) or 50) / 2
        half = np.array([hx, hy, hz], dtype=float)
        rx, ry, rz = rot.get('x',0) or 0, rot.get('y',0) or 0, rot.get('z',0) or 0
        R = _euler_rotation_matrix(np.radians(rx), np.radians(ry), np.radians(rz))
        extent = np.abs(R) @ half
        return ("box", BoxGeometry(center=center, half_extents=half,
                                   aabb_min=center-extent, aabb_max=center+extent,
                                   rotation=R))
    elif shape == "圆柱体":
        radius = dims.get('length_or_radius',30) or 30
        height = dims.get('height',30) or 30
        half_h = height / 2
        rx, ry, rz = (rot.get('x',0) or 0, rot.get('y',0) or 0, rot.get('z',0) or 0)
        R = _euler_rotation_matrix(np.radians(rx), np.radians(ry), np.radians(rz))
        axis = R @ np.array([0.0, 0.0, 1.0])
        axis /= np.linalg.norm(axis) + 1e-12
        ext = np.abs(axis) * half_h + radius * np.sqrt(np.maximum(0.0, 1.0 - axis**2))
        return ("cylinder", CylinderGeometry(center=center, radius=radius, half_height=half_h,
                                              axis=axis, aabb_min=center-ext, aabb_max=center+ext))
    elif shape == "拉伸多边形":
        ext_len = dims.get('extrusion_length', 100)
        start_x = pos.get('x', 0)
        end_x = start_x + ext_len
        verts_yz = np.array(geom.get('vertices_yz', []))
        if len(verts_yz) < 3:
            # Fallback
            return ("box", BoxGeometry(center=center, half_extents=np.full(3, 30),
                                       aabb_min=center-30, aabb_max=center+30))
        min_y, min_z = verts_yz.min(axis=0)
        max_y, max_z = verts_yz.max(axis=0)
        aabb_mn = np.array([min(start_x, end_x), min_y, min_z])
        aabb_mx = np.array([max(start_x, end_x), max_y, max_z])
        return ("extruded_poly", ExtrudedPolygonGeometry(
            start_x=min(start_x, end_x), end_x=max(start_x, end_x),
            vertices_yz=verts_yz, aabb_min=aabb_mn, aabb_max=aabb_mx))
    else:
        r = 30.0
        return ("box", BoxGeometry(center=center, half_extents=np.full(3,r),
                                   aabb_min=center-r, aabb_max=center+r))


# ---- 射线相交 ----

def ray_aabb(origin, dirn, mn, mx):
    """Slab法 → (hit, t, normal)"""
    t_min = -np.inf; t_max = np.inf; normal = np.zeros(3)
    for i in range(3):
        if abs(dirn[i]) < 1e-10:
            if origin[i] < mn[i] or origin[i] > mx[i]:
                return (False, 0.0, normal)
        else:
            t1 = (mn[i] - origin[i]) / dirn[i]
            t2 = (mx[i] - origin[i]) / dirn[i]
            if t1 > t2: t1, t2 = t2, t1; sign = 1.0
            else: sign = -1.0
            if t1 > t_min:
                t_min = t1; normal = np.zeros(3); normal[i] = sign
            t_max = min(t_max, t2)
            if t_min > t_max:
                return (False, 0.0, np.zeros(3))
    if t_min > 0: return (True, t_min, normal)
    elif t_max > 0: return (True, t_max, -normal)
    return (False, 0.0, np.zeros(3))


def ray_box(origin, dirn, box: BoxGeometry):
    """射线-有向包围盒；AABB 仅作为宽相位，不再替代旋转几何。"""
    if not ray_aabb(origin, dirn, box.aabb_min, box.aabb_max)[0]:
        return (False, 0.0, np.zeros(3))
    origin_local = box.rotation.T @ (origin - box.center)
    dir_local = box.rotation.T @ dirn
    hit, t, normal_local = ray_aabb(
        origin_local, dir_local, -box.half_extents, box.half_extents)
    if not hit:
        return (False, 0.0, np.zeros(3))
    normal_world = box.rotation @ normal_local
    normal_world /= np.linalg.norm(normal_world) + 1e-12
    return (True, t, normal_world)


def ray_cyl(origin, dirn, cyl: CylinderGeometry):
    """射线-有限长圆柱 → (hit, t, normal)"""
    if not ray_aabb(origin, dirn, cyl.aabb_min, cyl.aabb_max)[0]:
        return (False, 0.0, np.zeros(3))
    best_t = np.inf; best_n = np.zeros(3); hit = False
    ax = cyl.axis
    d_perp = dirn - np.dot(dirn, ax) * ax
    oc = origin - cyl.center
    oc_perp = oc - np.dot(oc, ax) * ax
    a = np.dot(d_perp, d_perp)
    b = 2 * np.dot(d_perp, oc_perp)
    c = np.dot(oc_perp, oc_perp) - cyl.radius**2
    disc = b*b - 4*a*c
    if disc >= 0 and a > 1e-10:
        sq = np.sqrt(disc)
        for tc in [(-b-sq)/(2*a), (-b+sq)/(2*a)]:
            if 0 < tc < best_t:
                p = origin + tc * dirn
                h = np.dot(p - cyl.center, ax)
                if abs(h) <= cyl.half_height:
                    best_t = tc; hit = True
                    rad = p - cyl.center - h * ax
                    best_n = rad / (np.linalg.norm(rad) + 1e-12)
    for sign in [1.0, -1.0]:
        cc = cyl.center + sign * cyl.half_height * ax
        cn = sign * ax
        den = np.dot(dirn, cn)
        if abs(den) < 1e-10: continue
        tc = np.dot(cc - origin, cn) / den
        if 0 < tc < best_t:
            p = origin + tc * dirn
            off = p - cc
            if np.dot(off, off) - np.dot(off, ax)**2 <= cyl.radius**2:
                best_t = tc; best_n = cn; hit = True
    return (hit, best_t, best_n)


def point_in_polygon_2d(pt_y: float, pt_z: float, verts_yz: np.ndarray) -> bool:
    """射线法判断二维点是否在多边形内"""
    inside = False
    n = len(verts_yz)
    j = n - 1
    for i in range(n):
        vi = verts_yz[i]
        vj = verts_yz[j]
        if ((vi[1] > pt_z) != (vj[1] > pt_z)) and \
           (pt_y < (vj[0] - vi[0]) * (pt_z - vi[1]) / (vj[1] - vi[1] + 1e-12) + vi[0]):
            inside = not inside
        j = i
    return inside

def ray_extruded_polygon(origin: np.ndarray, dirn: np.ndarray, poly: ExtrudedPolygonGeometry):
    """射线-拉伸多边形(X轴拉伸) → (hit, t, normal)"""
    if not ray_aabb(origin, dirn, poly.aabb_min, poly.aabb_max)[0]:
        return (False, 0.0, np.zeros(3))

    best_t = np.inf
    best_n = np.zeros(3)
    hit = False

    # 1. 检查两个X轴端面 (X = start_x, X = end_x)
    for nx, px in [(1.0, poly.end_x), (-1.0, poly.start_x)]:
        if abs(dirn[0]) > 1e-10:
            t = (px - origin[0]) / dirn[0]
            if 0 < t < best_t:
                pt = origin + t * dirn
                if point_in_polygon_2d(pt[1], pt[2], poly.vertices_yz):
                    best_t = t
                    best_n = np.array([nx, 0.0, 0.0])
                    hit = True

    # 2. 检查侧棱面
    n_verts = len(poly.vertices_yz)
    for i in range(n_verts):
        v1 = poly.vertices_yz[i]
        v2 = poly.vertices_yz[(i+1) % n_verts]
        # 面上的一个点：(poly.start_x, v1[0], v1[1])
        # 面的法线：Y-Z平面上边 (v1->v2) 的右侧法线 (正方形时朝外)
        dy = v2[0] - v1[0]
        dz = v2[1] - v1[1]
        line_len = np.hypot(dy, dz)
        if line_len < 1e-10: continue
        # 面板法线：X分量为0，YZ平面上垂直于直线，指向外部
        ny = dz / line_len
        nz = -dy / line_len
        normal = np.array([0.0, ny, nz])

        # 射线与平面求交
        denom = np.dot(dirn, normal)
        if abs(denom) > 1e-10:
            p0 = np.array([poly.start_x, v1[0], v1[1]])
            t = np.dot(p0 - origin, normal) / denom
            if 0 < t < best_t:
                pt = origin + t * dirn
                # 检测交点是否在面的范围内：X在[start_x, end_x]，YZ在两点连线上
                if poly.start_x <= pt[0] <= poly.end_x:
                    # 投影到线段上看是否在线段中间
                    proj_len = (pt[1]-v1[0])*dy + (pt[2]-v1[1])*dz
                    if 0 <= proj_len <= dy**2 + dz**2:
                        best_t = t
                        best_n = normal
                        hit = True

    return (hit, best_t, best_n)


def ray_geometry(origin: np.ndarray, dirn: np.ndarray, shape_type: str, geometry: Any):
    """Dispatch a ray query to the exact supported component geometry."""
    if shape_type == "box":
        return ray_box(origin, dirn, geometry)
    if shape_type == "cylinder":
        return ray_cyl(origin, dirn, geometry)
    if shape_type == "extruded_poly":
        return ray_extruded_polygon(origin, dirn, geometry)
    return (False, 0.0, np.zeros(3))


def line_of_sight_armor_mm(origin: np.ndarray, destination: np.ndarray,
                           parsed_components: List[Tuple],
                           armor_plates: List[ArmorPlate] = None,
                           exclude_component_id: int = None) -> float:
    """Sum armor thickness intersected before a destination component.

    This is a lightweight geometric shielding model for blast transmission.  It
    replaces the previous ID-only shielding lookup whenever a line of sight
    crosses a modeled armor solid.
    """
    ray = np.asarray(destination, dtype=float) - np.asarray(origin, dtype=float)
    distance = float(np.linalg.norm(ray))
    if distance < 1e-9:
        return 0.0
    direction = ray / distance
    overrides = defaultdict(list)
    for plate in armor_plates or []:
        overrides[plate.name].append(float(plate.thickness_mm))

    total = 0.0
    seen_ids = set()
    for cid, cname, st, geometry, cdata in parsed_components:
        if cid == exclude_component_id or "装甲" not in cname or cid in seen_ids:
            continue
        hit, t, _ = ray_geometry(origin, direction, st, geometry)
        if hit and 1e-6 < t < distance - 1e-6:
            default_thickness = cdata.get('material', {}).get(
                'equivalent_thickness', 30.0) or 30.0
            values = overrides.get(cname)
            total += float(np.mean(values)) if values else float(default_thickness)
            seen_ids.add(cid)
    return total



# [Fix2] 内部部件ID — 受装甲壳体保护
INTERNAL_IDS = {
    1, 2, 3,          # 传动/发动机/油箱
    43, 44,           # 火控
    46, 49,           # 弹药架/供弹
    53, 54,           # 辅弹
    57,               # 电台
    58, 59, 60,       # 驾驶员/车长/炮长
    *range(61, 68),   # 步兵
}


def detect_all_hits(
    fragments: List[FragmentState],
    components: List[Dict],
    warhead: Warhead,
    armor_plates: List[ArmorPlate] = None, # 兼容老代码接口
    max_range_cm: float = 5000.0,
) -> List[HitRecord]:
    """
    全部破片碰撞检测

    [Fix2] 装甲壳体碰撞: 射线收集所有经过的带"装甲"名字的部件厚度，并最终击中第一个非装甲部件。
           将装甲击中本身也作为 HitRecord 返回，用于统计装甲毁伤。
    """
    parsed = []
    for comp in components:
        st, go = parse_component_geometry(comp)
        parsed.append((comp['id'], comp['name'], st, go, comp))

    # armor.csv is the authoritative thickness override for matching hull names.
    # The component geometry remains authoritative for intersection because it is
    # a closed extruded solid, while the CSV contains separate side polygons.
    armor_thickness_by_name = {}
    for plate in armor_plates or []:
        armor_thickness_by_name.setdefault(plate.name, []).append(plate.thickness_mm)
    armor_thickness_by_name = {
        name: float(np.mean(values)) for name, values in armor_thickness_by_name.items()
    }

    fb = warhead.fragment_bed
    records = []

    for frag in fragments:
        d = frag.direction_T
        dn = np.linalg.norm(d)
        if dn < 1e-10: continue
        dirn = d / dn
        origin = frag.origin_T

        # 收集射线上的所有交点
        # hits_on_ray 格式: list of (t, cid, cname, normal, is_armor)
        hits_on_ray = []

        for cid, cname, st, go, cdata in parsed:
            ih, it, inl = ray_geometry(origin, dirn, st, go)

            if ih and 0 < it < max_range_cm:
                is_armor = "装甲" in cname
                hits_on_ray.append((it, cid, cname, inl, cdata, is_armor))

        if not hits_on_ray:
            continue

        # 按距离排序
        hits_on_ray.sort(key=lambda x: x[0])

        armor_traversed_mm = 0.0

        for (t, cid, cname, normal, cdata, is_armor) in hits_on_ray:
            hit_pt = origin + t * dirn
            speed = _arrival_speed(frag, hit_pt, fb)

            cos_a = np.clip(abs(np.dot(dirn, normal)), 0, 1)
            impact_deg = np.degrees(np.arccos(cos_a))

            if is_armor:
                component_thickness = cdata.get('material', {}).get(
                    'equivalent_thickness', 30.0) or 30.0
                armor_thickness = armor_thickness_by_name.get(
                    cname, float(component_thickness))
                # 记录击中装甲
                records.append(HitRecord(
                    fragment_id=frag.frag_id, component_id=cid, component_name=cname,
                    hit_point=hit_pt, arrival_speed=speed,
                    impact_angle_deg=impact_deg, distance_cm=t,
                    surface_normal=normal, armor_traversed_mm=armor_traversed_mm,
                    self_thickness_mm=armor_thickness))

                # 累加装甲厚度，供后续的部件使用
                armor_traversed_mm += armor_thickness

            else:
                # 击中第一个非装甲部件（内部部件）
                actual_armor_mm = armor_traversed_mm
                if cid in INTERNAL_IDS and armor_traversed_mm == 0:
                    actual_armor_mm = 30.0

                records.append(HitRecord(
                    fragment_id=frag.frag_id, component_id=cid, component_name=cname,
                    hit_point=hit_pt, arrival_speed=speed,
                    impact_angle_deg=impact_deg, distance_cm=t,
                    surface_normal=normal, armor_traversed_mm=actual_armor_mm))
                break # 破片在非装甲部件内部停留在或者不再追踪后面的部件

    return records


# ============================================================================
#  Thor侵彻模型 [Fix5]
# ============================================================================

class ThorPenetrationModel:
    """
    [Fix5] Thor方程取代de Marre

    V50 = K × t^α × m^β × (sec θ)^γ   (m/s)

    钢质预制破片 vs 均质装甲钢 (赵国志《终点效应学》):
        K = 270, α = 0.75, β = -0.37, γ = 1.2
        t: mm, m: g, θ: rad
    """
    K = 270.0
    ALPHA = 0.75
    BETA = -0.37
    GAMMA = 1.2

    @staticmethod
    def v50(armor_mm: float, mass_g: float, impact_rad: float = 0.0) -> float:
        """极限穿透速度 V50 (m/s)"""
        if armor_mm <= 0 or mass_g <= 0:
            return 0.0
        cos_t = max(np.cos(impact_rad), 0.1)  # 掠射限幅
        sec_t = 1.0 / cos_t
        return ThorPenetrationModel.K * (armor_mm ** ThorPenetrationModel.ALPHA) * \
               (mass_g ** ThorPenetrationModel.BETA) * (sec_t ** ThorPenetrationModel.GAMMA)

    @staticmethod
    def is_penetrated(speed: float, armor_mm: float, mass_g: float,
                      impact_rad: float = 0.0) -> Tuple[bool, float, float]:
        """(穿透?, V50, margin=V/V50)"""
        v = ThorPenetrationModel.v50(armor_mm, mass_g, impact_rad)
        if v <= 0: return (True, 0.0, float('inf'))
        margin = speed / v
        return (margin >= 1.0, v, margin)

    @staticmethod
    def residual_speed(speed: float, v50: float) -> float:
        """穿透后剩余速度估算: V_r = V × sqrt(1 - (V50/V)^2)"""
        if speed <= v50 or speed <= 0:
            return 0.0
        return speed * np.sqrt(1.0 - (v50 / speed) ** 2)

    @staticmethod
    def damage_prob(margin: float) -> float:
        """S型概率: margin=1→P=0.5"""
        return 1.0 / (1.0 + np.exp(-8.0 * (margin - 1.0)))


# ============================================================================
#  冲击波模型 (含装甲遮蔽修正)
# ============================================================================

class ShockwaveModel:
    """Sadovsky超压 + 反射/透射 + 部件分类"""

    # 部件暴露分类
    EXPOSED_IDS = set(range(6,18)) | set(range(18,24)) | set(range(24,30)) | \
                  {30,31} | set(range(32,40)) | {40,47,48,55,56}
    SEMI_EXPOSED_IDS = {41, 42, 45, 50, 51, 52}

    # 其余为内部部件

    # 内部部件遮蔽厚度 (mm)
    SHIELDING = {
        1:15, 2:15, 3:12,          # 传动/发动机/油箱
        4:5, 5:5,                   # 主动轮(半遮蔽)
        43:25, 44:25,               # 火控
        46:25, 49:25, 53:25, 54:25, # 弹药
        57:15,                      # 电台
        58:15, 59:25, 60:25,        # 乘员
        **{i:12 for i in range(61,68)},  # 步兵
    }

    @staticmethod
    def overpressure_sadovsky(dist_m: float, tnt_kg: float) -> float:
        """Sadovsky公式: 入射超压 (MPa)"""
        if dist_m <= 0.01 or tnt_kg <= 0:
            return 100.0
        Z = dist_m / (tnt_kg ** (1/3))
        return 0.084/Z + 0.27/Z**2 + 0.705/Z**3

    @staticmethod
    def reflected_pressure(incident: float, angle_deg: float = 0) -> float:
        """反射超压 (简化: Cr=2~8取决于入射超压)"""
        cr = 2.0 + min(incident, 1.0) * 6.0
        angle_factor = np.cos(np.radians(min(angle_deg, 89)))
        return incident * cr * angle_factor

    @staticmethod
    def transmitted_pressure(incident: float, thickness_mm: float) -> float:
        """装甲透射超压: τ = exp(-β·t)"""
        if thickness_mm <= 0: return incident
        if incident < 0.5: beta = 0.25
        elif incident < 2.0: beta = 0.18
        else: beta = 0.12
        return incident * np.exp(-beta * thickness_mm)

    @staticmethod
    def effective_overpressure(comp: dict, incident_mpa: float,
                               geometric_shielding_mm: float = 0.0) -> float:
        cid = comp['id']
        cname = comp['name']
        if cid in ShockwaveModel.EXPOSED_IDS:
            return ShockwaveModel.reflected_pressure(incident_mpa, 45)
        elif cid in ShockwaveModel.SEMI_EXPOSED_IDS:
            return ShockwaveModel.transmitted_pressure(incident_mpa, 5.0)
        elif "装甲" in cname:
            return ShockwaveModel.reflected_pressure(incident_mpa, 0)
        else:
            # Prefer actual line-of-sight armor when available; retain the table
            # as a conservative compatibility fallback for incomplete geometry.
            t = (geometric_shielding_mm if geometric_shielding_mm > 0
                 else ShockwaveModel.SHIELDING.get(cid, 15.0))
            return ShockwaveModel.transmitted_pressure(incident_mpa, t)

    @staticmethod
    def armor_threshold(comp: dict) -> float:
        if "装甲" in comp['name']: return 8.0
        return None

    @staticmethod
    def damage_prob_shock(overpressure: float, threshold: float) -> float:
        """S型概率: threshold→P=0.5"""
        if threshold <= 0: return 0.0
        ratio = overpressure / threshold
        return 1.0 / (1.0 + np.exp(-6.0 * (ratio - 1.0)))

# ============================================================================
#  毁伤评估结果数据结构
# ============================================================================

@dataclass
class ComponentDamageResult:
    component_id: int
    component_name: str = ""
    fragment_hits: int = 0
    fragment_penetrations: int = 0
    fragment_damage_prob: float = 0.0
    # [Fix7] 存储每次穿透的margin, 逐个合成
    penetration_margins: List[float] = field(default_factory=list)
    overpressure_mpa: float = 0.0
    overpressure_threshold: float = 0.0
    shockwave_damage_prob: float = 0.0
    combined_damage_prob: float = 0.0
    is_damaged: bool = False
    damage_level: DamageLevel = DamageLevel.NONE

    def to_dict(self):
        return {
            "id": self.component_id, "name": self.component_name,
            "frag_hits": self.fragment_hits, "frag_pen": self.fragment_penetrations,
            "P_frag": round(self.fragment_damage_prob, 4),
            "overpressure": round(self.overpressure_mpa, 4),
            "P_shock": round(self.shockwave_damage_prob, 4),
            "P_comb": round(self.combined_damage_prob, 4),
            "damaged": self.is_damaged, "level": self.damage_level.value,
        }


@dataclass
class VehicleDamageResult:
    component_results: List[ComponentDamageResult] = field(default_factory=list)
    total_components: int = 0
    total_fragments: int = 0
    total_hits: int = 0
    total_penetrations: int = 0
    damaged_count: int = 0
    damage_tree: Optional[Any] = None  # DamageTreeResult
    # Mechanism-only counterfactuals.  They use the same component states as the
    # combined result and therefore add no Monte-Carlo cost, but make it possible
    # to audit whether a label is driven by fragments or by blast.
    damage_tree_fragment: Optional[Any] = None
    damage_tree_shockwave: Optional[Any] = None
    encounter: Optional[EncounterCondition] = None
    projectile_name: str = ""

    def to_dict(self):
        d = {
            "projectile": self.projectile_name,
            "total_components": self.total_components,
            "total_fragments": self.total_fragments,
            "total_hits": self.total_hits,
            "total_penetrations": self.total_penetrations,
            "damaged_count": self.damaged_count,
            "damage_ratio": round(self.damaged_count / max(1, self.total_components), 4),
            "components": [r.to_dict() for r in self.component_results],
        }
        if self.damage_tree:
            d["damage_tree"] = self.damage_tree.to_dict()
        if self.damage_tree_fragment:
            d["damage_tree_fragment"] = self.damage_tree_fragment.to_dict()
        if self.damage_tree_shockwave:
            d["damage_tree_shockwave"] = self.damage_tree_shockwave.to_dict()
        return d


# ============================================================================
#  毁伤评估引擎
# ============================================================================

class DamageEngine:
    """
    主引擎: 破片 + 冲击波 → 综合毁伤 → 毁伤树

    [Fix2] 装甲壳体碰撞: 内部部件侵彻时叠加装甲厚度
    [Fix5] Thor方程替代de Marre
    [Fix7] 多次穿透逐个计算margin并独立合成概率
    """
    def __init__(self, enable_fragments=True, enable_shockwave=True,
                 armor_plates: List[ArmorPlate] = None):
        self.enable_fragments = enable_fragments
        self.enable_shockwave = enable_shockwave
        self.armor_plates = armor_plates or []

    def evaluate(
        self,
        projectile: Projectile,
        encounter: EncounterCondition,
        components: List[Dict],
        rng_seed: int = None,
        fragment_spread_sign: float = 1.0,
        shockwave_probability_cache: Optional[Dict[int, float]] = None,
    ) -> VehicleDamageResult:

        warhead = projectile.warhead
        fb = warhead.fragment_bed
        result = VehicleDamageResult()
        result.encounter = encounter
        result.projectile_name = projectile.name
        result.total_components = len(components)

        # 初始化各部件结果
        cmap = {}
        for comp in components:
            cid = comp['id']
            cr = ComponentDamageResult(component_id=cid, component_name=comp['name'])
            cr.overpressure_threshold = comp.get('material', {}).get('overpressure_threshold', 0.3) or 0.3
            cmap[cid] = cr

        # ---- 破片毁伤 ----
        if self.enable_fragments:
            fragments = generate_fragment_field(
                projectile,
                encounter,
                rng_seed=rng_seed,
                spread_sign=fragment_spread_sign,
            )
            result.total_fragments = len(fragments)

            hits = detect_all_hits(fragments, components, warhead,
                                   armor_plates=self.armor_plates)
            result.total_hits = len(hits)

            for hit in hits:
                cid = hit.component_id
                if cid not in cmap: continue
                cr = cmap[cid]
                cr.fragment_hits += 1

                # [Fix2] 总装甲厚度 = 途中装甲板 + 部件自身等效厚度
                comp_data = next((c for c in components if c['id'] == cid), None)
                if comp_data is None: continue
                comp_thickness_mm = (
                    hit.self_thickness_mm
                    if hit.self_thickness_mm is not None
                    else comp_data.get('material', {}).get('equivalent_thickness', 10.0) or 10.0
                )
                total_armor_mm = hit.armor_traversed_mm + comp_thickness_mm

                # [Fix5] Thor方程侵彻判定
                pen, v50_val, margin = ThorPenetrationModel.is_penetrated(
                    speed=hit.arrival_speed,
                    armor_mm=total_armor_mm,
                    mass_g=fb.single_mass_g,
                    impact_rad=np.radians(hit.impact_angle_deg),
                )

                if pen:
                    cr.fragment_penetrations += 1
                # [Fix7] 存储每次的margin
                cr.penetration_margins.append(margin)

            # 计算各部件破片毁伤概率
            for cr in cmap.values():
                if cr.fragment_hits == 0:
                    cr.fragment_damage_prob = 0.0
                    continue

                comp_data = next((c for c in components if c['id'] == cr.component_id), None)
                vr = 1.0
                if comp_data:
                    vr = comp_data.get('material', {}).get('vulnerable_area_ratio', 1.0) or 1.0

                if cr.fragment_penetrations > 0:
                    # [Fix7] 逐个穿透独立合成
                    # P = 1 - ∏(1 - p_i × vr), 其中 p_i = sigmoid(margin_i)
                    survival = 1.0
                    for margin in cr.penetration_margins:
                        # 只合成真实穿透事件。旧逻辑在任一破片穿透后还会把
                        # 0.5<=margin<1 的未穿透破片计入，造成不连续的概率跃迁。
                        if margin >= 1.0:
                            p_single = ThorPenetrationModel.damage_prob(margin) * vr
                            survival *= (1.0 - p_single)
                    cr.fragment_damage_prob = min(1.0, 1.0 - survival)
                else:
                    cr.fragment_damage_prob = min(1.0, 0.02 * cr.fragment_hits * vr)

        # ---- 冲击波毁伤 ----
        if shockwave_probability_cache is not None and not self.enable_shockwave:
            raise ValueError(
                "shockwave_probability_cache requires shockwave evaluation.")
        if self.enable_shockwave:
            if shockwave_probability_cache is not None:
                expected_ids = {int(comp['id']) for comp in components}
                observed_ids = {
                    int(component_id)
                    for component_id in shockwave_probability_cache
                }
                if observed_ids != expected_ids:
                    raise ValueError(
                        "shockwave_probability_cache component IDs do not "
                        "match the vehicle model.")
                for component_id, probability in (
                        shockwave_probability_cache.items()):
                    value = float(probability)
                    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                        raise ValueError(
                            "Cached shockwave probabilities must be finite "
                            "and lie in [0,1].")
                    cmap[int(component_id)].shockwave_damage_prob = value
            else:
                expl_pos = encounter.position_cm
                parsed_components = []
                for comp in components:
                    st, geometry = parse_component_geometry(comp)
                    parsed_components.append((comp['id'], comp['name'], st, geometry, comp))
                for comp in components:
                    cid = comp['id']
                    cr = cmap[cid]
                    pos = comp['geometry']['position']
                    cc = np.array([pos.get('x',0) or 0, pos.get('y',0) or 0,
                                   pos.get('z',0) or 0], dtype=float)
                    dist_m = np.linalg.norm(expl_pos - cc) / 100.0
                    tnt_kg = warhead.tnt_equivalent_mass_kg
                    incident = ShockwaveModel.overpressure_sadovsky(dist_m, tnt_kg)
                    shielding_mm = line_of_sight_armor_mm(
                        expl_pos, cc, parsed_components,
                        armor_plates=self.armor_plates,
                        exclude_component_id=cid,
                    )
                    cr.overpressure_mpa = ShockwaveModel.effective_overpressure(
                        comp, incident, geometric_shielding_mm=shielding_mm)

                    eff_threshold = ShockwaveModel.armor_threshold(comp)
                    if eff_threshold is None:
                        eff_threshold = cr.overpressure_threshold
                    cr.shockwave_damage_prob = ShockwaveModel.damage_prob_shock(
                        cr.overpressure_mpa, eff_threshold)

        # ---- 综合 ----
        for cr in cmap.values():
            pf = cr.fragment_damage_prob
            ps = cr.shockwave_damage_prob
            cr.combined_damage_prob = 1.0 - (1.0 - pf) * (1.0 - ps)
            p = cr.combined_damage_prob
            if p >= 0.9:   cr.damage_level = DamageLevel.CATASTROPHIC
            elif p >= 0.7: cr.damage_level = DamageLevel.SEVERE
            elif p >= 0.4: cr.damage_level = DamageLevel.MODERATE
            elif p >= 0.1: cr.damage_level = DamageLevel.LIGHT
            else:          cr.damage_level = DamageLevel.NONE
            cr.is_damaged = p >= 0.5

        result.component_results = sorted(cmap.values(), key=lambda r: r.component_id)
        result.total_penetrations = sum(r.fragment_penetrations for r in result.component_results)
        result.damaged_count = sum(1 for r in result.component_results if r.is_damaged)

        # ---- 毁伤树 ----
        evaluator = DamageTreeEvaluator()
        comp_probs = {cr.component_id: cr.combined_damage_prob for cr in result.component_results}
        result.damage_tree = evaluator.evaluate(comp_probs)
        result.damage_tree_fragment = evaluator.evaluate({
            cr.component_id: cr.fragment_damage_prob for cr in result.component_results
        })
        result.damage_tree_shockwave = evaluator.evaluate({
            cr.component_id: cr.shockwave_damage_prob for cr in result.component_results
        })

        return result


# ============================================================================
#  毁伤树 — 概率传递
# ============================================================================

@dataclass
class DamageCondition:
    condition_type: str
    component_ids: List[int]
    logic: str = "OR"
    ratio_threshold: Optional[float] = None
    description: str = ""

@dataclass
class DamageLevelRule:
    category: DamageCategory
    level: int
    conditions: List[DamageCondition]
    logic: str = "OR"
    description: str = ""

@dataclass
class DamageTreeResult:
    K_level: int = 0; M_level: int = 0; F_level: int = 0; C_level: int = 0
    K1_prob: float = 0.0; K2_prob: float = 0.0
    M1_prob: float = 0.0; M2_prob: float = 0.0
    F1_prob: float = 0.0; F2_prob: float = 0.0
    C1_prob: float = 0.0; C2_prob: float = 0.0
    triggered_rules: List[str] = field(default_factory=list)

    def to_dict(self):
        data = {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
                for k, v in self.__dict__.items()}
        data.update({k: round(v, 4) for k, v in self.ordinal_probability_dict.items()})
        return data

    @property
    def overall_score(self) -> float:
        ordinal = self.ordinal_probability_dict
        s = (
            0.35 * 0.5 * (ordinal["K_ge1_prob"] + ordinal["K_ge2_prob"]) +
            0.25 * 0.5 * (ordinal["F_ge1_prob"] + ordinal["F_ge2_prob"]) +
            0.22 * 0.5 * (ordinal["M_ge1_prob"] + ordinal["M_ge2_prob"]) +
            0.18 * 0.5 * (ordinal["C_ge1_prob"] + ordinal["C_ge2_prob"])
        )
        return float(np.clip(s, 0, 1))

    @property
    def level_vector(self) -> np.ndarray:
        """Legacy raw rule-event vector; not an ordinal probability vector."""
        return np.array([self.K1_prob, self.K2_prob, self.M1_prob, self.M2_prob,
                         self.F1_prob, self.F2_prob, self.C1_prob, self.C2_prob])

    @property
    def ordinal_probability_dict(self) -> Dict[str, float]:
        """Return explicit ``P(level>=1/2)`` values.

        K/M/F rule-1 and rule-2 are separate physical routes, so level>=1 is
        their probabilistic OR. C thresholds are nested by construction.
        """
        result = {}
        for name, p1, p2, nested in (
            ("K", self.K1_prob, self.K2_prob, False),
            ("M", self.M1_prob, self.M2_prob, False),
            ("F", self.F1_prob, self.F2_prob, False),
            ("C", self.C1_prob, self.C2_prob, True),
        ):
            ge1 = p1 if nested else 1.0 - (1.0 - p1) * (1.0 - p2)
            ge2 = min(p2, ge1)
            result[f"{name}_ge1_prob"] = float(np.clip(ge1, 0.0, 1.0))
            result[f"{name}_ge2_prob"] = float(np.clip(ge2, 0.0, 1.0))
        return result

    @property
    def ordinal_probability_vector(self) -> np.ndarray:
        values = self.ordinal_probability_dict
        return np.array([
            values[f"{name}_ge{level}_prob"]
            for name in ("K", "M", "F", "C") for level in (1, 2)
        ], dtype=float)


def build_damage_tree_rules() -> Dict[str, DamageLevelRule]:
    rules = {}
    # K
    rules["K1"] = DamageLevelRule(DamageCategory.K, 1, [
        DamageCondition("single", [3], "OR", description="油箱引燃")], description="K1: 油箱引燃")
    rules["K2"] = DamageLevelRule(DamageCategory.K, 2, [
        DamageCondition("single", [46], "OR", description="弹药架殉爆")], description="K2: 弹药殉爆")
    # M
    rules["M1"] = DamageLevelRule(DamageCategory.M, 1, [
        DamageCondition("ratio", list(range(6,18)), "RATIO", 0.2, "负重轮≥20%损毁"),
        DamageCondition("group", [30,31], "OR", description="诱导轮损毁"),
    ], description="M1: 行走部件部分损毁")
    rules["M2"] = DamageLevelRule(DamageCategory.M, 2, [
        DamageCondition("group", [1,2,3], "OR", description="动力系统任一损毁"),
        DamageCondition("group", [4,5], "OR", description="主动轮损毁"),
        DamageCondition("ratio", list(range(32,40)), "RATIO", 0.25, "履带≥25%损毁"),
        DamageCondition("single", [58], "OR", description="驾驶员阵亡"),
    ], description="M2: 丧失机动力")
    # F
    rules["F1"] = DamageLevelRule(DamageCategory.F, 1, [
        DamageCondition("group", [50,51,52,53,54,47,48], "OR", description="次要武器损毁"),
        DamageCondition("combination", [41,42], "EXCLUSIVE", description="主观瞄毁/辅助好"),
        DamageCondition("group", [43,44], "EXCLUSIVE_OR", description="火控恰好一个故障"),
        DamageCondition("combination", [60,59], "EXCLUSIVE", description="炮长亡/车长存"),
    ], description="F1: 火力降级")
    rules["F2"] = DamageLevelRule(DamageCategory.F, 2, [
        DamageCondition("group", [45,49], "AND", description="主炮+供弹同毁"),
        DamageCondition("group", [41,42], "AND", description="主副观瞄全毁"),
        DamageCondition("group", [43,44], "AND", description="火控全毁"),
        DamageCondition("group", [59,60], "AND", description="车长炮长同亡"),
    ], description="F2: 丧失火力")
    # C
    rules["C1"] = DamageLevelRule(DamageCategory.C, 1, [
        DamageCondition("ratio", list(range(58,68)), "RATIO", 0.2, "≥20%乘员阵亡")
    ], description="C1: 20%乘员阵亡")
    rules["C2"] = DamageLevelRule(DamageCategory.C, 2, [
        DamageCondition("ratio", list(range(58,68)), "RATIO", 0.6, "≥60%乘员阵亡")
    ], description="C2: 60%乘员阵亡")
    return rules


class DamageTreeEvaluator:
    def __init__(self, rules=None):
        self.rules = rules or build_damage_tree_rules()

    def evaluate(self, comp_probs: Dict[int,float], threshold=0.5) -> DamageTreeResult:
        result = DamageTreeResult()
        for rk, rule in self.rules.items():
            prob = self._eval_rule(rule, comp_probs)
            setattr(result, f"{rk}_prob", prob)
            if prob >= threshold:
                result.triggered_rules.append(f"{rk} ({rule.description}) P={prob:.2f}")
        ordinal = result.ordinal_probability_dict
        for name in ("K", "M", "F", "C"):
            ge1 = ordinal[f"{name}_ge1_prob"]
            ge2 = ordinal[f"{name}_ge2_prob"]
            setattr(result, f"{name}_level", 2 if ge2 >= threshold else (1 if ge1 >= threshold else 0))
        return result

    def _eval_rule(self, rule, probs):
        cps = [self._eval_cond(c, probs) for c in rule.conditions]
        return self._combine(cps, rule.logic)

    def _eval_cond(self, cond, probs):
        ps = [probs.get(cid, 0.0) for cid in cond.component_ids]
        if not ps: return 0.0
        if cond.logic == "OR": return self._p_or(ps)
        elif cond.logic == "AND": return self._p_and(ps)
        elif cond.logic == "RATIO": return self._p_ratio(ps, cond.ratio_threshold or 0.5)
        elif cond.logic == "EXCLUSIVE":
            return ps[0]*(1-ps[1]) if len(ps)>=2 else (ps[0] if ps else 0)
        elif cond.logic == "EXCLUSIVE_OR":
            if len(ps)>=2: return ps[0]*(1-ps[1]) + ps[1]*(1-ps[0])
            return ps[0] if ps else 0
        return self._p_or(ps)

    @staticmethod
    def _p_or(ps, rho=0.5):
        if not ps: return 0.0
        s = 1.0
        for p in ps: s *= (1-p)
        p_indep = 1.0 - s
        p_max = max(ps)
        return rho * p_max + (1.0 - rho) * p_indep

    @staticmethod
    def _p_and(ps):
        r = 1.0
        for p in ps: r *= p
        return r

    @staticmethod
    def _p_ratio(ps, threshold):
        n = len(ps)
        if n == 0: return 0.0
        k = threshold * n
        mu = sum(ps)
        var = sum(p*(1-p) for p in ps)
        if var < 1e-6:
            return 1.0 if mu >= k else 0.0
        z = (mu - k + 0.5) / np.sqrt(var)
        # 数学上极端 z 已饱和到 0/1；截断指数自变量避免批量生成时
        # 产生无意义的 overflow warning，同时保持双精度结果不变。
        logistic_arg = float(np.clip(1.7 * z, -60.0, 60.0))
        return float(1.0 / (1.0 + np.exp(-logistic_arg)))

    @staticmethod
    def _combine(ps, logic):
        if not ps: return 0.0
        if logic == "AND": return DamageTreeEvaluator._p_and(ps)
        return DamageTreeEvaluator._p_or(ps)


# ============================================================================
#  数据加载
# ============================================================================

def load_vehicle_model(filepath: str = None) -> List[Dict]:
    """加载 vehicle_model.json"""
    if filepath is None:
        filepath = bundled_resource_path("vehicle_model.json")
    if not filepath: raise FileNotFoundError("vehicle_model.json not found")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['components']
