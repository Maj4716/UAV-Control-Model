#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
装甲车 JSON 模型学术风格可视化脚本

默认行为：
1. 自动扫描当前目录下包含 components 字段的 JSON 文件；
2. 为每个模型生成 4 视角科研风格可视化；
3. 输出交互式 HTML，若环境支持 kaleido，则同时导出 PNG。

用法示例：
    python academic_vehicle_vis.py
    python academic_vehicle_vis.py vehicle_model.json
    python academic_vehicle_vis.py vehicle_model.json vehicle_model_1.json --output-dir report\\vehicle_figures
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


EDGE_COLOR = "#21313f"
LABEL_COLOR = "#1b2630"
GRID_COLOR = "rgba(28, 52, 70, 0.14)"
ZERO_LINE_COLOR = "rgba(28, 52, 70, 0.32)"
BACKGROUND_COLOR = "#fbfcfe"
PLANE_COLOR = "rgba(243, 246, 250, 1.0)"
LIGHTING = dict(ambient=0.58, diffuse=0.72, specular=0.08, roughness=0.96, fresnel=0.02)
LIGHT_POSITION = dict(x=1500, y=-1200, z=1200)

DISPLAY_STYLES = {
    "armor": {
        "label": "外装甲",
        "palette": ["#d9eef7", "#e7f4fb", "#dbeaf5"],
        "opacity": 0.16,
        "edge_color": "rgba(108, 143, 167, 0.50)",
        "edge_width": 1.8,
        "legend_color": "#dceef7",
    },
    "powertrain": {
        "label": "动力系统",
        "palette": ["#ef8354", "#f29f67", "#d96c47"],
        "opacity": 0.93,
        "edge_color": "rgba(139, 65, 41, 0.82)",
        "edge_width": 2.4,
        "legend_color": "#ef8354",
    },
    "running_gear": {
        "label": "轮系部件",
        "palette": ["#00a6a6", "#2ec4b6", "#38b2ac", "#1c7c7d"],
        "opacity": 0.90,
        "edge_color": "rgba(20, 88, 88, 0.80)",
        "edge_width": 2.2,
        "legend_color": "#2ec4b6",
    },
    "track": {
        "label": "履带组件",
        "palette": ["#6a994e", "#7fb069", "#90be6d", "#5f8f3f"],
        "opacity": 0.91,
        "edge_color": "rgba(64, 97, 38, 0.82)",
        "edge_width": 2.2,
        "legend_color": "#7fb069",
    },
    "suspension": {
        "label": "悬挂机构",
        "palette": ["#a7c957", "#8fb339", "#b5c99a"],
        "opacity": 0.91,
        "edge_color": "rgba(89, 105, 36, 0.82)",
        "edge_width": 2.2,
        "legend_color": "#a7c957",
    },
    "sensor": {
        "label": "观瞄火控",
        "palette": ["#4c6ef5", "#4895ef", "#3f8efc", "#5c7cfa"],
        "opacity": 0.93,
        "edge_color": "rgba(33, 65, 155, 0.82)",
        "edge_width": 2.2,
        "legend_color": "#4c6ef5",
    },
    "weapon": {
        "label": "武器系统",
        "palette": ["#d1495b", "#c44536", "#b23a48", "#e76f51"],
        "opacity": 0.93,
        "edge_color": "rgba(110, 33, 42, 0.84)",
        "edge_width": 2.3,
        "legend_color": "#d1495b",
    },
    "ammo": {
        "label": "弹药与供弹",
        "palette": ["#f4a261", "#ffb703", "#e9c46a", "#f6bd60"],
        "opacity": 0.92,
        "edge_color": "rgba(148, 92, 24, 0.84)",
        "edge_width": 2.3,
        "legend_color": "#ffb703",
    },
    "crew": {
        "label": "乘员与载员",
        "palette": ["#9d4edd", "#b185db", "#8e7dbe", "#a06cd5"],
        "opacity": 0.90,
        "edge_color": "rgba(82, 50, 130, 0.82)",
        "edge_width": 2.1,
        "legend_color": "#9d4edd",
    },
    "comms": {
        "label": "通信设备",
        "palette": ["#00b4d8", "#48cae4", "#219ebc"],
        "opacity": 0.91,
        "edge_color": "rgba(11, 98, 115, 0.82)",
        "edge_width": 2.1,
        "legend_color": "#00b4d8",
    },
    "other": {
        "label": "辅助设备",
        "palette": ["#577590", "#7d8597", "#6c757d"],
        "opacity": 0.88,
        "edge_color": "rgba(64, 76, 92, 0.78)",
        "edge_width": 2.0,
        "legend_color": "#7d8597",
    },
}

LABEL_WEIGHTS = {
    "powertrain": 1.00,
    "weapon": 0.96,
    "ammo": 0.95,
    "sensor": 0.94,
    "crew": 0.84,
    "comms": 0.78,
    "other": 0.72,
    "suspension": 0.68,
    "running_gear": 0.56,
    "track": 0.52,
    "armor": 0.24,
}


@dataclass
class ComponentMesh:
    component_id: int
    name: str
    kind: str
    display_category: str
    thickness_mm: float
    vulnerable_ratio: float
    overpressure_threshold: float
    volume_proxy: float
    centroid: np.ndarray
    vertices: np.ndarray
    i: List[int]
    j: List[int]
    k: List[int]
    edge_segments: List[np.ndarray]


def rotation_matrix_xyz(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    rx_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    return rz_m @ ry_m @ rx_m


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm <= 1e-12:
        return v
    return v / norm


def orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = normalize(axis)
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = normalize(np.cross(axis, ref))
    v = normalize(np.cross(axis, u))
    return u, v


def component_kind(geometry: dict) -> str:
    if geometry.get("vertices_yz"):
        return "extruded_polygon"

    dims = geometry.get("dimensions", {})
    width = dims.get("width")
    shape = str(geometry.get("shape", "")).lower()

    if width is None:
        return "cylinder"
    if "poly" in shape:
        return "extruded_polygon"
    if "cyl" in shape:
        return "cylinder"
    return "box"


def polygon_area(points_2d: np.ndarray) -> float:
    if len(points_2d) < 3:
        return 0.0
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def rgba_with_alpha(color: str, alpha: float) -> str:
    if not color.startswith("rgb("):
        return color
    channel_text = color[4:-1]
    return f"rgba({channel_text}, {alpha})"


def classify_display_category(name: str) -> str:
    if "装甲" in name:
        return "armor"
    if any(token in name for token in ["发动机", "传动机构", "油箱"]):
        return "powertrain"
    if "履带" in name:
        return "track"
    if any(token in name for token in ["主动轮", "负重轮", "托带轮", "诱导轮"]):
        return "running_gear"
    if "悬挂" in name:
        return "suspension"
    if any(token in name for token in ["测距", "观瞄", "火控"]):
        return "sensor"
    if any(token in name for token in ["主炮", "机炮", "烟幕"]):
        return "weapon"
    if any(token in name for token in ["弹药", "供弹"]):
        return "ammo"
    if any(token in name for token in ["通讯", "无线电", "天线"]):
        return "comms"
    if any(token in name for token in ["驾驶员", "车长", "炮长", "步兵"]):
        return "crew"
    return "other"


def stable_variant_index(name: str, component_id: int, palette_len: int) -> int:
    signature = sum(ord(char) for char in name) + 17 * component_id
    return signature % max(1, palette_len)


def component_visual_style(component: ComponentMesh) -> dict:
    style = DISPLAY_STYLES[component.display_category].copy()
    palette = style["palette"]
    style["fill_color"] = palette[stable_variant_index(component.name, component.component_id, len(palette))]

    if component.display_category == "armor":
        if component.name in {"上装甲", "下装甲"}:
            style["opacity"] = 0.11
        elif component.name == "炮台装甲":
            style["opacity"] = 0.16
        else:
            style["opacity"] = 0.14

    return style


def make_box_mesh(center: np.ndarray, lengths: np.ndarray, rotation: np.ndarray | None) -> tuple[np.ndarray, List[int], List[int], List[int], List[np.ndarray], float]:
    hx, hy, hz = lengths / 2.0
    vertices = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=float,
    )
    if rotation is not None:
        vertices = (rotation @ vertices.T).T
    vertices = vertices + center

    faces = [
        (0, 1, 2), (0, 2, 3),
        (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4),
        (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6),
        (3, 0, 4), (3, 4, 7),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    edge_segments = [vertices[[a, b]] for a, b in edges]
    volume = float(lengths[0] * lengths[1] * lengths[2])
    i, j, k = zip(*faces)
    return vertices, list(i), list(j), list(k), edge_segments, volume


def make_cylinder_mesh(center: np.ndarray, radius: float, height: float, axis: np.ndarray, segments: int = 36) -> tuple[np.ndarray, List[int], List[int], List[int], List[np.ndarray], float]:
    axis = normalize(axis)
    u, v = orthonormal_basis(axis)
    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    circle = np.array([math.cos(t) * u + math.sin(t) * v for t in theta])

    top_center = center + axis * (height / 2.0)
    bottom_center = center - axis * (height / 2.0)
    top_ring = top_center + radius * circle
    bottom_ring = bottom_center + radius * circle

    vertices = np.vstack([top_ring, bottom_ring, top_center[None, :], bottom_center[None, :]])
    top_idx = 2 * segments
    bottom_idx = 2 * segments + 1

    i: List[int] = []
    j: List[int] = []
    k: List[int] = []

    for idx in range(segments):
        nxt = (idx + 1) % segments
        i.extend([idx, nxt, top_idx, bottom_idx])
        j.extend([nxt, segments + nxt, idx, segments + nxt])
        k.extend([segments + idx, segments + idx, nxt, segments + idx])

    edge_segments: List[np.ndarray] = [np.vstack([top_ring, top_ring[0]]), np.vstack([bottom_ring, bottom_ring[0]])]
    stride = max(1, segments // 6)
    for idx in range(0, segments, stride):
        edge_segments.append(np.vstack([top_ring[idx], bottom_ring[idx]]))

    volume = float(np.pi * radius * radius * height)
    return vertices, i, j, k, edge_segments, volume


def make_extruded_polygon_mesh(start_x: float, end_x: float, verts_yz: np.ndarray) -> tuple[np.ndarray, List[int], List[int], List[int], List[np.ndarray], float]:
    count = len(verts_yz)
    front = np.column_stack([np.full(count, start_x), verts_yz])
    back = np.column_stack([np.full(count, end_x), verts_yz])
    vertices = np.vstack([front, back])

    i: List[int] = []
    j: List[int] = []
    k: List[int] = []

    for idx in range(count):
        nxt = (idx + 1) % count
        i.extend([idx, nxt])
        j.extend([nxt, count + nxt])
        k.extend([count + idx, count + idx])

    for idx in range(1, count - 1):
        i.append(0)
        j.append(idx)
        k.append(idx + 1)
        i.append(count)
        j.append(count + idx + 1)
        k.append(count + idx)

    edge_segments: List[np.ndarray] = [np.vstack([front, front[0]]), np.vstack([back, back[0]])]
    for idx in range(count):
        edge_segments.append(np.vstack([front[idx], back[idx]]))

    area = polygon_area(verts_yz)
    volume = float(abs(end_x - start_x) * area)
    return vertices, i, j, k, edge_segments, volume


def build_component_mesh(component: dict) -> ComponentMesh:
    geom = component["geometry"]
    dims = geom.get("dimensions", {})
    pos = geom.get("position", {})
    rot = geom.get("rotation", {})
    center = np.array(
        [
            float(pos.get("x", 0.0) or 0.0),
            float(pos.get("y", 0.0) or 0.0),
            float(pos.get("z", 0.0) or 0.0),
        ],
        dtype=float,
    )
    rotation = rotation_matrix_xyz(
        float(rot.get("x", 0.0) or 0.0),
        float(rot.get("y", 0.0) or 0.0),
        float(rot.get("z", 0.0) or 0.0),
    )

    kind = component_kind(geom)
    if kind == "box":
        lengths = np.array(
            [
                float(dims.get("length_or_radius", 1.0) or 1.0),
                float(dims.get("width", 1.0) or 1.0),
                float(dims.get("height", 1.0) or 1.0),
            ],
            dtype=float,
        )
        vertices, i, j, k, edges, volume = make_box_mesh(center, lengths, rotation)
    elif kind == "cylinder":
        radius = float(dims.get("length_or_radius", 1.0) or 1.0)
        height = float(dims.get("height", 1.0) or 1.0)
        axis = rotation @ np.array([0.0, 0.0, 1.0], dtype=float)
        vertices, i, j, k, edges, volume = make_cylinder_mesh(center, radius, height, axis)
    else:
        extrusion_length = float(dims.get("extrusion_length", 1.0) or 1.0)
        start_x = float(pos.get("x", 0.0) or 0.0)
        end_x = start_x + extrusion_length
        verts_yz = np.asarray(geom.get("vertices_yz", []), dtype=float)
        if verts_yz.ndim != 2 or verts_yz.shape[0] < 3:
            raise ValueError(f"Component {component.get('id')} 缺少有效的 vertices_yz 数据。")
        vertices, i, j, k, edges, volume = make_extruded_polygon_mesh(start_x, end_x, verts_yz)

    material = component.get("material", {})
    thickness = float(material.get("equivalent_thickness", 0.0) or 0.0)
    vulnerable_ratio = float(material.get("vulnerable_area_ratio", 0.0) or 0.0)
    overpressure_threshold = float(material.get("overpressure_threshold", 0.0) or 0.0)
    display_category = classify_display_category(str(component["name"]))

    return ComponentMesh(
        component_id=int(component["id"]),
        name=str(component["name"]),
        kind=kind,
        display_category=display_category,
        thickness_mm=thickness,
        vulnerable_ratio=vulnerable_ratio,
        overpressure_threshold=overpressure_threshold,
        volume_proxy=volume,
        centroid=np.mean(vertices, axis=0),
        vertices=vertices,
        i=i,
        j=j,
        k=k,
        edge_segments=edges,
    )


def find_model_files(paths: Sequence[str]) -> List[Path]:
    candidates: List[Path] = []
    if paths:
        candidates = [Path(p).resolve() for p in paths]
    else:
        candidates = sorted(Path.cwd().glob("*.json"))

    valid_files: List[Path] = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict) and isinstance(data.get("components"), list):
                valid_files.append(path)
        except Exception:
            continue

    return valid_files


def collect_outline_segments(components: Iterable[ComponentMesh]) -> tuple[List[float], List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    for comp in components:
        for seg in comp.edge_segments:
            xs.extend(seg[:, 0].tolist())
            ys.extend(seg[:, 1].tolist())
            zs.extend(seg[:, 2].tolist())
            xs.append(None)
            ys.append(None)
            zs.append(None)
    return xs, ys, zs


def extent_box_segments(min_xyz: np.ndarray, max_xyz: np.ndarray) -> tuple[List[float], List[float], List[float]]:
    x0, y0, z0 = min_xyz
    x1, y1, z1 = max_xyz
    vertices = np.array(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ],
        dtype=float,
    )
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    for a, b in edges:
        xs.extend([vertices[a, 0], vertices[b, 0], None])
        ys.extend([vertices[a, 1], vertices[b, 1], None])
        zs.extend([vertices[a, 2], vertices[b, 2], None])
    return xs, ys, zs


def shape_label(kind: str) -> str:
    mapping = {
        "box": "长方体",
        "cylinder": "圆柱体",
        "extruded_polygon": "拉伸多边形",
    }
    return mapping.get(kind, kind)


def make_scene_traces(
    components: Sequence[ComponentMesh],
    scene_name: str,
    label_count: int,
    extent_min: np.ndarray,
    extent_max: np.ndarray,
) -> List[go.BaseTraceType]:
    traces: List[go.BaseTraceType] = []
    render_sequence = sorted(components, key=lambda comp: comp.display_category == "armor")

    for comp in render_sequence:
        style = component_visual_style(comp)
        fill = go.Mesh3d(
            x=comp.vertices[:, 0],
            y=comp.vertices[:, 1],
            z=comp.vertices[:, 2],
            i=comp.i,
            j=comp.j,
            k=comp.k,
            color=style["fill_color"],
            opacity=style["opacity"],
            flatshading=False,
            lighting=LIGHTING,
            lightposition=LIGHT_POSITION,
            name=comp.name,
            legendgroup=comp.display_category,
            showlegend=False,
            hovertemplate=(
                f"<b>{comp.name}</b><br>"
                f"ID: {comp.component_id}<br>"
                f"展示类别: {style['label']}<br>"
                f"几何体: {shape_label(comp.kind)}<br>"
                f"等效厚度: {comp.thickness_mm:.2f} mm<br>"
                f"脆弱面积比: {comp.vulnerable_ratio:.2f}<br>"
                f"超压阈值: {comp.overpressure_threshold:.2f} MPa"
                "<extra></extra>"
            ),
            scene=scene_name,
        )
        traces.append(fill)

        edge_x: List[float] = []
        edge_y: List[float] = []
        edge_z: List[float] = []
        for seg in comp.edge_segments:
            edge_x.extend(seg[:, 0].tolist())
            edge_y.extend(seg[:, 1].tolist())
            edge_z.extend(seg[:, 2].tolist())
            edge_x.append(None)
            edge_y.append(None)
            edge_z.append(None)
        traces.append(
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                line=dict(color=style["edge_color"], width=style["edge_width"]),
                opacity=0.82,
                hoverinfo="skip",
                showlegend=False,
                scene=scene_name,
            )
        )

    if label_count > 0:
        ranked = sorted(
            components,
            key=lambda item: LABEL_WEIGHTS.get(item.display_category, 0.6) * item.volume_proxy,
            reverse=True,
        )[:label_count]
        traces.append(
            go.Scatter3d(
                x=[comp.centroid[0] for comp in ranked],
                y=[comp.centroid[1] for comp in ranked],
                z=[comp.centroid[2] for comp in ranked],
                mode="text",
                text=[comp.name for comp in ranked],
                textposition="top center",
                textfont=dict(size=12, color=LABEL_COLOR, family="Times New Roman, SimSun, serif"),
                hoverinfo="skip",
                showlegend=False,
                scene=scene_name,
            )
        )

    extent_x, extent_y, extent_z = extent_box_segments(extent_min, extent_max)
    traces.append(
        go.Scatter3d(
            x=extent_x,
            y=extent_y,
            z=extent_z,
            mode="lines",
            line=dict(color="rgba(52, 79, 98, 0.45)", width=2, dash="dot"),
            hoverinfo="skip",
            showlegend=False,
            scene=scene_name,
        )
    )

    return traces


def aspect_ratio_from_extent(min_xyz: np.ndarray, max_xyz: np.ndarray) -> dict:
    span = np.maximum(max_xyz - min_xyz, 1.0)
    scale = span / span.max()
    return dict(x=float(scale[0]), y=float(scale[1]), z=float(scale[2]))


def scene_layout(axis_title: str, aspect_ratio: dict, camera: dict) -> dict:
    axis_template = dict(
        showbackground=True,
        backgroundcolor=PLANE_COLOR,
        gridcolor=GRID_COLOR,
        zerolinecolor=ZERO_LINE_COLOR,
        gridwidth=1,
        zerolinewidth=1.3,
        showspikes=False,
        ticks="outside",
        tickfont=dict(size=10),
    )
    return dict(
        camera=camera,
        camera_projection=dict(type="orthographic"),
        aspectmode="manual",
        aspectratio=aspect_ratio,
        xaxis={**axis_template, "title": dict(text="X / cm", font=dict(size=13))},
        yaxis={**axis_template, "title": dict(text="Y / cm", font=dict(size=13))},
        zaxis={**axis_template, "title": dict(text="Z / cm", font=dict(size=13))},
        dragmode="orbit",
        annotations=[],
    )


def build_summary(metadata: dict, components: Sequence[ComponentMesh], extent_min: np.ndarray, extent_max: np.ndarray, path: Path) -> str:
    counts = {"box": 0, "cylinder": 0, "extruded_polygon": 0}
    for comp in components:
        counts[comp.kind] = counts.get(comp.kind, 0) + 1

    thicknesses = np.array([comp.thickness_mm for comp in components], dtype=float)
    vulnerable = np.array([comp.vulnerable_ratio for comp in components], dtype=float)
    span = extent_max - extent_min

    created_time = metadata.get("created_time", "未知")
    source_file = metadata.get("source_file", "未知")
    version = metadata.get("version", "未知")

    return (
        f"<b>数据文件</b>: {path.name}<br>"
        f"<b>版本</b>: {version} &nbsp;&nbsp; <b>源文件</b>: {source_file}<br>"
        f"<b>创建时间</b>: {created_time}<br>"
        f"<b>部件数量</b>: {len(components)}<br>"
        f"<b>几何构成</b>: 长方体 {counts['box']} / 圆柱体 {counts['cylinder']} / 拉伸多边形 {counts['extruded_polygon']}<br>"
        f"<b>展示策略</b>: 外装甲采用浅色高透处理，内部部件按功能类别分色展示<br>"
        f"<b>厚度统计</b>: {thicknesses.min():.1f} - {thicknesses.max():.1f} mm，均值 {thicknesses.mean():.1f} mm<br>"
        f"<b>脆弱面积比</b>: 均值 {vulnerable.mean():.2f}<br>"
        f"<b>空间包络</b>: {span[0]:.1f} × {span[1]:.1f} × {span[2]:.1f} cm"
    )


def make_legend_traces(scene_name: str) -> List[go.BaseTraceType]:
    legend_order = [
        "armor",
        "powertrain",
        "running_gear",
        "track",
        "suspension",
        "sensor",
        "weapon",
        "ammo",
        "comms",
        "crew",
        "other",
    ]
    traces: List[go.BaseTraceType] = []
    for key in legend_order:
        style = DISPLAY_STYLES[key]
        traces.append(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker=dict(size=8, color=style["legend_color"], opacity=0.95),
                name=style["label"],
                legendgroup=key,
                showlegend=True,
                hoverinfo="skip",
                scene=scene_name,
            )
        )
    return traces


def build_figure(model_path: Path, label_count: int) -> go.Figure:
    with model_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    components_raw = data["components"]
    components = [build_component_mesh(comp) for comp in components_raw]

    all_vertices = np.vstack([comp.vertices for comp in components])
    extent_min = all_vertices.min(axis=0)
    extent_max = all_vertices.max(axis=0)

    padding = np.maximum((extent_max - extent_min) * 0.08, 20.0)
    extent_min = extent_min - padding
    extent_max = extent_max + padding

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}], [{"type": "scene"}, {"type": "scene"}]],
        horizontal_spacing=0.04,
        vertical_spacing=0.08,
        subplot_titles=("三维透视", "俯视投影", "侧视投影", "正视投影"),
    )

    scene_names = ["scene", "scene2", "scene3", "scene4"]
    scene_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    label_counts = [label_count, 0, 0, 0]

    for idx, (scene_name, position) in enumerate(zip(scene_names, scene_positions)):
        traces = make_scene_traces(
            components=components,
            scene_name=scene_name,
            label_count=label_counts[idx],
            extent_min=extent_min,
            extent_max=extent_max,
        )
        for trace in traces:
            fig.add_trace(trace, row=position[0], col=position[1])

    for trace in make_legend_traces("scene"):
        fig.add_trace(trace, row=1, col=1)

    aspect_ratio = aspect_ratio_from_extent(extent_min, extent_max)
    cameras = [
        dict(eye=dict(x=1.85, y=-1.65, z=1.20), up=dict(x=0, y=0, z=1)),
        dict(eye=dict(x=0.0, y=0.0, z=2.8), up=dict(x=0, y=1, z=0)),
        dict(eye=dict(x=2.8, y=0.0, z=0.40), up=dict(x=0, y=0, z=1)),
        dict(eye=dict(x=0.0, y=-2.8, z=0.40), up=dict(x=0, y=0, z=1)),
    ]

    fig.update_layout(
        title=dict(
            text=f"装甲车三维结构展示<br><sup>{model_path.stem} · Academic Visualization</sup>",
            x=0.5,
            y=0.97,
            xanchor="center",
            yanchor="top",
            font=dict(size=24, family="Times New Roman, SimSun, serif", color="#172531"),
        ),
        font=dict(family="Times New Roman, SimSun, serif", size=12, color="#1c2d39"),
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        margin=dict(l=20, r=36, t=120, b=30),
        width=1600,
        height=1180,
        showlegend=True,
        legend=dict(
            x=0.985,
            y=0.78,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.84)",
            bordercolor="rgba(46, 73, 92, 0.18)",
            borderwidth=1,
            font=dict(size=11, family="Times New Roman, SimSun, serif", color="#20303d"),
            itemsizing="constant",
        ),
        annotations=[
            dict(
                text=build_summary(data.get("metadata", {}), components, extent_min, extent_max, model_path),
                x=0.5,
                y=1.035,
                xref="paper",
                yref="paper",
                xanchor="center",
                yanchor="top",
                align="left",
                showarrow=False,
                bordercolor="rgba(46, 73, 92, 0.22)",
                borderwidth=1,
                borderpad=10,
                bgcolor="rgba(255,255,255,0.82)",
                font=dict(size=12, family="Times New Roman, SimSun, serif", color="#20303d"),
            )
        ],
    )

    scene_updates = {}
    for idx, camera in enumerate(cameras, start=1):
        key = "scene" if idx == 1 else f"scene{idx}"
        scene_updates[key] = scene_layout(key, aspect_ratio, camera)
        scene_updates[key]["xaxis"]["range"] = [float(extent_min[0]), float(extent_max[0])]
        scene_updates[key]["yaxis"]["range"] = [float(extent_min[1]), float(extent_max[1])]
        scene_updates[key]["zaxis"]["range"] = [float(extent_min[2]), float(extent_max[2])]

    fig.update_layout(**scene_updates)
    return fig


def export_figure(fig: go.Figure, model_path: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"{model_path.stem}_academic_visualization.html"
    fig.write_html(
        str(html_path),
        include_plotlyjs="cdn",
        full_html=True,
        config={"displaylogo": False, "responsive": True, "scrollZoom": True},
    )

    exported = [html_path]
    png_path = output_dir / f"{model_path.stem}_academic_visualization.png"
    try:
        fig.write_image(str(png_path), scale=2.0)
        exported.append(png_path)
    except Exception:
        pass
    return exported


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="装甲车 JSON 模型学术风格可视化脚本")
    parser.add_argument("inputs", nargs="*", help="待可视化的 JSON 文件；省略时自动扫描当前目录")
    parser.add_argument(
        "--output-dir",
        default="report/academic_vehicle_vis",
        help="输出目录，默认 report/academic_vehicle_vis",
    )
    parser.add_argument(
        "--labels",
        type=int,
        default=10,
        help="主视图中显示标签的部件数量，默认 10",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_files = find_model_files(args.inputs)
    if not model_files:
        raise SystemExit("未找到包含 components 字段的装甲车 JSON 文件。")

    output_dir = Path(args.output_dir).resolve()
    for model_path in model_files:
        fig = build_figure(model_path, label_count=max(0, args.labels))
        exported = export_figure(fig, model_path, output_dir)
        print(f"[OK] {model_path.name}")
        for path in exported:
            print(f"  -> {path}")


if __name__ == "__main__":
    main()
