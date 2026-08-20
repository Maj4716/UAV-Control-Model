# -*- coding: utf-8 -*-
"""
毁伤效应空间三维可视化模块
===========================

功能:
  - 3D 散点图: 颜色映射毁伤评分
  - 装甲车轮廓线参考
  - 多弹型分面对比
  - 切面热力图
  - 交互式 HTML 导出
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, List, Any


# ============================================================================
#  颜色映射工具
# ============================================================================

# 自定义毁伤评分色阶: 绿(安全) → 黄 → 红(毁伤)
DAMAGE_COLORSCALE = [
    [0.0, "rgb(40, 167, 69)"],   # 深绿
    [0.2, "rgb(92, 184, 92)"],   # 浅绿
    [0.4, "rgb(255, 193, 7)"],   # 金黄
    [0.6, "rgb(253, 126, 20)"],  # 橙色
    [0.8, "rgb(220, 53, 69)"],   # 红色
    [1.0, "rgb(136, 14, 79)"],   # 深紫红
]


# ============================================================================
#  装甲车 3D 部件渲染
# ============================================================================

def _rot_xyz(rx_deg, ry_deg, rz_deg):
    rx, ry, rz = np.radians(rx_deg), np.radians(ry_deg), np.radians(rz_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def _box_verts(cx, cy, cz, lx, ly, lz, R=None):
    signs = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)
    pts = signs * np.array([lx / 2, ly / 2, lz / 2])
    if R is not None: pts = (R @ pts.T).T
    pts += [cx, cy, cz]
    return pts[:, 0].tolist(), pts[:, 1].tolist(), pts[:, 2].tolist()

def _cyl_surface(cx, cy, cz, radius, half_h, axis_vec, nt=20, nh=8):
    ax = axis_vec / (np.linalg.norm(axis_vec) + 1e-12)
    ref = np.array([1, 0, 0]) if abs(ax[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(ax, ref); u /= np.linalg.norm(u)
    v = np.cross(ax, u)
    theta = np.linspace(0, 2 * np.pi, nt)
    ts = np.linspace(-half_h, half_h, nh)
    xg, yg, zg = np.zeros((nh, nt)), np.zeros((nh, nt)), np.zeros((nh, nt))
    center = np.array([cx, cy, cz])
    for i, t in enumerate(ts):
        for j, th in enumerate(theta):
            pt = center + t * ax + radius * (np.cos(th) * u + np.sin(th) * v)
            xg[i, j], yg[i, j], zg[i, j] = pt
    return xg, yg, zg

BOX_I = [0, 0, 1, 1, 2, 2, 4, 4, 0, 0, 1, 1]
BOX_J = [1, 2, 2, 3, 3, 0, 5, 6, 4, 1, 5, 2]
BOX_K = [5, 6, 6, 7, 7, 4, 6, 7, 1, 5, 6, 6]

def _extruded_poly_mesh(start_x: float, end_x: float, verts_yz: np.ndarray):
    """生成 Y-Z 平面多边形沿 X 轴拉伸后的 3D 网格顶点和面索引"""
    n = len(verts_yz)
    # 前端盖顶点 (x = start_x)
    front_verts = np.column_stack([np.full(n, start_x), verts_yz])
    # 后端盖顶点 (x = end_x)
    back_verts = np.column_stack([np.full(n, end_x), verts_yz])

    vx = np.concatenate([front_verts[:, 0], back_verts[:, 0]])
    vy = np.concatenate([front_verts[:, 1], back_verts[:, 1]])
    vz = np.concatenate([front_verts[:, 2], back_verts[:, 2]])

    i, j, k = [], [], []

    # 侧边面 (每个矩形分成两个三角形)
    for idx in range(n):
        next_idx = (idx + 1) % n
        # front edge: idx, next_idx
        # back edge: n + idx, n + next_idx
        # Triangle 1
        i.append(idx)
        j.append(next_idx)
        k.append(n + idx)
        # Triangle 2
        i.append(next_idx)
        j.append(n + next_idx)
        k.append(n + idx)

    # 端盖 (简单三角剖分，假设多边形是凸的或足够规则可被扇形分割)
    # 前端盖 (法线朝-X，顶点顺序逆序保证法线正确)
    for idx in range(1, n - 1):
        i.append(0)
        j.append(idx + 1)
        k.append(idx)
    # 后端盖 (法线朝+X)
    for idx in range(1, n - 1):
        i.append(n + 0)
        j.append(n + idx)
        k.append(n + idx + 1)

    return vx, vy, vz, i, j, k

def render_vehicle_components(components: List[dict], opacity: float = 0.15) -> List[Any]:
    """将车辆部件列表转换为 Plotly 3D 对象列表 (Mesh3d/Surface)"""
    traces = []
    base_color = 'rgba(120, 130, 140, 1.0)'  # 统一的车辆部件颜色(带有透明度在Trace级别设置)

    for comp in components:
        cid = comp['id']
        geom = comp['geometry']
        pos = geom['position']
        dims = geom['dimensions']
        shape = geom['shape']
        rot = geom.get('rotation') or {}

        cx = pos.get('x', 0) or 0; cy = pos.get('y', 0) or 0; cz = pos.get('z', 0) or 0
        rx = rot.get('x', 0) or 0; ry = rot.get('y', 0) or 0; rz = rot.get('z', 0) or 0
        has_rot = abs(rx) > 0.5 or abs(ry) > 0.5 or abs(rz) > 0.5

        if shape == "长方体":
            l = dims.get('length_or_radius', 50) or 50
            w = dims.get('width', 50) or 50
            h = dims.get('height', 50) or 50
            R = _rot_xyz(rx, ry, rz) if has_rot else None
            vx, vy, vz = _box_verts(cx, cy, cz, l, w, h, R)
            traces.append(go.Mesh3d(
                x=vx, y=vy, z=vz, i=BOX_I, j=BOX_J, k=BOX_K,
                color=base_color, opacity=opacity, name=comp['name'],
                showlegend=False, hoverinfo='skip'
            ))

        elif shape == "圆柱体":
            radius = dims.get('length_or_radius', 20) or 20
            h = dims.get('height', 50) or 50
            if has_rot:
                axis_vec = _rot_xyz(rx, ry, rz) @ np.array([0, 0, 1.0])
            else:
                axis_vec = np.array([0, 0, 1.0])
            xg, yg, zg = _cyl_surface(cx, cy, cz, radius, h/2, axis_vec)
            traces.append(go.Surface(
                x=xg, y=yg, z=zg,
                colorscale=[[0, base_color], [1, base_color]], showscale=False,
                opacity=opacity, name=comp['name'],
                showlegend=False, hoverinfo='skip'
            ))
        elif shape == "拉伸多边形":
            ext_len = dims.get('extrusion_length', 100)
            start_x = pos.get('x', 0)
            end_x = start_x + ext_len
            verts_yz = np.array(geom.get('vertices_yz', []))
            if len(verts_yz) >= 3:
                vx, vy, vz, mesh_i, mesh_j, mesh_k = _extruded_poly_mesh(start_x, end_x, verts_yz)
                # 使用不同颜色区分装甲部件
                color = 'rgba(150, 110, 80, 1.0)' if "装甲" in comp['name'] else base_color
                traces.append(go.Mesh3d(
                    x=vx, y=vy, z=vz, i=mesh_i, j=mesh_j, k=mesh_k,
                    color=color, opacity=max(0.3, opacity + 0.1), name=comp['name'],
                    showlegend=False, hoverinfo='skip'
                ))

    return traces


# ============================================================================
#  核心可视化: 3D 毁伤散点图
# ============================================================================

def plot_damage_scatter_3d(
    df: pd.DataFrame,
    components: Optional[List[dict]] = None,
    color_col: str = "overall_score",
    title: str = "毁伤效应空间分布",
    point_size: int = 4,
    opacity: float = 0.7,
    show_vehicle: bool = True,
    height: int = 750,
) -> go.Figure:
    """
    在 3D 空间中绘制采样点散点图, 颜色映射毁伤效应。

    参数:
        df:           包含 x_cm, y_cm, z_cm 和 color_col 的 DataFrame
        components:   车辆部件实体列表 (若提供将绘制精确 3D 模型)
        color_col:    颜色映射列名 (默认 overall_score)
        title:        图表标题
        point_size:   散点大小
        opacity:      透明度
        show_vehicle: 是否显示装甲车模型
        height:       图表高度 (px)

    返回:
        Plotly Figure
    """
    fig = go.Figure()

    # 装甲车实体模型
    if show_vehicle and components:
        veh_traces = render_vehicle_components(components, opacity=0.15)
        for t in veh_traces:
            fig.add_trace(t)

    # 原点标记
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=8, color='gold', symbol='diamond',
                    line=dict(width=2, color='black')),
        name='目标中心',
        showlegend=True,
    ))

    # 毁伤散点
    hover_text = [
        f"位置: ({row.x_cm:.0f}, {row.y_cm:.0f}, {row.z_cm:.0f}) cm<br>"
        f"速度: ({row.vx_ms:.0f}, {row.vy_ms:.0f}, {row.vz_ms:.0f}) m/s<br>"
        f"评分: {row.overall_score:.3f}<br>"
        f"命中/穿透: {row.total_hits}/{row.total_penetrations}<br>"
        f"K{row.K_level}M{row.M_level}F{row.F_level}C{row.C_level}"
        for _, row in df.iterrows()
    ]

    fig.add_trace(go.Scatter3d(
        x=df["x_cm"], y=df["y_cm"], z=df["z_cm"],
        mode='markers',
        marker=dict(
            size=point_size,
            color=df[color_col],
            colorscale=DAMAGE_COLORSCALE,
            cmin=0, cmax=1,
            opacity=opacity,
            colorbar=dict(
                title=dict(text="毁伤评分", font=dict(size=14)),
                thickness=20,
                len=0.7,
                tickformat=".1%",
            ),
            line=dict(width=0),
        ),
        text=hover_text,
        hoverinfo='text',
        name='采样点',
        showlegend=False,
    ))

    # 半球参考面 (半透明)
    if len(df) > 0:
        r_max = np.sqrt(df["x_cm"]**2 + df["y_cm"]**2 + df["z_cm"]**2).max() * 1.05
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi / 2, 15)
        xs = r_max * np.outer(np.sin(v), np.cos(u))
        ys = r_max * np.outer(np.sin(v), np.sin(u))
        zs = r_max * np.outer(np.cos(v), np.ones_like(u))
        fig.add_trace(go.Surface(
            x=xs, y=ys, z=zs,
            colorscale=[[0, 'rgba(70,130,180,0.05)'], [1, 'rgba(70,130,180,0.05)']],
            showscale=False,
            opacity=0.1,
            name='半球边界',
            hoverinfo='skip',
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        scene=dict(
            xaxis_title='X (cm)', yaxis_title='Y (cm)', zaxis_title='Z (cm)',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        showlegend=True,
        legend=dict(x=0, y=1, font=dict(size=11)),
        margin=dict(l=0, r=0, t=50, b=0),
        height=height,
    )

    return fig


# ============================================================================
#  多弹型分面散点图
# ============================================================================

def plot_multi_projectile_scatter(
    df: pd.DataFrame,
    components: Optional[List[dict]] = None,
    color_col: str = "overall_score",
    height: int = 700,
) -> go.Figure:
    """
    按弹型名称分面的 3D 散点图。

    参数:
        df: 合并后的多弹型 DataFrame (需含 projectile_name 列)
        components: 车辆部件实体列表
    """
    proj_names = df["projectile_name"].unique()
    n = len(proj_names)

    fig = make_subplots(
        rows=1, cols=n,
        specs=[[{"type": "scatter3d"}] * n],
        subplot_titles=[name[:12] for name in proj_names],
        horizontal_spacing=0.02,
    )

    for i, name in enumerate(proj_names):
        # 加入装甲车模型
        if components:
            veh_traces = render_vehicle_components(components, opacity=0.1)
            for t in veh_traces:
                fig.add_trace(t, row=1, col=i+1)

        sub = df[df["projectile_name"] == name]
        fig.add_trace(go.Scatter3d(
            x=sub["x_cm"], y=sub["y_cm"], z=sub["z_cm"],
            mode='markers',
            marker=dict(
                size=3,
                color=sub[color_col],
                colorscale=DAMAGE_COLORSCALE,
                cmin=0, cmax=1,
                opacity=0.6,
                colorbar=dict(title="评分", len=0.5) if i == n-1 else None,
                showscale=(i == n - 1),
            ),
            name=name[:12],
            showlegend=True,
        ), row=1, col=i+1)

    fig.update_layout(height=height, margin=dict(t=40, b=10))
    return fig


# ============================================================================
#  统计概览图
# ============================================================================

def plot_score_distribution(
    df: pd.DataFrame,
    title: str = "毁伤评分分布",
) -> go.Figure:
    """毁伤评分直方图 + 箱线图"""
    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3],
                        shared_xaxes=True, vertical_spacing=0.05)

    fig.add_trace(go.Histogram(
        x=df["overall_score"], nbinsx=50,
        marker_color='rgba(220,53,69,0.6)',
        name="频数分布",
    ), row=1, col=1)

    fig.add_trace(go.Box(
        x=df["overall_score"],
        marker_color='rgba(220,53,69,0.8)',
        name="箱线图",
        boxmean=True,
    ), row=2, col=1)

    fig.update_layout(
        title=title, height=400,
        xaxis2_title="Overall Score",
        showlegend=False,
    )
    return fig


def plot_feature_correlation(
    df: pd.DataFrame,
    target_col: str = "overall_score",
) -> go.Figure:
    """特征与目标变量的相关系数条形图"""
    feature_cols = ["x_cm", "y_cm", "z_cm", "vx_ms", "vy_ms", "vz_ms",
                    "yaw_deg", "pitch_deg", "roll_deg", "speed_ms"]
    existing = [c for c in feature_cols if c in df.columns]

    corrs = df[existing].corrwith(df[target_col]).sort_values()

    colors = ['rgba(220,53,69,0.7)' if v < 0 else 'rgba(40,167,69,0.7)' for v in corrs]

    fig = go.Figure(go.Bar(
        y=corrs.index,
        x=corrs.values,
        orientation='h',
        marker_color=colors,
        text=[f"{v:.3f}" for v in corrs],
        textposition='outside',
    ))

    fig.update_layout(
        title="特征-评分相关系数",
        xaxis_title="Pearson 相关系数",
        height=400,
        margin=dict(l=100),
    )
    return fig


# ============================================================================
#  导出为交互式 HTML
# ============================================================================

def save_html(fig: go.Figure, path: str) -> str:
    """保存 Plotly 图表为独立 HTML 文件"""
    fig.write_html(path, include_plotlyjs='cdn')
    print(f"[可视化] 已保存 → {path}")
    return path
