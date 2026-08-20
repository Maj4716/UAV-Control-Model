# -*- coding: utf-8 -*-
"""
巡飞弹毁伤评估可视化系统 v2
运行: streamlit run vis_app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json

from loitering_munition_damage_twin.simulation.engine import (
    EncounterCondition, DamageEngine, Projectile, Warhead, FragmentBed,
    create_small_loitering_munition, create_medium_loitering_munition,
    create_medium_rear_det, create_heavy_loitering_munition,
    load_vehicle_model, load_armor_plates, generate_fragment_field,
    VehicleDamageResult, DamageTreeResult,
    GurneyModel, TaylorAngleModel, DetonationPoint,
)

# ---- Page config ----
st.set_page_config(page_title="巡飞弹毁伤评估系统", page_icon="🎯",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
.main-header { font-size:2rem; font-weight:700; text-align:center; color:#1a1a2e; margin-bottom:0.3rem; }
.sub-header  { font-size:1rem; color:#666; text-align:center; margin-bottom:1.5rem; }
.kmfc-card   { padding:0.8rem; border-radius:8px; margin:0.3rem 0; text-align:center; font-weight:600; }
.score-box   { font-size:2.5rem; font-weight:800; text-align:center; padding:1rem; border-radius:12px; margin:0.5rem 0; }
</style>""", unsafe_allow_html=True)


# ---- Cached data ----
@st.cache_data
def cached_load_components():
    return load_vehicle_model()

@st.cache_data
def cached_load_armor():
    return load_armor_plates()


# ============================================================================
#  Sidebar
# ============================================================================

def render_sidebar():
    st.sidebar.markdown("## ⚙️ 仿真参数")

    # -- 弹型 --
    st.sidebar.markdown("### 🚀 弹型")
    pc = st.sidebar.selectbox("选择弹型", [
        "中型巡飞弹 (前端起爆)", "中型巡飞弹 (后端起爆)",
        "小型巡飞弹 (前端起爆)", "大型巡飞弹 (前端起爆)", "自定义"])

    if   pc == "中型巡飞弹 (前端起爆)": proj = create_medium_loitering_munition()
    elif pc == "中型巡飞弹 (后端起爆)": proj = create_medium_rear_det()
    elif pc == "小型巡飞弹 (前端起爆)": proj = create_small_loitering_munition()
    elif pc == "大型巡飞弹 (前端起爆)": proj = create_heavy_loitering_munition()
    else:
        st.sidebar.markdown("#### 自定义战斗部")
        c1, c2 = st.sidebar.columns(2)
        with c1:
            charge_kg = st.number_input("装药量(kg)", 0.5, 20.0, 3.0, 0.5)
            frag_count = st.number_input("破片数量", 20, 1000, 150, 10)
        with c2:
            frag_mass = st.number_input("单片质量(g)", 1.0, 50.0, 8.0, 1.0)
            det_pt = st.selectbox("起爆点", ["前端","后端","中心"])
        dp = {"前端":DetonationPoint.FRONT,"后端":DetonationPoint.REAR,
              "中心":DetonationPoint.CENTER}[det_pt]
        proj = Projectile(
            name=f"自定义({charge_kg}kg/{frag_count}x{frag_mass}g)",
            warhead=Warhead(charge_mass_kg=charge_kg, detonation_point=dp,
                fragment_bed=FragmentBed(total_count=int(frag_count),
                    single_mass_g=frag_mass, num_rings=max(4,int(frag_count)//12))))

    # -- 交汇条件 --
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 交汇条件 (7 DOF)")

    preset = st.sidebar.selectbox("快速预设", [
        "自定义","垂直俯冲(顶部)","45°俯冲(侧面)",
        "低角度(前方)","近距侧面","后方攻击"])
    presets = {
        "垂直俯冲(顶部)":  dict(dx=0,dy=0,dz=200,yaw=0,pitch=-90,roll=0,vel=100),
        "45°俯冲(侧面)":   dict(dx=200,dy=0,dz=200,yaw=90,pitch=-45,roll=0,vel=100),
        "低角度(前方)":     dict(dx=0,dy=-400,dz=50,yaw=0,pitch=-15,roll=0,vel=120),
        "近距侧面":        dict(dx=250,dy=0,dz=80,yaw=90,pitch=-10,roll=0,vel=100),
        "后方攻击":        dict(dx=0,dy=400,dz=150,yaw=180,pitch=-30,roll=0,vel=100),
    }

    if preset != "自定义":
        p = presets[preset]
        dx,dy,dz = p['dx'],p['dy'],p['dz']
        yaw,pitch,roll,vel = p['yaw'],p['pitch'],p['roll'],p['vel']
    else:
        c1,c2,c3 = st.sidebar.columns(3)
        with c1: dx = st.number_input("dx(cm)", -500.0, 500.0, 0.0, 10.0)
        with c2: dy = st.number_input("dy(cm)", -500.0, 500.0, 0.0, 10.0)
        with c3: dz = st.number_input("dz(cm)", 0.0, 800.0, 200.0, 10.0)
        c1,c2,c3 = st.sidebar.columns(3)
        with c1: yaw   = st.number_input("偏航(°)", -180.0, 180.0, 0.0, 5.0)
        with c2: pitch  = st.number_input("俯仰(°)", -90.0, 0.0, -90.0, 5.0)
        with c3: roll   = st.number_input("滚转(°)", -180.0, 180.0, 0.0, 5.0)
        vel = st.sidebar.slider("末速度(m/s)", 50, 300, 100, 10)

    enc = EncounterCondition.from_speed_and_attitude(
        dx=dx, dy=dy, dz=dz, yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll, speed=vel)

    # -- 仿真选项 --
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 仿真选项")
    enable_frag  = st.sidebar.checkbox("启用破片毁伤", True)
    enable_shock = st.sidebar.checkbox("启用冲击波毁伤", True)
    rng_seed = st.sidebar.number_input("随机种子", 0, 9999, 42, 1)

    run = st.sidebar.button("🚀 执行仿真", type="primary", use_container_width=True)

    return proj, enc, enable_frag, enable_shock, int(rng_seed), run


# ============================================================================
#  3D helpers
# ============================================================================

def _rot_xyz(rx_deg, ry_deg, rz_deg):
    rx,ry,rz = np.radians(rx_deg), np.radians(ry_deg), np.radians(rz_deg)
    cx,sx = np.cos(rx),np.sin(rx)
    cy,sy = np.cos(ry),np.sin(ry)
    cz,sz = np.cos(rz),np.sin(rz)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rz @ Ry @ Rx

def _prob_color(p):
    p = max(0.0, min(1.0, p))
    if p < 0.5:
        r_ = int(255*p*2); g_ = int(200+55*(1-p*2)); b_ = int(100*(1-p*2))
    else:
        r_ = 255; g_ = int(200*max(0,1-(p-0.5)*2)); b_ = 0
    return f'rgba({r_},{g_},{b_},0.75)'

def _box_verts(cx,cy,cz, lx,ly,lz, R=None):
    signs = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                       [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], dtype=float)
    pts = signs * np.array([lx/2,ly/2,lz/2])
    if R is not None: pts = (R @ pts.T).T
    pts += [cx,cy,cz]
    return pts[:,0].tolist(), pts[:,1].tolist(), pts[:,2].tolist()

def _cyl_surface(cx,cy,cz, radius, half_h, axis_vec, nt=20, nh=8):
    ax = axis_vec / (np.linalg.norm(axis_vec)+1e-12)
    ref = np.array([1,0,0]) if abs(ax[0])<0.9 else np.array([0,1,0])
    u = np.cross(ax,ref); u /= np.linalg.norm(u)
    v = np.cross(ax,u)
    theta = np.linspace(0,2*np.pi,nt)
    ts = np.linspace(-half_h,half_h,nh)
    xg,yg,zg = np.zeros((nh,nt)),np.zeros((nh,nt)),np.zeros((nh,nt))
    center = np.array([cx,cy,cz])
    for i,t in enumerate(ts):
        for j,th in enumerate(theta):
            pt = center + t*ax + radius*(np.cos(th)*u + np.sin(th)*v)
            xg[i,j],yg[i,j],zg[i,j] = pt
    return xg,yg,zg

def _extruded_poly_mesh(start_x: float, end_x: float, verts_yz: np.ndarray):
    """生成 Y-Z 平面多边形沿 X 轴拉伸后的 3D 网格顶点和面索引"""
    n = len(verts_yz)
    front_verts = np.column_stack([np.full(n, start_x), verts_yz])
    back_verts = np.column_stack([np.full(n, end_x), verts_yz])

    vx = np.concatenate([front_verts[:, 0], back_verts[:, 0]])
    vy = np.concatenate([front_verts[:, 1], back_verts[:, 1]])
    vz = np.concatenate([front_verts[:, 2], back_verts[:, 2]])

    i, j, k = [], [], []
    for idx in range(n):
        next_idx = (idx + 1) % n
        i.append(idx)
        j.append(next_idx)
        k.append(n + idx)
        i.append(next_idx)
        j.append(n + next_idx)
        k.append(n + idx)

    for idx in range(1, n - 1):
        i.append(0); j.append(idx + 1); k.append(idx)
    for idx in range(1, n - 1):
        i.append(n + 0); j.append(n + idx); k.append(n + idx + 1)

    return vx, vy, vz, i, j, k


# ============================================================================
#  3D scene
# ============================================================================

BOX_I = [0,0,1,1,2,2,4,4,0,0,1,1]
BOX_J = [1,2,2,3,3,0,5,6,4,1,5,2]
BOX_K = [5,6,6,7,7,4,6,7,1,5,6,6]

def create_3d_scene(components, enc, result=None, fragments=None, show_frags=True):
    fig = go.Figure()
    prob_map = {}
    if result:
        for cr in result.component_results:
            prob_map[cr.component_id] = cr.combined_damage_prob

    for comp in components:
        cid = comp['id']
        geom = comp['geometry']
        pos = geom['position']
        dims = geom['dimensions']
        shape = geom['shape']
        rot = geom.get('rotation') or {}
        p = prob_map.get(cid, 0.0)
        color = _prob_color(p)
        cx = pos.get('x',0) or 0; cy = pos.get('y',0) or 0; cz = pos.get('z',0) or 0
        rx = rot.get('x',0) or 0; ry = rot.get('y',0) or 0; rz = rot.get('z',0) or 0
        has_rot = abs(rx)>0.5 or abs(ry)>0.5 or abs(rz)>0.5

        if shape == "长方体":
            l = dims.get('length_or_radius',50) or 50
            w = dims.get('width',50) or 50
            h = dims.get('height',50) or 50
            R = _rot_xyz(rx,ry,rz) if has_rot else None
            vx,vy,vz = _box_verts(cx,cy,cz, l,w,h, R)
            fig.add_trace(go.Mesh3d(x=vx,y=vy,z=vz,i=BOX_I,j=BOX_J,k=BOX_K,
                color=color, opacity=0.7, name=comp['name'],
                hovertemplate=f"<b>{comp['name']}</b> (ID:{cid})<br>P={p:.2%}<extra></extra>"))

        elif shape == "圆柱体":
            radius = dims.get('length_or_radius',20) or 20
            h = dims.get('height',50) or 50
            if has_rot:
                axis_vec = _rot_xyz(rx,ry,rz) @ np.array([0,0,1.0])
            else:
                axis_vec = np.array([0,0,1.0])
            xg,yg,zg = _cyl_surface(cx,cy,cz, radius, h/2, axis_vec)
            fig.add_trace(go.Surface(x=xg,y=yg,z=zg,
                colorscale=[[0,color],[1,color]], showscale=False, opacity=0.6,
                name=comp['name'],
                hovertemplate=f"<b>{comp['name']}</b> (ID:{cid})<br>P={p:.2%}<extra></extra>"))

        elif shape == "拉伸多边形":
            ext_len = dims.get('extrusion_length', 100)
            start_x = pos.get('x', 0)
            end_x = start_x + ext_len
            verts_yz = np.array(geom.get('vertices_yz', []))
            if len(verts_yz) >= 3:
                vx, vy, vz, mesh_i, mesh_j, mesh_k = _extruded_poly_mesh(start_x, end_x, verts_yz)
                # 装甲不透明度稍微低一点，颜色偏暗
                fig.add_trace(go.Mesh3d(
                    x=vx, y=vy, z=vz, i=mesh_i, j=mesh_j, k=mesh_k,
                    color=color, opacity=0.35, name=comp['name'],
                    hovertemplate=f"<b>{comp['name']}</b> (ID:{cid})<br>P={p:.2%}<extra></extra>"
                ))

    # 爆炸点
    fig.add_trace(go.Scatter3d(x=[enc.dx],y=[enc.dy],z=[enc.dz],
        mode='markers', marker=dict(size=10,color='gold',symbol='diamond',
                                     line=dict(width=2,color='red')),
        name='起爆点'))

    # 来袭方向
    v = enc.velocity_vector_ms
    vn = v/(np.linalg.norm(v)+1e-10)*150
    tip = enc.position_cm + vn
    fig.add_trace(go.Scatter3d(x=[enc.dx,tip[0]],y=[enc.dy,tip[1]],z=[enc.dz,tip[2]],
        mode='lines', line=dict(color='red',width=6), name='来袭方向'))

    # 破片射线
    if show_frags and fragments:
        step = max(1, len(fragments)//60)
        xs,ys,zs = [],[],[]
        for f in fragments[::step]:
            o = f.origin_T; end = o + f.direction_T*300
            xs += [o[0],end[0],None]; ys += [o[1],end[1],None]; zs += [o[2],end[2],None]
        fig.add_trace(go.Scatter3d(x=xs,y=ys,z=zs,
            mode='lines', line=dict(color='orange',width=1),
            opacity=0.3, name=f'破片射线 ({len(fragments)}枚)'))

    fig.update_layout(
        scene=dict(xaxis_title='X(cm)',yaxis_title='Y(cm)',zaxis_title='Z(cm)',
                   aspectmode='data', camera=dict(eye=dict(x=1.5,y=1.5,z=1.0))),
        showlegend=True, legend=dict(x=0,y=1),
        margin=dict(l=0,r=0,t=30,b=0), height=650)
    return fig


# ============================================================================
#  K/M/F/C cards
# ============================================================================

def render_kmfc_cards(dt):
    cols = st.columns(5)
    score = dt.overall_score
    sc = "#dc3545" if score>=0.8 else "#fd7e14" if score>=0.5 else "#ffc107" if score>=0.3 else "#28a745"
    with cols[0]:
        st.markdown(f'<div class="score-box" style="background:linear-gradient(135deg,{sc}22,{sc}44);border:2px solid {sc};"><div style="font-size:0.8rem;color:#666;">综合评分</div><div style="color:{sc};">{score:.2f}</div></div>', unsafe_allow_html=True)

    cats = [("K","灾难性",dt.K_level,dt.K1_prob,dt.K2_prob,["无","油箱引燃","弹药殉爆"]),
            ("M","机动",dt.M_level,dt.M1_prob,dt.M2_prob,["正常","降级","丧失"]),
            ("F","火力",dt.F_level,dt.F1_prob,dt.F2_prob,["正常","降级","丧失"]),
            ("C","乘员",dt.C_level,dt.C1_prob,dt.C2_prob,["安全","20%阵亡","60%阵亡"])]
    clrs = ["#28a745","#ffc107","#dc3545"]
    for col,(cat,name,lv,p1,p2,labels) in zip(cols[1:],cats):
        c = clrs[lv]
        with col:
            st.markdown(f'<div class="kmfc-card" style="background:{c}18;border-left:4px solid {c};"><div style="font-size:0.75rem;color:#888;">{name}毁伤</div><div style="font-size:1.8rem;color:{c};">{cat}{lv}</div><div style="font-size:0.7rem;color:#666;">{labels[lv]}</div><div style="font-size:0.65rem;color:#999;margin-top:4px;">P1={p1:.2f} | P2={p2:.2f}</div></div>', unsafe_allow_html=True)


# ============================================================================
#  Heatmap / charts
# ============================================================================

def create_subsystem_heatmap(result, components):
    systems = {"动力":[1,2,3],"右负重轮":list(range(6,12)),"左负重轮":list(range(12,18)),
        "托带轮":list(range(18,24)),"悬挂":list(range(24,30)),"主动轮":[4,5],"诱导轮":[30,31],
        "右履带":list(range(32,36)),"左履带":list(range(36,40)),"观瞄/测距":[40,41,42],
        "火控":[43,44],"主炮/供弹":[45,46,49],"辅助武器":[47,48,50,51,52,53,54],
        "通讯":[55,56,57],"乘员":list(range(58,68))}
    pm = {cr.component_id:cr.combined_damage_prob for cr in result.component_results}
    nm = {cr.component_id:cr.component_name for cr in result.component_results}
    rows = []
    for sn,ids in systems.items():
        for cid in ids:
            rows.append({"子系统":sn,"部件":nm.get(cid,f"ID:{cid}"),"ID":cid,"概率":pm.get(cid,0)})
    df = pd.DataFrame(rows)
    fig = px.treemap(df, path=["子系统","部件"], values=[1]*len(df),
        color="概率", color_continuous_scale="RdYlGn_r", range_color=[0,1],
        hover_data={"概率":":.2%","ID":True})
    fig.update_layout(title="部件毁伤概率分布(按子系统)", height=500, margin=dict(t=40,b=10,l=10,r=10))
    return fig, df


# ============================================================================
#  Scans
# ============================================================================

def run_height_scan(proj, components, armor, rng_seed):
    heights = list(range(50,601,25))
    data = []
    engine = DamageEngine(armor_plates=armor)
    for dz in heights:
        r = engine.evaluate(proj, EncounterCondition.from_speed_and_attitude(
                                dz=dz, pitch_deg=-90, speed=100),
                            components, rng_seed=rng_seed)
        dt = r.damage_tree
        data.append(dict(dz=dz, score=dt.overall_score, hits=r.total_hits, pen=r.total_penetrations,
                         K=dt.K_level, M=dt.M_level, F=dt.F_level, C=dt.C_level))
    df = pd.DataFrame(data)
    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,
                        subplot_titles=["综合评分 vs 起爆高度","命中/穿透 vs 起爆高度"])
    fig.add_trace(go.Scatter(x=df.dz,y=df.score,mode='lines+markers',name='评分',
                             line=dict(color='#dc3545',width=3)),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.dz,y=df.hits,mode='lines+markers',name='命中',
                             line=dict(color='#007bff')),row=2,col=1)
    fig.add_trace(go.Scatter(x=df.dz,y=df.pen,mode='lines+markers',name='穿透',
                             line=dict(color='#28a745')),row=2,col=1)
    fig.update_xaxes(title_text="dz(cm)",row=2,col=1)
    fig.update_layout(height=500, margin=dict(t=40,b=30))
    return fig, df

def run_angle_scan(proj, components, armor, rng_seed):
    yaws = list(range(0,361,15))
    scores = []
    engine = DamageEngine(armor_plates=armor)
    for y in yaws:
        enc = EncounterCondition.from_speed_and_attitude(
            dx=200*np.sin(np.radians(y)), dy=-200*np.cos(np.radians(y)),
            dz=200, yaw_deg=y, pitch_deg=-45, speed=100)
        scores.append(engine.evaluate(proj,enc,components,rng_seed=rng_seed).damage_tree.overall_score)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=scores,theta=yaws,mode='lines+markers',
        line=dict(color='#dc3545',width=2),fill='toself',fillcolor='rgba(220,53,69,0.15)'))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0,1])),
        title="方位角vs评分(45°俯冲,R=200,dz=200)", height=500)
    return fig


# ============================================================================
#  Detail table
# ============================================================================

def _sys(cid):
    if cid<=3: return "动力"
    if cid<=5: return "主动轮"
    if cid<=17: return "负重轮"
    if cid<=23: return "托带轮"
    if cid<=29: return "悬挂"
    if cid<=31: return "诱导轮"
    if cid<=39: return "履带"
    if cid<=42: return "观瞄"
    if cid<=44: return "火控"
    if cid<=54: return "武器"
    if cid<=57: return "通讯"
    if cid<=67: return "乘员"
    return "装甲"

def render_detail_table(result, components):
    rows = []
    for cr in result.component_results:
        rows.append({"ID":cr.component_id, "部件":cr.component_name,
            "子系统":_sys(cr.component_id),
            "命中":cr.fragment_hits, "穿透":cr.fragment_penetrations,
            "破片P":f"{cr.fragment_damage_prob:.2%}",
            "超压MPa":f"{cr.overpressure_mpa:.3f}",
            "冲击波P":f"{cr.shockwave_damage_prob:.2%}",
            "综合P":f"{cr.combined_damage_prob:.2%}",
            "等级":cr.damage_level.value})
    return pd.DataFrame(rows)


# ============================================================================
#  JSON safe serializer (fixes numpy bool / float)
# ============================================================================

def _json_safe(obj):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ============================================================================
#  Main
# ============================================================================

def main():
    st.markdown('<h1 class="main-header">🎯 巡飞弹毁伤评估系统</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">基于物理仿真的效果驱动末制导分析平台 | sim_engine v2</p>',
                unsafe_allow_html=True)

    components = cached_load_components()
    armor_plates = cached_load_armor()

    proj, enc, enable_frag, enable_shock, rng_seed, run = render_sidebar()

    # ---- 点击按钮时用当前参数计算，结果存 session_state ----
    if run:
        with st.spinner("正在执行毁伤仿真..."):
            engine = DamageEngine(enable_fragments=enable_frag,
                                  enable_shockwave=enable_shock,
                                  armor_plates=armor_plates)
            result = engine.evaluate(proj, enc, components, rng_seed=rng_seed)
            frags = generate_fragment_field(proj, enc, rng_seed=rng_seed) if enable_frag else []
            st.session_state['sim_result'] = result
            st.session_state['sim_frags'] = frags
            st.session_state['sim_enc'] = enc
            st.session_state['sim_proj'] = proj
        st.success("✅ 仿真完成！修改参数后请再次点击「执行仿真」更新结果。")

    # ---- 有结果就显示，没有就显示空模型 ----
    if 'sim_result' not in st.session_state:
        st.info("👈 设置参数后点击 **执行仿真** 按钮开始计算")
        fig_init = create_3d_scene(components, enc)
        st.plotly_chart(fig_init, use_container_width=True)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("动力系统", f"{sum(1 for c in components if c['id']<=3)} 部件")
        c2.metric("行走系统", f"{sum(1 for c in components if 4<=c['id']<=39)} 部件")
        c3.metric("火力系统", f"{sum(1 for c in components if 40<=c['id']<=54)} 部件")
        c4.metric("乘员", f"{sum(1 for c in components if 58<=c['id']<=67)} 人")
        return

    result = st.session_state['sim_result']
    frags  = st.session_state['sim_frags']
    enc_s  = st.session_state['sim_enc']
    proj_s = st.session_state['sim_proj']
    dt = result.damage_tree

    render_kmfc_cards(dt)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("总破片", result.total_fragments)
    c2.metric("命中", result.total_hits)
    c3.metric("穿透", result.total_penetrations)
    c4.metric("毁伤部件", f"{result.damaged_count}/{result.total_components}")
    c5.metric("弹型", proj_s.name[:8])

    st.markdown("---")

    tab_3d, tab_heat, tab_table, tab_scan, tab_export = st.tabs(
        ["🚗 3D模型","🔥 毁伤分布","📋 详细数据","📈 参数扫描","📥 导出"])

    with tab_3d:
        show_frags = st.checkbox("显示破片射线", True)
        fig3d = create_3d_scene(components, enc_s, result, frags, show_frags)
        st.plotly_chart(fig3d, use_container_width=True)

    with tab_heat:
        fig_tree, _ = create_subsystem_heatmap(result, components)
        st.plotly_chart(fig_tree, use_container_width=True)
        frag_dmg = sum(1 for r in result.component_results if r.fragment_damage_prob>0.5)
        shock_dmg = sum(1 for r in result.component_results if r.shockwave_damage_prob>0.5)
        fig_bar = go.Figure(data=[
            go.Bar(name='破片穿透',x=['毁伤方式'],y=[frag_dmg],marker_color='#e74c3c'),
            go.Bar(name='冲击波',x=['毁伤方式'],y=[shock_dmg],marker_color='#3498db')])
        fig_bar.update_layout(barmode='group',height=300,title="毁伤方式对比(P>50%)")
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab_table:
        df = render_detail_table(result, components)
        c1,c2 = st.columns(2)
        with c1: fs = st.multiselect("筛选子系统",df["子系统"].unique(),default=df["子系统"].unique())
        with c2: fd = st.selectbox("筛选状态",["全部","仅毁伤","仅正常"])
        dfs = df[df["子系统"].isin(fs)]
        if fd=="仅毁伤":   dfs = dfs[~dfs["等级"].str.contains("未毁伤")]
        elif fd=="仅正常": dfs = dfs[dfs["等级"].str.contains("未毁伤")]
        st.dataframe(dfs, use_container_width=True, hide_index=True, height=500)

    with tab_scan:
        st.radio("扫描类型",["起爆高度扫描","攻击方位角扫描"],horizontal=True, key="scan_type")
        if st.button("运行扫描"):
            if st.session_state.scan_type == "起爆高度扫描":
                with st.spinner("扫描中..."): fig_s, df_s = run_height_scan(proj_s,components,armor_plates,rng_seed)
                st.plotly_chart(fig_s, use_container_width=True)
                st.dataframe(df_s, use_container_width=True, hide_index=True)
            else:
                with st.spinner("扫描中..."): fig_p = run_angle_scan(proj_s,components,armor_plates,rng_seed)
                st.plotly_chart(fig_p, use_container_width=True)

    with tab_export:
        st.markdown("#### 导出仿真结果")
        report = result.to_dict()
        report["encounter"] = {"dx":enc_s.dx,"dy":enc_s.dy,"dz":enc_s.dz,
            "vx":enc_s.vx,"vy":enc_s.vy,"vz":enc_s.vz,
            "yaw":enc_s.yaw_deg,"pitch":enc_s.pitch_deg,
            "roll":enc_s.roll_deg,"velocity":enc_s.velocity}
        report_json = json.dumps(report, ensure_ascii=False, indent=2, default=_json_safe)
        st.download_button("📥 下载JSON报告", report_json, "damage_report.json", "application/json")

        if dt.triggered_rules:
            st.markdown("#### 毁伤树触发规则")
            for rule in dt.triggered_rules:
                st.markdown(f"- {rule}")


if __name__ == "__main__":
    main()
