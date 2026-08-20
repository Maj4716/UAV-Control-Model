#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成测试 — 验证全部7项审计修正

对比项:
  1. Taylor角修正后, 前端起爆 vs 后端起爆的命中率差异
  2. 装甲壳体碰撞后, 内部部件穿透率下降
  3. 破片总数校正 (150→150, 不再丢失)
  4. Thor方程的V50值
  5. 毁伤树概率传递
"""

from loitering_munition_damage_twin.simulation.engine import (
    EncounterCondition, DamageEngine,
    create_medium_loitering_munition, create_medium_rear_det,
    create_small_loitering_munition, create_heavy_loitering_munition,
    load_vehicle_model, load_armor_plates,
    ThorPenetrationModel, TaylorAngleModel, GurneyModel,
)
import numpy as np


def print_result(result, name):
    print(f"\n{'='*70}")
    print(f"场景: {name}")
    print(f"弹型: {result.projectile_name}")
    print(f"{'='*70}")
    print(f"  破片: {result.total_fragments} → 命中{result.total_hits} → 穿透{result.total_penetrations}")
    print(f"  毁伤部件: {result.damaged_count} / {result.total_components}")

    dt = result.damage_tree
    if dt:
        ln = {0:"无", 1:"轻度", 2:"重度"}
        print(f"  --- 毁伤树 ---")
        print(f"  K(灾难): {ln[dt.K_level]} | K1={dt.K1_prob:.2f}  K2={dt.K2_prob:.2f}")
        print(f"  M(机动): {ln[dt.M_level]} | M1={dt.M1_prob:.2f}  M2={dt.M2_prob:.2f}")
        print(f"  F(火力): {ln[dt.F_level]} | F1={dt.F1_prob:.2f}  F2={dt.F2_prob:.2f}")
        print(f"  C(乘员): {ln[dt.C_level]} | C1={dt.C1_prob:.2f}  C2={dt.C2_prob:.2f}")
        print(f"  综合评分: {dt.overall_score:.4f}")
        if dt.triggered_rules:
            print(f"  触发:")
            for r in dt.triggered_rules:
                print(f"    → {r}")

    damaged = [r for r in result.component_results if r.is_damaged]
    if damaged:
        print(f"\n  受损部件({len(damaged)}个):")
        for r in damaged:
            print(f"    [{r.component_id:2d}] {r.component_name:10s} | "
                  f"命中{r.fragment_hits}穿{r.fragment_penetrations} "
                  f"Pf={r.fragment_damage_prob:.2f} | "
                  f"超压{r.overpressure_mpa:.3f}MPa "
                  f"Ps={r.shockwave_damage_prob:.2f} | "
                  f"Pc={r.combined_damage_prob:.2f} [{r.damage_level.value}]")


def main():
    print("=" * 70)
    print("巡飞弹毁伤仿真器 v2 — 集成测试 (含全部审计修正)")
    print("=" * 70)

    # 加载
    components = load_vehicle_model()
    armor_plates = load_armor_plates()
    print(f"\n目标模型: {len(components)} 个部件")
    print(f"装甲板: {len(armor_plates)} 块")
    for ap in armor_plates:
        print(f"  {ap.name}: {ap.thickness_mm}mm, "
              f"Y=[{ap.aabb_min[1]:.0f},{ap.aabb_max[1]:.0f}] "
              f"Z=[{ap.aabb_min[2]:.0f},{ap.aabb_max[2]:.0f}]")

    # ---- 验证Fix1: Taylor角 ----
    print(f"\n{'='*70}")
    print("验证Fix1: Taylor角 (从弹轴度量)")
    print(f"{'='*70}")
    proj_med = create_medium_loitering_munition()
    w = proj_med.warhead
    vg = GurneyModel.cylinder_velocity(w.gurney_energy_mps, w.metal_to_charge_ratio)
    base_t = TaylorAngleModel.base_angle_rad(w.detonation_velocity_mps, vg)
    print(f"  Gurney速度: {vg:.1f} m/s")
    print(f"  Taylor角(从弹轴): {np.degrees(base_t):.1f}° (修正前是60.2°)")
    print(f"  半锥角≈30° → 破片聚焦在弹轴前后30°锥面内")

    # ---- 验证Fix4: 破片总数 ----
    print(f"\n验证Fix4: 破片总数")
    from loitering_munition_damage_twin.simulation.engine import generate_fragment_field
    enc_test = EncounterCondition.from_speed_and_attitude(dz=200, pitch_deg=-90)
    for name, proj in [("小型", create_small_loitering_munition()),
                       ("中型", create_medium_loitering_munition()),
                       ("大型", create_heavy_loitering_munition())]:
        frags = generate_fragment_field(proj, enc_test, rng_seed=42)
        print(f"  {name}: 声明{proj.warhead.fragment_bed.total_count}枚, "
              f"实际生成{len(frags)}枚")

    # ---- 验证Fix5: Thor方程 ----
    print(f"\n验证Fix5: Thor V50 (8g破片 vs RHA)")
    for t_mm in [2, 6, 10, 20, 30]:
        v50 = ThorPenetrationModel.v50(t_mm, 8.0)
        print(f"  t={t_mm:2d}mm: V50={v50:.0f} m/s")

    # ---- 核心场景对比 ----
    engine = DamageEngine(armor_plates=armor_plates)

    # 场景1: 垂直俯冲 — 前端起爆 vs 后端起爆
    print(f"\n{'='*70}")
    print("核心对比: 前端起爆 vs 后端起爆 (垂直俯冲, dz=200cm)")
    print(f"{'='*70}")
    enc1 = EncounterCondition.from_speed_and_attitude(dz=200, pitch_deg=-90, speed=100)

    proj_front = create_medium_loitering_munition()
    proj_rear = create_medium_rear_det()

    r_front = engine.evaluate(proj_front, enc1, components, rng_seed=42)
    r_rear = engine.evaluate(proj_rear, enc1, components, rng_seed=42)

    print_result(r_front, "垂直俯冲 - 前端起爆")
    print_result(r_rear, "垂直俯冲 - 后端起爆")

    print(f"\n  ★ 前端起爆命中{r_front.total_hits}穿{r_front.total_penetrations} "
          f"评分={r_front.damage_tree.overall_score:.4f}")
    print(f"  ★ 后端起爆命中{r_rear.total_hits}穿{r_rear.total_penetrations} "
          f"评分={r_rear.damage_tree.overall_score:.4f}")

    # 场景2: 45°侧面
    enc2 = EncounterCondition.from_speed_and_attitude(dx=200, dz=200, pitch_deg=-45, yaw_deg=90, speed=100)
    r2 = engine.evaluate(proj_front, enc2, components, rng_seed=42)
    print_result(r2, "45°俯冲从右侧 (dx=200, dz=200, yaw=90)")

    # 场景3: 低角度前方
    enc3 = EncounterCondition.from_speed_and_attitude(dy=-400, dz=50, pitch_deg=-15, speed=100)
    r3 = engine.evaluate(proj_front, enc3, components, rng_seed=42)
    print_result(r3, "低角度前方 (dy=-400, dz=50, pitch=-15)")

    # ---- 起爆高度扫描 (后端起爆) ----
    print(f"\n{'='*70}")
    print("起爆高度扫描 (中型弹后端起爆, 垂直俯冲, v=100m/s)")
    print(f"{'='*70}")
    for dz in [100, 150, 200, 250, 300, 400, 500]:
        enc_h = EncounterCondition.from_speed_and_attitude(dz=dz, pitch_deg=-90, speed=100)
        rh = engine.evaluate(proj_rear, enc_h, components, rng_seed=42)
        dt = rh.damage_tree
        print(f"  dz={dz:3d}cm | 命中{rh.total_hits:3d} 穿透{rh.total_penetrations:3d} | "
              f"毁伤{rh.damaged_count:2d}/{rh.total_components} | "
              f"评分={dt.overall_score:.4f} | "
              f"K{dt.K_level}M{dt.M_level}F{dt.F_level}C{dt.C_level}")

    # ---- 弹型对比 (后端起爆, dz=200) ----
    print(f"\n{'='*70}")
    print("弹型对比 (垂直俯冲, dz=200cm, v=100m/s)")
    print(f"{'='*70}")
    for label, proj in [("小型(前起)", create_small_loitering_munition()),
                        ("中型(前起)", create_medium_loitering_munition()),
                        ("中型(后起)", create_medium_rear_det()),
                        ("大型(前起)", create_heavy_loitering_munition())]:
        enc_c = EncounterCondition.from_speed_and_attitude(dz=200, pitch_deg=-90, speed=100)
        rc = engine.evaluate(proj, enc_c, components, rng_seed=42)
        dt = rc.damage_tree
        print(f"  {label:10s} | 破片{rc.total_fragments:3d}→命中{rc.total_hits:3d}→穿{rc.total_penetrations:3d} | "
              f"毁伤{rc.damaged_count:2d}/{rc.total_components} | "
              f"评分={dt.overall_score:.4f} | K{dt.K_level}M{dt.M_level}F{dt.F_level}C{dt.C_level}")

    print("\n[OK] 全部测试完成")


if __name__ == "__main__":
    main()
