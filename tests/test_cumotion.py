"""Unit tests & speed benchmarks for the cuMotion-style collision-aware planner."""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_local.classical.cumotion import (
    BatchedUR12eIK,
    CuMotionPlanner,
    PlanningScene,
    grip_from_gap,
    gripper_qpos_from_grip,
    make_held_bolt_spec,
    make_held_nut_spec,
    make_held_washer_spec,
)
from vlm_local.classical.kinematics import HOME_Q, down_mat, site_rot_from_zx
from vlm_local.classical.scene_interface import (
    NOMINAL_FIXTURE_XY,
    NOMINAL_FIXTURE_YAW,
    SEAT_LOCAL_GEOM,
    PartEstimate,
)


def test_scene_load_and_query_speed() -> None:
    t0 = time.perf_counter()
    scene = PlanningScene()
    load_s = time.perf_counter() - t0
    assert load_s < 2.0, f"PlanningScene load took {load_s:.3f}s (> 2.0s budget)"

    t1 = time.perf_counter()
    n_queries = 200
    for _ in range(n_queries):
        rep = scene.check_config(HOME_Q, 0.75)
    us_per_query = (time.perf_counter() - t1) * 1e6 / n_queries
    assert rep.collision_free
    assert us_per_query < 500.0, f"Per-query collision check too slow: {us_per_query:.1f} us"


def test_batched_fk_and_jacobian_match_mujoco() -> None:
    for mount_yaw in (0.0, 7.0 * math.pi / 8.0):
        scene = PlanningScene(tool_mount_yaw=mount_yaw)
        bik = BatchedUR12eIK(tool_mount_yaw=mount_yaw)
        rng = np.random.default_rng(123)
        qs = rng.uniform(-2.0, 2.0, size=(24, 6))
        ee_p, ee_r, jac = bik.fk_and_jacobian_batch(qs)

        jp = np.zeros((3, scene.model.nv))
        jr = np.zeros((3, scene.model.nv))
        for i in range(qs.shape[0]):
            scene.set_q(qs[i], 0.50)
            mujoco.mj_comPos(scene.model, scene.data)
            mj_p = scene.data.site_xpos[scene.site_id]
            mj_r = scene.data.site_xmat[scene.site_id].reshape(3, 3)
            mujoco.mj_jacSite(scene.model, scene.data, jp, jr, scene.site_id)
            mj_j = np.vstack([jp[:, scene.arm_dofs], jr[:, scene.arm_dofs]])
            assert float(np.linalg.norm(ee_p[i] - mj_p)) < 2e-4
            assert float(np.linalg.norm(ee_r[i] - mj_r)) < 5e-4
            assert float(np.max(np.abs(jac[i] - mj_j))) < 5e-4


def test_spring_link_signed_clearance_and_m24_insertion() -> None:
    planner = CuMotionPlanner()
    planner.sync_scene(fixture_xy=NOMINAL_FIXTURE_XY, fixture_yaw=NOMINAL_FIXTURE_YAW, parts=[])
    c, s = math.cos(NOMINAL_FIXTURE_YAW), math.sin(NOMINAL_FIXTURE_YAW)
    lx, ly = SEAT_LOCAL_GEOM["bore_m24"]["local_xy"]
    sxy = np.array([NOMINAL_FIXTURE_XY[0] + c * lx - s * ly, NOMINAL_FIXTURE_XY[1] + s * lx + c * ly, 0.115])

    # Naive 13.5mm pad clearance (sz=0.1305) at yaw +60 deg puts left_spring_link (-3.95 mm) into the block!
    bad_yaw = NOMINAL_FIXTURE_YAW + math.radians(60.0)
    bad_rot = site_rot_from_zx(np.array([math.cos(bad_yaw), math.sin(bad_yaw), 0.0]), np.array([0.0, 0.0, -1.0]))
    bad_off = bad_rot @ np.array([0.0015, 0.0, 0.130])
    bad_pos = np.array([sxy[0] - bad_off[0], sxy[1] - bad_off[1], 0.1305])
    q_bad = [x.q for x in planner.batch_ik.solve_batch(bad_pos[None, :], bad_rot[None, :], HOME_Q) if x.converged][0]
    rep_bad = planner.scene.check_config(q_bad, grip_from_gap(0.024), min_clearance=0.002, ignore_held_part=True)
    assert not rep_bad.collision_free
    assert rep_bad.spring_link_clearance < 0.0

    # CuMotionPlanner selects a heading + height keeping spring links >= 5 mm clear across +-6mm search ring!
    held_m24 = make_held_bolt_spec("bolt_m24x70", d_along=0.0015, z_s_held=0.130, head_along_neg_x=True)

    def build_fn(yaw: float) -> dict:
        rot = site_rot_from_zx(np.array([math.cos(yaw), math.sin(yaw), 0.0]), np.array([0.0, 0.0, -1.0]))
        off = rot @ np.array([0.0015, 0.0, 0.130])
        s_pre = np.array([sxy[0], sxy[1], 0.182]) - off
        s_ins = np.array([sxy[0] - off[0], sxy[1] - off[1], 0.1366])
        return {"rot": rot, "s_pre": s_pre, "s_end": s_ins, "s_transit": np.array([s_pre[0], s_pre[1], 0.285])}

    plan, diag = planner.select_bolt_insertion_plan(
        build_plan_fn=build_fn,
        fyaw=NOMINAL_FIXTURE_YAW,
        q_ref=HOME_Q,
        grip_held=grip_from_gap(0.024),
        held_spec=held_m24,
        support_top_z=0.115,
    )
    assert "q_end" in plan and "q_transit" in plan
    assert diag["spring_mm"] >= 5.0
    assert diag["clear_mm"] >= 2.5


def test_cartesian_and_joint_segment_via_points() -> None:
    planner = CuMotionPlanner()

    # 1. Washer transport across 115mm fixture block at z=0.258m must insert elevated bridge >= 0.305m
    held_w = make_held_washer_spec("washer_m33")
    planner.sync_scene(fixture_xy=NOMINAL_FIXTURE_XY, fixture_yaw=NOMINAL_FIXTURE_YAW, parts=[], held=held_w)
    p0 = np.array([0.66, -0.30, 0.258])
    p1 = np.array([0.59, 0.08, 0.258])
    r0 = down_mat(0.0)
    q0, _, ok0 = planner.solve_ik(p0, r0, HOME_Q, 0.44)
    assert ok0
    qs = planner.plan_cartesian_waypoints(p0, r0, p1, r0, q0, 0.44, pos_speed=0.028)
    ee_zs = [float(planner.batch_ik.fk_and_jacobian_batch(q[None, :])[0][0, 2]) for q in qs]
    assert max(ee_zs) >= 0.30
    for q in qs:
        assert planner.scene.check_config(q, 0.44, min_clearance=0.002).collision_free

    # 2. Joint-space move dipping held M30 bolt into table/fixture must insert collision-free via-points
    held_b = make_held_bolt_spec("bolt_m30x100", d_along=0.004, z_s_held=0.130, head_along_neg_x=True)
    planner.sync_scene(fixture_xy=NOMINAL_FIXTURE_XY, fixture_yaw=NOMINAL_FIXTURE_YAW, parts=[], held=held_b)
    q_a = np.array([-0.54, -1.19, 2.42, -2.80, -1.57, -2.51])
    q_b = np.array([0.39, -0.77, 1.77, -1.00, -2.49, 0.0])
    raw_ok, _, _ = planner.check_joint_segment(q_a, q_b, 0.25, include_endpoints=False, allow_start_escape=False)
    assert not raw_ok
    wps = planner.plan_joint_waypoints(q_a, q_b, 0.25)
    assert len(wps) >= 2
    for seg_a, seg_b in zip([q_a] + wps[:-1], wps):
        ok_seg, _, _ = planner.check_joint_segment(seg_a, seg_b, 0.25, include_endpoints=False)
        assert ok_seg


def test_dynamic_obstacle_avoidance_and_speed_scaler() -> None:
    planner = CuMotionPlanner()
    # Standing M30 bolt obstacle at (0.62, -0.10)
    standing_bolt = PartEstimate(
        kind="bolt",
        raw_kind="bolt_m30x100",
        position=np.array([0.62, -0.10, 0.060]),
        axis=np.array([0.0, 0.0, -1.0]),
        yaw=0.0,
        confidence=0.95,
        pose_hint="standing",
    )
    held_n = make_held_nut_spec("nut_m30")
    planner.sync_scene(fixture_xy=NOMINAL_FIXTURE_XY, fixture_yaw=NOMINAL_FIXTURE_YAW, parts=[standing_bolt], held=held_n)
    p0 = np.array([0.62, -0.25, 0.24])
    p1 = np.array([0.62, 0.04, 0.28])
    r0 = down_mat(0.0)
    q0, _, _ = planner.solve_ik(p0, r0, HOME_Q, 0.42)
    qs = planner.plan_cartesian_waypoints(p0, r0, p1, r0, q0, 0.42, pos_speed=0.025)
    for q in qs:
        assert planner.scene.check_config(q, 0.42, min_clearance=0.002, require_elbow_up=False).collision_free

    # SpeedScaler check: huge 0.05 rad step on all joints is scaled so TCP/elbow <= 1.45 m/s
    dq = np.full(6, 0.0495)
    s = planner.scene.safe_step_scale(HOME_Q, dq)
    tcp, elbow = planner.scene.commanded_speeds(HOME_Q + dq * s, dq * s)
    assert tcp <= 1.46 and elbow <= 1.46


def test_gripper_4bar_table() -> None:
    q_open = gripper_qpos_from_grip(1.0)
    q_closed = gripper_qpos_from_grip(0.0)
    assert q_open[0] > -0.01 and q_closed[0] < -0.70
    assert 0.15 < grip_from_gap(0.016) < 0.35
