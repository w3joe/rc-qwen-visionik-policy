"""Closed-loop manipulation state machine and motion primitives for RGB vision + IK.

Pure public-observation controller:
  - Uses ONLY proprioception (`joint_pos`, `joint_vel`, `gripper`, `ee_pos`, `ee_quat`),
    `task_id`, `instruction`, and the normalized output of `vision.estimate_scene(obs)`.
  - Uses pure-NumPy UR12e FK/Jacobian/DLS-IK (`policy.kinematics`).
  - Tracks commanded `q_target` starting from `HOME_Q` with strictly bounded increments.
  - Implements gripper sequencing, proprio grasp verification, nudge recovery,
    contact-guided bolt insertion, hex nut seating with contact wiggle, and retries.
"""
from __future__ import annotations

import math
import os
from typing import Any, Iterator

import numpy as np

from .cell import BASE_WORLD, LAYOUT_SHIFT, reach_yaw, shifted
from .kinematics import (
    HOME_Q,
    NUM_JOINTS,
    PAD_HALF_WIDTH_X,
    PAD_TIP_BOLT_M16,
    PAD_TIP_BOLT_M30,
    PAD_TIP_CLOSED,
    PAD_TIP_NUT_FLAT,
    PAD_TIP_WASHER,
    SAFE_STEP_LIMIT,
    BoundedDampedIK,
    closest_hex_yaw,
    compute_safe_delta,
    down_mat,
    fk_pose,
    nearest_pi_yaw,
    normalize,
    pad_tip_for_gap,
    quat_to_mat,
    rot_error,
    site_rot_from_zx,
    site_rot_from_zy,
    slerp_rot,
    wrap_angle,
)
from .scene_interface import (
    FASTENER_GEOM,
    FIXTURE_BLOCK_TOP_Z,
    SEAT_LOCAL_GEOM,
    TRAY_FLOOR_Z,
    GoalSpec,
    PartEstimate,
    SceneEstimate,
    TemporalSceneFilter,
    is_part_in_seat,
    parse_task_goals,
    select_target_part,
)

# Step command yielded by the generator: (dq_6, gripper_01, want_vision_next_step)
StepCommand = tuple[np.ndarray, float, bool]

# Measured Robotiq 2F-140 empty-closed proprio reading is ~0.073; held parts read higher:
#   M16 bolt (16mm) ~ 0.22, M30 bolt (30mm) ~ 0.32, M30 nut flat (46mm) ~ 0.42, washer (62mm) ~ 0.53
EMPTY_GRIPPER_THRESH = 0.135


def _extract_proprio(obs: dict[str, Any]) -> dict[str, Any]:
    raw = obs.get("proprio") if isinstance(obs.get("proprio"), dict) else {}
    jp = np.asarray(raw.get("joint_pos", HOME_Q), dtype=np.float64).reshape(-1)
    if jp.size != NUM_JOINTS or not np.all(np.isfinite(jp)):
        jp = HOME_Q.copy()
    jv = np.asarray(raw.get("joint_vel", np.zeros(NUM_JOINTS)), dtype=np.float64).reshape(-1)
    if jv.size != NUM_JOINTS or not np.all(np.isfinite(jv)):
        jv = np.zeros(NUM_JOINTS, dtype=np.float64)
    try:
        grip = float(raw.get("gripper", 1.0))
        if not math.isfinite(grip):
            grip = 1.0
    except (TypeError, ValueError):
        grip = 1.0

    ee_pos_raw = raw.get("ee_pos")
    ee_quat_raw = raw.get("ee_quat")
    if ee_pos_raw is not None and ee_quat_raw is not None:
        ee_pos = np.asarray(ee_pos_raw, dtype=np.float64).reshape(-1)
        if ee_pos.size == 3:
            ee_pos = ee_pos + BASE_WORLD  # observation is in the robot base frame
        ee_quat = np.asarray(ee_quat_raw, dtype=np.float64).reshape(-1)
        if ee_pos.size == 3 and np.all(np.isfinite(ee_pos)) and ee_quat.size == 4 and np.all(np.isfinite(ee_quat)):
            ee_rot = quat_to_mat(ee_quat)
        else:
            ee_pos, ee_rot = fk_pose(jp)
    else:
        ee_pos, ee_rot = fk_pose(jp)

    return {
        "joint_pos": jp,
        "joint_vel": jv,
        "gripper": grip,
        "ee_pos": ee_pos,
        "ee_rot": ee_rot,
    }


def _min_grip_for_part(part_model: str) -> float:
    if "washer" in part_model:
        return 0.34
    if "nut" in part_model:
        af = float(FASTENER_GEOM.get(part_model, FASTENER_GEOM["nut_m30"]).get("across_flats", 0.046))
        return max(EMPTY_GRIPPER_THRESH + 0.02, min(0.38, 0.06 + 5.4 * af))
    if "m30" in part_model:
        return 0.19
    return EMPTY_GRIPPER_THRESH


class EpisodeController:
    """Step-driven closed-loop controller for one episode."""

    def __init__(self, first_obs: dict[str, Any]) -> None:
        self.ik = BoundedDampedIK()
        self.goals: list[GoalSpec] = parse_task_goals(first_obs)
        self.tracker = TemporalSceneFilter(window=7)

        proprio = _extract_proprio(first_obs)
        # At real episode start, simulator target is HOME_Q; fall back to joint_pos for synthetic tests
        if float(np.max(np.abs(proprio["joint_pos"] - HOME_Q))) < 0.25:
            self.q_target = HOME_Q.copy()
        else:
            self.q_target = proprio["joint_pos"].copy()

        self.obs: dict[str, Any] = first_obs
        self.proprio: dict[str, Any] = proprio
        self.scene: SceneEstimate | None = None
        self.want_vision: bool = True
        self.step_idx: int = 0
        self._decoy_positions: list[np.ndarray] = []

        self._gen: Iterator[StepCommand] = self._run_episode()

    def step(self, obs: dict[str, Any], scene: SceneEstimate | None) -> tuple[np.ndarray, float]:
        """Advance controller by one 20 Hz step and return (safe_dq, gripper)."""
        self.obs = obs
        self.proprio = _extract_proprio(obs)
        if scene is not None:
            self.scene = scene
            self.tracker.update_fixture(scene)
            # Record any explicitly identified decoys on early unobstructed observations
            if self.step_idx <= 5:
                for p in scene.parts:
                    if (p.role or "").lower() in ("decoy", "large_bolt", "bolt_m24") and p.confidence >= 0.50:
                        if not any(float(np.linalg.norm(p.position[:2] - d[:2])) < 0.04 for d in self._decoy_positions):
                            self._decoy_positions.append(p.position.copy())
        self.step_idx += 1

        try:
            dq, grip, want_vis = next(self._gen)
        except StopIteration:
            dq = np.zeros(NUM_JOINTS, dtype=np.float64)
            grip = 0.80
            want_vis = False

        self.want_vision = bool(want_vis)
        return dq, float(np.clip(grip, 0.0, 1.0))

    # ------------------------------------------------------------------ low-level primitives

    def _emit_toward(
        self,
        q_goal: np.ndarray,
        grip: float,
        *,
        max_step: float = SAFE_STEP_LIMIT,
        want_vision: bool = False,
    ) -> StepCommand:
        dq, q_next = compute_safe_delta(self.q_target, q_goal, max_step=max_step)
        self.q_target = q_next
        return dq, float(np.clip(grip, 0.0, 1.0)), want_vision

    def _emit_hold(self, grip: float, *, want_vision: bool = False) -> StepCommand:
        return np.zeros(NUM_JOINTS, dtype=np.float64), float(np.clip(grip, 0.0, 1.0)), want_vision

    def _cur_ee_yaw(self) -> float:
        R = self.proprio["ee_rot"]
        return math.atan2(float(R[1, 0]), float(R[0, 0]))

    def _cur_y_yaw(self) -> float:
        R = self.proprio["ee_rot"]
        return math.atan2(float(R[1, 1]), float(R[0, 1]))

    def _move_joint(
        self,
        q_goal: np.ndarray,
        grip: float,
        *,
        max_step: float = SAFE_STEP_LIMIT,
        settle: int = 2,
        tol: float = 0.010,
        max_steps: int = 180,
        want_vision: bool = False,
    ) -> Iterator[StepCommand]:
        q_goal = np.asarray(q_goal, dtype=np.float64)
        for _ in range(max_steps):
            if float(np.max(np.abs(q_goal - self.q_target))) <= 2e-4:
                break
            yield self._emit_toward(q_goal, grip, max_step=max_step, want_vision=want_vision)
        for _ in range(settle):
            yield self._emit_toward(q_goal, grip, max_step=max_step, want_vision=want_vision)
            err = float(np.max(np.abs(self.proprio["joint_pos"] - q_goal)))
            vel = float(np.max(np.abs(self.proprio["joint_vel"])))
            if err < tol and vel < 0.06:
                break

    def _move_cartesian(
        self,
        pos_goal: np.ndarray,
        rot_goal: np.ndarray,
        grip: float,
        *,
        pos_speed: float = 0.018,
        rot_speed: float = 0.14,
        max_step: float = SAFE_STEP_LIMIT,
        settle: int = 2,
        tol: float = 0.010,
        wrist_up: bool = False,
        want_vision: bool = False,
    ) -> Iterator[StepCommand]:
        pos_goal = np.asarray(pos_goal, dtype=np.float64)
        rot_goal = np.asarray(rot_goal, dtype=np.float64)
        pos0, rot0 = fk_pose(self.q_target)
        dist = float(np.linalg.norm(pos_goal - pos0))
        ang = float(np.linalg.norm(rot_error(rot0, rot_goal)))
        n_seg = max(1, int(math.ceil(max(dist / pos_speed, ang / rot_speed))))
        q_prev = self.q_target.copy()
        for i in range(1, n_seg + 1):
            t = i / n_seg
            p_t = pos0 + t * (pos_goal - pos0)
            r_t = slerp_rot(rot0, rot_goal, t)
            q_t = self.ik.solve(p_t, r_t, q_prev, wrist_up=wrist_up).q
            q_prev = q_t
            for _ in range(35):
                if float(np.max(np.abs(q_t - self.q_target))) <= 2e-4:
                    break
                yield self._emit_toward(q_t, grip, max_step=max_step, want_vision=want_vision)
        for _ in range(settle):
            yield self._emit_toward(q_prev, grip, max_step=max_step, want_vision=want_vision)
            err = float(np.max(np.abs(self.proprio["joint_pos"] - q_prev)))
            vel = float(np.max(np.abs(self.proprio["joint_vel"])))
            if err < tol and vel < 0.05:
                break

    def _grip_hold(
        self,
        grip: float,
        *,
        steps: int = 14,
        early_stop: bool = True,
        want_vision: bool = False,
    ) -> Iterator[StepCommand]:
        prev_g = self.proprio["gripper"]
        still = 0
        for i in range(steps):
            # Request vision on the last step if caller wants fresh perception next
            wv = want_vision and (i == steps - 1)
            yield self._emit_hold(grip, want_vision=wv)
            cur_g = self.proprio["gripper"]
            if early_stop and i >= 6 and abs(cur_g - prev_g) < 0.0025:
                still += 1
                if still >= 3:
                    break
            else:
                still = 0
            prev_g = cur_g

    def _ensure_safe_height(self, min_z: float, grip: float, *, wrist_up: bool = False) -> Iterator[StepCommand]:
        cur_p, cur_r = fk_pose(self.q_target)
        if cur_p[2] < min_z:
            yield from self._move_cartesian(
                np.array([cur_p[0], cur_p[1], min_z + 0.02], dtype=np.float64),
                cur_r,
                grip,
                pos_speed=0.024,
                settle=1,
                wrist_up=wrist_up,
            )

    def _move_to_vantage(self, grip: float = 0.75) -> Iterator[StepCommand]:
        """Move arm high over +y / fixture side on the standard wrist-down branch."""
        cur_p, _ = fk_pose(self.q_target)
        if cur_p[2] < 0.28:
            yield from self._ensure_safe_height(0.28, grip)
        vantage_pos = np.array(shifted((0.62, 0.14, 0.34)), dtype=np.float64)
        vantage_rot = down_mat(0.0)
        # Solve near HOME_Q so we cleanly reset to the standard elbow-up wrist-down branch when empty
        q_v = self.ik.solve(vantage_pos, vantage_rot, HOME_Q).q
        q_cur = self.q_target.copy()
        if q_cur[4] * q_v[4] < 0.0:
            q_mid = 0.5 * (q_cur + q_v)
            q_mid[1] = min(float(q_mid[1]), -1.55)
            q_mid[2] = min(float(q_mid[2]), 1.75)
            yield from self._move_joint(q_mid, grip, settle=0, want_vision=False)
        yield from self._move_joint(q_v, grip, settle=2, want_vision=False)
        yield self._emit_hold(grip, want_vision=True)

    # ------------------------------------------------------------------ perception helpers

    def _select(
        self,
        goal: GoalSpec,
        *,
        min_confidence: float = 0.25,
        ignore_seated: bool = True,
    ) -> PartEstimate | None:
        if self.scene is None:
            return None
        return select_target_part(
            self.scene,
            goal,
            min_confidence=min_confidence,
            ignore_seated=ignore_seated,
            exclude_positions=self._decoy_positions,
        )

    def _observe_target(
        self,
        goal: GoalSpec,
        *,
        grip: float = 0.75,
        min_samples: int = 1,
        max_wait: int = 5,
    ) -> Iterator[StepCommand]:
        """Collect filtered vision observations of `goal`'s target part without moving."""
        for _ in range(max_wait):
            cand = self._select(goal)
            if cand is not None:
                self.tracker.update_part(cand)
                if self.tracker.part_count >= min_samples:
                    return
            yield self._emit_hold(grip, want_vision=True)

    def _get_filtered_fixture(self) -> tuple[dict[str, np.ndarray], float, np.ndarray]:
        if self.scene is None:
            from .scene_interface import normalize_scene
            return self.tracker.filtered_fixture(normalize_scene({}))
        return self.tracker.filtered_fixture(self.scene)

    def _clear_table_spot(self, exclude_xy: np.ndarray | None = None) -> np.ndarray:
        """Choose a free table XY on the -y side away from fixture and detected parts."""
        others: list[np.ndarray] = []
        if self.scene is not None:
            for p in self.scene.parts:
                if exclude_xy is not None and float(np.linalg.norm(p.position[:2] - exclude_xy)) < 0.03:
                    continue
                others.append(p.position[:2])

        best_xy = np.array(shifted((0.66, -0.22)), dtype=np.float64)
        best_score = -1e9
        for x in np.linspace(0.56, 0.84, 8) + LAYOUT_SHIFT[0]:
            for y in np.linspace(-0.34, -0.14, 7) + LAYOUT_SHIFT[1]:
                p = np.array([x, y], dtype=np.float64)
                if math.hypot(x, y) > 0.95:
                    continue
                d_fix = self.scene.fixture_distance(p) if self.scene is not None else 0.12
                if d_fix < 0.075:
                    continue
                d_oth = min((float(np.linalg.norm(p - o)) for o in others), default=1.0)
                score = min(d_fix, d_oth) - 0.1 * float(np.linalg.norm(p - np.array(shifted((0.70, -0.22)))))
                if score > best_score:
                    best_score = score
                    best_xy = p
        return best_xy

    def _hover_and_descend_with_nudge_watch(
        self,
        goal: GoalSpec,
        compute_grasp_fn,
        *,
        attempt: int = 0,
    ) -> Iterator[StepCommand]:
        """Hover over target, refine pose with vision, then descend watching for nudges.

        `compute_grasp_fn(part, seats, fyaw, ref_yaw)` returns
        `(grasp_pos, grasp_rot, g_open, hover_z, meta_dict)`.
        Stores final grasp result in `self._last_grasp_meta`.
        """
        part = self.tracker.filtered_part()
        assert part is not None
        seats, fyaw, _ = self._get_filtered_fixture()
        ref_yaw = self._cur_ee_yaw()
        grasp_pos, grasp_rot, g_open, hover_z, meta = compute_grasp_fn(part, seats, fyaw, ref_yaw)

        # Small deterministic retry offset if previous attempt missed
        retry_offsets = (
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, -0.002]),
            np.array([0.004, 0.0, -0.002]),
            np.array([-0.004, 0.004, -0.002]),
        )
        r_off = retry_offsets[attempt % len(retry_offsets)]

        # 1. Lift clear first if currently low
        yield from self._ensure_safe_height(max(0.24, hover_z - 0.02), g_open)

        # 2. Move to hover above part (no need to run vision during transit before arrival)
        hover_pos = np.array([grasp_pos[0], grasp_pos[1], hover_z], dtype=np.float64)
        q_hover = self.ik.solve(hover_pos, grasp_rot, self.q_target).q
        yield from self._move_joint(q_hover, g_open, settle=2, want_vision=False)
        yield self._emit_hold(g_open, want_vision=True)

        # 3. Collect 1 clean close-range frame at hover (arm settled; avoids redundant identical renders)
        self.tracker.reset_part()
        yield from self._observe_target(goal, grip=g_open, min_samples=1, max_wait=3)

        refined = self.tracker.filtered_part()
        if refined is not None and refined.confidence >= 0.45:
            # Keep original estimate if hover view was degraded/occluded by the gripper itself
            if float(np.linalg.norm(refined.position[:2] - part.position[:2])) < 0.08 or refined.confidence >= 0.65:
                part = refined
        seats, fyaw, _ = self._get_filtered_fixture()
        grasp_pos, grasp_rot, g_open, hover_z, meta = compute_grasp_fn(part, seats, fyaw, self._cur_ee_yaw())
        grasp_pos = grasp_pos + r_off

        # Re-align at hover if refined xy shifted > 4 mm
        cur_p, _ = fk_pose(self.q_target)
        if float(np.linalg.norm(grasp_pos[:2] - cur_p[:2])) > 0.004:
            hover_pos = np.array([grasp_pos[0], grasp_pos[1], hover_z], dtype=np.float64)
            yield from self._move_cartesian(hover_pos, grasp_rot, g_open, pos_speed=0.020, settle=2, want_vision=False)

        # 4. Descend vertically to grasp_pos, with a single midpoint nudge check before gripper occlusion
        nudge_recoveries = 0
        while True:
            pos0, rot0 = fk_pose(self.q_target)
            dist = float(np.linalg.norm(grasp_pos - pos0))
            n_seg = max(1, int(math.ceil(dist / 0.014)))
            mid_seg = max(1, n_seg // 3)
            q_prev = self.q_target.copy()
            nudged = False
            checked_mid = False

            for i in range(1, n_seg + 1):
                t = i / n_seg
                p_t = pos0 + t * (grasp_pos - pos0)
                r_t = slerp_rot(rot0, grasp_rot, t)
                q_t = self.ik.solve(p_t, r_t, q_prev).q
                q_prev = q_t

                do_check_now = (
                    (not checked_mid)
                    and (i == mid_seg)
                    and (p_t[2] > grasp_pos[2] + 0.045)
                    and (nudge_recoveries < 2)
                )
                for k_step in range(25):
                    if float(np.max(np.abs(q_t - self.q_target))) <= 2e-4:
                        break
                    req_vis = do_check_now and (k_step == 0)
                    yield self._emit_toward(q_t, g_open, max_step=0.042, want_vision=req_vis)
                    if do_check_now and (k_step == 1):
                        checked_mid = True
                        live = self._select(goal, min_confidence=0.60)
                        if live is not None:
                            shift = float(np.linalg.norm(live.position[:2] - part.position[:2]))
                            if shift > 0.025 and live.confidence >= 0.65:
                                nudged = True
                                nudge_recoveries += 1
                                break
                if nudged:
                    break

            if not nudged:
                # Settle briefly at grasp pose
                for _ in range(4):
                    yield self._emit_toward(q_prev, g_open, max_step=0.035, want_vision=False)
                    err = float(np.max(np.abs(self.proprio["joint_pos"] - q_prev)))
                    vel = float(np.max(np.abs(self.proprio["joint_vel"])))
                    if err < 0.008 and vel < 0.04:
                        break
                break

            # Recover from nudge: lift back to hover and re-estimate
            cur_p, cur_r = fk_pose(self.q_target)
            yield from self._move_cartesian(
                np.array([cur_p[0], cur_p[1], hover_z], dtype=np.float64),
                cur_r,
                g_open,
                pos_speed=0.025,
                settle=1,
                want_vision=False,
            )
            yield self._emit_hold(g_open, want_vision=True)
            self.tracker.reset_part()
            yield from self._observe_target(goal, grip=g_open, min_samples=1, max_wait=4)
            updated = self.tracker.filtered_part()
            if updated is not None:
                part = updated
            seats, fyaw, _ = self._get_filtered_fixture()
            grasp_pos, grasp_rot, g_open, hover_z, meta = compute_grasp_fn(part, seats, fyaw, self._cur_ee_yaw())
            hover_pos = np.array([grasp_pos[0], grasp_pos[1], hover_z], dtype=np.float64)
            yield from self._move_cartesian(hover_pos, grasp_rot, g_open, pos_speed=0.022, settle=2, want_vision=False)

        meta = dict(meta)
        meta["part"] = part
        meta["grasp_ee_pos"] = self.proprio["ee_pos"].copy()
        meta["grasp_rot"] = grasp_rot.copy()
        self._last_grasp_meta = meta

    # ------------------------------------------------------------------ Skill 1: Tray pick-and-place

    def _compute_tray_grasp(
        self,
        goal: GoalSpec,
        part: PartEstimate,
        seats: dict[str, np.ndarray],
        fyaw: float,
        ref_yaw: float,
    ) -> tuple[np.ndarray, np.ndarray, float, float, dict[str, Any]]:
        pos = part.position
        fyaw = reach_yaw(fyaw)  # headings below were designed for a yaw-0 fixture (see cell.reach_yaw)
        if goal.part_kind == "washer":
            geom = FASTENER_GEOM.get(goal.part_model, FASTENER_GEOM["washer_m33"])
            grasp_yaw = nearest_pi_yaw(fyaw, ref_yaw)
            # a flat washer's centre sits at half its thickness; camera depth from the low views reads it high
            z_c = min(float(pos[2]), 0.5 * geom["thickness"] + 0.002)
            # pads at table level (they stop on it): a 5 mm PLA washer is pinched only by its lower rim
            z_tip = max(z_c - 0.5 * geom["thickness"], 0.0) + PAD_TIP_WASHER - float(os.environ.get("RC_WASHER_DZ", "0.0025"))
            grasp_pos = np.array([float(pos[0]), float(pos[1]), z_tip], dtype=np.float64)
            return grasp_pos, down_mat(grasp_yaw), 0.82, 0.24, {"mode": "washer", "grasp_yaw": grasp_yaw}

        # Bolt (M30x60 or generic)
        geom = FASTENER_GEOM.get(goal.part_model, FASTENER_GEOM["bolt_m30x60"])
        mode = part.pose_hint
        if mode == "standing":
            # Standing on head, shank points straight up: grip upper shank above head
            grasp_yaw = nearest_pi_yaw(fyaw + math.pi / 2.0, ref_yaw)
            z_tip = geom["head_height"] + 0.015 + PAD_TIP_BOLT_M30
            grasp_pos = np.array([float(pos[0]), float(pos[1]), z_tip], dtype=np.float64)
            return grasp_pos, down_mat(grasp_yaw), 0.65, 0.27, {"mode": "standing", "grasp_yaw": grasp_yaw}

        if mode == "leaning":
            # Leaning against fixture -y face: point +x toward fixture wall so camera (-x) stays away
            # Fixture -y inward normal is [-sin(fyaw), cos(fyaw)]
            ax_xy = part.axis[:2]
            if float(np.linalg.norm(ax_xy)) > 0.15:
                # Ensure we point gripper +x into the fixture wall (+y_fixture)
                wall_in = np.array([-math.sin(fyaw), math.cos(fyaw)], dtype=np.float64)
                dir_xy = ax_xy / float(np.linalg.norm(ax_xy))
                if float(np.dot(dir_xy, wall_in)) < 0.0:
                    dir_xy = -dir_xy
                grasp_yaw = math.atan2(float(dir_xy[1]), float(dir_xy[0]))
            else:
                grasp_yaw = math.atan2(math.cos(fyaw), -math.sin(fyaw))
            z_tip = max(float(pos[2]) - 0.010, 0.022) + PAD_TIP_BOLT_M30
            grasp_pos = np.array([float(pos[0]), float(pos[1]), z_tip], dtype=np.float64)
            return grasp_pos, down_mat(grasp_yaw), 0.65, 0.27, {"mode": "leaning", "grasp_yaw": grasp_yaw}

        # Lying bolt: align gripper +x along bolt horizontal axis so fingers pinch across shank
        ax_xy = part.axis[:2]
        if float(np.linalg.norm(ax_xy)) > 0.15:
            bolt_yaw = math.atan2(float(ax_xy[1]), float(ax_xy[0]))
        else:
            bolt_yaw = part.yaw
        grasp_yaw = nearest_pi_yaw(bolt_yaw, ref_yaw)
        # If axis is explicitly directed (tip->head), shift +3 mm toward head on shank; else stay at center
        p_shank = pos.copy()
        if part.axis_directed:
            p_shank = pos + 0.003 * part.axis
        z_axis = min(float(p_shank[2]), float(geom.get("lying_radius", 0.5 * geom["diameter"])))  # geometry, not camera depth
        z_tip = max(z_axis - 0.5 * geom["diameter"], 0.004) + PAD_TIP_BOLT_M30
        grasp_pos = np.array([float(p_shank[0]), float(p_shank[1]), z_tip], dtype=np.float64)
        return (
            grasp_pos,
            down_mat(grasp_yaw),
            0.65,
            0.25,
            {"mode": "lying", "grasp_yaw": grasp_yaw, "bolt_yaw": bolt_yaw},
        )

    def _execute_tray_goal(self, goal: GoalSpec) -> Iterator[StepCommand]:
        max_attempts = 4
        min_grip = _min_grip_for_part(goal.part_model)

        for attempt in range(max_attempts):
            # 1. Observe target part (move to clear vantage if not yet detected or on retry)
            self.tracker.reset_part()
            if attempt > 0:
                yield from self._move_to_vantage(grip=0.75)
            yield from self._observe_target(goal, grip=0.75, min_samples=1, max_wait=3)

            if self.tracker.filtered_part() is None:
                # Could be occluded by parked gripper at HOME_Q -> move to clear vantage and look again
                yield from self._move_to_vantage(grip=0.75)
                yield from self._observe_target(goal, grip=0.75, min_samples=1, max_wait=4)

            part = self.tracker.filtered_part()
            if part is None:
                # Check if target part is already seated in tray
                if self.scene is not None:
                    all_kind = [p for p in self.scene.parts if p.kind == goal.part_kind]
                    if all_kind and all(is_part_in_seat(p, goal.seat_id, self.scene) for p in all_kind):
                        return
                continue

            # 2. Hover, refine, descend (with nudge watch)
            yield from self._hover_and_descend_with_nudge_watch(
                goal,
                lambda p, s, fy, ry: self._compute_tray_grasp(goal, p, s, fy, ry),
                attempt=attempt,
            )
            meta = self._last_grasp_meta
            mode = meta["mode"]

            # 3. Close gripper
            close_steps = 16 if goal.part_kind == "washer" else 14
            yield from self._grip_hold(0.0, steps=close_steps, early_stop=True, want_vision=False)

            # 4. Lift and verify grasp purely via proprio gripper reading
            cur_p, cur_r = fk_pose(self.q_target)
            lift_z = 0.26 if goal.part_kind == "washer" else 0.295
            yield from self._move_cartesian(
                np.array([cur_p[0], cur_p[1], lift_z], dtype=np.float64),
                cur_r,
                0.0,
                pos_speed=0.026,
                settle=1,
            )

            g_now = self.proprio["gripper"]
            if g_now < min_grip or g_now > 0.78:
                # Missed grasp or dropped during lift -> open and retry
                yield from self._grip_hold(0.75, steps=8, early_stop=True, want_vision=True)
                continue

            # 5. Carry over tray center, aligning lying bolt with tray's 222 mm long axis
            seats, fyaw, _ = self._get_filtered_fixture()
            tray_xyz = seats["tray"]
            cur_yaw = self._cur_ee_yaw()
            if goal.part_kind == "bolt" and mode == "lying":
                # Since gripper +x was aligned with bolt axis at grasp, aligning +x with tray long axis
                # (fyaw + pi/2) puts the 78.7 mm bolt lengthwise inside the 222 mm tray!
                place_yaw = nearest_pi_yaw(fyaw + math.pi / 2.0, cur_yaw)
            else:
                place_yaw = cur_yaw

            place_rot = down_mat(place_yaw)
            above_tray = np.array([float(tray_xyz[0]), float(tray_xyz[1]), lift_z], dtype=np.float64)
            yield from self._move_cartesian(
                above_tray, place_rot, 0.0, pos_speed=0.028, rot_speed=0.18, settle=2
            )

            # Re-verify still holding part before lowering
            if self.proprio["gripper"] < min_grip:
                yield from self._grip_hold(0.75, steps=8, early_stop=True, want_vision=True)
                continue

            # 6. Lower gently into tray
            if goal.part_kind == "washer":
                drop_z = TRAY_FLOOR_Z + 0.006 + PAD_TIP_WASHER
            elif mode == "standing":
                drop_z = TRAY_FLOOR_Z + 0.036 + PAD_TIP_BOLT_M30
            elif mode == "leaning":
                drop_z = TRAY_FLOOR_Z + 0.032 + PAD_TIP_BOLT_M30
            else:
                drop_z = TRAY_FLOOR_Z + 0.018 + PAD_TIP_BOLT_M30

            lower_pos = np.array([float(tray_xyz[0]), float(tray_xyz[1]), drop_z], dtype=np.float64)
            yield from self._move_cartesian(
                lower_pos, place_rot, 0.0, pos_speed=0.018, max_step=0.040, settle=2
            )

            # 7. Release and retract straight up
            g_rel = 0.82 if goal.part_kind == "washer" else 0.72
            yield from self._grip_hold(g_rel, steps=10, early_stop=True, want_vision=False)
            retract_pos = np.array([float(tray_xyz[0]), float(tray_xyz[1]), lift_z], dtype=np.float64)
            yield from self._move_cartesian(
                retract_pos, place_rot, g_rel, pos_speed=0.028, settle=1, want_vision=False
            )

            # 8. Hold for HOLD_STEPS settling and check via vision once settled
            for i in range(10):
                yield self._emit_hold(g_rel, want_vision=(i == 8))

            still_out = self._select(goal, min_confidence=0.60, ignore_seated=True)
            if still_out is not None:
                # Part didn't land inside tray -> retry
                continue
            return

    # ------------------------------------------------------------------ Skill 2: M16 bolt -> bore_m16

    def _unlean_bolt(self, part: PartEstimate, fyaw: float) -> Iterator[StepCommand]:
        """Grasp a leaning M16 bolt off the fixture -y face and lay it flat on clear table."""
        pos = part.position
        axis = part.axis.copy()
        fyaw = reach_yaw(fyaw)  # lean face = the block face toward world -y
        # Ensure axis points down & away from fixture wall (toward head)
        w_out = np.array([math.sin(fyaw), -math.cos(fyaw), 0.0], dtype=np.float64)
        if float(axis[2]) > 0.0:
            axis = -axis
        if float(np.linalg.norm(axis)) < 1e-6:
            axis = normalize(np.array([0.5 * w_out[0], 0.5 * w_out[1], -0.866], dtype=np.float64))
        else:
            axis = normalize(axis)

        p_grasp = pos - 0.002 * axis
        x_site = axis  # +X along head (down-out) => wrist camera (-X) points UP-in, clear of table
        z_site = normalize((-w_out) - float(np.dot(-w_out, x_site)) * x_site)
        y_site = np.cross(z_site, x_site)
        rot_grasp = np.column_stack([x_site, y_site, z_site])

        p_site_grasp = p_grasp - 0.129 * z_site
        p_site_pre = p_site_grasp - 0.075 * z_site + np.array([0.0, 0.0, 0.03], dtype=np.float64)
        grip_open = 0.46

        yield from self._ensure_safe_height(0.24, grip_open)
        q_pre = self.ik.solve(p_site_pre, rot_grasp, self.q_target).q
        yield from self._move_joint(q_pre, grip_open, settle=3)
        yield from self._move_cartesian(p_site_grasp, rot_grasp, grip_open, pos_speed=0.012, settle=3)
        yield from self._grip_hold(0.0, steps=14, early_stop=True)

        # Pull back from fixture wall and lift
        p_pull = p_site_grasp - 0.070 * z_site + np.array([0.0, 0.0, 0.095], dtype=np.float64)
        yield from self._move_cartesian(p_pull, rot_grasp, 0.0, pos_speed=0.018, settle=1)

        if self.proprio["gripper"] < EMPTY_GRIPPER_THRESH:
            # Missed leaning grasp; open and back away so we can re-observe (or bolt already toppled!)
            yield from self._grip_hold(0.55, steps=8, early_stop=True, want_vision=True)
            yield from self._ensure_safe_height(0.26, 0.55)
            return

        # Carry to clear spot on -y table and lay flat
        drop_xy = self._clear_table_spot(exclude_xy=pos[:2])
        rot_flat = site_rot_from_zx(np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0]))
        p_above = np.array([float(drop_xy[0]), float(drop_xy[1]), 0.25], dtype=np.float64)
        q_above = self.ik.solve(p_above, rot_flat, self.q_target).q
        yield from self._move_joint(q_above, 0.0, settle=2)

        p_drop = np.array([float(drop_xy[0]), float(drop_xy[1]), 0.146], dtype=np.float64)
        yield from self._move_cartesian(p_drop, rot_flat, 0.0, pos_speed=0.015, settle=3)
        yield from self._grip_hold(0.55, steps=10, early_stop=True)
        yield from self._move_cartesian(p_above, rot_flat, 0.55, pos_speed=0.025, settle=1, want_vision=False)
        yield from self._grip_hold(0.55, steps=6, early_stop=False, want_vision=True)

    def _compute_lying_m16_grasp(
        self,
        part: PartEstimate,
        seats: dict[str, np.ndarray],
        fyaw: float,
        ref_yaw: float,
    ) -> tuple[np.ndarray, np.ndarray, float, float, dict[str, Any]]:
        pos = part.position
        ax_xy = np.array([float(part.axis[0]), float(part.axis[1]), 0.0], dtype=np.float64)
        if float(np.linalg.norm(ax_xy)) < 1e-4:
            ax_xy = np.array([math.cos(part.yaw), math.sin(part.yaw), 0.0], dtype=np.float64)
        else:
            ax_xy = normalize(ax_xy)

        # Grasp on shank near head (+4 mm along +axis toward head => 2.5 mm clear of head bearing face).
        # Point gripperframe -X along +axis (head) so wrist camera (-X) stays UP on vertical insertion!
        d_along = 0.004
        p_grasp = pos + d_along * ax_xy
        rot_grasp = site_rot_from_zx(np.array([0.0, 0.0, -1.0]), -ax_xy)
        z_grasp = max(0.002, min(float(pos[2]), FASTENER_GEOM["bolt_m16x50"]["lying_radius"]) - 0.008) + PAD_TIP_BOLT_M16
        grasp_pos = np.array([float(p_grasp[0]), float(p_grasp[1]), z_grasp], dtype=np.float64)
        hover_z = max(0.27, z_grasp + 0.12)
        return grasp_pos, rot_grasp, 0.50, hover_z, {"d_along": d_along, "axis_xy": ax_xy}

    def _contact_descend_bolt(
        self,
        s_pre: np.ndarray,
        s_ins: np.ndarray,
        rot_insert: np.ndarray,
    ) -> Iterator[StepCommand]:
        """Vertical descent into bore_m16 with proprio contact-stall detection and spiral search."""
        # Spiral XY offsets (metres) in case vision/grasp offset puts tip slightly on bore rim
        offsets = (
            (0.0, 0.0),
            (0.005, 0.0),
            (-0.005, 0.0),
            (0.0, 0.005),
            (0.0, -0.005),
            (0.006, 0.006),
            (-0.006, -0.006),
            (0.006, -0.006),
            (-0.006, 0.006),
        )
        for idx, (ox, oy) in enumerate(offsets):
            p_start = s_pre + np.array([ox, oy, 0.0], dtype=np.float64)
            p_end = s_ins + np.array([ox, oy, 0.0], dtype=np.float64)
            if idx > 0:
                # Lift slightly above mouth and shift to next spiral point
                cur_p, _ = fk_pose(self.q_target)
                p_lift = np.array([cur_p[0], cur_p[1], max(float(cur_p[2]) + 0.014, float(s_pre[2]) - 0.006)])
                yield from self._move_cartesian(
                    p_lift, rot_insert, 0.0, pos_speed=0.016, max_step=0.035, settle=1, wrist_up=True
                )
                yield from self._move_cartesian(
                    np.array([p_start[0], p_start[1], p_lift[2]]),
                    rot_insert,
                    0.0,
                    pos_speed=0.016,
                    max_step=0.035,
                    settle=2,
                    wrist_up=True,
                )

            pos0, rot0 = fk_pose(self.q_target)
            dist = float(np.linalg.norm(p_end - pos0))
            n_seg = max(1, int(math.ceil(dist / 0.006)))
            q_prev = self.q_target.copy()
            stall_steps = 0
            stalled = False

            for i in range(1, n_seg + 1):
                t = i / n_seg
                p_t = pos0 + t * (p_end - pos0)
                r_t = slerp_rot(rot0, rot_insert, t)
                q_t = self.ik.solve(p_t, r_t, q_prev, wrist_up=True).q
                q_prev = q_t
                for _ in range(30):
                    if float(np.max(np.abs(q_t - self.q_target))) <= 2e-4:
                        break
                    yield self._emit_toward(q_t, 0.0, max_step=0.030, want_vision=False)
                    # Check proprio vertical lag (contact with fixture top around hole mouth)
                    cmd_p, _ = fk_pose(self.q_target)
                    z_lag = float(self.proprio["ee_pos"][2] - cmd_p[2])
                    if z_lag > 0.0075 and cmd_p[2] > p_end[2] + 0.008:
                        stall_steps += 1
                        if stall_steps >= 6:
                            stalled = True
                            break
                    else:
                        stall_steps = max(0, stall_steps - 1)
                if stalled:
                    break

            if not stalled:
                # Reached insertion depth freely! Settle cleanly.
                for _ in range(5):
                    yield self._emit_toward(q_prev, 0.0, max_step=0.025, want_vision=False)
                    err = float(np.max(np.abs(self.proprio["joint_pos"] - q_prev)))
                    vel = float(np.max(np.abs(self.proprio["joint_vel"])))
                    if err < 0.008 and vel < 0.04:
                        break
                return

    def _execute_bolt_hole_goal(self, goal: GoalSpec) -> Iterator[StepCommand]:
        geom = FASTENER_GEOM.get(goal.part_model, FASTENER_GEOM["bolt_m16x50"])
        seat_geom = SEAT_LOCAL_GEOM.get(goal.seat_id, SEAT_LOCAL_GEOM["bore_m16"])
        min_grip = _min_grip_for_part(goal.part_model)

        for attempt in range(3):
            self.tracker.reset_part()
            if attempt > 0:
                yield from self._move_to_vantage(grip=0.55)
            yield from self._observe_target(goal, grip=0.55, min_samples=1, max_wait=3)
            if self.tracker.filtered_part() is None:
                yield from self._move_to_vantage(grip=0.55)
                yield from self._observe_target(goal, grip=0.55, min_samples=1, max_wait=4)

            part = self.tracker.filtered_part()
            if part is None:
                return
            seats, fyaw, _ = self._get_filtered_fixture()

            # 1. If leaning or standing, lay bolt flat on clear table first, then re-observe
            if part.pose_hint in ("leaning", "standing"):
                yield from self._unlean_bolt(part, fyaw)
                self.tracker.reset_part()
                yield from self._observe_target(goal, grip=0.55, min_samples=1, max_wait=4)
                part = self.tracker.filtered_part()
                if part is None:
                    continue
                seats, fyaw, _ = self._get_filtered_fixture()

            # 2. Hover, refine, descend onto lying bolt shank
            yield from self._hover_and_descend_with_nudge_watch(
                goal, self._compute_lying_m16_grasp, attempt=attempt
            )
            meta = self._last_grasp_meta
            d_along = float(meta["d_along"])
            ee_at_grasp = meta["grasp_ee_pos"]

            # 3. Close gripper & lift to verify grasp
            yield from self._grip_hold(0.0, steps=14, early_stop=True)
            cur_p, cur_r = fk_pose(self.q_target)
            yield from self._move_cartesian(
                np.array([cur_p[0], cur_p[1], 0.28], dtype=np.float64),
                cur_r,
                0.0,
                pos_speed=0.024,
                settle=2,
            )
            if self.proprio["gripper"] < min_grip or self.proprio["gripper"] > 0.65:
                yield from self._grip_hold(0.55, steps=8, early_stop=True, want_vision=True)
                continue

            # 4. Compute exact insertion pose using pure proprio + known M16 bolt geometry
            seats, fyaw, _ = self._get_filtered_fixture()
            seat_xyz = seats[goal.seat_id]
            top_z = float(seat_geom["top_z"])
            # Bolt shank center on table is ~0.0103 m above table; ee_at_grasp[2] gives exact z_s
            z_s_held = float(np.clip(ee_at_grasp[2] - 0.0103, 0.120, 0.138))
            p_s_b = np.array([d_along, 0.0, z_s_held], dtype=np.float64)

            # Insertion orientation: bolt +axis (-x_site) -> [0, 0, +1] (so x_site = [0, 0, -1]),
            # z_site horizontal along +x_fixture (from tray toward block)
            approach_xy = np.array([math.cos(reach_yaw(fyaw)), math.sin(reach_yaw(fyaw)), 0.0], dtype=np.float64)
            rot_insert = site_rot_from_zx(approach_xy, np.array([0.0, 0.0, -1.0]))
            offset_s_to_b = rot_insert @ p_s_b

            # Keep lowest pad edge >= top_z + 2 mm
            pad_z_drop = (
                abs(float(rot_insert[2, 0])) * PAD_HALF_WIDTH_X
                + abs(float(rot_insert[2, 1])) * 0.012
                + max(0.0, -float(rot_insert[2, 2])) * PAD_TIP_BOLT_M16
            )
            min_site_z = top_z + 0.002 + pad_z_drop

            # Pre-insert: bolt tip 25 mm above fixture top, centered on bore
            b_pre = np.array([float(seat_xyz[0]), float(seat_xyz[1]), top_z + 0.025 - geom["tip_x"]])
            s_pre = b_pre - offset_s_to_b

            # Seated bolt depth (head hex corners sit in 8 mm 45 deg chamfer)
            corner = geom["across_flats"] / math.sqrt(3.0)
            head_sink = float(
                np.clip(0.5 * seat_geom["diameter"] + seat_geom["chamfer"] - corner, 0.0, seat_geom["chamfer"])
            )
            b_seated_z = top_z - head_sink - geom["under_head_x"]
            s_ins_z = max(min_site_z, b_seated_z - float(offset_s_to_b[2]) + 0.002)
            s_ins = np.array(
                [float(seat_xyz[0]) - float(offset_s_to_b[0]), float(seat_xyz[1]) - float(offset_s_to_b[1]), s_ins_z],
                dtype=np.float64,
            )

            # Solve wrist-up IK backwards from s_ins -> s_pre -> s_transit to stay on ONE wrist-up branch
            q_ins_ref = self.ik.solve(s_ins, rot_insert, self.q_target, wrist_up=True).q
            q_pre_ref = self.ik.solve(s_pre, rot_insert, q_ins_ref, wrist_up=True).q
            s_transit = np.array([s_pre[0], s_pre[1], 0.275], dtype=np.float64)
            q_transit = self.ik.solve(s_transit, rot_insert, q_pre_ref, wrist_up=True).q

            # Safe elevated midpoint if flipping wrist_2 sign across 0
            q_cur = self.q_target.copy()
            if q_cur[4] * q_transit[4] < 0.0:
                q_mid = 0.5 * (q_cur + q_transit)
                q_mid[1] = min(float(q_mid[1]), -1.55)
                q_mid[2] = min(float(q_mid[2]), 1.75)
                yield from self._move_joint(q_mid, 0.0, settle=0)

            yield from self._move_joint(q_transit, 0.0, settle=2)
            yield from self._move_cartesian(s_pre, rot_insert, 0.0, pos_speed=0.016, settle=3, wrist_up=True)

            # 5. Contact-guided vertical insertion into bore_m16
            yield from self._contact_descend_bolt(s_pre, s_ins, rot_insert)

            # 6. Release cleanly and retract straight up
            yield from self._grip_hold(0.65, steps=12, early_stop=True)
            yield from self._move_cartesian(s_transit, rot_insert, 0.65, pos_speed=0.025, settle=1, wrist_up=True)

            # Move to clear vantage (which captures 1 fresh unobstructed vision frame) and verify bolt is seated
            yield from self._move_to_vantage(grip=0.65)
            still_out = self._select(goal, min_confidence=0.60, ignore_seated=True)
            if still_out is not None and float(still_out.position[1]) < 0.05 + LAYOUT_SHIFT[1]:
                # Target small bolt fell back onto the -y table -> retry!
                continue
            return

    # ------------------------------------------------------------------ Skill 3: M30 nut -> hex_m30

    def _tip_edge_nut(
        self,
        part: PartEstimate,
        model: str | None = None,
        yaw_offset: float = 0.0,
        axis_override: np.ndarray | None = None,
    ) -> Iterator[StepCommand]:
        """Tip an on-edge nut flat onto the table with a size-parametric horizontal sweep across its upper face."""
        nut_key = model or (part.raw_kind if part.raw_kind in FASTENER_GEOM else "nut_m30")
        geom = FASTENER_GEOM.get(nut_key, FASTENER_GEOM["nut_m30"])
        af = float(geom.get("across_flats", 0.046))
        th = float(geom.get("thickness", 0.0256))

        grip_push = float(np.clip(0.03 + 8.0 * th, 0.19, 0.28))  # inner gap < nut thickness (M24->0.20, M30->0.235, M36->0.28)
        if axis_override is None:
            # Hover over the nut with parted fingers so the wrist camera refines its XY (and axis when still seen on_edge)
            yield from self._ensure_safe_height(0.25, grip_push)
            p_look = np.array([float(part.position[0]), float(part.position[1]), 0.28], dtype=np.float64)
            q_look = self.ik.solve(p_look, down_mat(0.0), self.q_target).q
            yield from self._move_joint(q_look, grip_push, settle=2, want_vision=False)
            yield self._emit_hold(grip_push, want_vision=True)
            yield self._emit_hold(grip_push, want_vision=False)
            if self.scene is not None:
                near_same = [
                    p for p in self.scene.parts
                    if p.kind == "nut" and p.confidence >= 0.30
                    and float(np.linalg.norm(p.position[:2] - part.position[:2])) < 0.08
                    and (p.raw_kind == nut_key or nut_key not in FASTENER_GEOM)
                ]
                if near_same:
                    refined = min(near_same, key=lambda p: float(np.linalg.norm(p.position[:2] - part.position[:2])))
                    if refined.pose_hint == "on_edge":
                        part = refined
                    else:
                        # Overhead wrist view can look square at 45 deg; keep original on_edge axis, refine XY only
                        part = PartEstimate(
                            kind=part.kind, raw_kind=part.raw_kind,
                            position=np.array([float(refined.position[0]), float(refined.position[1]), float(part.position[2])]),
                            axis=part.axis.copy(), yaw=part.yaw, confidence=refined.confidence,
                            role=part.role, pose_hint="on_edge", axis_directed=part.axis_directed, extra=dict(part.extra),
                        )

        pos = part.position
        if axis_override is not None and float(np.linalg.norm(axis_override[:2])) > 1e-4:
            n_ax = normalize(np.array([float(axis_override[0]), float(axis_override[1]), 0.0], dtype=np.float64))
        else:
            n_ax = np.array([float(part.axis[0]), float(part.axis[1]), 0.0], dtype=np.float64)
            if float(np.linalg.norm(n_ax)) < 1e-4:
                ang = part.yaw + yaw_offset
                n_ax = np.array([math.cos(ang), math.sin(ang), 0.0], dtype=np.float64)
            else:
                n_ax = normalize(n_ax)
                if abs(yaw_offset) > 1e-6:
                    c_o, s_o = math.cos(yaw_offset), math.sin(yaw_offset)
                    n_ax = np.array([c_o * n_ax[0] - s_o * n_ax[1], s_o * n_ax[0] + c_o * n_ax[1], 0.0], dtype=np.float64)

        # Pick push direction along +-n_ax staying on table and clear of fixture/other parts
        others = []
        if self.scene is not None:
            for p in self.scene.parts:
                if float(np.linalg.norm(p.position[:2] - pos[:2])) > 0.03:
                    others.append(p.position[:2])

        best_dir = -n_ax
        best_score = -1e9
        for sign in (-1.0, 1.0):
            d = sign * n_ax
            cand = pos[:2] + 0.05 * d[:2]
            in_bounds = (
                0.48 + LAYOUT_SHIFT[0] <= cand[0] <= 0.98 + LAYOUT_SHIFT[0]
                and -0.36 + LAYOUT_SHIFT[1] <= cand[1] <= -0.08 + LAYOUT_SHIFT[1]
                and math.hypot(float(cand[0]), float(cand[1])) <= 0.99
            )
            d_fix = self.scene.fixture_distance(cand) if self.scene is not None else 0.12
            d_oth = min((float(np.linalg.norm(cand - o)) for o in others), default=1.0)
            score = (10.0 if in_bounds else 0.0) + min(d_fix, d_oth) - 0.5 * abs(float(cand[0]) - (0.85 + LAYOUT_SHIFT[0]))
            if score > best_score:
                best_score = score
                best_dir = d

        rot_push = site_rot_from_zx(np.array([0.0, 0.0, -1.0]), best_dir)
        z_site = max(0.014, 0.68 * af) + pad_tip_for_gap(0.75 * th)
        d_start = max(0.052, 0.5 * th + 0.038)
        d_end = max(0.044, 0.90 * af)
        p_start = np.array([pos[0] - d_start * best_dir[0], pos[1] - d_start * best_dir[1], z_site], dtype=np.float64)
        p_end = np.array([pos[0] + d_end * best_dir[0], pos[1] + d_end * best_dir[1], z_site], dtype=np.float64)
        p_hover = np.array([p_start[0], p_start[1], 0.25], dtype=np.float64)

        q_hover = self.ik.solve(p_hover, rot_push, self.q_target).q
        yield from self._move_joint(q_hover, grip_push, settle=2)
        yield from self._move_cartesian(p_start, rot_push, grip_push, pos_speed=0.020, settle=2)
        yield from self._move_cartesian(p_end, rot_push, grip_push, pos_speed=0.015, settle=2)
        tipped_xy = np.array([pos[0] + 0.025 * best_dir[0], pos[1] + 0.025 * best_dir[1]], dtype=np.float64)
        self._last_tip_xy = tipped_xy.copy()
        yield from self._move_cartesian(
            np.array([tipped_xy[0], tipped_xy[1], 0.27], dtype=np.float64), down_mat(0.0), 0.72, pos_speed=0.025, settle=2,
            want_vision=False,
        )
        yield from self._grip_hold(0.72, steps=6, early_stop=False, want_vision=True)

    def _compute_flat_nut_grasp(
        self,
        part: PartEstimate,
        seats: dict[str, np.ndarray],
        fyaw: float,
        ref_yaw: float,
    ) -> tuple[np.ndarray, np.ndarray, float, float, dict[str, Any]]:
        pos = part.position
        cur_y_yaw = self._cur_y_yaw()
        # Align finger-opening (+Y) with one of the nut's 6 flat normals
        grasp_y_yaw = closest_hex_yaw(cur_y_yaw, part.yaw)
        y_site = np.array([math.cos(grasp_y_yaw), math.sin(grasp_y_yaw), 0.0], dtype=np.float64)
        rot_grasp = site_rot_from_zy(np.array([0.0, 0.0, -1.0]), y_site)

        # Grip upper 11 mm of 25.6 mm nut so fingertips stay > top_z (0.075) when nut bottom hits floor_z (0.062)
        z_grasp = 0.0145 + PAD_TIP_NUT_FLAT
        grasp_pos = np.array([float(pos[0]), float(pos[1]), z_grasp], dtype=np.float64)
        return grasp_pos, rot_grasp, 0.78, 0.25, {"grasp_y_yaw": grasp_y_yaw}

    def _contact_descend_nut(
        self,
        s_pre: np.ndarray,
        s_place: np.ndarray,
        des_y_yaw: float,
    ) -> Iterator[StepCommand]:
        """Descend nut into hex_m30 with proprio contact-stall detection and yaw/XY seating wiggle."""
        # Try nominal pose first, then small yaw wiggles (+-7 deg, +-13 deg) and +-3 mm XY offsets
        trials = (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, math.radians(7.0)),
            (0.0, 0.0, math.radians(-7.0)),
            (0.0035, 0.0, 0.0),
            (-0.0035, 0.0, 0.0),
            (0.0, 0.0035, 0.0),
            (0.0, -0.0035, 0.0),
            (0.0, 0.0, math.radians(14.0)),
            (0.0, 0.0, math.radians(-14.0)),
        )
        for idx, (ox, oy, dyaw) in enumerate(trials):
            y_yaw = des_y_yaw + dyaw
            y_site = np.array([math.cos(y_yaw), math.sin(y_yaw), 0.0], dtype=np.float64)
            rot_place = site_rot_from_zy(np.array([0.0, 0.0, -1.0]), y_site)
            p_end = s_place + np.array([ox, oy, 0.0], dtype=np.float64)

            if idx > 0:
                cur_p, _ = fk_pose(self.q_target)
                p_lift = np.array([cur_p[0], cur_p[1], max(float(cur_p[2]) + 0.010, float(s_place[2]) + 0.015)])
                yield from self._move_cartesian(p_lift, rot_place, 0.0, pos_speed=0.016, max_step=0.035, settle=1)
                yield from self._move_cartesian(
                    np.array([p_end[0], p_end[1], p_lift[2]]),
                    rot_place,
                    0.0,
                    pos_speed=0.016,
                    max_step=0.035,
                    settle=2,
                )

            pos0, rot0 = fk_pose(self.q_target)
            dist = float(np.linalg.norm(p_end - pos0))
            n_seg = max(1, int(math.ceil(dist / 0.006)))
            q_prev = self.q_target.copy()
            stall_steps = 0
            stalled = False

            for i in range(1, n_seg + 1):
                t = i / n_seg
                p_t = pos0 + t * (p_end - pos0)
                r_t = slerp_rot(rot0, rot_place, t)
                q_t = self.ik.solve(p_t, r_t, q_prev).q
                q_prev = q_t
                for _ in range(30):
                    if float(np.max(np.abs(q_t - self.q_target))) <= 2e-4:
                        break
                    yield self._emit_toward(q_t, 0.0, max_step=0.030, want_vision=False)
                    cmd_p, _ = fk_pose(self.q_target)
                    z_lag = float(self.proprio["ee_pos"][2] - cmd_p[2])
                    if z_lag > 0.0055 and cmd_p[2] > p_end[2] + 0.004:
                        stall_steps += 1
                        if stall_steps >= 6:
                            stalled = True
                            break
                    else:
                        stall_steps = max(0, stall_steps - 1)
                if stalled:
                    break

            if not stalled:
                for _ in range(5):
                    yield self._emit_toward(q_prev, 0.0, max_step=0.025, want_vision=False)
                    err = float(np.max(np.abs(self.proprio["joint_pos"] - q_prev)))
                    vel = float(np.max(np.abs(self.proprio["joint_vel"])))
                    if err < 0.008 and vel < 0.04:
                        break
                return

    def _execute_nut_pocket_goal(self, goal: GoalSpec) -> Iterator[StepCommand]:
        seat_geom = SEAT_LOCAL_GEOM.get(goal.seat_id, SEAT_LOCAL_GEOM["hex_m30"])
        min_grip = _min_grip_for_part(goal.part_model)

        for attempt in range(3):
            self.tracker.reset_part()
            if attempt > 0:
                yield from self._move_to_vantage(grip=0.72)
            yield from self._observe_target(goal, grip=0.72, min_samples=1, max_wait=3)
            if self.tracker.filtered_part() is None:
                yield from self._move_to_vantage(grip=0.72)
                yield from self._observe_target(goal, grip=0.72, min_samples=1, max_wait=4)

            part = self.tracker.filtered_part()
            if part is None:
                return

            # 1. If nut stands on edge, tip it flat first and re-observe
            if part.pose_hint == "on_edge":
                yield from self._tip_edge_nut(part)
                self.tracker.reset_part()
                yield from self._observe_target(goal, grip=0.72, min_samples=1, max_wait=4)
                part = self.tracker.filtered_part()
                if part is None:
                    continue

            # 2. Hover, refine, descend onto upper 11 mm of flat nut
            yield from self._hover_and_descend_with_nudge_watch(
                goal, self._compute_flat_nut_grasp, attempt=attempt
            )

            # 3. Close gripper (parallel jaws self-align nut flats to gripper +Y!) & lift
            yield from self._grip_hold(0.0, steps=14, early_stop=True)
            cur_p, cur_r = fk_pose(self.q_target)
            yield from self._move_cartesian(
                np.array([cur_p[0], cur_p[1], 0.27], dtype=np.float64),
                cur_r,
                0.0,
                pos_speed=0.022,
                settle=2,
            )
            if self.proprio["gripper"] < min_grip or self.proprio["gripper"] > 0.72:
                yield from self._grip_hold(0.75, steps=8, early_stop=True, want_vision=True)
                continue

            # 4. Align gripper +Y (which is now the grasped nut's flat normal!) with pocket flat normal
            seats, fyaw, _ = self._get_filtered_fixture()
            seat_xyz = seats[goal.seat_id]
            top_z = float(seat_geom["top_z"])
            flat_normal_yaw = math.pi / 2.0 + fyaw
            des_y_yaw = closest_hex_yaw(self._cur_y_yaw(), flat_normal_yaw)
            y_site = np.array([math.cos(des_y_yaw), math.sin(des_y_yaw), 0.0], dtype=np.float64)
            rot_place = site_rot_from_zy(np.array([0.0, 0.0, -1.0]), y_site)

            # Fingertips stay 1.5 mm above top_z (0.075), placing nut bottom ~0.0 mm above floor_z (0.062)
            s_place_z = top_z + 0.0015 + PAD_TIP_NUT_FLAT
            s_place = np.array([float(seat_xyz[0]), float(seat_xyz[1]), s_place_z], dtype=np.float64)
            s_pre = np.array([float(seat_xyz[0]), float(seat_xyz[1]), top_z + 0.045 + PAD_TIP_NUT_FLAT])
            s_transit = np.array([float(seat_xyz[0]), float(seat_xyz[1]), 0.295], dtype=np.float64)

            # Use smooth Cartesian transit while holding the nut so arm never flips IK branches mid-carry
            yield from self._move_cartesian(
                s_transit, rot_place, 0.0, pos_speed=0.025, rot_speed=0.16, settle=2
            )
            yield from self._move_cartesian(s_pre, rot_place, 0.0, pos_speed=0.016, settle=3)

            # 5. Contact-guided hex seating
            yield from self._contact_descend_nut(s_pre, s_place, des_y_yaw)

            # 6. Release, lift slightly, then perform a gentle blunt-finger top-seat press
            #    (settles any nut perched 2-6 mm high on the 4 mm hex pocket lead-in chamfer)
            yield from self._grip_hold(0.80, steps=10, early_stop=True)
            s_clear_top = np.array([float(seat_xyz[0]), float(seat_xyz[1]), top_z + 0.042 + PAD_TIP_CLOSED])
            yield from self._move_cartesian(s_clear_top, rot_place, 0.80, pos_speed=0.024, settle=1)
            yield from self._grip_hold(0.0, steps=10, early_stop=True)
            # Seated M30 nut top face is floor_z (0.062) + thickness (0.0256) = 0.0876 m.
            # Pressing closed fingertip to 0.0885 m settles perched nuts without jamming seated ones.
            s_tap = np.array([float(seat_xyz[0]), float(seat_xyz[1]), float(seat_geom["floor_z"]) + 0.0265 + PAD_TIP_CLOSED])
            yield from self._move_cartesian(
                s_tap, rot_place, 0.0, pos_speed=0.010, max_step=0.030, settle=2
            )
            yield from self._move_cartesian(s_transit, rot_place, 0.80, pos_speed=0.026, settle=1, want_vision=False)

            # Hold briefly to let nut settle and verify via vision once settled
            for i in range(10):
                yield self._emit_hold(0.80, want_vision=(i == 8))
            still_out = self._select(goal, min_confidence=0.60, ignore_seated=True)
            if still_out is not None and float(still_out.position[1]) < 0.05 + LAYOUT_SHIFT[1]:
                continue
            return

    # ------------------------------------------------------------------ top-level episode loop

    def _run_episode(self) -> Iterator[StepCommand]:
        # Let step 0 observation settle into tracker
        for goal in self.goals:
            seat_kind = SEAT_LOCAL_GEOM.get(goal.seat_id, {}).get("kind", "tray")
            if seat_kind == "tray":
                yield from self._execute_tray_goal(goal)
            elif seat_kind == "hole_vertical":
                yield from self._execute_bolt_hole_goal(goal)
            elif seat_kind == "hex_pocket":
                yield from self._execute_nut_pocket_goal(goal)
            else:
                yield from self._execute_tray_goal(goal)

        # Final clean hold for HOLD_STEPS completion
        while True:
            yield self._emit_hold(0.80, want_vision=False)
