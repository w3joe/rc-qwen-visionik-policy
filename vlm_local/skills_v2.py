"""Size-aware grasp + place skills for bolt->bore and nut->hex-pocket, built on `finisher.PlacementFinisher`.

`PlacementV2Mixin` overrides `_execute_bolt_hole_goal` / `_execute_nut_pocket_goal` of the vision-IK controller.
The grasp stages (observe, unlean/tip, hover-refine-descend, close, lift, grip check) are the old ones, parametrised by the
part model; everything from "held part above the seat" to "released, retreated, verified" is the finisher.
Placement failures: re-grasp and retry once (verification said the part is not in the seat), then give up cleanly.
"""
from __future__ import annotations

import math
import os
from typing import Any, Iterator

import numpy as np

from .classical.kinematics import (PAD_TIP_NUT_FLAT, closest_hex_yaw, down_mat, fk_pose, site_rot_from_zx, site_rot_from_zy)
from .classical.planner import EMPTY_GRIPPER_THRESH, StepCommand, _min_grip_for_part
from .classical.scene_interface import FASTENER_GEOM, FIXTURE_BLOCK_TOP_Z, SEAT_LOCAL_GEOM
from .finisher import (HeldPart, PlacementFinisher, bolt_axis_height_when_lying, bolt_grasp_d_along, bolt_pad_tip, nut_pad_tip)

PLACEMENT_V2 = os.environ.get("PLACEMENT_V2", "1") not in ("0", "", "false", "False")
PLACEMENT_V2_BOLT = os.environ.get("PLACEMENT_V2_BOLT", "1") not in ("0", "", "false", "False")  # ON since the real-cell port: the old bolt skill never inserts there (no search, no verification)
MAX_PLACE_TRIES = 2          # one placement + one re-grasp/retry


def _bolt_model(goal: Any, part: Any = None) -> str:
    for m in (getattr(part, "raw_kind", None), getattr(goal, "part_model", None)):
        if m in FASTENER_GEOM and m.startswith("bolt"):
            return m
    return "bolt_m16x50"


def _nut_model(goal: Any, part: Any = None) -> str:
    for m in (getattr(part, "raw_kind", None), getattr(goal, "part_model", None)):
        if m in FASTENER_GEOM and m.startswith("nut"):
            return m
    seat = str(getattr(goal, "seat_id", ""))
    cand = "nut_" + seat.split("_")[-1]
    return cand if cand in FASTENER_GEOM else "nut_m30"


class PlacementV2Mixin:
    placement_v2: bool = PLACEMENT_V2
    finisher_results: list

    # ---------------------------------------------------------------- generic grasps
    def _compute_lying_bolt_grasp(self, part, seats, fyaw, ref_yaw, model: str | None = None):
        model = model or _bolt_model(None, part)
        pos = part.position
        ax_xy = np.array([float(part.axis[0]), float(part.axis[1]), 0.0])
        if float(np.linalg.norm(ax_xy)) < 1e-4:
            ax_xy = np.array([math.cos(part.yaw), math.sin(part.yaw), 0.0])
        else:
            ax_xy = ax_xy / np.linalg.norm(ax_xy)
        d_along = bolt_grasp_d_along(model)
        p_grasp = pos + d_along * ax_xy
        rot_grasp = site_rot_from_zx(np.array([0.0, 0.0, -1.0]), -ax_xy)
        # A lying bolt rests at a known height (shank on the table, tilted up by its head): camera depth from the low
        # side views overestimates it by ~2 cm, so the pads closed above the shank. Trust geometry, cap by perception.
        z_axis = min(float(pos[2]), float(FASTENER_GEOM.get(model, FASTENER_GEOM["bolt_m16x50"]).get("lying_radius", 0.012)))
        z_grasp = max(0.002, z_axis - 0.008) + bolt_pad_tip(model)
        grasp_pos = np.array([float(p_grasp[0]), float(p_grasp[1]), z_grasp])
        return grasp_pos, rot_grasp, 0.50, max(0.27, z_grasp + 0.12), {"d_along": d_along, "axis_xy": ax_xy, "model": model}

    def _compute_flat_nut_grasp_v2(self, part, seats, fyaw, ref_yaw, model: str | None = None, seat_id: str | None = None):
        model = model or _nut_model(None, part)
        geom = FASTENER_GEOM.get(model, FASTENER_GEOM["nut_m30"])
        th = float(geom["thickness"])
        af = float(geom["across_flats"])
        sg = SEAT_LOCAL_GEOM.get(seat_id or "") if seat_id else None
        if sg is not None and sg.get("kind") == "hex_pocket":
            pocket_depth = float(sg["top_z"] - sg["floor_z"])
            pad_bottom = float(np.clip(pocket_depth + 0.0015, 0.006, th - 0.0085))
        else:
            pad_bottom = float(np.clip(0.55 * th, 0.006, th - 0.008))
        grasp_y_yaw = closest_hex_yaw(self._cur_y_yaw(), part.yaw)
        rot = site_rot_from_zy(np.array([0.0, 0.0, -1.0]), np.array([math.cos(grasp_y_yaw), math.sin(grasp_y_yaw), 0.0]))
        pos = part.position
        # Flat nut on the table sits on z=0 (ignore noisy side-camera z); only lift z_surf when perched on the fixture block
        on_block = float(pos[2]) >= 0.085 and self.scene is not None and self.scene.fixture_distance(pos[:2]) < 0.025
        z_surf = (FIXTURE_BLOCK_TOP_Z - 0.002) if on_block else 0.0
        pad_tip = nut_pad_tip(model)
        g_open = float(np.clip(0.32 + 8.0 * af, 0.58, 0.80))
        hover_z = max(0.25, z_surf + pad_bottom + pad_tip + 0.095)
        grasp_pos = np.array([float(pos[0]), float(pos[1]), z_surf + pad_bottom + pad_tip])
        return grasp_pos, rot, g_open, hover_z, {"grasp_y_yaw": grasp_y_yaw, "pad_bottom": pad_bottom, "model": model, "g_open": g_open}

    # ---------------------------------------------------------------- bolt -> bore
    def _execute_bolt_hole_goal(self, goal) -> Iterator[StepCommand]:
        if not (self.placement_v2 and PLACEMENT_V2_BOLT):
            yield from super()._execute_bolt_hole_goal(goal)
            return
        min_grip = _min_grip_for_part(goal.part_model)
        placements = 0
        for attempt in range(4):
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
            if part.pose_hint in ("leaning", "standing"):
                yield from self._unlean_bolt(part, fyaw)
                self.tracker.reset_part()
                yield from self._observe_target(goal, grip=0.55, min_samples=1, max_wait=4)
                part = self.tracker.filtered_part()
                if part is None:
                    continue
            model = _bolt_model(goal, part)
            yield from self._hover_and_descend_with_nudge_watch(
                goal, lambda p, s, fy, ry, m=model: self._compute_lying_bolt_grasp(p, s, fy, ry, m), attempt=attempt)
            meta = self._last_grasp_meta
            yield from self._grip_hold(0.0, steps=14, early_stop=True)
            cur_p, cur_r = fk_pose(self.q_target)
            yield from self._move_cartesian(np.array([cur_p[0], cur_p[1], 0.28]), cur_r, 0.0, pos_speed=0.024, settle=2)
            if self.proprio["gripper"] < min_grip or self.proprio["gripper"] > 0.65:
                yield from self._grip_hold(0.55, steps=8, early_stop=True, want_vision=True)
                continue
            held = HeldPart(model=model, kind="bolt", ee_at_grasp=np.asarray(meta["grasp_ee_pos"], float),
                            d_along=float(meta["d_along"]), axis_height=bolt_axis_height_when_lying(model))
            res = yield from PlacementFinisher(self).finish(held, goal.seat_id, goal)
            self._log("finish", goal=getattr(goal, "goal_id", "?"), ok=res.ok, why=res.reason)
            self.finisher_results.append((getattr(goal, "goal_id", "?"), res))
            placements += 1
            if res.ok or placements >= MAX_PLACE_TRIES:
                return

    # ---------------------------------------------------------------- nut -> hex pocket
    def _execute_nut_pocket_goal(self, goal) -> Iterator[StepCommand]:
        if not self.placement_v2:
            yield from super()._execute_nut_pocket_goal(goal)
            return
        placements = 0
        for attempt in range(4):
            self.tracker.reset_part()
            if attempt > 0:
                yield from self._move_to_vantage(grip=0.72)
            yield from self._observe_target(goal, grip=0.72, min_samples=1, max_wait=3)
            if self.tracker.filtered_part() is None:
                yield from self._move_to_vantage(grip=0.72)
                yield from self._observe_target(goal, grip=0.72, min_samples=1, max_wait=4)
            if self.tracker.filtered_part() is None and getattr(goal, "target_xy", None) is not None:
                # Arm at vantage can occlude parts behind the fixture from the left camera; look directly over target_xy
                tx, ty = float(goal.target_xy[0]), float(goal.target_xy[1])
                yield from self._ensure_safe_height(0.25, 0.72)
                q_look = self.ik.solve(np.array([tx, ty, 0.26]), down_mat(0.0), self.q_target).q
                yield from self._move_joint(q_look, 0.72, settle=2, want_vision=False)
                yield self._emit_hold(0.72, want_vision=True)
                yield from self._observe_target(goal, grip=0.72, min_samples=1, max_wait=3)
            part = self.tracker.filtered_part()
            if part is None:
                return
            model = _nut_model(goal, part)
            for tip_try in range(2):
                if part.pose_hint != "on_edge":
                    break
                yield from self._tip_edge_nut(part, model=model, yaw_offset=(0.0 if tip_try == 0 else 0.5 * math.pi))
                self.tracker.reset_part()
                yield from self._observe_target(goal, grip=0.72, min_samples=1, max_wait=4)
                part = self.tracker.filtered_part()
                if part is None:
                    break
            if part is None or part.pose_hint == "on_edge":
                continue
            model = _nut_model(goal, part)
            min_grip = _min_grip_for_part(model)
            yield from self._hover_and_descend_with_nudge_watch(
                goal, lambda p, s, fy, ry, m=model, sid=goal.seat_id: self._compute_flat_nut_grasp_v2(p, s, fy, ry, m, sid),
                attempt=attempt,
            )
            meta = self._last_grasp_meta
            g_open = float(meta.get("g_open", 0.75))
            yield from self._grip_hold(0.0, steps=14, early_stop=True)
            cur_p, cur_r = fk_pose(self.q_target)
            yield from self._move_cartesian(np.array([cur_p[0], cur_p[1], 0.27]), cur_r, 0.0, pos_speed=0.022, settle=2)
            g_now = float(self.proprio["gripper"])
            if g_now < min_grip or g_now > min(0.74, g_open - 0.02):
                if EMPTY_GRIPPER_THRESH < g_now < min_grip:
                    # Pinched across thickness (nut was on edge): set back down gently before opening
                    yield from self._move_cartesian(np.array([cur_p[0], cur_p[1], 0.175]), cur_r, 0.0, pos_speed=0.020, settle=1)
                yield from self._grip_hold(g_open, steps=8, early_stop=True, want_vision=True)
                continue
            held = HeldPart(model=model, kind="nut", ee_at_grasp=np.asarray(meta["grasp_ee_pos"], float),
                            nut_pad_bottom=float(meta["pad_bottom"]))
            res = yield from PlacementFinisher(self).finish(held, goal.seat_id, goal)
            self._log("finish", goal=getattr(goal, "goal_id", "?"), ok=res.ok, why=res.reason)
            self.finisher_results.append((getattr(goal, "goal_id", "?"), res))
            placements += 1
            if not res.ok and "perched" in res.reason and getattr(goal, "target_xy", None) is not None:
                seats_now, _, _ = self._get_filtered_fixture()
                if goal.seat_id in seats_now:
                    goal.target_xy = np.asarray(seats_now[goal.seat_id], float)[:2].copy()
            if res.ok or placements >= MAX_PLACE_TRIES:
                return
