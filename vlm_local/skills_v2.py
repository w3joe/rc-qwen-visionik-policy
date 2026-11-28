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

from .classical.kinematics import (PAD_TIP_NUT_FLAT, closest_hex_yaw, fk_pose, site_rot_from_zx, site_rot_from_zy)
from .classical.planner import StepCommand, _min_grip_for_part
from .classical.scene_interface import FASTENER_GEOM, SEAT_LOCAL_GEOM
from .finisher import (HeldPart, PlacementFinisher, bolt_axis_height_when_lying, bolt_grasp_d_along, bolt_pad_tip)

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
        z_grasp = max(0.002, float(pos[2]) - 0.008) + bolt_pad_tip(model)
        grasp_pos = np.array([float(p_grasp[0]), float(p_grasp[1]), z_grasp])
        return grasp_pos, rot_grasp, 0.50, max(0.27, z_grasp + 0.12), {"d_along": d_along, "axis_xy": ax_xy, "model": model}

    def _compute_flat_nut_grasp_v2(self, part, seats, fyaw, ref_yaw, model: str | None = None):
        model = model or _nut_model(None, part)
        th = FASTENER_GEOM[model]["thickness"]
        pad_bottom = 0.0145 if th > 0.0145 + 0.004 else max(0.006, th - 0.007)   # as the validated M30 skill: grip the upper part
        grasp_y_yaw = closest_hex_yaw(self._cur_y_yaw(), part.yaw)
        rot = site_rot_from_zy(np.array([0.0, 0.0, -1.0]), np.array([math.cos(grasp_y_yaw), math.sin(grasp_y_yaw), 0.0]))
        pos = part.position
        grasp_pos = np.array([float(pos[0]), float(pos[1]), pad_bottom + PAD_TIP_NUT_FLAT])
        return grasp_pos, rot, 0.78, 0.25, {"grasp_y_yaw": grasp_y_yaw, "pad_bottom": pad_bottom, "model": model}

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
        min_grip = _min_grip_for_part(goal.part_model)
        placements = 0
        for attempt in range(4):
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
            if part.pose_hint == "on_edge":
                yield from self._tip_edge_nut(part)
                self.tracker.reset_part()
                yield from self._observe_target(goal, grip=0.72, min_samples=1, max_wait=4)
                part = self.tracker.filtered_part()
                if part is None:
                    continue
            model = _nut_model(goal, part)
            yield from self._hover_and_descend_with_nudge_watch(
                goal, lambda p, s, fy, ry, m=model: self._compute_flat_nut_grasp_v2(p, s, fy, ry, m), attempt=attempt)
            meta = self._last_grasp_meta
            yield from self._grip_hold(0.0, steps=14, early_stop=True)
            cur_p, cur_r = fk_pose(self.q_target)
            yield from self._move_cartesian(np.array([cur_p[0], cur_p[1], 0.27]), cur_r, 0.0, pos_speed=0.022, settle=2)
            if self.proprio["gripper"] < min_grip or self.proprio["gripper"] > 0.72:
                yield from self._grip_hold(0.75, steps=8, early_stop=True, want_vision=True)
                continue
            held = HeldPart(model=model, kind="nut", ee_at_grasp=np.asarray(meta["grasp_ee_pos"], float),
                            nut_pad_bottom=float(meta["pad_bottom"]))
            res = yield from PlacementFinisher(self).finish(held, goal.seat_id, goal)
            self._log("finish", goal=getattr(goal, "goal_id", "?"), ok=res.ok, why=res.reason)
            self.finisher_results.append((getattr(goal, "goal_id", "?"), res))
            placements += 1
            if res.ok or placements >= MAX_PLACE_TRIES:
                return
