"""Vision-IK motion library driven by a planner-supplied goal list (+ a new relative/stacking placement skill).

`VLMEpisodeController` subclasses the vision-IK `EpisodeController` and changes only:
  * where the goals come from (planner instead of `parse_task_goals`),
  * how a goal's target part is selected in the perception output (nearest to the planner's chosen part, not the
    instruction-keyword roles),
  * the dispatch: tray / hole / pocket goals use the original skills; nut->tray, washer->hole mouth and relative
    placement (`on` / `next_to`) use `_execute_place_goal`, a generic pick-carry-drop skill built from the same
    primitives (hover/refine/descend, grasp verification by gripper reading, Cartesian carry).
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterator

import numpy as np

from .classical.kinematics import (PAD_TIP_BOLT_M30, PAD_TIP_CLOSED, PAD_TIP_NUT_FLAT, PAD_TIP_WASHER, down_mat, fk_pose,
                                   nearest_pi_yaw)
from .classical.cell import LAYOUT_SHIFT
from .classical.planner import EpisodeController, StepCommand, _min_grip_for_part
from .classical.scene_interface import (FASTENER_GEOM, SEAT_LOCAL_GEOM, TRAY_FLOOR_Z, PartEstimate, is_part_in_seat,
                                        select_target_part)
from .plan import PlannedGoal
from .skills_v2 import PlacementV2Mixin

TRACK_RADIUS = 0.10          # a tracked part may have moved this far between detections
# top surface heights (m) a relative destination offers
TABLE_Z = 0.0


class VLMEpisodeController(PlacementV2Mixin, EpisodeController):
    def __init__(self, first_obs: dict[str, Any], goals: list) -> None:
        super().__init__(first_obs)
        self.goals = list(goals)
        self.instruction = str(first_obs.get("instruction") or "")
        self.events: list[dict] = []
        self.finisher_results: list = []

    def _log(self, kind: str, **kw: Any) -> None:
        ev = {"step": self.step_idx, "event": kind, **kw}
        self.events.append(ev)
        try:
            import json as _j, sys as _s
            print("[ev] " + _j.dumps(ev, default=str)[:400], file=_s.stderr, flush=True)
        except Exception:
            pass

    # ------------------------------------------------------------------ target selection
    def _select(self, goal, *, min_confidence: float = 0.25, ignore_seated: bool = True) -> PartEstimate | None:
        if self.scene is None:
            return None
        if not isinstance(goal, PlannedGoal) or goal.target_xy is None:
            return super()._select(goal, min_confidence=min_confidence, ignore_seated=ignore_seated)
        # Track the planner's chosen part. Model labels from perception are reliable for bolts (and for the goal's own
        # part after it has been unleaned/tipped, which moves it by tens of cm), so prefer same-model candidates at any
        # distance; otherwise the nearest same-kind detection within TRACK_RADIUS.
        best = None
        for floor in (min_confidence, 0.15):  # same relaxation as the original selector
            pool = [p for p in self.scene.parts if p.kind == goal.part_kind and p.confidence >= floor
                    and not (ignore_seated and goal.seat_id and is_part_in_seat(p, goal.seat_id, self.scene))]
            same = [p for p in pool if p.raw_kind == goal.part_model]
            if goal.part_kind == "bolt" and same:
                pool, radius = same, 0.5
            else:
                radius = TRACK_RADIUS
            best_d = radius
            for p in pool:
                d = float(np.linalg.norm(p.position[:2] - goal.target_xy))
                if d < best_d:
                    best, best_d = p, d
            if best is not None:
                break
        if best is not None and best.confidence >= 0.4:
            goal.target_xy = best.position[:2].copy()  # follow the part if it was nudged
        return best

    # ------------------------------------------------------------------ grasp for any part kind
    def _grasp_fn(self, goal):
        if goal.part_kind == "nut":
            return self._compute_flat_nut_grasp
        return lambda p, s, fy, ry: self._compute_tray_grasp(goal, p, s, fy, ry)

    def _surface_z(self, goal, seats) -> float:
        """Height of the top surface the carried part will rest on at the destination."""
        if goal.relation == "on" and goal.ref_part_kind == "bolt":
            geom = FASTENER_GEOM.get(goal.ref_model or "bolt_m30x60", FASTENER_GEOM["bolt_m30x60"])
            return geom["head_height"]  # washer resting on the head of a bolt standing on its head
        if goal.seat_id in SEAT_LOCAL_GEOM:
            g = SEAT_LOCAL_GEOM[goal.seat_id]
            return float(g["floor_z"] if g["kind"] == "tray" else g["top_z"])
        if goal.relation == "on":
            return 0.03
        return TABLE_Z

    def _dest_xy(self, goal, seats) -> np.ndarray:
        if goal.seat_id in seats:
            return np.asarray(seats[goal.seat_id], float)[:2]
        ref_xy = goal.ref_part_xy
        if goal.ref_part_xy is not None and self.scene is not None:  # refresh from the latest detection
            best = min((p for p in self.scene.parts if p.kind == goal.ref_part_kind),
                       key=lambda p: float(np.linalg.norm(p.position[:2] - goal.ref_part_xy)), default=None)
            if best is not None and float(np.linalg.norm(best.position[:2] - goal.ref_part_xy)) < 0.06:
                ref_xy = best.position[:2]
        if ref_xy is None and goal.ref_seat in seats:
            ref_xy = np.asarray(seats[goal.ref_seat], float)[:2]
        if ref_xy is None:
            ref_xy = np.asarray(seats["tray"], float)[:2]
        if goal.relation == "on":
            return np.asarray(ref_xy, float)
        return self._free_spot_near(np.asarray(ref_xy, float), goal)

    def _free_spot_near(self, ref_xy: np.ndarray, goal) -> np.ndarray:
        """A table spot ~9 cm from ref, away from every other detected part and the fixture, inside the easy-reach box."""
        others = [p.position[:2] for p in (self.scene.parts if self.scene else [])
                  if float(np.linalg.norm(p.position[:2] - ref_xy)) > 0.02 and float(np.linalg.norm(p.position[:2] - (goal.target_xy if goal.target_xy is not None else ref_xy))) > 0.02]
        best, best_s = ref_xy + np.array([0.0, -0.09]), -1e9
        for ang in np.linspace(0, 2 * math.pi, 16, endpoint=False):
            for r in (0.09, 0.11):
                c = ref_xy + r * np.array([math.cos(ang), math.sin(ang)])
                if not (0.45 + LAYOUT_SHIFT[0] <= c[0] <= 1.0 + LAYOUT_SHIFT[0] and -0.42 + LAYOUT_SHIFT[1] <= c[1] <= 0.42 + LAYOUT_SHIFT[1]) or math.hypot(*c) > 1.0:
                    continue
                d_fix = self.scene.fixture_distance(c) if self.scene is not None else 0.2
                d_oth = min((float(np.linalg.norm(c - o)) for o in others), default=1.0)
                s = min(d_fix - 0.05, d_oth - 0.06, 0.1) - 0.2 * abs(r - 0.09)
                if s > best_s:
                    best, best_s = c, s
        return best

    # ------------------------------------------------------------------ generic pick -> carry -> drop skill
    def _execute_place_goal(self, goal) -> Iterator[StepCommand]:
        min_grip = _min_grip_for_part(goal.part_model)
        for attempt in range(4):
            self.tracker.reset_part()
            if attempt > 0:
                yield from self._move_to_vantage(grip=0.75)
            yield from self._observe_target(goal, grip=0.75, min_samples=1, max_wait=3)
            if self.tracker.filtered_part() is None:
                yield from self._move_to_vantage(grip=0.75)
                yield from self._observe_target(goal, grip=0.75, min_samples=1, max_wait=4)
            part = self.tracker.filtered_part()
            if part is None:
                self._log("place_no_part", goal=goal.goal_id)
                return
            if goal.part_kind == "nut" and part.pose_hint == "on_edge":
                yield from self._tip_edge_nut(part)
                self.tracker.reset_part()
                yield from self._observe_target(goal, grip=0.75, min_samples=1, max_wait=4)
                part = self.tracker.filtered_part()
                if part is None:
                    continue
            yield from self._hover_and_descend_with_nudge_watch(goal, self._grasp_fn(goal), attempt=attempt)
            meta = self._last_grasp_meta
            mode = meta.get("mode", "flat")
            yield from self._grip_hold(0.0, steps=16 if goal.part_kind == "washer" else 14, early_stop=True)
            cur_p, cur_r = fk_pose(self.q_target)
            lift_z = 0.28 if goal.part_kind != "bolt" else 0.295
            yield from self._move_cartesian(np.array([cur_p[0], cur_p[1], lift_z]), cur_r, 0.0, pos_speed=0.026, settle=1)
            g_now = self.proprio["gripper"]
            if g_now < min_grip or g_now > 0.78:
                self._log("place_grasp_failed", gripper=round(float(g_now), 3), attempt=attempt)
                yield from self._grip_hold(0.75, steps=8, early_stop=True, want_vision=True)
                continue
            # fresh look from the lifted pose (destination part may have moved), then carry
            yield self._emit_hold(0.0, want_vision=True)
            seats, fyaw, _ = self._get_filtered_fixture()
            dxy = self._dest_xy(goal, seats)
            surf = self._surface_z(goal, seats)
            cur_yaw = self._cur_ee_yaw()
            place_yaw = nearest_pi_yaw(fyaw + math.pi / 2.0, cur_yaw) if (goal.part_kind == "bolt" and mode == "lying") else cur_yaw
            rot = down_mat(place_yaw)
            yield from self._move_cartesian(np.array([dxy[0], dxy[1], lift_z]), rot, 0.0, pos_speed=0.028, rot_speed=0.18, settle=2)
            if self.proprio["gripper"] < min_grip:
                yield from self._grip_hold(0.75, steps=8, early_stop=True, want_vision=True)
                continue
            # lowest allowed palm z: part bottom ~ surface + clearance
            if goal.part_kind == "washer":
                drop_z = surf + 0.006 + PAD_TIP_WASHER + (0.004 if goal.relation == "on" else 0.0)
            elif goal.part_kind == "nut":
                drop_z = surf + 0.0145 + 0.010 + PAD_TIP_NUT_FLAT
            elif mode == "standing":
                drop_z = surf + 0.036 + PAD_TIP_BOLT_M30
            elif mode == "leaning":
                drop_z = surf + 0.032 + PAD_TIP_BOLT_M30
            else:
                drop_z = surf + 0.018 + PAD_TIP_BOLT_M30
            yield from self._move_cartesian(np.array([dxy[0], dxy[1], drop_z]), rot, 0.0, pos_speed=0.018, max_step=0.040, settle=2)
            g_rel = 0.82 if goal.part_kind in ("washer", "nut") else 0.72
            yield from self._grip_hold(g_rel, steps=10, early_stop=True)
            yield from self._move_cartesian(np.array([dxy[0], dxy[1], lift_z]), rot, g_rel, pos_speed=0.028, settle=1)
            self._log("placed", goal=goal.goal_id, dest_xy=[round(float(dxy[0]), 3), round(float(dxy[1]), 3)], surf=surf)
            for i in range(10):
                yield self._emit_hold(g_rel, want_vision=(i == 8))
            return

    # ------------------------------------------------------------------ dispatch
    @staticmethod
    def _execution_order(goals: list, instruction: str = "") -> list:
        """Skill-library prior: with a bolt->hole and a nut->pocket goal and no stacking relation, insert the bolt first
        (the nut sitting in its pocket gets knocked by the insertion approach otherwise; the baseline always does this).
        An instruction that states an order ("first", "then", "before", "after") keeps the planner's order: ordered tasks
        only count placements made in order."""
        if any(getattr(g, "relation", None) for g in goals):
            return goals
        if re.search(r"\b(first|then|before|after|afterwards|followed|next|finally)\b", instruction.lower()):
            return goals
        def rank(g):
            kind = SEAT_LOCAL_GEOM.get(getattr(g, "seat_id", ""), {}).get("kind")
            return 0 if (kind == "hole_vertical" and g.part_kind == "bolt") else (2 if kind == "hex_pocket" else 1)
        seats = [getattr(g, "seat_id", "") for g in goals]
        if len(set(seats)) < len(seats):  # two goals over one seat (washer then bolt) are order-sensitive: keep as planned
            return goals
        return sorted(goals, key=rank)

    def _run_episode(self) -> Iterator[StepCommand]:
        self.goals = self._execution_order(self.goals, self.instruction)
        for goal in self.goals:
            seat = SEAT_LOCAL_GEOM.get(getattr(goal, "seat_id", ""), {})
            kind = seat.get("kind")
            rel = getattr(goal, "relation", None)
            if rel:
                yield from self._execute_place_goal(goal)
            elif kind == "tray" and goal.part_kind in ("bolt", "washer"):
                yield from self._execute_tray_goal(goal)
            elif kind == "tray":
                yield from self._execute_place_goal(goal)          # nut -> tray
            elif kind == "hole_vertical" and goal.part_kind == "bolt":
                yield from self._execute_bolt_hole_goal(goal)
            elif kind == "hole_vertical":
                yield from self._execute_place_goal(goal)          # washer over a hole mouth
            elif kind == "hex_pocket" and goal.part_kind == "nut":
                yield from self._execute_nut_pocket_goal(goal)
            else:
                yield from self._execute_place_goal(goal)
            self._log("goal_done", goal=getattr(goal, "goal_id", "?"))
        while True:
            yield self._emit_hold(0.80, want_vision=False)
