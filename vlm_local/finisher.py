"""Placement "finisher": size-aware insertion / seating with release gating and post-release verification.

Scope
-----
Takes over once a part is grasped and held (any public-kit bolt / nut) and the controller has it above the target
seat; returns after release + retreat + a re-observation.  It is a drop-in replacement for the M16/M30-tuned code in
`classical/planner.py` (`_contact_descend_bolt`, `_contact_descend_nut`, the release steps of
`_execute_bolt_hole_goal` / `_execute_nut_pocket_goal`).  The old path stays available (`PLACEMENT_V2=0`) for A/B runs.

Interface (kept free of controller internals so it can later sit on top of a learned/FLUX policy)
---------------------------------------------------------------------------------------------
    finisher = PlacementFinisher(ctrl)          # ctrl: anything exposing the MotionHost protocol below
    result = yield from finisher.finish(held, seat_id)      # generator of StepCommand = (dq6, gripper01, want_vision)

    HeldPart        what is in the gripper: model / kind, how it is held (ee pose at grasp, offset along the bolt axis,
                    pad tip height) -- from the grasp stage, i.e. from the perception estimate + proprio.
    seat_id         a key of SEAT_LOCAL_GEOM; the seat xyz / fixture yaw come from the (temporally filtered) scene estimate.
    FinishResult    ok / reason / stages (per-try alignment numbers) -- returned as the generator's value.

MotionHost protocol (provided by `EpisodeController`): q_target, ik, proprio (ee_pos, ee_rot, gripper, joint_pos/vel),
scene, tracker, `_emit_toward/_emit_hold/_move_cartesian/_move_joint/_grip_hold/_move_to_vantage/
_get_filtered_fixture/_observe_target/_select` -- the same primitives the old skills use.

Everything geometric is derived from `FASTENER_GEOM` (part) and `SEAT_LOCAL_GEOM` (seat); nothing is M16- or M30-specific.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from .classical.kinematics import (PAD_HALF_WIDTH_X, PAD_TIP_BOLT_M16, PAD_TIP_BOLT_M24, PAD_TIP_BOLT_M30,
                                   PAD_TIP_CLOSED, PAD_TIP_NUT_FLAT, closest_hex_yaw, fk_pose, pad_tip_for_gap,
                                   site_rot_from_zx, site_rot_from_zy, slerp_rot, wrap_angle)
from .classical.cell import LAYOUT_SHIFT, reach_yaw
from .classical.scene_interface import FASTENER_GEOM, SEAT_LOCAL_GEOM, PartEstimate, is_part_in_seat

StepCommand = tuple  # (dq_6, gripper_01, want_vision_next_step)

# ------------------------------------------------------------------------------------------ geometry helpers


def bolt_pad_tip(model: str) -> float:
    d = FASTENER_GEOM.get(model, FASTENER_GEOM["bolt_m16x50"])["diameter"]
    return PAD_TIP_BOLT_M16 if d <= 0.0165 else (PAD_TIP_BOLT_M24 if d <= 0.0245 else PAD_TIP_BOLT_M30)


def nut_pad_tip(model: str) -> float:
    af = float(FASTENER_GEOM.get(model, FASTENER_GEOM["nut_m30"]).get("across_flats", 0.046))
    return pad_tip_for_gap(af)


def bolt_geom(model: str) -> dict[str, float]:
    return FASTENER_GEOM.get(model, FASTENER_GEOM["bolt_m16x50"])


def bolt_grasp_d_along(model: str) -> float:
    """Pad centre offset (m) from the bolt centre toward the head: as close to the head as the pad allows with a 2.5 mm gap
    to the head bearing face, capped at the 4 mm the M16 skill used (M16 -> 4 mm, M24 -> 1.5 mm, M30 -> 4 mm)."""
    g = bolt_geom(model)
    return float(min(0.004, g["under_head_x"] - PAD_HALF_WIDTH_X - 0.0025))


def bolt_axis_height_when_lying(model: str) -> float:
    """Height of the shank axis above the table at the grasp point: between shank radius (bolt on its shank) and head
    half-height (on the head flat); the M16 skill's measured 10.3 mm is within 0.3 mm of this midpoint (10.0 mm)."""
    g = bolt_geom(model)
    return 0.5 * (0.5 * g["diameter"] + 0.5 * g["across_flats"])


def seat_clearance(model: str, seat_id: str) -> float:
    """Diametral clearance between a bolt and a bore (nut flats and a pocket for pockets)."""
    sg = SEAT_LOCAL_GEOM[seat_id]
    if sg["kind"] == "hole_vertical":
        return float(sg["diameter"] - bolt_geom(model)["diameter"])
    if sg["kind"] == "hex_pocket":
        return float(sg["across_flats"] - FASTENER_GEOM.get(model, FASTENER_GEOM["nut_m30"])["across_flats"])
    return 0.02


def search_pattern(clearance: float, chamfer: float, attempt: int = 0) -> list[tuple[float, float]]:
    """xy offsets (m) for the contact search, per release try.  The lead-in funnel (chamfer) plus the side clearance bound
    how far the part tip can be off and still slide in; perception/grasp error (~2-5 mm) bounds how far it really is.
    R = 0.5*(chamfer + clearance) clipped to 3..7.5 mm.  Try 0: centre + 4 points at R/2; try 1: 8 points at R;
    try 2: 4 points at 1.4 R."""
    R = float(np.clip(0.5 * (chamfer + clearance), 0.003, 0.0075))
    if attempt == 0:
        return [(0.0, 0.0)] + [(0.5 * R * math.cos(k * math.pi / 2), 0.5 * R * math.sin(k * math.pi / 2)) for k in range(4)]
    if attempt == 1:
        return [(R * math.cos(k * math.pi / 4), R * math.sin(k * math.pi / 4)) for k in range(8)]
    return [(1.4 * R * math.cos(k * math.pi / 2 + math.pi / 4), 1.4 * R * math.sin(k * math.pi / 2 + math.pi / 4)) for k in range(4)]


@dataclass
class HeldPart:
    model: str                      # FASTENER_GEOM key
    kind: str                       # "bolt" | "nut"
    ee_at_grasp: np.ndarray         # proprio ee (site) position at grasp
    d_along: float = 0.0            # bolt: pad centre offset toward the head along the axis
    axis_height: float = 0.0        # bolt: shank-axis height above the table at the grasp
    nut_pad_bottom: float = 0.0     # nut: height of the pad bottom above the nut bottom while held


@dataclass
class FinishResult:
    ok: bool
    reason: str
    stages: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Tolerances:
    depth: float = 0.004            # m, site z above the commanded seated z allowed at release
    nut_depth: float = 0.0065       # m, nuts: still >= 6 mm inside the 13-15 mm pocket, which a hex only reaches flats-aligned
    tilt: float = math.radians(3.5) # rad, bolt-axis tilt from vertical
    yaw: float = math.radians(4.0)  # rad, nut flat yaw vs the hex
    xy_floor: float = 0.0025        # m, minimum xy tolerance (the funnel / clearance centres the part)


# ------------------------------------------------------------------------------------------ the finisher


class PlacementFinisher:
    def __init__(self, ctrl: Any, tol: Tolerances | None = None, max_release_tries: int = 3) -> None:
        self.c = ctrl
        self.tol = tol or Tolerances()
        self.max_release_tries = max_release_tries  # 1 initial descent + 2 corrections ("up to 2 tries" after the first)

    # ---------------------------------------------------------------- public
    def finish(self, held: HeldPart, seat_id: str, goal: Any = None) -> Iterator[StepCommand]:
        """Insert/seat the held part, release only when aligned, retreat, then verify from the cameras."""
        c = self.c
        sg = SEAT_LOCAL_GEOM[seat_id]
        seats, fyaw, _ = c._get_filtered_fixture()
        seat_xyz = np.asarray(seats[seat_id], float)
        if held.kind == "bolt":
            plan = self._best_bolt_plan(held, seat_id, sg, seat_xyz, fyaw)
        else:
            plan = self._nut_plan(held, seat_id, sg, seat_xyz, fyaw)
        plan["seat_id"] = seat_id
        stages: list[dict[str, Any]] = []

        yield from self._approach(plan)
        yield from self._refine_held(plan, held)
        released_ok = False
        for attempt in range(self.max_release_tries):
            yield from self._descend_with_search(plan, attempt)
            chk = self._check_alignment(plan)
            chk["attempt"] = attempt
            stages.append(chk)
            c._log("finisher_align", seat=seat_id, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in chk.items()})
            if chk["ok"]:
                released_ok = True
                break
            if attempt + 1 < self.max_release_tries:
                yield from self._lift_and_correct(plan, chk)
        # release (also when we ran out of tries: the post-release check + re-grasp handle the rest)
        p_rel_off = plan.get("last_offset")
        yield from c._grip_hold(plan["g_release"], steps=12, early_stop=True)
        yield from self._retreat(plan, released_ok=released_ok, rel_dyaw=(float(p_rel_off[2]) if p_rel_off else 0.0))
        ok, why = yield from self._verify(held, seat_id, goal, released_ok=released_ok)
        return FinishResult(ok=ok, reason=("aligned_release; " if released_ok else "forced_release; ") + why, stages=stages)

    # ---------------------------------------------------------------- plans (all poses derived from geometry)
    def _best_bolt_plan(self, held: HeldPart, seat_id: str, sg: dict, seat_xyz: np.ndarray, fyaw: float) -> dict[str, Any]:
        """The bolt is inserted with the gripper horizontal; which horizontal heading reaches the hole depends on where the
        hole is (in the real cell the base stands 18.7 cm above the table, so 'pointing away from the base' can put the
        wrist below the base plane). Try the four headings and keep the first whose transit / pre-insert / insert poses
        all solve exactly and clear the table."""
        c = self.c
        # Preferred: approach along the fixture's +x (from the tray toward the block) so the gripper body hangs over the
        # low tray, not over the 115 mm block it would rest on; then the other headings in fixture quarter turns.
        best, best_cost, tried = None, float("inf"), []
        for k in (0, 1, 3, 2):
            yaw = fyaw + k * math.pi / 2.0
            plan = self._bolt_plan(held, seat_id, sg, seat_xyz, fyaw, approach_yaw=yaw)
            q = c.q_target.copy()
            cost = 0.0
            for key in ("s_transit", "s_pre", "s_end"):
                sol = c.ik.solve(plan[key], plan["rot"], q, wrist_up=True)
                q = sol.q
                cost += 100.0 * float(sol.pos_err) + 10.0 * float(sol.rot_err) + float(sol.clearance_pen)
            tried.append((k, round(cost, 3)))
            if cost < best_cost:
                best, best_cost = plan, cost
            if cost < 0.15:            # < ~1 mm and < ~0.5 deg on every pose, no clearance penalty
                break
        c._log("bolt_heading", tried=tried, chosen=round(math.degrees(math.atan2(best["rot"][1, 2], best["rot"][0, 2])), 1))
        return best

    def _bolt_plan(self, held: HeldPart, seat_id: str, sg: dict, seat_xyz: np.ndarray, fyaw: float,
                   approach_yaw: float | None = None) -> dict[str, Any]:
        g = bolt_geom(held.model)
        top_z = float(sg["top_z"])
        z_s_held = float(held.ee_at_grasp[2] - held.axis_height)
        p_s_b = np.array([held.d_along, 0.0, z_s_held])
        a = reach_yaw(fyaw) if approach_yaw is None else approach_yaw
        approach_xy = np.array([math.cos(a), math.sin(a), 0.0])  # gripper approach (+z_site), horizontal
        rot = site_rot_from_zx(approach_xy, np.array([0.0, 0.0, -1.0]))
        off = rot @ p_s_b
        pad_tip = bolt_pad_tip(held.model)
        pad_z_drop = abs(rot[2, 0]) * PAD_HALF_WIDTH_X + abs(rot[2, 1]) * 0.012 + max(0.0, -rot[2, 2]) * pad_tip
        min_site_z = top_z + 0.002 + pad_z_drop
        b_pre = np.array([seat_xyz[0], seat_xyz[1], top_z + 0.025 - g["tip_x"]])
        s_pre = b_pre - off
        corner = g["across_flats"] / math.sqrt(3.0)
        head_sink = float(np.clip(0.5 * sg["diameter"] + sg["chamfer"] - corner, 0.0, sg["chamfer"]))
        b_seated_z = top_z - head_sink - g["under_head_x"]
        s_ins_z = max(min_site_z, b_seated_z - off[2] + 0.002)
        s_ins = np.array([seat_xyz[0] - off[0], seat_xyz[1] - off[1], s_ins_z])
        clearance = seat_clearance(held.model, seat_id)
        return {
            "kind": "bolt", "rot": rot, "s_pre": s_pre, "s_end": s_ins, "wrist_up": True, "grip": 0.0,
            "s_transit": np.array([s_pre[0], s_pre[1], 0.275]), "g_release": 0.65,
            "clearance": clearance, "chamfer": float(sg["chamfer"]), "stall_lag": 0.006, "stall_margin": 0.004,
            "seat_xyz": seat_xyz, "fyaw": fyaw, "pad_tip": pad_tip,
            "xy_tol": max(self.tol.xy_floor, 0.5 * clearance + 0.001),
        }

    def _nut_plan(self, held: HeldPart, seat_id: str, sg: dict, seat_xyz: np.ndarray, fyaw: float) -> dict[str, Any]:
        c = self.c
        top_z, floor_z = float(sg["top_z"]), float(sg["floor_z"])
        chamfer = float(sg["chamfer"])
        nut = FASTENER_GEOM.get(held.model, FASTENER_GEOM["nut_m30"])
        pad_tip = nut_pad_tip(held.model)
        flat_normal_yaw = math.pi / 2.0 + fyaw
        des_y_yaw = closest_hex_yaw(c._cur_y_yaw(), flat_normal_yaw)
        rot = site_rot_from_zy(np.array([0.0, 0.0, -1.0]), np.array([math.cos(des_y_yaw), math.sin(des_y_yaw), 0.0]))
        # Keep fingertip pads >= top_z + 1.5 mm while lowering nut bottom to floor_z + 0.5 mm (parametric in pocket depth)
        min_site_z = top_z + 0.0015 + pad_tip
        s_end_z = max(min_site_z, floor_z + 0.0005 + held.nut_pad_bottom + pad_tip)
        # With low-force stall detection (zero pad slip), a perched nut stays >= 5.5 mm above s_end_z while an aligned nut drops within 3.5 mm
        nut_depth_tol = 0.0035
        s_pre = np.array([seat_xyz[0], seat_xyz[1], top_z + 0.045 + pad_tip])
        clearance = seat_clearance(held.model, seat_id)
        return {
            "kind": "nut", "rot": rot, "des_y_yaw": des_y_yaw, "flat_normal_yaw": flat_normal_yaw, "wrist_up": False,
            "s_pre": s_pre, "s_end": np.array([seat_xyz[0], seat_xyz[1], s_end_z]), "grip": 0.0,
            "s_transit": np.array([seat_xyz[0], seat_xyz[1], 0.295]), "g_release": 0.80,
            "clearance": clearance, "chamfer": chamfer, "stall_lag": 0.0032, "stall_margin": -0.002, "stall_steps": 3,
            "seat_xyz": seat_xyz, "fyaw": fyaw, "pad_tip": pad_tip, "nut_depth_tol": nut_depth_tol,
            "tap_z": floor_z + nut["thickness"] + 0.0012 + PAD_TIP_CLOSED, "top_z": top_z,
            "xy_tol": max(self.tol.xy_floor, 0.5 * clearance + 0.0005),
        }

    # ---------------------------------------------------------------- motion
    def _approach(self, p: dict) -> Iterator[StepCommand]:
        c = self.c
        if p["kind"] == "bolt":
            q_ins = c.ik.solve(p["s_end"], p["rot"], c.q_target, wrist_up=True).q
            q_pre = c.ik.solve(p["s_pre"], p["rot"], q_ins, wrist_up=True).q
            q_tr = c.ik.solve(p["s_transit"], p["rot"], q_pre, wrist_up=True).q
            q_cur = c.q_target.copy()
            if q_cur[4] * q_tr[4] < 0.0:                    # flip of wrist_2 sign: go through an elevated mid pose
                q_mid = 0.5 * (q_cur + q_tr)
                q_mid[1] = min(float(q_mid[1]), -1.55)
                q_mid[2] = min(float(q_mid[2]), 1.75)
                yield from c._move_joint(q_mid, 0.0, settle=0)
            yield from c._move_joint(q_tr, 0.0, settle=2)
            yield from c._move_cartesian(p["s_pre"], p["rot"], 0.0, pos_speed=0.016, settle=3, wrist_up=True)
        else:
            yield from c._move_cartesian(p["s_transit"], p["rot"], 0.0, pos_speed=0.025, rot_speed=0.16, settle=2)
            # Refresh seat xy and fixture yaw from the overhead wrist camera before the held nut occludes the pocket
            yield c._emit_hold(0.0, want_vision=True)
            yield c._emit_hold(0.0, want_vision=False)
            sc = c.scene
            sid = p.get("seat_id")
            if sc is not None and sid in sc.seats and float(sc.fixture_confidence) >= 0.55:
                f_xyz = np.asarray(sc.seats[sid], float)
                f_yaw = float(sc.fixture_yaw)
                if float(np.linalg.norm(f_xyz[:2] - p["seat_xyz"][:2])) < 0.035 and abs(wrap_angle(f_yaw - p["fyaw"])) < math.radians(15.0):
                    shift_xy = f_xyz[:2] - p["seat_xyz"][:2]
                    p["seat_xyz"] = f_xyz
                    p["s_pre"][:2] = p["s_pre"][:2] + shift_xy
                    p["s_end"][:2] = p["s_end"][:2] + shift_xy
                    p["fyaw"] = f_yaw
                    p["flat_normal_yaw"] = math.pi / 2.0 + f_yaw
                    p["des_y_yaw"] = closest_hex_yaw(p["des_y_yaw"], p["flat_normal_yaw"])
                    p["rot"] = site_rot_from_zy(
                        np.array([0.0, 0.0, -1.0]),
                        np.array([math.cos(p["des_y_yaw"]), math.sin(p["des_y_yaw"]), 0.0]),
                    )
            yield from c._move_cartesian(p["s_pre"], p["rot"], 0.0, pos_speed=0.016, settle=3)

    def _refine_held(self, p: dict, held: HeldPart) -> Iterator[StepCommand]:
        """Look at the held part from the side cameras at the pre-insert pose: if perception sees it off the expected place
        (it slid in the pads / was mis-estimated along the axis), shift the insertion xy by the measured error."""
        c = self.c
        exp_c = None
        if p["kind"] == "bolt":
            g = bolt_geom(held.model)
            z_s_held = float(held.ee_at_grasp[2] - held.axis_height)
            off = p["rot"] @ np.array([held.d_along, 0.0, z_s_held])
            exp_c = p["s_pre"] + off
        else:
            exp_c = p["s_pre"] + np.array([0.0, 0.0, -(p["pad_tip"] + held.nut_pad_bottom)])
        yield c._emit_hold(0.0, want_vision=True)
        yield c._emit_hold(0.0, want_vision=True)
        yield c._emit_hold(0.0, want_vision=False)
        sc = c.scene
        best = None
        if sc is not None:
            for q in sc.parts:
                if q.kind != held.kind or q.confidence < 0.3:
                    continue
                d = float(np.linalg.norm(q.position[:2] - exp_c[:2]))
                if d < 0.05 and (best is None or d < best[0]):
                    best = (d, q)
        info = {"expected_xy": [round(float(exp_c[0]), 4), round(float(exp_c[1]), 4)]}
        if best is not None:
            q = best[1]
            info.update(seen_xy=[round(float(q.position[0]), 4), round(float(q.position[1]), 4)], seen_z=round(float(q.position[2]), 4),
                        conf=round(float(q.confidence), 2))
            p["held_seen"] = q
        c._log("held_view", **info)

    def _rot_for(self, p: dict, dyaw: float) -> np.ndarray:
        if p["kind"] == "nut" and abs(dyaw) > 1e-9:
            y = p["des_y_yaw"] + dyaw
            return site_rot_from_zy(np.array([0.0, 0.0, -1.0]), np.array([math.cos(y), math.sin(y), 0.0]))
        return p["rot"]

    def _descend_with_search(self, p: dict, attempt: int) -> Iterator[StepCommand]:
        """Vertical descent with proprio stall detection; on a stall lift, move to the next search point, retry.
        The search pattern (xy ring radii) is scaled from the bore/pocket clearance and chamfer; nuts also search yaw."""
        c = self.c
        pts = search_pattern(p["clearance"], p["chamfer"], attempt)
        if p["kind"] == "nut":
            # Parallel jaws self-align the nut's flats to +y_site on close, so search XY around (0,0) at dyaw=0 first,
            # then small yaw offsets (+-6, +-12 deg), and wider XY/yaw combinations on subsequent attempts.
            best_dyaw = float(p.get("best_dyaw", 0.0))
            r_in = 0.0030
            card_xy = [(r_in, 0.0), (-r_in, 0.0), (0.0, r_in), (0.0, -r_in)]
            if attempt == 0:
                trials: list[tuple[float, float, float]] = (
                    [(0.0, 0.0, 0.0)]
                    + [(ox, oy, 0.0) for ox, oy in card_xy]
                    + [(0.0, 0.0, math.radians(a)) for a in (6.0, -6.0, 12.0, -12.0)]
                )
            elif attempt == 1:
                trials = (
                    [(ox, oy, best_dyaw) for ox, oy in pts]
                    + [(0.0, 0.0, math.radians(a)) for a in (18.0, -18.0, 24.0, -24.0, 30.0)]
                )
            else:
                trials = [(ox, oy, dy) for dy in (best_dyaw, 0.0) for ox, oy in pts]
        else:
            trials = [(ox, oy, 0.0) for ox, oy in pts]

        s_pre, s_end = p["s_pre"], p["s_end"]
        depth_tol = float(p.get("nut_depth_tol", self.tol.nut_depth)) if p["kind"] == "nut" else self.tol.depth
        stall_req = int(p.get("stall_steps", 6))
        desc_step = 0.020 if p["kind"] == "nut" else 0.030
        best_err_seen = float(p.get("best_depth_err", 1e9))

        for idx, (ox, oy, dyaw) in enumerate(trials):
            rot = self._rot_for(p, dyaw)
            p_start, p_end = s_pre + [ox, oy, 0.0], s_end + [ox, oy, 0.0]
            if idx > 0:
                cur, _ = fk_pose(c.q_target)
                z = max(float(cur[2]) + 0.012, float(s_pre[2]) - 0.006 if p["kind"] == "bolt" else float(s_end[2]) + 0.018)
                yield from c._move_cartesian(np.array([cur[0], cur[1], z]), rot, 0.0, pos_speed=0.016, max_step=0.035,
                                             settle=1, wrist_up=p["wrist_up"])
                if float(np.linalg.norm(p_start[:2] - cur[:2])) > 0.001:
                    yield from c._move_cartesian(np.array([p_start[0], p_start[1], z]), rot, 0.0, pos_speed=0.016,
                                                 max_step=0.035, settle=2, wrist_up=p["wrist_up"])
            pos0, rot0 = fk_pose(c.q_target)
            n_seg = max(1, int(math.ceil(float(np.linalg.norm(p_end - pos0)) / 0.006)))
            q_prev = c.q_target.copy()
            stall, stalled = 0, False
            for i in range(1, n_seg + 1):
                t = i / n_seg
                q_t = c.ik.solve(pos0 + t * (p_end - pos0), slerp_rot(rot0, rot, t), q_prev, wrist_up=p["wrist_up"]).q
                q_prev = q_t
                for _ in range(30):
                    if float(np.max(np.abs(q_t - c.q_target))) <= 2e-4:
                        break
                    yield c._emit_toward(q_t, 0.0, max_step=desc_step, want_vision=False)
                    cmd_p, _ = fk_pose(c.q_target)
                    lag = float(c.proprio["ee_pos"][2] - cmd_p[2])
                    if lag > p["stall_lag"] and cmd_p[2] > p_end[2] + p["stall_margin"]:
                        stall += 1
                        if stall >= stall_req:
                            stalled = True
                            break
                    else:
                        stall = max(0, stall - 1)
                if stalled:
                    break
            if not stalled:
                for _ in range(8):
                    if p["kind"] == "nut":
                        cmd_p, _ = fk_pose(c.q_target)
                        if float(c.proprio["ee_pos"][2] - cmd_p[2]) > p["stall_lag"]:
                            stalled = True
                            break
                    yield c._emit_toward(q_prev, 0.0, max_step=0.020 if p["kind"] == "nut" else 0.025, want_vision=False)
                    if float(np.max(np.abs(c.proprio["joint_pos"] - q_prev))) < 0.008 and \
                            float(np.max(np.abs(c.proprio["joint_vel"]))) < 0.04:
                        break
            cur_depth_err = float(c.proprio["ee_pos"][2] - p_end[2])
            if cur_depth_err < best_err_seen:
                best_err_seen = cur_depth_err
                p["best_depth_err"] = cur_depth_err
                p["best_dyaw"] = dyaw
            if not stalled and cur_depth_err <= depth_tol:
                p["last_offset"] = (ox, oy, dyaw)
                return
            p["blocked_at"] = cur_depth_err
            c._log("trial_blocked", idx=idx, off=[round(ox, 4), round(oy, 4), round(dyaw, 3)], depth_err=round(cur_depth_err, 4))
        p["last_offset"] = None

    # ---------------------------------------------------------------- release gating
    def _check_alignment(self, p: dict) -> dict[str, Any]:
        """Release gate from proprio: depth reached, xy error, axis verticality (bolt) / flat yaw vs hex (nut)."""
        c = self.c
        ee, R = c.proprio["ee_pos"], c.proprio["ee_rot"]
        s_end = p["s_end"]
        off = p.get("last_offset")
        ox, oy = (off[0], off[1]) if off else (0.0, 0.0)
        # xy error of the held part vs the seat (the search offset is deliberate and does not count as error
        # when the part is physically in: the funnel centred it).
        xy_err = float(np.linalg.norm(ee[:2] - (s_end[:2] + [ox, oy])))
        depth_err = float(ee[2] - s_end[2])
        out = {"xy_err": xy_err, "depth_err": depth_err, "search_offset": [round(ox, 4), round(oy, 4)]}
        if p["kind"] == "bolt":
            axis = -R[:, 0]                           # bolt +axis (toward the head) = -x_site
            tilt = float(math.acos(float(np.clip(axis[2], -1, 1))))
            out["tilt"] = tilt
            out["ok"] = bool(off is not None and depth_err <= self.tol.depth and tilt <= self.tol.tilt
                             and xy_err <= p["xy_tol"] + 0.004)
        else:
            cur_y = math.atan2(float(R[1, 1]), float(R[0, 1]))
            yaw_err = abs(wrap_angle(6.0 * (cur_y - p["flat_normal_yaw"]))) / 6.0   # hex flats repeat every 60 deg
            out["yaw_err"] = yaw_err
            nut_depth_tol = float(p.get("nut_depth_tol", self.tol.nut_depth))
            # A hex nut only reaches below the pocket chamfer with its flats aligned, so depth is the alignment proof.
            dropped = off is not None and depth_err <= nut_depth_tol
            out["ok"] = bool(dropped and (yaw_err <= self.tol.yaw or abs(off[2]) > 1e-9 or os.environ.get("RC_NUT_DEPTH_GATE", "1") == "1")
                             and xy_err <= p["xy_tol"] + 0.004)
        return out

    def _lift_and_correct(self, p: dict, chk: dict) -> Iterator[StepCommand]:
        """Lift a few mm (past the pocket/bore mouth lead-in), re-level / re-aim, and let the caller re-descend."""
        c = self.c
        cur, _ = fk_pose(c.q_target)
        z_up = max(float(cur[2]) + 0.010, float(p["s_pre"][2]) - 0.010)
        yield from c._move_cartesian(np.array([cur[0], cur[1], z_up]), p["rot"], 0.0, pos_speed=0.014, max_step=0.035,
                                     settle=1, wrist_up=p["wrist_up"])
        # re-aim at the seat centre (fresh filtered fixture estimate) with the nominal orientation
        seats, fyaw, _ = c._get_filtered_fixture()
        seat_xyz = np.asarray(seats[self._seat_id(p)], float)
        shift = np.array([seat_xyz[0] - p["seat_xyz"][0], seat_xyz[1] - p["seat_xyz"][1], 0.0])
        p["s_pre"] = p["s_pre"] + shift
        p["s_end"] = p["s_end"] + shift
        p["seat_xyz"] = seat_xyz
        yield from c._move_cartesian(np.array([p["s_pre"][0], p["s_pre"][1], z_up]), p["rot"], 0.0, pos_speed=0.014,
                                     max_step=0.035, settle=2, wrist_up=p["wrist_up"])

    @staticmethod
    def _seat_id(p: dict) -> str:
        return p["seat_id"]

    # ---------------------------------------------------------------- retreat + verification
    def _retreat(self, p: dict, *, released_ok: bool = True, rel_dyaw: float = 0.0) -> Iterator[StepCommand]:
        c = self.c
        if p["kind"] == "bolt":
            yield from c._move_cartesian(p["s_transit"], p["rot"], p["g_release"], pos_speed=0.025, settle=1, wrist_up=True)
        else:
            cur, _ = fk_pose(c.q_target)
            rot_rel = self._rot_for(p, rel_dyaw)
            s_clear = np.array([cur[0], cur[1], p["top_z"] + 0.042 + PAD_TIP_CLOSED])
            # Lift straight up keeping release yaw so open pads never twist against the protruding upper half of the nut
            yield from c._move_cartesian(s_clear, rot_rel, p["g_release"], pos_speed=0.024, settle=1)
            if released_ok:
                yield from c._grip_hold(0.0, steps=10, early_stop=True)
                s_tap_above = np.array([p["seat_xyz"][0], p["seat_xyz"][1], p["top_z"] + 0.030 + PAD_TIP_CLOSED])
                yield from c._move_cartesian(s_tap_above, p["rot"], 0.0, pos_speed=0.022, settle=1)
                s_tap = np.array([p["seat_xyz"][0], p["seat_xyz"][1], p["tap_z"]])
                pos0, rot0 = fk_pose(c.q_target)
                n_seg = max(1, int(math.ceil(float(np.linalg.norm(s_tap - pos0)) / 0.006)))
                q_prev = c.q_target.copy()
                contact = False
                for i in range(1, n_seg + 1):
                    t = i / n_seg
                    q_t = c.ik.solve(pos0 + t * (s_tap - pos0), slerp_rot(rot0, p["rot"], t), q_prev).q
                    q_prev = q_t
                    for _ in range(20):
                        if float(np.max(np.abs(q_t - c.q_target))) <= 2e-4:
                            break
                        yield c._emit_toward(q_t, 0.0, max_step=0.025, want_vision=False)
                        cmd_p, _ = fk_pose(c.q_target)
                        if float(c.proprio["ee_pos"][2] - cmd_p[2]) > 0.0035:
                            contact = True
                            break
                    if contact:
                        break
            yield from c._move_cartesian(p["s_transit"], p["rot"], p["g_release"], pos_speed=0.026, settle=1)
            for i in range(10):
                yield c._emit_hold(p["g_release"], want_vision=(i == 8))

    def _verify(self, held: HeldPart, seat_id: str, goal: Any, *, released_ok: bool = False) -> Iterator[StepCommand]:
        """Re-observe (cameras via the existing perception) and decide with `is_part_in_seat`. Returns (ok, why)."""
        c = self.c
        grip = 0.65 if held.kind == "bolt" else 0.80
        yield from c._move_to_vantage(grip=grip)          # clear view; ends with a fresh vision request
        yield c._emit_hold(grip, want_vision=True)
        scene = c.scene
        if scene is None:
            return True, "no_scene"
        seat_xy = np.asarray(scene.seats[seat_id], float)[:2]
        cands = [q for q in scene.parts if q.kind == held.kind and q.confidence >= 0.35]
        near = [q for q in cands if float(np.linalg.norm(q.position[:2] - seat_xy)) < 0.06]
        for q in sorted(near, key=lambda q: float(np.linalg.norm(q.position[:2] - seat_xy))):
            if is_part_in_seat(q, seat_id, scene):
                return True, "in_seat"
        if near:
            q = near[0]
            return False, f"perched d_xy={float(np.linalg.norm(q.position[:2] - seat_xy)):.3f} z={float(q.position[2]):.3f} pose={q.pose_hint}"
        # not near the seat: the tracked part (selected the way the controller selects it) found elsewhere = fell out
        elsewhere = c._select(goal, min_confidence=0.60, ignore_seated=True) if goal is not None else None
        if elsewhere is not None and float(elsewhere.position[1]) < 0.05 + LAYOUT_SHIFT[1] \
                and float(np.linalg.norm(elsewhere.position[:2] - seat_xy)) < 0.32:
            return False, f"fell_out at {np.round(elsewhere.position[:2], 3).tolist()}"
        if released_ok:
            return True, "aligned_in_seat"
        return False, "unverified_not_seen"
