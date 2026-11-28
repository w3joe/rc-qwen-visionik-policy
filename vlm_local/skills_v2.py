"""Size-aware grasp + place skills for bolt->bore, nut->hex-pocket, washer->bore, and washer->standing-bolt.

`PlacementV2Mixin` overrides `_execute_bolt_hole_goal` / `_execute_nut_pocket_goal` of the vision-IK controller
and adds the two stacking skills (`_execute_washer_bore_goal`, `_execute_washer_on_bolt_goal`) plus size-aware
leaning-bolt recovery and wrist-camera refinement.
"""
from __future__ import annotations

import math
import os
from typing import Any, Iterator

import numpy as np

from .classical import cell as _cell
from .classical import detector as D
from .classical import vision as V
from .classical.kinematics import (PAD_TIP_NUT_FLAT, PAD_TIP_WASHER, closest_hex_yaw, down_mat, fk_pose,
                                   nearest_pi_yaw, normalize, site_rot_from_zx, site_rot_from_zy)
from .classical.planner import EMPTY_GRIPPER_THRESH, StepCommand, _min_grip_for_part
from .classical.scene_interface import FASTENER_GEOM, FIXTURE_BLOCK_TOP_Z, SEAT_LOCAL_GEOM, is_part_in_seat
from .finisher import (HeldPart, PlacementFinisher, bolt_axis_height_when_lying, bolt_geom, bolt_grasp_d_along,
                       bolt_pad_tip, nut_pad_tip)

PLACEMENT_V2 = os.environ.get("PLACEMENT_V2", "1") not in ("0", "", "false", "False")
PLACEMENT_V2_BOLT = os.environ.get("PLACEMENT_V2_BOLT", "1") not in ("0", "", "false", "False")
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


def _washer_model(goal: Any, part: Any = None) -> str:
    for m in (getattr(part, "raw_kind", None), getattr(goal, "part_model", None)):
        if m in FASTENER_GEOM and m.startswith("washer"):
            return m
    return "washer_m33"


def _washer_pad_tip(model: str) -> float:
    od = float(FASTENER_GEOM.get(model, FASTENER_GEOM["washer_m33"]).get("outer_diameter", 0.060))
    return PAD_TIP_WASHER if od >= 0.050 else (0.1360 if od >= 0.034 else 0.1369)


def _washer_min_grip(model: str) -> float:
    od = float(FASTENER_GEOM.get(model, FASTENER_GEOM["washer_m33"]).get("outer_diameter", 0.060))
    return 0.34 if od >= 0.050 else (0.22 if od >= 0.034 else 0.16)


class PlacementV2Mixin:
    placement_v2: bool = PLACEMENT_V2
    finisher_results: list

    def _log_ev(self, kind: str, **kw: Any) -> None:
        fn = getattr(self, "_log", None)
        if callable(fn):
            fn(kind, **kw)

    # ---------------------------------------------------------------- wrist-camera helpers (exact FK, zero extrinsic jitter)
    def _wrist_detect(self):
        det = D.get_detector()
        if det is None or not isinstance(getattr(self, "obs", None), dict):
            return None, [], None
        raw_imgs = self.obs.get("images")
        if not isinstance(raw_imgs, dict) or not raw_imgs.get("wrist"):
            return None, [], None
        w_im = V._decode_image(raw_imgs.get("wrist"))
        if w_im is None:
            return None, [], None
        h, w = w_im.shape[:2]
        wcam = D.make_wrist_camera(self.proprio["joint_pos"], proprio=self.proprio, width=w, height=h)
        dets = det.predict_views({"wrist": w_im}).get("wrist", [])
        return wcam, dets, det

    def _refine_washer_xy_from_wrist(self, model: str, expected_xy: np.ndarray) -> np.ndarray | None:
        wcam, dets, _ = self._wrist_detect()
        if wcam is None or not dets:
            return None
        th = float(FASTENER_GEOM.get(model, FASTENER_GEOM["washer_m33"]).get("thickness", 0.005))
        want_large = model in ("washer_m33", "washer_m30")
        cands = []
        for d in dets:
            if d.kind != "washer" or not d.usable_kpt(0, wcam.width, wcam.height, 0.30):
                continue
            if want_large and d.class_name in ("washer_m12", "washer_m16", "washer_m20") and d.conf > 0.60:
                continue
            p = wcam.intersect_z(float(d.kpts_xy[0, 0]), float(d.kpts_xy[0, 1]), 0.5 * th)
            if p is None:
                continue
            dist = float(np.linalg.norm(p[:2] - expected_xy[:2]))
            if dist < 0.095:
                cands.append((0 if d.class_name == model else 1, dist, p[:2].copy()))
        if not cands:
            return None
        cands.sort(key=lambda t: (t[0], t[1]))
        return cands[0][2]

    def _refine_fixture_seat_from_wrist(self, seat_id: str, init_xy: np.ndarray, rot: np.ndarray,
                                        z_obs: float = 0.31) -> Iterator[StepCommand]:
        x_dir = rot[:2, 0]
        obs_pos = np.array([float(init_xy[0] + 0.040 * x_dir[0]), float(init_xy[1] + 0.040 * x_dir[1]), z_obs])
        cur_grip = 0.0 if self.proprio["gripper"] > 0.16 and self.proprio["gripper"] < 0.72 else 0.80
        yield from self._move_cartesian(obs_pos, rot, cur_grip, pos_speed=0.026, settle=4, tol=0.006, want_vision=True)
        yield self._emit_hold(cur_grip, want_vision=False)
        wcam, dets, det = self._wrist_detect()
        seats, fyaw, _ = self._get_filtered_fixture()
        seat_xy = np.asarray(seats.get(seat_id, init_xy), float)[:2].copy()
        if wcam is not None and det is not None:
            fd = [d for d in dets if d.kind == "fixture" and d.conf >= 0.35]
            if fd:
                fd.sort(key=lambda d: -d.conf)
                fix_info = D.reconstruct_fixture({"wrist": fd[0]}, {"wrist": wcam},
                                                 det.fixture_kpt_seats, det.fixture_seat_spec)
                w_seats = {k: np.asarray(v, float).copy() for k, v in fix_info["seats"].items()}
                w_fyaw = float(fix_info["fixture_yaw"])
                fit_xy = w_seats.get(seat_id, np.array([init_xy[0], init_xy[1], FIXTURE_BLOCK_TOP_Z]))[:2].copy()
                top_z = float(SEAT_LOCAL_GEOM.get(seat_id, {}).get("top_z", FIXTURE_BLOCK_TOP_Z))
                dir_xy = None
                for k_idx, sid in det.fixture_kpt_seats.items():
                    if sid == seat_id and fd[0].usable_kpt(k_idx, wcam.width, wcam.height, 0.50):
                        p_dir = wcam.intersect_z(float(fd[0].kpts_xy[k_idx, 0]), float(fd[0].kpts_xy[k_idx, 1]), top_z)
                        if p_dir is not None:
                            dir_xy = p_dir[:2].copy()
                        break
                cand_xy = 0.5 * (fit_xy + dir_xy) if (dir_xy is not None and float(np.linalg.norm(fit_xy - dir_xy)) < 0.008) \
                    else (dir_xy if dir_xy is not None and len(fix_info.get("observed_seats", {})) < 2 else fit_xy)
                if float(np.linalg.norm(cand_xy - init_xy[:2])) < 0.050:
                    seat_xy = cand_xy
                    if seat_id in w_seats:
                        w_seats[seat_id][:2] = seat_xy
                    self._wrist_fixture = (w_seats, w_fyaw)
                    fyaw = w_fyaw
        self._log_ev("wrist_seat_refine", seat=seat_id, init_xy=[round(float(init_xy[0]), 4), round(float(init_xy[1]), 4)],
                     refined_xy=[round(float(seat_xy[0]), 4), round(float(seat_xy[1]), 4)],
                     shift_mm=round(float(np.linalg.norm(seat_xy - init_xy[:2])) * 1000.0, 1))
        return seat_xy, fyaw

    def _grasp_washer_for_stack(self, goal: Any, *, attempt: int = 0, lift_z: float = 0.305) -> Iterator[StepCommand]:
        model = _washer_model(goal)
        geom = FASTENER_GEOM.get(model, FASTENER_GEOM["washer_m33"])
        pad_tip = _washer_pad_tip(model)
        min_grip = _washer_min_grip(model)

        self.tracker.reset_part()
        if attempt > 0:
            yield from self._move_to_vantage(grip=0.82)
        yield from self._observe_target(goal, grip=0.82, min_samples=1, max_wait=3)
        part = self.tracker.filtered_part()
        if part is None or float(part.position[2]) > 0.055:
            scan_pos = np.array(_cell.shifted((0.58, -0.18, 0.29)), dtype=np.float64)
            yield from self._ensure_safe_height(0.26, 0.82)
            yield from self._move_cartesian(scan_pos, down_mat(0.0), 0.82, pos_speed=0.028, settle=3, want_vision=True)
            yield self._emit_hold(0.82, want_vision=True)
            self.tracker.reset_part()
            yield from self._observe_target(goal, grip=0.82, min_samples=1, max_wait=3)
            part = self.tracker.filtered_part()
            if part is None or float(part.position[2]) > 0.055:
                wxy_scan = self._refine_washer_xy_from_wrist(model, scan_pos[:2])
                if wxy_scan is None:
                    self._log_ev("washer_no_part", goal=getattr(goal, "goal_id", "?"), attempt=attempt)
                    return None
                w_xy = wxy_scan
            else:
                w_xy = part.position[:2].copy()
        else:
            w_xy = part.position[:2].copy()

        _, fyaw, _ = self._get_filtered_fixture()
        grasp_yaw = nearest_pi_yaw(_cell.reach_yaw(fyaw), self._cur_ee_yaw())
        rot = down_mat(grasp_yaw)

        yield from self._ensure_safe_height(0.25, 0.82)
        yield from self._move_cartesian(np.array([float(w_xy[0]), float(w_xy[1]), 0.24]), rot, 0.82,
                                        pos_speed=0.028, settle=3, tol=0.006, want_vision=False)
        yield self._emit_hold(0.82, want_vision=False)
        refined1 = self._refine_washer_xy_from_wrist(model, w_xy)
        if refined1 is not None:
            w_xy = refined1
            yield from self._move_cartesian(np.array([float(w_xy[0]), float(w_xy[1]), 0.24]), rot, 0.82,
                                            pos_speed=0.020, settle=3, tol=0.005, want_vision=False)
            yield self._emit_hold(0.82, want_vision=False)
            refined2 = self._refine_washer_xy_from_wrist(model, w_xy)
            if refined2 is not None and float(np.linalg.norm(refined2 - w_xy)) < 0.020:
                w_xy = refined2
                if float(np.linalg.norm(w_xy - fk_pose(self.q_target)[0][:2])) > 0.0015:
                    yield from self._move_cartesian(np.array([float(w_xy[0]), float(w_xy[1]), 0.24]), rot, 0.82,
                                                    pos_speed=0.016, settle=2, tol=0.005, want_vision=False)

        dz = min(float(os.environ.get("RC_WASHER_DZ", "0.0025")), 0.6 * float(geom.get("thickness", 0.005)))
        if attempt >= 2:
            dz += 0.001
        z_grasp = pad_tip - dz
        yield from self._move_cartesian(np.array([float(w_xy[0]), float(w_xy[1]), z_grasp]), rot, 0.82,
                                        pos_speed=0.014, max_step=0.036, settle=6, tol=0.006, want_vision=False)
        grasp_ee_z = float(self.proprio["ee_pos"][2])
        yield from self._grip_hold(0.0, steps=16, early_stop=True, want_vision=False)
        yield from self._move_cartesian(np.array([float(w_xy[0]), float(w_xy[1]), lift_z]), rot, 0.0,
                                        pos_speed=0.025, settle=2, want_vision=True)
        g_now = float(self.proprio["gripper"])
        ok = min_grip <= g_now <= 0.78
        self._log_ev("washer_grasp", model=model, xy=[round(float(w_xy[0]), 4), round(float(w_xy[1]), 4)],
                     grasp_ee_z=round(grasp_ee_z, 4), gripper=round(g_now, 3), attempt=attempt, ok=ok)
        if not ok:
            yield from self._grip_hold(0.82, steps=8, early_stop=True, want_vision=True)
            return None
        if hasattr(goal, "target_xy") and goal.target_xy is not None:
            goal.target_xy = w_xy.copy()
        return {"model": model, "w_xy": w_xy, "grasp_yaw": grasp_yaw, "rot": rot,
                "grasp_ee_z": grasp_ee_z, "pad_tip": pad_tip, "min_grip": min_grip}

    # ---------------------------------------------------------------- Skill A1: washer -> bore mouth
    def _execute_washer_bore_goal(self, goal) -> Iterator[StepCommand]:
        seat_id = goal.seat_id
        sg = SEAT_LOCAL_GEOM[seat_id]
        top_z = float(sg["top_z"])
        model = _washer_model(goal)
        w_th = float(FASTENER_GEOM.get(model, FASTENER_GEOM["washer_m33"]).get("thickness", 0.005))

        for attempt in range(3):
            held = yield from self._grasp_washer_for_stack(goal, attempt=attempt, lift_z=0.305)
            if held is None:
                continue
            rot = held["rot"]
            pad_tip = held["pad_tip"]
            grasp_ee_z = held["grasp_ee_z"]
            min_grip = held["min_grip"]

            seats, _, _ = self._get_filtered_fixture()
            init_xy = np.asarray(seats[seat_id], float)[:2]
            seat_xy, _ = yield from self._refine_fixture_seat_from_wrist(seat_id, init_xy, rot, z_obs=0.31)

            z_pre = max(0.30, top_z + 0.045 + pad_tip)
            yield from self._move_cartesian(np.array([float(seat_xy[0]), float(seat_xy[1]), z_pre]), rot, 0.0,
                                            pos_speed=0.022, settle=3, tol=0.005)
            if float(self.proprio["gripper"]) < min_grip:
                self._log_ev("washer_bore_dropped", seat=seat_id, attempt=attempt)
                yield from self._grip_hold(0.82, steps=8, early_stop=True, want_vision=True)
                continue

            drop_z = top_z + 0.002 + max(pad_tip, grasp_ee_z)
            yield from self._move_cartesian(np.array([float(seat_xy[0]), float(seat_xy[1]), drop_z]), rot, 0.0,
                                            pos_speed=0.012, max_step=0.032, settle=3, tol=0.005)
            yield from self._grip_hold(0.80, steps=10, early_stop=True)
            yield from self._move_cartesian(np.array([float(seat_xy[0]), float(seat_xy[1]), 0.305]), rot, 0.80,
                                            pos_speed=0.025, settle=2)

            # Verify washer sits flat over the bore mouth within WASHER_CENTRE_TOL (5 mm)
            x_dir = rot[:2, 0]
            v_pos = np.array([float(seat_xy[0] + 0.040 * x_dir[0]), float(seat_xy[1] + 0.040 * x_dir[1]), 0.31])
            yield from self._move_cartesian(v_pos, rot, 0.80, pos_speed=0.026, settle=3, want_vision=True)
            yield self._emit_hold(0.80, want_vision=True)
            wcam, dets, _ = self._wrist_detect()
            err_mm = None
            if wcam is not None:
                for d in dets:
                    if d.kind == "washer" and d.usable_kpt(0, wcam.width, wcam.height, 0.30):
                        pw = wcam.intersect_z(float(d.kpts_xy[0, 0]), float(d.kpts_xy[0, 1]), top_z + 0.5 * w_th)
                        if pw is not None and float(np.linalg.norm(pw[:2] - seat_xy)) < 0.035:
                            err_mm = round(float(np.linalg.norm(pw[:2] - seat_xy)) * 1000.0, 2)
                            break
            table_washer = self._select(goal, min_confidence=0.45, ignore_seated=True)
            seated_ok = (err_mm is not None and err_mm <= 5.5) or (
                err_mm is None and (
                    table_washer is None
                    or (self.scene is not None and any(p.kind == "washer" and is_part_in_seat(p, seat_id, self.scene) for p in self.scene.parts))
                )
            )
            self._log_ev("washer_bore_placed", seat=seat_id, seat_xy=[round(float(seat_xy[0]), 4), round(float(seat_xy[1]), 4)],
                         drop_z=round(drop_z, 4), err_mm=err_mm, ok=seated_ok, attempt=attempt)
            if seated_ok or (err_mm is None and attempt >= 1):
                for _ in range(6):
                    yield self._emit_hold(0.80, want_vision=False)
                return

    # ---------------------------------------------------------------- Skill B: washer -> standing bolt
    def _execute_washer_on_bolt_goal(self, goal) -> Iterator[StepCommand]:
        w_model = _washer_model(goal)
        w_th = float(FASTENER_GEOM.get(w_model, FASTENER_GEOM["washer_m33"]).get("thickness", 0.005))
        b_model = goal.ref_model if getattr(goal, "ref_model", None) in FASTENER_GEOM else "bolt_m30x100"
        b_geom = FASTENER_GEOM[b_model]
        z_tip = float(b_geom["length"] + b_geom["head_height"])
        z_k0 = float(0.5 * b_geom["length"] + b_geom["head_height"])
        z_head = float(b_geom["head_height"])
        z_clear = max(0.31, z_tip + 0.055 + _washer_pad_tip(w_model))

        for attempt in range(3):
            held = yield from self._grasp_washer_for_stack(goal, attempt=attempt, lift_z=z_clear)
            if held is None:
                continue
            rot = held["rot"]
            pad_tip = held["pad_tip"]
            grasp_ee_z = held["grasp_ee_z"]
            min_grip = held["min_grip"]

            b_est = np.asarray(goal.ref_part_xy if goal.ref_part_xy is not None else [0.71, -0.32], float)[:2].copy()
            if self.scene is not None:
                st_cands = [p for p in self.scene.parts if p.kind == "bolt"
                            and float(np.linalg.norm(p.position[:2] - b_est)) < 0.08]
                if st_cands:
                    best_b = min(st_cands, key=lambda p: (0 if p.pose_hint == "standing" else 1,
                                                          float(np.linalg.norm(p.position[:2] - b_est))))
                    b_est = best_b.position[:2].copy()
                    if best_b.raw_kind in FASTENER_GEOM and best_b.pose_hint == "standing":
                        b_model = best_b.raw_kind
                        b_geom = FASTENER_GEOM[b_model]
                        z_tip = float(b_geom["length"] + b_geom["head_height"])
                        z_k0 = float(0.5 * b_geom["length"] + b_geom["head_height"])
                        z_head = float(b_geom["head_height"])

            # Unoccluded wrist-camera localisation of the standing bolt's shank tip (k2) and midpoint (k0)
            x_dir = rot[:2, 0]
            obs_pos = np.array([float(b_est[0] + 0.035 * x_dir[0]), float(b_est[1] + 0.035 * x_dir[1]), z_clear])
            yield from self._move_cartesian(obs_pos, rot, 0.0, pos_speed=0.028, settle=4, tol=0.005, want_vision=False)
            yield self._emit_hold(0.0, want_vision=False)
            wcam, dets, _ = self._wrist_detect()
            bolt_xy = b_est.copy()
            b_cls_seen = "fallback"
            if wcam is not None and dets:
                bd_cands = []
                for d in dets:
                    if d.kind != "bolt" or not d.usable_kpt(0, wcam.width, wcam.height, 0.30):
                        continue
                    p0 = wcam.intersect_z(float(d.kpts_xy[0, 0]), float(d.kpts_xy[0, 1]), z_k0)
                    if p0 is None:
                        continue
                    dist = float(np.linalg.norm(p0[:2] - b_est))
                    if dist < 0.075:
                        bd_cands.append((dist, d, p0[:2].copy()))
                if bd_cands:
                    bd_cands.sort(key=lambda t: t[0])
                    _, bd0, k0_xy = bd_cands[0]
                    b_cls_seen = bd0.class_name
                    if bd0.usable_kpt(2, wcam.width, wcam.height, 0.30):
                        p2 = wcam.intersect_z(float(bd0.kpts_xy[2, 0]), float(bd0.kpts_xy[2, 1]), z_tip)
                        if p2 is not None and float(np.linalg.norm(p2[:2] - k0_xy)) < 0.016:
                            bolt_xy = 0.65 * p2[:2] + 0.35 * k0_xy
                        else:
                            bolt_xy = k0_xy
                    else:
                        bolt_xy = k0_xy
                    goal.ref_part_xy = bolt_xy.copy()

            self._log_ev("wrist_bolt_refine", b_model=b_model, seen_cls=b_cls_seen,
                         init_xy=[round(float(b_est[0]), 4), round(float(b_est[1]), 4)],
                         bolt_xy=[round(float(bolt_xy[0]), 4), round(float(bolt_xy[1]), 4)],
                         shift_mm=round(float(np.linalg.norm(bolt_xy - b_est)) * 1000.0, 1))

            # Align ring hole strictly over standing bolt axis at z_clear, settle, then lower onto head
            yield from self._move_cartesian(np.array([float(bolt_xy[0]), float(bolt_xy[1]), z_clear]), rot, 0.0,
                                            pos_speed=0.022, settle=4, tol=0.005, want_vision=False)
            if float(self.proprio["gripper"]) < min_grip:
                self._log_ev("washer_on_bolt_dropped", attempt=attempt)
                yield from self._grip_hold(0.82, steps=8, early_stop=True, want_vision=True)
                continue

            drop_z = z_head + 0.0025 + max(pad_tip, grasp_ee_z)
            yield from self._move_cartesian(np.array([float(bolt_xy[0]), float(bolt_xy[1]), drop_z]), rot, 0.0,
                                            pos_speed=0.012, max_step=0.032, settle=3, tol=0.005, want_vision=False)
            self._log_ev("washer_on_bolt_release", bolt_xy=[round(float(bolt_xy[0]), 4), round(float(bolt_xy[1]), 4)],
                         drop_z=round(drop_z, 4), attempt=attempt)
            yield from self._grip_hold(0.82, steps=10, early_stop=True, want_vision=False)
            # Retract straight up clear of the standing shank before any lateral motion
            yield from self._move_cartesian(np.array([float(bolt_xy[0]), float(bolt_xy[1]), z_clear]), rot, 0.82,
                                            pos_speed=0.025, settle=2, want_vision=False)

            # Quick verification from wrist offset
            v_pos = np.array([float(bolt_xy[0] + 0.035 * x_dir[0]), float(bolt_xy[1] + 0.035 * x_dir[1]), z_clear])
            yield from self._move_cartesian(v_pos, rot, 0.82, pos_speed=0.026, settle=3, want_vision=True)
            yield self._emit_hold(0.82, want_vision=False)
            wcam2, dets2, _ = self._wrist_detect()
            missed_table = False
            bolt_tipped = False
            if wcam2 is not None and dets2:
                for d in dets2:
                    if d.kind == "bolt" and d.usable_kpt(0, wcam2.width, wcam2.height, 0.30) and d.usable_kpt(2, wcam2.width, wcam2.height, 0.30):
                        q0 = wcam2.intersect_z(float(d.kpts_xy[0, 0]), float(d.kpts_xy[0, 1]), z_k0)
                        q2 = wcam2.intersect_z(float(d.kpts_xy[2, 0]), float(d.kpts_xy[2, 1]), z_tip)
                        if q0 is not None and q2 is not None and float(np.linalg.norm(q0[:2] - bolt_xy)) < 0.06:
                            if float(np.linalg.norm(q2[:2] - q0[:2])) > 0.028:
                                bolt_tipped = True
                    if d.kind == "washer" and d.class_name == w_model and d.usable_kpt(0, wcam2.width, wcam2.height, 0.35):
                        qw_head = wcam2.intersect_z(float(d.kpts_xy[0, 0]), float(d.kpts_xy[0, 1]), z_head + 0.5 * w_th)
                        qw_tab = wcam2.intersect_z(float(d.kpts_xy[0, 0]), float(d.kpts_xy[0, 1]), 0.5 * w_th)
                        if qw_head is not None and float(np.linalg.norm(qw_head[:2] - bolt_xy)) > 0.032:
                            if qw_tab is not None and float(np.linalg.norm(qw_tab[:2] - bolt_xy)) > 0.032:
                                missed_table = True

            ok = (not bolt_tipped) and (not missed_table)
            self._log_ev("washer_on_bolt_placed", bolt_xy=[round(float(bolt_xy[0]), 4), round(float(bolt_xy[1]), 4)],
                         drop_z=round(drop_z, 4), ok=ok, bolt_tipped=bolt_tipped, missed_table=missed_table, attempt=attempt)
            if bolt_tipped:
                return
            if ok:
                for _ in range(8):
                    yield self._emit_hold(0.82, want_vision=False)
                return

    # ---------------------------------------------------------------- size-aware leaning bolt recovery + wrist bolt grasp
    def _unlean_bolt(self, part, fyaw: float, goal: Any = None) -> Iterator[StepCommand]:
        model = _bolt_model(goal, part)
        g = bolt_geom(model)
        if g["diameter"] <= 0.017:
            yield from super()._unlean_bolt(part, fyaw)
            return
        pos = part.position.copy()
        fy_r = _cell.reach_yaw(fyaw)
        w_out = np.array([math.sin(fy_r), -math.cos(fy_r), 0.0], dtype=np.float64)
        z_lean = 0.5 * (g["length"] + g["head_height"])

        # Refine leaning bolt centre with the wrist camera so large M24/M30 bolts are grasped cleanly on the shank
        yield from self._ensure_safe_height(0.26, 0.62)
        obs_p = np.array([float(pos[0]), float(pos[1]) - 0.04, 0.28], dtype=np.float64)
        yield from self._move_cartesian(obs_p, down_mat(0.0), 0.62, pos_speed=0.028, settle=3, want_vision=False)
        yield self._emit_hold(0.62, want_vision=False)
        wcam, dets, _ = self._wrist_detect()
        p_ln = np.array([float(pos[0]), float(pos[1]), z_lean], dtype=np.float64)
        if wcam is not None and dets:
            cands = []
            for d in dets:
                if d.kind == "bolt" and d.usable_kpt(0, wcam.width, wcam.height, 0.30):
                    pl = wcam.intersect_z(float(d.kpts_xy[0, 0]), float(d.kpts_xy[0, 1]), z_lean)
                    if pl is not None and float(np.linalg.norm(pl[:2] - pos[:2])) < 0.07:
                        cands.append((float(np.linalg.norm(pl[:2] - pos[:2])), pl))
            if cands:
                cands.sort(key=lambda t: t[0])
                p_ln = cands[0][1]

        axis = normalize(np.array([0.5 * w_out[0], 0.5 * w_out[1], -0.866], dtype=np.float64))
        x_site = axis
        z_site = normalize((-w_out) - float(np.dot(-w_out, x_site)) * x_site)
        y_site = np.cross(z_site, x_site)
        rot_grasp = np.column_stack([x_site, y_site, z_site])

        p_site_grasp = p_ln - 0.005 * axis - 0.126 * z_site
        p_site_pre = p_site_grasp - 0.080 * z_site + np.array([0.0, 0.0, 0.03], dtype=np.float64)
        grip_open = 0.62

        q_pre = self.ik.solve(p_site_pre, rot_grasp, self.q_target).q
        yield from self._move_joint(q_pre, grip_open, settle=2)
        yield from self._move_cartesian(p_site_grasp, rot_grasp, grip_open, pos_speed=0.012, settle=3)
        yield from self._grip_hold(0.0, steps=14, early_stop=True)

        p_pull = p_site_grasp - 0.070 * z_site + np.array([0.0, 0.0, 0.095], dtype=np.float64)
        yield from self._move_cartesian(p_pull, rot_grasp, 0.0, pos_speed=0.018, settle=2)
        g_now = float(self.proprio["gripper"])
        if g_now < EMPTY_GRIPPER_THRESH:
            self._log_ev("unlean_missed", model=model, gripper=round(g_now, 3))
            yield from self._grip_hold(0.62, steps=8, early_stop=True, want_vision=True)
            yield from self._ensure_safe_height(0.27, 0.62)
            return

        drop_xy = self._clear_table_spot(exclude_xy=p_ln[:2])
        rot_flat = site_rot_from_zx(np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0]))
        p_above = np.array([float(drop_xy[0]), float(drop_xy[1]), 0.27], dtype=np.float64)
        q_above = self.ik.solve(p_above, rot_flat, self.q_target).q
        yield from self._move_joint(q_above, 0.0, settle=2)

        p_drop = np.array([float(drop_xy[0]), float(drop_xy[1]), 0.138 + g["lying_radius"]], dtype=np.float64)
        yield from self._move_cartesian(p_drop, rot_flat, 0.0, pos_speed=0.015, settle=3)
        yield from self._grip_hold(0.62, steps=10, early_stop=True)
        yield from self._move_cartesian(p_above, rot_flat, 0.62, pos_speed=0.025, settle=2, want_vision=True)
        yield self._emit_hold(0.62, want_vision=True)
        if goal is not None and hasattr(goal, "target_xy") and goal.target_xy is not None:
            goal.target_xy = drop_xy.copy()
        self._log_ev("unlean_bolt", model=model, p_ln=[round(float(x), 3) for x in p_ln],
                     drop_xy=[round(float(drop_xy[0]), 3), round(float(drop_xy[1]), 3)], gripper=round(g_now, 3))

    def _grasp_lying_bolt_wrist_refined(self, goal: Any, part: Any, model: str, attempt: int = 0) -> Iterator[StepCommand]:
        g = bolt_geom(model)
        ax_xy = np.array([float(part.axis[0]), float(part.axis[1]), 0.0], dtype=np.float64)
        if float(np.linalg.norm(ax_xy)) < 1e-4:
            ax_xy = np.array([math.cos(part.yaw), math.sin(part.yaw), 0.0], dtype=np.float64)
        else:
            ax_xy = ax_xy / float(np.linalg.norm(ax_xy))
        rot_init = site_rot_from_zx(np.array([0.0, 0.0, -1.0]), -ax_xy)
        g_open = 0.55 if g["diameter"] > 0.020 else 0.50

        yield from self._ensure_safe_height(0.26, g_open)
        hover0 = np.array([float(part.position[0]), float(part.position[1]), 0.28], dtype=np.float64)
        q_h0 = self.ik.solve(hover0, rot_init, self.q_target).q
        yield from self._move_joint(q_h0, g_open, settle=3, want_vision=False)
        yield self._emit_hold(g_open, want_vision=False)

        bpos = part.position.copy()
        bax = ax_xy.copy()
        wcam, dets, _ = self._wrist_detect()
        if wcam is not None and dets:
            full_len = g["length"] + g["head_height"]
            cands = []
            for d in dets:
                if d.kind != "bolt" or not (d.usable_kpt(1, wcam.width, wcam.height, 0.30) and d.usable_kpt(2, wcam.width, wcam.height, 0.30)):
                    continue
                pk1 = wcam.intersect_z(float(d.kpts_xy[1, 0]), float(d.kpts_xy[1, 1]), 0.5 * g["across_flats"])
                pk2 = wcam.intersect_z(float(d.kpts_xy[2, 0]), float(d.kpts_xy[2, 1]), 0.5 * g["diameter"])
                if pk1 is None or pk2 is None:
                    continue
                mid = 0.5 * (pk1 + pk2)
                dist = float(np.linalg.norm(mid[:2] - part.position[:2]))
                len_err = abs(float(np.linalg.norm(pk1 - pk2)) - full_len)
                if dist < 0.085 and len_err < 0.030:
                    cands.append((0 if d.class_name == model else 1, dist, mid, pk1 - pk2))
            if cands:
                cands.sort(key=lambda t: (t[0], t[1]))
                _, _, mid, vec = cands[0]
                vec_xy = np.array([float(vec[0]), float(vec[1]), 0.0], dtype=np.float64)
                if float(np.linalg.norm(vec_xy)) > 1e-4:
                    bpos = mid
                    bax = vec_xy / float(np.linalg.norm(vec_xy))

        d_along = bolt_grasp_d_along(model)
        retry_offsets = (np.zeros(3), np.array([0.0, 0.0, -0.002]), np.array([0.003, 0.0, -0.002]), np.array([-0.003, 0.003, -0.002]))
        p_grasp = bpos + d_along * bax + retry_offsets[attempt % len(retry_offsets)]
        rot_grasp = site_rot_from_zx(np.array([0.0, 0.0, -1.0]), -bax)
        z_grasp = max(0.002, g["lying_radius"] - 0.008) + bolt_pad_tip(model) + float(retry_offsets[attempt % len(retry_offsets)][2])

        yield from self._move_cartesian(np.array([float(p_grasp[0]), float(p_grasp[1]), 0.28]), rot_grasp, g_open,
                                        pos_speed=0.022, settle=3, tol=0.006, want_vision=False)
        yield from self._move_cartesian(np.array([float(p_grasp[0]), float(p_grasp[1]), z_grasp]), rot_grasp, g_open,
                                        pos_speed=0.014, max_step=0.036, settle=5, tol=0.006, want_vision=False)
        ee_at_grasp = self.proprio["ee_pos"].copy()
        self._last_grasp_meta = {"d_along": d_along, "axis_xy": bax, "model": model,
                                 "part": part, "grasp_ee_pos": ee_at_grasp, "grasp_rot": rot_grasp}
        self._log_ev("wrist_bolt_grasp", model=model, xy=[round(float(p_grasp[0]), 4), round(float(p_grasp[1]), 4)],
                     ee_z=round(float(ee_at_grasp[2]), 4), attempt=attempt)

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

    # ---------------------------------------------------------------- bolt -> bore (with optional washer under head)
    def _execute_bolt_hole_goal(self, goal) -> Iterator[StepCommand]:
        if not (self.placement_v2 and PLACEMENT_V2_BOLT):
            yield from super()._execute_bolt_hole_goal(goal)
            return
        min_grip = _min_grip_for_part(goal.part_model)
        on_washer = getattr(goal, "on_washer_model", None)
        w_th = float(FASTENER_GEOM.get(on_washer, FASTENER_GEOM["washer_m33"])["thickness"]) if on_washer else 0.0
        w_hole = float(FASTENER_GEOM.get(on_washer, FASTENER_GEOM["washer_m33"])["inner_diameter"]) if on_washer else 0.0

        # If inserting through a pre-placed washer and we haven't calibrated the seat with the wrist camera yet, do so now
        if on_washer and getattr(self, "_wrist_fixture", None) is None:
            seats0, _, _ = self._get_filtered_fixture()
            yield from self._ensure_safe_height(0.28, 0.60)
            yield from self._refine_fixture_seat_from_wrist(goal.seat_id, np.asarray(seats0[goal.seat_id], float)[:2], down_mat(0.0))

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
            wfix = getattr(self, "_wrist_fixture", None)
            if wfix is not None:
                fyaw = float(wfix[1])
            if part.pose_hint in ("leaning", "standing"):
                yield from self._unlean_bolt(part, fyaw, goal=goal)
                self.tracker.reset_part()
                yield from self._observe_target(goal, grip=0.55, min_samples=1, max_wait=4)
                part = self.tracker.filtered_part()
                if part is None:
                    continue
            model = _bolt_model(goal, part)
            if on_washer or bolt_geom(model)["diameter"] > 0.020:
                yield from self._grasp_lying_bolt_wrist_refined(goal, part, model, attempt=attempt)
            else:
                yield from self._hover_and_descend_with_nudge_watch(
                    goal, lambda p, s, fy, ry, m=model: self._compute_lying_bolt_grasp(p, s, fy, ry, m), attempt=attempt)
            meta = self._last_grasp_meta
            yield from self._grip_hold(0.0, steps=14, early_stop=True)
            cur_p, cur_r = fk_pose(self.q_target)
            yield from self._move_cartesian(np.array([cur_p[0], cur_p[1], 0.29 if on_washer else 0.28]), cur_r, 0.0, pos_speed=0.024, settle=2)
            if self.proprio["gripper"] < min_grip or self.proprio["gripper"] > 0.65:
                yield from self._grip_hold(0.55, steps=8, early_stop=True, want_vision=True)
                continue
            held = HeldPart(model=model, kind="bolt", ee_at_grasp=np.asarray(meta["grasp_ee_pos"], float),
                            d_along=float(meta["d_along"]), axis_height=bolt_axis_height_when_lying(model),
                            support_thickness=w_th, washer_hole=w_hole)
            res = yield from PlacementFinisher(self).finish(held, goal.seat_id, goal)
            self._log_ev("finish", goal=getattr(goal, "goal_id", "?"), ok=res.ok, why=res.reason, on_washer=bool(on_washer))
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
                if getattr(self, "_last_tip_xy", None) is not None and getattr(goal, "target_xy", None) is not None:
                    goal.target_xy = self._last_tip_xy.copy()
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
                if EMPTY_GRIPPER_THRESH + 0.02 < g_now < min_grip:
                    # Pinched an on-edge nut across its thickness: parallel jaws aligned its flat normal to +y_site!
                    # Set it gently back onto the table and immediately tip it flat along +y_site.
                    z_put = 0.012 + nut_pad_tip(model)
                    yield from self._move_cartesian(np.array([cur_p[0], cur_p[1], z_put]), cur_r, 0.0, pos_speed=0.018, settle=2)
                    yield from self._grip_hold(g_open, steps=8, early_stop=True, want_vision=False)
                    yield from self._move_cartesian(np.array([cur_p[0], cur_p[1], 0.25]), cur_r, g_open, pos_speed=0.024, settle=1)
                    edge_part = meta["part"]
                    edge_part.position = np.array([float(cur_p[0]), float(cur_p[1]), float(edge_part.position[2])])
                    yield from self._tip_edge_nut(edge_part, model=model, axis_override=cur_r[:, 1].copy())
                    if getattr(self, "_last_tip_xy", None) is not None and getattr(goal, "target_xy", None) is not None:
                        goal.target_xy = self._last_tip_xy.copy()
                    continue
                yield from self._grip_hold(g_open, steps=8, early_stop=True, want_vision=True)
                continue
            held = HeldPart(model=model, kind="nut", ee_at_grasp=np.asarray(meta["grasp_ee_pos"], float),
                            nut_pad_bottom=float(meta["pad_bottom"]))
            res = yield from PlacementFinisher(self).finish(held, goal.seat_id, goal)
            self._log_ev("finish", goal=getattr(goal, "goal_id", "?"), ok=res.ok, why=res.reason)
            self.finisher_results.append((getattr(goal, "goal_id", "?"), res))
            placements += 1
            if not res.ok and "perched" in res.reason and getattr(goal, "target_xy", None) is not None:
                seats_now, _, _ = self._get_filtered_fixture()
                if goal.seat_id in seats_now:
                    goal.target_xy = np.asarray(seats_now[goal.seat_id], float)[:2].copy()
            if res.ok or placements >= MAX_PLACE_TRIES:
                return
