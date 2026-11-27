"""Offline unit tests for detector 3D projection/triangulation round-trip and scene schema."""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlm_local.classical import detector as D
from vlm_local.classical.kinematics import HOME_Q
from vlm_local.classical.scene_interface import FIXTURE_BLOCK_TOP_Z, SEAT_LOCAL_GEOM, normalize_scene


def test_side_and_wrist_projection_triangulation_roundtrip() -> None:
    cam_l = D.make_side_camera("left")
    cam_r = D.make_side_camera("right")
    cam_w = D.make_wrist_camera(HOME_Q)

    pts_world = [
        np.array([0.70, -0.18, 0.012], dtype=np.float64),
        np.array([0.82, -0.12, 0.025], dtype=np.float64),
        np.array([0.66, 0.15, FIXTURE_BLOCK_TOP_Z], dtype=np.float64),
    ]
    for p_true in pts_world:
        ul, vl, dl = cam_l.project(p_true[None, :])
        ur, vr, dr = cam_r.project(p_true[None, :])
        assert float(dl[0]) > 0.1 and float(dr[0]) > 0.1

        ol, rl = cam_l.unproject_ray(float(ul[0]), float(vl[0]))
        or_, rr = cam_r.unproject_ray(float(ur[0]), float(vr[0]))
        p_rec, epi = D.triangulate_rays([ol, or_], [rl, rr])
        assert epi < 1e-9
        assert float(np.linalg.norm(p_rec - p_true)) < 1e-6

        hit_l = cam_l.intersect_z(float(ul[0]), float(vl[0]), float(p_true[2]))
        assert hit_l is not None
        assert float(np.linalg.norm(hit_l - p_true)) < 1e-6

    # 3-view round-trip including FK wrist camera on a point inside wrist FOV
    p_w = np.array([0.88, -0.05, 0.015], dtype=np.float64)
    origins, dirs, weights = [], [], []
    for cam, w in ((cam_l, 1.0), (cam_r, 1.0), (cam_w, 2.5)):
        u, v, dep = cam.project(p_w[None, :])
        assert float(dep[0]) > 0.05
        o, d = cam.unproject_ray(float(u[0]), float(v[0]))
        origins.append(o)
        dirs.append(d)
        weights.append(w)
    p_3v, epi_3v = D.triangulate_rays(origins, dirs, weights)
    assert epi_3v < 1e-6
    assert float(np.linalg.norm(p_3v - p_w)) < 1e-5


def test_reconstruct_scene_3d_bolt_and_fixture_roundtrip() -> None:
    cams = {
        "left": D.make_side_camera("left"),
        "right": D.make_side_camera("right"),
        "wrist": D.make_wrist_camera(HOME_Q),
    }

    true_fx, true_fy, true_fyaw = 0.675, 0.135, math.radians(-8.0)
    c_f, s_f = math.cos(true_fyaw), math.sin(true_fyaw)

    # Synthetic lying M16x50 bolt at (0.72, -0.20, 0.010), yaw 25 deg
    bolt_pos = np.array([0.72, -0.20, 0.010], dtype=np.float64)
    bolt_yaw = math.radians(25.0)
    bolt_axis = np.array([math.cos(bolt_yaw), math.sin(bolt_yaw), 0.0], dtype=np.float64)
    p_k0 = bolt_pos - 0.005 * bolt_axis
    p_k1 = bolt_pos + 0.030 * bolt_axis
    p_k2 = bolt_pos - 0.030 * bolt_axis

    dets_by_view: dict[str, list[D.Detection2D]] = {}
    for vname in ("left", "right"):
        cam = cams[vname]
        # Fixture 6 keypoints
        fk_xy = np.zeros((6, 2), dtype=np.float64)
        fk_cf = np.ones(6, dtype=np.float64) * 0.95
        for idx, sid in enumerate(D.DEFAULT_FIXTURE_KPTS):
            lx, ly = SEAT_LOCAL_GEOM[sid]["local_xy"]
            w_pt = np.array([true_fx + c_f * lx - s_f * ly, true_fy + s_f * lx + c_f * ly, FIXTURE_BLOCK_TOP_Z])
            u, v, _ = cam.project(w_pt[None, :])
            fk_xy[idx] = [float(u[0]), float(v[0])]
        fix_det = D.Detection2D(vname, 13, "fixture", "fixture", 0.96, (180.0, 60.0, 440.0, 260.0), fk_xy, fk_cf)

        # Bolt 3 keypoints
        bk_xy = np.zeros((6, 2), dtype=np.float64)
        bk_cf = np.array([0.95, 0.95, 0.95, 0.0, 0.0, 0.0], dtype=np.float64)
        for ki, kp3 in enumerate((p_k0, p_k1, p_k2)):
            u, v, _ = cam.project(kp3[None, :])
            bk_xy[ki] = [float(u[0]), float(v[0])]
        u_mid, v_mid = bk_xy[0]
        bolt_det = D.Detection2D(
            vname, 0, "bolt_m16x50", "bolt", 0.92, (u_mid - 20, v_mid - 15, u_mid + 20, v_mid + 15), bk_xy, bk_cf
        )
        dets_by_view[vname] = [fix_det, bolt_det]

    out = D.reconstruct_scene_3d(dets_by_view, cams)
    assert out["fixture_detected"] is True
    assert math.hypot(out["fixture_position"][0] - true_fx, out["fixture_position"][1] - true_fy) < 1e-4
    assert abs(out["fixture_yaw"] - true_fyaw) < 1e-4
    assert set(out["seats"].keys()) >= {"tray", "bore_m16", "bore_m24", "bore_m30", "hex_m24", "hex_m30", "hex_m36"}

    assert len(out["parts"]) == 1
    part = out["parts"][0]
    assert part["kind"] == "bolt" and part["class_name"] == "bolt_m16x50"
    assert float(np.linalg.norm(np.asarray(part["position"]) - bolt_pos)) < 1e-4
    assert float(np.dot(np.asarray(part["axis"]), bolt_axis)) > 0.9999

    norm = normalize_scene(
        {
            "parts": [{**part, "kind": part["class_name"], "shape": part["kind"]}],
            "seats": out["seats"],
            "fixture_yaw": out["fixture_yaw"],
            "fixture_confidence": out["fixture_confidence"],
        }
    )
    assert len(norm.parts) == 1 and norm.parts[0].raw_kind == "bolt_m16x50"


if __name__ == "__main__":
    test_side_and_wrist_projection_triangulation_roundtrip()
    test_reconstruct_scene_3d_bolt_and_fixture_roundtrip()
    print("ALL GEOMETRY TESTS PASSED")
