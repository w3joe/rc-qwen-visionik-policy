"""Per-episode side-camera self-calibration anchored on the wrist camera.

The tripod cameras are jittered every episode (about +-1.5 cm, +-1.5 deg), which moves points
seen through their nominal poses by 2-3 cm on the table. The wrist camera's pose is exact
(forward kinematics of the measured joints). So:

1. seats the wrist camera sees confidently are located exactly (ray / seat-top plane);
2. each side camera fits the whole fixture (6 seat keypoints, rigid 2D fit to the known layout)
   through its nominal pose;
3. a planar rigid correction (dx, dy and, with >= 2 anchors, dyaw about the anchor centroid) maps
   that camera's fixture onto the wrist anchors; it is applied to the camera pose, so every later
   ray from that camera (parts included) is corrected too.

Without a wrist anchor, a camera is left nominal.
"""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

WRIST_MIN_KCONF = float(os.environ.get("RC_SELFCAL_WRIST_KCONF", "0.6"))
SIDE_MIN_KCONF = float(os.environ.get("RC_SELFCAL_SIDE_KCONF", "0.5"))
MAX_SHIFT_M = 0.08
MAX_YAW_RAD = math.radians(6.0)


def _rigid2d(src: np.ndarray, dst: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Least-squares planar rotation+translation mapping src->dst, rotation about src centroid."""
    cs, cd = src.mean(0), dst.mean(0)
    if len(src) == 1:
        return float(cd[0] - cs[0]), float(cd[1] - cs[1]), 0.0, cs
    a, b = src - cs, dst - cd
    yaw = math.atan2(float(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])), float(np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1])))
    return float(cd[0] - cs[0]), float(cd[1] - cs[1]), yaw, cs


def _fixture_fit(points: dict[str, np.ndarray], layout: dict[str, tuple[float, float]]) -> tuple[float, float, float] | None:
    ids = [s for s in points if s in layout]
    if len(ids) < 3:
        return None
    L = np.array([layout[s] for s in ids], float)
    W = np.array([points[s][:2] for s in ids], float)
    dx, dy, yaw, c = _rigid2d(L, W)
    # convert "rotate about local centroid c then shift" into pose (x, y, yaw) of the local origin
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    ox = c[0] + dx - (cy_ * c[0] - sy_ * c[1])
    oy = c[1] + dy - (sy_ * c[0] + cy_ * c[1])
    return ox, oy, yaw


def _seat_world(pose: tuple[float, float, float], lxy: tuple[float, float]) -> np.ndarray:
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([x + c * lxy[0] - s * lxy[1], y + s * lxy[0] + c * lxy[1]])


def calibrate_side_cameras(
    fixture_dets: dict[str, Any],
    cams: dict[str, Any],
    kpt_seats: dict[int, str],
    layout: dict[str, tuple[float, float]],
    top_z: dict[str, float],
) -> dict[str, Any]:
    """Mutates cams['left'/'right'] in place. Returns a small report dict."""
    report: dict[str, Any] = {"anchors": 0, "cams": {}}
    if os.environ.get("RC_SELFCAL", "1") == "0":
        return report
    wdet, wcam = fixture_dets.get("wrist"), cams.get("wrist")
    anchors: dict[str, np.ndarray] = {}
    if wdet is not None and wcam is not None:
        for k, sid in kpt_seats.items():
            if sid not in layout or not wdet.usable_kpt(k, wcam.width, wcam.height, min_conf=WRIST_MIN_KCONF):
                continue
            p = wcam.intersect_z(float(wdet.kpts_xy[k, 0]), float(wdet.kpts_xy[k, 1]), top_z.get(sid, 0.115))
            if p is not None:
                anchors[sid] = np.asarray(p, float)
    report["anchors"] = len(anchors)
    if not anchors:
        return report
    for name in ("left", "right"):
        det, cam = fixture_dets.get(name), cams.get(name)
        if det is None or cam is None:
            continue
        pts: dict[str, np.ndarray] = {}
        for k, sid in kpt_seats.items():
            if sid in layout and det.usable_kpt(k, cam.width, cam.height, min_conf=SIDE_MIN_KCONF):
                p = cam.intersect_z(float(det.kpts_xy[k, 0]), float(det.kpts_xy[k, 1]), top_z.get(sid, 0.115))
                if p is not None:
                    pts[sid] = np.asarray(p, float)
        pose = _fixture_fit(pts, layout)
        if pose is None:
            continue
        ids = list(anchors)
        src = np.array([_seat_world(pose, layout[s]) for s in ids])
        dst = np.array([anchors[s][:2] for s in ids])
        dx, dy, yaw, pivot = _rigid2d(src, dst)
        if len(ids) < 2:
            yaw = 0.0
        if math.hypot(dx, dy) > MAX_SHIFT_M or abs(yaw) > MAX_YAW_RAD:
            report["cams"][name] = {"skipped": True, "dx": dx, "dy": dy, "yaw_deg": math.degrees(yaw)}
            continue
        c, s = math.cos(yaw), math.sin(yaw)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        piv = np.array([pivot[0], pivot[1], 0.0])
        cam.pos = Rz @ (np.asarray(cam.pos, float) - piv) + piv + np.array([dx, dy, 0.0])
        cam.mat = Rz @ np.asarray(cam.mat, float)
        report["cams"][name] = {"dx": dx, "dy": dy, "yaw_deg": math.degrees(yaw), "n_side": len(pts)}
    return report
