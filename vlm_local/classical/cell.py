"""The real-arm cell the simulator reproduces (engine `assets/cell/cell.json`, engine ad65e2e).

Frames. The policy works internally in the simulator's world frame: the table top is z = 0 and the
robot base stands at (0, 0, BASE_Z). Observations (`ee_pos`, `ee_quat`) arrive in the robot base frame
(UR `base_link`), so `ee_pos` gets +BASE_Z on the way in. There is no rotation between the two frames.

Layout. The engine designed its task spawns on a fixture at REFERENCE_ORIGIN and moves every spawn,
bound and view by LAYOUT_SHIFT to follow the real fixture's nominal position. Constants this policy
inherited from that reference layout go through `shifted()` the same way.
"""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

TABLE_Z_IN_BASE = -0.187
BASE_Z = -TABLE_Z_IN_BASE  # robot base plane above the table top (world z)
BASE_WORLD = np.array([0.0, 0.0, BASE_Z], dtype=np.float64)

# Table, in the base frame (= world xy): centre (0.621, 0.0361), 2.5473 m along y, 1.0639 m along x.
TABLE_X = (0.621 - 0.5 * 1.0639, 0.621 + 0.5 * 1.0639)
TABLE_Y = (0.0361 - 0.5 * 2.5473, 0.0361 + 0.5 * 2.5473)

FIXTURE_NOMINAL_XY = (0.594, 0.009)
FIXTURE_NOMINAL_YAW = math.radians(-90.0)

REFERENCE_ORIGIN = (0.66, 0.15)
LAYOUT_SHIFT = (FIXTURE_NOMINAL_XY[0] - REFERENCE_ORIGIN[0], FIXTURE_NOMINAL_XY[1] - REFERENCE_ORIGIN[1])

SIDE_FOVY = 54.9
WRIST_FOVY = 61.1  # Gemini 335Lg on the bracket, factory fy 305.2 px at 640x360

# Tripod cameras: base-frame position and MuJoCo xyaxes (camera +x right, +y up, looks along -z).
# Nominal poses from engine `assets/cell/cell.json` (used by simulator).
SIDE_CAMERAS_BASE_SIM: dict[str, tuple[tuple[float, float, float], tuple[float, ...]]] = {
    "left": ((1.0258, 0.3027, 0.0043), (-0.57209, 0.82018, -0.00376, -0.09034, -0.05846, 0.99419)),
    "right": ((1.1212, -0.1976, 0.3731), (0.43279, 0.90074, -0.03695, -0.59615, 0.3167, 0.73777)),
}

# PnP-refined extrinsics measured against UR12e FK fingertips across 36 real-cell frames
# (reduces stereo 3D error from 48.7 mm to 4.6 mm / 1.0 mm epipolar RMS).
SIDE_CAMERAS_BASE_REAL: dict[str, tuple[tuple[float, float, float], tuple[float, ...]]] = {
    "left": ((1.0046, 0.3061, 0.0000), (-0.580386, 0.814122, -0.018922, -0.072740, -0.028686, 0.996938)),
    "right": ((1.0998, -0.1988, 0.3546), (0.401880, 0.914728, -0.042023, -0.569877, 0.285766, 0.770440)),
}

SIDE_CAMERAS_BASE = SIDE_CAMERAS_BASE_SIM

# Real-cell wrist bracket is mounted rotated +157.5 deg (7*pi/8) around tool0 +Z relative to sim embodiment.py,
# with PnP-refined Gemini 335Lg optical pose in tool0 frame.
REAL_BRACKET_YAW_RAD = math.radians(157.5)
WRIST_CAMERA_POS_REAL = ( -0.07435, -0.09079, 0.05161 )
WRIST_CAMERA_XYAXES_REAL = ( 0.927147, -0.372104, -0.044005, -0.340564, -0.885831, 0.315149 )

# Real Robotiq 2F-140 ROS 2 bridge reports `1 - finger_joint / 0.695` (~0.009 empty, 0.203 M30 shank,
# 0.308 M30 nut, 0.396 M33 washer) vs MuJoCo's `1 + q / 0.800` (~0.073 empty, 0.315 M30 shank, 0.420 M30 nut).
REAL_GRIPPER_OFFSET = 0.073
REAL_GRIPPER_SCALE = 1.08

_AUTO_REAL_CELL: bool | None = None


def is_real_cell() -> bool:
    """Return True when operating on the physical cell (`RC_REAL_CELL=1` or auto-detected from real TCP)."""
    mode = os.environ.get("RC_REAL_CELL", "auto").strip().lower()
    if mode in ("1", "true", "yes", "on", "real"):
        return True
    if mode in ("0", "false", "no", "off", "sim"):
        return False
    return bool(_AUTO_REAL_CELL)


def detect_real_cell_from_obs(obs: dict[str, Any] | None) -> bool:
    """Auto-detect physical cell vs simulator from proprioceptive TCP convention when `RC_REAL_CELL=auto`."""
    global _AUTO_REAL_CELL
    mode = os.environ.get("RC_REAL_CELL", "auto").strip().lower()
    if mode in ("1", "true", "yes", "on", "real"):
        return True
    if mode in ("0", "false", "no", "off", "sim"):
        return False
    if not isinstance(obs, dict):
        return bool(_AUTO_REAL_CELL)
    raw = obs.get("proprio")
    if not isinstance(raw, dict):
        return bool(_AUTO_REAL_CELL)
    jp = raw.get("joint_pos")
    ee_raw = raw.get("ee_pos")
    if jp is not None and ee_raw is not None:
        q = np.asarray(jp, dtype=np.float64).reshape(-1)
        ee = np.asarray(ee_raw, dtype=np.float64).reshape(-1)
        if q.size == 6 and ee.size == 3 and np.all(np.isfinite(q)) and np.all(np.isfinite(ee)):
            from .kinematics import fk_pose_sim

            sim_ee, _ = fk_pose_sim(q)
            # Simulator ee_pos + BASE_WORLD matches sim_ee within <0.5 mm; real ROS URDF TCP differs by 151.3 mm.
            _AUTO_REAL_CELL = bool(float(np.linalg.norm((ee + BASE_WORLD) - sim_ee)) > 0.050)
    return bool(_AUTO_REAL_CELL)


def normalize_gripper_reading(grip_raw: float, *, real_cell: bool | None = None) -> float:
    """Map real Robotiq 2F-140 gripper opening onto the simulator's proprio scale when on real hardware."""
    g = float(np.clip(grip_raw, 0.0, 1.0))
    use_real = is_real_cell() if real_cell is None else bool(real_cell)
    if not use_real:
        return g
    if g >= 0.86:
        return g
    return float(np.clip(REAL_GRIPPER_OFFSET + REAL_GRIPPER_SCALE * g, 0.0, 1.0))


def shifted(p):
    """A point of the engine's reference layout -> the real cell layout (xy shift, z unchanged)."""
    return (float(p[0]) + LAYOUT_SHIFT[0], float(p[1]) + LAYOUT_SHIFT[1], *[float(v) for v in p[2:]])


def xyaxes_to_mat(xyaxes) -> np.ndarray:
    x = np.asarray(xyaxes[:3], float)
    x = x / np.linalg.norm(x)
    y = np.asarray(xyaxes[3:6], float)
    y = y - float(np.dot(x, y)) * x
    y = y / np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


def side_camera_pose(name: str, *, real_cell: bool | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(world position, 3x3 camera-to-world rotation with OpenGL/MuJoCo axes) of a tripod camera."""
    use_real = is_real_cell() if real_cell is None else bool(real_cell)
    table = SIDE_CAMERAS_BASE_REAL if use_real else SIDE_CAMERAS_BASE_SIM
    pos, xyaxes = table[name]
    return np.asarray(pos, float) + BASE_WORLD, xyaxes_to_mat(xyaxes)


def base_to_world(p) -> np.ndarray:
    return np.asarray(p, float) + BASE_WORLD


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_off_nominal(yaw: float) -> float:
    return abs(wrap(float(yaw) - FIXTURE_NOMINAL_YAW))


# The motion code chose its approach and grasp headings, and the "leaning" face, relative to a fixture at
# yaw 0 (tray toward the robot). Those choices are about the world (reach away from the base, keep the
# wrist camera clear, lean face toward world -y), so they use the fixture's deviation from nominal:
# reach_yaw(fyaw) == fyaw at the old layout and ~0 here. Seat-aligned choices (tray long axis, hex flats)
# keep the true fixture yaw.
REACH_YAW_OFFSET = -FIXTURE_NOMINAL_YAW


def reach_yaw(fyaw: float) -> float:
    return float(fyaw) + REACH_YAW_OFFSET
