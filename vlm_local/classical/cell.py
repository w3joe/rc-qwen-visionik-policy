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
SIDE_CAMERAS_BASE: dict[str, tuple[tuple[float, float, float], tuple[float, ...]]] = {
    "left": ((1.0258, 0.3027, 0.0043), (-0.57209, 0.82018, -0.00376, -0.09034, -0.05846, 0.99419)),
    "right": ((1.1212, -0.1976, 0.3731), (0.43279, 0.90074, -0.03695, -0.59615, 0.3167, 0.73777)),
}


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


def side_camera_pose(name: str) -> tuple[np.ndarray, np.ndarray]:
    """(world position, 3x3 camera-to-world rotation with OpenGL/MuJoCo axes) of a tripod camera."""
    pos, xyaxes = SIDE_CAMERAS_BASE[name]
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
