"""Classical calibrated multi-view RGB perception for Reality Check tabletop scenes.

Uses ONLY policy-wire inputs:
  - RGB images (`left`, `right`, `wrist`, optional `goal`)
  - Proprioception (`joint_pos`, `ee_pos`, `ee_quat`, `gripper`)
  - Task `instruction`
  - Public kit geometry, public fixture v3 geometry, and nominal camera poses

Zero engine state, zero seeds/indices, zero solution tables, zero teacher access.

Main entry point:
    estimate_scene(obs: dict) -> dict
"""
from __future__ import annotations
# NOTE: vendored from the vision-IK ("classical") policy; the only change is the extra `cameras` key at the end
# of estimate_scene().

import base64
import io
import math
import re
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from . import cell as _cell

# ==============================================================================
# Public Geometry & Nominal Camera Constants (from public participant engine)
# ==============================================================================

# Real cell (engine ad65e2e, cell.py): table top z = 0 in the world, robot base at (0, 0, 0.187).
TABLE_X = _cell.TABLE_X
TABLE_Y = _cell.TABLE_Y

# Nominal side cameras (640x360): the calibrated Gemini 336s. (eye, look-at) kept for legacy callers;
# exact rotations come from `side_camera_pose` (cell.json xyaxes).
SIDE_CAMERA_FOVY = _cell.SIDE_FOVY


def _eye_target(name: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    pos, mat = _cell.side_camera_pose(name)
    tgt = pos - mat[:, 2]
    return tuple(float(v) for v in pos), tuple(float(v) for v in tgt)


SIDE_CAMERA_POSES: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    n: _eye_target(n) for n in ("left", "right")
}
side_camera_pose = _cell.side_camera_pose

# Goal views are the same three cameras (schema v3 `goal` dict); the single-image goal paths use the
# left view through the nominal left camera.
GOAL_VIEW = "left"
GOAL_CAMERA_POS, GOAL_CAMERA_TARGET = SIDE_CAMERA_POSES[GOAL_VIEW]
GOAL_CAMERA_FOVY = _cell.SIDE_FOVY

# Wrist camera rigid mount relative to tool0 and gripperframe (ee)
GRIPPER_SPACER_M = 0.0244
BASE_MOUNT_Z_M = 0.007
GRIPPERFRAME_POS = (0.0, 0.0, 0.100)
GRIPPERFRAME_QUAT = (0.7071067811865476, 0.0, 0.0, -0.7071067811865476)  # Rz(-90 deg)
WRIST_CAMERA_POS = (0.0140, 0.1103, 0.0649)
WRIST_CAMERA_XYAXES = (-1.0, 0.0, 0.0, 0.0, 0.92388, 0.38268)
WRIST_CAMERA_FOVY = _cell.WRIST_FOVY

# Printed fixture v3 rev-3 geometry (metres, fixture-local: origin at footprint centre on z=0)
FIXTURE_ORIGIN = (_cell.FIXTURE_NOMINAL_XY[0], _cell.FIXTURE_NOMINAL_XY[1], 0.0)
FIXTURE_NOMINAL_YAW = _cell.FIXTURE_NOMINAL_YAW
FIXTURE_X0, FIXTURE_XM, FIXTURE_X1 = -0.125, -0.026, 0.125
FIXTURE_Y0, FIXTURE_Y1 = -0.115, 0.115
FIXTURE_TRAY_FLOOR_Z = 0.004
FIXTURE_TRAY_WALL_Z = 0.022
FIXTURE_BLOCK_TOP_Z = 0.115
FIXTURE_TRAY_INNER_X = (-0.121, -0.026)
FIXTURE_TRAY_INNER_Y = (-0.111, 0.111)
# REALITY CHECK inlay text bounding box on -y face (metres, fixture-local)
FIXTURE_TEXT_X = (-0.013, 0.112)
FIXTURE_TEXT_Z = (0.021, 0.061)

FIXTURE_BORES: tuple[dict[str, Any], ...] = (
    {"id": "bore_m24", "xy": (0.0072, 0.0860), "diameter": 0.028, "floor_z": 0.040, "top_z": 0.115, "part": "bolt_m24x70"},
    {"id": "bore_m16", "xy": (0.0072, 0.0035), "diameter": 0.020, "floor_z": 0.060, "top_z": 0.115, "part": "bolt_m16x50"},
    {"id": "bore_m30", "xy": (0.0072, -0.0825), "diameter": 0.035, "floor_z": 0.010, "top_z": 0.115, "part": "bolt_m30x100"},
)
FIXTURE_HEX_POCKETS: tuple[dict[str, Any], ...] = (
    {"id": "hex_m30", "xy": (0.07735, 0.0775), "across_flats": 0.051, "floor_z": 0.102, "top_z": 0.115, "part": "nut_m30"},
    {"id": "hex_m36", "xy": (0.07735, -0.0050), "across_flats": 0.060, "floor_z": 0.100, "top_z": 0.115, "part": "nut_m36"},
    {"id": "hex_m24", "xy": (0.07735, -0.0825), "across_flats": 0.041, "floor_z": 0.105, "top_z": 0.115, "part": "nut_m24"},
)


@dataclass(frozen=True)
class KitPart:
    part_id: str
    shape: str  # "bolt", "nut", "washer"
    diameter: float = 0.0       # bolt shank
    length: float = 0.0         # bolt under-head length
    head_height: float = 0.0    # bolt head height
    across_flats: float = 0.0   # bolt head or nut AF
    thickness: float = 0.0      # nut or washer thickness
    hole: float = 0.0           # nut hole or washer ID
    outer_diameter: float = 0.0 # washer OD
    tip_over_deg: float = 30.0  # leaning bolt tip-over angle

    @property
    def head_local_x(self) -> float:
        return 0.5 * self.length

    @property
    def under_head_x(self) -> float:
        return 0.5 * (self.length - self.head_height)

    @property
    def tip_x(self) -> float:
        return -0.5 * (self.length + self.head_height)

    @property
    def full_length(self) -> float:
        return self.length + self.head_height

    @property
    def lying_tilt(self) -> float:
        """Natural tilt (rad) of a lying bolt resting on head flat & shank tip."""
        if self.shape != "bolt":
            return 0.0
        head_r = 0.5 * self.across_flats
        shank_r = 0.5 * self.diameter
        # Distance between head bearing flat contact and shank tip contact
        span = max(self.length + 0.5 * self.head_height, 1e-4)
        return math.asin(min(0.4, max(0.0, (head_r - shank_r) / span)))

    @property
    def lying_z(self) -> float:
        if self.shape == "bolt":
            t = self.lying_tilt
            # Head flat at x=+head_local_x sits at z = 0.5*across_flats; body origin sits slightly lower
            return 0.5 * self.across_flats * math.cos(t) - self.head_local_x * math.sin(t) + 0.0012
        return 0.5 * self.thickness

    @property
    def standing_z(self) -> float:
        if self.shape == "bolt":
            return 0.5 * self.full_length
        return 0.5 * self.across_flats

    @property
    def radius_xy(self) -> float:
        if self.shape == "bolt":
            return 0.5 * self.full_length
        if self.shape == "nut":
            return self.across_flats / math.sqrt(3.0)
        return 0.5 * self.outer_diameter


KIT_PARTS: dict[str, KitPart] = {
    "bolt_m16x50": KitPart("bolt_m16x50", "bolt", diameter=0.016, length=0.050, head_height=0.010, across_flats=0.024, tip_over_deg=25.0),
    "bolt_m24x70": KitPart("bolt_m24x70", "bolt", diameter=0.024, length=0.070, head_height=0.015, across_flats=0.036, tip_over_deg=27.0),
    "bolt_m30x100": KitPart("bolt_m30x100", "bolt", diameter=0.030, length=0.100, head_height=0.0187, across_flats=0.046, tip_over_deg=24.0),
    "bolt_m24x50": KitPart("bolt_m24x50", "bolt", diameter=0.024, length=0.050, head_height=0.015, across_flats=0.036, tip_over_deg=35.0),
    "bolt_m30x60": KitPart("bolt_m30x60", "bolt", diameter=0.030, length=0.060, head_height=0.0187, across_flats=0.046, tip_over_deg=37.0),
    "nut_m12": KitPart("nut_m12", "nut", across_flats=0.018, thickness=0.0108, hole=0.012),
    "nut_m16": KitPart("nut_m16", "nut", across_flats=0.024, thickness=0.0148, hole=0.016),
    "nut_m20": KitPart("nut_m20", "nut", across_flats=0.030, thickness=0.0180, hole=0.020),
    "nut_m24": KitPart("nut_m24", "nut", across_flats=0.036, thickness=0.0215, hole=0.024),
    "nut_m30": KitPart("nut_m30", "nut", across_flats=0.046, thickness=0.0256, hole=0.030),
    "nut_m36": KitPart("nut_m36", "nut", across_flats=0.055, thickness=0.0310, hole=0.036),
    "washer_m12": KitPart("washer_m12", "washer", thickness=0.0025, hole=0.013, outer_diameter=0.024),
    "washer_m16": KitPart("washer_m16", "washer", thickness=0.003, hole=0.017, outer_diameter=0.030),
    "washer_m20": KitPart("washer_m20", "washer", thickness=0.003, hole=0.021, outer_diameter=0.037),
    "washer_m33": KitPart("washer_m33", "washer", thickness=0.005, hole=0.034, outer_diameter=0.060),
    "washer_m30": KitPart("washer_m30", "washer", thickness=0.006, hole=0.035, outer_diameter=0.062),
}


# ==============================================================================
# Math, Image Decoding & Camera Model
# ==============================================================================

def _decode_image(img: Any) -> np.ndarray | None:
    if img is None:
        return None
    if isinstance(img, dict):  # schema v3 goal: {"wrist", "left", "right"}
        img = img.get(GOAL_VIEW) or next((v for v in img.values() if v), None)
        if img is None:
            return None
    if isinstance(img, np.ndarray):
        if img.ndim == 3 and img.shape[2] >= 3:
            return np.asarray(img[..., :3], dtype=np.uint8)
        return None
    if isinstance(img, str):
        raw = base64.b64decode(img)
        with Image.open(io.BytesIO(raw)) as pil:
            return np.asarray(pil.convert("RGB"), dtype=np.uint8)
    if isinstance(img, (bytes, bytearray)):
        with Image.open(io.BytesIO(img)) as pil:
            return np.asarray(pil.convert("RGB"), dtype=np.uint8)
    return None


def _is_valid_goal_image(img: np.ndarray | None) -> bool:
    if img is None or img.ndim != 3 or img.shape[0] < 100 or img.shape[1] < 100:
        return False
    # Reject flat stand-in goal pictures (constant interior color)
    interior = img[40:-40, 40:-40]
    return float(interior.std()) > 8.0


def _quat_to_mat(q: np.ndarray | tuple[float, ...]) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=float)


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _xyaxes_to_mat(xyaxes: tuple[float, ...]) -> np.ndarray:
    x = np.asarray(xyaxes[:3], float)
    x /= np.linalg.norm(x)
    y = np.asarray(xyaxes[3:], float)
    y = y - np.dot(x, y) * x
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


def _look_mat(eye: tuple[float, float, float], target: tuple[float, float, float]) -> np.ndarray:
    e = np.asarray(eye, float)
    t = np.asarray(target, float)
    fwd = t - e
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return np.column_stack([right, up, -fwd])


def _rodrigues_batch(rvecs: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rvecs, axis=-1, keepdims=True)
    safe = np.where(theta > 1e-12, theta, 1.0)
    k = np.where(theta > 1e-12, rvecs / safe, 0.0)
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    B = len(rvecs)
    K = np.zeros((B, 3, 3), dtype=float)
    K[:, 0, 1], K[:, 0, 2] = -kz, ky
    K[:, 1, 0], K[:, 1, 2] = kz, -kx
    K[:, 2, 0], K[:, 2, 1] = -ky, kx
    st = np.sin(theta)[:, :, None]
    ct = (1.0 - np.cos(theta))[:, :, None]
    return np.eye(3)[None, :, :] + st * K + ct * (K @ K)


@dataclass
class Camera:
    name: str
    pos: np.ndarray   # (3,) world position
    mat: np.ndarray   # (3, 3) columns = [right, up, -forward] in world frame
    fovy_deg: float
    width: int
    height: int

    @property
    def f(self) -> float:
        return 0.5 * self.height / math.tan(math.radians(self.fovy_deg) * 0.5)

    @property
    def cx(self) -> float:
        return 0.5 * self.width

    @property
    def cy(self) -> float:
        return 0.5 * self.height

    def project(self, pts_w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pts = np.asarray(pts_w, float)
        p_c = (pts - self.pos) @ self.mat
        depth = -p_c[..., 2]
        safe_d = np.where(depth > 1e-4, depth, 1e-4)
        f = self.f
        u = self.cx + f * (p_c[..., 0] / safe_d)
        v = self.cy - f * (p_c[..., 1] / safe_d)
        return u, v, depth

    def unproject_ray(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        d_c = np.array([(u - self.cx) / self.f, -(v - self.cy) / self.f, -1.0], dtype=float)
        d_c /= np.linalg.norm(d_c)
        return self.pos.copy(), self.mat @ d_c

    def intersect_z(self, u: float, v: float, z0: float) -> np.ndarray | None:
        o, d = self.unproject_ray(u, v)
        if abs(float(d[2])) < 1e-6:
            return None
        t = (z0 - float(o[2])) / float(d[2])
        if t <= 0.02:
            return None
        return o + t * d


def _wrist_camera_from_proprio(proprio: dict[str, Any] | None, width: int = 640, height: int = 360) -> Camera | None:
    if not proprio:
        return None
    q_raw = proprio.get("joint_pos")
    if q_raw is not None:
        q = np.asarray(q_raw, dtype=np.float64).reshape(-1)
        if q.size == 6 and np.all(np.isfinite(q)):
            from .kinematics import wrist_camera_pose

            p_w_c, R_w_c = wrist_camera_pose(q)
            return Camera("wrist", p_w_c, R_w_c, WRIST_CAMERA_FOVY, width, height)
    if "ee_pos" not in proprio or "ee_quat" not in proprio:
        return None
    p_w_ee = np.asarray(proprio["ee_pos"], float) + _cell.BASE_WORLD  # base frame -> world
    R_w_ee = _quat_to_mat(proprio["ee_quat"])
    p_t_ee = np.array([0.0, 0.0, GRIPPER_SPACER_M + BASE_MOUNT_Z_M + GRIPPERFRAME_POS[2]], dtype=float)
    R_t_ee = _quat_to_mat(GRIPPERFRAME_QUAT)
    R_w_t = R_w_ee @ R_t_ee.T
    p_w_t = p_w_ee - R_w_t @ p_t_ee
    p_w_c = p_w_t + R_w_t @ np.asarray(WRIST_CAMERA_POS, float)
    R_w_c = R_w_t @ _xyaxes_to_mat(WRIST_CAMERA_XYAXES)
    return Camera("wrist", p_w_c, R_w_c, WRIST_CAMERA_FOVY, width, height)


def _triangulate_rays(origins: list[np.ndarray], dirs: list[np.ndarray]) -> tuple[np.ndarray, float]:
    A = np.zeros((3, 3), dtype=float)
    b = np.zeros(3, dtype=float)
    eye3 = np.eye(3)
    for o, d in zip(origins, dirs):
        P = eye3 - np.outer(d, d)
        A += P
        b += P @ o
    pt = np.linalg.solve(A + 1e-9 * eye3, b)
    errs = [float(np.linalg.norm((pt - o) - np.dot(pt - o, d) * d)) for o, d in zip(origins, dirs)]
    return pt, float(np.sqrt(np.mean(np.square(errs))))


def _bilinear_sample(arr: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    H, W = arr.shape[:2]
    u_c = np.clip(u, 0.0, W - 1.001)
    v_c = np.clip(v, 0.0, H - 1.001)
    u0 = u_c.astype(np.int32)
    v0 = v_c.astype(np.int32)
    du = u_c - u0
    dv = v_c - v0
    out = (
        arr[v0, u0] * (1.0 - du) * (1.0 - dv)
        + arr[v0, u0 + 1] * du * (1.0 - dv)
        + arr[v0 + 1, u0] * (1.0 - du) * dv
        + arr[v0 + 1, u0 + 1] * du * dv
    )
    valid = (u >= 1.0) & (u < W - 2.0) & (v >= 1.0) & (v < H - 2.0)
    return out, valid


def _box_blur(arr: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return arr
    k = 2 * r + 1
    p = np.pad(arr, ((0, 0), (r, r)), mode="edge")
    cs = np.pad(np.cumsum(p, axis=1), ((0, 0), (1, 0)), mode="constant")
    tmp = (cs[:, k:] - cs[:, :-k]) / k
    p2 = np.pad(tmp, ((r, r), (0, 0)), mode="edge")
    cs2 = np.pad(np.cumsum(p2, axis=0), ((1, 0), (0, 0)), mode="constant")
    return (cs2[k:, :] - cs2[:-k, :]) / k


def _dilate_mask(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0 or not np.any(mask):
        return mask
    return _box_blur(mask.astype(np.float32), r) > (0.5 / ((2 * r + 1) ** 2))


def _erode_mask(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask
    return ~_dilate_mask(~mask, r)


# ==============================================================================
# Analytical UR12e FK & Robot Self-Occlusion Mask
# ==============================================================================

def _ur12e_capsules(proprio: dict[str, Any] | None) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Compute world-frame 3D capsules (p0, p1, radius) covering the UR12e + bracket + 2F-140."""
    if not proprio:
        return []
    q = proprio.get("joint_pos")
    capsules: list[tuple[np.ndarray, np.ndarray, float]] = []

    if q is not None and len(q) == 6:
        q = [float(x) for x in q]
        # Exact body chain from ur12e.xml
        def step(p: np.ndarray, R: np.ndarray, dp: tuple[float, float, float], dq: tuple[float, float, float, float], angle: float):
            p_new = p + R @ np.asarray(dp, float)
            R_new = R @ _quat_to_mat(dq) @ _rot_z(angle)
            return p_new, R_new

        p0 = np.zeros(3)
        R0 = _quat_to_mat((0.0, 0.0, 0.0, 1.0))  # base_link_inertia
        capsules.append((np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.22]), 0.105))

        p1, R1 = step(p0, R0, (0.0, 0.0, 0.1807), (1.0, 0.0, 0.0, 0.0), q[0])
        p2, R2 = step(p1, R1, (0.0, 0.0, 0.0), (0.707107, 0.707107, 0.0, 0.0), q[1])
        p3, R3 = step(p2, R2, (-0.6127, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), q[2])
        p4, R4 = step(p3, R3, (-0.57155, 0.0, 0.17415), (1.0, 0.0, 0.0, 0.0), q[3])
        p5, R5 = step(p4, R4, (0.0, -0.11985, 0.0), (0.707107, 0.707107, 0.0, 0.0), q[4])
        p6, R6 = step(p5, R5, (0.0, 0.11655, 0.0), (0.707107, -0.707107, 0.0, 0.0), q[5])

        # Shoulder & upper arm
        capsules.append((p2 + R2 @ np.array([0.0, 0.0, 0.0]), p2 + R2 @ np.array([0.0, 0.0, 0.18]), 0.095))
        capsules.append((p2 + R2 @ np.array([-0.04, 0.0, 0.1762]), p2 + R2 @ np.array([-0.58, 0.0, 0.1762]), 0.080))
        # Forearm
        capsules.append((p3 + R3 @ np.array([-0.03, 0.0, 0.04]), p3 + R3 @ np.array([-0.54, 0.0, 0.04]), 0.068))
        # Wrists 1, 2, 3
        capsules.append((p4 + R4 @ np.array([0.0, 0.02, -0.05]), p4 + R4 @ np.array([0.0, -0.12, 0.02]), 0.062))
        capsules.append((p5 + R5 @ np.array([0.0, -0.02, -0.05]), p5 + R5 @ np.array([0.0, 0.10, 0.02]), 0.058))
        capsules.append((p6 + R6 @ np.array([0.0, 0.0, -0.07]), p6 + R6 @ np.array([0.0, 0.0, 0.02]), 0.058))

    # Gripper & wrist camera bracket from ee_pos, ee_quat, gripper
    if "ee_pos" in proprio and "ee_quat" in proprio:
        p_ee = np.asarray(proprio["ee_pos"], float)
        R_ee = _quat_to_mat(proprio["ee_quat"])
        # tool0 frame
        p_t_ee = np.array([0.0, 0.0, GRIPPER_SPACER_M + BASE_MOUNT_Z_M + GRIPPERFRAME_POS[2]])
        R_t_ee = _quat_to_mat(GRIPPERFRAME_QUAT)
        R_t = R_ee @ R_t_ee.T
        p_t = p_ee - R_t @ p_t_ee

        # Wrist bracket + Gemini 336 housing (on tool0 +y)
        capsules.append((p_t + R_t @ np.array([0.0, 0.0, 0.0]), p_t + R_t @ np.array([0.0, 0.135, 0.045]), 0.042))
        capsules.append((p_t + R_t @ np.array([-0.045, 0.122, 0.052]), p_t + R_t @ np.array([0.045, 0.122, 0.052]), 0.028))
        # Gripper palm body (along tool0 +z from 0.024 to 0.115)
        capsules.append((p_t + R_t @ np.array([0.0, 0.0, 0.02]), p_t + R_t @ np.array([0.0, 0.0, 0.115]), 0.052))
        # Two 2F-140 finger linkages + pads (opening along tool0 ±x)
        g = float(proprio.get("gripper", 1.0))
        half_open = 0.015 + 0.064 * max(0.0, min(1.0, g))
        for sgn in (-1.0, 1.0):
            # Linkage from palm to fingertip pad (generous radius covers linkage + bright pad tip)
            knuckle = p_t + R_t @ np.array([sgn * (half_open + 0.025), 0.0, 0.145])
            tip = p_t + R_t @ np.array([sgn * half_open, 0.0, 0.262])
            base_f = p_t + R_t @ np.array([sgn * 0.034, 0.0, 0.095])
            capsules.append((base_f, knuckle, 0.032))
            capsules.append((knuckle, tip, 0.032))
    return capsules


def _project_capsules_mask(
    capsules: list[tuple[np.ndarray, np.ndarray, float]],
    cam: Camera,
    pad_px: float = 4.0,
) -> np.ndarray:
    mask = np.zeros((cam.height, cam.width), dtype=bool)
    if not capsules:
        return mask
    H, W = cam.height, cam.width
    f = cam.f
    for p0, p1, rad in capsules:
        ts = np.linspace(0.0, 1.0, 8)
        pts = p0[None, :] * (1.0 - ts[:, None]) + p1[None, :] * ts[:, None]
        u, v, d = cam.project(pts)
        for ui, vi, di in zip(u, v, d):
            if di <= 0.08:
                continue
            rpx = f * (rad / di) + pad_px
            if ui + rpx < 0 or ui - rpx >= W or vi + rpx < 0 or vi - rpx >= H:
                continue
            x0 = max(0, int(math.floor(ui - rpx)))
            x1 = min(W, int(math.ceil(ui + rpx)) + 1)
            y0 = max(0, int(math.floor(vi - rpx)))
            y1 = min(H, int(math.ceil(vi + rpx)) + 1)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            mask[y0:y1, x0:x1] |= ((xx - ui) ** 2 + (yy - vi) ** 2) <= rpx * rpx
    return mask


# ==============================================================================
# Fixture Pose Estimation & Side-Camera Self-Calibration
# ==============================================================================

def _fixture_silhouette_samples(cam_pos_local: np.ndarray, n_per_edge: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Exact outer silhouette sample points & 3D inward directions in fixture-local metres."""
    X0, XM, X1 = FIXTURE_X0, FIXTURE_XM, FIXTURE_X1
    Y0, Y1 = FIXTURE_Y0, FIXTURE_Y1
    Z0, ZT, ZB = 0.0, FIXTURE_TRAY_WALL_Z, FIXTURE_BLOCK_TOP_Z
    NX, PX = np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
    NY, PY = np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0])
    NZ, PZ = np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 1.0])

    edges = [
        ([X0, Y0, Z0], [X1, Y0, Z0], NZ, NY),
        ([X1, Y0, Z0], [X1, Y1, Z0], NZ, PX),
        ([X1, Y1, Z0], [X0, Y1, Z0], NZ, PY),
        ([X0, Y1, Z0], [X0, Y0, Z0], NZ, NX),
        ([X0, Y0, ZT], [XM, Y0, ZT], PZ, NY),
        ([X0, Y1, ZT], [XM, Y1, ZT], PZ, PY),
        ([X0, Y0, ZT], [X0, Y1, ZT], PZ, NX),
        ([XM, Y0, ZB], [X1, Y0, ZB], PZ, NY),
        ([X1, Y0, ZB], [X1, Y1, ZB], PZ, PX),
        ([X1, Y1, ZB], [XM, Y1, ZB], PZ, PY),
        ([XM, Y1, ZB], [XM, Y0, ZB], PZ, NX),
        ([X0, Y0, Z0], [X0, Y0, ZT], NX, NY),
        ([X0, Y1, Z0], [X0, Y1, ZT], NX, PY),
        ([XM, Y0, ZT], [XM, Y0, ZB], NX, NY),
        ([XM, Y1, ZT], [XM, Y1, ZB], NX, PY),
        ([X1, Y0, Z0], [X1, Y0, ZB], PX, NY),
        ([X1, Y1, Z0], [X1, Y1, ZB], PX, PY),
    ]
    pts, in_dirs = [], []
    for p0, p1, n1, n2 in edges:
        a = np.array(p0, float)
        b = np.array(p1, float)
        v = cam_pos_local - 0.5 * (a + b)
        vis1 = float(np.dot(v, n1)) > 0.0
        vis2 = float(np.dot(v, n2)) > 0.0
        if vis1 != vis2:
            n_hid = n2 if vis1 else n1
            for t in np.linspace(0.10, 0.90, n_per_edge):
                pts.append(a + t * (b - a))
                in_dirs.append(-n_hid)
    return np.array(pts, float), np.array(in_dirs, float)


def _table_border_samples() -> tuple[np.ndarray, np.ndarray]:
    pts, ins = [], []
    for y in np.linspace(TABLE_Y[0] + 0.04, TABLE_Y[1] - 0.04, 20):
        pts.append([TABLE_X[1], y, 0.0]); ins.append([-1.0, 0.0, 0.0])
        pts.append([TABLE_X[0], y, 0.0]); ins.append([1.0, 0.0, 0.0])
    for x in np.linspace(TABLE_X[0] + 0.06, TABLE_X[1] - 0.04, 24):
        pts.append([x, TABLE_Y[0], 0.0]); ins.append([0.0, 1.0, 0.0])
        pts.append([x, TABLE_Y[1], 0.0]); ins.append([0.0, -1.0, 0.0])
    return np.array(pts, float), np.array(ins, float)


def _edge_maps(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute soft fixture score, soft table score, fine & coarse gradient maps."""
    f = img.astype(np.float32)
    R, G, B = f[..., 0], f[..., 1], f[..., 2]
    lum = 0.299 * R + 0.587 * G + 0.114 * B
    br = B - R
    s_br = np.clip((22.0 - br) / 10.0, -1.0, 1.0)
    soft_fix = np.where(lum < 48.0, np.minimum(s_br, -0.5), s_br)
    soft_tbl = 0.6 * np.clip((br - 10.0) / 12.0, -1.0, 1.0) + 0.4 * np.clip((105.0 - lum) / 25.0, -1.0, 1.0)
    gx = np.zeros_like(lum)
    gy = np.zeros_like(lum)
    gx[:, 1:-1] = np.abs(f[:, 2:, :] - f[:, :-2, :]).mean(axis=-1)
    gy[1:-1, :] = np.abs(f[2:, :, :] - f[:-2, :, :]).mean(axis=-1)
    grad = np.hypot(gx, gy)
    grad_fine = _box_blur(grad, 1)
    grad_coarse = _box_blur(grad, 4)
    return soft_fix, soft_tbl, grad_fine, grad_coarse


def _project_batch_cam(pts_w: np.ndarray, cpos: np.ndarray, cmat: np.ndarray, f: float, cx: float, cy: float):
    rel = pts_w[None, :, :] - cpos[:, None, :]
    p_c = np.einsum("bni,bij->bnj", rel, cmat)
    depth = -p_c[..., 2]
    safe_d = np.where(depth > 1e-4, depth, 1e-4)
    u = cx + f * (p_c[..., 0] / safe_d)
    v = cy - f * (p_c[..., 1] / safe_d)
    return u, v, depth


def _eval_fixture_poses_in_view(
    poses_xyyaw: np.ndarray,
    pts_loc: np.ndarray,
    in_loc: np.ndarray,
    cam: Camera,
    soft_fix: np.ndarray,
    grad_map: np.ndarray,
    robot_mask: np.ndarray | None,
    off_px: float = 2.5,
) -> np.ndarray:
    fx = poses_xyyaw[:, 0:1]
    fy = poses_xyyaw[:, 1:2]
    yaw = poses_xyyaw[:, 2:3]
    c = np.cos(yaw)
    s = np.sin(yaw)
    px, py, pz = pts_loc[:, 0][None, :], pts_loc[:, 1][None, :], pts_loc[:, 2][None, :]
    B, N = len(poses_xyyaw), len(pts_loc)
    pts_w = np.stack([fx + c * px - s * py, fy + s * px + c * py, np.broadcast_to(pz, (B, N))], axis=-1)
    ix, iy, iz = in_loc[:, 0][None, :], in_loc[:, 1][None, :], in_loc[:, 2][None, :]
    d_w = np.stack([c * ix - s * iy, s * ix + c * iy, np.broadcast_to(iz, (B, N))], axis=-1)

    u0, v0, d0 = cam.project(pts_w)
    u1, v1, _ = cam.project(pts_w + 0.008 * d_w)
    du, dv = u1 - u0, v1 - v0
    nrm = np.hypot(du, dv) + 1e-6
    nu, nv = du / nrm, dv / nrm

    s_in, ok1 = _bilinear_sample(soft_fix, u0 + off_px * nu, v0 + off_px * nv)
    s_out, ok2 = _bilinear_sample(soft_fix, u0 - off_px * nu, v0 - off_px * nv)
    ge, ok3 = _bilinear_sample(grad_map, u0, v0)
    ok = ok1 & ok2 & ok3 & (d0 > 0.1)
    if robot_mask is not None:
        ui = np.clip(np.round(u0).astype(np.int32), 0, cam.width - 1)
        vi = np.clip(np.round(v0).astype(np.int32), 0, cam.height - 1)
        ok &= ~robot_mask[vi, ui]
    sc = np.where(ok, np.clip(s_in - s_out, -1.2, 2.0) + 0.025 * np.clip(ge, 0.0, 60.0), 0.0)
    cnt = np.maximum(ok.sum(axis=-1), 1)
    return sc.sum(axis=-1) / cnt


def _fit_fixture_pose(
    views: list[tuple[Camera, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]],
) -> tuple[np.ndarray, float]:
    """Fit fixture (fx, fy, fyaw) across one or more views.

    Each view tuple is (cam, soft_fix, grad_fine, grad_coarse, robot_mask).
    """
    nom = np.array([FIXTURE_ORIGIN[0], FIXTURE_ORIGIN[1], 0.0])
    coarse_samples = []
    for cam, _sf, _gf, _gc, _rm in views:
        pl, il = _fixture_silhouette_samples(cam.pos - nom, n_per_edge=8)
        coarse_samples.append((pl, il))

    xs = np.linspace(FIXTURE_ORIGIN[0] - 0.055, FIXTURE_ORIGIN[0] + 0.055, 11)
    ys = np.linspace(FIXTURE_ORIGIN[1] - 0.055, FIXTURE_ORIGIN[1] + 0.055, 11)
    yaws = np.radians(np.linspace(-20.0, 20.0, 11))
    gx, gy, gyaw = np.meshgrid(xs, ys, yaws, indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel(), gyaw.ravel()], axis=-1)

    total_sc = np.zeros(len(grid), dtype=float)
    for (cam, sf, _gf, gc, rm), (pl, il) in zip(views, coarse_samples):
        total_sc += _eval_fixture_poses_in_view(grid, pl, il, cam, sf, gc, rm, off_px=3.0)
    top_idx = np.argsort(total_sc)[-3:][::-1]

    best_pose = grid[top_idx[0]].copy()
    best_sc = -1e9
    for idx in top_idx:
        pose = grid[idx].copy()
        fine_samples = []
        c, s = math.cos(pose[2]), math.sin(pose[2])
        for cam, _sf, _gf, _gc, _rm in views:
            rel = cam.pos - np.array([pose[0], pose[1], 0.0])
            cam_loc = np.array([c * rel[0] + s * rel[1], -s * rel[0] + c * rel[1], rel[2]])
            fine_samples.append(_fixture_silhouette_samples(cam_loc, n_per_edge=12))

        def score_batch(cands: np.ndarray) -> np.ndarray:
            acc = np.zeros(len(cands), dtype=float)
            for (cam, sf, gf, _gc, rm), (pf, inf) in zip(views, fine_samples):
                acc += _eval_fixture_poses_in_view(cands, pf, inf, cam, sf, gf, rm, off_px=2.5)
            return acc / max(len(views), 1)

        sc = float(score_batch(pose[None])[0])
        for step_xy, step_y in (
            (0.005, math.radians(2.0)),
            (0.002, math.radians(0.8)),
            (0.0008, math.radians(0.3)),
            (0.0003, math.radians(0.12)),
        ):
            for _ in range(4):
                cands = []
                for dx in (-step_xy, 0.0, step_xy):
                    for dy in (-step_xy, 0.0, step_xy):
                        for dyaw in (-step_y, 0.0, step_y):
                            if dx != 0.0 or dy != 0.0 or dyaw != 0.0:
                                cands.append([pose[0] + dx, pose[1] + dy, pose[2] + dyaw])
                cands_a = np.array(cands, dtype=float)
                scs = score_batch(cands_a)
                k = int(np.argmax(scs))
                if scs[k] > sc + 1e-6:
                    sc, pose = float(scs[k]), cands_a[k]
                else:
                    break
        if sc > best_sc:
            best_sc, best_pose = sc, pose

    conf = float(np.clip((best_sc - 0.8) / 1.8, 0.0, 1.0))
    return best_pose, conf


def _calibrate_side_camera(
    img: np.ndarray,
    cam_name: str,
    maps: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    robot_mask: np.ndarray | None,
    fixture_xyyaw: np.ndarray | None,
) -> tuple[Camera, dict[str, Any]]:
    nom_pos, nom_mat = side_camera_pose(cam_name)
    H, W = img.shape[:2]
    f = 0.5 * H / math.tan(math.radians(SIDE_CAMERA_FOVY) * 0.5)
    cx, cy = 0.5 * W, 0.5 * H

    soft_fix, soft_tbl, grad_fine, grad_coarse = maps
    tbl_pts, tbl_ins = _table_border_samples()

    fix_pts_w = fix_ins_w = None
    if fixture_xyyaw is not None:
        fx, fy, fyaw = float(fixture_xyyaw[0]), float(fixture_xyyaw[1]), float(fixture_xyyaw[2])
        c, s = math.cos(fyaw), math.sin(fyaw)
        rel = nom_pos - np.array([fx, fy, 0.0])
        cam_loc = np.array([c * rel[0] + s * rel[1], -s * rel[0] + c * rel[1], rel[2]])
        p_loc, i_loc = _fixture_silhouette_samples(cam_loc, n_per_edge=10)
        R_f = _rot_z(fyaw)
        fix_pts_w = p_loc @ R_f.T + np.array([fx, fy, 0.0])
        fix_ins_w = i_loc @ R_f.T

    def eval_cam_batch(dpos_b: np.ndarray, drot_b: np.ndarray, gmap: np.ndarray, off_px: float) -> np.ndarray:
        B = len(dpos_b)
        cpos = nom_pos[None, :] + dpos_b
        R_d = _rodrigues_batch(drot_b)
        cmat = np.einsum("ij,bjk->bik", nom_mat, R_d)
        total = np.zeros(B, dtype=float)

        # Table edges
        u0, v0, d0 = _project_batch_cam(tbl_pts, cpos, cmat, f, cx, cy)
        u1, v1, _ = _project_batch_cam(tbl_pts + 0.015 * tbl_ins, cpos, cmat, f, cx, cy)
        du, dv = u1 - u0, v1 - v0
        nrm = np.hypot(du, dv) + 1e-6
        nu, nv = du / nrm, dv / nrm
        s_in, ok1 = _bilinear_sample(soft_tbl, u0 + off_px * nu, v0 + off_px * nv)
        s_out, ok2 = _bilinear_sample(soft_tbl, u0 - off_px * nu, v0 - off_px * nv)
        ge, ok3 = _bilinear_sample(gmap, u0, v0)
        ok = ok1 & ok2 & ok3 & (d0 > 0.1)
        if robot_mask is not None:
            ui = np.clip(np.round(u0).astype(np.int32), 0, W - 1)
            vi = np.clip(np.round(v0).astype(np.int32), 0, H - 1)
            ok &= ~robot_mask[vi, ui]
        pt_sc = np.where(ok, np.clip(s_in - s_out, -0.4, 1.8) + 0.02 * np.clip(ge, 0.0, 60.0), 0.0)
        total += pt_sc.sum(axis=-1) / np.maximum(ok.sum(axis=-1), 1)

        # Fixture edges
        if fix_pts_w is not None:
            u0, v0, d0 = _project_batch_cam(fix_pts_w, cpos, cmat, f, cx, cy)
            u1, v1, _ = _project_batch_cam(fix_pts_w + 0.008 * fix_ins_w, cpos, cmat, f, cx, cy)
            du, dv = u1 - u0, v1 - v0
            nrm = np.hypot(du, dv) + 1e-6
            nu, nv = du / nrm, dv / nrm
            s_in, ok1 = _bilinear_sample(soft_fix, u0 + off_px * nu, v0 + off_px * nv)
            s_out, ok2 = _bilinear_sample(soft_fix, u0 - off_px * nu, v0 - off_px * nv)
            ge, ok3 = _bilinear_sample(gmap, u0, v0)
            ok = ok1 & ok2 & ok3 & (d0 > 0.1)
            if robot_mask is not None:
                ui = np.clip(np.round(u0).astype(np.int32), 0, W - 1)
                vi = np.clip(np.round(v0).astype(np.int32), 0, H - 1)
                ok &= ~robot_mask[vi, ui]
            pt_sc = np.where(ok, np.clip(s_in - s_out, -0.6, 2.0) + 0.025 * np.clip(ge, 0.0, 60.0), 0.0)
            total += 1.7 * (pt_sc.sum(axis=-1) / np.maximum(ok.sum(axis=-1), 1))

        prior = 0.5 * (np.sum((dpos_b / 0.020) ** 2, axis=-1) + np.sum((drot_b / math.radians(2.2)) ** 2, axis=-1))
        return total - 0.03 * prior

    ang = np.radians(np.linspace(-1.8, 1.8, 5))
    rx, ry, rz = np.meshgrid(ang, ang, ang, indexing="ij")
    drot_grid = np.stack([rx.ravel(), ry.ravel(), rz.ravel()], axis=-1)
    dpos_grid = np.zeros_like(drot_grid)
    scs = eval_cam_batch(dpos_grid, drot_grid, grad_coarse, 3.5)
    top2 = np.argsort(scs)[-2:][::-1]

    best_dp = np.zeros(3)
    best_dr = drot_grid[top2[0]].copy()
    best_sc = -1e9
    for idx in top2:
        dp = np.zeros(3)
        dr = drot_grid[idx].copy()
        sc = float(eval_cam_batch(dp[None], dr[None], grad_fine, 2.5)[0])
        for r_step, p_step in (
            (math.radians(0.55), 0.010),
            (math.radians(0.25), 0.005),
            (math.radians(0.11), 0.0025),
            (math.radians(0.045), 0.001),
        ):
            for _ in range(4):
                dps, drs = [], []
                for ax in range(3):
                    for sgn in (-1.0, 1.0):
                        r2 = dr.copy()
                        r2[ax] += sgn * r_step
                        if abs(r2[ax]) <= math.radians(4.5):
                            dps.append(dp); drs.append(r2)
                        p2 = dp.copy()
                        p2[ax] += sgn * p_step
                        if abs(p2[ax]) <= 0.040:
                            dps.append(p2); drs.append(dr)
                dps_a, drs_a = np.array(dps), np.array(drs)
                c_scs = eval_cam_batch(dps_a, drs_a, grad_fine, 2.5)
                k = int(np.argmax(c_scs))
                if c_scs[k] > sc + 1e-6:
                    sc, dp, dr = float(c_scs[k]), dps_a[k], drs_a[k]
                else:
                    break
        if sc > best_sc:
            best_sc, best_dp, best_dr = sc, dp, dr

    cam = Camera(cam_name, nom_pos + best_dp, nom_mat @ _rodrigues_batch(best_dr[None])[0], SIDE_CAMERA_FOVY, W, H)
    info = {
        "dpos": best_dp.tolist(),
        "drot_deg": np.degrees(best_dr).tolist(),
        "score": float(best_sc),
    }
    return cam, info


def _seats_from_fixture_pose(fx: float, fy: float, fyaw: float) -> tuple[dict[str, list[float]], dict[str, dict[str, Any]]]:
    c, s = math.cos(fyaw), math.sin(fyaw)

    def to_w(lx: float, ly: float) -> tuple[float, float]:
        return (fx + c * lx - s * ly, fy + s * lx + c * ly)

    tx, ty = to_w(0.5 * sum(FIXTURE_TRAY_INNER_X), 0.5 * sum(FIXTURE_TRAY_INNER_Y))
    seats: dict[str, list[float]] = {"tray": [float(tx), float(ty), FIXTURE_TRAY_FLOOR_Z]}
    meta: dict[str, dict[str, Any]] = {
        "tray": {
            "kind": "tray",
            "floor_z": FIXTURE_TRAY_FLOOR_Z,
            "top_z": FIXTURE_TRAY_WALL_Z,
            "half": [
                0.5 * (FIXTURE_TRAY_INNER_X[1] - FIXTURE_TRAY_INNER_X[0]),
                0.5 * (FIXTURE_TRAY_INNER_Y[1] - FIXTURE_TRAY_INNER_Y[0]),
            ],
        }
    }
    for b in FIXTURE_BORES:
        bx, by = to_w(*b["xy"])
        seats[b["id"]] = [float(bx), float(by), float(b["top_z"])]
        meta[b["id"]] = {
            "kind": "hole_vertical",
            "floor_z": float(b["floor_z"]),
            "top_z": float(b["top_z"]),
            "diameter": float(b["diameter"]),
            "intended_part": b["part"],
        }
    for p in FIXTURE_HEX_POCKETS:
        px, py = to_w(*p["xy"])
        seats[p["id"]] = [float(px), float(py), float(p["floor_z"])]
        meta[p["id"]] = {
            "kind": "hex_pocket",
            "floor_z": float(p["floor_z"]),
            "top_z": float(p["top_z"]),
            "across_flats": float(p["across_flats"]),
            "flat_normal_yaw": float(math.pi / 2.0 + fyaw),
            "intended_part": p["part"],
        }
    return seats, meta


# ==============================================================================
# 2D Polygon / Fixture / Workspace Mask Rasterization
# ==============================================================================

def _fill_convex_poly(mask: np.ndarray, pts_uv: np.ndarray) -> None:
    """Fill convex hull of 2D points (N, 2) into boolean mask (H, W) in-place."""
    if len(pts_uv) < 3:
        return
    # Andrew's monotone chain convex hull
    pts = sorted(set((float(p[0]), float(p[1])) for p in pts_uv))
    if len(pts) < 3:
        return

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1], dtype=float)
    if len(hull) < 3:
        return
    H, W = mask.shape
    x0 = max(0, int(math.floor(hull[:, 0].min())))
    x1 = min(W, int(math.ceil(hull[:, 0].max())) + 1)
    y0 = max(0, int(math.floor(hull[:, 1].min())))
    y1 = min(H, int(math.ceil(hull[:, 1].max())) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = np.ones((y1 - y0, x1 - x0), dtype=bool)
    n = len(hull)
    for i in range(n):
        p = hull[i]
        q = hull[(i + 1) % n]
        inside &= ((q[0] - p[0]) * (yy - p[1]) - (q[1] - p[1]) * (xx - p[0])) >= -1e-6
    mask[y0:y1, x0:x1] |= inside


def _project_box_mask(
    cam: Camera,
    corners_w: np.ndarray,
    pad_px: int = 0,
) -> np.ndarray:
    mask = np.zeros((cam.height, cam.width), dtype=bool)
    u, v, d = cam.project(corners_w)
    if np.all(d > 0.05):
        _fill_convex_poly(mask, np.column_stack([u, v]))
    elif np.any(d > 0.05):
        # Clip edges crossing z_cam = -0.05
        valid = np.column_stack([u[d > 0.05], v[d > 0.05]])
        if len(valid) >= 3:
            _fill_convex_poly(mask, valid)
    if pad_px > 0:
        mask = _dilate_mask(mask, pad_px)
    return mask


def _fixture_box_corners(fx: float, fy: float, fyaw: float, x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> np.ndarray:
    R = _rot_z(fyaw)
    t = np.array([fx, fy, 0.0])
    pts = np.array([
        [x, y, z]
        for x in (x0, x1)
        for y in (y0, y1)
        for z in (z0, z1)
    ], dtype=float)
    return pts @ R.T + t


def _fixture_masks(cam: Camera, fx: float, fy: float, fyaw: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (full_fixture_mask, neg_y_face_mask, text_box_mask) in image coordinates."""
    tray_c = _fixture_box_corners(fx, fy, fyaw, FIXTURE_X0, FIXTURE_XM, FIXTURE_Y0, FIXTURE_Y1, 0.0, FIXTURE_TRAY_WALL_Z)
    block_c = _fixture_box_corners(fx, fy, fyaw, FIXTURE_XM, FIXTURE_X1, FIXTURE_Y0, FIXTURE_Y1, 0.0, FIXTURE_BLOCK_TOP_Z)
    full = _project_box_mask(cam, tray_c, pad_px=2) | _project_box_mask(cam, block_c, pad_px=2)

    negy_c = _fixture_box_corners(fx, fy, fyaw, FIXTURE_XM, FIXTURE_X1, FIXTURE_Y0 - 0.002, FIXTURE_Y0 + 0.002, 0.0, FIXTURE_BLOCK_TOP_Z)
    negy = _project_box_mask(cam, negy_c, pad_px=1)

    text_c = _fixture_box_corners(fx, fy, fyaw, FIXTURE_TEXT_X[0], FIXTURE_TEXT_X[1], FIXTURE_Y0 - 0.002, FIXTURE_Y0 + 0.002, FIXTURE_TEXT_Z[0], FIXTURE_TEXT_Z[1])
    text_m = _project_box_mask(cam, text_c, pad_px=2)
    return full, negy, text_m


def _workspace_mask(cam: Camera) -> np.ndarray:
    corners = np.array([
        [x, y, z]
        for x in (0.36, 1.07)
        for y in (-0.47, 0.46)
        for z in (0.0, 0.07)
    ], dtype=float)
    return _project_box_mask(cam, corners, pad_px=2)


# ==============================================================================
# Per-View Adaptive Color Segmentation & Connected Components
# ==============================================================================

@dataclass
class Blob2D:
    view: str
    color_family: str     # "gold" or "light"
    sub_color: str        # "gold", "white", "silver", "grey"
    mask_crop: np.ndarray # bool (bh, bw)
    bbox: tuple[int, int, int, int]  # (x0, y0, x1, y1)
    area: int
    centroid_uv: tuple[float, float]
    mean_rgb: np.ndarray
    max_lum: float
    boundary_grad: float
    has_hole: bool
    hole_ratio: float
    touches_fixture: bool
    on_negy_face: bool


def _connected_components(mask: np.ndarray, min_area: int = 16) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    """Fast pure-NumPy/Python BFS connected components on sparse foreground mask."""
    ys, xs = np.nonzero(mask)
    if len(ys) < min_area:
        return []
    H, W = mask.shape
    visited = np.zeros((H, W), dtype=bool)
    comps: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    # Restrict search to bounding box of all foreground
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if visited[sy, sx]:
            continue
        q = deque([(sy, sx)])
        visited[sy, sx] = True
        pix_y = [sy]
        pix_x = [sx]
        x0 = x1 = sx
        y0 = y1 = sy
        while q:
            cy, cx = q.popleft()
            for dy in (-1, 0, 1):
                ny = cy + dy
                if ny < 0 or ny >= H:
                    continue
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    nx = cx + dx
                    if 0 <= nx < W and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))
                        pix_y.append(ny)
                        pix_x.append(nx)
                        if nx < x0: x0 = nx
                        if nx > x1: x1 = nx
                        if ny < y0: y0 = ny
                        if ny > y1: y1 = ny
        if len(pix_y) >= min_area:
            bh, bw = y1 - y0 + 1, x1 - x0 + 1
            crop = np.zeros((bh, bw), dtype=bool)
            crop[np.asarray(pix_y) - y0, np.asarray(pix_x) - x0] = True
            comps.append(((x0, y0, x1 + 1, y1 + 1), crop))
    return comps


def _fill_holes_2d(crop: np.ndarray) -> np.ndarray:
    """Fill interior background holes in a small binary mask (bh, bw)."""
    bh, bw = crop.shape
    padded = np.pad(crop, 1, mode="constant", constant_values=False)
    ph, pw = padded.shape
    ext = np.zeros((ph, pw), dtype=bool)
    q = deque([(0, 0)])
    ext[0, 0] = True
    while q:
        cy, cx = q.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < ph and 0 <= nx < pw and not padded[ny, nx] and not ext[ny, nx]:
                ext[ny, nx] = True
                q.append((ny, nx))
    return ~ext[1:-1, 1:-1]


def _segment_view(
    img: np.ndarray,
    cam: Camera,
    robot_mask: np.ndarray,
    fixture_xyyaw: np.ndarray,
) -> tuple[list[Blob2D], dict[str, Any]]:
    f = img.astype(np.float32)
    R, G, B = f[..., 0], f[..., 1], f[..., 2]
    lum = 0.299 * R + 0.587 * G + 0.114 * B
    min_c = np.minimum(np.minimum(R, G), B)
    max_c = np.maximum(np.maximum(R, G), B)
    br = B - R
    rb = R - B

    ws_mask = _workspace_mask(cam)
    fix_full, fix_negy, fix_text = _fixture_masks(cam, float(fixture_xyyaw[0]), float(fixture_xyyaw[1]), float(fixture_xyyaw[2]))

    # Estimate table RGB baseline robustly from table pixels outside robot & fixture
    tbl_sample_mask = ws_mask & (~robot_mask) & (~fix_full) & (br > 12.0) & (lum < 115.0)
    if int(tbl_sample_mask.sum()) >= 50:
        tbl_r = float(np.median(R[tbl_sample_mask]))
        tbl_g = float(np.median(G[tbl_sample_mask]))
        tbl_b = float(np.median(B[tbl_sample_mask]))
    else:
        tbl_r, tbl_g, tbl_b = 42.0, 69.0, 86.0
    tbl_lum = 0.299 * tbl_r + 0.587 * tbl_g + 0.114 * tbl_b

    # Gradient map for boundary sharpness (reject blurry table scuffs)
    gx = np.zeros_like(lum)
    gy = np.zeros_like(lum)
    gx[:, 1:-1] = np.abs(f[:, 2:, :] - f[:, :-2, :]).mean(axis=-1)
    gy[1:-1, :] = np.abs(f[2:, :, :] - f[:-2, :, :]).mean(axis=-1)
    grad = np.hypot(gx, gy)

    # Color masks: strict high-confidence seeds + soft non-table grow masks (captures self-shaded sides)
    is_gold_seed = (rb > 30.0) & ((G - B) > 12.0) & (R > 76.0) & (R > 0.90 * G)
    is_gold_soft = (rb > 14.0) & ((G - B) > 4.0) & (R > 48.0) & (R > 0.85 * G)

    b_rel = (B - R) / (max_c + 1.0)
    is_light_seed = (
        (lum > max(100.0, tbl_lum + 36.0))
        & (min_c > max(86.0, tbl_r + 44.0))
        & (b_rel < 0.165)
        & (rb <= 32.0)
    )
    # Self-shaded sides of silver/grey/white parts stay neutral (low b_rel) unlike blue table & table shadows!
    is_light_soft = (
        (lum > max(68.0, tbl_lum + 10.0))
        & (min_c > max(58.0, tbl_r + 18.0))
        & (b_rel < 0.21)
        & (rb <= 28.0)
    )

    valid_table = ws_mask & (~robot_mask) & (~fix_full)
    valid_gold = ws_mask & (~robot_mask)

    negy_plain = fix_negy & (~fix_text) & (~robot_mask) & (~is_gold_seed)
    if int(negy_plain.sum()) >= 20:
        negy_lum = float(np.median(lum[negy_plain]))
    else:
        negy_lum = 95.0

    # Hysteresis grow: keep only soft pixels connected to seeds within 6 px on valid area
    def _hysteresis(seed_m: np.ndarray, soft_m: np.ndarray) -> np.ndarray:
        if not np.any(seed_m):
            return seed_m
        # Dilate seed by 6 px as spatial gate, then flood-fill within soft_m from seed_m
        allowed = soft_m & _dilate_mask(seed_m, 6)
        out = seed_m.copy()
        ys, xs = np.nonzero(seed_m)
        H, W = seed_m.shape
        q = deque(zip(ys.tolist(), xs.tolist()))
        while q:
            cy, cx = q.popleft()
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < H and 0 <= nx < W and allowed[ny, nx] and not out[ny, nx]:
                    out[ny, nx] = True
                    q.append((ny, nx))
        return out

    gold_mask = _hysteresis(is_gold_seed & valid_gold, is_gold_soft & valid_gold)
    light_mask = _hysteresis(is_light_seed & valid_table, is_light_soft & valid_table)

    # Also build a dilated fixture mask to detect parts adjacent to fixture
    fix_near = _dilate_mask(fix_full, 4)

    blobs: list[Blob2D] = []
    for family, raw_mask, seed_ref in (
        ("gold", gold_mask, is_gold_seed & valid_gold),
        ("light", light_mask, is_light_seed & valid_table),
    ):
        comps = _connected_components(raw_mask, min_area=14)
        for (x0, y0, x1, y1), crop in comps:
            area = int(crop.sum())
            seed_area = int((seed_ref[y0:y1, x0:x1] & crop).sum())
            if seed_area < 8:
                continue
            eroded = _erode_mask(crop, 1)
            boundary = crop & (~eroded)
            if not np.any(boundary):
                boundary = crop
            b_grad = float(grad[y0:y1, x0:x1][boundary].mean())

            pad = 5
            rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
            rx1, ry1 = min(cam.width, x1 + pad), min(cam.height, y1 + pad)
            big = np.zeros((ry1 - ry0, rx1 - rx0), dtype=bool)
            big[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0] = crop
            ring = _dilate_mask(big, 4) & (~_dilate_mask(big, 1))
            ring &= (~robot_mask[ry0:ry1, rx0:rx1])
            patch_lum = lum[y0:y1, x0:x1][crop]
            mean_rgb = f[y0:y1, x0:x1][crop].mean(axis=0)
            max_l = float(patch_lum.max())
            p90_l = float(np.percentile(patch_lum, 90))

            if int(ring.sum()) >= 10:
                ring_lum = float(np.median(lum[ry0:ry1, rx0:rx1][ring]))
            else:
                ring_lum = tbl_lum

            if family == "light":
                lum_jump = p90_l - ring_lum
                # Reject worn teal-grey workbench scuffs (blurry grad < 35, greenish G-R > 9, dim p90 < 135)
                if b_grad < 38.0:
                    continue
                if (float(mean_rgb[1]) - float(mean_rgb[0])) > 8.5 and p90_l < 155.0:
                    continue
                if p90_l < max(112.0, tbl_lum + 42.0) and lum_jump < 55.0:
                    continue
            else:
                # Reject warm floor edges leaking into gold soft mask
                if (float(mean_rgb[0]) - float(mean_rgb[2])) < 38.0:
                    continue
                if b_grad < 22.0:
                    continue

            filled = _fill_holes_2d(crop)
            hole_px = filled & (~crop)
            hole_area = int(hole_px.sum())
            filled_area = int(filled.sum())
            hole_ratio = hole_area / max(filled_area, 1)
            has_hole = False
            if hole_area >= 6 and hole_ratio >= 0.04:
                h_lum = float(lum[y0:y1, x0:x1][hole_px].mean())
                if h_lum < float(patch_lum.mean()) - 25.0:
                    has_hole = True

            fy_idx, fx_idx = np.nonzero(filled)
            cu = float(x0 + fx_idx.mean())
            cv = float(y0 + fy_idx.mean())

            if family == "gold":
                sub_color = "gold"
            else:
                if p90_l > 225.0 and (mean_rgb[2] - mean_rgb[0]) < 8.0:
                    sub_color = "white"
                elif p90_l > 205.0:
                    sub_color = "silver"
                else:
                    sub_color = "grey"

            touches_fix = bool(np.any(fix_near[y0:y1, x0:x1] & crop))
            on_negy = bool(np.any(fix_negy[y0:y1, x0:x1] & crop))

            blobs.append(Blob2D(
                view=cam.name,
                color_family=family,
                sub_color=sub_color,
                mask_crop=crop,
                bbox=(x0, y0, x1, y1),
                area=area,
                centroid_uv=(cu, cv),
                mean_rgb=mean_rgb,
                max_lum=max_l,
                boundary_grad=b_grad,
                has_hole=has_hole,
                hole_ratio=float(hole_ratio),
                touches_fixture=touches_fix,
                on_negy_face=on_negy,
            ))

    ctx = {
        "gold_mask": gold_mask,
        "light_mask": light_mask,
        "is_gold": is_gold_soft,
        "is_gold_seed": is_gold_seed,
        "is_light": is_light_soft,
        "is_light_seed": is_light_seed,
        "fix_full": fix_full,
        "fix_negy": fix_negy,
        "fix_text": fix_text,
        "robot_mask": robot_mask,
        "lum": lum,
        "br": br,
        "tbl_rgb": (tbl_r, tbl_g, tbl_b),
        "negy_lum": negy_lum,
        "grad": grad,
    }
    return blobs, ctx


# ==============================================================================
# 3D Part Model Generation & Multi-View Silhouette Fitting
# ==============================================================================

def _build_part_primitives(part: KitPart) -> dict[str, Any]:
    """Exact 3D vertex primitives for zero-hole, zero-bloat convex-hull silhouette rasterization."""
    if part.shape == "bolt":
        # Hex head: 12 vertices (flats facing +-z, corners at k*60 deg in yz plane)
        r_corner = part.across_flats / math.sqrt(3.0)
        hx0, hx1 = part.under_head_x, part.under_head_x + part.head_height
        head = np.array([
            [x, r_corner * math.cos(k * math.pi / 3.0), r_corner * math.sin(k * math.pi / 3.0)]
            for x in (hx0, hx1) for k in range(6)
        ], dtype=float)
        # Shank cylinder: 24 vertices (12-gon at tip and under_head)
        r_sh = 0.5 * part.diameter
        shank = np.array([
            [x, r_sh * math.cos(2.0 * math.pi * k / 12.0), r_sh * math.sin(2.0 * math.pi * k / 12.0)]
            for x in (part.tip_x, part.under_head_x) for k in range(12)
        ], dtype=float)
        return {"shape": "bolt", "solids": (head, shank), "hole_rings": None}

    if part.shape == "nut":
        # Hex prism: 12 vertices (flats facing +-x, corners at 30 + k*60 deg)
        r_corner = part.across_flats / math.sqrt(3.0)
        hz = 0.5 * part.thickness
        outer = np.array([
            [r_corner * math.cos(math.radians(30.0 + 60.0 * k)),
             r_corner * math.sin(math.radians(30.0 + 60.0 * k)), z]
            for z in (-hz, hz) for k in range(6)
        ], dtype=float)
        r_in = 0.5 * part.hole
        ring_lo = np.array([
            [r_in * math.cos(2.0 * math.pi * k / 12.0), r_in * math.sin(2.0 * math.pi * k / 12.0), -hz]
            for k in range(12)
        ], dtype=float)
        ring_hi = np.array([
            [r_in * math.cos(2.0 * math.pi * k / 12.0), r_in * math.sin(2.0 * math.pi * k / 12.0), hz]
            for k in range(12)
        ], dtype=float)
        return {"shape": "nut", "solids": (outer,), "hole_rings": (ring_lo, ring_hi)}

    # Washer: outer disc (24 verts) minus through-hole (top & bottom inner rings)
    r_out = 0.5 * part.outer_diameter
    r_in = 0.5 * part.hole
    hz = 0.5 * part.thickness
    outer = np.array([
        [r_out * math.cos(2.0 * math.pi * k / 16.0), r_out * math.sin(2.0 * math.pi * k / 16.0), z]
        for z in (-hz, hz) for k in range(16)
    ], dtype=float)
    ring_lo = np.array([
        [r_in * math.cos(2.0 * math.pi * k / 12.0), r_in * math.sin(2.0 * math.pi * k / 12.0), -hz]
        for k in range(12)
    ], dtype=float)
    ring_hi = np.array([
        [r_in * math.cos(2.0 * math.pi * k / 12.0), r_in * math.sin(2.0 * math.pi * k / 12.0), hz]
        for k in range(12)
    ], dtype=float)
    return {"shape": "washer", "solids": (outer,), "hole_rings": (ring_lo, ring_hi)}


_PART_PRIMS_CACHE: dict[str, dict[str, Any]] = {
    pid: _build_part_primitives(p) for pid, p in KIT_PARTS.items()
}


def _fill_convex_hull_local(mask: np.ndarray, pts_uv: np.ndarray, x0: int, y0: int) -> None:
    """Fill 2D convex hull of pts_uv (in full-image coords) into local crop mask starting at (x0, y0)."""
    if len(pts_uv) < 3:
        return
    # Fast rounded-float deduplication and monotone chain convex hull
    pts = sorted(set((float(p[0]) - x0, float(p[1]) - y0) for p in pts_uv))
    if len(pts) < 3:
        return

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2:
            o, a = lower[-2], lower[-1]
            if (a[0] - o[0]) * (p[1] - o[1]) - (a[1] - o[1]) * (p[0] - o[0]) <= 0.0:
                lower.pop()
            else:
                break
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2:
            o, a = upper[-2], upper[-1]
            if (a[0] - o[0]) * (p[1] - o[1]) - (a[1] - o[1]) * (p[0] - o[0]) <= 0.0:
                upper.pop()
            else:
                break
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float32)
    if len(hull) < 3:
        return
    bh, bw = mask.shape
    cx0 = max(0, int(math.floor(float(hull[:, 0].min()))))
    cx1 = min(bw, int(math.ceil(float(hull[:, 0].max()))) + 1)
    cy0 = max(0, int(math.floor(float(hull[:, 1].min()))))
    cy1 = min(bh, int(math.ceil(float(hull[:, 1].max()))) + 1)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    # Vectorized halfspace test over all hull edges at once (no mgrid allocation)
    q = np.roll(hull, -1, axis=0)
    dx = (q[:, 0] - hull[:, 0])[:, None, None]
    dy = (q[:, 1] - hull[:, 1])[:, None, None]
    px = hull[:, 0][:, None, None]
    py = hull[:, 1][:, None, None]
    yy = np.arange(cy0, cy1, dtype=np.float32)[None, :, None]
    xx = np.arange(cx0, cx1, dtype=np.float32)[None, None, :]
    inside = np.all(dx * (yy - py) - dy * (xx - px) >= -1e-4, axis=0)
    mask[cy0:cy1, cx0:cx1] |= inside


def _rasterize_part_in_crop(
    part: KitPart,
    pos: np.ndarray,
    R: np.ndarray,
    cam: Camera,
    obs_bbox: tuple[int, int, int, int] | None,
    pad: int = 3,
) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray] | None:
    """Exact 2D silhouette rasterization (solid_mask, hole_mask) in a tight crop.

    Returns ((x0, y0, x1, y1), solid_mask, hole_mask) or None if behind camera / out of frame.
    """
    prims = _PART_PRIMS_CACHE[part.part_id]
    proj_solids = []
    umin, vmin = 1e9, 1e9
    umax, vmax = -1e9, -1e9
    for verts in prims["solids"]:
        vw = verts @ R.T + pos[None, :]
        u, v, d = cam.project(vw)
        if np.any(d <= 0.08):
            return None
        umin = min(umin, float(u.min()))
        umax = max(umax, float(u.max()))
        vmin = min(vmin, float(v.min()))
        vmax = max(vmax, float(v.max()))
        proj_solids.append(np.column_stack([u, v]))

    if umax < 0 or umin >= cam.width or vmax < 0 or vmin >= cam.height:
        return None
    if obs_bbox is not None:
        x0 = max(0, min(obs_bbox[0], int(math.floor(umin))) - pad)
        y0 = max(0, min(obs_bbox[1], int(math.floor(vmin))) - pad)
        x1 = min(cam.width, max(obs_bbox[2], int(math.ceil(umax)) + 1) + pad)
        y1 = min(cam.height, max(obs_bbox[3], int(math.ceil(vmax)) + 1) + pad)
    else:
        x0 = max(0, int(math.floor(umin)) - pad)
        y0 = max(0, int(math.floor(vmin)) - pad)
        x1 = min(cam.width, int(math.ceil(umax)) + 1 + pad)
        y1 = min(cam.height, int(math.ceil(vmax)) + 1 + pad)
    if x1 <= x0 or y1 <= y0 or (x1 - x0) > 190 or (y1 - y0) > 190:
        return None

    solid = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    for uv in proj_solids:
        _fill_convex_hull_local(solid, uv, x0, y0)

    hole = np.zeros_like(solid)
    if prims["hole_rings"] is not None:
        # Exact ray-traced cylindrical through-hole: pixels inside BOTH top & bottom bore rings!
        r_lo, r_hi = prims["hole_rings"]
        u1, v1, d1 = cam.project(r_lo @ R.T + pos[None, :])
        u2, v2, d2 = cam.project(r_hi @ R.T + pos[None, :])
        if np.all(d1 > 0.08) and np.all(d2 > 0.08):
            m1 = np.zeros_like(solid)
            m2 = np.zeros_like(solid)
            _fill_convex_hull_local(m1, np.column_stack([u1, v1]), x0, y0)
            _fill_convex_hull_local(m2, np.column_stack([u2, v2]), x0, y0)
            hole = m1 & m2
            solid = solid & (~hole)
    return (x0, y0, x1, y1), solid, hole


def _part_pose_world(
    part: KitPart,
    pose_type: str,
    x: float,
    y: float,
    yaw: float,
    fixture_xyyaw: np.ndarray | None = None,
    lean_u: float = 0.05,
    lean_theta: float = 0.65,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (pos_w (3,), R_w (3,3), axis_w (3,)) for a kit part in canonical resting pose."""
    if pose_type == "leaning" and part.shape == "bolt" and fixture_xyyaw is not None:
        fx, fy, fyaw = float(fixture_xyyaw[0]), float(fixture_xyyaw[1]), float(fixture_xyyaw[2])
        n = np.array([-math.sin(fyaw), math.cos(fyaw), 0.0])  # into -y face
        up = np.array([0.0, 0.0, 1.0])
        c, s = math.cos(lean_theta), math.sin(lean_theta)
        ax_x = -c * up - s * n  # body +x toward head (down & away from face)
        ax_z = (n + s * ax_x) / max(c, 1e-4)
        ax_z /= np.linalg.norm(ax_z)
        ax_y = np.cross(ax_z, ax_x)
        R = np.column_stack([ax_x, ax_y, ax_z])
        full = part.full_length
        cf, sf = math.cos(fyaw), math.sin(fyaw)
        face = np.array([fx + cf * lean_u - sf * FIXTURE_Y0, fy + sf * lean_u + cf * FIXTURE_Y0, 0.0])
        out = s * full - c * 0.5 * (part.across_flats - part.diameter) + 0.0005
        edge = face - out * n + 0.0005 * up
        pos = edge - 0.5 * full * ax_x - 0.5 * part.across_flats * ax_z
        return pos, R, ax_x

    if part.shape == "bolt":
        if pose_type == "standing":
            pos = np.array([x, y, part.standing_z], dtype=float)
            # Head down: body +x -> world -z
            R_hd = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=float)
            R = _rot_z(yaw) @ R_hd
            return pos, R, R[:, 0]
        # Lying on side: slight natural tilt (head higher than tip)
        tilt = part.lying_tilt
        ct, st = math.cos(-tilt), math.sin(-tilt)
        R_tilt = np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]], dtype=float)
        R = _rot_z(yaw) @ R_tilt
        pos = np.array([x, y, part.lying_z], dtype=float)
        return pos, R, R[:, 0]

    if part.shape == "nut" and pose_type == "on_edge":
        pos = np.array([x, y, 0.5 * part.across_flats], dtype=float)
        # Nut axis (body +z) horizontal along yaw
        ax_z = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=float)
        ax_y = np.array([0.0, 0.0, -1.0], dtype=float)
        ax_x = np.cross(ax_y, ax_z)
        R = np.column_stack([ax_x, ax_y, ax_z])
        return pos, R, ax_z

    # Lying flat nut or washer
    pos = np.array([x, y, part.lying_z], dtype=float)
    R = _rot_z(yaw)
    return pos, R, np.array([0.0, 0.0, 1.0], dtype=float)


def _score_part_hypothesis(
    part: KitPart,
    pose_type: str,
    pos: np.ndarray,
    R: np.ndarray,
    view_obs: list[tuple[Camera, np.ndarray, np.ndarray | None, tuple[int, int, int, int], np.ndarray | None]],
    color_family: str,
) -> float:
    """Multi-view exact silhouette IoU + hole consistency score for a 3D part hypothesis."""
    scores = []
    for cam, fg_mask, robot_mask, bbox, _extra in view_obs:
        rast = _rasterize_part_in_crop(part, pos, R, cam, bbox, pad=2)
        if rast is None:
            return -1.0
        (x0, y0, x1, y1), pred, hole = rast
        obs_crop = fg_mask[y0:y1, x0:x1]

        if robot_mask is not None:
            rm = robot_mask[y0:y1, x0:x1]
            pred_e = pred & (~rm)
            obs_e = obs_crop & (~rm)
        else:
            pred_e = pred
            obs_e = obs_crop

        inter = int((pred_e & obs_e).sum())
        union = int((pred_e | obs_e).sum())
        iou = inter / max(union, 1)

        # Through-hole bonus: if exact 3D geometry predicts an open through-hole, reward dark/bg pixels in it!
        hole_bonus = 0.0
        h_cnt = int(hole.sum())
        if h_cnt >= 4:
            frac_open = float((~obs_e[hole]).mean())
            hole_bonus = 0.08 * (frac_open - 0.35)
        scores.append(iou + hole_bonus)
    if not scores:
        return -1.0
    return float(np.mean(scores))


def _fit_candidate_3d(
    init_xyz: np.ndarray,
    color_family: str,
    sub_color: str,
    blobs_by_view: dict[str, Blob2D],
    cams: dict[str, Camera],
    view_ctxs: dict[str, dict[str, Any]],
    fixture_xyyaw: np.ndarray,
    allow_leaning: bool = False,
) -> dict[str, Any] | None:
    """Fit best (kit_part, pose_type, pos, axis, yaw, iou_score) across visible views."""
    view_obs = []
    for vname, blob in blobs_by_view.items():
        cam = cams[vname]
        ctx = view_ctxs[vname]
        x0, y0, x1, y1 = blob.bbox
        fg_iso = np.zeros((cam.height, cam.width), dtype=bool)
        fg_iso[y0:y1, x0:x1] = blob.mask_crop
        view_obs.append((cam, fg_iso, ctx["robot_mask"], blob.bbox, None))

    best_vname = "right" if "right" in blobs_by_view else next(iter(blobs_by_view))
    b0 = blobs_by_view[best_vname]
    cam0 = cams[best_vname]
    ys_b, xs_b = np.nonzero(b0.mask_crop)
    bx0, by0 = b0.bbox[0], b0.bbox[1]
    pts_table = []
    step_s = max(1, len(ys_b) // 100)
    z_init = float(np.clip(init_xyz[2], 0.003, 0.045))
    for py, px in zip(ys_b[::step_s], xs_b[::step_s]):
        hit = cam0.intersect_z(float(bx0 + px), float(by0 + py), z_init)
        if hit is not None:
            pts_table.append(hit[:2])
    init_yaw = 0.0
    if len(pts_table) >= 6:
        P2 = np.array(pts_table, float)
        Pc = P2 - P2.mean(axis=0, keepdims=True)
        cov = (Pc.T @ Pc) / len(Pc)
        _vals, vecs = np.linalg.eigh(cov)
        major = vecs[:, 1]
        init_yaw = math.atan2(float(major[1]), float(major[0]))

    any_hole = any(b.has_hole for b in blobs_by_view.values())

    hypotheses: list[tuple[str, str]] = []
    if color_family == "gold":
        hypotheses.extend([
            ("bolt_m30x100", "lying"),
            ("bolt_m30x100", "standing"),
            ("bolt_m24x70", "lying"),
            ("bolt_m16x50", "lying"),
            ("washer_m33", "lying"),
            ("nut_m24", "lying"),
            ("nut_m30", "lying"),
            ("nut_m36", "lying"),
            ("nut_m30", "on_edge"),
            ("nut_m24", "on_edge"),
            ("nut_m36", "on_edge"),
        ])
        if allow_leaning:
            hypotheses.extend([("bolt_m30x100", "leaning"), ("bolt_m24x70", "leaning"), ("bolt_m16x50", "leaning")])
    else:
        hypotheses.extend([
            ("washer_m33", "lying"),
            ("bolt_m16x50", "lying"),
            ("bolt_m24x70", "lying"),
            ("bolt_m30x100", "lying"),
            ("nut_m24", "lying"),
            ("nut_m30", "lying"),
            ("nut_m36", "lying"),
            ("nut_m30", "on_edge"),
            ("nut_m24", "on_edge"),
            ("bolt_m16x50", "standing"),
            ("bolt_m30x100", "standing"),
        ])
        if allow_leaning:
            hypotheses.extend([("bolt_m16x50", "leaning"), ("bolt_m30x100", "leaning"), ("bolt_m24x70", "leaning")])

    has_stereo = len(blobs_by_view) >= 2

    def _prior_adj(part: KitPart, ptype: str) -> float:
        if ptype == "lying":
            z_exp = part.lying_z
        elif ptype == "standing":
            z_exp = part.standing_z
        else:
            z_exp = 0.5 * part.across_flats
        # Mild stereo Z penalty (perspective visible centroid can be ~4-7mm off body origin for tall parts)
        z_err = max(0.0, abs(z_exp - float(init_xyz[2])) - 0.005) if has_stereo else 0.0
        adj = -4.0 * z_err
        if part.shape == "washer":
            if has_stereo and float(init_xyz[2]) > 0.011:
                adj -= 0.16
            if not any_hole:
                adj -= 0.14
        elif part.shape == "bolt" and ptype == "lying" and any_hole:
            adj -= 0.12
        return adj

    # Pass 1: fast coarse screen of all non-leaning hypotheses
    coarse_ranked: list[tuple[float, str, str, float, float, float]] = []
    leaning_hyps: list[tuple[str, str]] = []
    for pid, ptype in hypotheses:
        if ptype == "leaning":
            leaning_hyps.append((pid, ptype))
            continue
        part = KIT_PARTS[pid]
        z_exp = part.lying_z if ptype == "lying" else (part.standing_z if ptype == "standing" else 0.5 * part.across_flats)
        if has_stereo and abs(z_exp - float(init_xyz[2])) > 0.028:
            continue
        if not has_stereo:
            hit = cam0.intersect_z(b0.centroid_uv[0], b0.centroid_uv[1], z_exp)
            if hit is None:
                continue
            x_cur, y_cur = float(hit[0]), float(hit[1])
        else:
            x_cur, y_cur = float(init_xyz[0]), float(init_xyz[1])

        if part.shape == "bolt" and ptype == "lying":
            yaw_inits = [init_yaw + k * (math.pi / 4.0) for k in range(8)]
        elif part.shape == "nut" and ptype == "on_edge":
            yaw_inits = [i * math.pi / 4.0 for i in range(4)]
        else:
            yaw_inits = [0.0]

        best_c = -1e9
        best_y = yaw_inits[0]
        for ya in yaw_inits:
            pos, R, _ = _part_pose_world(part, ptype, x_cur, y_cur, ya)
            sc = _score_part_hypothesis(part, ptype, pos, R, view_obs, color_family)
            if sc > best_c:
                best_c, best_y = sc, ya
        coarse_ranked.append((best_c + _prior_adj(part, ptype), pid, ptype, x_cur, y_cur, best_y))

    coarse_ranked.sort(key=lambda t: t[0], reverse=True)
    # Refine top 4 coarse candidates so slight initial centroid offsets don't bias size selection
    best_res: dict[str, Any] | None = None
    best_total = -1e9

    for _c_sc, pid, ptype, x_cur, y_cur, y_init in coarse_ranked[:4]:
        part = KIT_PARTS[pid]
        rx, ry, ryaw = x_cur, y_cur, y_init
        pos0, R0, _ = _part_pose_world(part, ptype, rx, ry, ryaw)
        best_h_sc = _score_part_hypothesis(part, ptype, pos0, R0, view_obs, color_family)

        # For lying bolts, test BOTH head directions and axial shifts along bolt axis
        if part.shape == "bolt" and ptype == "lying":
            for y_try in (ryaw, ryaw + math.pi):
                d_ax = np.array([math.cos(y_try), math.sin(y_try)])
                for sh in (-0.008, -0.004, 0.0, 0.004, 0.008):
                    tx, ty = x_cur + sh * d_ax[0], y_cur + sh * d_ax[1]
                    pos, R, _ = _part_pose_world(part, ptype, tx, ty, y_try)
                    sc = _score_part_hypothesis(part, ptype, pos, R, view_obs, color_family)
                    if sc > best_h_sc:
                        best_h_sc, rx, ry, ryaw = sc, tx, ty, y_try
        elif part.shape == "nut" and ptype == "on_edge":
            for ya in (i * math.pi / 6.0 for i in range(6)):
                pos, R, _ = _part_pose_world(part, ptype, x_cur, y_cur, ya)
                sc = _score_part_hypothesis(part, ptype, pos, R, view_obs, color_family)
                if sc > best_h_sc:
                    best_h_sc, ryaw = sc, ya

        do_yaw = (part.shape == "bolt" and ptype == "lying") or (part.shape == "nut" and ptype in ("on_edge", "lying"))
        for sxy, syaw in ((0.005, math.radians(12.0)), (0.002, math.radians(5.0)), (0.0008, math.radians(2.0))):
            for _ in range(3):
                improved = False
                moves = [(-sxy, 0.0, 0.0), (sxy, 0.0, 0.0), (0.0, -sxy, 0.0), (0.0, sxy, 0.0)]
                if do_yaw:
                    moves.extend([(0.0, 0.0, -syaw), (0.0, 0.0, syaw)])
                for dx, dy, dyaw in moves:
                    pos, R, _ = _part_pose_world(part, ptype, rx + dx, ry + dy, ryaw + dyaw)
                    sc = _score_part_hypothesis(part, ptype, pos, R, view_obs, color_family)
                    if sc > best_h_sc + 1e-5:
                        best_h_sc = sc
                        rx, ry, ryaw = rx + dx, ry + dy, ryaw + dyaw
                        improved = True
                if not improved:
                    break

        tot = best_h_sc + _prior_adj(part, ptype)
        if tot > best_total:
            pos, R, axis = _part_pose_world(part, ptype, rx, ry, ryaw)
            yw = (ryaw + math.pi) % (2.0 * math.pi) - math.pi
            best_total = tot
            best_res = {
                "kind": pid, "shape": part.shape, "pose": ptype,
                "position": pos, "axis": axis, "yaw": float(yw),
                "iou": float(best_h_sc), "total_score": float(tot),
            }

    # Leaning hypotheses (if candidate sits against fixture -y face)
    if leaning_hyps and "right" in cams and "right" in view_ctxs:
        fx, fy, fyaw = float(fixture_xyyaw[0]), float(fixture_xyyaw[1]), float(fixture_xyyaw[2])
        c, s = math.cos(fyaw), math.sin(fyaw)
        dx0, dy0 = float(init_xyz[0]) - fx, float(init_xyz[1]) - fy
        u_init = float(np.clip(c * dx0 + s * dy0, FIXTURE_XM + 0.012, FIXTURE_X1 - 0.012))
        cam_r = cams["right"]
        ctx_r = view_ctxs["right"]
        for pid, _ptype in leaning_hyps:
            part = KIT_PARTS[pid]
            pos, R, axis, sc_l = _fit_leaning_bolt_right(part, u_init, fixture_xyyaw, cam_r, ctx_r, color_family)
            tot = sc_l + 0.05
            if tot > best_total:
                best_total = tot
                best_res = {
                    "kind": pid, "shape": part.shape, "pose": "leaning",
                    "position": pos, "axis": axis,
                    "yaw": float(math.atan2(float(axis[1]), float(axis[0]))),
                    "iou": float(sc_l), "total_score": float(tot),
                }

    return best_res


def _score_leaning_bolt_right(
    part: KitPart,
    pos: np.ndarray,
    R: np.ndarray,
    cam_r: Camera,
    ctx_r: dict[str, Any],
    color_family: str,
) -> float:
    """Text-immune exact-silhouette scorer for a bolt leaning against the fixture -y face in `right`."""
    rast = _rasterize_part_in_crop(part, pos, R, cam_r, None, pad=6)
    if rast is None:
        return -1.0
    (x0, y0, x1, y1), pred, _hole = rast

    rm = ctx_r["robot_mask"][y0:y1, x0:x1]
    pred = pred & (~rm)
    if int(pred.sum()) < 16:
        return -1.0

    fix_full = ctx_r["fix_full"][y0:y1, x0:x1]
    fix_text = ctx_r["fix_text"][y0:y1, x0:x1]
    lum = ctx_r["lum"][y0:y1, x0:x1]
    negy_lum = float(ctx_r["negy_lum"])

    if color_family == "gold":
        fg = ctx_r["is_gold"][y0:y1, x0:x1] & (~rm)
        # No text ambiguity for gold
        inter = int((pred & fg).sum())
        union = int((pred | fg).sum())
        return inter / max(union, 1)

    # Light (grey/silver) leaning bolt:
    # Foreground on table (outside fixture) OR bright on -y face
    fg_table = ctx_r["is_light"][y0:y1, x0:x1] & (~fix_full) & (~rm)
    fg_face = fix_full & (lum > max(112.0, negy_lum + 22.0)) & (~rm)
    fg_all = fg_table | fg_face

    # Ignore REALITY CHECK text pixels that lie OUTSIDE our predicted bolt silhouette!
    eval_mask = (~rm) & (~(fix_text & (~pred)))
    pred_e = pred & eval_mask
    fg_e = fg_all & eval_mask

    # Recall of bolt silhouette + precision against non-text background
    cov = int((pred & fg_all).sum()) / max(int(pred.sum()), 1)
    # Require BOTH table portion (head on table) AND upper face portion (shank leaning up) to match!
    pred_tbl = pred & (~fix_full)
    pred_fac = pred & fix_full
    cov_tbl = int((pred_tbl & fg_table).sum()) / max(int(pred_tbl.sum()), 1) if int(pred_tbl.sum()) >= 5 else cov
    cov_fac = int((pred_fac & fg_face).sum()) / max(int(pred_fac.sum()), 1) if int(pred_fac.sum()) >= 5 else 0.0

    # Local ring immediately around the predicted bolt (excluding text) should be background (measures tightness/size fit!)
    ring = _dilate_mask(pred, 4) & (~pred) & eval_mask
    bg_clean = 1.0 - (int((ring & fg_e).sum()) / max(int(ring.sum()), 1))
    # Table-only IoU ensures bolt head fits the observed table blob tightly (distinguishes M16 vs M24 vs nut!)
    tbl_inter = int((pred_tbl & fg_table).sum())
    tbl_union = int((pred_tbl | fg_table).sum())
    tbl_iou = tbl_inter / max(tbl_union, 1)

    if cov_fac < 0.35 or cov_tbl < 0.35:
        return 0.25 * (0.5 * cov + 0.5 * tbl_iou)
    return 0.42 * cov + 0.28 * tbl_iou + 0.20 * bg_clean + 0.10 * min(cov_tbl, cov_fac)


def _fit_leaning_bolt_right(
    part: KitPart,
    u_init: float,
    fixture_xyyaw: np.ndarray,
    cam_r: Camera,
    ctx_r: dict[str, Any],
    color_family: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    th_base = math.radians(part.tip_over_deg + 5.5)
    best_u, best_th, best_sc = u_init, th_base, -1e9
    for du in np.linspace(-0.024, 0.024, 11):
        uc = float(np.clip(u_init + du, FIXTURE_XM + 0.012, FIXTURE_X1 - 0.012))
        for dth in np.radians([-3.0, 0.0, 3.0, 6.0]):
            thc = th_base + dth
            pos, R, _ = _part_pose_world(part, "leaning", 0.0, 0.0, 0.0, fixture_xyyaw, uc, thc)
            sc = _score_leaning_bolt_right(part, pos, R, cam_r, ctx_r, color_family)
            if sc > best_sc:
                best_sc, best_u, best_th = sc, uc, thc
    for su, sth in ((0.004, math.radians(2.0)), (0.0015, math.radians(0.8))):
        for _ in range(3):
            imp = False
            for du, dth in ((-su, 0.0), (su, 0.0), (0.0, -sth), (0.0, sth)):
                uc = float(np.clip(best_u + du, FIXTURE_XM + 0.01, FIXTURE_X1 - 0.01))
                thc = float(np.clip(best_th + dth, math.radians(part.tip_over_deg + 1.0), math.radians(part.tip_over_deg + 13.0)))
                pos, R, _ = _part_pose_world(part, "leaning", 0.0, 0.0, 0.0, fixture_xyyaw, uc, thc)
                sc = _score_leaning_bolt_right(part, pos, R, cam_r, ctx_r, color_family)
                if sc > best_sc + 1e-5:
                    best_sc, best_u, best_th, imp = sc, uc, thc, True
            if not imp:
                break
    pos, R, axis = _part_pose_world(part, "leaning", 0.0, 0.0, 0.0, fixture_xyyaw, best_u, best_th)
    return pos, R, axis, float(best_sc)


# ==============================================================================
# Special Leaning-Bolt Check on Fixture -y Face
# ==============================================================================

def _detect_leaning_bolt_negy(
    cams: dict[str, Camera],
    view_ctxs: dict[str, dict[str, Any]],
    fixture_xyyaw: np.ndarray,
    existing_parts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Specifically check if a bolt leans against the fixture -y face (visible in `right`)."""
    if "right" not in cams or "right" not in view_ctxs:
        return []
    cam_r = cams["right"]
    ctx_r = view_ctxs["right"]
    fx, fy, fyaw = float(fixture_xyyaw[0]), float(fixture_xyyaw[1]), float(fixture_xyyaw[2])
    c, s = math.cos(fyaw), math.sin(fyaw)

    # Check if an existing detected part is ALREADY a leaning bolt
    if any(p["pose"] == "leaning" for p in existing_parts):
        return []

    # Build table-strip mask immediately in front of the fixture -y face:
    # local x in [-0.025, 0.120], local y in [-0.185, -0.115], z in [0.0, 0.025]
    strip_c = _fixture_box_corners(fx, fy, fyaw, FIXTURE_XM, FIXTURE_X1, FIXTURE_Y0 - 0.065, FIXTURE_Y0, 0.0, 0.024)
    strip_m = _project_box_mask(cam_r, strip_c, pad_px=2) & (~ctx_r["robot_mask"]) & (~ctx_r["fix_full"])

    found: list[dict[str, Any]] = []
    for family, part_cands in (
        ("gold", ("bolt_m30x100", "bolt_m16x50", "bolt_m24x70")),
        ("light", ("bolt_m16x50", "bolt_m30x100", "bolt_m24x70")),
    ):
        seed_m = strip_m & (ctx_r["is_gold_seed"] if family == "gold" else ctx_r["is_light_seed"])
        if int(seed_m.sum()) < 8:
            continue
        ys, xs = np.nonzero(seed_m)
        cu, cv = float(xs.mean()), float(ys.mean())
        already = False
        for p in existing_parts:
            pu, pv, _ = cam_r.project(np.asarray(p["position"], float)[None, :])
            if math.hypot(float(pu[0]) - cu, float(pv[0]) - cv) < 22.0 and p["color"] in (("gold",) if family == "gold" else ("grey", "silver", "white")):
                if len(p.get("views", [])) >= 2 and float(p["position"][2]) < 0.022:
                    already = True
                    break
        if already:
            continue

        hit = cam_r.intersect_z(cu, cv, 0.015)
        if hit is None:
            continue
        dx0, dy0 = float(hit[0]) - fx, float(hit[1]) - fy
        u_init = float(np.clip(c * dx0 + s * dy0, FIXTURE_XM + 0.012, FIXTURE_X1 - 0.012))

        best_sc = -1e9
        best_item = None
        for pid in part_cands:
            part = KIT_PARTS[pid]
            pos, _R, axis, bsc = _fit_leaning_bolt_right(part, u_init, fixture_xyyaw, cam_r, ctx_r, family)
            if bsc > best_sc:
                best_sc = bsc
                best_item = {
                    "kind": pid,
                    "shape": "bolt",
                    "color": "gold" if family == "gold" else "grey",
                    "pose": "leaning",
                    "position": pos.tolist(),
                    "axis": axis.tolist(),
                    "yaw": float(math.atan2(float(axis[1]), float(axis[0]))),
                    "confidence": float(np.clip(0.55 + 0.45 * bsc, 0.25, 0.92)),
                    "occluded": True,
                    "views": ["right"],
                }
        if best_item is not None and best_sc >= 0.48:
            found.append(best_item)
    return found


# ==============================================================================
# Target Matching (Instruction + Geometry/Color/Shape + Goal Presence Check)
# ==============================================================================

def _check_part_present_in_goal(
    part: dict[str, Any],
    goal_img: np.ndarray,
    goal_cam: Camera,
) -> bool | None:
    """Project a detected part's 3D position into the unjittered goal image.

    Returns True if a part of the same color family is STILL sitting there in `goal`,
    False if that spot is empty blue table in `goal` (so this part must move!),
    or None if out of view / ambiguous.
    """
    pos = np.asarray(part["position"], float)
    u, v, d = goal_cam.project(pos[None, :])
    ui, vi, di = int(round(float(u[0]))), int(round(float(v[0]))), float(d[0])
    H, W = goal_img.shape[:2]
    r = 12
    if di <= 0.1 or ui < r or ui >= W - r or vi < r or vi >= H - r:
        return None
    patch = goal_img[vi - r:vi + r + 1, ui - r:ui + r + 1].astype(np.float32)
    R, G, B = patch[..., 0], patch[..., 1], patch[..., 2]
    lum = 0.299 * R + 0.587 * G + 0.114 * B
    min_c = np.minimum(np.minimum(R, G), B)
    max_c = np.maximum(np.maximum(R, G), B)
    if part.get("color") == "gold":
        fg = ((R - B) > 30.0) & ((G - B) > 12.0) & (R > 76.0)
    else:
        fg = (lum > 110.0) & (min_c > 95.0) & (((B - R) / (max_c + 1.0)) < 0.16) & ((R - B) <= 32.0)
    frac = float(fg.mean())
    if frac > 0.12:
        return True
    if frac < 0.03:
        return False
    return None


def _assign_roles(
    parts: list[dict[str, Any]],
    seats: dict[str, list[float]],
    instruction: str,
    goal_img: np.ndarray | None,
    goal_cam: Camera | None,
) -> None:
    """Assign `role` ('target', 'decoy', 'clutter') and optional `target_seat` to parts."""
    from .scene_interface import extract_part_material_and_color, normalize_material_name

    text = (instruction or "").lower()
    wants_tray = "tray" in text
    wants_hole = "hole" in text
    wants_pocket = "pocket" in text
    wants_washer = "washer" in text
    wants_bolt = "bolt" in text
    wants_nut = "nut" in text
    wants_smaller_bolt = (("smaller" in text) or ("smallest" in text)) and wants_bolt
    wants_larger_bolt = (("larger" in text) or ("bigger" in text) or ("largest" in text)) and wants_bolt
    wants_near_tray = ("nearer the tray" in text) or ("nearest the tray" in text) or ("closer to the tray" in text)
    wants_near_robot = ("nearer the robot" in text) or ("nearest the robot" in text) or ("closer to the robot" in text)

    # Goal-image presence cue (absent in goal => definitely needs to be moved!)
    goal_moved: list[bool | None] = []
    for p in parts:
        if goal_img is not None and goal_cam is not None:
            pres = _check_part_present_in_goal(p, goal_img, goal_cam)
            goal_moved.append(None if pres is None else (not pres))
        else:
            goal_moved.append(None)

    for p in parts:
        p["role"] = "clutter"

    tray_xy = np.asarray(seats.get("tray", [0.5865, 0.15, 0.004])[:2], float)

    def _pick_best(
        cands: list[int],
        prefer_small: bool = False,
        prefer_large: bool = False,
        prefer_near_tray: bool = False,
        prefer_near_robot: bool = False,
        prefer_color: str | None = None,
        prefer_material: str | None = None,
    ) -> int | None:
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]

        def key_fn(i: int) -> tuple:
            p = parts[i]
            kp = KIT_PARTS.get(p["kind"])
            size_val = (
                kp.diameter
                if kp and kp.shape == "bolt"
                else (kp.outer_diameter if kp and kp.shape == "washer" else (kp.across_flats if kp else 0.03))
            )
            xy = np.asarray(p["position"][:2], float)
            d_tray = float(np.linalg.norm(xy - tray_xy))
            d_rob = float(np.linalg.norm(xy))
            moved = goal_moved[i]
            moved_score = 1 if moved is True else (-1 if moved is False else 0)
            p_mat = normalize_material_name(p.get("material")) or normalize_material_name(p.get("color"))
            mat_scores = p.get("material_scores") if isinstance(p.get("material_scores"), dict) else {}
            mat_ok = 0.0
            if prefer_material:
                if p_mat == prefer_material:
                    mat_ok += 1.0
                mat_ok += float(mat_scores.get(prefer_material, 0.0))
            elif prefer_color and p.get("color") == prefer_color:
                mat_ok += 1.0
            # Primary: explicit instruction attribute, reinforced by goal_moved
            attr = 0.0
            if prefer_small:
                attr = -size_val
            elif prefer_large:
                attr = size_val
            elif prefer_near_tray:
                attr = -d_tray
            elif prefer_near_robot:
                attr = -d_rob
            return (round(mat_ok, 3), moved_score, attr, float(p["confidence"]))

        return max(cands, key=key_fn)

    bolt_mat, bolt_color = extract_part_material_and_color(text, "bolt")
    nut_mat, nut_color = extract_part_material_and_color(text, "nut")
    wash_mat, wash_color = extract_part_material_and_color(text, "washer")

    # Match washer target
    if wants_washer:
        w_idxs = [i for i, p in enumerate(parts) if p["shape"] == "washer"]
        big_w = [i for i in w_idxs if "m33" in parts[i]["kind"] or "m30" in parts[i]["kind"]]
        if wants_tray and big_w:
            w_idxs = big_w
        if not w_idxs:
            w_idxs = [i for i, p in enumerate(parts) if p["color"] in ("white", "silver", "grey") and goal_moved[i] is True]
        pick = _pick_best(w_idxs, prefer_color=wash_color, prefer_material=wash_mat)
        if pick is not None:
            parts[pick]["role"] = "target"
            if wants_tray:
                parts[pick]["target_seat"] = "tray"
            for j in [i for i, p in enumerate(parts) if p["shape"] == "washer"]:
                if j != pick:
                    parts[j]["role"] = "decoy"

    # Match bolt target
    if wants_bolt:
        b_idxs = [i for i, p in enumerate(parts) if p["shape"] == "bolt"]
        pick = _pick_best(
            b_idxs,
            prefer_small=wants_smaller_bolt,
            prefer_large=wants_larger_bolt,
            prefer_near_tray=wants_near_tray,
            prefer_near_robot=wants_near_robot,
            prefer_color=bolt_color,
            prefer_material=bolt_mat,
        )
        if pick is not None:
            parts[pick]["role"] = "target"
            if wants_tray:
                parts[pick]["target_seat"] = "tray"
            elif wants_hole:
                kind = parts[pick]["kind"]
                seat_map = {
                    "bolt_m16x50": "bore_m16",
                    "bolt_m24x70": "bore_m24",
                    "bolt_m30x100": "bore_m30",
                    "bolt_m24x50": "bore_m24",
                    "bolt_m30x60": "bore_m30",
                }
                parts[pick]["target_seat"] = seat_map.get(kind, "bore_m16")
            for j in b_idxs:
                if j != pick:
                    parts[j]["role"] = "decoy"

    # Match nut target
    if wants_nut:
        n_idxs = [i for i, p in enumerate(parts) if p["shape"] == "nut"]
        if wants_pocket:
            pocket_nuts = [i for i in n_idxs if any(sz in parts[i]["kind"] for sz in ("m24", "m30", "m36"))]
            if pocket_nuts:
                n_idxs = pocket_nuts
        pick = _pick_best(n_idxs, prefer_color=nut_color, prefer_material=nut_mat)
        if pick is not None:
            parts[pick]["role"] = "target"
            if wants_tray:
                parts[pick]["target_seat"] = "tray"
            elif wants_pocket:
                kind = parts[pick]["kind"]
                seat_map = {"nut_m24": "hex_m24", "nut_m30": "hex_m30", "nut_m36": "hex_m36"}
                parts[pick]["target_seat"] = seat_map.get(kind, "hex_m30")
            for j in [i for i, p in enumerate(parts) if p["shape"] == "nut"]:
                if j != pick:
                    parts[j]["role"] = "decoy"

    # If no instruction was given, still tag parts absent in goal as "target"
    if not (wants_washer or wants_bolt or wants_nut):
        for i, moved in enumerate(goal_moved):
            if moved is True:
                parts[i]["role"] = "target"


# ==============================================================================
# Public Entry Point: estimate_scene(obs: dict) -> dict
# ==============================================================================

def estimate_scene(obs: dict[str, Any]) -> dict[str, Any]:
    """Estimate 3D poses of tabletop parts and fixture seats from a wire observation.

    Uses the learned YOLO-pose detector (`$RC_WEIGHTS_DIR/perception/detector.pt`) when enabled
    (`RC_DETECTOR != '0'`) and confident; otherwise falls back to the classical perception pipeline.
    """
    if isinstance(obs, dict):
        _cell.detect_real_cell_from_obs(obs)
        try:
            from . import detector as _det

            if _det.detector_enabled() and _det.detector_weights_path().is_file():
                det_res = _det.estimate_scene_from_detector(obs)
                if det_res is not None:
                    return det_res
        except Exception:  # noqa: BLE001
            pass

    raw_images = obs.get("images") or {}
    img_left = _decode_image(raw_images.get("left"))
    img_right = _decode_image(raw_images.get("right"))
    img_wrist = _decode_image(raw_images.get("wrist"))
    img_goal = _decode_image(obs.get("goal"))
    proprio = obs.get("proprio") or {}
    instruction = str(obs.get("instruction") or "")

    # 1. Analytical robot capsules for self-occlusion masking
    capsules = _ur12e_capsules(proprio)

    # Nominal cameras & precomputed edge maps
    nom_cams: dict[str, Camera] = {}
    maps_by_view: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    rob_masks_nom: dict[str, np.ndarray] = {}
    for name, img in (("left", img_left), ("right", img_right)):
        if img is not None:
            H, W = img.shape[:2]
            _p, _m = side_camera_pose(name)
            cam = Camera(name, _p, _m, SIDE_CAMERA_FOVY, W, H)
            nom_cams[name] = cam
            maps_by_view[name] = _edge_maps(img)
            rob_masks_nom[name] = _project_capsules_mask(capsules, cam, pad_px=5.0)

    wrist_cam = None
    if img_wrist is not None:
        H, W = img_wrist.shape[:2]
        wrist_cam = _wrist_camera_from_proprio(proprio, W, H)

    # 2. Fixture pose estimation
    goal_cam: Camera | None = None
    use_goal = _is_valid_goal_image(img_goal)
    if use_goal and img_goal is not None:
        Hg, Wg = img_goal.shape[:2]
        goal_cam = Camera("goal", np.array(GOAL_CAMERA_POS, float), _look_mat(GOAL_CAMERA_POS, GOAL_CAMERA_TARGET), GOAL_CAMERA_FOVY, Wg, Hg)
        g_sf, _, g_gf, g_gc = _edge_maps(img_goal)
        fixture_xyyaw, fix_conf = _fit_fixture_pose([(goal_cam, g_sf, g_gf, g_gc, None)])
    else:
        # Fallback without goal: calibrate side cameras on fixed table borders first, then fit fixture multi-view
        pre_cams: dict[str, Camera] = {}
        for name, img in (("left", img_left), ("right", img_right)):
            if img is not None:
                c_pre, _ = _calibrate_side_camera(img, name, maps_by_view[name], rob_masks_nom[name], None)
                pre_cams[name] = c_pre
        fit_views = []
        for name in ("left", "right"):
            if name in pre_cams:
                sf, _, gf, gc = maps_by_view[name]
                rm = _project_capsules_mask(capsules, pre_cams[name], pad_px=5.0)
                fit_views.append((pre_cams[name], sf, gf, gc, rm))
        if fit_views:
            fixture_xyyaw, fix_conf = _fit_fixture_pose(fit_views)
        else:
            fixture_xyyaw = np.array([FIXTURE_ORIGIN[0], FIXTURE_ORIGIN[1], FIXTURE_NOMINAL_YAW])
            fix_conf = 0.0

    fx, fy, fyaw = float(fixture_xyyaw[0]), float(fixture_xyyaw[1]), float(fixture_xyyaw[2])
    seats, seat_meta = _seats_from_fixture_pose(fx, fy, fyaw)

    # 3. Calibrate side cameras using fixed table borders + fitted 3D fixture silhouette
    cams: dict[str, Camera] = {}
    cam_cal_info: dict[str, Any] = {}
    rob_masks: dict[str, np.ndarray] = {}
    for name, img in (("left", img_left), ("right", img_right)):
        if img is not None:
            cam_cal, info = _calibrate_side_camera(img, name, maps_by_view[name], rob_masks_nom[name], fixture_xyyaw)
            cams[name] = cam_cal
            cam_cal_info[name] = info
            rob_masks[name] = _project_capsules_mask(capsules, cam_cal, pad_px=4.0)

    if wrist_cam is not None and img_wrist is not None:
        cams["wrist"] = wrist_cam
        # Mask out the gripper fingers at the bottom corners of the wrist camera frame
        rm_w = np.zeros((wrist_cam.height, wrist_cam.width), dtype=bool)
        rm_w[int(wrist_cam.height * 0.68):, :int(wrist_cam.width * 0.30)] = True
        rm_w[int(wrist_cam.height * 0.68):, int(wrist_cam.width * 0.70):] = True
        rob_masks["wrist"] = rm_w

    # 4. Segment foreground part blobs in active views
    blobs_per_view: dict[str, list[Blob2D]] = {}
    view_ctxs: dict[str, dict[str, Any]] = {}
    seg_views = [("left", img_left), ("right", img_right)]
    if img_wrist is not None and "wrist" in cams and (img_left is None or img_right is None):
        seg_views.append(("wrist", img_wrist))
    for name, img in seg_views:
        if img is not None and name in cams:
            blobs, ctx = _segment_view(img, cams[name], rob_masks[name], fixture_xyyaw)
            blobs_per_view[name] = blobs
            view_ctxs[name] = ctx

    # 5. Stereo matching between `right` and `left`
    b_right = blobs_per_view.get("right", [])
    b_left = blobs_per_view.get("left", [])

    pairs: list[tuple[float, int, int, np.ndarray]] = []
    if "right" in cams and "left" in cams:
        for ir, br in enumerate(b_right):
            o_r, d_r = cams["right"].unproject_ray(*br.centroid_uv)
            for il, bl in enumerate(b_left):
                if br.color_family != bl.color_family:
                    continue
                o_l, d_l = cams["left"].unproject_ray(*bl.centroid_uv)
                pt3, epi = _triangulate_rays([o_r, o_l], [d_r, d_l])
                if epi > 0.024:
                    continue
                if not (0.36 <= pt3[0] <= 1.07 and -0.47 + _cell.LAYOUT_SHIFT[1] <= pt3[1] <= 0.45 and -0.012 <= pt3[2] <= 0.090):
                    continue
                pairs.append((epi, ir, il, pt3))
    pairs.sort(key=lambda t: t[0])

    used_r: set[int] = set()
    used_l: set[int] = set()
    matched_clusters: list[tuple[dict[str, Blob2D], np.ndarray, float, bool]] = []
    for epi, ir, il, pt3 in pairs:
        if ir in used_r or il in used_l:
            continue
        used_r.add(ir)
        used_l.add(il)
        matched_clusters.append(({"right": b_right[ir], "left": b_left[il]}, pt3, epi, False))

    # 6. Occlusion-aware handling for unmatched single-view blobs (including wrist fallback)
    def _is_occluded_in_view(pt_w: np.ndarray, target_view: str) -> bool:
        if target_view not in cams or target_view not in view_ctxs:
            return True
        c_t = cams[target_view]
        ctx_t = view_ctxs[target_view]
        u, v, d = c_t.project(pt_w[None, :])
        ui, vi, di = int(round(float(u[0]))), int(round(float(v[0]))), float(d[0])
        if di <= 0.08 or ui < 6 or ui >= c_t.width - 6 or vi < 6 or vi >= c_t.height - 6:
            return True
        x0, x1 = max(0, ui - 8), min(c_t.width, ui + 9)
        y0, y1 = max(0, vi - 8), min(c_t.height, vi + 9)
        if np.any(ctx_t["robot_mask"][y0:y1, x0:x1]) or np.any(ctx_t["fix_full"][y0:y1, x0:x1]):
            return True
        return False

    # Lazy wrist segmentation to upgrade any single-view occluded cluster to stereo if wrist sees it!
    def _ensure_wrist_segmented() -> None:
        if "wrist" not in blobs_per_view and img_wrist is not None and "wrist" in cams:
            bw, cw = _segment_view(img_wrist, cams["wrist"], rob_masks["wrist"], fixture_xyyaw)
            blobs_per_view["wrist"] = bw
            view_ctxs["wrist"] = cw

    for v_src, b_list, used_set, v_other in (
        ("right", b_right, used_r, "left"),
        ("left", b_left, used_l, "right"),
    ):
        if v_src not in cams:
            continue
        cam_s = cams[v_src]
        for idx, b in enumerate(b_list):
            if idx in used_set:
                continue
            hit = cam_s.intersect_z(b.centroid_uv[0], b.centroid_uv[1], 0.014)
            if hit is None:
                continue
            if not (0.36 <= hit[0] <= 1.07 and -0.47 + _cell.LAYOUT_SHIFT[1] <= hit[1] <= 0.45):
                continue
            occ = _is_occluded_in_view(hit, v_other)
            if not (occ or b.color_family == "gold" or (b.boundary_grad > 38.0 and b.max_lum > 180.0)):
                continue
            # Check if wrist camera also sees this single-view part (stereo upgrade!)
            bmap = {v_src: b}
            pt_est, epi_est = hit, 0.015
            if img_wrist is not None and "wrist" in cams:
                _ensure_wrist_segmented()
                o_s, d_s = cam_s.unproject_ray(*b.centroid_uv)
                best_w = None
                for bw in blobs_per_view.get("wrist", []):
                    if bw.color_family != b.color_family:
                        continue
                    o_w, d_w = cams["wrist"].unproject_ray(*bw.centroid_uv)
                    pw, ew = _triangulate_rays([o_s, o_w], [d_s, d_w])
                    if ew < 0.020 and 0.36 <= pw[0] <= 1.07 and -0.47 + _cell.LAYOUT_SHIFT[1] <= pw[1] <= 0.45 and -0.01 <= pw[2] <= 0.08:
                        if best_w is None or ew < best_w[0]:
                            best_w = (ew, pw, bw)
                if best_w is not None:
                    epi_est, pt_est, bw_match = best_w
                    bmap["wrist"] = bw_match
                    occ = False
            matched_clusters.append((bmap, pt_est, epi_est, occ))

    # Fallback for wrist-only when neither side camera is present
    if not matched_clusters and "wrist" in blobs_per_view:
        cam_fb = cams["wrist"]
        for b in blobs_per_view["wrist"]:
            hit = cam_fb.intersect_z(b.centroid_uv[0], b.centroid_uv[1], 0.014)
            if hit is not None and 0.36 <= hit[0] <= 1.07 and -0.47 + _cell.LAYOUT_SHIFT[1] <= hit[1] <= 0.45:
                matched_clusters.append(({"wrist": b}, hit, 0.015, True))

    # 7. Fit 3D kit part & pose for each cluster
    parts_out: list[dict[str, Any]] = []
    for blobs_map, init_xyz, epi, is_occ in matched_clusters:
        ref_blob = blobs_map.get("right") or next(iter(blobs_map.values()))
        # Check if near fixture -y face (candidate for leaning bolt)
        c_f, s_f = math.cos(fyaw), math.sin(fyaw)
        dx_f, dy_f = float(init_xyz[0]) - fx, float(init_xyz[1]) - fy
        loc_x = c_f * dx_f + s_f * dy_f
        loc_y = -s_f * dx_f + c_f * dy_f
        near_negy = (FIXTURE_XM - 0.03 <= loc_x <= FIXTURE_X1 + 0.03) and (FIXTURE_Y0 - 0.09 <= loc_y <= FIXTURE_Y0 + 0.02)
        allow_lean = near_negy and (ref_blob.on_negy_face or ref_blob.touches_fixture or is_occ)

        fit = _fit_candidate_3d(
            init_xyz,
            ref_blob.color_family,
            ref_blob.sub_color,
            blobs_map,
            cams,
            view_ctxs,
            fixture_xyyaw,
            allow_leaning=allow_lean,
        )
        if fit is None:
            continue
        # Reject spurious detections whose fitted center lies strictly inside the solid fixture block
        pos_np = np.asarray(fit["position"], float)
        dx_p, dy_p = float(pos_np[0]) - fx, float(pos_np[1]) - fy
        plx = c_f * dx_p + s_f * dy_p
        ply = -s_f * dx_p + c_f * dy_p
        if (
            fit["pose"] != "leaning"
            and (FIXTURE_XM + 0.005 <= plx <= FIXTURE_X1 - 0.005)
            and (FIXTURE_Y0 + 0.005 <= ply <= FIXTURE_Y1 - 0.005)
            and float(pos_np[2]) < 0.085
        ):
            continue
        # Filter weak single-view false positives
        if len(blobs_map) == 1 and fit["iou"] < 0.28:
            continue
        if len(blobs_map) >= 2 and fit["iou"] < 0.16:
            continue

        # Compute confidence
        if len(blobs_map) >= 2:
            epi_term = max(0.0, 1.0 - epi / 0.025)
            conf = float(np.clip(0.35 + 0.55 * fit["iou"] + 0.20 * epi_term, 0.1, 0.99))
        else:
            conf = float(np.clip(0.25 + 0.55 * fit["iou"] - (0.05 if is_occ else 0.15), 0.1, 0.88))

        sub_mat_map = {"gold": "brass", "white": "nylon", "silver": "aluminium", "grey": "stainless steel"}
        est_mat = sub_mat_map.get(ref_blob.sub_color, "aluminium")

        # Deduplicate against already added parts (within 22mm)
        dup = False
        for prev in parts_out:
            if np.linalg.norm(pos_np[:2] - np.asarray(prev["position"][:2], float)) < 0.022:
                if conf > prev["confidence"]:
                    prev.update({
                        "kind": fit["kind"], "shape": fit["shape"], "color": ref_blob.sub_color,
                        "material": est_mat,
                        "pose": fit["pose"], "position": pos_np.tolist(),
                        "axis": np.asarray(fit["axis"], float).tolist(),
                        "yaw": float(fit["yaw"]), "confidence": conf,
                        "occluded": bool(is_occ), "views": sorted(blobs_map.keys()),
                    })
                dup = True
                break
        if not dup:
            parts_out.append({
                "kind": fit["kind"],
                "shape": fit["shape"],
                "color": ref_blob.sub_color,
                "material": est_mat,
                "pose": fit["pose"],
                "position": pos_np.tolist(),
                "axis": np.asarray(fit["axis"], float).tolist(),
                "yaw": float(fit["yaw"]),
                "confidence": conf,
                "occluded": bool(is_occ),
                "views": sorted(blobs_map.keys()),
            })

    # 8. Dedicated leaning-bolt check on fixture -y face (catches grey M16 leaning bolts
    #    whose head/shank sits closely against the -y face and is hidden in `left`)
    extra_leaning = _detect_leaning_bolt_negy(cams, view_ctxs, fixture_xyyaw, parts_out)
    for lp in extra_leaning:
        lp_pos = np.asarray(lp["position"], float)
        replaced = False
        for prev in parts_out:
            if np.linalg.norm(lp_pos[:2] - np.asarray(prev["position"][:2], float)) < 0.045:
                if prev["occluded"] or prev["pose"] != "leaning":
                    prev.update(lp)
                replaced = True
                break
        if not replaced:
            parts_out.append(lp)

    # 9. Match target role by instruction + geometry/color/shape + goal presence
    _assign_roles(parts_out, seats, instruction, img_goal if use_goal else None, goal_cam)

    # Sort parts: targets first, then by descending confidence
    parts_out.sort(key=lambda p: (0 if p.get("role") == "target" else 1, -float(p["confidence"])))

    return {
        "parts": parts_out,
        "seats": seats,
        "fixture_yaw": float(fyaw),
        "fixture_position": [float(fx), float(fy), 0.0],
        "fixture_confidence": float(fix_conf),
        "seat_meta": seat_meta,
        "camera_calibration": cam_cal_info,
        # vlm-local patch: expose the calibrated cameras so the VLM overlay can project parts/seats
        # onto the same pixels the model sees.
        "cameras": {k: c for k, c in cams.items() if k in ("left", "right")},
        "perception_source": "classical",
    }
