"""Exact pure-NumPy UR12e + Robotiq 2F-140 forward kinematics, analytic Jacobian,
capsule clearance checks, and bounded damped least-squares IK.

Derived directly from the public participant kit UR12e model (ur12e.xml,
embodiment.py, 2f140.xml). Uses only NumPy and math (no MuJoCo at policy runtime).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

NUM_JOINTS = 6

# Home joint configuration (rad) from embodiment.py / docs/api.md
HOME_Q = np.array([-0.2055, -1.1091, 1.6007, -1.9708, -1.5708, 1.5708], dtype=np.float64)

# Protocol limits (docs/api.md, RULES.md)
PROTOCOL_JOINT_LO = np.array(
    [-6.28319, -6.28319, -2.792526803190927, -6.28319, -6.28319, -6.28319], dtype=np.float64
)
PROTOCOL_JOINT_HI = np.array(
    [6.28319, 6.28319, 2.792526803190927, 6.28319, 6.28319, 6.28319], dtype=np.float64
)
PROTOCOL_MAX_DELTA = 0.05

# Conservative internal bounds to guarantee zero runner clamps (RULES.md float32 margin note)
SAFE_STEP_LIMIT = 0.0495
IK_JOINT_LO = np.array([-6.22, -6.22, -2.75, -6.22, -6.22, -6.22], dtype=np.float64)
IK_JOINT_HI = np.array([6.22, 6.22, 2.75, 6.22, 6.22, 6.22], dtype=np.float64)

# Known Robotiq 2F-140 linkage pad tip reach along gripperframe +Z as a function of inner gap.
# Derived from public 2f140.xml linkage geometry:
#   gap ~0.6mm (closed) -> 0.1381 m
#   gap 16mm (M16 bolt) -> 0.1377 m
#   gap 26mm (M30 nut edge) -> 0.1372 m
#   gap 30mm (M30 bolt) -> 0.1369 m
#   gap 46mm (M30 nut flat) -> 0.1353 m
#   gap 62mm (M30 washer) -> 0.1329 m
#   gap 88mm (open ~0.72) -> 0.1275 m
PAD_TIP_CLOSED = 0.1381
PAD_TIP_BOLT_M16 = 0.1377
PAD_TIP_BOLT_M24 = 0.1373
PAD_TIP_BOLT_M30 = 0.1369
PAD_TIP_NUT_FLAT = 0.1353
PAD_TIP_NUT_EDGE = 0.1372
PAD_TIP_WASHER = 0.1329
PAD_HALF_WIDTH_X = 0.0135

# Team bracket + Gemini 336 camera housing in tool0 frame (embodiment.py)
GRIPPER_SPACER_M = 0.0244
BASE_MOUNT_Z_M = 0.007
GRIPPERFRAME_Z_M = 0.100
TOOL0_TO_SITE_Z = GRIPPER_SPACER_M + BASE_MOUNT_Z_M + GRIPPERFRAME_Z_M  # 0.1314 m

_PLATE_NORMAL = np.array([0.0, -0.38268, 0.92388], dtype=np.float64)
_PLATE_CENTRE = np.array([0.00027, 0.12209, 0.04292 - 0.0064], dtype=np.float64)
CAMERA_HOUSING_HALF = np.array([0.04475, 0.0125, 0.01535], dtype=np.float64)
CAMERA_HOUSING_POS = _PLATE_CENTRE + _PLATE_NORMAL * CAMERA_HOUSING_HALF[2]
WRIST_CAMERA_POS = np.array([0.0140, 0.1103, 0.0649], dtype=np.float64)
WRIST_CAMERA_XYAXES = np.array([-1.0, 0.0, 0.0, 0.0, 0.92388, 0.38268], dtype=np.float64)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion (w, x, y, z) to 3x3 rotation matrix."""
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to unit quaternion (w, x, y, z) with w >= 0."""
    R = np.asarray(R, dtype=np.float64)
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n > 1e-12:
        q /= n
    if q[0] < 0.0:
        q = -q
    return q


def _rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def wrap_angle(angle: float) -> float:
    """Wrap angle in radians to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def nearest_pi_yaw(target_yaw: float, ref_yaw: float) -> float:
    """Return target_yaw or target_yaw + pi (wrapped), whichever is closer to ref_yaw."""
    d0 = abs(wrap_angle(target_yaw - ref_yaw))
    d1 = abs(wrap_angle(target_yaw + math.pi - ref_yaw))
    return wrap_angle(target_yaw) if d0 <= d1 else wrap_angle(target_yaw + math.pi)


def closest_hex_yaw(current_yaw: float, target_mod60: float) -> float:
    """Return angle congruent to target_mod60 (mod pi/3) closest to current_yaw."""
    step = math.pi / 3.0
    k = round((current_yaw - target_mod60) / step)
    return target_mod60 + k * step


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("cannot normalize zero vector")
    return v / n


def rot_error(r_cur: np.ndarray, r_des: np.ndarray) -> np.ndarray:
    """World-frame axis-angle rotation vector rotating r_cur onto r_des."""
    r_err = r_des @ r_cur.T
    q = mat_to_quat(r_err)
    sin_half = float(np.linalg.norm(q[1:4]))
    if sin_half < 1e-11:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(sin_half, float(q[0]))
    return (angle / sin_half) * q[1:4]


def axis_angle_to_mat(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix for unit axis and angle (rad)."""
    ax = normalize(axis)
    c, s = math.cos(angle), math.sin(angle)
    kx, ky, kz = ax
    K = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + s * K + (1.0 - c) * (K @ K)


def slerp_rot(r0: np.ndarray, r1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two 3x3 rotation matrices."""
    err = rot_error(r0, r1)
    angle = float(np.linalg.norm(err))
    if angle < 1e-10 or t <= 0.0:
        return r0.copy()
    if t >= 1.0:
        return r1.copy()
    return axis_angle_to_mat(err / angle, angle * t) @ r0


def down_mat(x_yaw: float) -> np.ndarray:
    """Gripperframe pointing straight down: +z=[0,0,-1], +x=[cos(x_yaw), sin(x_yaw), 0].

    Note: fingers open along +y = z x x = [sin(x_yaw), -cos(x_yaw), 0].
    Wrist camera bracket sits on -x.
    """
    c, s = math.cos(x_yaw), math.sin(x_yaw)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)


def site_rot_from_zx(z_axis: np.ndarray, x_hint: np.ndarray) -> np.ndarray:
    """Right-handed 3x3 rotation matrix with +Z = z_axis and +X closest to x_hint."""
    z = normalize(z_axis)
    x = np.asarray(x_hint, dtype=np.float64)
    x = x - float(np.dot(x, z)) * z
    if float(np.linalg.norm(x)) < 1e-8:
        fallback = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = fallback - float(np.dot(fallback, z)) * z
    x = normalize(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def site_rot_from_zy(z_axis: np.ndarray, y_hint: np.ndarray) -> np.ndarray:
    """Right-handed 3x3 rotation matrix with +Z = z_axis and +Y closest to y_hint."""
    z = normalize(z_axis)
    y = np.asarray(y_hint, dtype=np.float64)
    y = y - float(np.dot(y, z)) * z
    if float(np.linalg.norm(y)) < 1e-8:
        fallback = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        y = fallback - float(np.dot(fallback, z)) * z
    y = normalize(y)
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


# Static link transforms from ur12e.xml: (pos, quat_wxyz) before each joint's local Z rotation.
_R_BASE_INERTIA = quat_to_mat(np.array([0.0, 0.0, 0.0, 1.0]))
_LINK_STATIC = (
    (np.array([0.0, 0.0, 0.1807]), np.eye(3)),  # shoulder_link
    (np.array([0.0, 0.0, 0.0]), quat_to_mat(np.array([0.707107, 0.707107, 0.0, 0.0]))),  # upper_arm_link
    (np.array([-0.6127, 0.0, 0.0]), np.eye(3)),  # forearm_link
    (np.array([-0.57155, 0.0, 0.17415]), np.eye(3)),  # wrist_1_link
    (
        np.array([0.0, -0.11985, -2.45816e-11]),
        quat_to_mat(np.array([0.707107, 0.707107, 0.0, 0.0])),
    ),  # wrist_2_link
    (
        np.array([0.0, 0.11655, -2.39048e-11]),
        quat_to_mat(np.array([0.707107, -0.707107, 0.0, 0.0])),
    ),  # wrist_3_link
)

# flange * tool0 = identity; verified against MuJoCo
_R_FLANGE = quat_to_mat(np.array([0.5, -0.5, -0.5, -0.5]))
_R_TOOL0 = _R_FLANGE @ quat_to_mat(np.array([0.5, 0.5, 0.5, 0.5]))
_P_SITE_IN_TOOL0 = np.array([0.0, 0.0, TOOL0_TO_SITE_Z], dtype=np.float64)
_R_SITE_IN_TOOL0 = quat_to_mat(np.array([0.7071, 0.0, 0.0, -0.7071]))

# Wrist camera orientation in tool0 from WRIST_CAMERA_XYAXES
_CAM_X = WRIST_CAMERA_XYAXES[:3]
_CAM_Y = WRIST_CAMERA_XYAXES[3:6]
_R_WRIST_CAM_IN_TOOL0 = np.column_stack([_CAM_X, _CAM_Y, np.cross(_CAM_X, _CAM_Y)])


@dataclass
class FKResult:
    """Full forward-kinematics result for UR12e + 2F-140."""

    ee_pos: np.ndarray  # gripperframe world position (3,)
    ee_rot: np.ndarray  # gripperframe world rotation (3, 3)
    joint_pos_world: np.ndarray  # (6, 3) origins of the 6 arm joints
    joint_axes_world: np.ndarray  # (6, 3) unit rotation axes of the 6 arm joints
    link_rots: tuple[np.ndarray, ...]  # 6 world rotation matrices for shoulder..wrist_3
    tool0_pos: np.ndarray  # (3,) tool0 world origin
    tool0_rot: np.ndarray  # (3, 3) tool0 world rotation
    camera_housing_pos: np.ndarray  # (3,) wrist camera collision box center


def forward_kinematics(q: np.ndarray) -> FKResult:
    """Compute exact UR12e + 2F-140 forward kinematics for joint angles q (6,)."""
    q = np.asarray(q, dtype=np.float64)
    p = np.zeros(3, dtype=np.float64)
    R = _R_BASE_INERTIA.copy()

    j_pos = np.empty((6, 3), dtype=np.float64)
    j_axes = np.empty((6, 3), dtype=np.float64)
    link_rots: list[np.ndarray] = []

    for i in range(6):
        offset, r_static = _LINK_STATIC[i]
        p = p + R @ offset
        R = R @ r_static @ _rot_z(float(q[i]))
        j_pos[i] = p
        j_axes[i] = R[:, 2]
        link_rots.append(R.copy())

    tool0_pos = p
    tool0_rot = R @ _R_TOOL0
    ee_pos = tool0_pos + tool0_rot @ _P_SITE_IN_TOOL0
    ee_rot = tool0_rot @ _R_SITE_IN_TOOL0
    cam_pos = tool0_pos + tool0_rot @ CAMERA_HOUSING_POS

    return FKResult(
        ee_pos=ee_pos,
        ee_rot=ee_rot,
        joint_pos_world=j_pos,
        joint_axes_world=j_axes,
        link_rots=tuple(link_rots),
        tool0_pos=tool0_pos,
        tool0_rot=tool0_rot,
        camera_housing_pos=cam_pos,
    )


def fk_pose(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (ee_pos, ee_rot) for gripperframe at joint configuration q."""
    res = forward_kinematics(q)
    return res.ee_pos, res.ee_rot


def wrist_camera_pose(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (cam_pos_world, cam_rot_world) for the Gemini 336 wrist camera.

    Columns of cam_rot_world are OpenGL/MuJoCo camera axes: +X right, +Y up, -Z forward.
    """
    res = forward_kinematics(q)
    pos = res.tool0_pos + res.tool0_rot @ WRIST_CAMERA_POS
    rot = res.tool0_rot @ _R_WRIST_CAM_IN_TOOL0
    return pos, rot


def jacobian(q: np.ndarray, fk: FKResult | None = None) -> np.ndarray:
    """Analytic 6x6 geometric Jacobian [J_p; J_r] for gripperframe in world frame."""
    res = forward_kinematics(q) if fk is None else fk
    J = np.empty((6, 6), dtype=np.float64)
    ee = res.ee_pos
    for i in range(6):
        axis = res.joint_axes_world[i]
        J[:3, i] = np.cross(axis, ee - res.joint_pos_world[i])
        J[3:, i] = axis
    return J


def clearance_penalty(q: np.ndarray, fk: FKResult | None = None, *, wrist_up: bool = False) -> float:
    """Compute collision-clearance penalty for UR12e arm capsules & wrist camera against table z=0.

    Matches the collision primitives in embodiment._apply_arm_collisions and _add_wrist_mount.
    Returns 0.0 when all capsules/links have safe clearance above the table and elbow-up posture.
    """
    res = forward_kinematics(q) if fk is None else fk
    pen = 0.0

    # Keep elbow-up branch (prevents forearm sweeping down into the table)
    if q[2] < 0.15:
        pen += 20.0 + 40.0 * (0.15 - float(q[2]))
    if q[1] > -0.05:
        pen += 20.0 + 40.0 * (float(q[1]) + 0.05)

    # Link origins: [shoulder, upper_arm, forearm, wrist_1, wrist_2, wrist_3]
    forearm_p, forearm_R = res.joint_pos_world[2], res.link_rots[2]
    w1_p, w1_R = res.joint_pos_world[3], res.link_rots[3]
    w2_p, w2_R = res.joint_pos_world[4], res.link_rots[4]
    w3_p, w3_R = res.joint_pos_world[5], res.link_rots[5]

    # Exact lowest-point z of collision capsules (radius subtracted)
    # forearm_collision: fromto (-0.08, 0, 0.04) to (-0.48, 0, 0.04), r=0.048
    fa_a = forearm_p + forearm_R @ np.array([-0.08, 0.0, 0.04])
    fa_b = forearm_p + forearm_R @ np.array([-0.48, 0.0, 0.04])
    fa_min_z = min(float(fa_a[2]), float(fa_b[2])) - 0.048

    # wrist_1_collision: fromto (0, -0.02, 0) to (0, -0.10, 0), r=0.045
    w1_a = w1_p + w1_R @ np.array([0.0, -0.02, 0.0])
    w1_b = w1_p + w1_R @ np.array([0.0, -0.10, 0.0])
    w1_min_z = min(float(w1_a[2]), float(w1_b[2])) - 0.045

    # wrist_2_collision: fromto (0, 0.02, 0) to (0, 0.08, 0), r=0.042
    w2_a = w2_p + w2_R @ np.array([0.0, 0.02, 0.0])
    w2_b = w2_p + w2_R @ np.array([0.0, 0.08, 0.0])
    w2_min_z = min(float(w2_a[2]), float(w2_b[2])) - 0.042

    # wrist_3_collision: cylinder at (0, 0, -0.025), r=0.045, half=0.025
    w3_c = w3_p + w3_R @ np.array([0.0, 0.0, -0.025])
    w3_min_z = float(w3_c[2]) - 0.045

    # wrist camera housing box center minus conservative radius
    cam_min_z = float(res.camera_housing_pos[2]) - 0.030

    # Safe heights above table (z = 0)
    checks = (
        (fa_min_z, 0.030),
        (w1_min_z, 0.025),
        (w2_min_z, 0.022),
        (w3_min_z, 0.020),
        (cam_min_z, 0.025),
        (float(w1_p[2]), 0.075),
        (float(w2_p[2]), 0.070),
        (float(w3_p[2]), 0.065),
        (float(forearm_p[2]), 0.080),
        (float(res.camera_housing_pos[2]), 0.055),
    )
    for z, min_z in checks:
        if z < min_z:
            pen += 500.0 * (min_z - z) + 5.0

    if wrist_up:
        w1z = float(w1_p[2])
        if w1z < 0.14:
            pen += 60.0 * (0.14 - w1z) + 2.0

    return pen


def unwrap_near(
    q: np.ndarray,
    q_ref: np.ndarray,
    lo: np.ndarray = IK_JOINT_LO,
    hi: np.ndarray = IK_JOINT_HI,
) -> np.ndarray:
    """Shift 2*pi-periodic joints (all except shoulder_lift & elbow) closest to q_ref within limits."""
    out = np.asarray(q, dtype=np.float64).copy()
    for i in (0, 3, 4, 5):
        best = out[i]
        best_d = abs(best - q_ref[i])
        for k in (-2, -1, 1, 2):
            cand = out[i] + k * 2.0 * math.pi
            if lo[i] <= cand <= hi[i]:
                d = abs(cand - q_ref[i])
                if d < best_d:
                    best_d = d
                    best = cand
        out[i] = best
    return np.clip(out, lo, hi)


@dataclass
class IKSolution:
    q: np.ndarray
    pos_err: float
    rot_err: float
    clearance_pen: float
    converged: bool


class BoundedDampedIK:
    """Bounded damped least-squares IK solver with multi-branch clearance selection."""

    def __init__(
        self,
        lo: np.ndarray = IK_JOINT_LO,
        hi: np.ndarray = IK_JOINT_HI,
    ) -> None:
        self.lo = np.asarray(lo, dtype=np.float64).copy()
        self.hi = np.asarray(hi, dtype=np.float64).copy()

    def _dls_single(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        q_init: np.ndarray,
        *,
        max_iter: int = 80,
        damping: float = 1e-2,
        pos_tol: float = 1e-4,
        rot_tol: float = 1e-3,
    ) -> tuple[np.ndarray, float, float]:
        q = np.clip(np.asarray(q_init, dtype=np.float64).copy(), self.lo, self.hi)
        eye6 = np.eye(6, dtype=np.float64)
        pos_err = 1e9
        r_err_norm = 1e9

        for _ in range(max_iter):
            fk = forward_kinematics(q)
            dp = target_pos - fk.ee_pos
            dr = rot_error(fk.ee_rot, target_rot)
            pos_err = float(np.linalg.norm(dp))
            r_err_norm = float(np.linalg.norm(dr))
            if pos_err < pos_tol and r_err_norm < rot_tol:
                break
            J = jacobian(q, fk)
            err = np.concatenate([dp, dr])
            lam = damping + 0.05 * min(pos_err, 0.2)
            A = J @ J.T + (lam * lam) * eye6
            try:
                dq = J.T @ np.linalg.solve(A, err)
            except np.linalg.LinAlgError:
                dq = J.T @ np.linalg.lstsq(A + 1e-3 * eye6, err, rcond=None)[0]
            peak = float(np.max(np.abs(dq)))
            if peak > 0.35:
                dq *= 0.35 / peak
            q = np.clip(q + dq, self.lo, self.hi)

        return q, pos_err, r_err_norm

    def solve(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        q_ref: np.ndarray,
        *,
        wrist_up: bool = False,
        prefer_close: bool = True,
        max_iter: int = 80,
    ) -> IKSolution:
        """Solve bounded DLS IK for gripperframe pose (target_pos, target_rot) near q_ref."""
        target_pos = np.asarray(target_pos, dtype=np.float64)
        target_rot = np.asarray(target_rot, dtype=np.float64)
        q_ref = np.asarray(q_ref, dtype=np.float64)

        # Fast path from q_ref
        q0, pe0, re0 = self._dls_single(target_pos, target_rot, q_ref, max_iter=max_iter)
        q0 = unwrap_near(q0, q_ref, self.lo, self.hi)
        pen0 = clearance_penalty(q0, wrist_up=wrist_up)
        if pe0 < 1e-3 and re0 < 5e-3 and pen0 == 0.0:
            return IKSolution(q=q0, pos_err=pe0, rot_err=re0, clearance_pen=0.0, converged=True)

        # Multi-seed search covering both wrist_2 < 0 and wrist_2 > 0 elbow-up branches
        pan = math.atan2(float(target_pos[1]), float(target_pos[0]))
        seeds = [
            np.array([pan, -1.25, 2.20, -1.20, +1.5708, 0.0]),
            np.array([pan - 0.35, -1.25, 2.50, -1.35, +0.95, 0.0]),
            HOME_Q.copy(),
            np.array([pan, -1.10, 1.60, -2.00, -1.5708, float(q_ref[5])]),
            np.array([pan, -0.90, 1.30, -1.90, -1.5708, 1.5708]),
            np.array([pan, -1.30, 1.80, -0.50, +1.5708, 0.0]),
            np.array([pan + 0.25, -1.35, 2.10, -1.80, -1.5708, 0.0]),
        ]

        candidates = [(q0, pe0, re0, pen0)]
        for seed in seeds:
            qc, pec, rec = self._dls_single(target_pos, target_rot, seed, max_iter=120)
            qc = unwrap_near(qc, q_ref, self.lo, self.hi)
            penc = clearance_penalty(qc, wrist_up=wrist_up)
            candidates.append((qc, pec, rec, penc))

        best = candidates[0]
        best_cost = float("inf")
        dist_weight = 0.05 if prefer_close else 0.005
        for qc, pec, rec, penc in candidates:
            err_cost = 1000.0 * (pec + 0.2 * rec)
            dist_cost = float(np.linalg.norm(qc - q_ref))
            cost = err_cost + penc + dist_weight * dist_cost
            if cost < best_cost:
                best_cost = cost
                best = (qc, pec, rec, penc)

        q_best, pe_best, re_best, pen_best = best
        return IKSolution(
            q=q_best,
            pos_err=pe_best,
            rot_err=re_best,
            clearance_pen=pen_best,
            converged=(pe_best < 3e-3 and re_best < 1.5e-2),
        )


def compute_safe_delta(
    q_target_now: np.ndarray,
    q_goal: np.ndarray,
    *,
    max_step: float = SAFE_STEP_LIMIT,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a strictly legal joint delta from q_target_now toward q_goal.

    Guarantees:
      - |delta[i]| <= SAFE_STEP_LIMIT < 0.05
      - q_target_next = q_target_now + delta stays strictly inside PROTOCOL_JOINT_LO/HI
    Returns (delta, q_target_next).
    """
    q_now = np.asarray(q_target_now, dtype=np.float64)
    q_des = np.clip(np.asarray(q_goal, dtype=np.float64), IK_JOINT_LO, IK_JOINT_HI)
    step = min(float(max_step), SAFE_STEP_LIMIT)
    dq = q_des - q_now
    peak = float(np.max(np.abs(dq)))
    if peak > step:
        dq = dq * (step / peak)
    # Enforce per-joint target limits strictly inside protocol bounds
    q_next = np.clip(q_now + dq, IK_JOINT_LO, IK_JOINT_HI)
    dq = np.clip(q_next - q_now, -SAFE_STEP_LIMIT, SAFE_STEP_LIMIT)
    q_next = q_now + dq
    return dq, q_next
