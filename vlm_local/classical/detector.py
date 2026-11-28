"""Learned multi-view YOLO-pose perception for Reality Check tabletop scenes (engine b23043a).

Lazy-loads an Ultralytics YOLO-pose model once from `$RC_WEIGHTS_DIR/perception/detector.pt`
with `$RC_WEIGHTS_DIR/perception/classes.json`, strictly offline (no network, no telemetry,
no font downloads, writable config under `/scratch` or temp dir).

Runs one batched call across available `left`, `right`, and `wrist` 640x360 RGB frames,
reconstructs 3D parts and fixture seats via multi-view keypoint triangulation (using nominal
side-camera poses from `cameras.py` and exact FK wrist-camera pose from `kinematics.py`)
with single-view ray-plane fallback, estimates part material from foreground box crops, and
emits the exact dict schema of `vision.estimate_scene(obs)`.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import cell as _cell
from .kinematics import HOME_Q, NUM_JOINTS, wrap_angle, wrist_camera_pose
from .scene_interface import (
    FASTENER_GEOM,
    FIXTURE_BLOCK_TOP_Z,
    FIXTURE_BLOCK_Y_MIN,
    NOMINAL_FIXTURE_XY,
    NOMINAL_FIXTURE_YAW,
    SEAT_LOCAL_GEOM,
    TRAY_FLOOR_Z,
)


def _configure_ultralytics_offline() -> str:
    """Configure environment variables before importing Ultralytics so it never touches the network
    or attempts writes outside `/scratch` / system temp."""
    os.environ["YOLO_OFFLINE"] = "1"
    os.environ["ULTRALYTICS_OFFLINE"] = "1"
    os.environ.setdefault("YOLO_AUTOINSTALL", "0")
    os.environ.setdefault("YOLO_VERBOSE", "False")

    cfg_dir = os.environ.get("YOLO_CONFIG_DIR")
    if not cfg_dir:
        base_candidates = [
            os.environ.get("RC_SCRATCH"),
            "/scratch" if os.path.isdir("/scratch") and os.access("/scratch", os.W_OK) else None,
            tempfile.gettempdir(),
        ]
        chosen = next((b for b in base_candidates if b), tempfile.gettempdir())
        cfg_path = Path(chosen) / "ultralytics"
        try:
            cfg_path.mkdir(parents=True, exist_ok=True)
            cfg_dir = str(cfg_path)
            os.environ["YOLO_CONFIG_DIR"] = cfg_dir
        except OSError:
            cfg_dir = tempfile.gettempdir()
    return cfg_dir


_configure_ultralytics_offline()

# Nominal camera geometry from engine/rc_engine/cameras.py and embodiment.py
GEMINI_336_SIZE: tuple[int, int] = (640, 360)
SIDE_CAMERA_FOVY: float = _cell.SIDE_FOVY
WRIST_CAMERA_FOVY: float = _cell.WRIST_FOVY
SIDE_CAMERA_POSES: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "left": ((0.68, 0.66, 0.66), (0.73, -0.04, 0.03)),
    "right": ((1.02, -0.62, 0.65), (0.73, -0.04, 0.03)),
}

DEFAULT_CLASS_NAMES: tuple[str, ...] = (
    "bolt_m16x50",
    "bolt_m24x70",
    "bolt_m30x100",
    "nut_m12",
    "nut_m16",
    "nut_m20",
    "nut_m24",
    "nut_m30",
    "nut_m36",
    "washer_m12",
    "washer_m16",
    "washer_m20",
    "washer_m33",
    "fixture",
)

DEFAULT_FIXTURE_KPTS: tuple[str, ...] = (
    "bore_m16",
    "bore_m24",
    "bore_m30",
    "hex_m24",
    "hex_m30",
    "hex_m36",
)

MATERIAL_TO_COLOR: dict[str, str] = {
    "brass": "gold",
    "nylon": "white",
    "aluminium": "silver",
    "aluminum": "silver",
    "stainless steel": "grey",
    "stainless_steel": "grey",
}


# ---------------------------------------------------------------------------
# Environment & Path Helpers
# ---------------------------------------------------------------------------

def detector_enabled() -> bool:
    """Return False when `RC_DETECTOR=0` (or false/off/no) forces the classical fallback."""
    val = os.environ.get("RC_DETECTOR", "1").strip().lower()
    return val not in ("0", "false", "no", "off")


def detector_weights_path() -> Path:
    """Resolve runtime detector weights path (`$RC_WEIGHTS_DIR/perception/detector.pt`)."""
    override = os.environ.get("RC_DETECTOR_PATH", "").strip()
    if override:
        return Path(override)
    weights_root = Path(os.environ.get("RC_WEIGHTS_DIR") or "/weights")
    return weights_root / "perception" / "detector.pt"


def detector_classes_path(weights_path: Path | None = None) -> Path:
    """Resolve runtime classes spec path (`$RC_WEIGHTS_DIR/perception/classes.json`)."""
    override = os.environ.get("RC_DETECTOR_CLASSES", "").strip()
    if override:
        return Path(override)
    w = weights_path or detector_weights_path()
    return w.parent / "classes.json"


def canonical_kind(class_name: str) -> str:
    c = str(class_name or "").strip().lower()
    if c == "fixture" or "fixture" in c:
        return "fixture"
    if "bolt" in c:
        return "bolt"
    if "nut" in c:
        return "nut"
    if "washer" in c:
        return "washer"
    return "unknown"


def parse_classes_spec(
    raw_spec: Any,
    model_names: dict[int, str] | list[str] | None = None,
) -> tuple[list[str], dict[int, str], dict[str, tuple[tuple[float, float], float]]]:
    """Parse `classes.json` plus optional `model.names` fallback.

    Returns `(class_names_in_index_order, fixture_kpt_index_to_seat_id, fixture_seat_local_spec)`.
    `fixture_seat_local_spec` maps `seat_id -> ((local_x_m, local_y_m), top_z_m)`.
    """
    class_names: list[str] = []
    kpt_to_seat: dict[int, str] = {i: sid for i, sid in enumerate(DEFAULT_FIXTURE_KPTS)}
    seat_spec: dict[str, tuple[tuple[float, float], float]] = {
        sid: (
            (float(SEAT_LOCAL_GEOM[sid]["local_xy"][0]), float(SEAT_LOCAL_GEOM[sid]["local_xy"][1])),
            float(SEAT_LOCAL_GEOM[sid].get("top_z", FIXTURE_BLOCK_TOP_Z)),
        )
        for sid in DEFAULT_FIXTURE_KPTS
        if sid in SEAT_LOCAL_GEOM
    }

    def _extract_names(obj: Any) -> list[str]:
        if isinstance(obj, (list, tuple)):
            return [str(x).strip() for x in obj]
        if isinstance(obj, dict):
            try:
                items = sorted(((int(k), str(v).strip()) for k, v in obj.items()), key=lambda kv: kv[0])
                return [v for _, v in items]
            except (TypeError, ValueError):
                pass
        return []

    def _parse_kpt_index(key: Any) -> int | None:
        s = str(key).strip().lower()
        if s.startswith("k"):
            s = s[1:]
        try:
            idx = int(s)
            return idx if 0 <= idx < 6 else None
        except ValueError:
            return None

    if isinstance(raw_spec, (list, tuple)):
        class_names = _extract_names(raw_spec)
    elif isinstance(raw_spec, dict):
        for key in ("names", "classes", "class_names", "id_to_name"):
            if key in raw_spec:
                class_names = _extract_names(raw_spec[key])
                if class_names:
                    break
        if not class_names and all(str(k).isdigit() for k in raw_spec.keys() if k is not None) and raw_spec:
            class_names = _extract_names(raw_spec)

        # Parse `keypoints.fixture` (e.g. ["bore_m16", "bore_m24", ...]) if present
        kpts_block = raw_spec.get("keypoints")
        if isinstance(kpts_block, dict) and isinstance(kpts_block.get("fixture"), (list, tuple)):
            fk_list = kpts_block["fixture"]
            kpt_to_seat = {i: str(s).strip() for i, s in enumerate(fk_list[:6]) if s and str(s).strip()}
        else:
            for key in (
                "fixture_keypoints",
                "fixture_seats",
                "fixture_kpt_to_seat",
                "fixture_keypoint_to_seat",
                "keypoint_to_seat",
                "kpt_to_seat",
                "fixture_kpts",
                "seat_keypoints",
            ):
                val = raw_spec.get(key)
                if isinstance(val, (list, tuple)) and val:
                    kpt_to_seat = {i: str(s).strip() for i, s in enumerate(val[:6]) if s and str(s).strip()}
                    break
                if isinstance(val, dict) and val:
                    parsed: dict[int, str] = {}
                    for mk, mv in val.items():
                        idx = _parse_kpt_index(mk)
                        if idx is not None and isinstance(mv, str):
                            parsed[idx] = mv.strip()
                        else:
                            idx_rev = _parse_kpt_index(mv)
                            if idx_rev is not None and isinstance(mk, str):
                                parsed[idx_rev] = mk.strip()
                    if parsed:
                        kpt_to_seat.update(parsed)
                        break

        # Parse `fixture_seat_spec` (xy_mm / xy_m, top_z_mm / top_z_m)
        raw_fspec = raw_spec.get("fixture_seat_spec")
        if isinstance(raw_fspec, dict):
            for sid, sinfo in raw_fspec.items():
                if not isinstance(sinfo, dict):
                    continue
                if "xy_mm" in sinfo and isinstance(sinfo["xy_mm"], (list, tuple)) and len(sinfo["xy_mm"]) >= 2:
                    lx, ly = float(sinfo["xy_mm"][0]) * 0.001, float(sinfo["xy_mm"][1]) * 0.001
                elif "xy" in sinfo and isinstance(sinfo["xy"], (list, tuple)) and len(sinfo["xy"]) >= 2:
                    lx, ly = float(sinfo["xy"][0]), float(sinfo["xy"][1])
                else:
                    continue
                if "top_z_mm" in sinfo:
                    tz = float(sinfo["top_z_mm"]) * 0.001
                elif "top_z" in sinfo:
                    tz = float(sinfo["top_z"])
                else:
                    tz = FIXTURE_BLOCK_TOP_Z
                seat_spec[str(sid).strip()] = ((lx, ly), tz)

    if not class_names and model_names is not None:
        class_names = _extract_names(model_names)
    if not class_names:
        class_names = list(DEFAULT_CLASS_NAMES)

    return class_names, kpt_to_seat, seat_spec


# ---------------------------------------------------------------------------
# Camera Model & 3D Triangulation / Projection Primitives
# ---------------------------------------------------------------------------

def look_at_matrix(eye: tuple[float, float, float] | np.ndarray, target: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    """OpenGL/MuJoCo camera rotation matrix `R_wc` (`columns = [right, up, -forward]`)."""
    e = np.asarray(eye, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    fwd = t - e
    fwd /= float(np.linalg.norm(fwd))
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    right /= float(np.linalg.norm(right))
    up = np.cross(right, fwd)
    return np.column_stack([right, up, -fwd])


def make_side_camera(name: str, width: int = GEMINI_336_SIZE[0], height: int = GEMINI_336_SIZE[1]):
    """Construct nominal pinhole `Camera` for `'left'` or `'right'` from `cameras.py`."""
    from .vision import Camera

    pos, mat = _cell.side_camera_pose(name)
    return Camera(name, pos, mat, SIDE_CAMERA_FOVY, width, height)


def make_wrist_camera(
    joint_pos: Any = None,
    proprio: dict[str, Any] | None = None,
    width: int = GEMINI_336_SIZE[0],
    height: int = GEMINI_336_SIZE[1],
):
    """Construct pinhole `Camera` for `'wrist'` using forward kinematics of `joint_pos`."""
    from .vision import Camera, _wrist_camera_from_proprio

    q_raw = joint_pos if joint_pos is not None else (proprio.get("joint_pos") if isinstance(proprio, dict) else None)
    if q_raw is not None:
        q = np.asarray(q_raw, dtype=np.float64).reshape(-1)
        if q.size == NUM_JOINTS and np.all(np.isfinite(q)):
            pos, rot = wrist_camera_pose(q)
            return Camera("wrist", pos, rot, WRIST_CAMERA_FOVY, width, height)
    if isinstance(proprio, dict):
        cam = _wrist_camera_from_proprio(proprio, width, height)
        if cam is not None:
            return cam
    pos, rot = wrist_camera_pose(HOME_Q)
    return Camera("wrist", pos, rot, WRIST_CAMERA_FOVY, width, height)


def build_nominal_cameras(
    obs: dict[str, Any],
    decoded_images: dict[str, np.ndarray | None] | None = None,
) -> dict[str, Any]:
    """Build `{'left', 'right', 'wrist'}` Camera objects matching image resolutions in `obs`."""
    proprio = obs.get("proprio") if isinstance(obs.get("proprio"), dict) else {}
    imgs = decoded_images or {}
    cams: dict[str, Any] = {}
    for side in ("left", "right"):
        im = imgs.get(side)
        h, w = im.shape[:2] if isinstance(im, np.ndarray) and im.ndim >= 2 else (GEMINI_336_SIZE[1], GEMINI_336_SIZE[0])
        cams[side] = make_side_camera(side, w, h)
    im_w = imgs.get("wrist")
    hw, ww = im_w.shape[:2] if isinstance(im_w, np.ndarray) and im_w.ndim >= 2 else (GEMINI_336_SIZE[1], GEMINI_336_SIZE[0])
    wcam = make_wrist_camera(proprio.get("joint_pos"), proprio=proprio, width=ww, height=hw)
    if wcam is not None:
        cams["wrist"] = wcam
    return cams


def triangulate_rays(
    origins: list[np.ndarray],
    dirs: list[np.ndarray],
    weights: list[float] | None = None,
) -> tuple[np.ndarray, float]:
    """Weighted least-squares 3D triangulation of two or more unit rays.

    Returns `(point_world_3d, weighted_rms_ray_distance)`.
    """
    n = len(origins)
    if n < 2:
        raise ValueError("triangulate_rays requires at least 2 rays")
    ws = [1.0] * n if weights is None else [max(1e-6, float(w)) for w in weights]
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    eye3 = np.eye(3, dtype=np.float64)
    unit_dirs: list[np.ndarray] = []
    for o, d, w in zip(origins, dirs, ws):
        dn = np.asarray(d, dtype=np.float64)
        dn = dn / max(float(np.linalg.norm(dn)), 1e-12)
        unit_dirs.append(dn)
        P = eye3 - np.outer(dn, dn)
        A += w * P
        b += w * (P @ np.asarray(o, dtype=np.float64))
    pt = np.linalg.solve(A + 1e-9 * eye3, b)
    sq_errs = []
    w_sum = 0.0
    for o, dn, w in zip(origins, unit_dirs, ws):
        diff = pt - np.asarray(o, dtype=np.float64)
        perp = diff - float(np.dot(diff, dn)) * dn
        sq_errs.append(w * float(np.dot(perp, perp)))
        w_sum += w
    rms = math.sqrt(sum(sq_errs) / max(w_sum, 1e-9))
    return pt, rms


def ray_point_distance(origin: np.ndarray, direction: np.ndarray, pt: np.ndarray) -> tuple[float, float]:
    """Return `(perp_distance, depth_along_ray)` from `(origin, direction)` to `pt`."""
    d = np.asarray(direction, dtype=np.float64)
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    rel = np.asarray(pt, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    along = float(np.dot(rel, d))
    perp = float(np.linalg.norm(rel - along * d))
    return perp, along


def intersect_vertical_plane(
    origin: np.ndarray,
    direction: np.ndarray,
    plane_point_xy: np.ndarray,
    plane_normal_xy: np.ndarray,
) -> np.ndarray | None:
    """Intersect 3D ray `(origin, direction)` with vertical plane `(p[:2] - plane_point_xy) . plane_normal_xy = 0`."""
    n2 = np.asarray(plane_normal_xy, dtype=np.float64)
    denom = float(direction[0] * n2[0] + direction[1] * n2[1])
    if abs(denom) < 1e-6:
        return None
    num = float((plane_point_xy[0] - origin[0]) * n2[0] + (plane_point_xy[1] - origin[1]) * n2[1])
    t = num / denom
    if t <= 0.05:
        return None
    return np.asarray(origin, dtype=np.float64) + t * np.asarray(direction, dtype=np.float64)


# ---------------------------------------------------------------------------
# Material / Colour Estimation from Box Crops
# ---------------------------------------------------------------------------

def estimate_crop_features(
    img_rgb: np.ndarray,
    xyxy: tuple[float, float, float, float],
    kind: str,
) -> np.ndarray | None:
    """Extract foreground RGB pixel array `(M, 3)` float32 inside `xyxy`, rejecting table/fixture background."""
    H, W = img_rgb.shape[:2]
    x0 = int(max(0, math.floor(xyxy[0])))
    y0 = int(max(0, math.floor(xyxy[1])))
    x1 = int(min(W, math.ceil(xyxy[2])))
    y1 = int(min(H, math.ceil(xyxy[3])))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None

    crop = img_rgb[y0:y1, x0:x1].astype(np.float32)
    ch, cw = crop.shape[:2]

    # Estimate local background RGB from a 4-px ring just outside the detection box
    pad = 4
    rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
    rx1, ry1 = min(W, x1 + pad), min(H, y1 + pad)
    ring_patch = img_rgb[ry0:ry1, rx0:rx1].astype(np.float32)
    ring_mask = np.ones((ry1 - ry0, rx1 - rx0), dtype=bool)
    ring_mask[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0] = False
    if int(ring_mask.sum()) >= 8:
        bg_rgb = np.median(ring_patch[ring_mask], axis=0)
    else:
        bg_rgb = np.array([45.0, 68.0, 82.0], dtype=np.float32)

    # Mask out central through-hole for flat washers/nuts so background inside the bore isn't averaged in
    spatial_mask = np.ones((ch, cw), dtype=bool)
    if kind in ("washer", "nut") and ch >= 8 and cw >= 8:
        yy, xx = np.ogrid[:ch, :cw]
        cy, cx = 0.5 * (ch - 1), 0.5 * (cw - 1)
        ry = max(1.5, 0.22 * ch)
        rx = max(1.5, 0.22 * cw)
        in_hole = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0
        spatial_mask &= ~in_hole

    diff = np.linalg.norm(crop - bg_rgb[None, None, :], axis=-1)
    fg = spatial_mask & (diff > 20.0)
    if int(fg.sum()) >= 10:
        return crop[fg]

    # Fallback (e.g., mid-grey stainless steel over grey fixture): central region, upper 65% contrast/luminance
    yy, xx = np.ogrid[:ch, :cw]
    cy, cx = 0.5 * (ch - 1), 0.5 * (cw - 1)
    ry, rx = max(1.5, 0.38 * ch), max(1.5, 0.38 * cw)
    core = spatial_mask & ((((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0)
    pts = crop[core] if int(core.sum()) >= 6 else crop.reshape(-1, 3)
    if len(pts) >= 8:
        d_pts = np.linalg.norm(pts - bg_rgb[None, :], axis=-1)
        thresh = float(np.percentile(d_pts, 35))
        keep = pts[d_pts >= thresh]
        if len(keep) >= 4:
            return keep
    return pts


def classify_material_from_pixels(fg_pixels: np.ndarray | None) -> tuple[str, str, dict[str, float]]:
    """Classify SimReady fastener material (`brass`, `nylon`, `aluminium`, `stainless steel`) from RGB pixels.

    Returns `(material_word, color_tag, material_scores_dict)`.
    """
    default_scores = {"brass": 0.25, "nylon": 0.25, "aluminium": 0.25, "stainless steel": 0.25}
    if fg_pixels is None or len(fg_pixels) == 0:
        return "aluminium", "silver", default_scores

    R = fg_pixels[:, 0]
    G = fg_pixels[:, 1]
    B = fg_pixels[:, 2]
    lum = 0.299 * R + 0.587 * G + 0.114 * B

    mean_r = float(np.mean(R))
    mean_g = float(np.mean(G))
    mean_b = float(np.mean(B))
    med_lum = float(np.median(lum))
    p95_lum = float(np.percentile(lum, 95))
    std_lum = float(np.std(lum))

    rb_diff = mean_r - mean_b
    gb_diff = mean_g - mean_b
    rb_ratio = mean_r / max(mean_b, 1.0)
    spec_span = p95_lum - med_lum

    # Soft scores in [0, 1] for relative ranking across multiple parts in the same scene/lighting
    brass_score = float(np.clip((rb_diff - 12.0) / 40.0, 0.0, 1.0) * np.clip((rb_ratio - 1.15) / 0.35, 0.0, 1.0))
    non_brass = 1.0 - brass_score

    # Stainless steel: darker diffuse body (rgba ~0.58 vs ~0.92) + metallic highlight contrast
    darkness = float(np.clip((168.0 - med_lum) / 65.0, 0.0, 1.0))
    metallic = float(np.clip((spec_span + 0.5 * std_lum - 10.0) / 38.0, 0.0, 1.0))
    stainless_score = non_brass * (0.78 * darkness + 0.22 * metallic)

    bright = float(np.clip((med_lum - 135.0) / 65.0, 0.0, 1.0))
    matte = float(np.clip((34.0 - (0.6 * spec_span + 0.6 * std_lum)) / 26.0, 0.0, 1.0))
    warm_neutral = float(np.clip((rb_diff + 4.0) / 14.0, 0.0, 1.0))
    cool_metallic = float(np.clip((-rb_diff + 6.0) / 14.0, 0.0, 1.0))

    nylon_score = non_brass * (0.45 * bright + 0.35 * matte + 0.20 * warm_neutral)
    alum_score = non_brass * (0.45 * bright + 0.30 * metallic + 0.25 * cool_metallic)

    scores = {
        "brass": round(brass_score, 4),
        "nylon": round(nylon_score, 4),
        "aluminium": round(alum_score, 4),
        "stainless steel": round(stainless_score, 4),
    }

    if rb_diff > 24.0 and gb_diff > 8.0 and rb_ratio > 1.26:
        return "brass", "gold", scores
    if med_lum < 152.0:
        return "stainless steel", "grey", scores
    if (rb_diff > 2.5 and std_lum < 28.0 and spec_span < 30.0) or (med_lum > 198.0 and spec_span < 22.0 and rb_diff >= -1.5):
        return "nylon", "white", scores
    return "aluminium", "silver", scores


# ---------------------------------------------------------------------------
# 2D Detection Structure & Multi-View 3D Scene Reconstruction
# ---------------------------------------------------------------------------

@dataclass
class Detection2D:
    view: str
    class_id: int
    class_name: str
    kind: str  # "bolt", "nut", "washer", "fixture"
    conf: float
    xyxy: tuple[float, float, float, float]
    kpts_xy: np.ndarray    # (6, 2) float64 pixel (u, v)
    kpts_conf: np.ndarray  # (6,) float64 in [0, 1]

    def usable_kpt(self, idx: int, width: int, height: int, min_conf: float = 0.30) -> bool:
        if idx < 0 or idx >= len(self.kpts_xy):
            return False
        if float(self.kpts_conf[idx]) < min_conf:
            return False
        u, v = float(self.kpts_xy[idx, 0]), float(self.kpts_xy[idx, 1])
        if u <= 0.5 and v <= 0.5:
            return False
        return 0.5 <= u <= width - 0.5 and 0.5 <= v <= height - 0.5

    def anchor_uv(self, width: int, height: int) -> tuple[float, float]:
        if self.usable_kpt(0, width, height, min_conf=0.25):
            return float(self.kpts_xy[0, 0]), float(self.kpts_xy[0, 1])
        return 0.5 * (self.xyxy[0] + self.xyxy[2]), 0.5 * (self.xyxy[1] + self.xyxy[3])


def _part_dims(class_name: str, kind: str) -> dict[str, float]:
    if class_name in FASTENER_GEOM:
        return FASTENER_GEOM[class_name]
    defaults = {
        "bolt": FASTENER_GEOM["bolt_m16x50"],
        "nut": FASTENER_GEOM["nut_m30"],
        "washer": FASTENER_GEOM.get("washer_m33", FASTENER_GEOM["washer_m30"]),
    }
    return defaults.get(kind, FASTENER_GEOM["bolt_m16x50"])


def _fit_rigid_2d(
    local_pts: np.ndarray,
    world_pts: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float, np.ndarray]:
    """Weighted 2D rigid transformation fit (closed-form SO(2), never reflects).

    Maps `local_pts (N, 2)` -> `world_pts (N, 2)` via `[fx, fy] + R(fyaw) @ local`.
    Returns `(fx, fy, fyaw, per_point_residuals)`.
    """
    w = np.maximum(np.asarray(weights, dtype=np.float64), 1e-6)
    w_sum = float(np.sum(w))
    if len(local_pts) == 1:
        fx = float(world_pts[0, 0] - local_pts[0, 0])
        fy = float(world_pts[0, 1] - local_pts[0, 1])
        return fx, fy, 0.0, np.zeros(1, dtype=np.float64)

    l_bar = np.sum(local_pts * w[:, None], axis=0) / w_sum
    w_bar = np.sum(world_pts * w[:, None], axis=0) / w_sum
    dl = local_pts - l_bar[None, :]
    dw = world_pts - w_bar[None, :]

    cs = float(np.sum(w * (dl[:, 0] * dw[:, 0] + dl[:, 1] * dw[:, 1])))
    sn = float(np.sum(w * (dl[:, 0] * dw[:, 1] - dl[:, 1] * dw[:, 0])))
    fyaw = math.atan2(sn, cs) if (cs * cs + sn * sn) > 1e-12 else 0.0
    c, s = math.cos(fyaw), math.sin(fyaw)
    fx = float(w_bar[0] - (c * l_bar[0] - s * l_bar[1]))
    fy = float(w_bar[1] - (s * l_bar[0] + c * l_bar[1]))

    pred_x = fx + c * local_pts[:, 0] - s * local_pts[:, 1]
    pred_y = fy + s * local_pts[:, 0] + c * local_pts[:, 1]
    res = np.hypot(pred_x - world_pts[:, 0], pred_y - world_pts[:, 1])
    return fx, fy, wrap_angle(fyaw), res


def reconstruct_fixture(
    fixture_dets: dict[str, Detection2D],
    cams: dict[str, Any],
    fixture_kpt_seats: dict[int, str] | None = None,
    fixture_seat_spec: dict[str, tuple[tuple[float, float], float]] | None = None,
) -> dict[str, Any]:
    """Reconstruct fixture pose `(fx, fy, fyaw)` by least squares on the 6 seat keypoints
    using `fixture_seat_spec`, then output all seat positions from the fitted pose."""
    from .vision import _seats_from_fixture_pose

    kpt_map = fixture_kpt_seats or {i: sid for i, sid in enumerate(DEFAULT_FIXTURE_KPTS)}
    seat_spec = fixture_seat_spec or {
        sid: (
            (float(SEAT_LOCAL_GEOM[sid]["local_xy"][0]), float(SEAT_LOCAL_GEOM[sid]["local_xy"][1])),
            float(SEAT_LOCAL_GEOM[sid].get("top_z", FIXTURE_BLOCK_TOP_Z)),
        )
        for sid in DEFAULT_FIXTURE_KPTS
        if sid in SEAT_LOCAL_GEOM
    }
    nom_fx, nom_fy = float(NOMINAL_FIXTURE_XY[0]), float(NOMINAL_FIXTURE_XY[1])

    if not fixture_dets:
        seats, seat_meta = _seats_from_fixture_pose(nom_fx, nom_fy, NOMINAL_FIXTURE_YAW)
        return {
            "detected": False,
            "fixture_position": [nom_fx, nom_fy, 0.0],
            "fixture_yaw": NOMINAL_FIXTURE_YAW,
            "fixture_confidence": 0.0,
            "seats": seats,
            "seat_meta": seat_meta,
            "observed_seats": {},
        }

    observed_seats: dict[str, np.ndarray] = {}
    observed_weights: dict[str, float] = {}

    for k_idx, seat_id in kpt_map.items():
        origins: list[np.ndarray] = []
        dirs: list[np.ndarray] = []
        ray_ws: list[float] = []
        best_single: tuple[float, Any, float, float] | None = None
        top_z_seat = float(seat_spec[seat_id][1]) if seat_id in seat_spec else FIXTURE_BLOCK_TOP_Z

        for vname, det in fixture_dets.items():
            cam = cams.get(vname)
            if cam is None or not det.usable_kpt(k_idx, cam.width, cam.height, min_conf=0.30):
                continue
            u, v = float(det.kpts_xy[k_idx, 0]), float(det.kpts_xy[k_idx, 1])
            kc = float(det.kpts_conf[k_idx]) * float(det.conf)
            vw = 2.5 if vname == "wrist" else 1.0
            o, d = cam.unproject_ray(u, v)
            origins.append(o)
            dirs.append(d)
            ray_ws.append(kc * vw)
            if best_single is None or (kc * vw) > best_single[0]:
                best_single = (kc * vw, cam, u, v)

        pt_3d: np.ndarray | None = None
        w_total = sum(ray_ws)
        if len(origins) >= 2:
            cand_pt, epi = triangulate_rays(origins, dirs, ray_ws)
            depths_ok = all(float(np.dot(cand_pt - o, d)) > 0.08 for o, d in zip(origins, dirs))
            if depths_ok and epi <= 0.070:
                pt_3d = cand_pt
        if pt_3d is None and best_single is not None:
            _, bcam, bu, bv = best_single
            pt_3d = bcam.intersect_z(bu, bv, top_z_seat)
            w_total *= 0.65

        if pt_3d is not None and 0.30 <= float(pt_3d[0]) <= 1.15 and -0.45 + _cell.LAYOUT_SHIFT[1] <= float(pt_3d[1]) <= 0.50:
            observed_seats[seat_id] = pt_3d
            observed_weights[seat_id] = max(0.05, w_total)

    local_list: list[tuple[float, float]] = []
    world_list: list[np.ndarray] = []
    weight_list: list[float] = []
    for sid, w_xyz in observed_seats.items():
        if sid in seat_spec:
            local_list.append(seat_spec[sid][0])
            world_list.append(w_xyz[:2])
            weight_list.append(observed_weights[sid])
        elif sid in SEAT_LOCAL_GEOM:
            local_list.append(SEAT_LOCAL_GEOM[sid]["local_xy"])
            world_list.append(w_xyz[:2])
            weight_list.append(observed_weights[sid])

    if local_list:
        L = np.asarray(local_list, dtype=np.float64)
        W = np.asarray(world_list, dtype=np.float64)
        wt = np.asarray(weight_list, dtype=np.float64)
        fx, fy, fyaw, res = _fit_rigid_2d(L, W, wt)
        # Single-pass outlier rejection if >= 3 keypoints and one is clearly off
        if len(L) >= 3 and float(np.max(res)) > 0.024:
            keep = res < max(0.024, 1.8 * float(np.median(res)))
            if int(np.sum(keep)) >= 2:
                fx, fy, fyaw, res = _fit_rigid_2d(L[keep], W[keep], wt[keep])
        if math.hypot(fx - nom_fx, fy - nom_fy) > 0.16 or _cell.yaw_off_nominal(fyaw) > math.radians(42.0):
            fx, fy, fyaw = nom_fx, nom_fy, NOMINAL_FIXTURE_YAW
    else:
        fx, fy, fyaw = nom_fx, nom_fy, NOMINAL_FIXTURE_YAW

    seats, seat_meta = _seats_from_fixture_pose(fx, fy, fyaw)
    # Apply any custom seat local_xy from fixture_seat_spec not already identical to SEAT_LOCAL_GEOM
    c_f, s_f = math.cos(fyaw), math.sin(fyaw)
    for sid, ((lx, ly), tz) in seat_spec.items():
        wx = fx + c_f * lx - s_f * ly
        wy = fy + s_f * lx + c_f * ly
        if sid in seats:
            seats[sid] = [float(wx), float(wy), float(seats[sid][2])]
        else:
            seats[sid] = [float(wx), float(wy), float(tz)]

    det_confs = [float(d.conf) for d in fixture_dets.values()]
    kpt_factor = min(1.0, len(observed_seats) / 3.0) if observed_seats else 0.4
    fix_conf = float(np.clip(max(det_confs) * (0.55 + 0.45 * kpt_factor), 0.0, 1.0))

    return {
        "detected": True,
        "fixture_position": [float(fx), float(fy), 0.0],
        "fixture_yaw": float(fyaw),
        "fixture_confidence": fix_conf,
        "seats": seats,
        "seat_meta": seat_meta,
        "observed_seats": {k: v.tolist() for k, v in observed_seats.items()},
    }


def _fixture_local_xy(world_xy: np.ndarray, fx: float, fy: float, fyaw: float) -> tuple[float, float]:
    c, s = math.cos(fyaw), math.sin(fyaw)
    dx = float(world_xy[0]) - fx
    dy = float(world_xy[1]) - fy
    return c * dx + s * dy, -s * dx + c * dy


def _single_view_surface_z(
    cam: Any,
    u: float,
    v: float,
    fx: float,
    fy: float,
    fyaw: float,
    seats: dict[str, list[float]],
    kind: str,
) -> float:
    """First opaque horizontal support surface (`FIXTURE_BLOCK_TOP_Z`, pocket floor, `TRAY_FLOOR_Z`, or table `0.0`)
    intersected by pixel ray `(u, v)`."""
    hit_top = cam.intersect_z(u, v, FIXTURE_BLOCK_TOP_Z)
    if hit_top is not None:
        lx, ly = _fixture_local_xy(hit_top[:2], fx, fy, fyaw)
        if -0.026 <= lx <= 0.125 and -0.115 <= ly <= 0.115:
            if kind == "nut":
                for sid in ("hex_m24", "hex_m30", "hex_m36"):
                    if sid in seats and math.hypot(float(hit_top[0]) - seats[sid][0], float(hit_top[1]) - seats[sid][1]) < 0.030:
                        return float(SEAT_LOCAL_GEOM[sid]["floor_z"])
            return FIXTURE_BLOCK_TOP_Z

    hit_tray = cam.intersect_z(u, v, TRAY_FLOOR_Z)
    if hit_tray is not None:
        lx, ly = _fixture_local_xy(hit_tray[:2], fx, fy, fyaw)
        if -0.121 <= lx <= -0.026 and -0.111 <= ly <= 0.111:
            return TRAY_FLOOR_Z
    return 0.0


def _reconstruct_bolt_single_view(
    det: Detection2D,
    cam: Any,
    z_base: float,
    fx: float,
    fy: float,
    fyaw: float,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Reconstruct `(p_k0, p_k1, p_k2)` of a single-view bolt (supporting lying, standing, and fixture-leaning)."""
    dims = _part_dims(det.class_name, "bolt")
    r_head = 0.5 * float(dims.get("across_flats", 0.024))
    r_shank = 0.5 * float(dims.get("diameter", 0.016))
    full_len = float(dims.get("length", 0.050)) + float(dims.get("head_height", 0.010))
    z_mid_lying = 0.5 * (r_head + r_shank)

    has_k0 = det.usable_kpt(0, cam.width, cam.height, 0.28)
    has_k1 = det.usable_kpt(1, cam.width, cam.height, 0.28)
    has_k2 = det.usable_kpt(2, cam.width, cam.height, 0.28)

    p_k0 = cam.intersect_z(float(det.kpts_xy[0, 0]), float(det.kpts_xy[0, 1]), z_base + z_mid_lying) if has_k0 else None
    if not (has_k1 and has_k2):
        u_a, v_a = det.anchor_uv(cam.width, cam.height)
        if p_k0 is None:
            p_k0 = cam.intersect_z(u_a, v_a, z_base + z_mid_lying)
        p_k1 = cam.intersect_z(float(det.kpts_xy[1, 0]), float(det.kpts_xy[1, 1]), z_base + r_head) if has_k1 else None
        p_k2 = cam.intersect_z(float(det.kpts_xy[2, 0]), float(det.kpts_xy[2, 1]), z_base + r_shank) if has_k2 else None
        return p_k0, p_k1, p_k2

    u1, v1 = float(det.kpts_xy[1, 0]), float(det.kpts_xy[1, 1])
    u2, v2 = float(det.kpts_xy[2, 0]), float(det.kpts_xy[2, 1])

    # Hypothesis 1: lying on z_base
    k1_ly = cam.intersect_z(u1, v1, z_base + r_head)
    k2_ly = cam.intersect_z(u2, v2, z_base + r_shank)
    candidates: list[tuple[float, np.ndarray | None, np.ndarray, np.ndarray]] = []
    if k1_ly is not None and k2_ly is not None:
        len_err = abs(float(np.linalg.norm(k1_ly - k2_ly)) - full_len)
        candidates.append((len_err, p_k0, k1_ly, k2_ly))

    # If head (k1) is distinctly lower in the image than tip (k2) (v1 > v2), test standing and leaning
    if v1 > v2 + 6.0:
        # Hypothesis 2: standing upright on head on z_base
        k1_st = cam.intersect_z(u1, v1, z_base + 0.001)
        k2_st = cam.intersect_z(u2, v2, z_base + full_len)
        if k1_st is not None and k2_st is not None:
            xy_mismatch = float(np.linalg.norm(k1_st[:2] - k2_st[:2]))
            if xy_mismatch < 0.025 and abs(u1 - u2) < 0.45 * abs(v1 - v2):
                k0_st = 0.5 * (k1_st + k2_st)
                candidates.append((xy_mismatch, k0_st, k1_st, k2_st))

        # Hypothesis 3: leaning against fixture -y face (z_base == 0.0)
        if z_base < 0.002 and k1_st is not None:
            k1_ln = cam.intersect_z(u1, v1, 0.005)
            if k1_ln is not None:
                # Lean face = the block face toward world -y: the fixture's +x face in the real cell. In the
                # frame turned by reach_yaw it is the "-y" face at depth 0.125 m, spanning +-0.115 m.
                fyaw_v = _cell.reach_yaw(fyaw)
                lx1, ly1 = _fixture_local_xy(k1_ln[:2], fx, fy, fyaw_v)
                if -0.13 <= lx1 <= 0.13 and -0.22 <= ly1 <= -0.115:
                    c_f, s_f = math.cos(fyaw_v), math.sin(fyaw_v)
                    wall_pt = np.array([fx + s_f * 0.125, fy - c_f * 0.125], dtype=np.float64)
                    wall_n = np.array([-s_f, c_f], dtype=np.float64)
                    o2, d2 = cam.unproject_ray(u2, v2)
                    k2_ln = intersect_vertical_plane(o2, d2, wall_pt, wall_n)
                    if k2_ln is not None and 0.020 <= float(k2_ln[2]) <= FIXTURE_BLOCK_TOP_Z + 0.005:
                        len_err_ln = abs(float(np.linalg.norm(k2_ln - k1_ln)) - full_len)
                        if len_err_ln < 0.032:
                            # Slight bonus when head sits right in the leaning strip and length matches wall contact
                            candidates.append((max(0.0, len_err_ln - 0.004), 0.5 * (k1_ln + k2_ln), k1_ln, k2_ln))

    if not candidates:
        return p_k0, k1_ly, k2_ly
    candidates.sort(key=lambda t: t[0])
    _, best_k0, best_k1, best_k2 = candidates[0]
    return best_k0, best_k1, best_k2


def _box_iou_2d(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0.0:
        return 0.0
    union = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return float(inter / union)


def _proj_nut_box_wh(kp: Any, pose_type: str, cam: Any, uc: float, vc: float, z_c: float, yaw: float) -> tuple[float, float] | None:
    from .vision import _part_pose_world

    pt = cam.intersect_z(uc, vc, z_c)
    if pt is None:
        return None
    _, R, _ = _part_pose_world(kp, pose_type, float(pt[0]), float(pt[1]), yaw)
    r = kp.across_flats / math.sqrt(3.0)
    hz = 0.5 * kp.thickness
    verts = [[r * math.cos(k * math.pi / 3.0 + math.pi / 6.0), r * math.sin(k * math.pi / 3.0 + math.pi / 6.0), z]
             for z in (-hz, hz) for k in range(6)]
    vw = pt[None, :] + (np.asarray(verts, dtype=np.float64) @ R.T)
    u, v, dep = cam.project(vw)
    if np.any(dep <= 0.05):
        return None
    return float(u.max() - u.min()), float(v.max() - v.min())


def _estimate_nut_pose_and_yaw(
    class_name: str,
    pos: np.ndarray,
    cmap: dict[str, Detection2D],
    cams: dict[str, Any],
    imgs_rgb: dict[str, np.ndarray | None],
    fyaw: float = 0.0,
) -> tuple[np.ndarray, float, str]:
    """Estimate `(axis, yaw, pose_hint)` for a nut using per-view ray-plane projected 2D box geometry
    (immune to side-camera pitch bias) + silhouette yaw search."""
    from .vision import KIT_PARTS, _part_pose_world, _rasterize_part_in_crop

    dims = _part_dims(class_name, "nut")
    af = float(dims.get("across_flats", 0.046))
    th = float(dims.get("thickness", 0.0256))
    on_edge_z_thresh = 0.5 * (0.5 * th + 0.5 * af)

    kp = KIT_PARTS.get(class_name) or KIT_PARTS.get("nut_m30")
    view_masks: list[tuple[Any, Detection2D, np.ndarray, tuple[int, int, int, int]]] = []
    for vname, det in cmap.items():
        im = imgs_rgb.get(vname)
        cam = cams.get(vname)
        if not isinstance(im, np.ndarray) or cam is None:
            continue
        H, W = im.shape[:2]
        x0, y0 = max(0, int(math.floor(det.xyxy[0])) - 6), max(0, int(math.floor(det.xyxy[1])) - 6)
        x1, y1 = min(W, int(math.ceil(det.xyxy[2])) + 6), min(H, int(math.ceil(det.xyxy[3])) + 6)
        if x1 - x0 < 6 or y1 - y0 < 6:
            continue
        patch = im[y0:y1, x0:x1].astype(np.float32)
        ring = np.ones((y1 - y0, x1 - x0), dtype=bool)
        ring[4:-4, 4:-4] = False
        bg = np.median(patch[ring], axis=0) if int(ring.sum()) >= 8 else np.array([45.0, 68.0, 82.0], dtype=np.float32)
        fg_full = np.zeros((H, W), dtype=bool)
        fg_full[y0:y1, x0:x1] = np.linalg.norm(patch - bg[None, None, :], axis=-1) > 20.0
        view_masks.append((cam, det, fg_full, (x0, y0, x1, y1)))

    is_on_table = float(pos[2]) < 0.070

    if kp is not None and is_on_table:
        v_dets = [(cams[v], d) for v, d in cmap.items() if v in cams and d.conf >= 0.30]
        if v_dets:
            ly_errs: list[float] = []
            for cam, d in v_dets:
                bw, bh = max(2.0, d.xyxy[2] - d.xyxy[0]), max(2.0, d.xyxy[3] - d.xyxy[1])
                uc, vc = 0.5 * (d.xyxy[0] + d.xyxy[2]), 0.5 * (d.xyxy[1] + d.xyxy[3])
                es = []
                for deg in range(0, 60, 15):
                    wh = _proj_nut_box_wh(kp, "lying", cam, uc, vc, 0.5 * th, math.radians(float(deg)))
                    if wh is not None:
                        es.append(abs(wh[0] - bw) / bw + 1.5 * abs(wh[1] - bh) / bh)
                if es:
                    ly_errs.append(min(es))

            best_ed_err, best_ed_yaw = 1e9, 0.0
            for deg in range(0, 180, 15):
                ya = math.radians(float(deg))
                es = []
                for cam, d in v_dets:
                    bw, bh = max(2.0, d.xyxy[2] - d.xyxy[0]), max(2.0, d.xyxy[3] - d.xyxy[1])
                    uc, vc = 0.5 * (d.xyxy[0] + d.xyxy[2]), 0.5 * (d.xyxy[1] + d.xyxy[3])
                    wh = _proj_nut_box_wh(kp, "on_edge", cam, uc, vc, 0.5 * af, ya)
                    if wh is not None:
                        es.append(abs(wh[0] - bw) / bw + 1.5 * abs(wh[1] - bh) / bh)
                if es and float(np.mean(es)) < best_ed_err:
                    best_ed_err = float(np.mean(es))
                    best_ed_yaw = ya

            ly_mean = float(np.mean(ly_errs)) if ly_errs else 9.0
            if best_ed_err < ly_mean:
                # Resolve +yaw vs -yaw sign (ambiguous in axis-aligned box wh) via multi-view silhouette IoU
                if view_masks:
                    best_ed_iou = -1.0
                    for deg in range(0, 180, 10):
                        ya = math.radians(float(deg))
                        _, R_ed, _ = _part_pose_world(kp, "on_edge", float(pos[0]), float(pos[1]), ya)
                        ious = []
                        for cam, det, fg_full, bbox in view_masks:
                            uc, vc = 0.5 * (det.xyxy[0] + det.xyxy[2]), 0.5 * (det.xyxy[1] + det.xyxy[3])
                            pt_v = cam.intersect_z(uc, vc, 0.5 * af)
                            p_try = pt_v if pt_v is not None else np.array([float(pos[0]), float(pos[1]), 0.5 * af], dtype=np.float64)
                            rast = _rasterize_part_in_crop(kp, p_try, R_ed, cam, bbox, pad=2)
                            if rast is None:
                                continue
                            (rx0, ry0, rx1, ry1), pred, _hole = rast
                            obs_c = fg_full[ry0:ry1, rx0:rx1]
                            ious.append(int((pred & obs_c).sum()) / max(int((pred | obs_c).sum()), 1))
                        if ious and float(np.mean(ious)) > best_ed_iou:
                            best_ed_iou, best_ed_yaw = float(np.mean(ious)), ya
                ax = np.array([math.cos(best_ed_yaw), math.sin(best_ed_yaw), 0.0], dtype=np.float64)
                return ax, wrap_angle(best_ed_yaw), "on_edge"

    if kp is not None and view_masks:
        best_ly_sc, best_ly_yaw = -1.0, wrap_angle(fyaw + 0.5 * math.pi)
        z_ly = float(pos[2]) if not is_on_table else 0.5 * th
        for deg in range(0, 60, 10):
            ya = math.radians(float(deg))
            _, R_ly, _ = _part_pose_world(kp, "lying", float(pos[0]), float(pos[1]), ya)
            ious = []
            for cam, det, fg_full, bbox in view_masks:
                uc, vc = 0.5 * (det.xyxy[0] + det.xyxy[2]), 0.5 * (det.xyxy[1] + det.xyxy[3])
                pt_v = cam.intersect_z(uc, vc, z_ly) if is_on_table else None
                p_try = pt_v if pt_v is not None else np.array([float(pos[0]), float(pos[1]), z_ly], dtype=np.float64)
                rast = _rasterize_part_in_crop(kp, p_try, R_ly, cam, bbox, pad=2)
                if rast is None:
                    continue
                (rx0, ry0, rx1, ry1), pred, _hole = rast
                obs_c = fg_full[ry0:ry1, rx0:rx1]
                ious.append(int((pred & obs_c).sum()) / max(int((pred | obs_c).sum()), 1))
            if ious and float(np.mean(ious)) > best_ly_sc:
                best_ly_sc, best_ly_yaw = float(np.mean(ious)), ya
        return np.array([0.0, 0.0, 1.0], dtype=np.float64), wrap_angle(best_ly_yaw), "lying"

    if is_on_table and float(pos[2]) > on_edge_z_thresh + 0.002:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64), 0.0, "on_edge"
    return np.array([0.0, 0.0, 1.0], dtype=np.float64), wrap_angle(fyaw + 0.5 * math.pi), "lying"


def _pose_and_axes_from_keypoints(
    class_name: str,
    kind: str,
    p_k0: np.ndarray | None,
    p_k1: np.ndarray | None,
    p_k2: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str] | None:
    """Compute `(position, axis_tip_to_head, axis_head_to_tip, yaw, pose_hint)` from 3D keypoints.
    Note: nut/washer k1/k2 are intentionally ignored due to rotational/hex symmetry."""
    dims = _part_dims(class_name, kind)

    if kind == "bolt":
        head_h = float(dims.get("head_height", 0.010))
        if p_k1 is not None and p_k2 is not None and float(np.linalg.norm(p_k1 - p_k2)) > 1e-4:
            pos = 0.5 * (p_k1 + p_k2)
            vec = p_k1 - p_k2  # tip (k2) -> head (k1), matching vision.py / planner.py body +X convention
            axis = vec / float(np.linalg.norm(vec))
        elif p_k0 is not None and p_k1 is not None and float(np.linalg.norm(p_k1 - p_k0)) > 1e-4:
            vec = p_k1 - p_k0
            axis = vec / float(np.linalg.norm(vec))
            pos = p_k0 + 0.5 * head_h * axis
        elif p_k0 is not None and p_k2 is not None and float(np.linalg.norm(p_k0 - p_k2)) > 1e-4:
            vec = p_k0 - p_k2
            axis = vec / float(np.linalg.norm(vec))
            pos = p_k0 + 0.5 * head_h * axis
        elif p_k0 is not None:
            pos = p_k0.copy()
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            return None

        head_to_tip = -axis
        yaw = math.atan2(float(axis[1]), float(axis[0])) if math.hypot(float(axis[0]), float(axis[1])) > 1e-5 else 0.0
        az = float(axis[2])
        if az < -0.82 or (abs(az) > 0.85 and float(pos[2]) > 0.032 and float(pos[2]) < 0.070):
            pose = "standing"
        elif az < -0.35:
            pose = "leaning"
        else:
            pose = "lying"
        return pos, axis, head_to_tip, wrap_angle(yaw), pose

    # Nut or washer: use k0 centre only (k1/k2 ignored due to symmetry)
    if p_k0 is None:
        return None
    pos = p_k0.copy()
    axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return pos, axis, axis, 0.0, "lying"


def reconstruct_parts(
    part_dets_by_view: dict[str, list[Detection2D]],
    cams: dict[str, Any],
    imgs_rgb: dict[str, np.ndarray | None],
    fixture_info: dict[str, Any],
) -> list[dict[str, Any]]:
    """Associate multi-view detections by class & 3D epipolar consistency and reconstruct 3D parts."""
    fx, fy = float(fixture_info["fixture_position"][0]), float(fixture_info["fixture_position"][1])
    fyaw = float(fixture_info["fixture_yaw"])
    seats = fixture_info["seats"]

    def _plausible_stereo_3d(pt3: np.ndarray) -> bool:
        if not (0.25 <= float(pt3[0]) <= 1.22 and -0.55 + _cell.LAYOUT_SHIFT[1] <= float(pt3[1]) <= 0.55):
            return False
        lx, ly = _fixture_local_xy(pt3[:2], fx, fy, fyaw)
        on_fixture = (-0.04 <= lx <= 0.14 and -0.13 <= ly <= 0.13)
        return (-0.05 <= float(pt3[2]) <= 0.35) if on_fixture else (-0.035 <= float(pt3[2]) <= 0.068)

    all_classes: set[str] = set()
    for dets in part_dets_by_view.values():
        for d in dets:
            all_classes.add(d.class_name)

    clusters: list[tuple[dict[str, Detection2D], np.ndarray | None]] = []

    for cls_name in sorted(all_classes):
        d_left = [d for d in part_dets_by_view.get("left", []) if d.class_name == cls_name and d.conf >= 0.25]
        d_right = [d for d in part_dets_by_view.get("right", []) if d.class_name == cls_name and d.conf >= 0.25]
        d_wrist = [d for d in part_dets_by_view.get("wrist", []) if d.class_name == cls_name and d.conf >= 0.25]

        used_l: set[int] = set()
        used_r: set[int] = set()
        used_w: set[int] = set()
        cls_clusters: list[tuple[dict[str, Detection2D], np.ndarray]] = []

        # 1. Stereo match left <-> right
        if "left" in cams and "right" in cams and d_left and d_right:
            cl, cr = cams["left"], cams["right"]
            cand_pairs: list[tuple[float, int, int, np.ndarray]] = []
            for il, dl in enumerate(d_left):
                ul, vl = dl.anchor_uv(cl.width, cl.height)
                ol, rl = cl.unproject_ray(ul, vl)
                for ir, dr in enumerate(d_right):
                    ur, vr = dr.anchor_uv(cr.width, cr.height)
                    or_, rr = cr.unproject_ray(ur, vr)
                    pt3, epi = triangulate_rays([ol, or_], [rl, rr])
                    if epi > 0.055:
                        continue
                    if float(np.dot(pt3 - ol, rl)) <= 0.08 or float(np.dot(pt3 - or_, rr)) <= 0.08:
                        continue
                    if not _plausible_stereo_3d(pt3):
                        continue
                    # A washer outside the fixture block cannot float > 6.5 cm in the air (rejects false cross-part stereo pairs).
                    if cls_name.startswith("washer") and float(pt3[2]) > 0.065:
                        lx_p, ly_p = _fixture_local_xy(pt3[:2], fx, fy, fyaw)
                        if not (-0.16 <= lx_p <= 0.16 and -0.15 <= ly_p <= 0.15):
                            continue
                    cand_pairs.append((epi, il, ir, pt3))
            cand_pairs.sort(key=lambda t: t[0])
            for _epi, il, ir, pt3 in cand_pairs:
                if il in used_l or ir in used_r:
                    continue
                used_l.add(il)
                used_r.add(ir)
                cls_clusters.append(({"left": d_left[il], "right": d_right[ir]}, pt3))

        # 2. Attach wrist detections to matched stereo clusters, or pair wrist with unmatched left/right
        if "wrist" in cams and d_wrist:
            cw = cams["wrist"]
            for iw, dw in enumerate(d_wrist):
                uw, vw = dw.anchor_uv(cw.width, cw.height)
                ow, rw = cw.unproject_ray(uw, vw)

                best_c: tuple[float, int] | None = None
                for ci, (cmap, cpt) in enumerate(cls_clusters):
                    if "wrist" in cmap:
                        continue
                    perp, along = ray_point_distance(ow, rw, cpt)
                    if along > 0.05 and perp <= 0.045:
                        if best_c is None or perp < best_c[0]:
                            best_c = (perp, ci)
                if best_c is not None:
                    used_w.add(iw)
                    cls_clusters[best_c[1]][0]["wrist"] = dw
                    continue

                # Try pairing wrist with an unmatched side view
                best_side: tuple[float, str, int, np.ndarray] | None = None
                for sname, s_list, s_used in (("right", d_right, used_r), ("left", d_left, used_l)):
                    if sname not in cams:
                        continue
                    cs = cams[sname]
                    for idx_s, ds in enumerate(s_list):
                        if idx_s in s_used:
                            continue
                        us, vs = ds.anchor_uv(cs.width, cs.height)
                        os_, rs = cs.unproject_ray(us, vs)
                        pt_sw, epi_sw = triangulate_rays([os_, ow], [rs, rw], [1.0, 2.5])
                        if epi_sw <= 0.045 and float(np.dot(pt_sw - ow, rw)) > 0.05 and float(np.dot(pt_sw - os_, rs)) > 0.08:
                            if _plausible_stereo_3d(pt_sw):
                                if best_side is None or epi_sw < best_side[0]:
                                    best_side = (epi_sw, sname, idx_s, pt_sw)
                if best_side is not None:
                    _, sname, idx_s, pt_sw = best_side
                    used_w.add(iw)
                    if sname == "right":
                        used_r.add(idx_s)
                        cls_clusters.append(({"right": d_right[idx_s], "wrist": dw}, pt_sw))
                    else:
                        used_l.add(idx_s)
                        cls_clusters.append(({"left": d_left[idx_s], "wrist": dw}, pt_sw))

        for cmap, cpt in cls_clusters:
            clusters.append((cmap, cpt))

        # 3. Remaining confident single-view detections (wrist first, then right, then left)
        for vname, d_list, used_set, min_c in (
            ("wrist", d_wrist, used_w, 0.35),
            ("right", d_right, used_r, 0.40),
            ("left", d_left, used_l, 0.40),
        ):
            if vname not in cams:
                continue
            for idx_d, dd in enumerate(d_list):
                if idx_d not in used_set and dd.conf >= min_c:
                    clusters.append(({vname: dd}, None))

    # Reconstruct 3D pose & material for each cluster
    parts_raw: list[dict[str, Any]] = []
    for cmap, anchor_3d in clusters:
        sample_det = next(iter(cmap.values()))
        cls_name = sample_det.class_name
        kind = sample_det.kind
        dims = _part_dims(cls_name, kind)

        kpts_3d: list[np.ndarray | None] = [None, None, None]
        if len(cmap) >= 2:
            for ki in range(3):
                origins, dirs, rws = [], [], []
                for vname, det in cmap.items():
                    cam = cams[vname]
                    if det.usable_kpt(ki, cam.width, cam.height, min_conf=0.28):
                        o, d = cam.unproject_ray(float(det.kpts_xy[ki, 0]), float(det.kpts_xy[ki, 1]))
                        vw = 2.5 if vname == "wrist" else 1.0
                        origins.append(o)
                        dirs.append(d)
                        rws.append(float(det.kpts_conf[ki]) * float(det.conf) * vw)
                if len(origins) >= 2:
                    pt_k, epi_k = triangulate_rays(origins, dirs, rws)
                    if epi_k <= 0.060 and all(float(np.dot(pt_k - o, d)) > 0.05 for o, d in zip(origins, dirs)):
                        kpts_3d[ki] = pt_k

            if kpts_3d[0] is None and anchor_3d is not None:
                kpts_3d[0] = anchor_3d.copy()

            # Fill any single-view keypoint in a multi-view cluster via height prior relative to k0/anchor
            z_ref = float(kpts_3d[0][2]) if kpts_3d[0] is not None else 0.012
            for ki in (1, 2):
                if kpts_3d[ki] is not None:
                    continue
                for vname in ("wrist", "right", "left"):
                    if vname not in cmap:
                        continue
                    det = cmap[vname]
                    cam = cams[vname]
                    if det.usable_kpt(ki, cam.width, cam.height, min_conf=0.32):
                        if kind == "bolt" and z_ref < 0.032:
                            zk = 0.5 * float(dims.get("across_flats", 0.024)) if ki == 1 else 0.5 * float(dims.get("diameter", 0.016))
                        else:
                            zk = z_ref
                        kpts_3d[ki] = cam.intersect_z(float(det.kpts_xy[ki, 0]), float(det.kpts_xy[ki, 1]), zk)
                        break
        else:
            vname, det = next(iter(cmap.items()))
            cam = cams[vname]
            ua, va = det.anchor_uv(cam.width, cam.height)
            z_surf = _single_view_surface_z(cam, ua, va, fx, fy, fyaw, seats, kind)
            if kind == "bolt":
                kpts_3d[0], kpts_3d[1], kpts_3d[2] = _reconstruct_bolt_single_view(det, cam, z_surf, fx, fy, fyaw)
            else:
                th = float(dims.get("thickness", 0.0256 if kind == "nut" else 0.005))
                af = float(dims.get("across_flats", 0.046))
                bw = max(1.0, det.xyxy[2] - det.xyxy[0])
                bh = max(1.0, det.xyxy[3] - det.xyxy[1])
                edge_like = (
                    kind == "nut"
                    and z_surf < 0.002
                    and (bh / bw) > 0.88
                    and det.usable_kpt(1, cam.width, cam.height, 0.30)
                    and det.usable_kpt(2, cam.width, cam.height, 0.30)
                    and max(abs(float(det.kpts_xy[1, 1] - det.kpts_xy[0, 1])), abs(float(det.kpts_xy[2, 1] - det.kpts_xy[0, 1])))
                    > 1.35 * max(abs(float(det.kpts_xy[1, 0] - det.kpts_xy[0, 0])), abs(float(det.kpts_xy[2, 0] - det.kpts_xy[0, 0])))
                )
                zc = z_surf + (0.5 * af if edge_like else 0.5 * th)
                kpts_3d[0] = (
                    cam.intersect_z(float(det.kpts_xy[0, 0]), float(det.kpts_xy[0, 1]), zc)
                    if det.usable_kpt(0, cam.width, cam.height, 0.28)
                    else cam.intersect_z(ua, va, zc)
                )
                for ki in (1, 2):
                    if det.usable_kpt(ki, cam.width, cam.height, 0.28):
                        kpts_3d[ki] = cam.intersect_z(float(det.kpts_xy[ki, 0]), float(det.kpts_xy[ki, 1]), zc)

        res_pose = _pose_and_axes_from_keypoints(cls_name, kind, kpts_3d[0], kpts_3d[1], kpts_3d[2])
        if res_pose is None:
            continue
        pos, axis_t2h, axis_h2t, yaw, pose_hint = res_pose
        if kind == "nut":
            axis_t2h, yaw, pose_hint = _estimate_nut_pose_and_yaw(cls_name, pos, cmap, cams, imgs_rgb, fyaw)
            axis_h2t = axis_t2h
            if len(cmap) == 1:
                vname_1, det_1 = next(iter(cmap.items()))
                cam_1 = cams[vname_1]
                ua_1, va_1 = det_1.anchor_uv(cam_1.width, cam_1.height)
                z_surf_1 = _single_view_surface_z(cam_1, ua_1, va_1, fx, fy, fyaw, seats, kind)
                if z_surf_1 < 0.002:
                    zc_1 = float(dims.get("across_flats", 0.046)) * 0.5 if pose_hint == "on_edge" else float(dims.get("thickness", 0.0256)) * 0.5
                    u0_1, v0_1 = (float(det_1.kpts_xy[0, 0]), float(det_1.kpts_xy[0, 1])) if det_1.usable_kpt(0, cam_1.width, cam_1.height, 0.28) else (ua_1, va_1)
                    p_refined = cam_1.intersect_z(u0_1, v0_1, zc_1)
                    if p_refined is not None:
                        pos = p_refined

        if not (0.25 <= float(pos[0]) <= 1.20 and -0.52 + _cell.LAYOUT_SHIFT[1] <= float(pos[1]) <= 0.50 and -0.04 <= float(pos[2]) <= 0.30):
            continue

        # Pool foreground pixels across views in this cluster for material/colour classification
        fg_chunks: list[np.ndarray] = []
        uv_by_view: dict[str, list[float]] = {}
        for vname, det in cmap.items():
            cam = cams[vname]
            u_v, v_v = det.anchor_uv(cam.width, cam.height)
            uv_by_view[vname] = [float(u_v), float(v_v)]
            im_v = imgs_rgb.get(vname)
            if isinstance(im_v, np.ndarray):
                px = estimate_crop_features(im_v, det.xyxy, kind)
                if px is not None and len(px) > 0:
                    fg_chunks.append(px)
        fg_all = np.concatenate(fg_chunks, axis=0) if fg_chunks else None
        material, color_tag, mat_scores = classify_material_from_pixels(fg_all)

        mean_conf = float(np.mean([d.conf for d in cmap.values()]))
        max_conf = float(np.max([d.conf for d in cmap.values()]))
        multi_bonus = 0.06 if len(cmap) >= 2 else -0.05
        conf = float(np.clip(0.65 * max_conf + 0.35 * mean_conf + multi_bonus, 0.05, 0.99))

        parts_raw.append(
            {
                "kind": kind,
                "class_name": cls_name,
                "size_class": cls_name,
                "position": [float(pos[0]), float(pos[1]), float(pos[2])],
                "axis": [float(axis_t2h[0]), float(axis_t2h[1]), float(axis_t2h[2])],
                "axis_head_to_tip": [float(axis_h2t[0]), float(axis_h2t[1]), float(axis_h2t[2])],
                "yaw": float(yaw),
                "confidence": conf,
                "pose": pose_hint,
                "material": material,
                "color": color_tag,
                "material_scores": mat_scores,
                "occluded": len(cmap) == 1,
                "views": sorted(cmap.keys()),
                "uv_by_view": uv_by_view,
                "_boxes_by_view": {vn: dt.xyxy for vn, dt in cmap.items()},
            }
        )

    # Deduplicate overlapping 3D detections, shared 2D boxes across size classes, and single-view ghosts
    parts_raw.sort(key=lambda p: (len(p["views"]), p["confidence"]), reverse=True)
    deduped: list[dict[str, Any]] = []
    claimed_boxes: dict[str, list[tuple[float, float, float, float]]] = {"left": [], "right": [], "wrist": []}
    for cand in parts_raw:
        cpos = np.asarray(cand["position"], dtype=np.float64)
        cboxes = cand.pop("_boxes_by_view", {})
        if any(float(np.linalg.norm(cpos - np.asarray(prev["position"], dtype=np.float64))) < 0.020 for prev in deduped):
            continue
        if cboxes and all(any(_box_iou_2d(bx, prev_bx) > 0.50 for prev_bx in claimed_boxes.get(vn, ())) for vn, bx in cboxes.items()):
            continue
        if len(cand["views"]) == 1:
            vn = cand["views"][0]
            cam_v = cams.get(vn)
            uv_v = cand["uv_by_view"].get(vn)
            if cam_v is not None and uv_v is not None:
                o_v, r_v = cam_v.unproject_ray(float(uv_v[0]), float(uv_v[1]))
                if any(
                    len(prev["views"]) >= 2
                    and prev["kind"] == cand["kind"]
                    and ray_point_distance(o_v, r_v, np.asarray(prev["position"], dtype=np.float64))[0] < 0.028
                    for prev in deduped
                ):
                    continue
        for vn, bx in cboxes.items():
            claimed_boxes.setdefault(vn, []).append(bx)
        deduped.append(cand)
    return deduped


def reconstruct_scene_3d(
    dets_by_view: dict[str, list[Detection2D]],
    cams: dict[str, Any],
    imgs_rgb: dict[str, np.ndarray | None] | None = None,
    fixture_kpt_seats: dict[int, str] | None = None,
    fixture_seat_spec: dict[str, tuple[tuple[float, float], float]] | None = None,
) -> dict[str, Any]:
    """Full 3D reconstruction from per-view `Detection2D` lists (used by runtime detector and unit tests)."""
    imgs = imgs_rgb or {}
    fixture_dets: dict[str, Detection2D] = {}
    part_dets: dict[str, list[Detection2D]] = {}

    for vname, dlist in dets_by_view.items():
        best_fix: Detection2D | None = None
        plist: list[Detection2D] = []
        for d in dlist:
            if d.kind == "fixture":
                if d.conf >= 0.25 and (best_fix is None or d.conf > best_fix.conf):
                    best_fix = d
            elif d.kind in ("bolt", "nut", "washer"):
                plist.append(d)
        if best_fix is not None:
            fixture_dets[vname] = best_fix
        part_dets[vname] = plist

    from .selfcal import calibrate_side_cameras

    kmap = fixture_kpt_seats or {i: sid for i, sid in enumerate(DEFAULT_FIXTURE_KPTS)}
    layout = {sid: tuple(SEAT_LOCAL_GEOM[sid]["local_xy"]) for sid in kmap.values() if sid in SEAT_LOCAL_GEOM}
    tops = {sid: float(SEAT_LOCAL_GEOM[sid].get("top_z", FIXTURE_BLOCK_TOP_Z)) for sid in layout}
    selfcal = calibrate_side_cameras(fixture_dets, cams, kmap, layout, tops)

    fix_info = reconstruct_fixture(fixture_dets, cams, fixture_kpt_seats, fixture_seat_spec)
    parts = reconstruct_parts(part_dets, cams, imgs, fix_info)

    return {
        "parts": parts,
        "seats": fix_info["seats"],
        "seat_meta": fix_info["seat_meta"],
        "fixture_position": fix_info["fixture_position"],
        "fixture_yaw": fix_info["fixture_yaw"],
        "fixture_confidence": fix_info["fixture_confidence"],
        "fixture_detected": fix_info["detected"],
        "selfcal": selfcal,
        "cameras": {k: c for k, c in cams.items() if k in ("left", "right", "wrist")},
    }


# ---------------------------------------------------------------------------
# Lazy-Loaded Ultralytics YOLO-Pose Wrapper
# ---------------------------------------------------------------------------

class _YOLOPartDetector:
    def __init__(self, weights_path: Path, classes_path: Path) -> None:
        _configure_ultralytics_offline()
        import torch
        from ultralytics import YOLO, settings

        try:
            settings.update(
                {
                    "sync": False,
                    "api_key": "",
                    "hub": False,
                    "wandb": False,
                    "tensorboard": False,
                    "mlflow": False,
                    "comet": False,
                    "clearml": False,
                    "neptune": False,
                    "dvc": False,
                    "raytune": False,
                }
            )
        except Exception:  # noqa: BLE001
            pass

        self.torch = torch
        self.weights_path = weights_path
        self.classes_path = classes_path
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(str(weights_path), task="pose")

        raw_spec = None
        if classes_path.is_file():
            try:
                raw_spec = json.loads(classes_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                raw_spec = None
        self.class_names, self.fixture_kpt_seats, self.fixture_seat_spec = parse_classes_spec(
            raw_spec, getattr(self.model, "names", None)
        )

    def predict_views(self, imgs_rgb: dict[str, np.ndarray | None]) -> dict[str, list[Detection2D]]:
        """Run one batched YOLO-pose call across available `('left', 'right', 'wrist')` RGB frames."""
        order: list[str] = []
        bgr_batch: list[np.ndarray] = []
        for vname in ("left", "right", "wrist"):
            im = imgs_rgb.get(vname)
            if isinstance(im, np.ndarray) and im.ndim == 3 and im.shape[2] >= 3:
                order.append(vname)
                # Ultralytics LoadPilAndNumpy assumes numpy inputs are BGR and flips to RGB in preprocess
                bgr_batch.append(np.ascontiguousarray(im[..., :3][..., ::-1], dtype=np.uint8))

        if not bgr_batch:
            return {}

        with self.torch.inference_mode():
            results = self.model.predict(
                bgr_batch,
                device=self.device,
                verbose=False,
                conf=0.20,
                iou=0.55,
            )

        out: dict[str, list[Detection2D]] = {}
        for vname, res in zip(order, results):
            dets: list[Detection2D] = []
            boxes = getattr(res, "boxes", None)
            kpts = getattr(res, "keypoints", None)
            if boxes is None or len(boxes) == 0:
                out[vname] = dets
                continue

            cls_arr = boxes.cls.detach().cpu().numpy().astype(int)
            conf_arr = boxes.conf.detach().cpu().numpy().astype(np.float64)
            xyxy_arr = boxes.xyxy.detach().cpu().numpy().astype(np.float64)

            if kpts is not None and kpts.xy is not None:
                kxy_arr = kpts.xy.detach().cpu().numpy().astype(np.float64)
                if kpts.conf is not None:
                    kconf_arr = kpts.conf.detach().cpu().numpy().astype(np.float64)
                else:
                    kconf_arr = ((kxy_arr[..., 0] > 0.5) | (kxy_arr[..., 1] > 0.5)).astype(np.float64)
            else:
                kxy_arr = np.zeros((len(cls_arr), 6, 2), dtype=np.float64)
                kconf_arr = np.zeros((len(cls_arr), 6), dtype=np.float64)

            for i in range(len(cls_arr)):
                cid = int(cls_arr[i])
                cname = self.class_names[cid] if 0 <= cid < len(self.class_names) else f"class_{cid}"
                kind = canonical_kind(cname)
                kxy = np.zeros((6, 2), dtype=np.float64)
                kcf = np.zeros(6, dtype=np.float64)
                n_k = min(6, kxy_arr.shape[1])
                if n_k > 0:
                    kxy[:n_k] = kxy_arr[i, :n_k, :2]
                    kcf[:n_k] = kconf_arr[i, :n_k]
                dets.append(
                    Detection2D(
                        view=vname,
                        class_id=cid,
                        class_name=cname,
                        kind=kind,
                        conf=float(conf_arr[i]),
                        xyxy=(float(xyxy_arr[i, 0]), float(xyxy_arr[i, 1]), float(xyxy_arr[i, 2]), float(xyxy_arr[i, 3])),
                        kpts_xy=kxy,
                        kpts_conf=kcf,
                    )
                )
            out[vname] = dets
        return out


_DETECTOR_LOCK = threading.Lock()
_CACHED_DETECTOR: _YOLOPartDetector | None = None
_CACHED_KEY: tuple[str, str] | None = None


def get_detector() -> _YOLOPartDetector | None:
    """Lazy-load and cache the YOLO-pose detector singleton if enabled and weights exist."""
    global _CACHED_DETECTOR, _CACHED_KEY
    if not detector_enabled():
        return None
    w_path = detector_weights_path()
    if not w_path.is_file():
        return None
    c_path = detector_classes_path(w_path)
    key = (str(w_path.resolve()), str(c_path.resolve()) if c_path.exists() else str(c_path))
    with _DETECTOR_LOCK:
        if _CACHED_DETECTOR is not None and _CACHED_KEY == key:
            return _CACHED_DETECTOR
        det = _YOLOPartDetector(w_path, c_path)
        _CACHED_DETECTOR = det
        _CACHED_KEY = key
        return det


def warmup_detector() -> bool:
    """Optional startup warmup when detector weights are present."""
    det = get_detector()
    if det is None:
        return False
    dummy = np.full((GEMINI_336_SIZE[1], GEMINI_336_SIZE[0], 3), 90, dtype=np.uint8)
    det.predict_views({"left": dummy, "right": dummy, "wrist": dummy})
    return True


def detect_scene(obs: dict[str, Any]) -> dict[str, Any] | None:
    """Run YOLO-pose on `obs['images']` (`left`, `right`, `wrist`) and reconstruct 3D parts + seats."""
    from .vision import _decode_image

    det = get_detector()
    if det is None:
        return None

    raw_images = obs.get("images") if isinstance(obs.get("images"), dict) else {}
    imgs_rgb = {
        k: _decode_image(raw_images.get(k))
        for k in ("left", "right", "wrist")
    }
    if not any(v is not None for v in imgs_rgb.values()):
        return None

    _cell.detect_real_cell_from_obs(obs)
    cams = build_nominal_cameras(obs, imgs_rgb)
    dets_2d = det.predict_views(imgs_rgb)
    return reconstruct_scene_3d(dets_2d, cams, imgs_rgb, det.fixture_kpt_seats, det.fixture_seat_spec)


def estimate_scene_from_detector(obs: dict[str, Any]) -> dict[str, Any] | None:
    """Run detector and convert confident results into the exact `vision.estimate_scene(obs)` schema.

    Returns `None` if detector is disabled, weights are absent, or detections are unconfident
    (neither confident fixture `>= 0.40` nor any confident part `>= 0.35`).
    """
    from .vision import (
        GOAL_CAMERA_FOVY,
        GOAL_CAMERA_POS,
        GOAL_CAMERA_TARGET,
        Camera,
        _assign_roles,
        _decode_image,
        _is_valid_goal_image,
        _look_mat,
    )

    det_out = detect_scene(obs)
    if det_out is None:
        return None

    fix_conf = float(det_out.get("fixture_confidence", 0.0))
    raw_parts = det_out.get("parts") or []
    has_confident_part = any(float(p.get("confidence", 0.0)) >= 0.35 for p in raw_parts)
    if fix_conf < 0.40 and not has_confident_part:
        return None

    # Convert detector part records (kind="bolt", class_name="bolt_m16x50") into vision.estimate_scene schema
    # (where item["kind"] is the kit model name e.g. "bolt_m16x50" and item["shape"] is "bolt"/"nut"/"washer").
    parts_out: list[dict[str, Any]] = []
    for p in raw_parts:
        parts_out.append(
            {
                "kind": str(p["class_name"]),
                "shape": str(p["kind"]),
                "color": str(p.get("color", "silver")),
                "material": str(p.get("material", "aluminium")),
                "material_scores": dict(p.get("material_scores") or {}),
                "pose": str(p.get("pose", "lying")),
                "position": list(p["position"]),
                "axis": list(p["axis"]),
                "axis_head_to_tip": list(p.get("axis_head_to_tip", p["axis"])),
                "yaw": float(p["yaw"]),
                "confidence": float(p["confidence"]),
                "occluded": bool(p.get("occluded", False)),
                "views": list(p.get("views", [])),
                "uv_by_view": dict(p.get("uv_by_view") or {}),
            }
        )

    seats = det_out["seats"]
    instruction = str(obs.get("instruction") or "")
    img_goal = _decode_image(obs.get("goal"))
    goal_cam = None
    use_goal = _is_valid_goal_image(img_goal)
    if use_goal and img_goal is not None:
        hg, wg = img_goal.shape[:2]
        goal_cam = Camera(
            "goal",
            np.asarray(GOAL_CAMERA_POS, dtype=np.float64),
            _look_mat(GOAL_CAMERA_POS, GOAL_CAMERA_TARGET),
            GOAL_CAMERA_FOVY,
            wg,
            hg,
        )

    _assign_roles(parts_out, seats, instruction, img_goal if use_goal else None, goal_cam)
    parts_out.sort(key=lambda item: (0 if item.get("role") == "target" else 1, -float(item["confidence"])))

    cams = det_out.get("cameras") or {}
    return {
        "parts": parts_out,
        "seats": seats,
        "fixture_yaw": float(det_out["fixture_yaw"]),
        "fixture_position": list(det_out["fixture_position"]),
        "fixture_confidence": fix_conf,
        "seat_meta": det_out["seat_meta"],
        "selfcal": det_out.get("selfcal") or {},
        "camera_calibration": {},
        "cameras": {k: c for k, c in cams.items() if k in ("left", "right")},
        "perception_source": "detector",
    }
