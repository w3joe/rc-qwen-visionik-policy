"""Vision scene interface, task instruction parsing, fixture seat completion,
confidence-based target selection, and temporal filtering.

Uses ONLY public observations (`obs`) and calls `vision.estimate_scene(obs)`
provided by the companion vision module. Never reads simulator internals.
"""
from __future__ import annotations

import importlib
import importlib.util
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .kinematics import wrap_angle

# Public fixture v3 geometry (from assets/fixture/fixture_v3.json, converted to metres)
NOMINAL_FIXTURE_XY = np.array([0.70, 0.10], dtype=np.float64)
FIXTURE_HALF_FOOTPRINT = (0.125, 0.115)
FIXTURE_BLOCK_TOP_Z = 0.075
FIXTURE_BLOCK_Y_MIN = -0.115
TRAY_FLOOR_Z = 0.004
TRAY_WALL_TOP_Z = 0.022
TRAY_HALF_XY = (0.0475, 0.111)

# Local (x, y) offsets in fixture frame (metres) and z heights for all standard seats
SEAT_LOCAL_GEOM: dict[str, dict[str, Any]] = {
    "tray": {
        "kind": "tray",
        "local_xy": (-0.0735, 0.0),
        "floor_z": 0.004,
        "top_z": 0.022,
        "half": TRAY_HALF_XY,
    },
    "bore_m24": {
        "kind": "hole_vertical",
        "local_xy": (0.0072, 0.0860),
        "floor_z": 0.020,
        "top_z": 0.075,
        "diameter": 0.028,
        "chamfer": 0.008,
    },
    "bore_m16": {
        "kind": "hole_vertical",
        "local_xy": (0.0072, 0.0035),
        "floor_z": 0.020,
        "top_z": 0.075,
        "diameter": 0.020,
        "chamfer": 0.008,
    },
    "bore_m30": {
        "kind": "hole_vertical",
        "local_xy": (0.0072, -0.0825),
        "floor_z": 0.010,
        "top_z": 0.075,
        "diameter": 0.035,
        "chamfer": 0.008,
    },
    "hex_m30": {
        "kind": "hex_pocket",
        "local_xy": (0.07735, 0.0775),
        "floor_z": 0.062,
        "top_z": 0.075,
        "across_flats": 0.051,
        "chamfer": 0.004,
    },
    "hex_m36": {
        "kind": "hex_pocket",
        "local_xy": (0.07735, -0.0050),
        "floor_z": 0.060,
        "top_z": 0.075,
        "across_flats": 0.060,
        "chamfer": 0.004,
    },
    "hex_m24": {
        "kind": "hex_pocket",
        "local_xy": (0.07735, -0.0825),
        "floor_z": 0.065,
        "top_z": 0.075,
        "across_flats": 0.041,
        "chamfer": 0.004,
    },
}

# Public fastener kit dimensions (from assets/fasteners/manifest.json)
FASTENER_GEOM: dict[str, dict[str, float]] = {
    "bolt_m16x50": {
        "diameter": 0.016,
        "across_flats": 0.024,
        "length": 0.050,
        "head_height": 0.010,
        "head_local_x": 0.025,
        "under_head_x": 0.020,
        "tip_x": -0.030,
        "lying_radius": 0.012,
    },
    "bolt_m24x50": {
        "diameter": 0.024,
        "across_flats": 0.036,
        "length": 0.050,
        "head_height": 0.015,
        "head_local_x": 0.025,
        "under_head_x": 0.0175,
        "tip_x": -0.0325,
        "lying_radius": 0.018,
    },
    "bolt_m30x60": {
        "diameter": 0.030,
        "across_flats": 0.046,
        "length": 0.060,
        "head_height": 0.0187,
        "head_local_x": 0.030,
        "under_head_x": 0.02065,
        "tip_x": -0.03935,
        "lying_radius": 0.023,
    },
    "nut_m24": {"across_flats": 0.036, "thickness": 0.0215, "hole": 0.024, "lying_radius": 0.01075},
    "nut_m30": {"across_flats": 0.046, "thickness": 0.0256, "hole": 0.030, "lying_radius": 0.0128},
    "nut_m36": {"across_flats": 0.055, "thickness": 0.0310, "hole": 0.036, "lying_radius": 0.0155},
    "washer_m30": {"outer_diameter": 0.062, "inner_diameter": 0.035, "thickness": 0.006, "lying_radius": 0.003},
}


def load_vision_estimator() -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Import and return `estimate_scene` from the companion `vision.py` module.

    Supports both `policy/vision.py` (relative package) and top-level `vision.py`
    under the policy repository root so package imports stay stable in all runners.
    """
    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parent

    # 1. Try policy.vision (if vision.py is placed inside policy/)
    try:
        mod = importlib.import_module("policy.vision")
        if hasattr(mod, "estimate_scene"):
            return mod.estimate_scene
    except ImportError:
        pass

    # 2. Try top-level `vision` (if vision.py is placed at repo root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        mod = importlib.import_module("vision")
        if hasattr(mod, "estimate_scene"):
            return mod.estimate_scene
    except ImportError:
        pass

    # 3. Direct file load fallback
    for candidate in (repo_root / "vision.py", pkg_dir / "vision.py"):
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location("vision", str(candidate))
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                sys.modules["vision"] = mod
                spec.loader.exec_module(mod)
                if hasattr(mod, "estimate_scene"):
                    return mod.estimate_scene

    raise RuntimeError(
        "vision.estimate_scene not found: expected vision.py at repository root or policy/vision.py"
    )


def _canonical_kind(raw_kind: str) -> str:
    k = str(raw_kind or "").strip().lower()
    if "bolt" in k:
        return "bolt"
    if "washer" in k:
        return "washer"
    if "nut" in k:
        return "nut"
    return k or "unknown"


@dataclass
class PartEstimate:
    kind: str  # "bolt", "nut", "washer", or "unknown"
    raw_kind: str
    position: np.ndarray  # (3,) world metres
    axis: np.ndarray  # (3,) unit vector
    yaw: float  # radians
    confidence: float  # [0, 1]
    role: str | None = None
    pose_hint: str = "lying"  # "lying", "standing", "leaning", "on_edge"
    axis_directed: bool = False  # True if axis sign (tip->head) is explicitly provided
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneEstimate:
    parts: list[PartEstimate]
    seats: dict[str, np.ndarray]  # seat_id -> (3,) world xyz
    fixture_yaw: float
    fixture_xy: np.ndarray  # (2,) world xy of fixture origin
    fixture_confidence: float = 1.0
    raw: dict[str, Any] = field(default_factory=dict)

    def fixture_to_world(self, local_xy: tuple[float, float]) -> np.ndarray:
        c, s = math.cos(self.fixture_yaw), math.sin(self.fixture_yaw)
        lx, ly = float(local_xy[0]), float(local_xy[1])
        return np.array(
            [self.fixture_xy[0] + c * lx - s * ly, self.fixture_xy[1] + s * lx + c * ly],
            dtype=np.float64,
        )

    def fixture_to_local(self, world_xy: np.ndarray) -> np.ndarray:
        c, s = math.cos(self.fixture_yaw), math.sin(self.fixture_yaw)
        dx = float(world_xy[0]) - float(self.fixture_xy[0])
        dy = float(world_xy[1]) - float(self.fixture_xy[1])
        return np.array([c * dx + s * dy, -s * dx + c * dy], dtype=np.float64)

    def fixture_distance(self, world_xy: np.ndarray) -> float:
        lx, ly = self.fixture_to_local(world_xy)
        hx, hy = FIXTURE_HALF_FOOTPRINT
        return float(math.hypot(max(abs(lx) - hx, 0.0), max(abs(ly) - hy, 0.0)))


def _infer_pose_hint(kind: str, pos: np.ndarray, axis: np.ndarray, extra: dict[str, Any]) -> str:
    explicit = extra.get("pose") or extra.get("start_pose") or extra.get("state")
    if isinstance(explicit, str):
        e = explicit.strip().lower()
        if e in ("lying", "standing", "leaning", "on_edge", "flat"):
            return "lying" if e == "flat" else e
    if kind == "washer":
        return "lying"
    az = float(axis[2])
    if kind == "bolt":
        # Standing on head: axis points down (-z) or nearly vertical with center z >= 0.032
        if az < -0.82 or (abs(az) > 0.85 and float(pos[2]) > 0.032):
            return "standing"
        # Leaning against fixture -y face: tilted 25-50 deg from vertical (|az| ~ 0.55..0.88)
        if az < -0.35 or (abs(az) > 0.38 and float(pos[2]) > 0.025):
            return "leaning"
        return "lying"
    if kind == "nut":
        # Lying M30 nut has center z ~ 0.0128 and axis ~ [0, 0, +-1]; on_edge has z ~ 0.023 and |az| < 0.6
        if abs(az) < 0.65:
            return "on_edge"
        if float(pos[2]) > 0.0195 and abs(az) < 0.88:
            return "on_edge"
        return "lying"
    return "lying"


def _safe_vec3(raw: Any, default: np.ndarray | None = None) -> np.ndarray | None:
    if raw is None:
        return default.copy() if default is not None else None
    try:
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        if arr.size >= 3 and np.all(np.isfinite(arr[:3])):
            return arr[:3].copy()
        if arr.size == 2 and np.all(np.isfinite(arr[:2])) and default is not None:
            return np.array([arr[0], arr[1], default[2]], dtype=np.float64)
    except (TypeError, ValueError):
        pass
    return default.copy() if default is not None else None


_SEAT_ALIASES: dict[str, str] = {
    "tray": "tray",
    "fixture_tray": "tray",
    "bore_m16": "bore_m16",
    "hole_m16": "bore_m16",
    "m16_hole": "bore_m16",
    "m16_bore": "bore_m16",
    "small_hole": "bore_m16",
    "bore_m24": "bore_m24",
    "hole_m24": "bore_m24",
    "m24_hole": "bore_m24",
    "m24_bore": "bore_m24",
    "bore_m30": "bore_m30",
    "hole_m30": "bore_m30",
    "m30_hole": "bore_m30",
    "m30_bore": "bore_m30",
    "hex_m30": "hex_m30",
    "pocket_m30": "hex_m30",
    "m30_pocket": "hex_m30",
    "hex_pocket_m30": "hex_m30",
    "hex_m36": "hex_m36",
    "pocket_m36": "hex_m36",
    "m36_pocket": "hex_m36",
    "hex_m24": "hex_m24",
    "pocket_m24": "hex_m24",
    "m24_pocket": "hex_m24",
}


def normalize_scene(raw: Any) -> SceneEstimate:
    """Convert raw output from `vision.estimate_scene(obs)` into a clean `SceneEstimate`."""
    if not isinstance(raw, dict):
        raw = {}

    fixture_yaw = raw.get("fixture_yaw", 0.0)
    try:
        fixture_yaw = float(fixture_yaw)
        if not math.isfinite(fixture_yaw):
            fixture_yaw = 0.0
    except (TypeError, ValueError):
        fixture_yaw = 0.0
    fixture_yaw = wrap_angle(fixture_yaw)

    raw_seats = raw.get("seats") if isinstance(raw.get("seats"), dict) else {}
    seats_in: dict[str, np.ndarray] = {}
    for k, v in raw_seats.items():
        canon_k = _SEAT_ALIASES.get(str(k).strip().lower(), str(k).strip())
        if canon_k in SEAT_LOCAL_GEOM:
            default_z = float(SEAT_LOCAL_GEOM[canon_k].get("top_z", 0.075))
            vec = _safe_vec3(v, default=np.array([0.0, 0.0, default_z]))
            if vec is not None:
                seats_in[canon_k] = vec
        else:
            vec = _safe_vec3(v)
            if vec is not None:
                seats_in[canon_k] = vec

    # Infer fixture origin xy from any reported seats + fixture_yaw
    c, s = math.cos(fixture_yaw), math.sin(fixture_yaw)
    origins: list[np.ndarray] = []
    for sid, world_xyz in seats_in.items():
        if sid in SEAT_LOCAL_GEOM:
            lx, ly = SEAT_LOCAL_GEOM[sid]["local_xy"]
            rx = c * lx - s * ly
            ry = s * lx + c * ly
            origins.append(np.array([world_xyz[0] - rx, world_xyz[1] - ry], dtype=np.float64))

    if origins:
        fixture_xy = np.median(np.vstack(origins), axis=0)
        # Sanity clamp to plausible fixture table region
        if float(np.linalg.norm(fixture_xy - NOMINAL_FIXTURE_XY)) > 0.12:
            fixture_xy = NOMINAL_FIXTURE_XY.copy()
    else:
        fixture_xy = NOMINAL_FIXTURE_XY.copy()

    # Complete all standard seats from fixture_xy + fixture_yaw while preserving explicit seats
    seats: dict[str, np.ndarray] = {}
    for sid, geom in SEAT_LOCAL_GEOM.items():
        lx, ly = geom["local_xy"]
        wx = float(fixture_xy[0] + c * lx - s * ly)
        wy = float(fixture_xy[1] + s * lx + c * ly)
        wz = float(geom["floor_z"] if geom["kind"] == "tray" else geom["top_z"])
        if sid in seats_in:
            # Trust vision seat xy if within 8 cm of nominal fixture range; keep canonical z if vision z is 0
            sv = seats_in[sid].copy()
            if sv[2] <= 1e-4 and geom["kind"] != "tray":
                sv[2] = wz
            seats[sid] = sv
        else:
            seats[sid] = np.array([wx, wy, wz], dtype=np.float64)
    for sid, sv in seats_in.items():
        if sid not in seats:
            seats[sid] = sv

    # Normalize detected parts
    parts: list[PartEstimate] = []
    raw_parts = raw.get("parts") if isinstance(raw.get("parts"), (list, tuple)) else []
    for item in raw_parts:
        if not isinstance(item, dict):
            continue
        pos = _safe_vec3(item.get("position"))
        if pos is None:
            continue
        raw_kind = str(item.get("kind", "unknown"))
        kind = _canonical_kind(raw_kind)

        try:
            conf = float(item.get("confidence", 0.5))
            if not math.isfinite(conf):
                conf = 0.0
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))

        yaw_raw = item.get("yaw", None)
        yaw_val: float | None = None
        if yaw_raw is not None:
            try:
                y = float(yaw_raw)
                if math.isfinite(y):
                    yaw_val = wrap_angle(y)
            except (TypeError, ValueError):
                pass

        axis_raw = _safe_vec3(item.get("axis"))
        axis_directed = False
        if axis_raw is not None and float(np.linalg.norm(axis_raw)) > 1e-6:
            axis = axis_raw / float(np.linalg.norm(axis_raw))
            axis_directed = True
        else:
            if kind == "bolt":
                ang = yaw_val if yaw_val is not None else 0.0
                axis = np.array([math.cos(ang), math.sin(ang), 0.0], dtype=np.float64)
            else:
                axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        if yaw_val is None:
            if math.hypot(float(axis[0]), float(axis[1])) > 1e-5:
                yaw_val = math.atan2(float(axis[1]), float(axis[0]))
            else:
                yaw_val = 0.0

        role = item.get("role")
        role_str = str(role).strip().lower() if isinstance(role, str) and role.strip() else None

        extra = {
            k: v
            for k, v in item.items()
            if k not in {"kind", "position", "axis", "yaw", "confidence", "role"}
        }
        pose_hint = _infer_pose_hint(kind, pos, axis, extra)
        parts.append(
            PartEstimate(
                kind=kind,
                raw_kind=raw_kind,
                position=pos,
                axis=axis,
                yaw=yaw_val,
                confidence=conf,
                role=role_str,
                pose_hint=pose_hint,
                axis_directed=axis_directed,
                extra=extra,
            )
        )

    fixture_conf = float(raw.get("fixture_confidence", 1.0 if raw_seats else 0.5))
    return SceneEstimate(
        parts=parts,
        seats=seats,
        fixture_yaw=fixture_yaw,
        fixture_xy=fixture_xy,
        fixture_confidence=fixture_conf,
        raw=raw,
    )


@dataclass(frozen=True)
class GoalSpec:
    """One pick-and-place subgoal derived purely from public task_id and instruction."""

    goal_id: str
    part_kind: str  # "bolt", "washer", "nut"
    part_model: str  # e.g. "bolt_m30x60", "washer_m30", "bolt_m16x50", "nut_m30"
    seat_id: str  # "tray", "bore_m16", "hex_m30", etc.
    selector: str = "default"  # "default", "smaller_bolt", "nearer_tray_bolt", "nut"
    colour_hint: str | None = None


def parse_task_goals(obs: dict[str, Any]) -> list[GoalSpec]:
    """Infer goal list strictly from public `task_id` and `instruction`."""
    task_id = str(obs.get("task_id") or "").strip().lower()
    instr = str(obs.get("instruction") or "").strip().lower()

    if task_id == "bolt_to_tray" or ("tray" in instr and "pocket" not in instr and "hole" not in instr):
        if "washer" in instr:
            return [
                GoalSpec(
                    goal_id="washer_to_tray",
                    part_kind="washer",
                    part_model="washer_m30",
                    seat_id="tray",
                    selector="washer",
                    colour_hint="white" if "white" in instr else None,
                )
            ]
        return [
            GoalSpec(
                goal_id="bolt_to_tray",
                part_kind="bolt",
                part_model="bolt_m30x60",
                seat_id="tray",
                selector="bolt",
                colour_hint="gold" if "gold" in instr else None,
            )
        ]

    if task_id == "bolt_and_nut_to_fixture" or ("hole" in instr and "pocket" in instr):
        bolt_sel = "nearer_tray_bolt" if "nearer the tray" in instr else "smaller_bolt"
        nut_colour = "gold" if "gold nut" in instr else None
        return [
            GoalSpec(
                goal_id="small_bolt_to_bore_m16",
                part_kind="bolt",
                part_model="bolt_m16x50",
                seat_id="bore_m16",
                selector=bolt_sel,
            ),
            GoalSpec(
                goal_id="nut_to_hex_m30",
                part_kind="nut",
                part_model="nut_m30",
                seat_id="hex_m30",
                selector="nut",
                colour_hint=nut_colour,
            ),
        ]

    # Best-effort fallback for any other task instruction
    goals: list[GoalSpec] = []
    if "washer" in instr:
        seat = "tray" if "tray" in instr else "bore_m30"
        goals.append(GoalSpec("washer_goal", "washer", "washer_m30", seat, "washer"))
    if "bolt" in instr:
        if "tray" in instr:
            goals.append(GoalSpec("bolt_tray_goal", "bolt", "bolt_m30x60", "tray", "bolt"))
        else:
            sel = "smaller_bolt" if "smaller" in instr else "bolt"
            model = "bolt_m16x50" if "smaller" in instr else "bolt_m30x60"
            seat = "bore_m16" if "smaller" in instr else "bore_m30"
            goals.append(GoalSpec("bolt_hole_goal", "bolt", model, seat, sel))
    if "nut" in instr:
        seat = "tray" if "tray" in instr and not goals else "hex_m30"
        goals.append(GoalSpec("nut_goal", "nut", "nut_m30", seat, "nut"))
    if not goals:
        goals.append(GoalSpec("default_bolt", "bolt", "bolt_m30x60", "tray", "bolt"))
    return goals


def is_part_in_seat(part: PartEstimate, seat_id: str, scene: SceneEstimate) -> bool:
    """Check (from vision estimate alone) whether a part already lies in its target seat."""
    seat_xyz = scene.seats.get(seat_id)
    if seat_xyz is None:
        return False
    geom = SEAT_LOCAL_GEOM.get(seat_id, {})
    kind = geom.get("kind", "")
    if kind == "tray":
        # Check in fixture-local coordinates relative to tray center
        dx = float(part.position[0] - seat_xyz[0])
        dy = float(part.position[1] - seat_xyz[1])
        c, s = math.cos(scene.fixture_yaw), math.sin(scene.fixture_yaw)
        u = c * dx + s * dy
        v = -s * dx + c * dy
        hx, hy = geom.get("half", TRAY_HALF_XY)
        return abs(u) <= hx - 0.005 and abs(v) <= hy - 0.005 and float(part.position[2]) < 0.06
    if kind == "hole_vertical":
        d_xy = float(np.linalg.norm(part.position[:2] - seat_xyz[:2]))
        return d_xy < 0.018 and float(part.position[2]) > 0.035 and abs(float(part.axis[2])) > 0.85
    if kind == "hex_pocket":
        d_xy = float(np.linalg.norm(part.position[:2] - seat_xyz[:2]))
        return d_xy < 0.018 and 0.065 <= float(part.position[2]) <= 0.0775
    return False


def _in_reachable_table(pos: np.ndarray) -> bool:
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    r = math.hypot(x, y)
    return 0.30 <= x <= 1.15 and -0.50 <= y <= 0.45 and -0.04 <= z <= 0.25 and r <= 1.08


def select_target_part(
    scene: SceneEstimate,
    goal: GoalSpec,
    *,
    min_confidence: float = 0.25,
    ignore_seated: bool = True,
    exclude_positions: list[np.ndarray] | None = None,
) -> PartEstimate | None:
    """Confidence-based selection of the best part matching `goal`."""
    def _is_forbidden(p: PartEstimate) -> bool:
        role = (p.role or "").lower()
        rk = p.raw_kind.lower()
        if goal.selector in ("smaller_bolt", "nearer_tray_bolt"):
            # Hard-reject explicitly classified decoys or wrong bolt sizes (e.g. M24 decoy left on table after M16 is seated)
            if role in ("large_bolt", "decoy", "decoy_bolt", "bolt_m24"):
                return True
            if "m24" in rk or "m30" in rk or "large" in rk:
                return True
        elif goal.selector == "nut":
            if role in ("decoy", "clutter") and goal.part_model == "nut_m30":
                return True
        if exclude_positions:
            for ex in exclude_positions:
                if float(np.linalg.norm(p.position[:2] - ex[:2])) < 0.045:
                    return True
        return False

    candidates = [
        p
        for p in scene.parts
        if p.kind == goal.part_kind
        and p.confidence >= min_confidence
        and _in_reachable_table(p.position)
        and not _is_forbidden(p)
        and (not ignore_seated or not is_part_in_seat(p, goal.seat_id, scene))
    ]
    if not candidates:
        # Relax confidence floor slightly (still >= 0.15 and still enforcing forbidden/decoy checks!)
        candidates = [
            p
            for p in scene.parts
            if p.kind == goal.part_kind
            and p.confidence >= 0.15
            and _in_reachable_table(p.position)
            and not _is_forbidden(p)
            and (not ignore_seated or not is_part_in_seat(p, goal.seat_id, scene))
        ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    tray_xy = scene.seats.get("tray", np.array([0.6265, 0.10, 0.004]))[:2]

    def score_candidate(p: PartEstimate) -> float:
        s = 2.0 * p.confidence
        role = (p.role or "").lower()
        rk = p.raw_kind.lower()
        colour = str(p.extra.get("colour") or p.extra.get("color") or "").lower()
        target_seat = str(p.extra.get("target_seat") or "").lower()

        if target_seat and target_seat == goal.seat_id:
            s += 4.0
        if goal.colour_hint and goal.colour_hint in colour:
            s += 1.5

        if goal.selector in ("smaller_bolt", "nearer_tray_bolt"):
            if role in ("small_bolt", "target_bolt", "bolt_m16", "target"):
                s += 10.0
            if "m16" in rk or "small" in rk:
                s += 8.0
            # In bolt_and_nut_to_fixture, only the smaller M16 bolt ever leans against the fixture
            if p.pose_hint == "leaning":
                s += 4.0
            if goal.selector == "nearer_tray_bolt":
                d_tray = float(np.linalg.norm(p.position[:2] - tray_xy))
                s -= 6.0 * d_tray
            for size_key in ("diameter", "size", "length", "area", "pixel_area", "extent"):
                val = p.extra.get(size_key)
                if isinstance(val, (int, float)) and math.isfinite(val):
                    vals = [
                        float(c.extra[size_key])
                        for c in candidates
                        if isinstance(c.extra.get(size_key), (int, float))
                    ]
                    if len(vals) >= 2 and max(vals) > min(vals):
                        norm = (float(val) - min(vals)) / (max(vals) - min(vals))
                        s += 2.5 * (1.0 - norm)
                    break
        elif goal.selector == "nut":
            if role in ("nut", "target_nut", "nut_m30", "target"):
                s += 6.0
            if "m30" in rk:
                s += 4.0
            if "gold" in colour:
                s += 2.0
        elif goal.selector in ("bolt", "washer"):
            if role in ("bolt", "washer", "target"):
                s += 6.0
            elif role in ("clutter", "decoy"):
                s -= 6.0
        return s

    return max(candidates, key=score_candidate)


class TemporalSceneFilter:
    """Short-window confidence-weighted robust filter for part and fixture estimates.

    Accumulates observations while the arm is observing/hovering, rejects single-frame
    outliers, and detects nudges (> 2.2 cm horizontal shift of an ungrasped target).
    """

    def __init__(self, window: int = 7) -> None:
        self.window = max(1, int(window))
        self._part_buf: list[PartEstimate] = []
        self._fixture_xy_buf: list[tuple[np.ndarray, float]] = []
        self._fixture_yaw_buf: list[tuple[float, float]] = []
        self._seats_buf: dict[str, list[tuple[np.ndarray, float]]] = {}

    def reset_part(self) -> None:
        self._part_buf.clear()

    def reset_all(self) -> None:
        self._part_buf.clear()
        self._fixture_xy_buf.clear()
        self._fixture_yaw_buf.clear()
        self._seats_buf.clear()

    def update_fixture(self, scene: SceneEstimate) -> None:
        w = max(0.05, float(scene.fixture_confidence))
        self._fixture_xy_buf.append((scene.fixture_xy.copy(), w))
        self._fixture_yaw_buf.append((float(scene.fixture_yaw), w))
        if len(self._fixture_xy_buf) > self.window * 2:
            self._fixture_xy_buf.pop(0)
            self._fixture_yaw_buf.pop(0)
        for sid, xyz in scene.seats.items():
            buf = self._seats_buf.setdefault(sid, [])
            buf.append((xyz.copy(), w))
            if len(buf) > self.window * 2:
                buf.pop(0)

    def update_part(self, part: PartEstimate | None) -> bool:
        """Add part observation. Returns True if a sudden > 2.2 cm shift (nudge) was detected."""
        if part is None:
            return False
        nudged = False
        if len(self._part_buf) >= 2:
            cur = self.filtered_part()
            if cur is not None and part.confidence >= 0.40:
                shift_xy = float(np.linalg.norm(part.position[:2] - cur.position[:2]))
                if shift_xy > 0.022:
                    # Wait for confirmation or reset immediately if very confident
                    if part.confidence >= 0.55:
                        self._part_buf.clear()
                        nudged = True
        self._part_buf.append(part)
        if len(self._part_buf) > self.window:
            self._part_buf.pop(0)
        return nudged

    @property
    def part_count(self) -> int:
        return len(self._part_buf)

    def filtered_part(self) -> PartEstimate | None:
        if not self._part_buf:
            return None
        if len(self._part_buf) == 1:
            return self._part_buf[0]

        pts = np.array([p.position for p in self._part_buf], dtype=np.float64)
        med = np.median(pts, axis=0)
        dists = np.linalg.norm(pts - med, axis=1)
        inliers = [p for p, d in zip(self._part_buf, dists) if d <= 0.025]
        if not inliers:
            inliers = list(self._part_buf)

        weights = np.array([max(0.05, p.confidence) for p in inliers], dtype=np.float64)
        w_sum = float(np.sum(weights))
        pos = np.sum(np.array([p.position for p in inliers]) * weights[:, None], axis=0) / w_sum

        # Align axes to the highest-confidence inlier's hemisphere unless directed
        ref = max(inliers, key=lambda p: p.confidence)
        axes = []
        for p in inliers:
            ax = p.axis.copy()
            if not p.axis_directed and float(np.dot(ax, ref.axis)) < 0.0:
                ax = -ax
            axes.append(ax)
        axis_mean = np.sum(np.array(axes) * weights[:, None], axis=0)
        an = float(np.linalg.norm(axis_mean))
        axis = axis_mean / an if an > 1e-8 else ref.axis.copy()

        # Circular mean for yaw
        sin_y = float(np.sum(np.sin([p.yaw for p in inliers]) * weights))
        cos_y = float(np.sum(np.cos([p.yaw for p in inliers]) * weights))
        yaw = math.atan2(sin_y, cos_y) if (sin_y * sin_y + cos_y * cos_y) > 1e-10 else ref.yaw

        # Majority vote for pose_hint weighted by confidence
        pose_votes: dict[str, float] = {}
        for p, w in zip(inliers, weights):
            pose_votes[p.pose_hint] = pose_votes.get(p.pose_hint, 0.0) + float(w)
        pose_hint = max(pose_votes.items(), key=lambda kv: kv[1])[0]

        mean_conf = float(np.mean([p.confidence for p in inliers]))
        return PartEstimate(
            kind=ref.kind,
            raw_kind=ref.raw_kind,
            position=pos,
            axis=axis,
            yaw=yaw,
            confidence=mean_conf,
            role=ref.role,
            pose_hint=pose_hint,
            axis_directed=any(p.axis_directed for p in inliers),
            extra=dict(ref.extra),
        )

    def filtered_fixture(self, fallback_scene: SceneEstimate) -> tuple[dict[str, np.ndarray], float, np.ndarray]:
        if not self._fixture_xy_buf:
            return fallback_scene.seats, fallback_scene.fixture_yaw, fallback_scene.fixture_xy

        xys = np.array([xy for xy, _ in self._fixture_xy_buf], dtype=np.float64)
        fxy = np.median(xys, axis=0)

        ys = np.array([y for y, _ in self._fixture_yaw_buf], dtype=np.float64)
        ws = np.array([w for _, w in self._fixture_yaw_buf], dtype=np.float64)
        fyaw = math.atan2(float(np.sum(np.sin(ys) * ws)), float(np.sum(np.cos(ys) * ws)))

        seats: dict[str, np.ndarray] = {}
        for sid, buf in self._seats_buf.items():
            arr = np.array([v for v, _ in buf], dtype=np.float64)
            seats[sid] = np.median(arr, axis=0)
        for sid, v in fallback_scene.seats.items():
            if sid not in seats:
                seats[sid] = v.copy()
        return seats, fyaw, fxy
