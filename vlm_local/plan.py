"""Planner interface: a planner turns (instruction, goal picture, detected scene) into an ordered goal list.

The motion code (vision-IK skills in `classical/planner.py`) is unchanged; a planner only decides

  * which detected part to move (`part_id`, one of the P1..Pn ids shown to it),
  * where it goes: a fixture seat (`tray`, `bore_*`, `hex_*`) or a RELATIVE destination
    (`on:<part id>` = stack on that part, `next_to:<part id|seat>` = free table spot beside it),
  * in which order.

Compatible with exports/benchmarks/llm-oneshot: there a planner call answers one primitive (pick|place) with
pixel targets; here one call answers the whole ordered list with candidate ids and (optionally) 2D points.
`Plan.to_primitives()` yields the same pick/place primitive sequence if a caller wants the llm-oneshot shape.
"""
from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .classical.scene_interface import (FASTENER_GEOM, SEAT_LOCAL_GEOM, GoalSpec, PartEstimate, SceneEstimate,
                                        is_part_in_seat, parse_task_goals, select_target_part)

SEAT_IDS = tuple(SEAT_LOCAL_GEOM)  # tray, bore_m24, bore_m16, bore_m30, hex_m30, hex_m36, hex_m24
KIND_OF = {"bolt": "bolt", "nut": "nut", "washer": "washer"}


@dataclass
class Step:
    """One requested placement, as a planner states it (ids refer to the PlanContext candidate table)."""
    part: str                      # "P2"
    dest: str                      # "tray" | "bore_m16" | "hex_m30" | "on:P3" | "next_to:P3" | "next_to:bore_m24"
    pt_left: list[float] | None = None   # optional 2D point of the part in the LEFT image (px)
    pt_right: list[float] | None = None
    note: str = ""


@dataclass
class Plan:
    steps: list[Step]
    source: str = "?"
    analysis: str = ""
    raw: str = ""
    valid: bool = True
    problems: list[str] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)   # per-call stats: latency_s, tokens_in/out, model, kind

    def to_primitives(self) -> list[dict]:
        out = []
        for s in self.steps:
            out += [{"primitive": "pick", "target": s.part, "left": {"center": s.pt_left}, "right": {"center": s.pt_right}},
                    {"primitive": "place", "target": s.dest}]
        return out


@dataclass
class PlannedGoal:
    """Duck-types `GoalSpec` (goal_id, part_kind, part_model, seat_id, selector, colour_hint) and adds grounding."""
    goal_id: str
    part_kind: str
    part_model: str
    seat_id: str                       # "" for relative destinations
    selector: str = "vlm"
    colour_hint: str | None = None
    target_xy: np.ndarray | None = None  # world xy of the chosen part (updated while it is tracked)
    relation: str | None = None          # None | "on" | "next_to"
    ref_part_xy: np.ndarray | None = None
    ref_part_kind: str | None = None
    ref_seat: str | None = None
    ref_model: str | None = None
    on_washer_model: str | None = None
    pid: str = ""
    exclude_xys: list[np.ndarray] = field(default_factory=list)


class PlanContext:
    """Everything a planner may look at. Built only from the /act envelope + the perception module's output."""

    def __init__(self, obs: dict, scene: SceneEstimate, images: dict[str, np.ndarray], goal_img: np.ndarray | None):
        self.obs = obs
        self.instruction = str(obs.get("instruction") or "")
        self.scene = scene
        self.images = images
        self.goal_img = goal_img
        self.cameras = scene.raw.get("cameras", {}) if scene is not None else {}
        # candidate parts in a stable order (left-to-right is not needed: ids are just labels)
        self.parts: list[PartEstimate] = sorted(scene.parts, key=lambda p: (-p.confidence)) if scene else []
        self.ids = {f"P{i + 1}": p for i, p in enumerate(self.parts)}

    def part_by_id(self, pid: str) -> PartEstimate | None:
        return self.ids.get(str(pid).strip().upper())


class Planner(Protocol):
    name: str

    def plan(self, ctx: PlanContext) -> Plan: ...


class HardcodedPlanner:
    """Baseline: the original keyword rules (`parse_task_goals`) + vision's instruction-driven role assignment."""
    name = "hardcoded"

    def plan(self, ctx: PlanContext) -> Plan:
        goals = parse_task_goals(ctx.obs)
        steps = []
        for g in goals:
            p = select_target_part(ctx.scene, g) if ctx.scene is not None else None
            pid = next((k for k, v in ctx.ids.items() if v is p), "P?") if p is not None else "P?"
            steps.append(Step(part=pid, dest=g.seat_id, note=g.goal_id))
        plan = Plan(steps=steps, source=self.name)
        plan.legacy_goals = goals  # type: ignore[attr-defined]
        return plan


def _cos(a: float) -> float:
    return math.cos(a)


def part_kind(model: str) -> str:
    m = (model or "").lower()
    return "washer" if "washer" in m else ("nut" if "nut" in m else "bolt")


def _bolt_want_bore(model: str) -> str | None:
    if os.environ.get("RC_SIZE_OVERRIDE", "1") == "0":
        return None
    m = re.match(r"bolt_m(\d+)", str(model))
    if not m:
        return None
    want = "bore_m" + m.group(1)
    return want if want in SEAT_LOCAL_GEOM else None


def validate_and_ground(plan: Plan, ctx: PlanContext) -> tuple[list[PlannedGoal], list[str]]:
    """Turn planner steps into executable goals. Unknown ids/seats are reported and the step is skipped."""
    goals: list[PlannedGoal] = []
    problems: list[str] = []
    used: set[str] = set()
    by_pid: dict[str, PlannedGoal] = {}
    for i, s in enumerate(plan.steps):
        part = ctx.part_by_id(s.part)
        if part is None and (s.pt_left or s.pt_right):
            part = nearest_to_pixel(ctx, s.pt_left, s.pt_right)
        if part is None:
            problems.append(f"step {i}: unknown part {s.part!r}")
            continue
        pid = next(k for k, v in ctx.ids.items() if v is part)
        if pid in used:
            problems.append(f"step {i}: part {pid} used twice")
            continue
        used.add(pid)
        model = part.raw_kind if part.raw_kind and part.raw_kind != "unknown" else part.kind
        dest = (s.dest or "").strip().lower()
        # "on:<seat>" for a fixture seat (e.g. washer "on:bore_m30" or bolt "on:bore_m30") means placing at that seat.
        if dest.startswith("on:"):
            _cand_seat = dest.partition(":")[2].strip()
            if _cand_seat in SEAT_LOCAL_GEOM:
                dest = _cand_seat
        # A nut goes in the hex pocket of its own size and a bolt in the hole of its own size: the learned
        # detector's size class is reliable, the language planner's pocket choice is not.
        _m = re.match(r"(nut|bolt)_m(\d+)", str(model))
        if _m and os.environ.get("RC_SIZE_OVERRIDE", "1") != "0":
            _want = ("hex_m" if _m.group(1) == "nut" else "bore_m") + _m.group(2)
            # any fixture seat other than the tray: the hole/pocket of the part's own kind and size
            if _want in SEAT_LOCAL_GEOM and dest.startswith(("bore_", "hex_")) and dest != _want:
                print(f"[plan] size override {pid}: {dest} -> {_want} ({model})", file=sys.stderr, flush=True)
                dest = _want
        g = PlannedGoal(goal_id=f"{pid}->{dest}", part_kind=part.kind, part_model=model, seat_id="",
                        target_xy=part.position[:2].copy(),
                        colour_hint=str(part.extra.get("color") or part.extra.get("colour") or "") or None,
                        pid=pid)
        if dest in SEAT_LOCAL_GEOM:
            g.seat_id = dest
        elif dest.startswith("on:") or dest.startswith("next_to:"):
            rel, _, ref = dest.partition(":")
            ref = ref.strip()
            ref_pid = ref.upper()
            rp = ctx.part_by_id(ref_pid)
            if rp is not None:
                if rp is part:
                    problems.append(f"step {i}: relative to itself")
                    continue
                rp_model = rp.raw_kind if rp.raw_kind and rp.raw_kind != "unknown" else rp.kind
                # Bolt "on:<washer>" means inserting the bolt through that washer into the hole mouth underneath it.
                if rel == "on" and part.kind == "bolt" and rp.kind == "washer":
                    gw = by_pid.get(ref_pid)
                    want_bore = _bolt_want_bore(model)
                    seated_bore = None
                    if ctx.scene is not None:
                        for sid, sgeom in SEAT_LOCAL_GEOM.items():
                            if sgeom.get("kind") == "hole_vertical" and is_part_in_seat(rp, sid, ctx.scene):
                                seated_bore = sid
                                break
                    bore_seat = want_bore or (gw.seat_id if gw and gw.seat_id.startswith("bore_") else seated_bore) or "bore_m30"
                    if gw is not None and gw.seat_id.startswith("bore_") and gw.seat_id != bore_seat:
                        print(f"[plan] align washer {ref_pid} bore: {gw.seat_id} -> {bore_seat}", file=sys.stderr, flush=True)
                        gw.seat_id = bore_seat
                        gw.goal_id = f"{gw.pid}->{bore_seat}"
                    g.seat_id = bore_seat
                    g.relation = None
                    g.on_washer_model = rp_model
                    g.ref_part_kind = "washer"
                    g.ref_model = rp_model
                    g.goal_id = f"{pid}->{bore_seat}(on:{ref_pid})"
                else:
                    # Washer "on:<bolt>": ensure reference bolt is standing if another standing bolt matches
                    if rel == "on" and part.kind == "washer" and rp.kind == "bolt" and rp.pose_hint != "standing" and ref_pid not in by_pid:
                        w_hole = float(FASTENER_GEOM.get(model, FASTENER_GEOM["washer_m33"]).get("inner_diameter", 0.034))
                        st_bolts = [cand for cand in ctx.parts if cand.kind == "bolt" and cand.pose_hint == "standing"
                                    and float(FASTENER_GEOM.get(cand.raw_kind, {}).get("diameter", 0.030)) < w_hole]
                        if len(st_bolts) == 1:
                            rp = st_bolts[0]
                            rp_model = rp.raw_kind if rp.raw_kind and rp.raw_kind != "unknown" else rp.kind
                            print(f"[plan] redirect {pid} on standing bolt {rp_model}", file=sys.stderr, flush=True)
                    g.relation = rel
                    g.ref_part_xy, g.ref_part_kind = rp.position[:2].copy(), rp.kind
                    g.ref_model = rp_model
            elif ref in SEAT_LOCAL_GEOM:
                g.relation = rel
                g.ref_seat = ref
            else:
                problems.append(f"step {i}: unknown reference {ref!r}")
                continue
        else:
            problems.append(f"step {i}: unknown destination {s.dest!r}")
            continue
        goals.append(g)
        by_pid[pid] = g

    # Pair any washer->bore and bolt->same_bore goals (e.g. when both steps named bore_m30 directly), and keep washer first.
    for idx_b, gb in enumerate(list(goals)):
        if gb.part_kind != "bolt" or SEAT_LOCAL_GEOM.get(gb.seat_id, {}).get("kind") != "hole_vertical":
            continue
        matching_w = [gw for gw in goals if gw.part_kind == "washer"
                      and SEAT_LOCAL_GEOM.get(gw.seat_id, {}).get("kind") == "hole_vertical"
                      and (gw.seat_id == gb.seat_id or gb.on_washer_model == gw.part_model)]
        if matching_w:
            gw = matching_w[0]
            if gw.seat_id != gb.seat_id:
                gw.seat_id = gb.seat_id
                gw.goal_id = f"{gw.pid}->{gb.seat_id}"
            gb.on_washer_model = gw.part_model
            gb.ref_part_kind = "washer"
            gb.ref_model = gw.part_model
            ib, iw = goals.index(gb), goals.index(gw)
            if ib < iw:
                goals.remove(gw)
                goals.insert(ib, gw)
        elif gb.on_washer_model is None and ctx.scene is not None:
            for pw in ctx.scene.parts:
                if pw.kind == "washer" and is_part_in_seat(pw, gb.seat_id, ctx.scene):
                    gb.on_washer_model = pw.raw_kind if pw.raw_kind and pw.raw_kind != "unknown" else "washer_m33"
                    gb.ref_part_kind = "washer"
                    gb.ref_model = gb.on_washer_model
                    break

    non_goal_parts = [(p.raw_kind, p.position[:2].copy()) for k, p in ctx.ids.items() if k not in used and p.confidence >= 0.70]
    for g in goals:
        g.exclude_xys = [
            xy for rk, xy in non_goal_parts
            if rk != g.part_model and (g.target_xy is None or float(np.linalg.norm(xy - g.target_xy)) > 0.045)
        ]
    try:
        print("[plan] goals " + "; ".join(f"{g.goal_id}:{g.part_model}" + (f"[on:{g.on_washer_model}]" if g.on_washer_model else "") for g in goals)
              + " | parts " + "; ".join(f"{k}:{v.raw_kind}" for k, v in ctx.ids.items()), file=sys.stderr, flush=True)
    except Exception:
        pass
    return goals, problems


def nearest_to_pixel(ctx: PlanContext, pt_left, pt_right) -> PartEstimate | None:
    best, best_d = None, 60.0
    for p in ctx.parts:
        ds = []
        for name, pt in (("left", pt_left), ("right", pt_right)):
            cam = ctx.cameras.get(name)
            if cam is None or not pt:
                continue
            u, v, d = cam.project(np.asarray(p.position, float)[None, :])
            if float(d[0]) > 0:
                ds.append(math.hypot(float(u[0]) - pt[0], float(v[0]) - pt[1]))
        if ds and min(ds) < best_d:
            best, best_d = p, min(ds)
    return best


def goals_from_legacy(goals: list[GoalSpec]) -> list[Any]:
    return list(goals)
