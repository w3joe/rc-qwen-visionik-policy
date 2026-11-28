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

from .classical.scene_interface import (SEAT_LOCAL_GEOM, GoalSpec, PartEstimate, SceneEstimate, parse_task_goals,
                                        select_target_part)

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


def validate_and_ground(plan: Plan, ctx: PlanContext) -> tuple[list[PlannedGoal], list[str]]:
    """Turn planner steps into executable goals. Unknown ids/seats are reported and the step is skipped."""
    goals: list[PlannedGoal] = []
    problems: list[str] = []
    used: set[str] = set()
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
                        colour_hint=str(part.extra.get("color") or part.extra.get("colour") or "") or None)
        if dest in SEAT_LOCAL_GEOM:
            g.seat_id = dest
        elif dest.startswith("on:") or dest.startswith("next_to:"):
            rel, _, ref = dest.partition(":")
            ref = ref.strip()
            g.relation = rel
            rp = ctx.part_by_id(ref.upper())
            if rp is not None:
                g.ref_part_xy, g.ref_part_kind = rp.position[:2].copy(), rp.kind
                g.ref_model = rp.raw_kind
                if rp is part:
                    problems.append(f"step {i}: relative to itself")
                    continue
            elif ref in SEAT_LOCAL_GEOM:
                g.ref_seat = ref
            else:
                problems.append(f"step {i}: unknown reference {ref!r}")
                continue
        else:
            problems.append(f"step {i}: unknown destination {s.dest!r}")
            continue
        goals.append(g)
    non_goal_xys = [p.position[:2].copy() for k, p in ctx.ids.items() if k not in used and p.confidence >= 0.45]
    for g in goals:
        g.exclude_xys = [xy for xy in non_goal_xys if g.target_xy is None or float(np.linalg.norm(xy - g.target_xy)) > 0.04]
    try:
        print("[plan] goals " + "; ".join(f"{g.goal_id}:{g.part_model}" for g in goals)
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
