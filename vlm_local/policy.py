"""Planner-driven vision-IK policy (contract v0 `Policy`: load() + act(obs)).

Per episode: at step 0 the classical perception module estimates the scene, ONE planner call turns
(instruction + goal picture + candidates) into an ordered goal list, and from then on every /act is the unchanged
vision-IK controller (milliseconds). The original hard-coded planner is the fallback when the VLM call fails or
returns nothing executable (and is selectable as the baseline).
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from .classical import vision as V
from .classical.kinematics import NUM_JOINTS, SAFE_STEP_LIMIT
from .classical.scene_interface import normalize_scene
from .controller import VLMEpisodeController
from .plan import HardcodedPlanner, Plan, PlanContext, validate_and_ground


class VLMPolicy:
    def __init__(self, planner=None, *, fallback_to_hardcoded: bool = True) -> None:
        self.planner = planner or HardcodedPlanner()
        self.fallback = HardcodedPlanner()
        self.use_fallback = fallback_to_hardcoded
        self._ep: Any = None
        self._ctrl: VLMEpisodeController | None = None
        self._last: tuple | None = None
        self.reset_stats()

    # --------------------------------------------------------------- bookkeeping
    def reset_stats(self) -> None:
        self.act_times: list[float] = []
        self.vision_calls = 0
        self.vision_errors = 0
        self.plan_info: dict[str, Any] = {}
        self.planner_calls: list[dict] = []

    def load(self) -> None:
        self._ep, self._ctrl, self._last = None, None, None

    @property
    def events(self) -> list[dict]:
        return self._ctrl.events if self._ctrl else []

    # --------------------------------------------------------------- act
    @staticmethod
    def _action(dq, grip) -> dict[str, Any]:
        vals = [max(-SAFE_STEP_LIMIT, min(SAFE_STEP_LIMIT, float(dq[i]) if i < len(dq) and math.isfinite(float(dq[i])) else 0.0))
                for i in range(NUM_JOINTS)]
        g = float(grip) if math.isfinite(float(grip)) else 0.8
        return {"type": "joint_delta", "values": vals, "gripper": max(0.0, min(1.0, g))}

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            return self._act(obs)
        finally:
            self.act_times.append(time.perf_counter() - t0)

    def _act(self, obs: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(obs, dict):
            return self._action(np.zeros(NUM_JOINTS), 0.8)
        ep, step = obs.get("episode_id"), obs.get("step", 0)
        if self._last is not None and self._last[0] == (ep, step):
            return self._last[1]
        images = obs.get("images")
        if not isinstance(images, dict) or not images:  # warm-up request: hold
            return self._action(np.zeros(NUM_JOINTS), 1.0)
        scene = None
        if self._ctrl is None or ep != self._ep or step == 0:
            self._ep = ep
            self.reset_stats()
            scene = self._estimate(obs)
            goals = self._plan(obs, scene)
            self._ctrl = VLMEpisodeController(obs, goals)
        elif self._ctrl.want_vision or self._ctrl.scene is None:
            scene = self._estimate(obs)
        dq, grip = self._ctrl.step(obs, scene)
        action = self._action(dq, grip)
        self._last = ((ep, step), action)
        return action

    def _estimate(self, obs):
        self.vision_calls += 1
        try:
            return normalize_scene(V.estimate_scene(obs))
        except Exception:  # noqa: BLE001
            self.vision_errors += 1
            return None

    def _plan(self, obs, scene) -> list:
        t0 = time.perf_counter()
        imgs = {k: V._decode_image(v) for k, v in (obs.get("images") or {}).items()}
        ctx = PlanContext(obs, scene, imgs, V._decode_image(obs.get("goal")) if obs.get("goal") else None)
        plan: Plan | None = None
        goals: list = []
        err = None
        try:
            plan = self.planner.plan(ctx)
            self.planner_calls = list(plan.calls)
            if getattr(plan, "legacy_goals", None) is not None:
                goals = list(plan.legacy_goals)
                problems: list[str] = []
            else:
                goals, problems = validate_and_ground(plan, ctx)
        except Exception as e:  # noqa: BLE001
            err, problems = f"{type(e).__name__}: {str(e)[:160]}", [f"planner error"]
        used_fallback = False
        if not goals and self.use_fallback and not isinstance(self.planner, HardcodedPlanner):
            fb = self.fallback.plan(ctx)
            goals, used_fallback = list(fb.legacy_goals), True  # type: ignore[attr-defined]
        self.plan_info = {
            "planner": getattr(self.planner, "name", "?"), "wall_s": round(time.perf_counter() - t0, 3),
            "steps": [(s.part, s.dest) for s in plan.steps] if plan else [],
            "analysis": plan.analysis if plan else "", "raw": (plan.raw[:600] if plan else ""),
            "problems": problems, "error": err, "used_fallback": used_fallback,
            "candidates": {k: [p.raw_kind, p.pose_hint, [round(float(x), 3) for x in p.position[:2]], round(p.confidence, 2)]
                           for k, p in ctx.ids.items()},
            "goals": [getattr(g, "goal_id", "?") for g in goals],
        }
        return goals
