"""VLM planner: set-of-mark prompt (candidate ids drawn on the camera frames) -> ordered JSON plan.

Perception stays classical (vision.estimate_scene: calibrated stereo -> part models/colours/poses in 3D, seats from
the fitted fixture pose). The VLM only decides WHICH detected part goes WHERE and in which order, reading the
instruction and the goal picture. Inputs used: the /act images, goal picture, instruction (nothing else).
"""
from __future__ import annotations

import json
import math
import re
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .classical import vision as V
from .plan import SEAT_IDS, Plan, PlanContext, Step

SHORT = {"tray": "T", "bore_m16": "b16", "bore_m24": "b24", "bore_m30": "b30",
         "hex_m24": "h24", "hex_m30": "h30", "hex_m36": "h36"}

SYSTEM = """You are the task planner of a UR12e robot arm with a parallel gripper, working at a table with a printed \
fixture. A tested motion library executes your plan: for every step it picks the chosen part and puts it at the chosen \
destination. You do not output coordinates; you choose, from the labelled candidates, which part goes where and in \
which order. Reply with one JSON object only."""

KNOWLEDGE = """Fixture seats (labels drawn on the pictures in cyan; answer with the full seat id):
- tray (T): open tray; bolts, nuts and washers can all be dropped in it.
- bore_m16 (b16): round vertical hole, 20 mm; bore_m24 (b24): 28 mm; bore_m30 (b30): 35 mm. A bolt is inserted standing \
up (head up) into the hole of ITS size: bolt_m16x50 -> bore_m16, bolt_m24x50 -> bore_m24, bolt_m30x60 -> bore_m30. A bolt \
too fat for a hole cannot enter it; a thin bolt in a wide hole does not count. A washer can lie flat over a hole mouth.
- hex_m24 (h24): hex pocket 41 mm across flats; hex_m30 (h30): 51 mm; hex_m36 (h36): 60 mm. A nut is seated flat in the \
pocket of ITS size: nut_m24 -> hex_m24, nut_m30 -> hex_m30, nut_m36 -> hex_m36 ("the pocket that matches its size").
Part sizes: bolts M16x50 (smallest), M24x50, M30x60 (largest); nuts M24 < M30 < M36; washer_m30 (62 mm ring).
Relative destinations: "on:P3" = stack the part on part P3 (a washer is lowered over the shank of a bolt P3 that stands on \
its head; a washer can also go over a hole first and the bolt then goes through it into the same hole); "next_to:P3" or \
"next_to:tray" = a free spot on the table beside that part / seat (the moved part is the one the instruction says to move; the \
reference part stays where it is: "put the washer next to the bolt" -> part = the washer, dest = next_to:<the bolt>)."""

USER = """Instruction: "{instruction}"
Pictures, in order: 1) GOAL = zoomed crop of the fixture in the finished scene (every goal part seated; seat labels drawn next to the seats' centres); 2) LEFT camera and 3) RIGHT \
camera of the CURRENT scene with candidate part ids drawn in yellow (P1, P2, ...) and seat labels in cyan.
Detected parts (from the perception module; model/colour are estimates):
{parts}

How to decide:
- The INSTRUCTION says what to do: move exactly the objects it names, to the kind of destination it names ("tray" -> tray; "hole" -> a \
bore_*; "hex pocket"/"pocket" -> a hex_*; "on the bolt" -> on:Pk). Other parts on the table (clutter, decoys, e.g. extra nuts or the other \
bolt) must NOT be touched: if the instruction names ONE bolt, output exactly ONE bolt step.
- When the instruction leaves a choice (which of two bolts or nuts; which hole/pocket "matches its size"), use the detected sizes and \
tags above and the GOAL picture (which labelled seat holds which part there).
- Follow the instruction's order if it gives one ("then", "first"); otherwise any order.
{knowledge}

The model names of detected NUTS are unreliable (a nut standing on its edge is often mislabelled one size too small): for a nut, choose \
the pocket from the instruction and the GOAL picture, not from the detected size. Bolt models are reliable.
Reply with JSON only, no markdown:
{{"task": "<the instruction in your own words: which objects go to which kind of destination>", "steps": [{{"part": "<Pk>", "dest": "<seat id | on:Pk | next_to:Pk>"}}]}}
One step per moved object (usually 1 or 2); never use a part twice."""


def _font(sz: int):
    try:
        return ImageFont.load_default(size=sz)
    except TypeError:
        return ImageFont.load_default()


def _label(d: ImageDraw.ImageDraw, xy, text: str, fg, font, bg=(0, 0, 0, 170)) -> None:
    x, y = xy
    w = d.textlength(text, font=font)
    d.rectangle([x - 2, y - 1, x + w + 2, y + font.size + 2 if hasattr(font, "size") else y + 14], fill=bg)
    d.text((x, y), text, fill=fg, font=font)


def annotate(ctx: PlanContext) -> list[Image.Image]:
    """GOAL (seat labels) + LEFT + RIGHT (part ids + seat labels)."""
    out: list[Image.Image] = []
    seats = ctx.scene.seats if ctx.scene is not None else {}
    f = _font(14)
    if ctx.goal_img is not None:
        gimg = Image.fromarray(ctx.goal_img).convert("RGBA")
        gcam = V.Camera("goal", np.array(V.GOAL_CAMERA_POS, float), V._look_mat(V.GOAL_CAMERA_POS, V.GOAL_CAMERA_TARGET),
                        V.GOAL_CAMERA_FOVY, gimg.size[0], gimg.size[1])
        # zoomed crop of the fixture (the goal camera is fixed; the fixture pose comes from the perception module)
        corners = []
        for lx in (-0.125, 0.125):
            for ly in (-0.115, 0.115):
                for z in (0.0, 0.08):
                    w = ctx.scene.fixture_to_world((lx, ly)) if ctx.scene is not None else (0.7 + lx, 0.1 + ly)
                    corners.append([w[0], w[1], z])
        u, v, dep = gcam.project(np.asarray(corners, float))
        x0, x1 = max(0, int(u.min()) - 25), min(gimg.size[0], int(u.max()) + 25)
        y0, y1 = max(0, int(v.min()) - 35), min(gimg.size[1], int(v.max()) + 25)
        sc = min(4.0, 640.0 / max(x1 - x0, 1))
        crop = gimg.crop((x0, y0, x1, y1)).resize((int((x1 - x0) * sc), int((y1 - y0) * sc)), Image.LANCZOS)
        ov = Image.new("RGBA", crop.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov); fb = _font(18)
        for sid, xyz in seats.items():
            if sid in SHORT:
                su, sv, sd = gcam.project(np.asarray(xyz, float)[None, :])
                if sd[0] > 0:
                    cx, cy = (su[0] - x0) * sc, (sv[0] - y0) * sc
                    d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(0, 255, 255, 255))
                    _label(d, (cx + 7, cy - 9), SHORT[sid], (0, 255, 255, 255), fb)
        out.append(Image.alpha_composite(crop, ov).convert("RGB"))
    for name in ("left", "right"):
        if name not in ctx.images:
            continue
        im = Image.fromarray(ctx.images[name]).convert("RGBA")
        ov = Image.new("RGBA", im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
        cam = ctx.cameras.get(name)
        if cam is not None:
            for sid, xyz in seats.items():
                if sid in SHORT:
                    u, v, dep = cam.project(np.asarray(xyz, float)[None, :])
                    if dep[0] > 0:
                        d.ellipse([u[0] - 3, v[0] - 3, u[0] + 3, v[0] + 3], fill=(0, 255, 255, 255))
                        _label(d, (u[0] + 5, v[0] - 7), SHORT[sid], (0, 255, 255, 255), f)
            for pid, p in ctx.ids.items():
                u, v, dep = cam.project(np.asarray(p.position, float)[None, :])
                if dep[0] > 0:
                    d.ellipse([u[0] - 13, v[0] - 13, u[0] + 13, v[0] + 13], outline=(255, 230, 0, 255), width=2)
                    _label(d, (u[0] + 11, v[0] - 22), pid, (255, 230, 0, 255), f)
        out.append(Image.alpha_composite(im, ov).convert("RGB"))
    return out


def _msize(model: str) -> int:
    m = re.search(r"m(\d+)", model or "")
    return int(m.group(1)) if m else 0


def parts_table(ctx: PlanContext) -> str:
    """One row per detected part, with the derived facts an instruction may refer to (size rank, distance to the tray)."""
    tray = np.asarray(ctx.scene.seats.get("tray", [0.63, 0.1, 0.0]), float)[:2] if ctx.scene is not None else np.array([0.63, 0.1])
    dtray = {pid: float(np.linalg.norm(p.position[:2] - tray)) * 100 for pid, p in ctx.ids.items()}
    tags: dict[str, list[str]] = {pid: [] for pid in ctx.ids}
    for kind in ("bolt", "nut", "washer"):
        grp = [(pid, p) for pid, p in ctx.ids.items() if p.kind == kind]
        if len(grp) < 2:
            continue
        sizes = {pid: _msize(p.raw_kind) for pid, p in grp}
        if len(set(sizes.values())) > 1:
            tags[min(sizes, key=sizes.get)].append(f"smallest {kind}")
            tags[max(sizes, key=sizes.get)].append(f"largest {kind}")
        tags[min((pid for pid, _ in grp), key=dtray.get)].append(f"nearest the tray among {kind}s")
        tags[max((pid for pid, _ in grp), key=dtray.get)].append(f"farthest from the tray among {kind}s")
    rows = []
    for pid, p in ctx.ids.items():
        col = p.extra.get("color") or p.extra.get("colour") or "?"
        near_fix = ctx.scene.fixture_distance(p.position[:2]) < 0.03
        model = p.raw_kind
        if p.kind == "nut" and p.pose_hint == "on_edge":
            model = "nut (size not reliably detected: it stands on its edge; take the pocket from the instruction and the GOAL picture)"
        rows.append(f"- {pid}: {model}, colour {col}, pose {p.pose_hint}, {'against the fixture' if near_fix else 'on the table'}, "
                    f"{dtray[pid]:.0f} cm from the tray centre" + (f" [{'; '.join(tags[pid])}]" if tags[pid] else ""))
    return "\n".join(rows) if rows else "- (none detected)"


def build_prompt(ctx: PlanContext) -> str:
    return USER.format(instruction=ctx.instruction, parts=parts_table(ctx), knowledge=KNOWLEDGE)


def parse_plan_json(raw: str) -> dict | None:
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    for cand in (raw, (re.search(r"\{.*\}", raw, re.S) or [None])[0] if re.search(r"\{.*\}", raw, re.S) else None):
        if not cand:
            continue
        try:
            v = json.loads(cand)
            if isinstance(v, dict):
                return v
        except Exception:  # noqa: BLE001
            continue
    return None


def _norm_dest(d: Any) -> str:
    if isinstance(d, dict):  # tolerate {"on": "P3"} / {"seat": "tray"}
        for k, v in d.items():
            if k in ("on", "next_to"):
                return f"{k}:{v}"
            if k in ("seat", "dest"):
                return _norm_dest(v)
    s = str(d or "").strip().lower().replace(" ", "")
    inv = {v.lower(): k for k, v in SHORT.items()}
    if s in inv:
        s = inv[s]
    if s.startswith("on:") or s.startswith("next_to:"):
        head, _, ref = s.partition(":")
        ref = inv.get(ref, ref)
        return f"{head}:{ref}"
    return s


class VLMPlanner:
    """backend: object with chat(system, text, images, max_new_tokens) -> (text, stats)."""

    def __init__(self, backend, *, name: str = "vlm", max_new_tokens: int = 320, retries: int = 1) -> None:
        self.backend, self.name, self.max_new_tokens, self.retries = backend, name, max_new_tokens, retries

    def plan(self, ctx: PlanContext) -> Plan:
        images = annotate(ctx)
        prompt = build_prompt(ctx)
        calls, raw, data = [], "", None
        for attempt in range(1 + self.retries):
            t0 = time.time()
            raw, st = self.backend.chat(SYSTEM, prompt, images, self.max_new_tokens)
            st = dict(st); st["wall_s"] = round(time.time() - t0, 3); st["kind"] = "plan"; calls.append(st)
            data = parse_plan_json(raw)
            if data and isinstance(data.get("steps"), list):
                break
        if not data or not isinstance(data.get("steps"), list):
            return Plan(steps=[], source=self.name, raw=raw, valid=False, problems=["unparseable plan"], calls=calls)
        steps = []
        for s in data["steps"]:
            if not isinstance(s, dict) or "part" not in s:
                continue
            steps.append(Step(part=str(s["part"]), dest=_norm_dest(s.get("dest", s.get("destination", ""))),
                              pt_left=s.get("pt_left"), pt_right=s.get("pt_right")))
        return Plan(steps=steps, source=self.name, analysis=str(data.get("task", data.get("goal_state", data.get("analysis", ""))))[:300], raw=raw, calls=calls)
