#!/usr/bin/env python
"""Program-form policy server for docs/api.md (contract v0, schema v3): VLM planner + vision-IK controller.

    # judged-style: model in-process from RC_WEIGHTS_DIR/qwen3_vl_8b (e.g. /weights/qwen3_vl_8b), no network
    python serve.py
    # dev: planner calls go to a running vlm_server.py (e.g. over an SSH tunnel)
    RC_VLM_URL=http://127.0.0.1:9001 python policy_server.py
    # baseline
    RC_PLANNER=hardcoded python policy_server.py

/healthz answers at once, /readyz flips to ready after the model is loaded AND a dummy plan call has warmed the kernels.
POST /act echoes episode_id/step (body and X-RC-* headers) and is idempotent per (episode_id, step).
With RC_PRIVATE=1 nothing about prompts, plans or part layouts is printed (only counters).
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
CONTRACT, SCHEMA_VERSION = "v0", 3
STATE: dict = {"ready": False, "error": None, "policy": None}
LOCK = threading.Lock()
PRIVATE = os.environ.get("RC_PRIVATE") == "1"
STATS_PATH = os.environ.get("RC_STATS_PATH")  # per-episode timing counters (no content) appended as JSON lines


def load() -> None:
    try:
        from vlm_local.llm_planner import VLMPlanner
        from vlm_local.plan import HardcodedPlanner
        from vlm_local.policy import VLMPolicy

        which = os.environ.get("RC_PLANNER", "vlm")
        if which == "hardcoded":
            planner = HardcodedPlanner()
        elif os.environ.get("RC_VLM_URL"):
            from vlm_local.backends import RemoteBackend
            planner = VLMPlanner(RemoteBackend(os.environ["RC_VLM_URL"]), name="remote")
        else:
            from vlm_local.backends import HFQwen3VL
            path = Path(os.environ.get("RC_VLM_PATH") or Path(os.environ.get("RC_WEIGHTS_DIR") or "/weights") / "qwen3_vl_8b")
            if not (path / "config.json").is_file():
                raise RuntimeError("weights directory missing")  # never fall back to another model
            backend = HFQwen3VL(str(path), quant=os.environ.get("RC_VLM_QUANT", "bf16"))
            planner = VLMPlanner(backend, name="qwen3vl-local")
            # warm the kernels with a dummy call BEFORE /readyz flips, so the first real plan does not pay for it
            from PIL import Image
            t0 = time.time()
            backend.chat("", "Reply with the word ok.", [Image.new("RGB", (640, 480), (90, 90, 90))] * 3, 4)
            print(f"[vlm-local] model loaded + warmed in {time.time() - t0:.1f}s from {path}", flush=True)
        STATE["policy"] = VLMPolicy(planner)
        STATE["ready"] = True
        print("[vlm-local] ready", flush=True)
    except Exception:  # noqa: BLE001
        STATE["error"] = "policy failed to load"
        traceback.print_exc()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, code: int, body: dict, headers: dict | None = None) -> None:
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(raw)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers(); self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._json(200, {})
        elif self.path == "/readyz":
            body = {"ready": bool(STATE["ready"]), "contract": CONTRACT, "schema_version": SCHEMA_VERSION,
                    "capabilities": {"chunking": True, "caching": False}}  # advertised as in the starter (the runner ignores it; this policy always returns 1 action)
            if STATE["error"]:
                body["ready"] = False; body["error"] = STATE["error"]
            self._json(200, body)
        else:
            self._json(404, {"error": "no route"})

    def do_POST(self) -> None:
        if self.path != "/act":
            return self._json(404, {"error": "no route"})
        if not STATE["ready"]:
            return self._json(503, {"error": "not ready"})
        try:
            env = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
            ep, step = env["episode_id"], env["step"]
        except Exception:  # noqa: BLE001
            return self._json(400, {"error": "invalid request: body must be JSON with episode_id and step"})
        try:
            with LOCK:
                pol = STATE["policy"]
                action = pol.act(env)
                if step == 0 and pol.planner_calls:  # counters only: no instruction, plan or scene content
                    print("[plan] " + json.dumps({"calls": len(pol.planner_calls), "lat_s": [c.get("latency_s") for c in pol.planner_calls],
                          "tok": [(c.get("tokens_in"), c.get("tokens_out")) for c in pol.planner_calls],
                          "steps": len(pol.plan_info.get("steps", [])), "fallback": pol.plan_info.get("used_fallback"),
                          "act0_s": round(pol.act_times[-1], 2) if pol.act_times else None}), flush=True)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            return self._json(500, {"error": "policy act() failed"})
        if STATS_PATH and (step == 0 or step % 100 == 0):
            try:
                with open(STATS_PATH, "a") as f:
                    f.write(json.dumps({"t": time.time(), "step": step, "act_total_s": round(sum(pol.act_times), 3),
                                        "vlm_calls": len(pol.planner_calls),
                                        "vlm_latency_s": [c.get("latency_s") for c in pol.planner_calls],
                                        "tokens": [(c.get("tokens_in"), c.get("tokens_out")) for c in pol.planner_calls],
                                        "fallback": pol.plan_info.get("used_fallback")}) + "\n")
            except Exception:  # noqa: BLE001
                pass
        self._json(200, {"episode_id": ep, "step": step, "action": action},
                   {"X-RC-Episode-Id": str(ep), "X-RC-Step": str(step)})


def main() -> None:
    port = int(os.environ.get("RC_POLICY_PORT", "8080"))
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    srv.daemon_threads = True
    threading.Thread(target=load, daemon=True).start()  # listen first, load afterwards
    print(f"[vlm-local] serving contract {CONTRACT} on :{port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
