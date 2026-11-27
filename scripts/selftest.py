"""Self-test for your policy repository: python3 scripts/selftest.py [--weights DIR]

The last line is ALL PASS when your load() and act() ran, "CONTRACT ONLY: ..." (exit 0) naming why they
could not run here and the next step, or "FAILED: ..." (exit 1) naming the first failed check. It checks,
in four numbered sections and without downloading any model:
  1. manifest.json (contract v0, a valid entry_point);
  2. your entry point: a Policy class is imported (with the judged image's loader when rc_runtime
     is importable, otherwise a plain import), then load() runs on --weights DIR (your downloaded
     snapshot; the starter example uses a tiny generated checkpoint when torch and safetensors are
     installed) and act() runs on a real step-0 observation (scripts/sample_obs/); the action is
     validated like the runner does. A program entry point (serve.py) is started and checked over HTTP;
  3. the example servers in examples/ (FastAPI is skipped when it is not installed);
  4. the HTTP contract of the SDK server: /healthz at once, /readyz after load, /act echoes
     episode_id and step, a malformed /act gets 400, a failed load stays not-ready.

It imports rc_sdk from ./rc_sdk (or from RC_SDK_DIR when set). Inside the
published image run `python -I -m rc_runtime --selfcheck --submission /submission` instead.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import socket
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
# Package root is starter/rc_sdk (the installable project). Works from starter/ without pip. Without that
# folder, point RC_SDK_DIR at the bundle's starter/rc_sdk or pip install it; the published image ships rc_sdk.
SDK_DIRS = [d for d in (os.environ.get("RC_SDK_DIR"), os.path.join(ROOT, "rc_sdk")) if d]
for sdk in reversed(SDK_DIRS):
    if os.path.isdir(os.path.join(sdk, "rc_sdk")):
        sys.path.insert(0, sdk)
for candidate in (os.path.join(ROOT, "..", "runtime"),):  # monorepo checkout
    if os.path.isdir(os.path.join(candidate, "rc_runtime")):
        sys.path.append(candidate)

try:
    from rc_sdk.policy import Policy  # noqa: E402
    from rc_sdk.server import PolicyServer  # noqa: E402
except ImportError:
    sys.stderr.write(
        "FAIL rc_sdk not found. The self-test needs the starter SDK: keep rc_sdk/ in your copy,\n"
        "or set RC_SDK_DIR to the participant bundle's starter/rc_sdk, or\n"
        "`pip install <bundle>/starter/rc_sdk`, or run it inside the published image, which ships rc_sdk.\n")
    sys.exit(2)

sys.path.insert(0, HERE)  # the stdlib checkpoint writer shared with make_example_weights.py
from make_example_weights import write_example_weights  # noqa: E402

try:
    from rc_runtime import loader as rt_loader  # noqa: E402
    from rc_runtime.server import RuntimeServer  # noqa: E402
except ImportError:
    rt_loader = None
    RuntimeServer = None

DELIBERATE_FAILURE = "selftest: deliberate load failure (expected)"
HOME_Q = (-0.2055, -1.1091, 1.6007, -1.9708, -1.5708, 1.5708)  # the arm's joint target at step 0 (docs/api.md)
STEP_LIMIT = (0.10472, 0.10472, 0.15708, 0.15708, 0.15708, 0.15708)
SAMPLE_DIR = os.path.join(HERE, "sample_obs")  # step 0 of bolt_to_tray seed 0, as the engine renders it
EXAMPLE_ENTRY = "policy.example:ExamplePolicy"  # the only entry point the tiny generated checkpoint fits
ENTRY_POINT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*$")
MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
PATH_PART_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*$")


def entry_kind(value) -> str:
    """'class', 'module' or 'file'; raises ValueError with the reason otherwise (same grammar as rc_runtime)."""
    if not isinstance(value, str) or not value:
        raise ValueError("entry_point must be a non-empty string")
    if ":" in value:
        if ENTRY_POINT_RE.fullmatch(value):
            return "class"
        raise ValueError('a Policy class entry point must look like "package.module:Attr"')
    if "/" in value or value.endswith(".py") or "\\" in value:
        parts = value.split("/")
        if value.startswith("/") or re.match(r"^[A-Za-z]:", value):
            raise ValueError("absolute paths are not allowed; use a path relative to the repository root")
        if "\\" in value or any(p in ("", ".", "..") for p in parts) or not all(PATH_PART_RE.match(p) for p in parts):
            raise ValueError("a program path must stay inside the repository (no '..', '.', empty segments, backslashes)")
        if not value.endswith(".py"):
            raise ValueError("a program path must end in .py")
        return "file"
    if MODULE_RE.fullmatch(value):
        return "module"
    raise ValueError('entry_point must be "package.module:Attr", a dotted module path, or a relative .py path')


class FakePolicy(Policy):
    def load(self) -> None:
        time.sleep(0.3)  # simulate weight load

    def act(self, obs):
        return {"type": "joint_delta", "values": [0.0] * 6, "gripper": 0.0}


class FailedLoadPolicy(Policy):
    def load(self) -> None:
        raise RuntimeError(DELIBERATE_FAILURE)

    def act(self, obs):
        raise AssertionError("act must not run after failed load")


def sample_observation(step: int = 0, episode_id: str = "selftest") -> dict:
    """A /act request as the runner sends it (camera JPEGs from scripts/sample_obs/ when present)."""
    import base64

    def jpeg(name):
        path = os.path.join(SAMPLE_DIR, name + ".jpg")
        if not os.path.isfile(path):
            return None
        with open(path, "rb") as source:
            return base64.b64encode(source.read()).decode()
    obs = {"episode_id": episode_id, "step": step, "task_id": "bolt_to_tray",
           "instruction": "Move the gold bolt to the tray.", "schema_version": 2,
           "proprio": {"joint_pos": list(HOME_Q), "joint_vel": [0.0] * 6, "gripper": 0.9968,
                       "ee_pos": [0.8906, -0.0078, 0.2016], "ee_quat": [-0.0289, -0.6302, 0.7751, 0.0355]}}
    images = {name: jpeg(name) for name in ("wrist", "left", "right")}
    if all(images.values()):
        obs["images"] = images
    if jpeg("goal"):
        obs["goal"] = jpeg("goal")
    return obs


def _runner_motion():
    """The runner's own action boundary (rc_runner/motion.py, standard library only) when the participant
    bundle's runner/ sits next to this repository; None otherwise."""
    import importlib.util
    path = os.path.join(ROOT, "..", "runner", "rc_runner", "motion.py")
    if not os.path.isfile(path):
        return None
    spec = importlib.util.spec_from_file_location("_rc_runner_motion", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:  # noqa: BLE001 - fall back to the local copy of the rules
        return None
    return module if hasattr(module, "clamp_action") else None


def _number(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value or abs(value) == float("inf"):
        raise ValueError("expected a finite number")
    return float(value)


def local_check_action(action) -> list:
    """The runner's action rules (rc_runner/motion.py): raises ValueError for a held step, returns clamp notes."""
    if not isinstance(action, dict):
        raise ValueError("action must be an object")
    if action.get("type") == "action_chunk":
        if "values" in action or "gripper" in action:
            raise ValueError("both joint_delta and action_chunk fields")
        actions = action.get("actions")
        if not isinstance(actions, list) or not 1 <= len(actions) <= 16:
            raise ValueError("chunk must contain 1 to 16 actions")
        if "chunk_len" in action and (type(action["chunk_len"]) is not int or action["chunk_len"] != len(actions)):
            raise ValueError("chunk_len does not match actions")
    else:
        if "actions" in action or "chunk_len" in action:
            raise ValueError("both joint_delta and action_chunk fields")
        actions = [action]
    notes = []
    for index, item in enumerate(actions):
        if not isinstance(item, dict) or item.get("type") != "joint_delta":
            raise ValueError("expected joint_delta")
        if any(key not in ("type", "values", "gripper") for key in item):
            raise ValueError("unknown action field")
        values = item.get("values")
        if not isinstance(values, list) or len(values) != 6:
            raise ValueError("expected six joint deltas")
        for joint, value in enumerate(values):
            if abs(_number(value)) > STEP_LIMIT[joint] + 1e-9:
                notes.append(f"action {index} joint {joint} over the {STEP_LIMIT[joint]} rad step limit")
        if not 0.0 <= _number(item.get("gripper")) <= 1.0:
            notes.append(f"action {index} gripper outside [0, 1]")
    return notes


def check_action(action, label: str) -> str:
    """Validate an action like the runner; FAIL on what would be a held step. Returns a short summary."""
    motion = _runner_motion()
    try:
        if motion is not None:
            hits = []
            motion.clamp_action(action, list(HOME_Q), hits=hits)
            notes = sorted({f"{limit} {joint}" for element in hits for limit, joint in element})
        else:
            notes = local_check_action(action)
    except ValueError as exc:
        raise SystemExit(f"FAIL {label}: the runner would hold this step instead ({exc}): {json.dumps(action)[:300]}")
    for note in notes:
        report("NOTE", f"{label}: clamped by the runner: {note}")
    count = len(action["actions"]) if action.get("type") == "action_chunk" else 1
    return f"valid {action.get('type')} ({count} action{'s' if count > 1 else ''})"


def request(port: int, method: str, path: str, body=None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"} if body else {})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read()), dict(resp.headers)
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"{}"), dict(err.headers)


def make_server(policy_class):
    """The judged image's server when rc_runtime is available, else the rc_sdk server."""
    if RuntimeServer is not None:
        def loader():
            policy = policy_class()
            rt_loader.run_load(policy)
            return policy
        return RuntimeServer("127.0.0.1", 0, loader)
    return PolicyServer(0, policy_class())


def check_manifest() -> dict:
    manifest_path = os.path.join(ROOT, "manifest.json")
    try:
        with open(manifest_path, encoding="utf-8") as source:
            manifest = json.load(source)
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be a JSON object")
        if manifest.get("contract") != "v0":
            raise ValueError('contract must be "v0"')
        entry_kind(manifest.get("entry_point"))
        if "port" in manifest and (type(manifest["port"]) is not int or manifest["port"] != 8080):
            raise ValueError("port must be 8080 or omitted (the image always serves on 8080)")
        weights = manifest.get("weights")
        if weights is not None and not isinstance(weights, dict):
            raise ValueError("weights must be an object when present (placeholder; ignored)")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"FAIL manifest.json: {exc}") from exc
    report("PASS", "manifest.json is valid (model link is supplied in the dashboard; weights not downloaded)")
    return manifest


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def check_program(root, entry: str, label: str, weights_dir=None) -> None:
    """Start a program-form entry point for real and assert contract v0 over HTTP (no torch needed)."""
    from pathlib import Path
    root = Path(root)
    with tempfile.TemporaryDirectory() as weights, tempfile.TemporaryDirectory() as scratch:
        if weights_dir:
            weights = os.path.abspath(weights_dir)
        else:
            write_example_weights(weights, scale=0.1)
        port = free_port()
        env = {k: v for k, v in os.environ.items() if not k.startswith("RC_")}
        env.update(RC_WEIGHTS_DIR=weights, RC_SCRATCH=scratch, RC_POLICY_PORT=str(port), PYTHONUNBUFFERED="1")
        if rt_loader is not None:
            env["PYTHONPATH"] = os.pathsep.join([*SDK_DIRS, os.path.join(ROOT, "..", "runtime")])
            cmd = [sys.executable, "-m", "rc_runtime", "--submission", str(root), "--port", str(port)]
            how = "rc_runtime runs it as __main__"
        else:
            cmd = [sys.executable, entry] if entry.endswith(".py") else [sys.executable, "-m", entry]
            how = "plain python (rc_runtime not importable here)"
        log = tempfile.TemporaryFile(mode="w+")
        proc = subprocess.Popen(cmd, cwd=str(root), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)

        def fail(message):
            proc.kill()
            proc.wait()
            log.seek(0)
            raise SystemExit(f"FAIL {label}: {message}\n--- program output ---\n{log.read()[-3000:]}")

        try:
            deadline = time.monotonic() + 30
            while True:
                if proc.poll() is not None:
                    fail(f"program exited early with status {proc.returncode}")
                try:
                    status, body, _ = request(port, "GET", "/healthz")
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        fail(f"nothing listening on 0.0.0.0:$RC_POLICY_PORT ({port}) after 30 s")
                    time.sleep(0.1)
            assert status == 200 and body == {}, (status, body)
            report("PASS", f"{label}: /healthz 200 {{}} ({how})")
            while True:
                status, body, _ = request(port, "GET", "/readyz")
                if body.get("ready"):
                    break
                if body.get("error") or time.monotonic() > deadline:
                    fail(f"/readyz never became ready: {body}")
                time.sleep(0.1)
            assert status == 200 and body["contract"] == "v0" and body["schema_version"] == 2, body
            assert set(body["capabilities"]) >= {"chunking", "caching"} and body["capabilities"]["chunking"] is True, body
            report("PASS", f"{label}: /readyz ready after weights load, contract v0, schema_version 2")
            status, body, headers = request(port, "POST", "/act", sample_observation(3, "ep-9"))
            assert status == 200 and body["episode_id"] == "ep-9" and body["step"] == 3, (status, body)
            low = {k.lower(): v for k, v in headers.items()}  # header names are case-insensitive
            assert low.get("x-rc-episode-id") == "ep-9" and low.get("x-rc-step") == "3", headers
            summary = check_action(body["action"], label)
            report("PASS", f"{label}: /act echoes episode_id/step (body + headers) and returns a {summary}")
            status, body, _ = request(port, "POST", "/act", {"episode_id": "ep-9"})
            assert status == 400 and "step" in body["error"], (status, body)
            report("PASS", f"{label}: malformed /act (missing step) -> 400")
        finally:
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()


def stage_submission(source) -> "Path":
    import shutil
    from pathlib import Path
    import atexit
    staged = tempfile.mkdtemp(prefix="rc-submission-")
    atexit.register(shutil.rmtree, staged, True)
    root = Path(staged) / "submission"
    shutil.copytree(source, root, ignore=shutil.ignore_patterns(
        "rc_sdk", "Dockerfile", "Dockerfile.legacy", ".git", "__pycache__", "scripts", "examples"))
    return root


def check_example_programs() -> None:
    import importlib.util
    examples = os.path.join(ROOT, "examples")
    if not os.path.isdir(examples):
        return
    for name in sorted(os.listdir(examples)):
        folder = os.path.join(examples, name)
        manifest_path = os.path.join(folder, "manifest.json")
        if not os.path.isfile(manifest_path):
            continue
        with open(manifest_path, encoding="utf-8") as source:
            entry = json.load(source)["entry_point"]
        label = f"examples/{name}"
        try:
            kind = entry_kind(entry)
        except ValueError as exc:
            raise SystemExit(f"FAIL {label}: manifest entry_point: {exc}")
        assert kind != "class", f"{label} is expected to be a program"
        if "fastapi" in name and (importlib.util.find_spec("fastapi") is None or importlib.util.find_spec("uvicorn") is None):
            report("SKIP", f"{label}: fastapi/uvicorn are not installed here (they are in the judged image)")
            continue
        check_program(stage_submission(folder), entry, label)


def check_entry_point(entry: str, weights_dir=None):
    """Resolve the entry point the way the image does, then load() and act() it when it can run here.
    Returns None when load()/act() ran, else the reason they did not."""
    # The judged /submission is your repository without a Dockerfile, .git or any rc_sdk copy
    # (rc_sdk comes from the image). Mimic it so rc_sdk is imported from outside the tree.
    root = stage_submission(ROOT)
    if entry_kind(entry) != "class":
        check_program(root, entry, "your program entry point", weights_dir)
        return None
    try:
        if rt_loader is not None:
            reserved = rt_loader.load_reserved_modules(env={})
            policy = rt_loader.resolve_entry(root, entry, reserved)
            report("PASS", "entry point resolved by rc_runtime.loader (path appended last, shadow checks, Policy resolved)")
        else:
            import importlib
            sys.path.append(str(root))
            module, attr = entry.split(":")
            policy = getattr(importlib.import_module(module), attr)()
            report("PASS", f"{entry} imported (rc_runtime not importable here, so with importlib only; "
                           "`python -I -m rc_runtime --selfcheck` in the image runs the loader checks)")
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, ImportError) or "ImportError" in str(exc) or "ModuleNotFoundError" in str(exc):
            report("SKIP", f"resolving {entry}: it imports a package that is not installed here ({exc}). The judged "
                           "image has the supported packages (see SUBMIT.md); install them locally or run "
                           "`python -I -m rc_runtime --selfcheck --submission /submission` inside the image.")
            return f"{entry} imports a package that is not installed here ({exc})"
        raise
    assert isinstance(policy, Policy), "entry point must resolve to a Policy"
    with tempfile.TemporaryDirectory() as generated:
        if weights_dir:
            weights = os.path.abspath(weights_dir)
        elif entry == EXAMPLE_ENTRY:
            write_example_weights(generated)  # the same file scripts/make_example_weights.py writes
            weights = generated
        else:
            return "no weights here: pass --weights DIR (your downloaded model snapshot)"
        os.environ["RC_WEIGHTS_DIR"] = weights
        t0 = time.monotonic()
        try:
            policy.load()
        except ImportError as exc:
            return f"load() imports a package that is not installed here ({exc}; the judged image has it)"
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"FAIL load() with RC_WEIGHTS_DIR={weights}: {type(exc).__name__}: {exc}") from exc
        loaded = time.monotonic() - t0
        t0 = time.monotonic()
        try:
            action = policy.act(sample_observation())
        except ImportError as exc:
            return f"act() imports a package that is not installed here ({exc}; the judged image has it)"
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"FAIL act() on a step-0 observation: {type(exc).__name__}: {exc}") from exc
        acted = time.monotonic() - t0
    summary = check_action(action, "act()")
    report("PASS", f"load() in {loaded:.1f} s, act() in {acted * 1000:.0f} ms on a step-0 observation: {summary}")
    return None


COUNTS = {"PASS": 0, "SKIP": 0, "NOTE": 0}
STDOUT = sys.stdout  # report() writes here even while the SDK contract check mutes its own demo servers


def report(tag: str, text: str) -> None:
    """One result line: a four-letter tag, then what was checked ("PASS manifest.json is valid ...")."""
    COUNTS[tag] = COUNTS.get(tag, 0) + 1
    STDOUT.write(f"{tag} {text}\n")
    STDOUT.flush()


SECTION = {"name": ""}


def section(number: int, title: str) -> None:
    SECTION["name"] = title
    STDOUT.write(f"\n[{number}/4] {title}\n")
    STDOUT.flush()


def contract_only_step(reason: str) -> str:
    if reason.startswith("no weights here"):
        return "re-run with --weights DIR pointing at your downloaded model snapshot"
    if "not installed here" in reason:
        return ("install that package locally and re-run, or run the practice tool (contestants/PRACTICE.md), "
                "which uses the judged image")
    return "re-run with --weights DIR, or run the practice tool (contestants/PRACTICE.md)"


def check_sdk_server() -> None:
    """The SDK server's HTTP contract, on demo policies of this script (their server logs and the deliberate
    load-failure traceback are kept out of your output, and shown only if a check here fails)."""
    import contextlib
    import io
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            _check_sdk_server()
    except BaseException:
        sys.stderr.write("--- demo server output ---\n" + captured.getvalue()[-3000:])
        raise


def _check_sdk_server() -> None:
    server = make_server(FakePolicy)
    port = server.server_address[1]
    t0 = time.monotonic()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    server.start_load()

    status, body, _ = request(port, "GET", "/healthz")
    elapsed = time.monotonic() - t0
    assert status == 200 and body == {}, (status, body)
    assert elapsed < 0.2, f"healthz took {elapsed:.3f}s (liveness only!)"
    report("PASS", f"/healthz 200 {{}} immediately (after {elapsed * 1000:.0f} ms, pre-weights)")

    status, body, _ = request(port, "GET", "/readyz")
    assert status == 200 and body["ready"] is False, (status, body)
    assert body["contract"] == "v0" and body["schema_version"] == 2
    assert body["capabilities"] == {"chunking": True, "caching": False}
    report("PASS", "/readyz not-ready before load, contract v0, schema_version 2, capabilities")

    time.sleep(0.5)
    status, body, _ = request(port, "GET", "/readyz")
    assert status == 200 and body["ready"] is True, (status, body)
    report("PASS", "/readyz flips ready after load() completes")

    obs = {"episode_id": "ep-123", "step": 7, "task_id": "t1",
           "images": {"wrist": "PLACEHOLDER", "left": "PLACEHOLDER", "right": "PLACEHOLDER"},
           "proprio": {"joint_pos": [0.0] * 6}}
    status, body, headers = request(port, "POST", "/act", obs)
    assert status == 200, (status, body)
    assert body["episode_id"] == "ep-123" and body["step"] == 7  # verbatim echo
    assert body["action"]["type"] == "joint_delta"
    assert headers.get("X-RC-Episode-Id") == "ep-123" and headers.get("X-RC-Step") == "7"
    report("PASS", "/act echoes episode_id/step verbatim (body + X-RC-* headers) + action")

    status, body, _ = request(port, "POST", "/act", {"episode_id": "ep-123"})
    assert status == 400 and "step" in body["error"], (status, body)
    report("PASS", "malformed /act (missing step) -> 400 with readable message")

    server.shutdown()

    failed_server = make_server(FailedLoadPolicy)
    failed_port = failed_server.server_address[1]
    threading.Thread(target=failed_server.serve_forever, daemon=True).start()
    failed_server.start_load()
    for _ in range(100):
        if failed_server.load_error is not None:
            break
        time.sleep(0.01)
    status, body, _ = request(failed_port, "GET", "/readyz")
    assert status == 200 and body["ready"] is False, (status, body)
    assert body["error"] == "policy failed to load", body
    assert DELIBERATE_FAILURE not in json.dumps(body), body
    failed_server.shutdown()
    report("PASS", "a failed load() stays not-ready and exposes only a bounded error")


def run(argv=None):
    """Every check in order; returns the reason load()/act() did not run, or None when they did."""
    import argparse
    parser = argparse.ArgumentParser(description="Self-test for your policy repository.")
    parser.add_argument("--weights", metavar="DIR", default=None,
                        help="run load() on this model snapshot (as mounted at /weights) and then act()")
    args = parser.parse_args(argv)
    if args.weights and not os.path.isdir(args.weights):
        raise SystemExit(f"FAIL --weights {args.weights}: not a directory (point it at your downloaded model "
                         "snapshot, or create the example one with python3 scripts/make_example_weights.py DIR)")
    section(1, "manifest.json")
    manifest = check_manifest()
    section(2, f"your entry point ({manifest['entry_point']})")
    skipped = check_entry_point(manifest["entry_point"], args.weights)
    if skipped:
        report("SKIP", f"load()/act() not run: {skipped}")
    section(3, "example servers in examples/")
    check_example_programs()
    section(4, "SDK server HTTP contract")
    check_sdk_server()
    return skipped


def main(argv=None) -> None:
    try:
        skipped = run(argv)
    except SystemExit as exc:
        if not isinstance(exc.code, str):
            raise
        message = exc.code.rstrip()
        sys.stderr.write(message + "\n")
        first = message.splitlines()[0]
        sys.stderr.write(f"\nFAILED: {first[5:] if first.startswith('FAIL ') else first}\n")
        sys.exit(1)
    except AssertionError as exc:
        sys.stderr.write(f"FAIL {SECTION['name']}: unexpected response {exc}\n")
        sys.stderr.write(f"\nFAILED: {SECTION['name']}: unexpected response {exc}\n")
        sys.exit(1)
    STDOUT.write(f"\n{COUNTS['PASS']} passed · {COUNTS['SKIP']} skipped · {COUNTS['NOTE']} notes\n")
    if skipped:
        STDOUT.write(f"CONTRACT ONLY: the contract checks passed, but your load()/act() were NOT run: {skipped}. "
                     f"Next step: {contract_only_step(skipped)}.\n")
    else:
        STDOUT.write("ALL PASS\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
