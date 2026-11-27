# rc-qwen-visionik-policy

Reality Check policy: a **Qwen3-VL-8B-Instruct planner** decides *what* to place *where*; a **classical vision + inverse-kinematics
controller** with the **placement-v2 finisher** does all the motion. Python only; the one model it loads is stored in the Hugging Face repo
[`w3joe/rc-qwen-visionik-policy`](https://huggingface.co/w3joe/rc-qwen-visionik-policy), mounted at `/weights` (`RC_WEIGHTS_DIR`).

This is exactly the stack that scored **0.672** (engine-v1, 16/24 episodes) on the 8651b1b judged-path runner; see [Judged results](#judged-results).

## How it works

Entry point: `manifest.json` -> `serve.py` (program form, contract v0, port 8080). The server listens first, then loads, and reports ready on `/readyz`
after the planner has been loaded and warmed with a dummy call.

1. **Plan (once per episode, step 0):** classical perception estimates the scene from the RGB cameras; Qwen3-VL-8B
   (`/weights/qwen3_vl_8b`, bf16, greedy) turns instruction + goal picture + candidate parts into an ordered goal list (`vlm_local/llm_planner.py`,
   `plan.py`). If the call fails or returns nothing executable, a hard-coded instruction parser is used (`HardcodedPlanner`).
2. **Execute (every later `/act`):** the vision-IK controller (`vlm_local/controller.py`, `vlm_local/classical/`): colour/shape segmentation and multi-view triangulation,
   pure-NumPy UR12e FK/Jacobian/damped-least-squares IK with bounded joint steps, grasp verification from proprioception, nudge recovery,
   per-goal skills (tray, bolt -> hole, nut -> hex pocket). Each `/act` costs milliseconds.
3. **Placement-v2 finisher** (`vlm_local/finisher.py`, `skills_v2.py`): size-aware, contact-guided insertion/seating with release gating and post-release verification.
   Defaults as judged: `PLACEMENT_V2` **on**, `PLACEMENT_V2_BOLT` **off** (the v2 bolt path regressed the M24 bolt; the older bolt skill is used).

`/readyz` advertises `capabilities.chunking: true` like the starter (the runner ignores the flag; this policy returns one action per call, as judged).

No network, nothing downloaded, nothing written outside `/scratch`. Weights path: `$RC_WEIGHTS_DIR/qwen3_vl_8b` (default `/weights/qwen3_vl_8b`), loaded with
`from_pretrained(path)` offline. Optional env (not needed for judging): `RC_PLANNER=hardcoded` (no VLM), `RC_VLM_PATH` (override the model dir), `RC_VLM_QUANT`, `PLACEMENT_V2`, `PLACEMENT_V2_BOLT`.

## Judged results

Real runner (8651b1b, engine `rc-engine:8651b1b4`, pinned policy image, judged plan, 60 s cumulative `/act` budget, 2 public + 4 hidden tasks, 4 repetitions = 24 episodes, one H100):

| | runner total (engine-v1) | engine-v2 points (Q=0 .. Q=1) |
|---|---|---|
| **run 1** | **0.672** (16/24) | **46.4 .. 66.5** |
| run 2 (replication, same seeds) | 0.660 | 46.2 .. 66.2 |

| task | success | v1 mean | v2 (Q=0 .. Q=1) | mean steps |
|---|---|---|---|---|
| bolt_and_nut_to_fixture | 3/4 | 0.762 | 54.8 .. 68.2 | 682 |
| bolt_to_tray | 4/4 | 0.972 | 70.0 .. 110.3 | 113 |
| hidden_1 | 4/4 | 0.892 | 70.0 .. 89.8 | 431 |
| hidden_2 | 2/4 | 0.574 | 36.8 .. 59.3 | 637 |
| hidden_3 | 3/4 | 0.762 | 53.3 .. 82.2 | 396 |
| hidden_4 | 0/4 | 0.199 | 4.3 | 1200 |

All 8 unsolved episodes ended on the 1200-step limit; no crash, timeout, error or `/act` budget overrun. Per-episode `/act` time is at most ~30 s cumulative
(single `/act` <= ~6.7 s, planner call 3.3-4.8 s, per-step p50 ~2.5 ms), policy memory peak 19.6 GiB, model load ~17-19 s with warm page cache. Each 24-episode run took ~43 min of wall time (engine-bound) on that setup.
The seeds of both runs are identical, so the effective sample is 4 episodes per task: +-0.05 on the total is noise, and the replication is not independent evidence.
The v2 range depends on placement quality Q, which the runner does not report. Note that the contest rules now give 90 s of cumulative `/act` time per repetition (the judged run used 60 s), so there is more headroom than measured.

## Layout

```
manifest.json     {"contract": "v0", "entry_point": "serve.py", "port": 8080}
serve.py          HTTP server (/healthz, /readyz, /act), loads the planner then the policy
vlm_local/        policy.py, controller.py, plan.py, llm_planner.py, backends.py (HF Qwen3-VL loader), finisher.py, skills_v2.py
vlm_local/classical/   vision.py, planner.py (skills), kinematics.py, scene_interface.py
rc_sdk/ scripts/  starter files (selftest, sample observations), unchanged; the judged image ships its own rc_sdk
```

## Run it

The judged image has the dependencies (torch, transformers, ...); the host usually does not.

```sh
HF=w3joe/rc-qwen-visionik-policy; REV=<hf-commit-sha>          # pinned in the benchmark notes
IMG=ghcr.io/lucasucal/rc-policy@sha256:3c0eeb270fce3535799b6677ad2066434212476109efd2e415d86a8b33e93db8

# 1. snapshot of the model at the pinned commit (while the repo is private: hf auth login first)
hf download $HF --revision $REV --local-dir /tmp/snapshot && rm -rf /tmp/snapshot/.cache      # a .cache/ in the weights root fails the runner's weights check

# 2. self-test: manifest, HTTP contract, then load() + a real step-0 /act on the real weights (~1-2 min); must end with ALL PASS
docker run --rm --gpus device=0 --network none --user 10001:10001 --tmpfs /scratch:rw,size=8g,exec -e TMPDIR=/scratch/tmp \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e RC_WEIGHTS_DIR=/weights -v "$PWD":/submission:ro -v /tmp/snapshot:/weights:ro -w /submission \
  --entrypoint python $IMG scripts/selftest.py --weights /weights

# 3. practice / timing test, from the participant bundle root (the repo must be a git commit)
runner/.venv/bin/python -m practice run --repo /path/to/this/checkout \
  --model https://huggingface.co/w3joe/rc-qwen-visionik-policy --revision $REV --judged --seeds 0-1 \
  --tasks bolt_to_tray,bolt_and_nut_to_fixture                     # public tasks only; drop --tasks/--seeds for the full judged plan (6 tasks x 4 reps)
```

## Go public and submit

1. Hugging Face model repo: Settings -> change visibility to public, confirm **not gated**. Do not delete the repo or rewrite the pinned commit.
2. This GitHub repo: `gh repo edit w3joe/rc-qwen-visionik-policy --visibility public --accept-visibility-change-consequences`. It has a single commit; any later change must be a new commit, and the SHA to submit changes.
3. EngineExchange -> Policy runs: repo `https://github.com/w3joe/rc-qwen-visionik-policy` + the **git commit SHA** of the submitted commit; model `https://huggingface.co/w3joe/rc-qwen-visionik-policy` + the **HF commit SHA** (both SHAs are recorded in the benchmark notes' SUBMISSION.md; GitHub shows this repo's SHA as the latest commit).

## Licence

Code: Apache-2.0 (see `LICENSE`, `NOTICE`). Qwen3-VL-8B-Instruct: Apache-2.0, redistributed unmodified in the HF repo with its licence file.
