"""Policy base class — subclass this and swap in your model.

Schema v2: 6 arm `joint_delta` values (|Δq| ≤ 0.10472 rad base and shoulder, 0.15708 rad elbow and wrists)
plus gripper in [0, 1].
"""

from typing import Any, Dict


class Policy:
    """Your model behind the reality-check container contract (v0).

    Lifecycle:
      - `load()` runs in a background thread at server start; /readyz flips to
        ready when it returns. Load weights, build your model, warm up here.
      - `act(obs)` is called per step once ready.

    act() must be a PURE FUNCTION of the request envelope: same envelope for
    the same (episode_id, step) => valid action (deterministic recompute is
    fine). No external side effects; no assumed call rate. Judged evaluation
    does not retry /act; the practice/non-judged runner may retry byte-identical
    requests after a transport failure.
    """

    def load(self) -> None:
        """Load weights / initialize the model. Runs once, in the background.

        The judged runner mounts the selected public Hugging Face snapshot at
        /weights. The dashboard model link selects the current snapshot for each new submission.
        Load the mounted files rather than fetching a model by name at runtime.
        Do NOT commit weights to this repo.
        """
        raise NotImplementedError("implement load() to fetch/prepare your weights")

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """One observation in, one action dict out (schema v2: 6 joint_delta + gripper).

        `obs` is the parsed /act request envelope: {episode_id, step, task_id,
        instruction?, goal?, images?, proprio?}. `instruction` is the
        natural-language sentence for the task. `goal` is a 640×480 JPEG of
        the finished scene. Return the ACTION dict only — the server echoes
        episode_id/step for you.
        """
        raise NotImplementedError("implement act() to run your model")
