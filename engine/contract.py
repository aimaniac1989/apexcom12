"""The evaluation contract our submission must satisfy, transcribed from the deployed repo.

Every constant here is read off `spec.yaml`, `input.schema.json`, `player/launch.py`,
`referee/referee.py`, `referee/gym_v1/client.py`, `env/sim.py` and `env/scoring.py` at
humanoid_olympics v0.4.0. `check_submission()` enforces the parts a submission can violate.

Three layers can zero us, and they fail differently:

  L1  platform screener   runs BEFORE upload; rejects the artifact outright
  L2  player load         a failure means is_ready() False -> the referee writes
                          raw_score 0.0 / "submission_not_ready" for the WHOLE MEET
  L3  runtime             per-attempt typed zeros, plus one meet-wide wall-clock budget
"""

from __future__ import annotations

import pathlib
import time

import numpy as np

# -- interface (player/launch.py, env/sim.py) ------------------------------------------------
OBS_DIM, ACT_DIM, STATE_DIM = 104, 12, 256
OPSET = 17

# -- L1 platform screener (spec.yaml: screening) ---------------------------------------------
MAX_SIZE_MB = 15
MIN_WEIGHT_BYTES = 10_000

# -- L3 runtime (referee/referee.py, env/sim.py) ---------------------------------------------
MAX_ACTION_LEN = 1024          # longer list -> invalid_action -> 0.0
ACTION_ENV_CLIP = 10.0         # env/sim.py: np.clip(a, -10, 10) before the PD target
ACTION_SCALE = 0.25            # target = clip(a) * 0.25 + DEFAULT_ANGLES
DEADLINE_MS = 500              # input.schema.json: const
EVALUATION_BUDGET_S = 840.0    # referee stops SCHEDULING new attempts past this
INSTANCES_PER_EVENT = 4        # const
WIND_MAX_MS = 8.0              # const

# Events run in this order; each attempt has its own step cap (env/course.EVENT_MAX_STEPS).
# The budget matters because it is consumed IN THIS ORDER: overrun and the JUMPS are what
# get `round_timeout` rows worth 0.0 -- and they stay in the denominator.
EVENT_ORDER = ("sprint_100", "sprint_400", "hurdles_100", "high_jump", "long_jump", "triple_jump")
EVENT_MAX_STEPS = {"sprint_100": 1200, "sprint_400": 3600, "hurdles_100": 1900,
                   "high_jump": 900, "long_jump": 1000, "triple_jump": 1400}
MAX_CALLS = sum(EVENT_MAX_STEPS[e] for e in EVENT_ORDER) * INSTANCES_PER_EVENT   # 40,000
LONGEST_EPISODE = max(EVENT_MAX_STEPS.values())                                  # 3,600

# Terminal reasons that score a HARD ZERO rather than partial progress (env/scoring.py).
ZERO_REASONS = frozenset({"physics_glitch", "high_foul", "invalid_action", "jump_foul",
                          "player_error", "submission_not_ready", "out_of_bounds"})

# Route length at reset, per event -- this is what obs[50]*10 reads on the first step, and it
# is the only event identifier the policy ever gets. sprint_100 and hurdles_100 COLLIDE at
# 102.0 m; nothing in the observation distinguishes them on step 1.
RESET_REMAINING_M = {"sprint_100": 102.0, "sprint_400": 400.0, "hurdles_100": 102.0,
                     "high_jump": 14.0, "long_jump": 23.0, "triple_jump": 27.0}


class ContractError(AssertionError):
    """The artifact would be rejected or zeroed by the evaluation stack."""


def check_submission(path: str | pathlib.Path, verbose: bool = True) -> dict:
    """Run every gate a submission can fail. Raises ContractError on the first violation.

    This deliberately uses onnxruntime's reported shapes, not the graph proto: `player/launch.py`
    validates `session.get_inputs()/get_outputs()`, so ORT's shape inference is what actually
    decides. A graph with a symbolic last dim can still pass if ORT resolves it -- but relying on
    that is relying on the player image's ORT version, so we additionally require the graph
    itself to declare static trailing dims.
    """
    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper

    path = pathlib.Path(path)
    report: dict = {"path": str(path)}

    # -- L1 ----------------------------------------------------------------------------------
    size_mb = path.stat().st_size / 1e6
    report["size_mb"] = round(size_mb, 3)
    if size_mb > MAX_SIZE_MB:
        raise ContractError(f"L1 size {size_mb:.2f} MB > {MAX_SIZE_MB} MB")

    model = onnx.load(str(path))
    weight_bytes = sum(numpy_helper.to_array(i).nbytes for i in model.graph.initializer)
    report["weight_bytes"] = weight_bytes
    report["params"] = sum(numpy_helper.to_array(i).size for i in model.graph.initializer)
    if weight_bytes < MIN_WEIGHT_BYTES:
        raise ContractError(f"L1 weight_bytes {weight_bytes} < {MIN_WEIGHT_BYTES}")

    opsets = {(o.domain or "ai.onnx"): o.version for o in model.opset_import}
    report["opset"] = opsets
    if opsets.get("ai.onnx") != OPSET:
        raise ContractError(f"opset {opsets} != ai.onnx {OPSET}")

    # Static trailing dims in the graph itself, so we do not depend on ORT inference.
    def last_dim(vi):
        dims = vi.type.tensor_type.shape.dim
        return dims[-1].dim_value if dims and dims[-1].HasField("dim_value") else None

    for vi, want, what in ((model.graph.input[0], OBS_DIM, "obs"),
                           (model.graph.input[1], STATE_DIM, "state_in"),
                           (model.graph.output[0], ACT_DIM, "action"),
                           (model.graph.output[1], STATE_DIM, "state_out")):
        if last_dim(vi) != want:
            raise ContractError(
                f"graph declares {what} trailing dim {last_dim(vi)!r}, need static {want} "
                "(a symbolic dim may pass via ORT inference but is a bet on the player's "
                "onnxruntime version)")

    # -- L2: player/launch.py::_load_session, replicated exactly ------------------------------
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=opts,
                                   providers=["CPUExecutionProvider"])
    ins, outs = session.get_inputs(), session.get_outputs()
    if len(ins) != 2 or len(outs) != 2:
        raise ContractError(f"L2 needs exactly 2 inputs / 2 outputs, got {len(ins)}/{len(outs)}")
    for tensor, want, what in ((ins[0], OBS_DIM, "input 0 (obs)"),
                               (ins[1], STATE_DIM, "input 1 (state_in)"),
                               (outs[0], ACT_DIM, "output 0 (action)"),
                               (outs[1], STATE_DIM, "output 1 (state_out)")):
        if tensor.type != "tensor(float)":
            raise ContractError(f"L2 {what} must be float32, got {tensor.type}")
        if len(tensor.shape) != 2 or tensor.shape[-1] != want:
            raise ContractError(f"L2 {what} must be [batch, {want}], got {tensor.shape}")
    names = [i.name for i in ins]
    report["io"] = [(t.name, t.shape) for t in list(ins) + list(outs)]

    # -- L3: behaviour under the real calling convention --------------------------------------
    rng = np.random.default_rng(0)

    def step(obs, state):
        a, s = session.run(None, {names[0]: obs, names[1]: state})
        return (np.asarray(a, np.float32).reshape(1, ACT_DIM),
                np.asarray(s, np.float32).reshape(1, STATE_DIM))

    # determinism: the referee re-scores the incumbent, so identical inputs must give
    # identical actions or our own score is not reproducible.
    o0 = rng.standard_normal((1, OBS_DIM), dtype=np.float32)
    z = np.zeros((1, STATE_DIM), np.float32)
    a1, _ = step(o0, z)
    a2, _ = step(o0, z)
    if not np.array_equal(a1, a2):
        raise ContractError("L3 non-deterministic: same (obs, state) gave different actions")

    # A full-length episode of the longest event, threading state exactly as the player does.
    # Anything that drifts to NaN/inf here is invalid_action -> 0.0 for that attempt.
    state = np.zeros((1, STATE_DIM), np.float32)
    worst_abs = 0.0
    for t in range(LONGEST_EPISODE):
        obs = (rng.standard_normal((1, OBS_DIM), dtype=np.float32) * 0.5).astype(np.float32)
        obs[0, 50] = max(0.0, (400.0 - 0.11 * t)) / 10.0     # remaining distance, shrinking
        action, state = step(obs, state)
        if not np.all(np.isfinite(action)):
            raise ContractError(f"L3 non-finite action at step {t}")
        if not np.all(np.isfinite(state)):
            raise ContractError(f"L3 non-finite state_out at step {t}")
        worst_abs = max(worst_abs, float(np.abs(action).max()))
    report["max_abs_action_3600_steps"] = round(worst_abs, 4)
    report["state_absmax_after_episode"] = round(float(np.abs(state).max()), 4)

    # Hostile observations: the env never emits these, but a bounded head should survive them
    # rather than rail or overflow.
    for name, obs in (("zeros", np.zeros((1, OBS_DIM), np.float32)),
                      ("large", np.full((1, OBS_DIM), 1e3, np.float32)),
                      ("neg_large", np.full((1, OBS_DIM), -1e3, np.float32))):
        action, _ = step(obs, np.zeros((1, STATE_DIM), np.float32))
        if not np.all(np.isfinite(action)):
            raise ContractError(f"L3 non-finite action on {name} observation")
        if np.abs(action).max() > ACTION_ENV_CLIP + 1e-4:
            raise ContractError(f"L3 action {np.abs(action).max():.3f} exceeds the env clip "
                                f"+-{ACTION_ENV_CLIP} on {name} -- the head is not bounded")

    # -- latency against the meet-wide budget -------------------------------------------------
    obs = rng.standard_normal((1, OBS_DIM), dtype=np.float32)
    state = np.zeros((1, STATE_DIM), np.float32)
    for _ in range(50):
        step(obs, state)
    t0 = time.perf_counter()
    N = 500
    for _ in range(N):
        step(obs, state)
    ms = (time.perf_counter() - t0) / N * 1000
    report["ms_per_step_local"] = round(ms, 4)
    report["inference_s_at_max_calls"] = round(ms * MAX_CALLS / 1000, 1)
    report["budget_s"] = EVALUATION_BUDGET_S
    if ms > DEADLINE_MS:
        raise ContractError(f"L3 {ms:.1f} ms/step exceeds the {DEADLINE_MS} ms action deadline")

    if verbose:
        print(f"PASS  {path.name}")
        for k, v in report.items():
            print(f"   {k:32s} {v}")
    return report


if __name__ == "__main__":
    import sys
    for arg in sys.argv[1:]:
        check_submission(arg)
