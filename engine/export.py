"""Export our policy to a submittable ONNX artifact, then prove it against the contract.

    PYTHONPATH=. python engine/export.py --out engine/submission.onnx

The export is deliberately paranoid about two things the evaluation stack punishes hardest:

  * **Static trailing dims.** `player/launch.py` reads onnxruntime's reported shapes, so a
    symbolic last dim can pass if ORT happens to infer it -- but that is a bet on the player
    image's ORT version, and losing it costs the WHOLE MEET (`submission_not_ready`, raw 0.0),
    not one attempt. We export with only axis 0 dynamic and assert the graph itself declares
    104/256/12/256.

  * **Parity between torch and ONNX.** The score comes from the ONNX graph, not the checkpoint.
    Any divergence means we tuned one artifact and submitted another, so this asserts they agree
    to 1e-5 over a threaded multi-step episode rather than on a single forward pass.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch

from engine.contract import ACT_DIM, OBS_DIM, OPSET, STATE_DIM, check_submission
from engine.model import OlympicsPolicy


def export(policy: OlympicsPolicy, out: pathlib.Path) -> None:
    policy = policy.eval()
    obs = torch.zeros(1, OBS_DIM)
    state = torch.zeros(1, STATE_DIM)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        policy, (obs, state), str(out),
        input_names=["obs", "state_in"], output_names=["action", "state_out"],
        # Only the batch axis is dynamic. Everything else is pinned so the player's shape check
        # never depends on ORT inference.
        dynamic_axes={k: {0: "batch"} for k in ("obs", "state_in", "action", "state_out")},
        opset_version=OPSET, dynamo=False)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {policy.n_params():,} params)")


def check_parity(policy: OlympicsPolicy, path: pathlib.Path, steps: int = 400,
                 tol: float = 1e-5) -> float:
    """Thread the same episode through torch and through onnxruntime; return the worst delta."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]

    rng = np.random.default_rng(0)
    st_t = torch.zeros(1, STATE_DIM)
    st_o = np.zeros((1, STATE_DIM), np.float32)
    worst = 0.0
    # Sweep obs[50] across every event's reset value so the latch, the geometry channel and the
    # sticky overhead bit are all exercised, not just the generic path.
    for k in range(steps):
        o = (rng.standard_normal((1, OBS_DIM)) * 0.4).astype(np.float32)
        o[0, 50] = [1.40, 2.30, 2.70, 10.20, 40.00][k % 5] - 0.001 * k
        o[0, 97:104] = 2.0 if k % 7 else 1.2
        with torch.no_grad():
            a_t, st_t = policy(torch.from_numpy(o), st_t)
        a_o, st_o = sess.run(None, {names[0]: o, names[1]: st_o})
        st_o = np.asarray(st_o, np.float32).reshape(1, STATE_DIM)
        worst = max(worst, float(np.abs(a_t.numpy() - np.asarray(a_o)).max()),
                    float(np.abs(st_t.numpy() - st_o).max()))
    if worst > tol:
        raise AssertionError(f"torch/ONNX parity {worst:.3e} > {tol:.0e}")
    return worst


def check_event_latch(path: pathlib.Path) -> None:
    """The event must be decided on step 1 and then never move, for every event."""
    import onnxruntime as ort

    from engine.contract import RESET_REMAINING_M
    from engine.model import E_HI, E_LO, EVENT_NAMES

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]

    expected = {"high_jump": "high_jump", "long_jump": "long_jump", "triple_jump": "triple_jump",
                "sprint_100": "flat_100", "hurdles_100": "flat_100", "sprint_400": "circuit_400"}
    for event, remaining in RESET_REMAINING_M.items():
        st = np.zeros((1, STATE_DIM), np.float32)
        first = None
        for k in range(300):
            o = np.zeros((1, OBS_DIM), np.float32)
            # Distance to the finish shrinks as the attempt proceeds -- the exact thing that
            # would reclassify the event if the latch were not holding.
            o[0, 50] = max(0.0, remaining - 0.12 * k) / 10.0
            o[0, 97:104] = 2.0
            _, st = sess.run(None, {names[0]: o, names[1]: st})
            st = np.asarray(st, np.float32).reshape(1, STATE_DIM)
            onehot = st[0, E_LO:E_HI]
            if onehot.sum() != 1.0:
                raise AssertionError(f"{event}: event one-hot is not one-hot at step {k}: {onehot}")
            cls = EVENT_NAMES[int(np.argmax(onehot))]
            first = first or cls
            if cls != first:
                raise AssertionError(f"{event}: latch broke at step {k}: {first} -> {cls}")
        if first != expected[event]:
            raise AssertionError(f"{event}: classified as {first}, expected {expected[event]}")
        print(f"   latch ok  {event:12s} remaining {remaining:6.1f} m -> {first}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="engine/submission.onnx")
    ap.add_argument("--ckpt", help="optional trained checkpoint to load before exporting")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model = OlympicsPolicy()
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
        print(f"loaded {args.ckpt}")

    out = pathlib.Path(args.out)
    export(model, out)

    print("\n-- torch/ONNX parity --")
    print(f"   worst |delta| over 400 threaded steps: {check_parity(model, out):.3e}")

    print("\n-- event latch --")
    check_event_latch(out)

    print("\n-- evaluation contract --")
    check_submission(out)
