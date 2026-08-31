"""Phase 1: warm-start our policy from the published baseline walker.

`baseline/baseline.onnx` is Unitree's stock G1 locomotion policy (`motion.pt`, BSD-3) wrapped in
the competition interface by `tools/make_baseline.py`. It is the organisers' own published
reference, not another miner's artifact, and it already solves the one thing that is expensive to
learn from scratch: a stable bipedal gait. It cannot see terrain and it is slow (0.8 m/s forward
command), so it scores 0.0330 -- but as a GAIT PRIOR it removes the largest block of RL, which is
the ~200M steps a humanoid spends learning not to fall over.

This is DAgger, not plain behaviour cloning. Rolling out purely under the teacher would train the
student only on states the teacher visits; the student then drifts into states it has never seen
and falls. So `beta` anneals from 1.0 (teacher drives) toward 0.0 (student drives) while the
teacher keeps LABELLING every visited state -- the standard fix for compounding covariate shift.

Two mechanical details that are easy to get wrong:

  * The teacher is an LSTM carrying its state in the SAME 256-float channel our student uses for a
    completely different layout. The two states are threaded independently; they are never mixed.
  * `env/sim.py` caches one compiled MuJoCo model per event in a module global and
    `OlympicsSim.__init__` writes friction/wind into it. Two live sims of the same event in one
    process therefore corrupt each other. Rollouts here are strictly sequential for that reason.

    PYTHONPATH=. python engine/distill.py --iters 6 --episodes 24 --out engine/runs/warm.pt
"""

from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

from engine.model import IN_DIM, OBS_DIM, STATE_DIM, OlympicsPolicy
from env.course import EVENTS
from env.sim import OlympicsSim, instance_spec

BASELINE = "baseline/baseline.onnx"
BPTT = 64          # truncation length for the GRU


def teacher_session(path: str) -> tuple[ort.InferenceSession, list[str]]:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
    return sess, [i.name for i in sess.get_inputs()]


@torch.no_grad()
def collect(policy: OlympicsPolicy, sess, names, episodes: int, beta: float, seed: int,
            step_cap: int) -> tuple[list[dict], dict]:
    """Roll out under a beta-mix of teacher and student; label every state with the teacher."""
    policy.eval()
    out, stats = [], {"steps": 0, "reasons": {}}
    for ep in range(episodes):
        event = EVENTS[ep % len(EVENTS)]
        attempt = (ep // len(EVENTS)) % 4
        sim = OlympicsSim(instance_spec(event, attempt, seed + ep))
        obs = sim.reset()
        t_state = np.zeros((1, STATE_DIM), np.float32)     # teacher LSTM state
        s_state = torch.zeros(1, STATE_DIM)                 # our GRU + observers
        obs_log, act_log = [], []
        reason = None
        while reason is None and sim.steps < step_cap:
            o = obs.reshape(1, OBS_DIM).astype(np.float32)
            t_act, t_state = sess.run(None, {names[0]: o, names[1]: t_state})
            t_state = np.nan_to_num(np.asarray(t_state, np.float32).reshape(1, STATE_DIM))
            t_act = np.asarray(t_act, np.float32).ravel()

            s_act, s_state = policy(torch.from_numpy(o), s_state)
            obs_log.append(o[0].copy())
            act_log.append(t_act.copy())                    # label is ALWAYS the teacher

            drive = t_act if np.random.rand() < beta else s_act.numpy().ravel()
            result = sim.step(np.clip(drive, -10.0, 10.0).astype(np.float64))
            obs, reason = result.obs, result.terminal_reason
        stats["steps"] += sim.steps
        stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
        if obs_log:
            out.append({"obs": np.stack(obs_log), "act": np.stack(act_log), "event": event})
        del sim
    return out, stats


def fit_normalizer(policy: OlympicsPolicy, episodes: list[dict]) -> None:
    """Fit mean/istd on the 202-vector the encoder actually sees, under the CURRENT policy.

    D, the geometry channel and the sticky overhead bit are all functions of behaviour, so a
    normaliser fitted on random actions is wrong for a trained policy. Refit before every export.
    """
    feats = []
    with torch.no_grad():
        for ep in episodes:
            obs = torch.from_numpy(ep["obs"])
            state = torch.zeros(1, STATE_DIM)
            for t in range(obs.shape[0]):
                o = obs[t:t + 1]
                feats.append(policy.assemble(o, state)[0])
                _, state = policy(o, state)
    x = torch.stack(feats)
    policy.fit_normalizer(x)
    print(f"   normaliser fitted on {x.shape[0]:,} frames "
          f"(mean |mu| {policy.mean.abs().mean():.3f}, mean istd {policy.istd.mean():.3f})")


def train(policy: OlympicsPolicy, episodes: list[dict], epochs: int, lr: float,
          batch: int) -> float:
    """Truncated-BPTT behaviour cloning. Sequences, not shuffled frames: the student is
    recurrent, so its state has to be built the way it will be at evaluation time."""
    policy.train()
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    order = list(range(len(episodes)))
    last = float("nan")
    for epoch in range(epochs):
        np.random.shuffle(order)
        total, nseg = 0.0, 0
        for start in range(0, len(order), batch):
            group = [episodes[i] for i in order[start:start + batch]]
            T = min(len(g["obs"]) for g in group)
            obs = torch.from_numpy(np.stack([g["obs"][:T] for g in group]))     # [B,T,104]
            act = torch.from_numpy(np.stack([g["act"][:T] for g in group]))     # [B,T,12]
            state = torch.zeros(obs.shape[0], STATE_DIM)
            for chunk in range(0, T, BPTT):
                end = min(chunk + BPTT, T)
                state = state.detach()          # truncate the gradient, keep the value
                loss = 0.0
                for t in range(chunk, end):
                    pred, state = policy(obs[:, t], state)
                    loss = loss + F.mse_loss(pred, act[:, t])
                loss = loss / max(1, end - chunk)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                total += float(loss.detach())
                nseg += 1
        last = total / max(1, nseg)
        print(f"   epoch {epoch + 1}/{epochs}  bc_mse {last:.5f}", flush=True)
    return last


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=6, help="DAgger iterations")
    ap.add_argument("--episodes", type=int, default=24, help="episodes collected per iteration")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--step-cap", type=int, default=700)
    ap.add_argument("--teacher", default=BASELINE)
    ap.add_argument("--out", default="engine/runs/warm.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads() // 2))

    sess, names = teacher_session(args.teacher)
    policy = OlympicsPolicy()
    print(f"student {policy.n_params():,} params; teacher {args.teacher}")

    pool: list[dict] = []
    t0 = time.monotonic()
    for it in range(args.iters):
        # Teacher drives everything at first, then hands over. The teacher keeps labelling
        # throughout, which is what makes this DAgger rather than one-shot cloning.
        beta = 1.0 if it == 0 else max(0.0, 1.0 - it / max(1, args.iters - 1))
        fresh, stats = collect(policy, sess, names, args.episodes, beta,
                               seed=args.seed + 1000 * it, step_cap=args.step_cap)
        pool.extend(fresh)
        pool = pool[-args.episodes * 4:]          # keep a sliding window of recent states
        print(f"iter {it}  beta {beta:.2f}  +{len(fresh)} eps  {stats['steps']:,} steps  "
              f"pool {len(pool)}  {stats['reasons']}", flush=True)
        if it == 0:
            fit_normalizer(policy, pool)
        train(policy, pool, args.epochs, args.lr, args.batch)

    fit_normalizer(policy, pool)                   # refit under the final behaviour
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), out)
    print(f"\nwrote {out}  ({time.monotonic() - t0:.0f}s)")
