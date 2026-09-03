"""Search one event's action trajectory directly, with CEM, scored by the referee's own scorer.

PPO has spent >150M steps here without producing a single completion, and `instance_score` pays a
minimum of **0.25** for any completion against the 0.0194 the current artifact scores across the
whole meet. One cleared bar is worth `0.25/6 = +0.042` on the meet -- more than doubling it.

Why search rather than learn:

* **No shaping, no credit assignment, no basin.** CEM optimises `instance_score` itself. Every
  reward-shaping problem this project has hit -- `w_smooth` pricing the winning gait out, `fin`
  being unreachable, the fall gate having no airborne exemption -- simply does not apply.
* **The jumps are short.** `analysis/riv_0830_seed1.json` shows the field leader clearing the high
  bar at step 151 and landing the long jump at step 190. A 150-step horizon at 64 candidates x 4
  attempts is ~51k sim steps per iteration, a few seconds across 14 cores.
* **The geometry is fixed.** `bar_x` is always 12.0, the take-off board always 15.0. Only the bar
  height, friction stratum and wind vary across the four attempts -- and scoring the mean over all
  four is what keeps a solution from overfitting one operating point.

The trajectory is `knots x 12` joint offsets, linearly interpolated over the episode and fed
straight in as actions, so what comes out is open-loop. That is deliberate: it answers "is a
clearance reachable at all" before anything is spent on making it robust. To reach a submission it
has to be distilled into the event's own slice of `head1` (see `engine/model.py`), which the
per-event head exists to make possible without disturbing the other events.

    PYTHONPATH=. python tools/cem_event.py --event high_jump --iters 100
    PYTHONPATH=. python tools/cem_event.py --event long_jump --iters 100 --steps 260
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import pathlib
import time

import numpy as np

ACTION_CLIP = 10.0
_EVENT: str = ""
_SESSION = None      # onnxruntime session, when searching residuals on top of a policy


def _expand(knots: np.ndarray, steps: int) -> np.ndarray:
    """`[K, 12]` control points -> `[steps, 12]`, linearly interpolated and clipped to the env."""
    k = knots.shape[0]
    src = np.linspace(0.0, 1.0, k)
    dst = np.linspace(0.0, 1.0, steps)
    out = np.empty((steps, knots.shape[1]), np.float64)
    for j in range(knots.shape[1]):
        out[:, j] = np.interp(dst, src, knots[:, j])
    return np.clip(out, -ACTION_CLIP, ACTION_CLIP)


def _init(event: str, use_mjb: bool, policy: str | None) -> None:
    global _EVENT, _SESSION
    _EVENT = event
    if use_mjb:
        from engine.mjb import install
        install(event)
    if policy:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1
        _SESSION = ort.InferenceSession(policy, sess_options=opts,
                                        providers=["CPUExecutionProvider"])


def _score_one(args) -> float:
    """Mean `instance_score` over the four attempts. One live sim at a time, per env/sim.py."""
    from env.scoring import instance_score
    from env.sim import OlympicsSim, instance_spec

    from env.sim import OBS_DIM, STATE_DIM

    knots, steps, seed, shape_w = args
    total = 0.0
    for attempt in range(4):
        sim = OlympicsSim(instance_spec(_EVENT, attempt, seed))
        obs = sim.reset()
        # RESIDUAL search. A random open-loop joint trajectory cannot walk -- locomotion needs
        # feedback, and a from-scratch population scores ~0.8 m of a 14 m route, never reaching
        # the bar at 12. So the policy walks and CEM only perturbs it: the searched knots are
        # added on top. With `--policy` absent this degrades to the open-loop search, which is
        # only useful for confirming that.
        residual = _expand(knots, steps)
        state = np.zeros((1, STATE_DIM), np.float32)
        names = [i.name for i in _SESSION.get_inputs()] if _SESSION else []
        reason = None
        for t in range(steps):
            if _SESSION is None:
                a = residual[t]
            else:
                out, state = _SESSION.run(None, {names[0]: obs.reshape(1, OBS_DIM).astype(np.float32),
                                                 names[1]: state})
                state = np.asarray(state, np.float32).reshape(1, STATE_DIM)
                a = np.clip(np.asarray(out).ravel() + residual[t], -ACTION_CLIP, ACTION_CLIP)
            result = sim.step(a, max_steps=steps)
            obs, reason = result.obs, result.terminal_reason
            if reason is not None:
                break
        # `instance_score` alone is a flat zero on high_jump until the pelvis crosses the bar
        # plane (note 6 in engine/rewards.py: `_best_clearance` is written in exactly one place),
        # so a random population scores 0.0000 everywhere and CEM has nothing to climb. Route
        # fraction is the only signal available before a crossing. Weighted well under the 0.25
        # completion floor so it can never outrank an actual clearance.
        total += instance_score(_EVENT, reason or "timeout", sim.progress,
                                sim.steps, sim.max_steps, sim.metrics)
        total += shape_w * float(sim.progress)
        del sim
    return total / 4.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--event", required=True)
    ap.add_argument("--steps", type=int, default=200, help="episode horizon to search over")
    ap.add_argument("--knots", type=int, default=11, help="control points per joint")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--elite", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=2.0, help="initial exploration std")
    ap.add_argument("--sigma-min", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1, help="platform round seed")
    ap.add_argument("--workers", type=int, default=0, help="0 = cpu_count - 2")
    ap.add_argument("--init", help="warm-start from a saved .npy of knots")
    ap.add_argument("--out", default="analysis/cem_best.npy")
    ap.add_argument("--policy", help="ONNX policy to search residuals on top of; without it "
                                     "the search is open-loop and cannot produce a gait")
    ap.add_argument("--shape", type=float, default=0.05,
                    help="weight on route fraction, to give CEM a gradient before any "
                         "completion exists; must stay well under the 0.25 completion floor")
    ap.add_argument("--no-mjb", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    shape = (args.knots, 12)
    mean = np.load(args.init).reshape(shape) if args.init else np.zeros(shape)
    sigma = np.full(shape, args.sigma)

    workers = args.workers or max(1, mp.cpu_count() - 2)
    ctx = mp.get_context("spawn" if "fork" not in mp.get_all_start_methods() else "fork")
    best_score, best_knots, t0 = -1.0, mean.copy(), time.monotonic()

    print(f"CEM {args.event}: {args.knots} knots x 12, {args.steps} steps, "
          f"pop {args.pop} elite {args.elite}, {workers} workers, "
          f"{'residual on ' + args.policy if args.policy else 'OPEN-LOOP (no policy)'}")
    with ctx.Pool(workers, initializer=_init,
                  initargs=(args.event, not args.no_mjb, args.policy)) as pool:
        for it in range(args.iters):
            batch = mean[None] + sigma[None] * rng.standard_normal((args.pop, *shape))
            batch = np.clip(batch, -ACTION_CLIP, ACTION_CLIP)
            scores = np.array(pool.map(_score_one,
                                       [(c, args.steps, args.seed, args.shape)
                                        for c in batch]))
            elite = batch[np.argsort(-scores)[:args.elite]]
            mean = elite.mean(axis=0)
            # Floor the std so a converged population can still escape a local optimum; CEM
            # collapsing to zero variance in ~20 iterations is the usual way this search fails.
            sigma = np.maximum(elite.std(axis=0), args.sigma_min)
            if scores.max() > best_score:
                best_score, best_knots = float(scores.max()), batch[int(np.argmax(scores))].copy()
                pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                np.save(args.out, best_knots)
            print(f"  it {it:3d}  best {scores.max():.4f}  mean {scores.mean():.4f}  "
                  f"sigma {sigma.mean():.3f}  all-time {best_score:.4f}  "
                  f"[{time.monotonic()-t0:.0f}s]", flush=True)

    # 0.25 is the completion floor for every event; anything below it is partial credit only.
    verdict = "COMPLETION" if best_score >= 0.25 else "partial credit only"
    print(f"\nbest {best_score:.4f} -> {verdict}   saved {args.out}")
    print(f"meet contribution if it holds across the round: {best_score/6:.4f}")


if __name__ == "__main__":
    main()
