"""Score a torch checkpoint with the referee's OWN scorer, without exporting to ONNX first.

Used as the in-training progress signal. Two properties matter:

* **It uses `env.scoring.instance_score` and `env.meet_score` directly**, so the number is the
  same quantity the round is decided on -- not a proxy, and not our shaped reward. Shaped reward
  going up while this stays flat is the failure mode worth catching early, and it is invisible
  unless you measure the real thing.

* **It is multi-seed by default.** Per-seed sd on this meet is ~0.025 against a 1% takeover
  margin of ~0.006, so a single-seed comparison cannot resolve an improvement. Two seeds during
  training is a progress signal, not a verdict; use >= 8 before believing a delta.

    PYTHONPATH=. python engine/evaluate.py --ckpt engine/runs/ppo.pt --seeds 4
"""

from __future__ import annotations

import argparse
import statistics


def evaluate_policy(policy, seeds=(1, 2), instances_per_event: int = 4, step_cap: int = 0,
                    use_mjb: bool = True, verbose: bool = False) -> dict:
    import numpy as np
    import torch

    from env import EVENTS, instance_score, meet_score
    from env.sim import OlympicsSim, STATE_DIM, event_instances

    if use_mjb:
        from engine.mjb import build
        for event in EVENTS:
            build(event)

    policy = policy.eval()
    per_seed, rows_all = [], []
    for seed in seeds:
        rows = []
        # Attempts are grouped by event so the referee holds one compiled scene at a time;
        # mirroring that here keeps the .mjb swap to five transitions per meet.
        for params in sorted(event_instances(instances_per_event, seed),
                             key=lambda p: EVENTS.index(p.event)):
            if use_mjb:
                from engine.mjb import install
                install(params.event)
            sim = OlympicsSim(params)
            obs = sim.reset()
            state = torch.zeros(1, STATE_DIM)
            cap = min(sim.max_steps, step_cap) if step_cap else sim.max_steps
            reason = None
            with torch.no_grad():
                while reason is None:
                    action, state = policy(
                        torch.from_numpy(obs.reshape(1, -1).astype(np.float32)), state)
                    result = sim.step(action.numpy().ravel().astype(np.float64), max_steps=cap)
                    obs, reason = result.obs, result.terminal_reason
            rows.append({"event": sim.event, "terminal_reason": reason,
                         "score": instance_score(sim.event, reason, sim.progress,
                                                 sim.steps, cap, sim.metrics)})
            del sim
        per_seed.append(meet_score(rows, EVENTS))
        rows_all.extend(rows)
        if verbose:
            print(f"   seed {seed}: {per_seed[-1]:.5f}")

    event_scores = {}
    for event in EVENTS:
        vals = [r["score"] for r in rows_all if r["event"] == event]
        event_scores[event] = statistics.fmean(vals) if vals else 0.0
    reasons = {}
    for r in rows_all:
        reasons[r["terminal_reason"]] = reasons.get(r["terminal_reason"], 0) + 1
    return {
        "raw_score": statistics.fmean(per_seed),
        "raw_sd": statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0,
        "per_seed": per_seed,
        "event_scores": event_scores,
        "reasons": reasons,
        "num_finished": sum(1 for r in rows_all
                            if r["terminal_reason"] in {"completed", "cleared", "landed"}),
        "n": len(rows_all),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--instances-per-event", type=int, default=4)
    args = ap.parse_args()

    import json

    import torch

    from engine.model import OlympicsPolicy

    policy = OlympicsPolicy()
    policy.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    result = evaluate_policy(policy, seeds=tuple(range(1, args.seeds + 1)),
                             instances_per_event=args.instances_per_event, verbose=True)
    print(json.dumps(result, indent=2))
