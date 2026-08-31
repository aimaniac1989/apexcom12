"""Score an ONNX policy against a balanced Humanoid Olympics meet in-process."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time

import numpy as np
import onnxruntime as ort

from env import (EVENTS, OBS_DIM, STATE_DIM, OlympicsSim, event_instances, instance_score,
                 meet_score)
from env.history import DEFAULT_STRIDE, InstanceRecorder, write_instance


def rollout(session, sim: OlympicsSim, obs: np.ndarray, step_cap: int | None,
            rec: InstanceRecorder | None = None) -> str:
    state = np.zeros((1, STATE_DIM), np.float32)
    names = [item.name for item in session.get_inputs()]
    reason = None
    while reason is None:
        action, state = session.run(None, {names[0]: obs.reshape(1, OBS_DIM), names[1]: state})
        result = sim.step(np.asarray(action).ravel(), max_steps=min(sim.max_steps, step_cap)
                          if step_cap is not None else None)
        obs, reason = result.obs, result.terminal_reason
        if rec is not None:
            rec.capture(sim, action)
    return reason


def evaluate(path: str, instances_per_event: int, seed: int = 1, verbose: bool = True,
             record: str | None = None, stride: int = DEFAULT_STRIDE,
             step_cap: int | None = None):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
    rows, written, t0 = [], [], time.monotonic()
    tasks = event_instances(instances_per_event, seed)
    for task, params in enumerate(tasks):
        sim = OlympicsSim(params)
        obs = sim.reset()
        rec = InstanceRecorder(task, sim, stride) if record else None
        reason = rollout(session, sim, obs, step_cap, rec)
        score = instance_score(sim.event, reason, sim.progress, sim.steps,
                               min(sim.max_steps, step_cap) if step_cap else sim.max_steps, sim.metrics)
        row = {
            "task": task, "event": sim.event, "attempt": params.attempt,
            "friction_level": round(params.friction_level, 4),
            "wind_speed_ms": round(params.wind_speed, 2),
            "challenge": {k: round(float(v), 3) for k, v in params.challenge.items()},
            "terminal_reason": reason, "progress": round(sim.progress, 4),
            "distance_m": round(sim.distance_m, 2), "steps": sim.steps,
            "max_steps": min(sim.max_steps, step_cap) if step_cap else sim.max_steps,
            "metrics": {k: round(float(v), 4) for k, v in sim.metrics.items()},
            "score": round(score, 6),
        }
        rows.append(row)
        if rec is not None:
            written.append(write_instance(record, rec.record(
                sim, row, match_id=f"local:{pathlib.Path(path).stem}", num_instances=len(tasks))))
        if verbose:
            print(f"  [{task + 1:2d}/{len(tasks)}] {sim.event:14} {reason:14} "
                  f"{sim.distance_m:7.2f} m  score {score:.4f}")
        rec = None
        del sim

    event_scores = {
        event: statistics.fmean(float(row["score"]) for row in rows if row["event"] == event)
        for event in EVENTS
    }
    summary = {
        "artifact": path, "instances_per_event": instances_per_event, "n": len(rows), "seed": seed,
        "raw_score": round(meet_score(rows, EVENTS), 6),
        "event_scores": {event: round(value, 6) for event, value in event_scores.items()},
        "num_finished": sum(row["terminal_reason"] in {"completed", "cleared", "landed"}
                            for row in rows),
        "reasons": {reason: sum(row["terminal_reason"] == reason for row in rows)
                    for reason in sorted({row["terminal_reason"] for row in rows})},
        "wall_time_s": round(time.monotonic() - t0, 1), "instances": rows,
    }
    if written:
        summary["history_dir"] = str(record)
        summary["history_files"] = len(written)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact")
    parser.add_argument("-n", "--instances-per-event", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int,
                        help="local testing cap applied to every event; omitted uses official limits")
    parser.add_argument("--json")
    parser.add_argument("--record", metavar="DIR")
    parser.add_argument("--record-stride", type=int, default=DEFAULT_STRIDE, metavar="N")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()
    result = evaluate(args.artifact, args.instances_per_event, seed=args.seed,
                      verbose=not args.quiet, record=args.record, stride=args.record_stride,
                      step_cap=args.max_steps)
    print(json.dumps({key: value for key, value in result.items() if key != "instances"}, indent=2))
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(result, indent=2))
        print("wrote", args.json)
