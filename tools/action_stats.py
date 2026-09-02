"""Measure what a policy's actions actually look like, from a recorded meet.

`engine/rewards.py` prices `w_smooth` against ``||a - a_prev||^2`` and `w_effort` against
``||a||^2``, and both were tuned for a plausible walking gait rather than a measured one. That
turned out to matter: at the original 0.004, a large-amplitude gait went net negative on a
completed 100 m. This reads back a recorded history and reports what a policy actually does, so
those two weights can be set against a number instead of a guess -- including for the rival
artifacts in `original/`, whose gaits are the ones worth pricing for.

    PYTHONPATH=. python tools/local_eval.py original/code_submission_0830.onnx \
        -n 1 --seed 1 --record /tmp/leader --record-stride 1
    PYTHONPATH=. python tools/action_stats.py /tmp/leader

**Record with `--record-stride 1`.** The default stride of 2 keeps every second frame, which
inflates ``||a - a_prev||^2`` by sampling across two control steps.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

from env.history import read_instance, unpack


def stats(actions: np.ndarray) -> dict[str, float]:
    """Per-episode action statistics. Frame 0 and the terminal frame carry zeros, so drop them."""
    a = np.asarray(actions, np.float64)
    if len(a) < 3:
        return {}
    a = a[1:-1]
    d = np.diff(a, axis=0)
    return {"steps": len(a),
            "abs_a": float(np.abs(a).mean()),
            "max_a": float(np.abs(a).max()),
            "sq_a": float((a * a).sum(axis=1).mean()),        # mean ||a||^2 per step
            "sq_da": float((d * d).sum(axis=1).mean())}       # mean ||a - a_prev||^2 per step


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", help="a directory of instance_*.json written by --record")
    ap.add_argument("--w-smooth", type=float, default=0.0002)
    ap.add_argument("--w-effort", type=float, default=0.0001)
    args = ap.parse_args()

    rows = []
    for path in sorted(pathlib.Path(args.directory).glob("instance_*.json")):
        rec = read_instance(path)
        s = stats(unpack(rec["frames"]["action"]))
        if s:
            s["stride"] = int(rec["timing"].get("stride", 1))
            rows.append((rec["conditions"].get("event", "?"),
                         rec["outcome"].get("terminal_reason", "?"), s))

    if not rows:
        raise SystemExit(f"no usable instances in {args.directory}")

    print(f"{'event':<13}{'reason':<12}{'steps':>6}{'mean|a|':>9}{'max|a|':>8}"
          f"{'||a||^2':>9}{'||da||^2':>10}{'effort':>9}{'smooth':>9}")
    for event, reason, s in rows:
        ef = args.w_effort * s["sq_a"] * s["steps"]
        sm = args.w_smooth * s["sq_da"] * s["steps"]
        print(f"{event:<13}{reason:<12}{s['steps']:>6}{s['abs_a']:>9.2f}{s['max_a']:>8.2f}"
              f"{s['sq_a']:>9.1f}{s['sq_da']:>10.1f}{-ef:>9.0f}{-sm:>9.0f}")

    # A completed 100 m pays roughly 1512 all in (progress + success + w_score + alive + upright).
    # If `smooth` above approaches that, the shaping is forbidding the gait rather than tidying it.
    strides = {r[2]["stride"] for r in rows}
    if strides != {1}:
        print()
        print(f"WARNING stride {sorted(strides)}: frames are that many control steps "
              "apart, so ||da||^2 is inflated. Re-record with --record-stride 1.")
    print(f"\nweights: w_effort {args.w_effort}  w_smooth {args.w_smooth}")
    print("compare `smooth` against ~1512, what a completed 100 m pays in total")


if __name__ == "__main__":
    main()
