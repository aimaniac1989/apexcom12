"""A recorded run must still replay from its history files alone.

History is only worth writing if it can be read back, and the writer (env/history.py, shared with
the referee) and the reader (tools/replay.py) are different files — so assert the round trip, not
just that recording ran.

    PYTHONPATH=. python .github/scripts/check_replay_roundtrip.py <history-dir> <n>
"""

from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np

from env.history import read_all
from tools.preview import _lit_model
from tools.replay import Run

directory = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])

# One file per instance, named as the platform's FileType.HISTORY collector expects.
names = sorted(p.name for p in directory.glob("instance_*.json"))
assert names == [f"instance_{i:02d}.json" for i in range(expected)], names

runs = [Run(r) for r in read_all(directory)]
assert len(runs) == expected, len(runs)
for run in runs:
    # Straight-event route distance is world-x relative to its recorded start.
    # The circular 400 m is unwrapped angular progress, which cannot be rebuilt
    # from sparse positions without repeating the referee's route integrator.
    if run.event != "sprint_400":
        distance = float(run.qpos[:, 0].max() - run.qpos[0, 0])
        assert abs(distance - run.outcome["distance_m"]) < 0.02, run.outcome
    assert run.action.shape == (run.frames, 12), run.action.shape
    # ...and rebuild the scene and the pose with nothing but mujoco.
    model = _lit_model(run.event, run.frictions, run.challenge)
    data = mujoco.MjData(model)
    data.qpos[:] = run.qpos[-1]
    mujoco.mj_forward(model, data)
    assert np.isfinite(data.body("pelvis").xpos).all()

print(f"ok: replayed {sum(r.frames for r in runs)} frames from qpos alone")
