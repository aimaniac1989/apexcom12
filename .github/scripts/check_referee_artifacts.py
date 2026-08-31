"""The referee's two miner-facing artifacts behave, without needing containers.

Drives the REAL `OlympicsReferee.play_game()` against the REAL `OlympicsPlayer`: the stub client
below stands in for gym_v1's `PlayerClient`, so every /reset and /act goes through the code the
sandbox runs. That covers both artifacts at once — the per-instance history the referee writes,
and the API log the player prints — and lets the failure path be tested, which a container run
cannot easily do.

    PYTHONPATH=.:referee:player SUBMISSION_PATH=baseline/baseline.onnx \
        python .github/scripts/check_referee_artifacts.py
"""

from __future__ import annotations

import contextlib
import io
import pathlib

import referee as ref                      # referee/referee.py
from gym_v1.referee import RefereeContext
from launch import OlympicsPlayer          # player/launch.py

from env import EVENTS
from env.history import read_all

N, STEPS = 3, 150

player = OlympicsPlayer()
assert player.is_ready(), player.load_error


class Stub:
    """gym_v1's PlayerClient surface — the two methods the referee calls."""

    def reset(self, **kw):
        # The HTTP client owns its timeout; the in-process player does not.
        kw.pop("timeout_s", None)
        player.reset(**kw)

    def act(self, observation, deadline_ms, timeout_s=None):
        return player.act(observation=observation, deadline_ms=deadline_ms)


def play(history_dir: str, instances: int = N, steps: int = STEPS):
    """One scored suite against `history_dir`, returning (result, captured stdout)."""
    ref.HISTORY_DIR = pathlib.Path(history_dir)
    ctx = RefereeContext(match_id="ci", seed=1, num_players=1, player_urls=["http://stub"],
                         config={"seed": 1, "instances_per_event": instances,
                                 "test_step_cap": steps, "deadline_ms": 500})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = ref.OlympicsReferee().play_game(ctx, [Stub()])
    return result, buf.getvalue()


# 1. The referee writes one history file per instance, agreeing with what it scored.
result, log = play("/tmp/refhistory")
files = sorted(pathlib.Path("/tmp/refhistory").glob("*.json"))
expected = N * len(EVENTS)
assert [p.name for p in files] == [f"instance_{i:02d}.json" for i in range(expected)], files
assert result.metadata["history_files"] == expected, result.metadata
for record, row in zip(read_all("/tmp/refhistory"), result.metadata["tasks"]):
    assert record["match_id"] == "ci", record["match_id"]
    assert record["outcome"]["score"] == row["score"], (record["outcome"], row)
print(f"ok: {expected} history files, outcomes agree with result.json metadata")

# 2. The player logs every API call the referee made — one act line per control step.
acts = [ln for ln in log.splitlines() if ln.startswith("apex.api") and " call=act " in ln]
resets = [ln for ln in log.splitlines() if " call=reset " in ln]
assert len(resets) == expected, len(resets)
assert len(acts) == sum(r["steps"] for r in result.metadata["tasks"]), len(acts)
print(f"ok: {len(resets)} reset + {len(acts)} act lines logged (= total control steps)")

# 3. A failure to write a debugging artifact must never turn a scored round into a zero.
ok, _ = play("/tmp/writable_history", instances=2, steps=100)
bad, _ = play("/proc/nope/history", instances=2, steps=100)
assert ok.metadata["history_files"] == 2 * len(EVENTS), ok.metadata
assert bad.metadata["history_files"] == 0, bad.metadata
assert bad.raw_scores == ok.raw_scores, (bad.raw_scores, ok.raw_scores)
print(f"ok: unwritable history -> history_files=0, score unchanged at {ok.raw_scores[0]}")
