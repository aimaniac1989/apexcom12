"""The round seed must not reach a miner, on any surface, ever.

Before 0.4.0 the seed was inert: it selected nothing, so leaking it leaked nothing. Now it
selects the round's friction and wind, which makes every seed-bearing output a disclosure of
future rounds -- a platform that draws seeds from any predictable sequence turns one leaked
seed into the next round's conditions.

The reported conditions are NOT the leak, and are deliberately still reported: miners need to
know what they were scored on, and `friction_level`/`wind_speed_ms`/`wind_dir_deg` only let an
ALREADY-GUESSED seed be confirmed. That is why the secrecy has to rest on the seed being
unguessable platform-side, which this repo cannot enforce -- and why nothing here may hand out the
value itself, which would remove the guessing step entirely.

Scanned as serialized JSON rather than by key name: a seed leaked as a nested value, a string,
or under an innocent key is the same leak.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from env.history import DEFAULT_STRIDE, InstanceRecorder
from env.sim import WIND_MAX_MS, OlympicsSim, event_instances, instance_spec

# Distinctive, so a hit in the JSON is unambiguous rather than a coincidental score digit.
ROUND_SEED = 987654321
STEP_CAP = 3


def _numbers(blob) -> list:
    """Every scalar in the structure, at any depth."""
    if isinstance(blob, dict):
        return [n for value in blob.values() for n in _numbers(value)]
    if isinstance(blob, (list, tuple)):
        return [n for value in blob for n in _numbers(value)]
    return [blob]


def _assert_no_seed(blob, what: str) -> None:
    params = instance_spec("sprint_100", 0, ROUND_SEED, WIND_MAX_MS, 4)
    secrets = {ROUND_SEED, params.seed}
    text = json.dumps(blob)
    for secret in secrets:
        assert str(secret) not in text, f"{what} contains the seed {secret} in its JSON"
    for value in _numbers(blob):
        assert value not in secrets, f"{what} reports the seed as a value: {value!r}"
    for key in ("seed", "round_seed", "rng", "round_key"):
        assert key not in text, f"{what} carries a {key!r} field"


def test_the_instance_parameters_are_the_only_place_the_seed_lives():
    """The derived episode seed stays on InstanceParams; nothing downstream needs it."""
    params = instance_spec("sprint_100", 0, ROUND_SEED, WIND_MAX_MS, 4)
    assert params.seed != ROUND_SEED, "the episode seed is the round seed, unmixed"
    # The conditions a miner does see must not be the seed in disguise.
    _assert_no_seed({"friction_level": params.friction_level, "wind_speed": params.wind_speed,
                     "wind_dir": params.wind_dir}, "the reported conditions")


def test_the_history_record_does_not_carry_the_seed():
    """History is delivered to the miner post-round, so it is a miner-visible surface."""
    params = instance_spec("sprint_100", 0, ROUND_SEED, WIND_MAX_MS, 4)
    sim = OlympicsSim(params)
    sim.reset()
    rec = InstanceRecorder(0, sim, DEFAULT_STRIDE)
    for _ in range(STEP_CAP):
        sim.step(np.zeros(12, np.float64), max_steps=STEP_CAP)
        rec.capture(sim, np.zeros(12, np.float64))
    record = rec.record(sim, {"terminal_reason": "timeout", "score": 0.0},
                        match_id="m:0", num_instances=1)
    _assert_no_seed(record, "the history record")


def _load_referee_module():
    """The referee resolves its vendored `gym_v1` from its own directory, not the repo root."""
    root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "referee"))
    spec = importlib.util.spec_from_file_location("olympics_referee", root / "referee/referee.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_unrun_row_does_not_carry_the_seed():
    """The timeout path builds its row separately and must not diverge from the scored one."""
    params = instance_spec("sprint_100", 0, ROUND_SEED, WIND_MAX_MS, 4)
    referee = _load_referee_module()
    _assert_no_seed(referee.OlympicsReferee._unrun_row(0, params, "round_timeout"), "an unrun row")


def test_the_player_is_told_nothing_that_identifies_the_round():
    """The referee hands `player.reset` a fixed seed and an empty config, on purpose.

    A per-round value here would let a policy identify its conditions at runtime instead of
    sensing them, which is the whole point of not showing friction and wind in the observation.
    """
    source = pathlib.Path(__file__).resolve().parents[1].joinpath("referee/referee.py").read_text()
    assert "player_index=0, seed=0," in source, (
        "the referee no longer passes a fixed seed=0 into player.reset"
    )
    assert "config={}" in source, "the referee no longer passes an empty config into player.reset"


def test_conditions_do_not_narrow_the_seed_to_a_single_candidate():
    """Reported conditions must not be a lossless encoding of the seed.

    A phase is one float per event per round; the seed is 63 bits. Two seeds colliding on a
    phase is expected and fine -- what would be fatal is the reverse, a reported condition that
    is the seed itself rescaled.
    """
    levels = {}
    for seed in range(4000):
        key = round(instance_spec("sprint_100", 0, seed, WIND_MAX_MS, 4).friction_level, 4)
        levels.setdefault(key, []).append(seed)
    assert len(levels) < 4000, (
        "every seed maps to its own reported friction level at 4 decimal places, so the "
        "published condition identifies the seed exactly"
    )
