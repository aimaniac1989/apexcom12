"""The round seed must move the conditions, and must move them without unbalancing the meet.

Two properties are in tension here and both are load-bearing:

* Rounds must DIFFER, or the meet is 24 fixed operating points a policy can be tuned to
  rather than a friction band it has to hold up across.
* Rounds must be EQUALLY HARD IN SHAPE, or the 1% takeover margin is competing with luck of
  the draw. Every round keeps four evenly spaced strata spanning the whole envelope; only the
  phase moves. That is asserted here on the cyclic spacing, not on a score, because a score
  cannot distinguish a balanced sweep from four attempts that happened to bunch up.

Determinism per seed is the third: the incumbent is re-scored at the start of each round and
that comparison is only like-for-like if a seed reproduces its meet exactly.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from env.course import EVENTS, HIGH_JUMP_BARS_M
from env.sim import WIND_MAX_MS, event_instances, instance_spec

ATTEMPTS = 4
SEEDS = [0, 1, 7, 12345, 2**31 - 1, 2**63 + 17]


def _by_event(tasks):
    grouped: dict[str, list] = {event: [] for event in EVENTS}
    for params in tasks:
        grouped[params.event].append(params)
    return grouped


def _cyclic_gaps(values: list[float]) -> list[float]:
    """Gaps between sorted values on the unit circle, including the wrap-around gap."""
    ordered = sorted(values)
    gaps = [b - a for a, b in zip(ordered, ordered[1:])]
    gaps.append(1.0 - ordered[-1] + ordered[0])
    return gaps


def test_same_seed_reproduces_the_meet_exactly():
    """Without this the incumbent's re-score is not comparable to the challengers'."""
    first = event_instances(ATTEMPTS, seed=12345, wind_max=WIND_MAX_MS)
    again = event_instances(ATTEMPTS, seed=12345, wind_max=WIND_MAX_MS)
    assert first == again, "the same round seed produced two different meets"


def test_conditions_move_between_rounds():
    """Friction level, wind speed, wind direction, and the episode seed all follow the round."""
    meets = {seed: event_instances(ATTEMPTS, seed, WIND_MAX_MS) for seed in SEEDS}
    for field in ("friction_level", "wind_speed", "wind_dir", "seed"):
        signatures = {seed: tuple(getattr(p, field) for p in tasks)
                      for seed, tasks in meets.items()}
        distinct = set(signatures.values())
        assert len(distinct) == len(SEEDS), (
            f"{field} did not vary across all {len(SEEDS)} round seeds: "
            f"{len(distinct)} distinct meet(s)"
        )


def test_events_do_not_shift_in_lockstep():
    """A single global phase would move all six disciplines the same way every round."""
    for seed in SEEDS:
        grouped = _by_event(event_instances(ATTEMPTS, seed, WIND_MAX_MS))
        first_attempt_mu = {event: attempts[0].friction_level
                            for event, attempts in grouped.items()}
        assert len(set(first_attempt_mu.values())) == len(EVENTS), (
            f"seed {seed}: events share a stratum phase: {first_attempt_mu}"
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_every_round_is_a_balanced_sweep(seed: int):
    """Four evenly spaced strata across the full envelope, whatever the round drew."""
    for event, attempts in _by_event(event_instances(ATTEMPTS, seed, WIND_MAX_MS)).items():
        assert len(attempts) == ATTEMPTS

        mus = [p.friction_level for p in attempts]
        assert len(set(mus)) == ATTEMPTS, f"seed {seed}/{event}: strata collapsed: {mus}"
        for level in mus:
            assert 0.0 <= level < 1.0, f"seed {seed}/{event}: level {level} off the band"
        for gap in _cyclic_gaps(mus):
            assert gap == pytest.approx(1.0 / ATTEMPTS, abs=1e-9), (
                f"seed {seed}/{event}: friction strata are not evenly spaced: {mus}"
            )

        winds = [p.wind_speed / WIND_MAX_MS for p in attempts]
        assert len(set(winds)) == ATTEMPTS, f"seed {seed}/{event}: wind strata collapsed: {winds}"
        for level in winds:
            assert 0.0 <= level < 1.0, f"seed {seed}/{event}: wind {level} off the band"
        for gap in _cyclic_gaps(winds):
            assert gap == pytest.approx(1.0 / ATTEMPTS, abs=1e-9), (
                f"seed {seed}/{event}: wind strata are not evenly spaced: {winds}"
            )


@pytest.mark.parametrize("seed", SEEDS)
def test_opposing_wind_directions_survive_the_phase_shift(seed: int):
    """The paired-heading property is what stops a round favouring one direction of travel."""
    for event, attempts in _by_event(event_instances(ATTEMPTS, seed, WIND_MAX_MS)).items():
        dirs = [p.wind_dir for p in attempts]
        for a, b in ((0, 1), (2, 3)):
            separation = abs(dirs[a] - dirs[b]) % (2.0 * 3.141592653589793)
            assert min(separation, 2.0 * 3.141592653589793 - separation) == pytest.approx(
                3.141592653589793, abs=1e-9), (
                f"seed {seed}/{event}: attempts {a}/{b} are not opposed: {dirs}"
            )


def test_the_bar_ladder_is_not_a_round_condition():
    """High-jump bars are the published difficulty scale, so they must not follow the seed."""
    expected = [HIGH_JUMP_BARS_M[attempt % len(HIGH_JUMP_BARS_M)] for attempt in range(ATTEMPTS)]
    for seed in SEEDS:
        bars = [instance_spec("high_jump", attempt, seed, WIND_MAX_MS, ATTEMPTS)
                .challenge["bar_height_m"] for attempt in range(ATTEMPTS)]
        assert bars == expected, f"seed {seed}: bar ladder moved to {bars}"


def test_a_negative_seed_does_not_fault_the_referee():
    """ctx.seed arrives off the request; the schema's minimum is not enforced in here."""
    tasks = event_instances(ATTEMPTS, seed=-9, wind_max=WIND_MAX_MS)
    assert len(tasks) == ATTEMPTS * len(EVENTS)
