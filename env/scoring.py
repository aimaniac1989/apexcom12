"""Bounded, equally weighted scoring for a Humanoid Olympics meet."""

from __future__ import annotations

import math
from typing import Mapping

from .course import (HIGH_JUMP_BARS_M, LONG_LANDING_M, LONG_TAKEOFF_M, TRIPLE_LANDING_M,
                     TRIPLE_TAKEOFF_M)

RACE_EVENTS = frozenset({"sprint_100", "sprint_400", "hurdles_100"})
JUMP_EVENTS = frozenset({"long_jump", "triple_jump"})


def instance_score(event: str, terminal_reason: str, progress: float, steps: int, max_steps: int,
                   metrics: Mapping[str, float] | None = None) -> float:
    """Return one event result in ``[0, 1]``."""
    metrics = metrics or {}
    if terminal_reason in {"physics_glitch", "high_foul", "invalid_action", "jump_foul",
                           "player_error", "submission_not_ready", "out_of_bounds"}:
        return 0.0

    if event in RACE_EVENTS:
        if terminal_reason == "completed":
            # Completion earns [0.25, 1.00]; pace breaks ties without allowing
            # a timeout to outrank a legitimate finish.
            return 0.25 + 0.75 * max(0.0, 1.0 - steps / max(max_steps, 1))
        return 0.24 * min(max(float(progress), 0.0), 1.0)

    if event == "high_jump":
        target = max(float(metrics.get("bar_height_m", 0.95)), 1e-6)
        clearance = max(0.0, float(metrics.get("best_clearance_m", 0.0)))
        if terminal_reason == "cleared":
            normalized_height = min(max((target - HIGH_JUMP_BARS_M[0]) /
                                        (HIGH_JUMP_BARS_M[-1] - HIGH_JUMP_BARS_M[0]), 0.0), 1.0)
            return 0.25 + 0.75 * normalized_height
        return 0.24 * min(clearance / target, 1.0)

    if event in JUMP_EVENTS:
        distance = max(0.0, float(metrics.get("jump_distance_m", 0.0)))
        minimum = (LONG_LANDING_M - LONG_TAKEOFF_M if event == "long_jump"
                   else TRIPLE_LANDING_M - TRIPLE_TAKEOFF_M)
        target = 12.0 if event == "long_jump" else 18.0
        if terminal_reason == "landed":
            return 0.25 + 0.75 * min(max((distance - minimum) / (target - minimum), 0.0), 1.0)
        return 0.20 * min(max(float(progress), 0.0), 1.0)

    raise ValueError(f"unknown Olympic event {event!r}")


def meet_score(rows: list[Mapping[str, float | str]], events: tuple[str, ...]) -> float:
    """Mean the event means, so long races cannot outweigh jump disciplines."""
    if not rows:
        return 0.0
    means = []
    for event in events:
        scores = [float(row["score"]) for row in rows if row.get("event") == event]
        means.append(math.fsum(scores) / len(scores) if scores else 0.0)
    return math.fsum(means) / len(means)
