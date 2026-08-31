"""Public geometry for the six legs-only Humanoid Olympics events.

Every event is short enough to be an individual control problem, rather than a
general obstacle course. Conditions (friction, wind, and high-jump bar) are a
fixed public four-stratum launch suite in :mod:`env.sim`; the event geometry
itself is public.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

PLINTH_TOP = 0.8
PLINTH_THICK = 0.4
# A deliberately narrow 1.7 m lane.  This leaves a real running margin for the
# G1 while making a 400 m lap at race speed a route-control problem, not an
# open-floor sprint.
TRACK_HALF_W = 0.85
WORLD_GROUP = 2
OVERHEAD_GROUP = 3
GEOM_PREFIX = "course_"

# The launch preset is intentionally aspirational.  A complete all-round meet
# should be a milestone, not something a flat-ground walker solves immediately.
EVENTS = ("sprint_100", "sprint_400", "hurdles_100", "high_jump", "long_jump", "triple_jump")
EVENT_LABELS = {
    "sprint_100": "100 m sprint",
    "sprint_400": "400 m circular sprint",
    "hurdles_100": "100 m hurdles",
    "high_jump": "high jump",
    "long_jump": "long jump",
    "triple_jump": "triple jump",
}

# These are race standards, not generous simulation watchdogs: 100 m needs
# 4.25 m/s average, while the full 400 m lap needs 5.56 m/s including the
# start and sustained cornering.  Jumps receive enough time for a careful
# setup, but cannot spend a minute probing the obstacle.
EVENT_MAX_STEPS = {
    "sprint_100": 1200,
    "sprint_400": 3600,
    "hurdles_100": 1900,
    "high_jump": 900,
    "long_jump": 1000,
    "triple_jump": 1400,
}

# The escalating hurdles form a within-run curriculum: a policy can make
# meaningful early progress, but the 1.15 m final barrier is deliberately at
# the frontier of what this legs-only G1 should be able to clear at speed.
HURDLE_HEIGHTS_M = (0.55, 0.60, 0.65, 0.70, 0.75,
                     0.80, 0.85, 0.90, 1.00, 1.15)
HIGH_JUMP_BARS_M = (1.00, 1.10, 1.20, 1.30)
TAKEOFF_BOARD_BEFORE_M, TAKEOFF_BOARD_AFTER_M = 0.35, 0.05
LONG_TAKEOFF_M, LONG_LANDING_M = 15.0, 21.0
TRIPLE_TAKEOFF_M = 12.0
TRIPLE_HOP_START_M, TRIPLE_HOP_LENGTH_M = 14.0, 1.5
TRIPLE_STEP_START_M, TRIPLE_STEP_LENGTH_M = 18.0, 1.5
TRIPLE_LANDING_M = 25.0

# The low end is a real slip regime, not a merely-imperfect one: 0.30 is below the ~0.35 where a
# foot can still push hard enough to sprint comfortably, so the bottom stratum asks a policy to
# manage traction rather than assume it. Widening the band moves EVERY stratum, not just the last
# one, because a stratum is a fraction of the band (see `nominal_mu`): the second and third strata
# drop to ~0.80 and ~0.57 from ~0.90 and ~0.71. The +/-8% per-slab jitter is relative to the band
# too, so it widens to +/-0.076.
FRICTION_NOMINAL = (0.30, 1.25)
COLOR = {
    "track": ".54 .57 .61 1",
    "hurdle": ".85 .26 .32 1",
    "bar": ".98 .89 .22 1",
    "takeoff_board": ".93 .95 .98 1",
    "sand": ".78 .62 .35 1",
    "hop_pad": ".42 .72 .88 1",
    "step_pad": ".32 .60 .86 1",
}


@dataclass(frozen=True)
class Surface:
    """One physical box in the scene.

    ``walkable=False`` keeps a bar out of downward terrain rays while leaving
    it fully collidable. ``yaw`` is a world-z rotation in radians.
    """

    kind: str
    x: float
    y: float
    z: float
    hx: float
    hy: float
    hz: float
    yaw: float = 0.0
    walkable: bool = True


@dataclass(frozen=True)
class EventLayout:
    event: str
    surfaces: tuple[Surface, ...]
    start_x: float
    start_y: float
    start_yaw: float
    finish: float
    challenge: Mapping[str, float] = field(default_factory=dict)

    @property
    def is_circular(self) -> bool:
        return self.event == "sprint_400"


def _slab(x0: float, length: float, *, y: float = 0.0, half_w: float = TRACK_HALF_W,
          kind: str = "track") -> Surface:
    return Surface(kind, x0 + length / 2, y, PLINTH_TOP - PLINTH_THICK / 2,
                   length / 2, half_w, PLINTH_THICK / 2)


def _straight(length: float, *, kind: str = "track") -> list[Surface]:
    return [_slab(-3.0, length + 3.0, kind=kind)]


def _sprint_100() -> EventLayout:
    return EventLayout("sprint_100", tuple(_straight(100.0)), -2.0, 0.0, 0.0, 100.0)


def _hurdles_100() -> EventLayout:
    surfaces = _straight(100.0)
    # Ten progressively taller barriers. Each overhangs the lane boundary so a
    # runner cannot skim around its end while leaving its pelvis in bounds.
    for x, height in zip(np.linspace(12.0, 88.5, 10), HURDLE_HEIGHTS_M, strict=True):
        surfaces.append(Surface("hurdle", float(x), 0.0, PLINTH_TOP + height / 2,
                                0.12, TRACK_HALF_W + 0.20, height / 2,
                                walkable=False))
    return EventLayout("hurdles_100", tuple(surfaces), -2.0, 0.0, 0.0, 100.0)


def _sprint_400() -> EventLayout:
    # A genuine 400 m circumference. Overlapping tangent slabs approximate a
    # lane without creating seam gaps; 96 is smooth relative to the 1.5 m lane.
    radius, n = 400.0 / (2.0 * math.pi), 96
    arc = 400.0 / n
    surfaces = []
    for i in range(n):
        theta = 2.0 * math.pi * (i + 0.5) / n
        surfaces.append(Surface("track", radius * math.cos(theta), radius * math.sin(theta),
                                PLINTH_TOP - PLINTH_THICK / 2, arc * 0.53, TRACK_HALF_W,
                                PLINTH_THICK / 2, yaw=theta + math.pi / 2))
    # Counter-clockwise start: the tangent at (R, 0) points +y.
    return EventLayout("sprint_400", tuple(surfaces), radius, 0.0, math.pi / 2, 400.0,
                       {"radius_m": radius})


def _high_jump(challenge: Mapping[str, float]) -> EventLayout:
    bar_height = float(challenge.get("bar_height_m", HIGH_JUMP_BARS_M[0]))
    bar_x = 12.0
    surfaces = _straight(20.0)
    # The bar is collidable, but is not a ground sample. Its clearance is
    # judged at the vertical plane rather than by a fragile contact identity.
    surfaces.append(Surface("bar", bar_x, 0.0, PLINTH_TOP + bar_height,
                            0.045, TRACK_HALF_W + 0.20, 0.035, walkable=False))
    return EventLayout("high_jump", tuple(surfaces), -2.0, 0.0, 0.0, bar_x,
                       {"bar_height_m": bar_height, "bar_x_m": bar_x})


def _long_jump() -> EventLayout:
    takeoff, landing = LONG_TAKEOFF_M, LONG_LANDING_M
    surfaces = [
        _slab(-3.0, takeoff - TAKEOFF_BOARD_BEFORE_M + 3.0, kind="track"),
        _slab(takeoff - TAKEOFF_BOARD_BEFORE_M,
              TAKEOFF_BOARD_BEFORE_M + TAKEOFF_BOARD_AFTER_M, kind="takeoff_board"),
        _slab(landing, 24.0, kind="sand"),
    ]
    return EventLayout("long_jump", tuple(surfaces), -2.0, 0.0, 0.0, landing,
                       {"takeoff_x_m": takeoff, "landing_x_m": landing})


def _triple_jump() -> EventLayout:
    # The hop, step, and jump are deliberately wide real gaps. A legal final
    # landing only follows from three separate, controlled single-leg phases.
    takeoff, landing = TRIPLE_TAKEOFF_M, TRIPLE_LANDING_M
    surfaces = [
        _slab(-3.0, takeoff - TAKEOFF_BOARD_BEFORE_M + 3.0, kind="track"),
        _slab(takeoff - TAKEOFF_BOARD_BEFORE_M,
              TAKEOFF_BOARD_BEFORE_M + TAKEOFF_BOARD_AFTER_M, kind="takeoff_board"),
        _slab(TRIPLE_HOP_START_M, TRIPLE_HOP_LENGTH_M, kind="hop_pad"),
        _slab(TRIPLE_STEP_START_M, TRIPLE_STEP_LENGTH_M, kind="step_pad"),
        _slab(landing, 24.0, kind="sand"),
    ]
    return EventLayout("triple_jump", tuple(surfaces), -2.0, 0.0, 0.0, landing,
                       {"takeoff_x_m": takeoff, "landing_x_m": landing,
                        "hop_start_x_m": TRIPLE_HOP_START_M,
                        "step_start_x_m": TRIPLE_STEP_START_M})


def build_event(event: str, challenge: Mapping[str, float] | None = None) -> EventLayout:
    """Build the named event from its public geometry and round conditions."""
    challenge = challenge or {}
    if event == "sprint_100":
        return _sprint_100()
    if event == "sprint_400":
        return _sprint_400()
    if event == "hurdles_100":
        return _hurdles_100()
    if event == "high_jump":
        return _high_jump(challenge)
    if event == "long_jump":
        return _long_jump()
    if event == "triple_jump":
        return _triple_jump()
    raise ValueError(f"unknown Olympic event {event!r}")


def course_xml_fragment(layout: EventLayout, frictions: list[float] | None = None) -> str:
    """The layout's MJCF boxes, named in friction-array order."""
    out = []
    for i, surface in enumerate(layout.surfaces):
        mu = 1.0 if frictions is None else frictions[i]
        group = WORLD_GROUP if surface.walkable else OVERHEAD_GROUP
        out.append(
            f'    <geom name="{GEOM_PREFIX}{i}" type="box" '
            f'pos="{surface.x:.4f} {surface.y:.4f} {surface.z:.4f}" '
            f'size="{surface.hx:.4f} {surface.hy:.4f} {surface.hz:.4f}" '
            f'euler="0 0 {surface.yaw:.6f}" condim="3" group="{group}" '
            f'friction="{mu:.4f} .1 .1" rgba="{COLOR[surface.kind]}"/>'
        )
    return "\n".join(out)


def nominal_mu(level: float) -> float:
    lo, hi = FRICTION_NOMINAL
    return hi - (hi - lo) * float(level)


def sample_frictions(layout: EventLayout, level: float, rng: np.random.Generator) -> list[float]:
    """One rounded sliding-friction value per emitted geom."""
    lo, hi = FRICTION_NOMINAL
    base = hi - (hi - lo) * float(level)
    return [round(float(np.clip(base + (hi - lo) * 0.08 * rng.uniform(-1.0, 1.0), lo, hi)), 4)
            for _ in layout.surfaces]


if __name__ == "__main__":
    for event in EVENTS:
        layout = build_event(event)
        print(f"{event:14} {len(layout.surfaces):3d} geoms, start=({layout.start_x:.1f}, "
              f"{layout.start_y:.1f}), finish={layout.finish:.1f}")
