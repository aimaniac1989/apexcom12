"""Semantic checks for the deliberately hard Olympic event rules.

These checks exercise the referee-owned state machines directly with synthetic
foot contacts.  They complement a physics rollout: a weak policy rarely
reaches every jump phase, while these assertions catch a future rule change
that accidentally turns a void or wrong-foot touch into a valid landing.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from env import OlympicsSim, event_instances, instance_score, instance_spec
from env.course import (HIGH_JUMP_BARS_M, HURDLE_HEIGHTS_M, LONG_LANDING_M, LONG_TAKEOFF_M,
                        TAKEOFF_BOARD_AFTER_M, TAKEOFF_BOARD_BEFORE_M, TRIPLE_LANDING_M,
                        TRIPLE_TAKEOFF_M, build_event)
from env.sim import (HIGH_CLEARANCE_MARGIN_M, HIGH_LANDING_OFFSET_M, MIN_AIRBORNE_STEPS,
                     MIN_SUPPORT_STEPS, PLINTH_TOP, WIND_MAX_MS, _scene_xml)


def sim(event: str) -> OlympicsSim:
    result = OlympicsSim(instance_spec(event, 0, seed=7))
    result.reset()
    return result


# The public geometry carries the promised hard dimensions, and the decorative
# floor cannot ever become a lower route.
hurdles = build_event("hurdles_100")
assert len([s for s in hurdles.surfaces if s.kind == "hurdle"]) == 10
assert tuple(round(s.hz * 2, 2) for s in hurdles.surfaces if s.kind == "hurdle") == HURDLE_HEIGHTS_M
hurdle_sim = sim("hurdles_100")
hurdle_ids = hurdle_sim._obstacle_geom_ids["hurdle"]
assert len(hurdle_ids) == 10
assert all(hurdle_sim._geom_pair_hits(hurdle_ids, hurdle_id, -1) for hurdle_id in hurdle_ids)
long_layout = build_event("long_jump")
assert math.isclose(long_layout.challenge["landing_x_m"] - long_layout.challenge["takeoff_x_m"], 6.0)
board = next(s for s in long_layout.surfaces if s.kind == "takeoff_board")
assert math.isclose(board.x - board.hx, LONG_TAKEOFF_M - TAKEOFF_BOARD_BEFORE_M)
assert math.isclose(board.x + board.hx, LONG_TAKEOFF_M + TAKEOFF_BOARD_AFTER_M)
assert 'contype="0" conaffinity="0"' in _scene_xml(long_layout)

# 0.4.0 draws the conditions from the round seed, so the meet must MOVE between rounds and
# reproduce exactly within one.  The stratification must survive the phase shift: four evenly
# spaced friction and wind strata spanning the full envelope, every round.
assert event_instances(4, 1) == event_instances(4, 1)
assert event_instances(4, 1) != event_instances(4, 20)
for event in ("sprint_100", "sprint_400", "hurdles_100", "high_jump", "long_jump", "triple_jump"):
    for attempt in range(4):
        assert instance_spec(event, attempt, seed=1) == instance_spec(event, attempt, seed=1)
        assert instance_spec(event, attempt, seed=1) != instance_spec(event, attempt, seed=20)
    for seed in (1, 20, 987654321):
        specs = [instance_spec(event, attempt, seed=seed) for attempt in range(4)]
        # The high-jump bar ladder is the published difficulty scale, not a condition.
        assert [s.challenge for s in specs] == [instance_spec(event, a, seed=1).challenge
                                                for a in range(4)]
        for levels in ([s.friction_level for s in specs],
                       [s.wind_speed / WIND_MAX_MS for s in specs]):
            assert len(set(levels)) == 4, (event, seed, levels)
            assert all(0.0 <= level < 1.0 for level in levels), (event, seed, levels)
            ordered = sorted(levels)
            gaps = [b - a for a, b in zip(ordered, ordered[1:])] + [1.0 - ordered[-1] + ordered[0]]
            assert all(abs(gap - 0.25) < 1e-9 for gap in gaps), (event, seed, levels)

# Only a foot contact on a walkable surface's top, inside its footprint, can
# become support. A vertical side scrape and an edge contact cannot unlock a phase.
contact_sim = sim("long_jump")
board_gid = next(gid for gid, surface in contact_sim._surface_by_geom.items()
                 if surface.kind == "takeoff_board")
board_surface = contact_sim._surface_by_geom[board_gid]
top = board_surface.z + board_surface.hz
top_contact = SimpleNamespace(pos=np.array([board_surface.x, board_surface.y, top]),
                              frame=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
assert contact_sim._is_top_course_contact(board_gid, top_contact)
side_contact = SimpleNamespace(pos=np.array([board_surface.x, board_surface.y, top - 0.2]),
                               frame=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
edge_contact = SimpleNamespace(pos=np.array([board_surface.x + board_surface.hx + 0.1,
                                             board_surface.y, top]),
                               frame=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
assert not contact_sim._is_top_course_contact(board_gid, side_contact)
assert not contact_sim._is_top_course_contact(board_gid, edge_contact)
contact_sim._foot_contact_forces = {("left", "sand"): 0.0}
assert not any(contact_sim._accumulate_support({"left": {"sand"}}, "sand")
               for _ in range(MIN_SUPPORT_STEPS))

# Route-relative lateral sign is positive-left for every event. CCW runners
# are left of their tangent when they move toward the circle centre.
straight = sim("sprint_100")
straight.data.qpos[:2] = [0.0, 0.3]
assert math.isclose(straight._route()[1], 0.3)
circle = sim("sprint_400")
radius = float(circle.layout.challenge["radius_m"])
for angle in (0.0, math.pi / 2, math.pi, -math.pi / 2):
    circle.data.qpos[:2] = [(radius - 0.3) * math.cos(angle), (radius - 0.3) * math.sin(angle)]
    heading, lateral, _ = circle._route()
    assert math.isclose(lateral, 0.3, abs_tol=1e-9)
    assert math.isclose(math.sin(heading), math.sin(angle + math.pi / 2), abs_tol=1e-9)
    circle.data.qpos[:2] = [(radius + 0.3) * math.cos(angle), (radius + 0.3) * math.sin(angle)]
    assert math.isclose(circle._route()[1], -0.3, abs_tol=1e-9)

# Ducking under a high-jump bar cannot clear it.  A genuine airborne crossing
# followed by a far-side foot landing can.
high = sim("high_jump")
bar_x = float(high.layout.challenge["bar_x_m"])
bar_h = float(high.layout.challenge["bar_height_m"])
high._hits = lambda kind: False  # type: ignore[method-assign]
high._foot_contacts = lambda: {}  # type: ignore[method-assign]
high._observe_high_jump(bar_x - 0.01, bar_x + 0.01, PLINTH_TOP + bar_h)
assert high._event_reason == "bar_missed", high._event_reason

high = sim("high_jump")
bar_x = float(high.layout.challenge["bar_x_m"])
bar_h = float(high.layout.challenge["bar_height_m"])
high._hits = lambda kind: False  # type: ignore[method-assign]
high._foot_contacts = lambda: {}  # type: ignore[method-assign]
high._high_airborne_steps = MIN_AIRBORNE_STEPS - 1
high._observe_high_jump(bar_x - 0.01, bar_x + 0.01, PLINTH_TOP + bar_h + HIGH_CLEARANCE_MARGIN_M)
assert high._high_valid_crossing, high._event_reason
high._foot_contacts = lambda: {"left": {"track"}}  # type: ignore[method-assign]
high._jump_contact_illegal = lambda contacts, allowed: False  # type: ignore[method-assign]
for _ in range(MIN_SUPPORT_STEPS):
    high._observe_high_jump(bar_x + HIGH_LANDING_OFFSET_M, bar_x + HIGH_LANDING_OFFSET_M + 0.01,
                            PLINTH_TOP + bar_h)
assert high._event_reason == "cleared", high._event_reason

# A long jump only counts after sustained foot support on sand.  A runway/floor
# contact after take-off is a foul, not a shortcut.
long = sim("long_jump")
long._jump_state = "long_flight"
long._observe_long_jump(LONG_LANDING_M, LONG_TAKEOFF_M, {"left": {"track"}})
assert long._event_reason == "jump_foul", long._event_reason
long = sim("long_jump")
long._jump_state = "long_flight"
long._foot_contact_xs = {("left", "sand"): [LONG_LANDING_M + 0.25]}
for _ in range(MIN_SUPPORT_STEPS):
    long._observe_long_jump(LONG_LANDING_M + 4.0, LONG_TAKEOFF_M, {"left": {"sand"}})
assert long._event_reason == "landed", long._event_reason
assert math.isclose(long._jump_distance, 6.25)

long = sim("long_jump")
long._jump_state, long.max_x = "approach", LONG_TAKEOFF_M + TAKEOFF_BOARD_AFTER_M + 0.01
long._observe_long_jump(long.max_x, LONG_TAKEOFF_M, {})
assert long._event_reason == "jump_foul", long._event_reason

# Triple jump needs same-foot hop, a real second flight, opposite-foot step,
# another flight, then sustained sand support.  A wrong-foot first landing is
# terminally illegal.
triple = sim("triple_jump")
triple._jump_state, triple._takeoff_foot = "hop_flight", "left"
triple._observe_triple_jump(14.1, TRIPLE_TAKEOFF_M, {"right": {"hop_pad"}})
assert triple._event_reason == "jump_foul", triple._event_reason

triple = sim("triple_jump")
triple._jump_state, triple._takeoff_foot = "hop_flight", "left"
for _ in range(MIN_SUPPORT_STEPS):
    triple._observe_triple_jump(14.1, TRIPLE_TAKEOFF_M, {"left": {"hop_pad"}})
assert triple._jump_state == "hop_support" and triple._triple_phase == 1
for _ in range(MIN_AIRBORNE_STEPS):
    triple._observe_triple_jump(15.7, TRIPLE_TAKEOFF_M, {})
assert triple._jump_state == "step_flight"
for _ in range(MIN_SUPPORT_STEPS):
    triple._observe_triple_jump(18.1, TRIPLE_TAKEOFF_M, {"right": {"step_pad"}})
assert triple._jump_state == "step_support" and triple._triple_phase == 2
for _ in range(MIN_AIRBORNE_STEPS):
    triple._observe_triple_jump(19.7, TRIPLE_TAKEOFF_M, {})
assert triple._jump_state == "sand_flight"
for _ in range(MIN_SUPPORT_STEPS):
    triple._observe_triple_jump(TRIPLE_LANDING_M, TRIPLE_TAKEOFF_M, {"left": {"sand"}})
assert triple._event_reason == "landed" and triple._triple_phase == 3
assert math.isclose(triple._jump_distance, TRIPLE_LANDING_M - TRIPLE_TAKEOFF_M)

# Legal completions receive the reserved finish band and the old triple-score
# saturation is gone: a minimum legal landing is not a perfect triple jump.
assert math.isclose(instance_score("long_jump", "landed", 0, 0, 1,
                                   {"jump_distance_m": 6.0}), 0.25)
assert math.isclose(instance_score("triple_jump", "landed", 0, 0, 1,
                                   {"jump_distance_m": 13.0}), 0.25)
assert math.isclose(instance_score("triple_jump", "landed", 0, 0, 1,
                                   {"jump_distance_m": 18.0}), 1.0)
assert math.isclose(instance_score("high_jump", "cleared", 0, 0, 1,
                                   {"bar_height_m": HIGH_JUMP_BARS_M[-1]}), 1.0)

print("ok: hard geometry, anti-duck, legal jumps, and finish scoring")
