"""The course's sliding friction must reach the solver, not just the model.

Regression guard for a defect that shipped in 0.1.0 and was invisible in every score. MuJoCo mixes
contact parameters from both geoms in a pair, and for friction the mix is the element-wise MAXIMUM
when the two geoms have equal `geom_priority`. `g1_12dof.xml` sets no geom friction, so the robot's
feet sit at MuJoCo's default of 1.0. Every stratum this meet asks for below 1.0 was therefore taken
from the foot rather than the track: 18 of the 24 launch attempts solved at exactly 1.0000 while the
course asked for 0.52-0.98, so three of the four condition strata per event were duplicates of each
other at full grip. That 0.52-0.98 is what 0.2.0's `(0.50, 1.25)` band asked for; 0.3.0 widened the
band to `(0.30, 1.25)`, which only widens the gap this guards against.

These assert at CONTACT level on purpose. A score-based check cannot tell "friction was applied"
apart from "friction was ignored and the policy happens to be robust" -- which is exactly how this
survived the 20-seed native calibration, whose sample standard deviation was 0.0.

    python -m pytest tests/test_friction_reaches_contacts.py
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

import env.course as course
from env.sim import InstanceParams, OlympicsSim, event_instances

SETTLE_STEPS = 80
TOL = 1e-4


def _geom_name(model, gid: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""


def _settle(params: InstanceParams) -> OlympicsSim:
    """Hold the default pose long enough for foot-to-course contacts to form."""
    sim = OlympicsSim(params)
    sim.reset()
    for _ in range(SETTLE_STEPS):
        sim.step(np.zeros(12))
    return sim


def _course_contacts(sim: OlympicsSim) -> list[tuple[float, float]]:
    """(geom mu, solved contact mu) for every contact involving a course geom."""
    out = []
    for c in range(sim.data.ncon):
        con = sim.data.contact[c]
        for gid in (con.geom1, con.geom2):
            if _geom_name(sim.model, gid).startswith(course.GEOM_PREFIX):
                out.append((float(sim.model.geom_friction[gid, 0]), float(con.friction[0])))
    return out


def _params(event: str = "sprint_100", friction_level: float = 1.0) -> InstanceParams:
    """A settled-pose instance; wind off so only friction is under test."""
    return InstanceParams(event=event, attempt=0, seed=1, friction_level=friction_level,
                          wind_speed=0.0, wind_dir=0.0)


def test_course_geoms_outrank_the_robot():
    """Course geoms must carry a higher contact priority than everything else.

    At equal priority MuJoCo takes max() of the two frictions, and since the robot's feet default
    to 1.0 that discards every value the course asked for below it.
    """
    sim = OlympicsSim(_params())
    course_prio, other_prio = set(), set()
    for gid in range(sim.model.ngeom):
        target = course_prio if _geom_name(sim.model, gid).startswith(course.GEOM_PREFIX) else other_prio
        target.add(int(sim.model.geom_priority[gid]))

    assert course_prio, "no course geoms found"
    assert min(course_prio) > max(other_prio), (
        f"course priority {course_prio} must exceed everything else {other_prio}; "
        "at equal priority MuJoCo takes max() of the two frictions and the course loses"
    )


def test_solved_contact_friction_matches_the_course():
    """The mu MuJoCo actually solves with must be the mu the course asked for."""
    # friction_level 1.0 is the slippery end of the band, so this is the case the defect hid.
    sim = _settle(_params(friction_level=1.0))
    contacts = _course_contacts(sim)
    assert contacts, "no course contacts formed; the probe cannot conclude anything"
    for geom_mu, solved_mu in contacts:
        assert solved_mu == pytest.approx(geom_mu, abs=TOL), (
            f"solver used mu {solved_mu:.4f} where the course geom asked for {geom_mu:.4f}"
        )


def test_contact_friction_is_not_pinned_to_the_default():
    """A slippery stratum must not silently solve at MuJoCo's 1.0 foot default.

    This is the specific failure signature: geom mu well below 1.0, contact mu exactly 1.0.
    """
    sim = _settle(_params(friction_level=1.0))
    contacts = _course_contacts(sim)
    assert contacts, "no course contacts formed"
    asked = max(geom_mu for geom_mu, _ in contacts)
    assert asked < 1.0, (
        f"test is not exercising the defect: every course geom asked for mu >= 1.0 ({asked:.4f}), "
        "where max() with the foot's 1.0 is indistinguishable from a working fix"
    )
    for geom_mu, solved_mu in contacts:
        assert solved_mu != pytest.approx(1.0, abs=TOL), (
            f"contact mu is pinned to MuJoCo's 1.0 default while the course asked for {geom_mu:.4f}"
        )


def test_every_launch_stratum_reaches_the_solver():
    """All 24 attempts of a full meet, not just a hand-built instance.

    The original defect was stratum-dependent -- attempt 0 of each event happened to ask for mu
    above 1.0 and so appeared to work, while attempts 1-3 were clamped. A single-instance test
    would have passed. This is the assertion that actually pins the bug down.
    """
    tasks = event_instances(4, seed=12345, wind_max=8.0)
    assert len(tasks) == 24, f"expected the 24-attempt launch meet, got {len(tasks)}"

    clamped, checked, asked_below_one = [], 0, 0
    for params in tasks:
        contacts = _course_contacts(_settle(params))
        if not contacts:
            continue
        for geom_mu, solved_mu in contacts:
            checked += 1
            if geom_mu < 1.0 - TOL:
                asked_below_one += 1
            if abs(solved_mu - geom_mu) > TOL:
                clamped.append((params.event, params.attempt, geom_mu, solved_mu))

    assert checked, "no course contacts formed across the whole meet"
    assert asked_below_one, (
        "no attempt asked for mu below 1.0, so this meet cannot detect the max() clamp at all"
    )
    assert not clamped, (
        f"{len(clamped)} contacts did not solve at the course's mu, e.g. "
        + ", ".join(f"{e}/att{a}: asked {g:.4f} got {s:.4f}" for e, a, g, s in clamped[:4])
    )


def test_strata_are_distinct_at_the_solver():
    """The four strata must be four different friction regimes where it counts.

    Under the defect the three sub-1.0 strata all solved at exactly 1.0, collapsing a four-point
    condition sweep into two. Distinct geom values are not enough; they have to differ in the
    solved contact.
    """
    solved_by_attempt = {}
    for params in event_instances(4, seed=12345, wind_max=8.0):
        if params.event != "sprint_100":
            continue
        contacts = _course_contacts(_settle(params))
        assert contacts, f"no course contacts for attempt {params.attempt}"
        solved_by_attempt[params.attempt] = round(
            float(np.mean([solved for _, solved in contacts])), 4)

    assert len(solved_by_attempt) == 4, f"expected 4 strata, got {sorted(solved_by_attempt)}"
    distinct = set(solved_by_attempt.values())
    assert len(distinct) == 4, (
        f"the four strata collapsed to {len(distinct)} solved friction value(s): {solved_by_attempt}"
    )
