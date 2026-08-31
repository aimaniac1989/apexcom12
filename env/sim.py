"""Shared MuJoCo physics and event-neutral gates for Humanoid Olympics."""

from __future__ import annotations

import gc
import math
import pathlib
from dataclasses import dataclass, field
from typing import Mapping

import mujoco
import numpy as np

from .course import (EVENT_MAX_STEPS, EVENTS, GEOM_PREFIX, HIGH_JUMP_BARS_M, OVERHEAD_GROUP,
                     PLINTH_TOP, TAKEOFF_BOARD_AFTER_M, TAKEOFF_BOARD_BEFORE_M, TRACK_HALF_W,
                     WORLD_GROUP, EventLayout, build_event, course_xml_fragment, sample_frictions)

ASSETS = pathlib.Path(__file__).parent / "assets"

ACT_DIM = 12
STATE_DIM = 256
SCAN_NX, SCAN_NY = 9, 5
OVERHEAD_N = 7
OBS_DIM = 52 + SCAN_NX * SCAN_NY + OVERHEAD_N

# At 6 m/s the inherited 1.6 m horizon was only 0.27 seconds. These scans retain
# the 104-wide signature while letting a fast policy see its next hurdle or bend.
SCAN_X = np.linspace(-0.4, 6.0, SCAN_NX)
SCAN_Y = np.linspace(-0.7, 0.7, SCAN_NY)
OVERHEAD_X = np.linspace(0.0, 4.0, OVERHEAD_N)
SCAN_CLIP = 2.0

PHYS_DT = 0.002
FRAME_SKIP = 10
ACTION_SCALE = 0.25
QVEL_GLITCH_LIMIT = 100.0
RESET_NOISE = 0.01
FALL_CLEARANCE = 0.45
UPRIGHT_MIN = 0.40
RAY_FROM_ABOVE = 3.0
AIR_DENSITY = 1.204
WIND_MAX_MS = 8.0

KP = np.array([100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40], np.float64)
KD = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2], np.float64)
DEFAULT_ANGLES = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                           -0.1, 0.0, 0.0, 0.3, -0.2, 0.0], np.float64)
GAIT_PERIOD = 0.8

# Event validity is watched at 500 Hz.  These hysteresis windows reject a
# fleeting toe scrape while remaining tiny relative to a real flight phase.
MIN_AIRBORNE_STEPS = 20       # 40 ms with no supporting feet
MIN_SUPPORT_STEPS = 4         # 8 ms continuous legal support
HIGH_CLEARANCE_MARGIN_M = 0.08
HIGH_LANDING_OFFSET_M = 0.75
SUPPORT_TOP_TOLERANCE_M = 0.06
SUPPORT_NORMAL_Z_MIN = 0.65
MIN_SUPPORT_IMPULSE_NS = 0.50


class InvalidAction(ValueError):
    """The player returned something other than a finite 12-vector."""


@dataclass(frozen=True)
class StepResult:
    obs: np.ndarray
    terminal_reason: str | None


@dataclass(frozen=True)
class InstanceParams:
    event: str
    attempt: int
    seed: int
    friction_level: float
    wind_speed: float
    wind_dir: float
    challenge: Mapping[str, float] = field(default_factory=dict)

    @property
    def wind(self) -> tuple[float, float, float]:
        return (-self.wind_speed * float(np.cos(self.wind_dir)),
                -self.wind_speed * float(np.sin(self.wind_dir)), 0.0)


def event_instances(instances_per_event: int, seed: int,
                    wind_max: float = WIND_MAX_MS) -> list[InstanceParams]:
    """One round's meet: each discipline gets the same balanced trial set.

    ``seed`` is the platform round identifier and it now moves the conditions.
    Every round still runs the same SHAPE of meet -- the same events, the same
    attempt count, and the same four evenly spaced friction/wind strata spanning
    the full envelope -- but the stratum phase is drawn per round, so no two
    rounds present the same 24 attempts.  A policy therefore has to hold up
    across the band rather than at 24 memorised operating points.

    Determinism is per seed, not per round-number: the same seed reproduces its
    meet exactly, which is what keeps the incumbent's start-of-round re-score
    like-for-like against the challengers it is being compared to.
    """
    if instances_per_event < 1:
        raise ValueError("instances_per_event must be >= 1")
    return [instance_spec(event, attempt, seed, wind_max, attempts=instances_per_event)
            for event in EVENTS for attempt in range(instances_per_event)]


def instance_spec(event: str, attempt: int, seed: int, wind_max: float = WIND_MAX_MS,
                  attempts: int = 4) -> InstanceParams:
    """One deterministically stratified condition, drawn for this round's seed."""
    if event not in EVENTS:
        raise ValueError(f"unknown Olympic event {event!r}")
    event_index = EVENTS.index(event)
    if not 0 <= attempt < attempts:
        raise ValueError(f"attempt must be in [0, {attempts}), got {attempt}")
    # Non-negative and bounded: the platform seed reaches us straight off the request, and a
    # negative one would fault np.random inside the referee rather than at the schema edge.
    round_key = int(seed) % (1 << 63)
    # The stratum PHASE is per round and per event; the stratification is not. Attempts stay
    # 1/attempts apart across the whole envelope, so every round is an equally broad sweep of the
    # friction band and the wind range -- what moves is where in the band each attempt lands.
    # Mixing the round key per event also keeps the six disciplines from shifting in lockstep.
    rng = np.random.default_rng([event_index, round_key, 0x0A11])
    phase = float(rng.uniform(0.0, 1.0))
    friction = ((attempt + phase) / attempts) % 1.0
    wind_speed = wind_max * (((attempt + 0.5 + phase) / attempts) % 1.0)
    wind_dir = (2.0 * math.pi * phase + (attempt // 2) * math.pi / 2 +
                (attempt % 2) * math.pi) % (2.0 * math.pi)
    challenge: dict[str, float] = {}
    if event == "high_jump":
        # The bar ladder stays fixed across rounds. It is the event's published difficulty scale,
        # not a condition: a round that happened to draw four low bars would not be the same test.
        challenge["bar_height_m"] = HIGH_JUMP_BARS_M[attempt % len(HIGH_JUMP_BARS_M)]
    # Per-round too, so the within-stratum friction jitter and the reset perturbation move with
    # the meet instead of replaying one fixed set of 24 starts.
    episode_rng = np.random.default_rng([event_index, attempt, round_key, 0x5151])
    return InstanceParams(event=event, attempt=attempt, seed=int(episode_rng.integers(1 << 31)),
                          friction_level=friction,
                          wind_speed=wind_speed, wind_dir=wind_dir, challenge=challenge)


def _scene_xml(layout: EventLayout) -> str:
    robot = (ASSETS / "g1_12dof.xml").read_text()
    # Visible to downward terrain rays and the renderer, but never a support:
    # a gap must be a real fall, not a lower route a policy can walk along.
    floor = (f'    <geom name="floor" type="plane" size="180 180 0.1" pos="0 0 0" '
             f'contype="0" conaffinity="0" group="{WORLD_GROUP}" rgba=".15 .16 .19 1"/>\n')
    return robot.replace("</worldbody>", floor + course_xml_fragment(layout) + "\n  </worldbody>")


_MODEL: mujoco.MjModel | None = None
_MODEL_KEY: str | None = None
_COURSE_GEOMS: list[int] = []


def _shared_model(layout: EventLayout) -> tuple[mujoco.MjModel, list[int]]:
    """Compile only the current event; six G1 mesh models would exceed the cap."""
    global _MODEL, _MODEL_KEY, _COURSE_GEOMS
    # Challenge values are runtime fields (the high-jump bar position), so one
    # compiled scene per event is sufficient even when attempts use new bars.
    key = layout.event
    if _MODEL is None or _MODEL_KEY != key:
        _MODEL = None
        _COURSE_GEOMS = []
        gc.collect()
        model = mujoco.MjModel.from_xml_string(_scene_xml(layout), _mesh_assets())
        model.opt.timestep = PHYS_DT
        model.opt.density = AIR_DENSITY
        _MODEL = model
        _MODEL_KEY = key
        _COURSE_GEOMS = [model.geom(f"{GEOM_PREFIX}{i}").id for i in range(len(layout.surfaces))]
        # Make the course's friction authoritative for foot contacts. Writing geom_friction is
        # necessary but not sufficient: MuJoCo mixes contact parameters from BOTH geoms in a pair,
        # and for friction the mix is the element-wise MAXIMUM whenever the two have equal
        # priority. g1_12dof.xml declares no geom friction, so the robot's feet sit at MuJoCo's
        # default of 1.0. Every stratum this meet draws below 1.0 was therefore taken from the
        # foot, not the track: measured 18 of the 24 launch attempts solving at exactly 1.0000
        # while the course asked for 0.52-0.98 (that figure is 0.2.0's (0.50, 1.25) band; 0.3.0
        # widened it to (0.30, 1.25), so the gap is larger). Raising priority on the course side makes its
        # contact parameters win outright, which is MuJoCo's documented mechanism for this case.
        # Guarded by tests/test_friction_reaches_contacts.py, which asserts on the solved contact
        # friction rather than on a score -- a score cannot distinguish a band that applied from
        # one that was mixed away, which is exactly how this survived the 20-seed calibration.
        #
        # Constant per geom, so it belongs here; only geom_friction varies per instance.
        for gid in _COURSE_GEOMS:
            model.geom_priority[gid] = 1
    return _MODEL, _COURSE_GEOMS


class OlympicsSim:
    """One attempt in one Olympic discipline.

    Course-relative heading/lateral channels retain the old public ONNX shape,
    but now describe the tangent of the active event's route rather than +x.
    """

    def __init__(self, params: InstanceParams):
        self.params = params
        self.layout = build_event(params.event, params.challenge)
        rng = np.random.default_rng([params.seed, 0xC0FFEE])
        self.frictions = sample_frictions(self.layout, params.friction_level, rng)
        self.model, geoms = _shared_model(self.layout)
        for gid, mu in zip(geoms, self.frictions):
            self.model.geom_friction[gid, 0] = mu
        if self.event == "high_jump":
            bar_index = next(i for i, surface in enumerate(self.layout.surfaces) if surface.kind == "bar")
            self.model.geom_pos[geoms[bar_index], 2] = PLINTH_TOP + float(
                self.layout.challenge["bar_height_m"])
        self.model.opt.wind[:] = params.wind
        self.data = mujoco.MjData(self.model)
        self._pelvis = self.model.body("pelvis").id
        self._foot_bodies = {
            self.model.body("left_ankle_roll_link").id: "left",
            self.model.body("right_ankle_roll_link").id: "right",
        }
        self._surface_kind_by_geom = {geoms[i]: surface.kind
                                      for i, surface in enumerate(self.layout.surfaces)}
        self._surface_by_geom = {geoms[i]: surface for i, surface in enumerate(self.layout.surfaces)}
        self._surface_top_z_by_geom = {geoms[i]: surface.z + surface.hz
                                       for i, surface in enumerate(self.layout.surfaces)}
        self._obstacle_geom_ids: dict[str, set[int]] = {}
        for i, surface in enumerate(self.layout.surfaces):
            if not surface.walkable:
                self._obstacle_geom_ids.setdefault(surface.kind, set()).add(geoms[i])
        self._ray_mask = np.zeros(6, np.uint8)
        self._ray_mask[WORLD_GROUP] = 1
        self._up_mask = np.zeros(6, np.uint8)
        self._up_mask[WORLD_GROUP] = 1
        self._up_mask[OVERHEAD_GROUP] = 1
        self._geomid = np.zeros(1, np.int32)
        self._action = np.zeros(ACT_DIM)
        self.steps = 0
        self.max_x = self.layout.start_x
        self._circle_prev = 0.0
        self._circle_distance = 0.0
        self._best_clearance = 0.0
        self._jump_distance = 0.0
        self._jump_landed = False
        self._landing_contact_x: float | None = None
        self._takeoff_contact_x: float | None = None
        self._triple_phase = 0
        self._jump_state = "approach"
        self._takeoff_foot: str | None = None
        self._airborne_steps = 0
        self._support_steps = 0
        self._support_impulse_ns = 0.0
        self._prev_x = self.layout.start_x
        self._high_crossed = False
        self._high_valid_crossing = False
        self._high_airborne_steps = 0
        self._foot_contact_xs: dict[tuple[str, str], list[float]] = {}
        self._foot_contact_forces: dict[tuple[str, str], float] = {}
        self._event_reason: str | None = None

    @property
    def event(self) -> str:
        return self.params.event

    @property
    def max_steps(self) -> int:
        return EVENT_MAX_STEPS[self.event]

    @property
    def progress(self) -> float:
        if self.layout.is_circular:
            return float(np.clip(self._circle_distance / self.layout.finish, 0.0, 1.0))
        start = self.layout.start_x
        return float(np.clip((self.max_x - start) / max(self.layout.finish - start, 1e-6), 0.0, 1.0))

    @property
    def distance_m(self) -> float:
        """Maximum route distance, rather than a world-axis coordinate."""
        if self.layout.is_circular:
            return max(0.0, self._circle_distance)
        return max(0.0, self.max_x - self.layout.start_x)

    @property
    def metrics(self) -> dict[str, float]:
        return {"bar_height_m": float(self.layout.challenge.get("bar_height_m", 0.0)),
                "best_clearance_m": self._best_clearance,
                "jump_distance_m": self._jump_distance,
                "takeoff_contact_x_m": self._takeoff_contact_x or 0.0,
                "triple_phase": float(self._triple_phase)}

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        rng = np.random.default_rng([self.params.seed, 0xBADA55])
        self.data.qpos[0] = self.layout.start_x + float(rng.uniform(-RESET_NOISE, RESET_NOISE))
        self.data.qpos[1] = self.layout.start_y + float(rng.uniform(-RESET_NOISE, RESET_NOISE))
        self.data.qpos[2] = PLINTH_TOP + 0.793
        half = self.layout.start_yaw / 2.0
        self.data.qpos[3:7] = [math.cos(half), 0.0, 0.0, math.sin(half)]
        self.data.qpos[7:] = DEFAULT_ANGLES + rng.uniform(-RESET_NOISE, RESET_NOISE, ACT_DIM)
        self.data.qvel[:] = rng.uniform(-RESET_NOISE, RESET_NOISE, self.model.nv)
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.max_x = float(self.data.qpos[0])
        self._circle_prev = math.atan2(float(self.data.qpos[1]), float(self.data.qpos[0]))
        self._circle_distance = 0.0
        self._best_clearance = 0.0
        self._jump_distance = 0.0
        self._jump_landed = False
        self._landing_contact_x = None
        self._takeoff_contact_x = None
        self._triple_phase = 0
        self._jump_state = "approach"
        self._takeoff_foot = None
        self._airborne_steps = 0
        self._support_steps = 0
        self._support_impulse_ns = 0.0
        self._prev_x = float(self.data.qpos[0])
        self._high_crossed = False
        self._high_valid_crossing = False
        self._high_airborne_steps = 0
        self._foot_contact_xs = {}
        self._foot_contact_forces = {}
        self._event_reason = None
        return self._obs()

    def step(self, action, max_steps: int | None = None) -> StepResult:
        a = np.asarray(action, dtype=np.float64).ravel()
        if a.shape != (ACT_DIM,):
            raise InvalidAction(f"action must be {ACT_DIM} floats, got shape {a.shape}")
        if not np.all(np.isfinite(a)):
            raise InvalidAction("action must contain only finite floats")
        self._action = np.clip(a, -10.0, 10.0)
        target = self._action * ACTION_SCALE + DEFAULT_ANGLES
        for _ in range(FRAME_SKIP):
            self.data.ctrl[:] = (target - self.data.qpos[7:]) * KP - self.data.qvel[6:] * KD
            mujoco.mj_step(self.model, self.data)
            self._observe_physics_step()
            if self._event_reason is not None:
                break
        self.steps += 1
        reason = self._event_reason or self._terminal(max_steps or self.max_steps)
        return StepResult(obs=self._obs(), terminal_reason=reason)

    # -- route and event rules ------------------------------------------------------------

    def _observe_physics_step(self) -> None:
        x, y, z = (float(v) for v in self.data.qpos[:3])
        prev_x = self._prev_x
        self.max_x = max(self.max_x, x)
        if self.layout.is_circular:
            angle = math.atan2(y, x)
            delta = (angle - self._circle_prev + math.pi) % (2 * math.pi) - math.pi
            self._circle_distance += (400.0 / (2 * math.pi)) * delta
            self._circle_prev = angle
        if abs(self._route()[1]) > TRACK_HALF_W:
            self._event_reason = "out_of_bounds"
            return
        if self.event == "hurdles_100" and self._hits("hurdle"):
            self._event_reason = "hurdle_hit"
        elif self.event == "high_jump":
            self._observe_high_jump(prev_x, x, z)
        elif self.event in {"long_jump", "triple_jump"}:
            self._observe_jump(x)
        elif self.event in {"sprint_100", "hurdles_100"} and x >= self.layout.finish:
            self._event_reason = "completed"
        elif self.event == "sprint_400" and self._circle_distance >= 400.0:
            self._event_reason = "completed"
        self._prev_x = x

    def _observe_high_jump(self, prev_x: float, x: float, z: float) -> None:
        """Require a real flight over the bar and a supported far-side landing."""
        bar_x = float(self.layout.challenge["bar_x_m"])
        height = float(self.layout.challenge["bar_height_m"])
        if self._hits("bar"):
            self._event_reason = "bar_hit"
            return
        contacts = self._foot_contacts()
        self._high_airborne_steps = self._high_airborne_steps + 1 if not contacts else 0
        if prev_x <= bar_x <= x:
            self._high_crossed = True
            self._best_clearance = max(self._best_clearance, z - PLINTH_TOP)
            if (self._high_airborne_steps >= MIN_AIRBORNE_STEPS
                    and self._best_clearance >= height + HIGH_CLEARANCE_MARGIN_M):
                self._high_valid_crossing = True
            else:
                self._event_reason = "bar_missed"
                return
        if self._high_valid_crossing:
            if self._jump_contact_illegal(contacts, "track"):
                self._event_reason = "high_foul"
                return
            if x >= bar_x + HIGH_LANDING_OFFSET_M and contacts and self._only_surface(contacts, "track"):
                if self._accumulate_support(contacts, "track"):
                    self._event_reason = "cleared"
            else:
                self._support_steps = 0
                self._support_impulse_ns = 0.0
        elif self._high_crossed and x >= bar_x + HIGH_LANDING_OFFSET_M and not self._high_valid_crossing:
            self._event_reason = "bar_missed"

    def _observe_jump(self, x: float) -> None:
        """Run the event-specific legal-contact state machine at each physics step."""
        takeoff = float(self.layout.challenge["takeoff_x_m"])
        contacts = self._foot_contacts()
        if self.event == "triple_jump":
            self._observe_triple_jump(x, takeoff, contacts)
        else:
            self._observe_long_jump(x, takeoff, contacts)

    def _observe_takeoff(self, x: float, takeoff: float,
                         contacts: dict[str, set[str]]) -> bool:
        """Remember a one-foot take-off from the narrow legal board zone."""
        feet = self._feet_on(contacts, "takeoff_board")
        if len(feet) == 1:
            foot = next(iter(feet))
            if self._only_foot_on(contacts, "takeoff_board", foot):
                self._takeoff_foot = foot
                self._takeoff_contact_x = self._first_foot_contact_x(contacts, "takeoff_board", takeoff)
        if contacts:
            self._airborne_steps = 0
            valid_board = (self._takeoff_foot is not None
                           and self._only_foot_on(contacts, "takeoff_board", self._takeoff_foot))
            if self._takeoff_foot is not None and not valid_board:
                self._event_reason = "jump_foul"
            elif self.max_x > takeoff + TAKEOFF_BOARD_AFTER_M:
                self._event_reason = "jump_foul"
            return False
        if self._takeoff_foot is None:
            if self.max_x > takeoff + TAKEOFF_BOARD_AFTER_M:
                self._event_reason = "jump_foul"
            return False
        self._airborne_steps += 1
        if self._airborne_steps < MIN_AIRBORNE_STEPS:
            return False
        self._airborne_steps = 0
        return True

    def _observe_long_jump(self, x: float, takeoff: float,
                           contacts: dict[str, set[str]]) -> None:
        if self._jump_state == "approach":
            if self._observe_takeoff(x, takeoff, contacts):
                self._jump_state = "long_flight"
            return
        if self._jump_contact_illegal(contacts, "sand"):
            self._event_reason = "jump_foul"
            return
        if not contacts:
            self._support_steps = 0
            self._support_impulse_ns = 0.0
            return
        if not self._only_surface(contacts, "sand"):
            self._event_reason = "jump_foul"
            return
        if self._support_steps == 0:
            self._landing_contact_x = self._first_foot_contact_x(contacts, "sand", x)
        if self._accumulate_support(contacts, "sand"):
            landing_x = self._landing_contact_x if self._landing_contact_x is not None else x
            self._jump_distance = max(0.0, landing_x - takeoff)
            self._jump_landed = True
            self._event_reason = "landed"

    def _observe_triple_jump(self, x: float, takeoff: float,
                             contacts: dict[str, set[str]]) -> None:
        if self._jump_state == "approach":
            if self._observe_takeoff(x, takeoff, contacts):
                self._jump_state = "hop_flight"
            return
        if self._jump_state == "hop_flight":
            self._triple_arrival(contacts, "hop_pad", self._takeoff_foot, "hop_support", 1)
            return
        if self._jump_state == "hop_support":
            self._triple_departure(contacts, "hop_pad", self._takeoff_foot, "step_flight")
            return
        if self._jump_state == "step_flight":
            opposite = "right" if self._takeoff_foot == "left" else "left"
            self._triple_arrival(contacts, "step_pad", opposite, "step_support", 2)
            return
        if self._jump_state == "step_support":
            opposite = "right" if self._takeoff_foot == "left" else "left"
            self._triple_departure(contacts, "step_pad", opposite, "sand_flight")
            return
        if self._jump_state == "sand_flight":
            if self._jump_contact_illegal(contacts, "sand"):
                self._event_reason = "jump_foul"
                return
            if not contacts:
                self._support_steps = 0
                self._support_impulse_ns = 0.0
                return
            if not self._only_surface(contacts, "sand"):
                self._event_reason = "jump_foul"
                return
            if self._support_steps == 0:
                self._landing_contact_x = self._first_foot_contact_x(contacts, "sand", x)
            if self._accumulate_support(contacts, "sand"):
                landing_x = self._landing_contact_x if self._landing_contact_x is not None else x
                self._jump_distance = max(0.0, landing_x - takeoff)
                self._jump_landed = True
                self._triple_phase = 3
                self._event_reason = "landed"

    def _triple_arrival(self, contacts: dict[str, set[str]], kind: str, foot: str | None,
                        next_state: str, phase: int) -> None:
        if self._jump_contact_illegal(contacts, kind):
            self._event_reason = "jump_foul"
            return
        if not contacts:
            self._support_steps = 0
            self._support_impulse_ns = 0.0
            return
        if foot is None or not self._only_foot_on(contacts, kind, foot):
            self._event_reason = "jump_foul"
            return
        if self._accumulate_support(contacts, kind):
            self._jump_state = next_state
            self._triple_phase = phase
            self._support_steps = 0
            self._support_impulse_ns = 0.0

    def _triple_departure(self, contacts: dict[str, set[str]], kind: str, foot: str | None,
                          next_state: str) -> None:
        if self._jump_contact_illegal(contacts, kind):
            self._event_reason = "jump_foul"
            return
        if contacts:
            if foot is None or not self._only_foot_on(contacts, kind, foot):
                self._event_reason = "jump_foul"
            self._airborne_steps = 0
            return
        self._airborne_steps += 1
        if self._airborne_steps >= MIN_AIRBORNE_STEPS:
            self._jump_state = next_state
            self._airborne_steps = 0

    def _foot_contacts(self) -> dict[str, set[str]]:
        contacts: dict[str, set[str]] = {"left": set(), "right": set()}
        self._foot_contact_xs = {}
        self._foot_contact_forces = {}
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            b1 = int(self.model.geom_bodyid[contact.geom1])
            b2 = int(self.model.geom_bodyid[contact.geom2])
            kind2 = self._surface_kind_by_geom.get(int(contact.geom2))
            kind1 = self._surface_kind_by_geom.get(int(contact.geom1))
            force = np.zeros(6, np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            normal_force = max(0.0, float(force[0]))
            if (b1 in self._foot_bodies and kind2 is not None
                    and self._is_top_course_contact(int(contact.geom2), contact)):
                foot = self._foot_bodies[b1]
                contacts[foot].add(kind2)
                self._foot_contact_xs.setdefault((foot, kind2), []).append(float(contact.pos[0]))
                self._foot_contact_forces[(foot, kind2)] = (
                    self._foot_contact_forces.get((foot, kind2), 0.0) + normal_force)
            if (b2 in self._foot_bodies and kind1 is not None
                    and self._is_top_course_contact(int(contact.geom1), contact)):
                foot = self._foot_bodies[b2]
                contacts[foot].add(kind1)
                self._foot_contact_xs.setdefault((foot, kind1), []).append(float(contact.pos[0]))
                self._foot_contact_forces[(foot, kind1)] = (
                    self._foot_contact_forces.get((foot, kind1), 0.0) + normal_force)
        return {foot: kinds for foot, kinds in contacts.items() if kinds}

    def _is_top_course_contact(self, course_geom: int, contact) -> bool:
        top = self._surface_top_z_by_geom.get(course_geom)
        surface = self._surface_by_geom.get(course_geom)
        if top is None or surface is None or not surface.walkable:
            return False
        dx, dy = float(contact.pos[0]) - surface.x, float(contact.pos[1]) - surface.y
        c, s = math.cos(surface.yaw), math.sin(surface.yaw)
        local_x, local_y = c * dx + s * dy, -s * dx + c * dy
        return (abs(float(contact.pos[2]) - top) <= SUPPORT_TOP_TOLERANCE_M
                and abs(float(contact.frame[2])) >= SUPPORT_NORMAL_Z_MIN
                and abs(local_x) <= surface.hx + SUPPORT_TOP_TOLERANCE_M
                and abs(local_y) <= surface.hy + SUPPORT_TOP_TOLERANCE_M)

    @staticmethod
    def _feet_on(contacts: dict[str, set[str]], kind: str) -> set[str]:
        return {foot for foot, kinds in contacts.items() if kind in kinds}

    @staticmethod
    def _only_surface(contacts: dict[str, set[str]], kind: str) -> bool:
        return bool(contacts) and all(kinds == {kind} for kinds in contacts.values())

    @staticmethod
    def _only_foot_on(contacts: dict[str, set[str]], kind: str, foot: str) -> bool:
        return set(contacts) == {foot} and contacts.get(foot) == {kind}

    def _first_foot_contact_x(self, contacts: dict[str, set[str]], kind: str,
                              fallback_x: float) -> float:
        points = [point for foot, kinds in contacts.items() if kind in kinds
                  for point in self._foot_contact_xs.get((foot, kind), [])]
        return min(points, default=fallback_x)

    def _accumulate_support(self, contacts: dict[str, set[str]], kind: str) -> bool:
        """Confirm support only after both a short duration and real normal impulse."""
        force_keys = [(foot, kind) for foot, kinds in contacts.items() if kind in kinds]
        known_force = any(key in self._foot_contact_forces for key in force_keys)
        normal_force = (sum(self._foot_contact_forces.get(key, 0.0) for key in force_keys)
                        if known_force else MIN_SUPPORT_IMPULSE_NS / (MIN_SUPPORT_STEPS * PHYS_DT))
        self._support_steps += 1
        self._support_impulse_ns += normal_force * PHYS_DT
        return (self._support_steps >= MIN_SUPPORT_STEPS
                and self._support_impulse_ns >= MIN_SUPPORT_IMPULSE_NS)

    def _jump_contact_illegal(self, contacts: dict[str, set[str]], allowed: str) -> bool:
        """Reject wrong surfaces and any non-foot course contact after take-off."""
        if contacts and not self._only_surface(contacts, allowed):
            return True
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            for course_gid, robot_gid in ((int(contact.geom1), int(contact.geom2)),
                                          (int(contact.geom2), int(contact.geom1))):
                kind = self._surface_kind_by_geom.get(course_gid)
                if kind is None:
                    continue
                body = int(self.model.geom_bodyid[robot_gid])
                if body not in self._foot_bodies or not self._is_top_course_contact(course_gid, contact):
                    return True
        return False

    def _hits(self, kind: str) -> bool:
        gids = self._obstacle_geom_ids.get(kind)
        if not gids:
            return False
        return any(self._geom_pair_hits(gids, int(self.data.contact[i].geom1),
                                        int(self.data.contact[i].geom2))
                   for i in range(self.data.ncon))

    @staticmethod
    def _geom_pair_hits(gids: set[int], geom1: int, geom2: int) -> bool:
        return geom1 in gids or geom2 in gids

    # -- perception -----------------------------------------------------------------------

    def _yaw(self) -> float:
        qw, qx, qy, qz = self.data.qpos[3:7]
        return float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))

    def _route(self) -> tuple[float, float, float]:
        """(target yaw, signed cross-track offset, remaining metres)."""
        x, y = (float(v) for v in self.data.qpos[:2])
        if self.layout.is_circular:
            radius = float(self.layout.challenge["radius_m"])
            theta = math.atan2(y, x)
            # Positive cross-track is always to the runner's left. For CCW
            # travel that is toward the centre, not outward from the circle.
            return theta + math.pi / 2, radius - math.hypot(x, y), max(0.0, 400.0 - self._circle_distance)
        return 0.0, y, max(0.0, self.layout.finish - x)

    def _ray_down(self, x: float, y: float, z_from: float) -> float:
        d = mujoco.mj_ray(self.model, self.data, np.array([x, y, z_from]),
                          np.array([0.0, 0.0, -1.0]), self._ray_mask, 1, -1, self._geomid)
        return z_from - d if d >= 0 else -SCAN_CLIP

    def _ray_up(self, x: float, y: float, z_from: float) -> float:
        d = mujoco.mj_ray(self.model, self.data, np.array([x, y, z_from]),
                          np.array([0.0, 0.0, 1.0]), self._up_mask, 1, -1, self._geomid)
        return SCAN_CLIP if d < 0 else min(d, SCAN_CLIP)

    def _obs(self) -> np.ndarray:
        d, yaw = self.data, self._yaw()
        px, py, pz = (float(v) for v in d.qpos[:3])
        c, s = np.cos(yaw), np.sin(yaw)
        route_yaw, lateral, remaining = self._route()
        heading_error = (yaw - route_yaw + math.pi) % (2 * math.pi) - math.pi
        rot = np.array(d.xmat[self._pelvis]).reshape(3, 3)
        lin = rot.T @ d.qvel[:3]
        ang = rot.T @ d.qvel[3:6]
        grav = rot.T @ np.array([0.0, 0.0, -1.0])
        scan = np.empty(SCAN_NX * SCAN_NY)
        k = 0
        for dx in SCAN_X:
            for dy in SCAN_Y:
                wx, wy = px + c * dx - s * dy, py + s * dx + c * dy
                scan[k] = self._ray_down(wx, wy, pz + RAY_FROM_ABOVE) - pz
                k += 1
        np.clip(scan, -SCAN_CLIP, SCAN_CLIP, out=scan)
        over = np.array([self._ray_up(px + c * dx, py + s * dx, pz + 0.05) for dx in OVERHEAD_X])
        ground = self._ray_down(px, py, pz + RAY_FROM_ABOVE)
        phase = (self.steps * PHYS_DT * FRAME_SKIP % GAIT_PERIOD) / GAIT_PERIOD
        return np.concatenate([
            grav, ang * 0.25, lin * 2.0, d.qpos[7:] - DEFAULT_ANGLES, d.qvel[6:] * 0.05,
            self._action, [np.sin(2 * math.pi * phase), np.cos(2 * math.pi * phase)],
            [np.sin(heading_error), np.cos(heading_error)], [lateral, remaining / 10.0,
             np.clip(pz - ground, -SCAN_CLIP, SCAN_CLIP)], scan, over,
        ]).astype(np.float32)

    # -- common gates ---------------------------------------------------------------------

    def _terminal(self, max_steps: int) -> str | None:
        qpos, qvel = self.data.qpos, self.data.qvel
        if not (np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel))):
            return "physics_glitch"
        if np.max(np.abs(qvel)) > QVEL_GLITCH_LIMIT:
            return "physics_glitch"
        px, py, pz = (float(v) for v in qpos[:3])
        _, lateral, _ = self._route()
        if abs(lateral) > TRACK_HALF_W:
            return "out_of_bounds"
        if float(self.data.xmat[self._pelvis].reshape(3, 3)[2, 2]) < UPRIGHT_MIN:
            return "fell"
        if pz - self._ray_down(px, py, pz + RAY_FROM_ABOVE) < FALL_CLEARANCE:
            return "fell"
        if self.event in {"sprint_100", "hurdles_100"} and px >= self.layout.finish:
            return "completed"
        if self.event == "sprint_400" and self._circle_distance >= 400.0:
            return "completed"
        if self.steps >= max_steps:
            return "timeout"
        return None


def _mesh_assets() -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in (ASSETS / "meshes").glob("*.STL")}
