"""Dense reward shaping for Humanoid Olympics, one event at a time.

The competition score is almost entirely terminal -- a completed race pays 0.25 upward, everything
else pays `0.24 * progress` -- and terminal-only signal is far too sparse to learn a gait from. So
this file turns the scoring rules into per-step reward, and it is written against six things
measured in this repo rather than guessed:

1.  **In-flight uprightness must be continuous, not terminal.** The `fell` gate is
    `xmat[2,2] < 0.40` with NO airborne exemption, and a jump is killed mid-flight while still
    rising if the torso rotates past ~66 deg. A CEM search that rewarded attitude only on landing
    got zero gradient because landing never happened. `w_upright` is therefore paid every step.

2.  **Fouls are hard zeros, not partial credit.** `jump_foul`, `high_foul`, `out_of_bounds` and
    `physics_glitch` score 0.0 while a plain `timeout` still earns `0.24 * progress`. So a foul
    must be strictly worse than stopping, or the policy learns to gamble.

3.  **But refusing to act must not be optimal either.** Symmetrically, if fouls are penalised
    without rewarding the legal take-off, the optimum becomes "stop at the board" -- worth ~0.10
    and a local minimum that no amount of further training escapes. `w_takeoff` and the phase
    bonuses exist to make committing strictly better than balking.

4.  **The action must stay smooth -- but only barely.** `env/sim.py` clips to +-10, and a
    controller railed against that clip is sensitive to perturbation, so a small penalty is worth
    keeping. It was 20x larger until the rival artifacts were read: at 0.004 it cost an aggressive
    gait 42% of everything a completed 100 m pays, and made a large-amplitude one net negative.
    See `w_smooth` and note 7.

5.  **Falling must cost something, and progress must be paid as a FRACTION.** Both learned from an
    11.2M-step PPO run that reached raw 0.0519 with `fin 0/200` -- not one completion in 200
    episodes, every event dying at 13-16 m of route. Two causes, both in this file: `fell` had no
    terminal penalty at all (so under autoreset the same metres were simply re-earned, leaving the
    linear progress term indifferent to falling), and progress was paid per METRE against a scorer
    that pays per route fraction (so the 400 m was shaped 28x harder than the high jump for events
    `meet_score` weights equally). See `r_fell` and `w_progress` below.

6.  **high_jump has no progress credit at all, so nothing here pointed at the bar.** It is the
    only event `instance_score` does not pay on `progress`: it pays `0.24 * clearance/target`,
    and `sim._best_clearance` is written in exactly one place -- inside `if prev_x <= bar_x <= x`
    in `env/sim.py` -- so it is 0.0 until the pelvis has already crossed the bar plane. Two
    consequences. The `w_score` anchor is identically 0 on this event, so the one term that keeps
    shaping honest is switched off precisely where shaping is loosest. And `w_clearance` keys off
    that same field, making it a reward for a jump already made rather than a gradient toward
    making one -- so walking the 14 m route and timing out banked ~1300 against a real score of
    0.000, which is note 3's "stop at the board" local optimum with no `w_takeoff` to oppose it
    (that term is gated on `JUMP_EVENTS`, and high_jump is not in it). `w_apex` is the missing
    gradient: note 1 applied to height instead of attitude.

7.  **The winning regime is high-power ballistic, and unbounded power is why it exists.** MuJoCo
    `<motor>` actuators deliver their full torque at any joint velocity -- there is no torque-speed
    curve, so mechanical power is unbounded even though torque is capped. The 0.6154 leader in
    `analysis/riv_0830_seed1.json` runs the 400 m at 12.4 m/s, clears the high bar with 1.95 m of
    pelvis rise against a 1.30 m bar, and lands a 12.33 m long jump. None of that is physical; all
    of it is legal. A shaping function tuned for a plausible walking gait -- which this one was --
    prices the winning gait out of reach, which is what notes 4 and 5 were doing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Pelvis height at reset, above the plinth (`env/sim.py` sets qpos[2] = PLINTH_TOP + 0.793).
# `w_apex` pays rise ABOVE this, so merely standing earns nothing.
STAND_PELVIS_Z = 0.793

RACE_EVENTS = frozenset({"sprint_100", "sprint_400", "hurdles_100"})
JUMP_EVENTS = frozenset({"long_jump", "triple_jump"})

# Terminal reasons the scorer maps to a hard 0.0 (env/scoring.py). Everything else keeps
# progress credit, so these must carry a penalty a policy cannot gamble against.
FOUL_REASONS = frozenset({"jump_foul", "high_foul", "out_of_bounds", "physics_glitch",
                          "invalid_action"})
SUCCESS_REASONS = frozenset({"completed", "cleared", "landed"})
# `fell` is the fourth outcome, and the original version of this file paid it NOTHING terminal.
# Measured consequence over an 11.2M-step run: `fin 0/200` throughout, with every event dying at
# 13-16 m of route. Under autoreset a fall costs only the episode, and the same metres are simply
# re-earned after the reset, so a linear progress term is INDIFFERENT to falling -- there was no
# gradient anywhere in this function pushing the policy to stay up past the point it fell.
# `timeout` stays unpenalised deliberately: the scorer keeps `0.24 * progress` for it, and
# punishing it too is what turns "stop at the board" into the local optimum (note 3 above).
FALL_REASONS = frozenset({"fell"})


@dataclass
class RewardConfig:
    # Per unit of ROUTE FRACTION gained, not per metre. `instance_score` pays progress as a
    # fraction and `meet_score` weights the six events equally, but a per-metre term paid a full
    # sprint_400 route 4,800 against a full high_jump route's 168 -- a 28:1 shaping skew across
    # events the scorer treats 1:1. 1200 is chosen to leave sprint_100 (a ~102 m route at the old
    # 12.0/m) on exactly the scale every other weight in this dataclass was tuned against, so this
    # re-weights the events without rescaling the function.
    w_progress: float = 1200.0
    # Both were cut, because adding r_fell below is what makes standing still a candidate: these
    # are per-step but the step caps run 900 (high jump) to 3600 (400 m), so "stay upright and do
    # nothing" paid 4x more on the 400 m than on the jumps, for identical zero progress. At the old
    # 0.03/0.35 a 400 m timeout banked 864 -- more than 70% of that route's entire progress budget.
    w_alive: float = 0.01         # per surviving step; keeps early training from suiciding
    w_upright: float = 0.20       # per step, on (xmat22 - UPRIGHT_MIN); the flight-attitude term
    # Both cut 20x and 8x, measured against what actually wins. `analysis/riv_0830_seed1.json`:
    # the 0.6154 leader completes the 100 m in 528 steps (9.66 m/s), the 400 m at 12.4 m/s, clears
    # the high bar with 1.95 m of pelvis rise and lands a 12.33 m long jump. That regime exists
    # because MuJoCo `<motor>` has no torque-speed curve -- torque is capped (139 N.m knee and hip
    # roll, 88 hip pitch/yaw, 50 ankle) but mechanical POWER is not, and the joints carry
    # damping 0.001 / frictionloss 0.1. High-power ballistic locomotion is therefore reachable.
    #
    # At the old weights our function forbade it. A completed 100 m pays ~1512 all in. An
    # aggressive gait (|a|~5, |da|~5) paid -634 in smoothness alone, 42% of that; |a|=|da|=10 came
    # to -2534 and went net NEGATIVE at -1529; true bang-bang reached -10138. Note 4 called
    # w_smooth "cheap insurance against inheriting a bang-bang gait" -- the measurement says a
    # bang-bang gait is the winning gait, and 0.004 taxed it out of existence. Our policy sat at
    # 2.4 m/s, which is about where the smoothness-constrained optimum is.
    w_smooth: float = 0.0002      # per step, on ||a - a_prev||^2
    w_effort: float = 0.0001      # per step, on ||a||^2
    w_lateral: float = 0.25       # per step, on cross-track error^2 (out_of_bounds is a zero)
    w_takeoff: float = 15.0       # one-off, on a legal one-foot board departure
    w_phase: float = 40.0         # per triple-jump phase reached (hop, step, landing)
    w_clearance: float = 8.0      # per metre of new best high-jump clearance
    # Per metre of new-best pelvis rise, paid only within `apex_window_m` of the bar. This is the
    # only term on high_jump that pays before the crossing (note 6). Sized against the failure it
    # must beat: the lowest bar needs 1.00 + HIGH_CLEARANCE_MARGIN_M - STAND_PELVIS_Z = 0.287 m of
    # rise, and an early attempt at it mostly ends `fell` at -60. 300 pays ~86 for that rise, so
    # trying beats not trying, while staying below r_success so it cannot substitute for actually
    # clearing. Untuned -- this is the first coefficient to revisit once high_jump scores at all.
    # OFF, measured. Over ~6M high_jump steps it never produced one clearance, and the rise it
    # buys near the bar converts clean ducks into strikes: `bar_hit` records no clearance and so
    # scores 0.000, where a duck taking `bar_missed` scores 0.077. high_jump fell 0.0772 -> 0.0338
    # the moment it entered the pool, and by 35.7M `bar_missed` had collapsed 11.7 -> 2.6 per 200
    # as the extra height destabilised the approach and episodes started falling short of the bar.
    # Ducking as upright as each bar allows is worth 0.1671 and is reachable; a 0.29 m rise is not,
    # until the gait survives past 58 m. Restore this when that is true.
    w_apex: float = 0.0
    apex_window_m: float = 2.5    # a rise anywhere else on the route is not a high jump
    w_hurdle: float = 10.0        # per hurdle passed without contact
    r_success: float = 120.0      # terminal, on completed / cleared / landed
    # Terminal, on `fell`. The missing term: without it a fall is free (it even PAID, via w_score
    # on the progress already banked) and nothing opposed the progress term's push to accelerate
    # until the gait broke. Sized against the alternative failure it creates -- a 100 m fall at
    # 15% route still nets ~0.8/step against ~0.13/step for standing still, so moving stays
    # strictly better than balking. NOTE this is weak on the 400 m specifically, where 10 m of
    # route is worth so little that no fall penalty can beat standing; that event needs a
    # curriculum, not a coefficient.
    r_fell: float = -60.0
    # Was -40, against a progress term paying hundreds. A foul that bought a few metres was net
    # positive, and the 11.2M-step run showed it: fouls climbing 2-9 -> 23-28 per 200 episodes
    # over the last ~30 updates while high_jump fell 0.0403 -> 0.0163.
    r_foul: float = -150.0        # terminal, on a hard-zero reason
    w_score: float = 200.0        # terminal, on the REAL instance score, so shaping cannot
                                  # drift away from what actually wins the round


@dataclass
class EpisodeTracker:
    """Per-episode bookkeeping for terms that are deltas rather than levels."""

    cfg: RewardConfig = field(default_factory=RewardConfig)
    prev_progress: float = 0.0
    prev_action: np.ndarray | None = None
    prev_phase: int = 0
    prev_clearance: float = 0.0
    prev_apex: float = 0.0
    took_off: bool = False
    hurdles_passed: int = 0

    def reset(self, sim) -> None:
        self.prev_progress = float(sim.progress)
        self.prev_action = None
        self.prev_phase = 0
        self.prev_clearance = 0.0
        self.prev_apex = 0.0
        self.took_off = False
        self.hurdles_passed = 0

    def step(self, sim, action: np.ndarray, reason: str | None) -> float:
        from env.course import HURDLE_HEIGHTS_M, PLINTH_TOP  # noqa: F401  (hurdle parity)
        from env.scoring import instance_score
        from env.sim import UPRIGHT_MIN

        c = self.cfg
        r = 0.0

        # -- progress along the event's own route ------------------------------------------
        # `sim.progress` is the SAME clipped route fraction `instance_score` consumes, and it is
        # monotone (max_x / accumulated _circle_distance), so the delta is never negative.
        progress = float(sim.progress)
        r += c.w_progress * (progress - self.prev_progress)
        self.prev_progress = progress

        # -- posture: paid EVERY step, including airborne ones ------------------------------
        upright = float(sim.data.xmat[sim._pelvis].reshape(3, 3)[2, 2])
        r += c.w_upright * (upright - UPRIGHT_MIN)
        r += c.w_alive

        # -- lane discipline: leaving it is a hard zero --------------------------------------
        lateral = float(sim._route()[1])
        r -= c.w_lateral * lateral * lateral

        # -- smoothness, against the +-10 env clip -------------------------------------------
        a = np.asarray(action, np.float64).ravel()
        if self.prev_action is not None:
            d = a - self.prev_action
            r -= c.w_smooth * float(d @ d)
        r -= c.w_effort * float(a @ a)
        self.prev_action = a.copy()

        # -- event-specific structure ---------------------------------------------------------
        if sim.event in JUMP_EVENTS:
            if not self.took_off and sim._jump_state != "approach":
                r += c.w_takeoff             # committing to a legal departure beats balking
                self.took_off = True
            phase = int(sim._triple_phase)
            if phase > self.prev_phase:
                r += c.w_phase * (phase - self.prev_phase)
                self.prev_phase = phase
        elif sim.event == "high_jump":
            clearance = float(sim._best_clearance)
            if clearance > self.prev_clearance:
                r += c.w_clearance * (clearance - self.prev_clearance)
                self.prev_clearance = clearance
            # The gradient `w_clearance` cannot supply (note 6): pelvis rise near the bar, paid
            # whether or not the crossing lands. On the NEW-BEST delta only -- a level term would
            # pay every step of a hover, and the apex is what the bar actually judges.
            if abs(float(sim.data.qpos[0])
                   - float(sim.layout.challenge["bar_x_m"])) <= c.apex_window_m:
                rise = float(sim.data.qpos[2]) - PLINTH_TOP - STAND_PELVIS_Z
                if rise > self.prev_apex:
                    r += c.w_apex * (rise - self.prev_apex)
                    self.prev_apex = rise
        elif sim.event == "hurdles_100":
            from env.course import build_event
            px = float(sim.data.qpos[0])
            passed = sum(1 for s in sim.layout.surfaces if s.kind == "hurdle" and s.x < px)
            if passed > self.hurdles_passed:
                r += c.w_hurdle * (passed - self.hurdles_passed)
                self.hurdles_passed = passed

        # -- terminal ---------------------------------------------------------------------------
        if reason is not None:
            if reason in FOUL_REASONS:
                r += c.r_foul
            elif reason in FALL_REASONS:
                r += c.r_fell
            elif reason in SUCCESS_REASONS:
                r += c.r_success
            # Anchor the shaping to the real objective: whatever the dense terms encourage, the
            # thing that actually wins rounds is this number.
            r += c.w_score * float(instance_score(
                sim.event, reason, sim.progress, sim.steps, sim.max_steps, sim.metrics))
        return float(r)
