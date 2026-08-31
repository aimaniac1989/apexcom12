"""Precompiled MuJoCo scenes, so a worker costs ~150 MB instead of ~500 MB.

`env.sim._shared_model` compiles the scene from XML on first use. That is 27 STL meshes converted
to convex hulls: ~1-2 s and ~500 MB resident, PER WORKER PROCESS. At 10 workers that is ~5 GB of
identical geometry, and it is what makes the vec env die on a 16 GB box.

MuJoCo's binary model format is the compiler's own output, so loading a `.mjb` produces the same
`MjModel` far more cheaply. This module compiles each event once, caches it, and injects it into
`env.sim`'s module globals so `_shared_model` short-circuits.

**This touches the referee's own physics, so it is gated.** `selftest()` asserts a rollout under
the injected model is BIT-IDENTICAL to one under a freshly compiled model. If that ever fails,
every worker would be training on physics that is not what scores us -- the single worst failure
mode available -- so the gate is hard, not a warning.

    PYTHONPATH=. python engine/mjb.py            # build the cache and run the gate
"""

from __future__ import annotations

import hashlib
import pathlib

import numpy as np

CACHE = pathlib.Path(__file__).parent / "runs" / "mjb"


def _cache_path(event: str) -> pathlib.Path:
    return CACHE / f"{event}.mjb"


def build(event: str, force: bool = False) -> pathlib.Path:
    """Compile `event`'s scene once and write it to the cache."""
    import mujoco

    from env.sim import OlympicsSim, instance_spec

    out = _cache_path(event)
    if out.exists() and not force:
        return out
    CACHE.mkdir(parents=True, exist_ok=True)
    # Going through OlympicsSim rather than the XML directly guarantees we capture every
    # post-compile mutation _shared_model makes -- opt.timestep, opt.density, and the
    # geom_priority that makes course friction authoritative for foot contacts.
    sim = OlympicsSim(instance_spec(event, 0, 1))
    mujoco.mj_saveModel(sim.model, str(out), None)
    del sim
    return out


def install(event: str) -> None:
    """Load `event` from cache into `env.sim`'s globals so `_shared_model` returns it.

    Called once per worker process, before the first OlympicsSim of that event.
    """
    import mujoco

    from env import sim as envsim

    path = _cache_path(event)
    if not path.exists():
        build(event)
    model = mujoco.MjModel.from_binary_path(str(path))
    layout_n = len(_layout(event).surfaces)
    envsim._MODEL = model
    envsim._MODEL_KEY = event
    envsim._COURSE_GEOMS = [model.geom(f"{envsim.GEOM_PREFIX}{i}").id for i in range(layout_n)]


def _layout(event: str):
    from env.course import build_event
    from env.sim import HIGH_JUMP_BARS_M

    challenge = {"bar_height_m": HIGH_JUMP_BARS_M[0]} if event == "high_jump" else {}
    return build_event(event, challenge)


def _digest(event: str, steps: int = 120) -> str:
    """Fingerprint a deterministic rollout: qpos+qvel after every step, hashed."""
    from env.sim import OlympicsSim, instance_spec

    sim = OlympicsSim(instance_spec(event, 1, 7))
    sim.reset()
    h = hashlib.sha256()
    rng = np.random.default_rng(0)
    for _ in range(steps):
        a = rng.uniform(-1.0, 1.0, 12)
        result = sim.step(a)
        h.update(np.ascontiguousarray(sim.data.qpos, np.float64).tobytes())
        h.update(np.ascontiguousarray(sim.data.qvel, np.float64).tobytes())
        if result.terminal_reason is not None:
            sim.reset()
    del sim
    return h.hexdigest()[:16]


def selftest(events=None) -> None:
    """Hard gate: injected .mjb must reproduce a fresh compile bit for bit."""
    from env import sim as envsim
    from env.course import EVENTS

    events = events or EVENTS
    for event in events:
        # Fresh compile: clear the cache-injection so _shared_model rebuilds from XML.
        envsim._MODEL, envsim._MODEL_KEY, envsim._COURSE_GEOMS = None, None, []
        want = _digest(event)

        build(event, force=True)
        envsim._MODEL, envsim._MODEL_KEY, envsim._COURSE_GEOMS = None, None, []
        install(event)
        got = _digest(event)

        if got != want:
            raise AssertionError(
                f"{event}: .mjb rollout {got} != fresh-compile {want}. Training would run on "
                "different physics than the referee scores. Refusing to use the cache.")
        print(f"   {event:14s} bit-identical  {got}")


def _rss_mb() -> float:
    import os

    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except ImportError:
        if hasattr(os, "getrusage"):          # Linux: ru_maxrss is KiB
            import resource

            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        return float("nan")


if __name__ == "__main__":
    from env.course import EVENTS

    print("building cache")
    for ev in EVENTS:
        p = build(ev, force=True)
        print(f"   {ev:14s} {p.stat().st_size / 1e6:6.1f} MB  {p}")

    print("\ndeterminism gate (.mjb vs fresh compile)")
    selftest()

    print("\nresident memory, one worker's worth of scene")
    base = _rss_mb()
    install(EVENTS[0])
    from env.sim import OlympicsSim, instance_spec

    sim = OlympicsSim(instance_spec(EVENTS[0], 0, 1))
    sim.reset()
    print(f"   after .mjb load + sim: {_rss_mb():.0f} MB rss (delta {_rss_mb() - base:+.0f} MB)")
