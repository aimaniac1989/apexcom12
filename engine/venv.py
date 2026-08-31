"""Multiprocess vectorised Humanoid Olympics environment.

Two facts about `env/sim.py` shape this file, and getting either wrong corrupts training silently:

  * **`_shared_model` is a module global keyed by event, and `OlympicsSim.__init__` writes
    `geom_friction` and `opt.wind` into it.** Two live sims of the SAME event inside one process
    therefore overwrite each other's conditions -- the second sim's friction becomes the first
    sim's too, and nothing raises. Each worker here owns exactly one live sim at a time.

  * **Switching events recompiles the scene** (a fresh `mujoco.MjModel.from_xml_string` over 27
    STL convex hulls, ~1-2 s and a few hundred MB). A worker that round-robins events would spend
    most of its life compiling. So each worker is PINNED to one event for its whole life and
    cycles attempts/seeds within it; the pool covers all six by assigning workers round-robin.

Throughput on a 12-core i7-12700 is ~860 control steps/s per worker, so 10 workers give ~8.6k
steps/s. Physics is ~65% of that and is CPU-only -- see `engine/README.md` for why MJX is not an
option here.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np

from engine.rewards import EpisodeTracker, RewardConfig
from env.course import EVENTS
from env.sim import ACT_DIM, OBS_DIM, OlympicsSim, instance_spec


def _worker(remote, event: str, seed0: int, cfg: RewardConfig, step_cap: int,
            use_mjb: bool = True) -> None:
    if use_mjb:
        # Load the precompiled scene instead of compiling from XML in every worker.
        # Measured per process: +534 MB / 0.34 s compiling, vs +102 MB / 0.10 s loading --
        # 5.2x less resident memory, which is what decides how many workers fit.
        # engine/mjb.selftest() gates this as bit-identical to a fresh compile.
        from engine.mjb import install

        install(event)
    remote_recv, remote_send = remote.recv, remote.send
    tracker = EpisodeTracker(cfg)
    episode = 0

    def fresh():
        nonlocal episode
        # Attempts cycle the four condition strata; the seed advances so the policy sees the
        # whole envelope rather than four memorised operating points.
        params = instance_spec(event, episode % 4, seed0 + episode // 4)
        episode += 1
        sim = OlympicsSim(params)
        obs = sim.reset()
        tracker.reset(sim)
        return sim, obs

    sim, obs = fresh()
    try:
        while True:
            cmd, data = remote_recv()
            if cmd == "step":
                cap = min(sim.max_steps, step_cap) if step_cap else sim.max_steps
                try:
                    result = sim.step(np.asarray(data, np.float64).ravel(), max_steps=cap)
                    reason = result.terminal_reason
                    reward = tracker.step(sim, data, reason)
                    obs = result.obs
                except Exception:            # a malformed action is the learner's bug, not the env's
                    reason, reward = "invalid_action", cfg.r_foul
                info = {"event": event}
                if reason is not None:
                    info.update(reason=reason, score=float(sim.progress), steps=sim.steps,
                                distance_m=float(sim.distance_m))
                    del sim
                    sim, obs = fresh()
                remote_send((obs.astype(np.float32), np.float32(reward),
                             reason is not None, info))
            elif cmd == "close":
                remote_send(None)
                break
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        remote.close()


class VecOlympics:
    """`n_workers` independent attempts stepped in lockstep. Autoresets on termination."""

    def __init__(self, n_workers: int = 12, seed: int = 0,
                 cfg: RewardConfig | None = None, step_cap: int = 0,
                 use_mjb: bool = True) -> None:
        cfg = cfg or RewardConfig()
        # fork where available (Linux): the child inherits the parent image, so it neither
        # re-imports the entry module nor reloads torch. On Windows only spawn exists, which is
        # why every heavy import in engine/ppo.py is deferred into train().
        method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        ctx = mp.get_context(method)
        self.n = n_workers
        self.events = [EVENTS[i % len(EVENTS)] for i in range(n_workers)]
        if use_mjb:
            from engine.mjb import build            # compile once in the PARENT, before forking
            for event in sorted(set(self.events)):
                build(event)
        self._parents, self._procs = [], []
        for i, event in enumerate(self.events):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker,
                               args=(child, event, seed + 7919 * i, cfg, step_cap, use_mjb),
                               daemon=True)
            proc.start()
            child.close()
            self._parents.append(parent)
            self._procs.append(proc)

    def step(self, actions: np.ndarray):
        for parent, a in zip(self._parents, actions):
            parent.send(("step", np.asarray(a, np.float64)))
        obs = np.empty((self.n, OBS_DIM), np.float32)
        rew = np.empty(self.n, np.float32)
        done = np.zeros(self.n, bool)
        infos = []
        for i, parent in enumerate(self._parents):
            o, r, d, info = parent.recv()
            obs[i], rew[i], done[i] = o, r, d
            infos.append(info)
        return obs, rew, done, infos

    def reset(self) -> np.ndarray:
        """Autoreset makes a real reset unnecessary; a zero action just advances one step."""
        obs, _, _, _ = self.step(np.zeros((self.n, ACT_DIM)))
        return obs

    def close(self) -> None:
        for parent in self._parents:
            try:
                parent.send(("close", None))
                parent.recv()
            except (BrokenPipeError, EOFError):
                pass
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()


if __name__ == "__main__":
    import time
    venv = VecOlympics(n_workers=10, step_cap=400)
    obs = venv.reset()
    print(f"events: {venv.events}")
    t0, N = time.monotonic(), 300
    finished = {}
    for _ in range(N):
        obs, rew, done, infos = venv.step(np.zeros((venv.n, ACT_DIM)))
        for info in infos:
            if "reason" in info:
                finished[info["reason"]] = finished.get(info["reason"], 0) + 1
    dt = time.monotonic() - t0
    print(f"{N * venv.n / dt:,.0f} control steps/s across {venv.n} workers "
          f"({N * venv.n:,} steps in {dt:.1f}s)")
    print("terminations:", finished)
    venv.close()
