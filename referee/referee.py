"""Referee for a balanced, six-discipline Humanoid Olympics meet."""

from __future__ import annotations

import json
import math
import pathlib
import sys
import time
from dataclasses import asdict

from gym_v1 import GameResult, Referee, RefereeContext
from gym_v1.client import PlayerClient, PlayerError
from gym_v1.referee import RESULT_PATH

from env import (ACT_DIM, EVENT_LABELS, EVENTS, OBS_DIM, STATE_DIM, OlympicsSim, event_instances,
                 instance_score, meet_score, nominal_mu)
from env.history import DEFAULT_STRIDE, InstanceRecorder, write_instance
from env.sim import WIND_MAX_MS, InvalidAction

DEFAULT_INSTANCES_PER_EVENT = 4
DEFAULT_DEADLINE_MS = 500
EVALUATION_BUDGET_S = 840.0
HISTORY_DIR = pathlib.Path("/data/history")
PLAYER_FAULTS = (PlayerError, OSError, json.JSONDecodeError)
MAX_ACTION_LEN = 1024


class OlympicsReferee(Referee):
    def play_game(self, ctx: RefereeContext, players: list[PlayerClient]) -> GameResult:
        started = time.monotonic()
        cfg = ctx.config or {}
        seed = int(cfg.get("seed", ctx.seed))
        test_step_cap = int(cfg.get("test_step_cap", 0))
        # The release configuration is part of the score definition. Test-only
        # callers may reduce it under an explicit control-step cap, but a live
        # round always evaluates the same 24 recorded attempts under the same
        # stratification; only the conditions move, and they move with `seed`.
        if test_step_cap > 0:
            instances_per_event = int(cfg.get("instances_per_event", DEFAULT_INSTANCES_PER_EVENT))
            deadline_ms = int(cfg.get("deadline_ms", DEFAULT_DEADLINE_MS))
            wind_max = float(cfg.get("wind_max_ms", WIND_MAX_MS))
            record_history = bool(cfg.get("record_history", True))
            history_stride = int(cfg.get("history_stride", DEFAULT_STRIDE))
        else:
            instances_per_event = DEFAULT_INSTANCES_PER_EVENT
            deadline_ms = DEFAULT_DEADLINE_MS
            wind_max = WIND_MAX_MS
            record_history = True
            history_stride = DEFAULT_STRIDE
        player = players[0]
        tasks = event_instances(instances_per_event, seed, wind_max)

        rows: list[dict] = []
        history_written = 0
        for task_index, params in enumerate(tasks):
            if time.monotonic() - started >= EVALUATION_BUDGET_S:
                rows.append(self._unrun_row(task_index, params, "round_timeout"))
                continue

            sim = OlympicsSim(params)
            max_steps = min(sim.max_steps, test_step_cap) if test_step_cap > 0 else sim.max_steps
            obs = sim.reset()
            rec = InstanceRecorder(task_index, sim, history_stride) if record_history else None
            reason: str | None = None
            try:
                # A malicious reset must not consume the persistence allowance
                # after the 840 s scoring window.  Keep a tiny transport grace
                # so urllib can surface a typed PlayerError rather than a raw
                # socket error at the deadline.
                reset_timeout = min(30.0, max(0.1, EVALUATION_BUDGET_S -
                                               (time.monotonic() - started)))
                player.reset(match_id=f"{ctx.match_id}:{task_index}", player_index=0, seed=0,
                             config={}, timeout_s=reset_timeout)
            except PLAYER_FAULTS:
                reason = "player_error"

            while reason is None:
                if time.monotonic() - started >= EVALUATION_BUDGET_S:
                    reason = "round_timeout"
                    break
                try:
                    action_timeout = min(deadline_ms / 1000.0 + 1.0,
                                         max(0.1, EVALUATION_BUDGET_S -
                                             (time.monotonic() - started)))
                    action = player.act(observation=obs.tolist(), deadline_ms=deadline_ms,
                                        timeout_s=action_timeout)
                except PLAYER_FAULTS:
                    reason = "player_error"
                    break
                if isinstance(action, (list, tuple)) and len(action) > MAX_ACTION_LEN:
                    reason = "invalid_action"
                    break
                try:
                    result = sim.step(action, max_steps=max_steps)
                except (InvalidAction, TypeError, ValueError):
                    reason = "invalid_action"
                    break
                obs, reason = result.obs, result.terminal_reason
                if rec is not None:
                    rec.capture(sim, action)

            metrics = sim.metrics
            score = instance_score(sim.event, reason, sim.progress, sim.steps, max_steps, metrics)
            row = {
                "task": task_index,
                "event": sim.event,
                "event_label": EVENT_LABELS[sim.event],
                "attempt": params.attempt,
                "friction_level": round(params.friction_level, 4),
                "friction_mu": round(nominal_mu(params.friction_level), 4),
                "wind_speed_ms": round(params.wind_speed, 2),
                "wind_dir_deg": round(math.degrees(params.wind_dir), 1),
                "challenge": {k: round(float(v), 3) for k, v in params.challenge.items()},
                "terminal_reason": reason,
                "progress": round(sim.progress, 4),
                "distance_m": round(sim.distance_m, 2),
                "steps": sim.steps,
                "max_steps": max_steps,
                "sim_time_s": round(sim.steps * 0.02, 2),
                "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
                "score": round(score, 6),
            }
            rows.append(row)
            if rec is not None:
                try:
                    write_instance(HISTORY_DIR, rec.record(sim, row, match_id=ctx.match_id,
                                                           num_instances=len(tasks)))
                    history_written += 1
                except Exception as error:  # artifact production never changes a score
                    print(f"history write failed for task {task_index}: {type(error).__name__}: {error}",
                          file=sys.stderr, flush=True)
            # The next task may compile a different G1 scene. Drop this attempt's
            # MjData before doing so; only the current event model stays resident.
            rec = None
            del sim

        raw = meet_score(rows, EVENTS)
        event_scores = {
            event: round(sum(row["score"] for row in rows if row["event"] == event) /
                         max(1, sum(row["event"] == event for row in rows)), 6)
            for event in EVENTS
        }
        finishes = sum(row["terminal_reason"] in {"completed", "cleared", "landed"} for row in rows)
        return GameResult(
            raw_scores=[raw], winner=0 if raw > 0 else -1, terminal_reason="scored",
            steps=sum(int(row["steps"]) for row in rows),
            metadata={
                "tasks": rows,
                "instances_per_event": instances_per_event,
                "num_instances": len(rows),
                "event_scores": event_scores,
                "num_finished": finishes,
                "history_files": history_written,
                "wind_max_ms": wind_max,
                "history_stride": history_stride,
                "eval_time_in_seconds": round(time.monotonic() - started, 1),
            },
        )

    @staticmethod
    def _unrun_row(task: int, params, reason: str) -> dict:
        return {
            "task": task, "event": params.event, "event_label": EVENT_LABELS[params.event],
            "attempt": params.attempt, "friction_level": round(params.friction_level, 4),
            "friction_mu": round(nominal_mu(params.friction_level), 4),
            "wind_speed_ms": round(params.wind_speed, 2),
            "wind_dir_deg": round(math.degrees(params.wind_dir), 1),
            "challenge": {k: round(float(v), 3) for k, v in params.challenge.items()},
            "terminal_reason": reason, "progress": 0.0, "distance_m": 0.0,
            "steps": 0, "max_steps": 0, "sim_time_s": 0.0, "metrics": {}, "score": 0.0,
        }

    def run(self) -> None:
        """Attribute a never-ready player to its submission, not the referee."""
        ctx = RefereeContext.from_env()
        players = [PlayerClient(url) for url in ctx.player_urls]
        try:
            for player in players:
                player.wait_until_ready(self.readiness_timeout_s)
        except PlayerError as error:
            result = GameResult(
                raw_scores=[0.0], winner=-1, terminal_reason="submission_not_ready", steps=0,
                metadata={
                    "error": str(error),
                    "explanation": ("The ONNX submission never became ready. It must expose inputs obs "
                                    f"[batch, {OBS_DIM}] and state_in [batch, {STATE_DIM}], outputs "
                                    f"action [batch, {ACT_DIM}] and state_out [batch, {STATE_DIM}], all float32."),
                },
            )
        else:
            result = self.play_game(ctx, players)
        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULT_PATH.write_text(json.dumps(asdict(result)))


if __name__ == "__main__":
    OlympicsReferee().run()
