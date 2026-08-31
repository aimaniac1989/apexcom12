# engine — our Humanoid Olympics submission

Our own policy and training stack, built from scratch against the deployed v0.4.0 repo. Nothing
here derives from another miner's artifact; the only inherited weights are the organisers' own
published `baseline/baseline.onnx` (Unitree `motion.pt`, BSD-3), used as a gait prior.

| file | role |
|---|---|
| `contract.py` | the evaluation contract as executable assertions — run it before every submit |
| `model.py` | the policy: event latch, course-position geometry channel, GRU + EMA observers |
| `distill.py` | phase 1 — DAgger warm-start from the baseline walker |
| `venv.py` | multiprocess vectorised env (one event per worker, precompiled scenes) |
| `rewards.py` | dense shaping derived from the scoring rules |
| `ppo.py` | phase 2 — PPO with eval hooks, best-checkpoint tracking, resume |
| `evaluate.py` | scores a checkpoint with the referee's OWN scorer |
| `mjb.py` | precompiled scene cache + bit-identical determinism gate |
| `export.py` | ONNX export + parity + latch + contract checks |
| `setup_vps.sh` | provision and verify a Linux training box |

## The three gates that can zero a round

Run `python engine/contract.py engine/submission.onnx` before every submission.

1. **Screener** (pre-upload): ≤ 15 MB, ≥ 10,000 initializer bytes.
2. **Player load**: exactly 2 in / 2 out, positional order, float32, trailing dims 104/256/12/256.
   A failure here writes `submission_not_ready`, raw 0.0, **for the whole meet** — not one attempt.
3. **Runtime**: 12 finite floats per action, 500 ms deadline, and `EVALUATION_BUDGET_S = 840` for
   all 24 attempts. Events are scheduled in a fixed order, so overrunning the budget zeroes the
   **last three — all the jumps** — while they stay in the denominator.

Current artifact: 1.85 MB, 458k params, 0.11 ms/step → 4.4 s of the 840 s budget.

## Measured facts this design is built on

Each of these was measured in this repo, and several contradict the obvious guess:

- **Hurdles are nearly invisible.** They are `walkable=False`, so they never enter the height
  scan, and the upward rays start at `pelvis+0.05` ≈ 1.64 m — above the first six hurdles
  entirely. The 0.55 m hurdle is detected at **0 of 106** approach positions; the 1.15 m one at
  35 of 106 (rays are 0.667 m apart, a hurdle is 0.24 m wide). No reactive policy can clear them.
  Hence the geometry channel: latch the event, recover `px = finish_e − 10·obs[50]`, and feed
  signed range to all 17 public obstacle coordinates.
- **The event is identifiable only on step 1.** `obs[50]` is remaining-route/10 and decays, so
  the classification is latched into state. `sprint_100` and `hurdles_100` both reset at 102.0 m
  and are genuinely indistinguishable; a sticky "overhead seen" bit is the only later evidence.
- **The `fell` gate has no airborne exemption** (`xmat[2,2] < 0.40`). A jump is killed mid-flight
  if the torso passes ~66°, with no arms to counter-rotate. In-flight uprightness is therefore a
  **continuous** reward term — a terminal-only version gives a searcher zero gradient, measured.
- **Fouls are hard zeros** while a timeout keeps `0.24 × progress`. Committing to a jump must
  beat balking at the board, or "stop at the takeoff line" (~0.10) becomes a local optimum.
- **`.mjb` vs XML per worker: +102 MB / 0.10 s vs +534 MB / 0.34 s.** This is what decides worker
  count. `mjb.selftest()` gates it as bit-identical — training on physics that differs from the
  referee's is the worst failure available.

## Platform traps, both hit in practice

- **Windows `spawn` re-imports the entry module in every worker.** `import torch` at module level
  in `ppo.py` killed 10 workers with a bare `EOFError`. All heavy imports are deferred into
  `train()`, and `venv.py`/`rewards.py` are kept torch-free. On Linux `fork` avoids this entirely.
- **Recurrent PPO drifts.** Re-deriving states through a BPTT chunk under an already-updated
  policy sent KL 0.18 → 2.7 → 13.8 in three updates. `--replay-state 1` (default) feeds stored
  states, which both stabilises the ratio and makes timesteps i.i.d. — enabling the flat batched
  update that took throughput 710 → ~2,000 steps/s.

## Status

Infrastructure verified end to end. **The policy is not trained.** The warm start scores 0.0231
against the baseline's 0.0330 and the field leader's 0.6126 — BC does not beat its teacher, and
its value is as a PPO initialisation (≈9× better than random's 0.0026), not as a submission.
Early PPO eval moved 0.0273 → 0.0065 over 9k steps, which is either normal early thrash or the
known gait fragility; it needs a real run to distinguish, and that is what the eval hook and the
best-checkpoint tracking are for.
