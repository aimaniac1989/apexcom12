# Training log — what the numbers mean and what moved them

Companion to `engine/README.md`, which covers the architecture. This file covers the *run*: the
scoring geometry the training has to satisfy, what has actually been measured on the box, and the
runbook for driving a multi-day session over ssh.

Status as of 2026-09-02: best meet `raw` is **0.0633** (44M, four-event pool), past the baseline's
0.0330 and against the field leader's 0.6126. The binding constraint is **stability**: the policy
falls at ~12 m and multi-event training makes it worse, not better. See *The 12 m wall*.

## The scoring geometry

Everything below follows from `env/scoring.py` and `env/course.py`, and several points contradict
the obvious guess. Control step is `PHYS_DT * FRAME_SKIP` = 0.02 s.

| event | route | cap | m/s for a completion | what a completion actually needs |
|---|---|---|---|---|
| `sprint_100` | 102 m | 24.0 s | **4.25** | reach 100 m |
| `sprint_400` | 400 m | 72.0 s | **5.56** | reach 400 m, on a curve |
| `hurdles_100` | 102 m | 38.0 s | 2.68 | reach 100 m past ten 0.55-1.15 m barriers |
| `high_jump` | 14 m | 18.0 s | 0.78 | a 0.29 m pelvis rise over the bar |
| `long_jump` | 23 m | 20.0 s | 1.15 | a 6 m void, board 15 -> sand 21 |
| `triple_jump` | 27 m | 28.0 s | 0.96 | hop + step + landing, board 12 -> sand 25 |

**No event completes on stability alone.** The races gate on speed and the jumps gate on jumping.
A policy that stops falling entirely and walks a clean 1.5 m/s still scores `fin 0` on all six.

**`fin` is therefore the wrong instrument**, and reading it as the outcome number cost this project
real time. What moves continuously is route *fraction*, which is also what `instance_score` pays on
for five of the six events. `engine/ppo.py` now prints it (`894fbde`):

```
upd    5  steps 15,710,720   2010/s  rew   194.78  pi +0.0229  v  547.425  kl +0.0276  ...
        eps 200  prog 0.152  dist   15.5 m  fin 0  foul 11  | fell 167  timeout 22  jump_foul 8
```

`prog` and the reason histogram are the run's real vital signs. `fell` shrinking against `timeout`
is a fall fix working, and it shows well before `prog` climbs.

**`bar_missed` is not a hard zero, and the policy found that.** At 44M steps `high_jump`
scored 0.0772 *without being in the curriculum*. Backing it out: `0.24 * clearance/target` gives a
clearance of 0.37 m above the plinth, against a standing pelvis of 0.793 and a bar at 1.00-1.30.
It is **ducking under the bar plane**, taking `bar_missed`, and collecting partial credit on pelvis
height. That is what the scorer pays, and it is underexploited — ducking only as low as each bar
requires is worth **0.1671**.

**`high_jump` is the exception that pays nothing.** It is the only event `instance_score` does not
pay on `progress` — it pays `0.24 * clearance/target`, and `sim._best_clearance` is written in
exactly one place, inside `if prev_x <= bar_x <= x` in `env/sim.py`. So it is 0.0 until the pelvis
has already crossed the bar plane. See note 6 in `engine/rewards.py`.

## The 12 m wall

The reason histogram from `894fbde` settled a question that two months of `fin 0/200` could not.
At 27.4M steps on a five-event pool, over a 200-episode window:

```
  fell            165.7  (83%)
  bar_hit          21.4  (11%)
  bar_missed       11.7  (6%)
  out_of_bounds     1.0  (0%)
  timeout           0.0  (0%)   <- never once
```

By 35.7M it was `fell 181.2` (91%). **The policy is not choosing where to stop, it is collapsing** —
not one episode in 200 ever reaches its step cap. Every earlier reading of the 13-16 m band as a
"stop at the board" strategy was wrong, and so was the reasoning built on it.

Eval is deterministic (`evaluate.py` takes the mean action), and it reaches 17.5 m on `sprint_100`
against 12.1 m for the sampled training rollouts — so the pinned 0.25 action noise costs about 5 m
on top of a ceiling that is already the problem.

**The gait degrades monotonically with pool size:**

| pool | sprint_100 reach |
|---|---|
| `sprint_100` alone | 58.2 m |
| four events | 13.1 m |
| five events | 17.5 m deterministic / 12.1 m sampled |

Interference is real, but it destroys *stability*, not behavioural choice. That is why the
per-event `head1` is not sufficient on its own: the gait lives in `enc1`/`enc2`/the GRU, which the
head fork leaves shared and still being pulled five ways. Hence `--freeze-trunk`.

## Measured history

| steps | raw | sprint_100 | sprint_400 | hurdles | high | long | triple | note |
|---|---|---|---|---|---|---|---|---|
| 11.2M | 0.0519 | — | — | — | — | — | — | pre-`5f51125`, all events dying at 13-16 m |
| 3.0M | 0.03052 | 0.1257 | 0.0066 | 0.0323 | 0.0 | 0.0185 | 0.0 | `sprint_100`-only, post fall fix |
| 10.9M | 0.03040 | 0.1369 | 0.0004 | 0.0323 | 0.0 | 0.0 | 0.0129 | same run, +7.9M steps |
| 44M | 0.06331 | 0.0309 | 0.0081 | 0.0322 | 0.0772 | 0.1321 | 0.0992 | four-event pool |

Backed out as metres of route (`score / rate * route`):

| event | 3.0M | 10.9M | delta |
|---|---|---|---|
| `sprint_100` | 53.4 m | 58.2 m | **+4.8** |
| `sprint_400` | 11.0 m | 0.7 m | **-10.3** |
| `hurdles_100` | 13.7 m | 13.7 m | 0.0 |
| `long_jump` | 2.1 m | 0.0 m | **-2.1** |
| `triple_jump` | 0.0 m | 1.7 m | +1.7 |

Two lessons, both expensive:

1. **A single-event curriculum goes net-negative.** 7.9M steps bought `sprint_100` 4.8 m while
   `sprint_400` lost 10.3 m and `long_jump` lost 2.1 m. The meet mean did not move: 0.03052 ->
   0.03040. Catastrophic forgetting outran the specialisation.
2. **Interference is real, and it is behavioural.** On the four-event pool every event
   converged on the same 13-15 m: `sprint_100` 58.2 -> 13.1 m, `sprint_400` 0.7 -> 13.5 m,
   `long_jump` 0.0 -> 15.2 m, `triple_jump` 1.7 -> 13.4 m. The policy learned one behaviour —
   *walk about fourteen metres and stop* — which is optimal for three of the four (long jump's
   board ends at 15, triple's at 12) and ruinous for the fourth. `sprint_100` collapsed **while it
   was being trained**, which is interference rather than forgetting. It lives in the layers that
   pick a behaviour, not the ones that produce a gait; hence the per-event `head1`.
3. **`_best.pt` selects on the wrong thing during a curriculum.** `best` tracks full-meet `raw`,
   which stays flat while one event improves and five decay, so `<out>_best.pt` can easily hold a
   worse gait than the latest `<out>.pt`. Init from the latest weights during a curriculum run.

## Where the score is

The 58.2 m sprint gait kept *alongside* the jump approaches it was traded for, plus the high jump
duck taken to the height each bar actually allows. Nothing new — just not trading one for another:

| event | best seen | score | at 44M |
|---|---|---|---|
| `sprint_100` | 58.2 m (at 10.9M) | 0.1369 | 0.0309 |
| `sprint_400` | 174.5 m at 2.42 m/s | 0.1047 | 0.0081 |
| `hurdles_100` | 13.7 m (hurdle at 12) | 0.0322 | 0.0322 |
| `high_jump` | duck as upright as the bar allows | 0.1671 | 0.0772 |
| `long_jump` | 17.0 m (board ends at 15) | 0.1478 | 0.1321 |
| `triple_jump` | 14.0 m (board at 12) | 0.1037 | 0.0992 |
| **meet raw** | | **0.1154** | **0.0633** |

Note where the value sits: the two horizontal jumps pay most, not because of jumping but because
their routes are 23 m and 27 m, so reaching the board is most of the fraction.

**But partial credit has a ceiling near 0.13.** Every number above comes from never completing
anything. `riv_0830` scores 0.6126 because it *finishes* — its `long_jump` is 0.941, which means it
lands. The first real completion is the unlock, and `high_jump`'s 0.29 m pelvis rise is the
smallest one on the board.

`hurdles_100` stays out of the curriculum: it needs a 0.55 m barrier cleared at 12 m and the
observation barely sees it (`engine/README.md`). `high_jump` goes back in — the duck is worth
optimising deliberately, and `w_apex` has still never run.

## Changes and why

| commit | change |
|---|---|
| `5f51125` | `r_fell` added; progress paid as route fraction not metres; `r_foul` -40 -> -150 |
| `f684e44` | `w_apex` gives high_jump a pre-crossing gradient; `--events` curriculum subset |
| `894fbde` | log route progress + terminal-reason histogram; fix the resumed-run throughput figure |
| `61f9139` | per-event `head1`, sliced by the latched one-hot; `expand_head1` checkpoint migration |
| (this) | `--freeze-trunk`; `w_apex` to 0 |

**`w_apex` is off, measured.** Over ~6M `high_jump` steps it never produced one clearance, and the
rise it buys near the bar converts clean ducks into strikes — `bar_hit` records no clearance and so
scores 0.000, where a duck taking `bar_missed` scores 0.077. `high_jump` fell 0.0772 -> 0.0338 on
entering the pool, and by 35.7M `bar_missed` had collapsed 11.7 -> 2.6 per 200 as the extra height
destabilised the approach. Restore it when the gait survives past 58 m.

**A reward change invalidates the stored critic.** `--resume` restores the critic *and* Adam's
moments, which were fit to the old return distribution. Resuming across `5f51125` showed up as
value loss swinging 456 <-> 976 between updates while the advantages — and therefore the policy
updates — were garbage. After changing `engine/rewards.py`, restart with
`--init <weights> --resume 0`, never by resuming the state file.

**`--anchor 0` is mandatory with `--resume 0`.** The anchor net copies from whatever `--init`
loaded, and `--resume 0` puts `total_steps` back to 0, so `anchor_w` springs back to `--anchor`
and the run spends `--anchor-steps` BC-fitting the policy to its own current gait.

## Runbook

Always launch inside tmux — the trainer is then a child of the tmux server, not of the ssh
session, and survives a dropped connection.

### First start (new curriculum from a checkpoint)

```bash
tmux new -s meet
cd ~/apollo/apexcom12 && source .venv/bin/activate   # adjust path if different

PYTHONPATH=$(pwd) python engine/ppo.py \
    --events sprint_100,sprint_400,long_jump,triple_jump \
    --init engine/runs/sprint.pt --resume 0 --anchor 0 \
    --workers $(( $(nproc) - 2 )) --horizon 128 --minibatch 2048 \
    --eval-every 25 --eval-seeds 2 --save-every 10 --updates 20000 \
    --out engine/runs/meet.pt 2>&1 | tee engine/runs/meet.log
```

`ctrl-b` `d` detaches. Reattach with `tmux attach -t meet`, or in one line from a fresh login:
`ssh vps -t tmux attach -t meet`.

### Continue training (same run after crash / disconnect)

1. Check whether it is already running:

```bash
pgrep -af engine/ppo.py
tmux ls
tmux attach -t meet    # if the session still exists, you are done
```

2. If the process died but checkpoints remain, resume from `<out>_state.pt`. Use the **same**
   `--out` path, **omit** `--resume 0` (default `--resume 1`), and append the log with `tee -a`:

```bash
cd ~/apollo/apexcom12 && source .venv/bin/activate
tmux new -s meet

PYTHONPATH=$(pwd) python engine/ppo.py \
    --events sprint_100,sprint_400,long_jump,triple_jump \
    --init engine/runs/meet.pt \
    --anchor 0 \
    --workers $(( $(nproc) - 2 )) --horizon 128 --minibatch 2048 \
    --eval-every 25 --eval-seeds 2 --save-every 10 --updates 20000 \
    --out engine/runs/meet.pt 2>&1 | tee -a engine/runs/meet.log
```

You should see `resumed engine/runs/meet_state.pt at N steps (best ...)`. That restores policy,
critic, Adam moments, `log_std`, and step count. At `--save-every 10` you lose at most ten updates.

| file | role |
|---|---|
| `<out>_state.pt` | full resume (preferred for continue) |
| `<out>.pt` | latest policy weights |
| `<out>_best.pt` | best full-meet EVAL raw — prefer as `--init` only when starting a **new** run |

Match `--out` / `--init` to the run you actually have (`meet.pt`, `ppo.pt`, `sprint.pt`, …):

```bash
ls -lh engine/runs/*.pt
grep EVAL engine/runs/*.log | tail
```

### When to use `--resume 0` instead

Only after changing `engine/rewards.py`, or when starting a **new** curriculum from a weight file
(not continuing the same optimiser state). Always pair with `--anchor 0`. Do **not** load the old
`*_state.pt` in that case.

### Monitor without attaching

tmux scrollback caps at ~2000 lines; the log file does not:

```bash
tail -f engine/runs/meet.log
grep EVAL engine/runs/meet.log | tail
pgrep -af engine/ppo.py
```

Client-side keepalives stop most idle disconnects. In `~/.ssh/config`:

```
Host vps
    HostName <ip>
    Port <port>
    User root
    ServerAliveInterval 30
    ServerAliveCountMax 6
    TCPKeepAlive yes
```

## Open items

- **`std` has not moved in 449 updates.** It sits at exactly `--init-std` 0.25. Not a wiring bug —
  `log_std` is in the gradient graph — but a stalemate between the `--ent 1e-3` bonus pushing it up
  and the policy gradient pushing it down, with Adam normalising both. The policy therefore injects
  0.25 rad of noise into every joint target forever and never sharpens, which caps gait speed and
  precision. Lowering `--ent` is the lever; do it when a run plateaus, not while it is being asked
  to generalise to unseen routes.
- **`out_of_bounds` fouls returned at 5-9 per 200** once the gait reached ~58 m. On a `sprint_100`
  pool the only reachable foul reasons are `out_of_bounds` and `physics_glitch`, so the faster gait
  is drifting out of the 1.5 m lane. `w_lateral` 0.25 against `w_progress` 1200 is the ratio to
  revisit if it worsens.
- **`kl` reached 0.057 against `--target-kl` 0.03** late in the sprint run, with value loss back up
  near 95. The epoch loop early-stops the moment KL crosses target, so updates are being cut short.
  If it persists across an EVAL rather than being one noisy update, lower `--lr` or raise
  `--minibatch`.
- **`w_apex` has never been exercised by a real run.** Its gating and delta arithmetic are unit
  tested, but `high_jump` has been outside every curriculum since it landed.
- **Neither `hurdles_100` nor `high_jump` has a path to non-trivial score yet.** Both are blocked on
  capability the policy does not have, and together they are a third of the meet mean.
