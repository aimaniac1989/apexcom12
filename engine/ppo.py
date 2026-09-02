"""Phase 2: recurrent PPO, warm-started from the distilled gait.

Design notes that are specific to this problem rather than boilerplate:

* **Truncated BPTT over stored states.** The policy is recurrent and its 256-float state carries
  the latched event, the disturbance EMAs and the GRU hidden. Rollouts store the FULL state at
  every step; each optimisation chunk restarts from the stored (detached) state and re-runs
  forward with grad. Re-running is exact because the forward is deterministic, so the gradient
  sees the same trajectory the data came from.

* **A BC anchor to the frozen warm-start.** Policy-gradient updates scale as 1/sigma^2, and a
  locomotion gait is fragile: a few bad updates destroy it and PPO will not rediscover it. The
  anchor is an L2 pull toward the distilled policy's mean, annealed to zero over `anchor_steps`.
  Without it, warm-starting buys nothing -- the first hundred updates simply undo it.

* **The critic is NOT exported.** The submission graph must have exactly 2 outputs, so the value
  head lives here and is dropped at export time. It sees the same features as the actor; there is
  no privileged information to give it (friction and wind are genuinely unobservable, and a
  privileged critic measured EV 0.35 on a related task -- most of the variance here is chaotic
  contact, not something a better critic recovers).

* **Action noise stays small.** The exported head is `10*tanh(a/10)`, already smooth; the
  exploration std is initialised at ~0.25 and floored, because a humanoid gait tolerates very
  little action noise before it stops walking.

    PYTHONPATH=. python engine/ppo.py --init engine/runs/warm.pt --updates 400
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import time

import numpy as np

# NOTE: torch is imported INSIDE train(), never at module level, and neither are the modules
# that pull it in (engine.model). Windows uses spawn, so every one of the N env workers
# re-imports this file as `__mp_main__`. With `import torch` up here, all N children load torch
# and its DLLs on startup: at 4 workers it merely wastes seconds, and at 10 the children die
# during import and the parent sees only `EOFError` from a closed pipe. `engine.venv` and
# `engine.rewards` are deliberately torch-free so a worker's import stays cheap.


def _build(IN_DIM, H_DIM):
    """Define the critic lazily, so importing this module does not import torch."""
    import torch
    import torch.nn as nn

    class Critic(nn.Module):
        """Value head over the actor's own assembled features plus its recurrent latent."""

        def __init__(self, hidden=(256, 128)) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(IN_DIM + H_DIM, hidden[0]), nn.ELU(),
                nn.Linear(hidden[0], hidden[1]), nn.ELU(),
                nn.Linear(hidden[1], 1))

        def forward(self, feats: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            return self.net(torch.cat([feats, h], dim=1)).squeeze(-1)

    def gae(rew, val, done, last_val, gamma=0.99, lam=0.95):
        T, N = rew.shape
        adv = torch.zeros(T, N)
        run = torch.zeros(N)
        nxt = last_val
        for t in reversed(range(T)):
            mask = 1.0 - done[t]
            delta = rew[t] + gamma * nxt * mask - val[t]
            run = delta + gamma * lam * mask * run
            adv[t] = run
            nxt = val[t]
        return adv, adv + val

    return Critic, gae


def _report(update, total_steps, steps_at_start, t_start, b_rew, stats, anchor_w, log_std,
            recent, args, out, policy) -> None:
    """One line of optimiser health, one of what the robot actually did.

    `fin` is a lagging indicator and will read 0 for a long time: no event completes on
    stability alone. sprint_100 needs a sustained 4.25 m/s over its 102 m in the 24 s cap,
    sprint_400 5.56 m/s, and the jumps need a jump. What moves first is route FRACTION -- which
    is also what `instance_score` pays on for five of the six events -- so `prog` is the number
    to watch, and the reason histogram says what is ending the episodes.
    """
    import torch

    k = max(1, stats["n"])
    # Steps SINCE THIS PROCESS STARTED. `total_steps` is restored by --resume, and dividing the
    # cumulative count by this run's elapsed time reported ~3.6M/s on the first update after a
    # resume, decaying hyperbolically toward the truth.
    rate = (total_steps - steps_at_start) / max(1e-9, time.monotonic() - t_start)
    finished = sum(1 for i in recent if i.get("reason") in {"completed", "cleared", "landed"})
    fouls = sum(1 for i in recent if i.get("reason") in
                {"jump_foul", "high_foul", "out_of_bounds", "physics_glitch"})
    print(f"upd {update:4d}  steps {total_steps:9,}  {rate:6.0f}/s  "
          f"rew {b_rew.sum(0).mean():8.2f}  pi {stats['pi']/k:+.4f}  v {stats['v']/k:8.3f}  "
          f"kl {stats['kl']/k:+.4f}  bc_w {anchor_w:.3f}  std {log_std.exp().mean():.3f}",
          flush=True)
    if recent:
        n = len(recent)
        # `score` is sim.progress, the clipped route fraction instance_score consumes.
        prog = sum(float(i.get("score", 0.0)) for i in recent) / n
        dist = sum(float(i.get("distance_m", 0.0)) for i in recent) / n
        reasons = collections.Counter(i.get("reason", "?") for i in recent)
        hist = "  ".join(f"{r} {c}" for r, c in reasons.most_common())
        print(f"        eps {n:3d}  prog {prog:.3f}  dist {dist:6.1f} m  "
              f"fin {finished}  foul {fouls}  | {hist}", flush=True)
        # Per event, because the aggregate above is close to meaningless: it means a 400 m route
        # with a 14 m one, so `dist` cannot distinguish "one event reaches 58 m and four fall at
        # 3 m" from "all five fall at 12 m" -- and those want opposite responses.
        by = {}
        for i in recent:
            by.setdefault(i.get("event", "?"), []).append(i)
        cells = []
        for ev in sorted(by):
            rows = by[ev]
            pe = sum(float(r.get("score", 0.0)) for r in rows) / len(rows)
            de = sum(float(r.get("distance_m", 0.0)) for r in rows) / len(rows)
            fe = sum(1 for r in rows if r.get("reason") == "fell")
            cells.append(f"{ev} {pe:.2f} {de:4.1f}m fell {fe}/{len(rows)}")
        print("        " + "  ".join(cells), flush=True)


def _checkpoint(update, args, out, policy, critic, opt, log_std, total_steps, best) -> None:
    """Write the weights and, separately, everything needed to resume mid-run."""
    import torch

    if (update + 1) % args.save_every:
        return
    torch.save(policy.state_dict(), out)
    torch.save({"policy": policy.state_dict(), "critic": critic.state_dict(),
                "opt": opt.state_dict(), "log_std": log_std.detach().clone(),
                "total_steps": total_steps, "best": best},
               out.with_name(out.stem + "_state.pt"))


def _maybe_eval(update, args, policy, out, best):
    """Score against the REAL referee scorer periodically and keep the best checkpoint.

    Shaped reward is not the objective; `raw_score` is. Reward climbing while raw_score stays
    flat is the characteristic way reward shaping goes wrong, and on a multi-day run you want to
    see that on day one rather than at the end. The best-so-far checkpoint is kept separately
    because PPO can and does regress.
    """
    import json

    import torch

    if not args.eval_every or (update + 1) % args.eval_every:
        return best
    from engine.evaluate import evaluate_policy

    was_training = policy.training
    result = evaluate_policy(policy, seeds=tuple(range(1, args.eval_seeds + 1)))
    policy.train(was_training)
    raw = result["raw_score"]
    tag = ""
    if raw > best:
        best = raw
        torch.save(policy.state_dict(), out.with_name(out.stem + "_best.pt"))
        tag = "  <- best, saved"
    print(f"   EVAL raw {raw:.5f} (sd {result['raw_sd']:.5f})  finished "
          f"{result['num_finished']}/{result['n']}  "
          f"{ {k: round(v, 4) for k, v in result['event_scores'].items()} }{tag}", flush=True)
    with out.with_suffix(".eval.jsonl").open("a") as fh:
        fh.write(json.dumps({"update": update, "raw": raw, **result["event_scores"]}) + "\n")
    return best


def train(args) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from engine.model import (ACT_DIM, H_DIM, H_HI, H_LO, IN_DIM, OBS_DIM, STATE_DIM,
                              OlympicsPolicy, expand_head1)
    from engine.rewards import RewardConfig
    from engine.venv import VecOlympics

    Critic, gae = _build(IN_DIM, H_DIM)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(2)          # the workers own the cores; leave the trainer two

    policy = OlympicsPolicy()
    if args.init:
        policy.load_state_dict(expand_head1(torch.load(args.init, map_location="cpu")))
        print(f"warm-started from {args.init}")
    anchor = OlympicsPolicy()
    anchor.load_state_dict(policy.state_dict())
    anchor.eval()
    for p in anchor.parameters():
        p.requires_grad_(False)

    # Freeze the gait. Measured: sprint_100 reached 58.2 m on a single-event pool, 13.1 m on
    # four events and 12.1 m on five, and at 35.7M steps 91% of episodes end in `fell` with
    # `timeout` never once appearing. Multi-event training is not making the policy choose a
    # different behaviour, it is destroying the gait -- and the gait lives in enc1/enc2/gru, which
    # the per-event head1 leaves shared. With no gradient reaching them the trunk cannot degrade,
    # and each event still adapts through its own head. A frozen trunk cannot learn to JUMP, so
    # this is deliberately a harvest-the-partial-credit run.
    if args.freeze_trunk:
        for module in (policy.enc1, policy.enc2, policy.gru):
            for p in module.parameters():
                p.requires_grad_(False)
        print("trunk frozen: enc1/enc2/gru held, only the per-event heads train")

    critic = Critic()
    log_std = nn.Parameter(torch.full((ACT_DIM,), float(np.log(args.init_std))))
    opt = torch.optim.Adam(
        [{"params": [p for p in policy.parameters() if p.requires_grad], "lr": args.lr},
         {"params": critic.parameters(), "lr": args.lr * 3},
         {"params": [log_std], "lr": args.lr}], eps=1e-5)

    events = tuple(e.strip() for e in args.events.split(",") if e.strip()) or None
    venv = VecOlympics(n_workers=args.workers, seed=args.seed,
                       cfg=RewardConfig(), step_cap=args.step_cap, events=events)
    if events:
        # The eval hook still scores the FULL meet, so raw_score stays comparable across runs.
        print(f"curriculum: training {sorted(set(venv.events))} only")
    N, T = venv.n, args.horizon
    obs_np = venv.reset()
    state = torch.zeros(N, STATE_DIM)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    episode_returns, recent = {}, []
    t_start = time.monotonic()
    total_steps = 0
    best = -1.0
    # Resume: a multi-day run will be interrupted, and losing a day to a dropped ssh session is
    # avoidable. Optimiser and exploration std are restored too -- reloading only the weights
    # restarts Adam's moments and the entropy schedule, which shows up as a visible regression.
    resume_path = out.with_name(out.stem + "_state.pt")
    if args.resume and resume_path.exists():
        ck = torch.load(resume_path, map_location="cpu")
        policy.load_state_dict(expand_head1(ck["policy"]))
        critic.load_state_dict(ck["critic"])
        opt.load_state_dict(ck["opt"])
        with torch.no_grad():
            log_std.copy_(ck["log_std"])
        total_steps, best = ck.get("total_steps", 0), ck.get("best", -1.0)
        print(f"resumed {resume_path} at {total_steps:,} steps (best {best:.5f})")
    # After the resume, so throughput and the final wall-clock measure THIS run, not a total
    # that a restored step count would inflate.
    steps_at_start = total_steps
    t_start = time.monotonic()

    for update in range(args.updates):
        b_obs = torch.zeros(T, N, OBS_DIM)
        b_state = torch.zeros(T, N, STATE_DIM)
        b_act = torch.zeros(T, N, ACT_DIM)
        b_logp = torch.zeros(T, N)
        b_val = torch.zeros(T, N)
        b_rew = torch.zeros(T, N)
        b_done = torch.zeros(T, N)

        # -- collect --------------------------------------------------------------------------
        with torch.no_grad():
            for t in range(T):
                obs = torch.from_numpy(obs_np)
                b_obs[t], b_state[t] = obs, state
                mean, next_state = policy(obs, state)
                feats = policy.assemble(obs, state)
                b_val[t] = critic(feats, next_state[:, H_LO:H_HI])
                std = log_std.exp().clamp_min(args.min_std)
                action = mean + std * torch.randn_like(mean)
                b_logp[t] = (-0.5 * (((action - mean) / std) ** 2)
                             - log_std - 0.5 * float(np.log(2 * np.pi))).sum(-1)
                b_act[t] = action

                obs_np, rew, done, infos = venv.step(action.numpy())
                b_rew[t] = torch.from_numpy(rew)
                b_done[t] = torch.from_numpy(done.astype(np.float32))
                # State is zeroed on reset by the player, so mirror that exactly here.
                state = next_state * (1.0 - torch.from_numpy(done.astype(np.float32))).unsqueeze(1)
                for i, info in enumerate(infos):
                    if "reason" in info:
                        episode_returns.setdefault(info["event"], []).append(info)
                        recent.append(info)
            recent[:] = recent[-200:]
            last_mean, last_state = policy(torch.from_numpy(obs_np), state)
            last_val = critic(policy.assemble(torch.from_numpy(obs_np), state),
                              last_state[:, H_LO:H_HI])

        total_steps += T * N
        adv, ret = gae(b_rew, b_val, b_done, last_val, args.gamma, args.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # -- optimise ---------------------------------------------------------------------------
        anchor_w = args.anchor * max(0.0, 1.0 - total_steps / max(1, args.anchor_steps))
        stats = {"pi": 0.0, "v": 0.0, "ent": 0.0, "bc": 0.0, "kl": 0.0, "n": 0}

        if args.replay_state:
            # FLAT PATH. With stored states replayed, every timestep is independent given its
            # state, so the whole rollout is one [T*N, ...] batch of i.i.d. samples and PPO
            # reduces to its standard feed-forward form. That replaces T sequential passes of
            # width N with a handful of wide minibatches -- measured 0.41 ms/sample at batch 1
            # against 0.012 ms/sample at batch 512, so this is where the wall-clock is.
            # It is also what finally gives a GPU something worth doing.
            flat = lambda x: x.reshape(T * N, *x.shape[2:])          # noqa: E731
            f_obs, f_state = flat(b_obs), flat(b_state)
            f_act, f_logp = flat(b_act), flat(b_logp)
            f_adv, f_ret = flat(adv), flat(ret)
            idx_all = torch.randperm(T * N)
            mb = args.minibatch or (T * N)
            stop = False
            for _ in range(args.epochs):
                if stop:
                    break
                for s0 in range(0, T * N, mb):
                    idx = idx_all[s0:s0 + mb]
                    obs_b, st_b = f_obs[idx], f_state[idx]
                    mean, st_next = policy(obs_b, st_b)
                    value = critic(policy.assemble(obs_b, st_b), st_next[:, H_LO:H_HI])
                    std = log_std.exp().clamp_min(args.min_std)
                    logp = (-0.5 * (((f_act[idx] - mean) / std) ** 2)
                            - log_std - 0.5 * float(np.log(2 * np.pi))).sum(-1)
                    ratio = (logp - f_logp[idx]).exp()
                    a_b = f_adv[idx]
                    pi_loss = -torch.min(
                        ratio * a_b,
                        ratio.clamp(1 - args.clip, 1 + args.clip) * a_b).mean()
                    v_loss = F.mse_loss(value, f_ret[idx])
                    ent = (log_std + 0.5 * float(np.log(2 * np.pi * np.e))).sum()
                    bc_loss = torch.zeros((), device=mean.device)
                    if anchor_w > 0:
                        with torch.no_grad():
                            a_mean, _ = anchor(obs_b, st_b)
                        bc_loss = F.mse_loss(mean, a_mean)
                    loss = pi_loss + args.vf * v_loss - args.ent * ent + anchor_w * bc_loss
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(policy.parameters()) + list(critic.parameters()) + [log_std], 1.0)
                    opt.step()
                    with torch.no_grad():
                        approx_kl = float(((ratio - 1) - (logp - f_logp[idx])).mean())
                    stats["pi"] += float(pi_loss.detach())
                    stats["v"] += float(v_loss.detach())
                    stats["bc"] += float(bc_loss.detach())
                    stats["kl"] += approx_kl
                    stats["n"] += 1
                    if approx_kl > args.target_kl:
                        stop = True
                        break
            _report(update, total_steps, steps_at_start, t_start, b_rew, stats, anchor_w,
                    log_std, recent, args, out, policy)
            best = _maybe_eval(update, args, policy, out, best)
            _checkpoint(update, args, out, policy, critic, opt, log_std, total_steps, best)
            continue
        # Early-stop on KL. This matters more here than in feed-forward PPO: each chunk restarts
        # from the STORED state but then re-derives the next 31 states under a policy that has
        # already taken gradient steps this update. That drift is the standard truncated-BPTT
        # approximation, but it compounds -- measured KL running 0.18 -> 2.7 -> 13.8 over three
        # updates without this guard, which is a destroyed policy, not a noisy one.
        stop = False
        for _ in range(args.epochs):
            if stop:
                break
            for c0 in range(0, T, args.bptt):
                c1 = min(c0 + args.bptt, T)
                st = b_state[c0].clone()          # exact: forward is deterministic
                pi_loss = v_loss = bc_loss = ent_sum = kl_sum = 0.0
                for t in range(c0, c1):
                    obs = b_obs[t]
                    # `replay_state` feeds the STORED state at every t instead of the one this
                    # chunk re-derived. Re-deriving is what BPTT needs, but the states drift once
                    # the policy has taken a step this update, and the drift compounds: measured
                    # KL 1.75 -> 6.79 -> 8.07 over three updates at bptt=32, which is a destroyed
                    # gait rather than a noisy one. Replaying trades gradient-through-time for a
                    # ratio that means what PPO assumes it means. Off = true BPTT, and then bptt
                    # must be small (<= 8) and lr low.
                    if args.replay_state:
                        st = b_state[t]
                    mean, st_next = policy(obs, st)
                    feats = policy.assemble(obs, st)
                    value = critic(feats, st_next[:, H_LO:H_HI])
                    std = log_std.exp().clamp_min(args.min_std)
                    logp = (-0.5 * (((b_act[t] - mean) / std) ** 2)
                            - log_std - 0.5 * float(np.log(2 * np.pi))).sum(-1)
                    ratio = (logp - b_logp[t]).exp()
                    a_t = adv[t]
                    pi_loss = pi_loss - torch.min(
                        ratio * a_t,
                        ratio.clamp(1 - args.clip, 1 + args.clip) * a_t).mean()
                    v_loss = v_loss + F.mse_loss(value, ret[t])
                    ent_sum = ent_sum + (log_std + 0.5 * float(np.log(2 * np.pi * np.e))).sum()
                    with torch.no_grad():
                        a_mean, _ = anchor(obs, st)
                        kl_sum = kl_sum + (b_logp[t] - logp).mean()
                    if anchor_w > 0:
                        bc_loss = bc_loss + F.mse_loss(mean, a_mean)
                    # Detach the state we carry INTO the next timestep only at the chunk edge;
                    # inside the chunk the gradient must flow through the recurrence.
                    st = st_next
                n = max(1, c1 - c0)
                loss = (pi_loss / n + args.vf * v_loss / n - args.ent * ent_sum / n
                        + anchor_w * bc_loss / n)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(policy.parameters()) + list(critic.parameters()) + [log_std], 1.0)
                opt.step()
                stats["pi"] += float(pi_loss.detach()) / n
                stats["v"] += float(v_loss.detach()) / n
                stats["bc"] += float(bc_loss.detach()) / n if anchor_w > 0 else 0.0
                stats["kl"] += float(kl_sum.detach()) / n
                stats["n"] += 1
                if float(kl_sum.detach()) / n > args.target_kl:
                    stop = True
                    break

        # Sequential-BPTT path shares the reporting, eval and checkpointing of the flat one.
        _report(update, total_steps, steps_at_start, t_start, b_rew, stats, anchor_w,
                log_std, recent, args, out, policy)
        best = _maybe_eval(update, args, policy, out, best)
        _checkpoint(update, args, out, policy, critic, opt, log_std, total_steps, best)

    torch.save(policy.state_dict(), out)
    venv.close()
    print(f"done: {total_steps:,} steps in {(time.monotonic()-t_start)/3600:.2f} h -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="engine/runs/warm.pt")
    ap.add_argument("--out", default="engine/runs/ppo.pt")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--freeze-trunk", type=int, default=0,
                    help="hold enc1/enc2/gru and train only the per-event heads, so a "
                         "multi-event pool cannot degrade the gait it starts from")
    ap.add_argument("--events", default="",
                    help="comma-separated event subset to train on, e.g. sprint_100 "
                         "(default: all six, round-robin across workers)")
    ap.add_argument("--horizon", type=int, default=128)
    ap.add_argument("--bptt", type=int, default=32)
    ap.add_argument("--updates", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--ent", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--init-std", type=float, default=0.25)
    ap.add_argument("--min-std", type=float, default=0.05)
    ap.add_argument("--anchor", type=float, default=2.0)
    ap.add_argument("--anchor-steps", type=int, default=3_000_000)
    ap.add_argument("--step-cap", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--target-kl", type=float, default=0.03)
    ap.add_argument("--replay-state", type=int, default=1,
                    help="1 = feed stored states (stable, enables the flat batched update); "
                         "0 = true BPTT (needs small bptt/lr)")
    ap.add_argument("--minibatch", type=int, default=1024,
                    help="samples per gradient step on the flat path; 0 = one batch of T*N")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="score against the real referee every N updates (0 = never)")
    ap.add_argument("--eval-seeds", type=int, default=2)
    ap.add_argument("--resume", type=int, default=1, help="resume from <out>_state.pt if present")
    ap.add_argument("--seed", type=int, default=0)
    train(ap.parse_args())


