"""Our Humanoid Olympics policy. One recurrent controller, six events, built from scratch.

Designed against `engine/contract.py` and against three things measured in this repo's own
simulator rather than assumed:

1.  **The event is identifiable on step 1 and only on step 1.** `obs[50]` is `remaining_route_m/10`
    and each event has a distinct route length at reset -- high 14.0, long 23.0, triple 27.0,
    100 m 102.0, 400 m 400.0. It then decays, so the classification must be LATCHED into the
    recurrent state or it silently changes mid-attempt.

2.  **Obstacles are largely invisible, so course position is the real percept.** Hurdles are
    `walkable=False`, which puts them in `OVERHEAD_GROUP`: they never appear in the downward
    height scan, and the upward rays start at `pelvis + 0.05` ~ 1.64 m, above the first six
    hurdles entirely. Measured: the 0.55 m hurdle is detected at 0 of 106 approach positions; the
    1.15 m one at 35 of 106, because the 7 overhead rays are 0.667 m apart and a hurdle is 0.24 m
    wide. A reactive policy therefore cannot clear them. But the geometry is public and fixed, so
    once the event is known, `px = finish_e - 10*obs[50]` recovers absolute course position and
    every obstacle's range is computable. That is what `_geometry` feeds the network.

3.  **The action must be bounded at the env clip.** `env/sim.py` clips to +-10 before forming the
    PD target, so any policy railing past it is a bang-bang controller: maximally sensitive to
    perturbation, and unusable at the low action noise a fine-tune needs. The head is
    `10*tanh(a/10)`, which is smooth, cannot rail, and cannot emit NaN.

Recurrent memory is not decoration here: friction (mu in [0.30, 1.25]) and wind (0-8 m/s) are
drawn per attempt and are NOT observable. The only direct evidence is the PD tracking residual
`0.25*prev_action - (q - q_default)`, which is nonzero exactly when something external is
fighting the loop. Multi-timescale EMAs of it are handed to the network explicitly, because a
GRU would need a near-unity eigenvalue to learn a 300-step average and that is the classic hard
case for BPTT.

State layout of the 256 opaque floats (zeroed at every attempt reset):

    [  0:128)  h   GRU hidden                                 128
    [128:200)  D   3 timescales x 24-dim disturbance vector     72
    [200:203)  W   EMA warm-up scalars, 1 - lambda^t             3
    [203:208)  E   latched event one-hot                         5
    [208:209)  L   latch flag                                    1
    [209:210)  S   sticky "overhead obstacle seen" evidence bit   1
    [210:256)  unused, held at zero                              46
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# -- interface -------------------------------------------------------------------------------
OBS_DIM, ACT_DIM, STATE_DIM = 104, 12, 256
ACTION_ENV_CLIP = 10.0     # env/sim.py clips here; the head is bounded AT it, never past it
ACTION_SCALE = 0.25

# -- event classification, from the reset route length ---------------------------------------
# high 1.40 | long 2.30 | triple 2.70 | 100 m 10.20 | 400 m 40.00  (obs[50] at reset)
EVENT_NAMES = ("high_jump", "long_jump", "triple_jump", "flat_100", "circuit_400")
EVENT_EDGES = (1.85, 2.55, 6.0, 25.0)
# Route length used to recover absolute course position for each class.
EVENT_FINISH = (12.0, 21.0, 25.0, 100.0, 400.0)
N_EVENT = len(EVENT_NAMES)

# -- public, fixed obstacle coordinates (env/course.py) --------------------------------------
HURDLE_X = tuple(12.0 + i * (88.5 - 12.0) / 9 for i in range(10))   # linspace(12.0, 88.5, 10)
BAR_X = (12.0,)                                    # high jump bar plane
LONG_X = (15.0, 21.0)                              # take-off board, sand edge
TRIPLE_X = (12.0, 14.75, 18.75, 25.0)              # board, hop pad, step pad, sand edge
GEO_X = HURDLE_X + BAR_X + LONG_X + TRIPLE_X
GEO_DIM = len(GEO_X)                               # 17

# -- disturbance observer ---------------------------------------------------------------------
V_DIM = 24
LAMBDAS = (0.8, 0.97, 0.997)                       # tau ~ 5, 33, 333 steps at 50 Hz
N_TAU = len(LAMBDAS)
D_DIM = N_TAU * V_DIM                              # 72

H_DIM = 128
H_LO, H_HI = 0, H_DIM                              # [  0:128)
D_LO, D_HI = H_HI, H_HI + D_DIM                    # [128:200)
W_LO, W_HI = D_HI, D_HI + N_TAU                    # [200:203)
E_LO, E_HI = W_HI, W_HI + N_EVENT                  # [203:208)
L_LO, L_HI = E_HI, E_HI + 1                        # [208:209)
S_LO, S_HI = L_HI, L_HI + 1                        # [209:210)
assert S_HI <= STATE_DIM

IN_DIM = OBS_DIM + D_DIM + N_TAU + N_EVENT + 1 + GEO_DIM   # 104+72+3+5+1+17 = 202

# Overhead channel reads SCAN_CLIP (2.0) when nothing is above; anything lower is a real
# structure. 1.95 leaves margin for ray noise without firing on clear track.
OVERHEAD_CLEAR = 1.95


def expand_head1(sd: dict, head_dim: int = 256) -> dict:
    """Migrate a shared-head1 checkpoint to the per-event layout, or pass a new one through.

    Every event class starts as a COPY of the trained shared head, so the migrated policy is
    behaviourally identical to the one it came from -- day zero is a no-op and the classes
    diverge only as training separates them. That identity is the migration's own test: export
    before and after and the actions must match to float tolerance.
    """
    w, b = sd.get("head1.weight"), sd.get("head1.bias")
    if w is None or w.shape[0] == N_EVENT * head_dim:
        return sd                                    # already per-event, or not a policy dict
    sd = dict(sd)
    sd["head1.weight"] = w.repeat(N_EVENT, 1)        # tile along out_features: slice k == old
    sd["head1.bias"] = b.repeat(N_EVENT)
    return sd


class GRUCell(nn.Module):
    """PyTorch's GRU cell in elementary ops.

    `nn.GRU` exports to a fused ONNX `GRU` node wanting a sequence axis and a
    [layers, batch, hidden] state; spelling the cell out keeps the graph to
    Gemm/Sigmoid/Tanh/Mul/Add with every shape static, which is what the player's
    load-time shape check needs.
    """

    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.x2h = nn.Linear(in_dim, 3 * hidden)
        self.h2h = nn.Linear(hidden, 3 * hidden)
        bound = hidden ** -0.5
        for lin in (self.x2h, self.h2h):
            nn.init.uniform_(lin.weight, -bound, bound)
            nn.init.uniform_(lin.bias, -bound, bound)
        # Bias the update gate toward retention: h' = (1-z)*n + z*h, so z near 1 keeps memory.
        # Without this a fresh GRU forgets every step and the long-horizon signal never trains.
        with torch.no_grad():
            self.x2h.bias[hidden:2 * hidden].fill_(1.0)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        n = self.hidden
        gi, gh = self.x2h(x), self.h2h(h)
        r = torch.sigmoid(gi[:, :n] + gh[:, :n])
        z = torch.sigmoid(gi[:, n:2 * n] + gh[:, n:2 * n])
        c = torch.tanh(gi[:, 2 * n:] + r * gh[:, 2 * n:])
        return (1.0 - z) * c + z * h


class OlympicsPolicy(nn.Module):
    """obs[104] + state_in[256] -> action[12] + state_out[256].

        x  = [obs | D | W | E | S | geo]              202
        z  = clip((x - mean) * istd, -10, +10)
        e  = ELU(enc2(ELU(enc1(z))))                  202 -> 384 -> 256
        h' = GRUCell(e, h)                            256 x 128 -> 128
        g  = ELU(head2(ELU(head1([e | h']))))         384 -> 256 -> 128
        a  = 10 * tanh(head3(g) / 10)                 128 -> 12

    `e` reaches the head directly as well as through the GRU. Take-off timing and hurdle
    clearance are functions of the CURRENT geometry and must not be bottlenecked through a
    latent whose job is estimating friction and wind over hundreds of steps.
    """

    def __init__(self, enc=(384, 256), head=(256, 128), hidden: int = H_DIM) -> None:
        super().__init__()
        self.enc1 = nn.Linear(IN_DIM, enc[0])
        self.enc2 = nn.Linear(enc[0], enc[1])
        self.gru = GRUCell(enc[1], hidden)
        # PER-EVENT first head. Measured at 44M steps on a four-event pool: the shared head
        # cannot hold "run 100 m" and "stop at the 15 m board" at once. Every event converged on
        # 13-15 m -- optimal for long_jump and triple_jump, and it cost sprint_100 58.2 m -> 13.1 m
        # WHILE IT WAS BEING TRAINED. That is interference, not forgetting, and it lives in the
        # layers that pick a behaviour rather than in the ones that produce a gait. So enc1/enc2
        # and the GRU stay shared -- all six events are the same locomotion problem and splitting
        # them would divide the data for the skill that needs it most -- and only this layer forks.
        #
        # Laid out as one wide Gemm sliced by the latched one-hot rather than N_EVENT separate
        # Linears: the graph stays Gemm/Reshape/Mul/ReduceSum with every shape static, which is
        # what the player's load-time shape check needs (see GRUCell above).
        self.head_dim = head[0]
        self.head1 = nn.Linear(enc[1] + hidden, N_EVENT * head[0])
        self.head2 = nn.Linear(head[0], head[1])
        self.head3 = nn.Linear(head[1], ACT_DIM)

        # Start near the default pose: a locomotion policy that begins by flailing spends its
        # first million steps learning not to.
        nn.init.orthogonal_(self.head3.weight, gain=0.01)
        nn.init.zeros_(self.head3.bias)

        self.register_buffer("mean", torch.zeros(IN_DIM))
        self.register_buffer("istd", torch.ones(IN_DIM))
        lam = torch.tensor(LAMBDAS, dtype=torch.float32)
        self.register_buffer("lam_d", lam.repeat_interleave(V_DIM))       # [72]
        self.register_buffer("lam_w", lam)                                # [3]
        self.register_buffer("edges", torch.tensor(EVENT_EDGES, dtype=torch.float32))
        self.register_buffer("finish", torch.tensor(EVENT_FINISH, dtype=torch.float32))
        self.register_buffer("geo_x", torch.tensor(GEO_X, dtype=torch.float32))

    # -- perception helpers --------------------------------------------------------------------

    def _classify(self, obs: torch.Tensor) -> torch.Tensor:
        """One-hot event class from the reset route length. Valid ONLY on the first step."""
        d = obs[:, 50:51]
        lo = torch.cat([torch.full_like(d, -1e9), self.edges.expand(d.shape[0], -1)], dim=1)
        hi = torch.cat([self.edges.expand(d.shape[0], -1), torch.full_like(d, 1e9)], dim=1)
        return ((d >= lo) & (d < hi)).to(obs.dtype)

    def _geometry(self, obs: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        """Signed range to every fixed obstacle, from recovered absolute course position.

        This is the channel that makes the hurdles and the jump boards tractable: they are
        invisible to the height scan and mostly invisible overhead, but their coordinates are
        public and `obs[50]` still reports distance-to-finish, so range is recoverable exactly.
        """
        finish = (event * self.finish).sum(dim=1, keepdim=True)      # [B,1]
        px = finish - 10.0 * obs[:, 50:51]
        # Ahead is positive. Clipped so a far-away obstacle saturates instead of dominating.
        return torch.clamp((self.geo_x - px) / 5.0, -0.6, 2.0)

    def _disturbance(self, obs: torch.Tensor) -> torch.Tensor:
        """The 24-d load observer. `track_err` is the only direct evidence of friction/wind."""
        grav = obs[:, 0:3]
        angvel = obs[:, 3:6]
        linvel = obs[:, 6:9]
        q = obs[:, 9:21]
        prev_a = obs[:, 33:45]
        heading = obs[:, 47:49]
        pelvis = obs[:, 51:52]
        track_err = ACTION_SCALE * prev_a - q
        return torch.cat([grav, angvel, linvel, track_err, pelvis, heading], dim=1)

    # -- forward -------------------------------------------------------------------------------

    def forward(self, obs: torch.Tensor, state_in: torch.Tensor):
        h = state_in[:, H_LO:H_HI]
        d = state_in[:, D_LO:D_HI]
        w = state_in[:, W_LO:W_HI]
        e_prev = state_in[:, E_LO:E_HI]
        latch = state_in[:, L_LO:L_HI]
        seen = state_in[:, S_LO:S_HI]

        # Latch the event on the first step of the attempt and hold it. `obs[50]` decays during
        # the run, so re-classifying every step would silently walk the policy across classes.
        event = torch.where(latch >= 0.5, e_prev, self._classify(obs))

        # sprint_100 and hurdles_100 are indistinguishable at reset (both 102.0 m). The only
        # later evidence is an overhead return, and it is weak -- the first six hurdles are below
        # the ray origin entirely. Accumulate it as a sticky bit rather than trusting one frame.
        seen_now = (obs[:, 97:104].min(dim=1, keepdim=True).values < OVERHEAD_CLEAR).to(obs.dtype)
        seen_new = torch.maximum(seen, seen_now)

        geo = self._geometry(obs, event)
        x = torch.cat([obs, d, w, event, seen_new, geo], dim=1)
        z = torch.clamp((x - self.mean) * self.istd, -10.0, 10.0)
        enc = F.elu(self.enc2(F.elu(self.enc1(z))))

        h_new = self.gru(enc, h)
        # `event` is exactly one-hot (see _classify), so this selects one event's head rather
        # than mixing them; the other slices get a zero weight and no gradient.
        g = self.head1(torch.cat([enc, h_new], dim=1)).view(-1, N_EVENT, self.head_dim)
        g = F.elu((g * event.unsqueeze(-1)).sum(dim=1))
        g = F.elu(self.head2(g))
        # Bounded AT the env clip: smooth, cannot rail, cannot produce NaN or inf.
        action = ACTION_ENV_CLIP * torch.tanh(self.head3(g) / ACTION_ENV_CLIP)

        v = self._disturbance(obs)
        d_new = self.lam_d * d + (1.0 - self.lam_d) * v.repeat(1, N_TAU)
        w_new = self.lam_w * w + (1.0 - self.lam_w)
        pad = torch.zeros_like(state_in[:, S_HI:])
        state_out = torch.cat([h_new, d_new, w_new, event, torch.ones_like(latch),
                               seen_new, pad], dim=1)
        return action, state_out

    # -- utilities ------------------------------------------------------------------------------

    def assemble(self, obs: torch.Tensor, state_in: torch.Tensor) -> torch.Tensor:
        """The 202-vector the normaliser sees. Collect these over rollouts to fit it."""
        event = torch.where(state_in[:, L_LO:L_HI] >= 0.5, state_in[:, E_LO:E_HI],
                            self._classify(obs))
        return torch.cat([obs, state_in[:, D_LO:D_HI], state_in[:, W_LO:W_HI], event,
                          state_in[:, S_LO:S_HI], self._geometry(obs, event)], dim=1)

    @torch.no_grad()
    def fit_normalizer(self, x: torch.Tensor, min_std: float = 0.05) -> None:
        """Fit mean/istd from collected `assemble` outputs, [N, 202].

        Refit from rollouts of the CURRENT policy: D and the geometry channels are functions of
        the policy's own behaviour, so a normaliser fitted under random actions is wrong for a
        trained one. Refit and re-freeze before every export.

        `min_std` floors the scale at 0.05, capping istd at 20. Several inputs here are
        near-constant by construction -- the event one-hot is fixed for a whole attempt, the
        warm-up scalars saturate, the geometry channels clip -- and a 1e-4 floor turned those
        into gain-10,000 noise amplifiers (measured: mean istd 1196 across the vector). Leaving a
        constant feature effectively unnormalised is correct; multiplying its jitter by 10^4 is
        not.
        """
        if x.ndim != 2 or x.shape[1] != IN_DIM:
            raise ValueError(f"expected [N, {IN_DIM}], got {tuple(x.shape)}")
        self.mean.copy_(x.mean(0))
        self.istd.copy_(1.0 / x.std(0).clamp_min(min_std))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
