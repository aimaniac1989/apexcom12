# Competition onboarding manifest: `humanoid_olympics`

This document records the signed release image digests and release verification. It is not an
onboarding request by itself. As of 0.2.0 there is no calibration gate: `baseline_raw_score` is
0.0 by choice, so no measured baseline has to be certified before a release.

## 1. Goal statement & alignment plan

**What success looks like**

> A winning submission is one compact recurrent, legs-only Unitree G1 controller that can sprint,
> stay on a narrow lane through a sustained corner, clear progressively harder hurdles, and execute
> legal vertical and horizontal jumps under seeded surface and wind conditions. It must be an
> adaptable all-round athletic controller rather than six memorised or invalid trajectories.

**Alignment checks and review plan**

- Replays show the same controller completing or materially progressing across every event, with no
  event score carried by invalid contacts, boundary violations, or typed player faults.
- The 400 m diagnostic shows forward lap progress while inside the lane; jump diagnostics show
  legal board take-off, required flight phases, and foot-top landing contacts.
- Top policies fit the CPU/2 GiB budget and answer within the fixed 500 ms action deadline.
- After every round reveal, inspect top-event histories plus event-score distributions on fresh
  seeds. Watch for score spikes concentrated in one event, contact-foul patterns, long evaluator
  times, or policies that rank well without exhibiting the stated athletic behaviours.

## 2. Deliverables

| Item | Where | Status |
|---|---|---|
| Competition repo + final release tag | `macrocosm-os/apex-competition-humanoid-olympics` @ `v0.1.0` | final release [run 32039533955](https://github.com/macrocosm-os/apex-competition-humanoid-olympics/actions/runs/32039533955) passed all jobs |
| `spec.yaml` (`apex.competition.v1`) + preflight | `spec.yaml`, `fixtures/input.json` | local stage/prod preflight passes; repeat after digest pin |
| Player image | `ghcr.io/macrocosm-os/apex-competition-humanoid-olympics-player@sha256:92e6aab7bfc511f1e6d107432cf1e368962c6fc716e1abd30ff8fbc1debea8a0` | candidate signature verified |
| Referee image | `ghcr.io/macrocosm-os/apex-competition-humanoid-olympics-referee@sha256:2197d8bfd18196ad74e93a902860c4590caeefafcaa37c38af48a98edccc532b` | candidate signature verified |
| Layer-2 screen | n/a; fixed-shape ONNX artifact plus platform structural screen and player validation | n/a |
| Round generation | n/a; one platform master seed is sufficient | complete |
| Cosign identity + issuer | `.github/workflows/release.yml`; GitHub Actions OIDC | candidate and final workflows verify it |
| Input schema + fixture | `input.schema.json`, `fixtures/input.json` | complete |
| Baseline integration artifact | `baseline/baseline.onnx` | pinned; smoke-tested in release CI. No calibration gate as of 0.2.0 — `baseline_raw_score` is 0.0 by choice, so there is no measured number to certify |
| Miner documentation | `README.md`, `docs/design.md` | complete |
| Full end-to-end evidence | release CI two-container job + candidate calibration artifact | candidate [run 32039171270](https://github.com/macrocosm-os/apex-competition-humanoid-olympics/actions/runs/32039171270) and final [run 32039533955](https://github.com/macrocosm-os/apex-competition-humanoid-olympics/actions/runs/32039533955) passed all jobs |

Score-affecting pins already committed:

- player base/runtime: `python:3.12-slim`, `numpy==2.3.4`, `onnxruntime==1.28.0`;
- referee base/runtime: `python:3.12-slim`, `numpy==2.3.4`, `mujoco==3.11.0`;
- vendored protocol: `macrocosm-os/apex-competitions-builder@d063d9028dbec4bb15182794496f4aa2aac19d49`;
- Unitree stock-policy source: `unitree_rl_gym@276801e46c5d433564f24658bac64f254b7d2d4b`;
- `motion.pt`: `cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d`;
- baseline ONNX: `1d0a88ef2edcd13f9ad0401cc72faaea664951ec0763429ad31bbc907e2954f2`;
- datasets / held-out corpora: n/a; all course geometry is code-versioned in the referee image.

## 3. Ops parameters

| Parameter | Proposal | Reason |
|---|---|---|
| Process / kind | CPU / solo | A single 15 MB ONNX policy and deterministic MuJoCo physics fit CPU; score is absolute. |
| Round length / reveal | 2 days / 5 days | The fixed v0.1 meet can loop while trained-control breakthroughs retain meaningful IP. |
| Score direction | higher is better | Legal speed, height, and distance are monotone athletic improvements. |
| Launch meet | 4 attempts × 6 events; 40,000 maximum actions | Fixed shape makes cross-round scores comparable while sampling four condition strata. |
| Resources | 2 CPU, 2 GiB, 0 GPU | Native evidence peaked at 927.8 MiB, leaving >50% headroom; this ceiling is exercised in release CI. |
| Timeouts | player 1,200 s; referee 900 s; internal scheduler 840 s | Leaves persistence time after a full recorded 24-attempt meet. |
| Action deadline | 500 ms | Caps inference latency while remaining realistic for compact recurrent CPU ONNX. |
| Baseline score | 0.0 by choice | The platform entry bar, not a measurement. See `spec.yaml` `defaults` for why 0.1.0's measured 0.032985375 was withdrawn. |
| Submission fee | propose USD 1 in TAO | Discourages low-effort repetition and approximately covers simulator cost. |
| Incentive weight | propose 0.03 | Middle of the normal 0.02–0.05 range, subject to Macrocosmos. |

## 4. Evaluation sizing

The fixed launch unit is 24 attempts: four deterministic condition strata for each of six
equal-weight events, repeated identically for every round seed. The candidate workflow runs the baseline, a stationary valid ONNX reference,
and an independently seeded untrained ONNX reference across 20 master seeds in the actual
two-container native-amd64 setup. It records per-seed raw/event scores, standard deviation,
ordering, peak container memory, and wall time in `baseline-calibration-native-amd64`.

The figures below were measured under 0.1.0, **before** the contact-priority fix, and are retained
as resource and wall-time evidence only. Their score components are superseded: 0.1.0 ran 18 of the
24 attempts with friction clamped to mu 1.0, so every score here describes a grippier meet than
0.2.0 evaluates. They are NOT a calibration of the current physics, and nothing depends on them
being one — `baseline_raw_score` is 0.0 by choice as of 0.2.0.

- Evidence: [native calibration run 32037213638](https://github.com/macrocosm-os/apex-competition-humanoid-olympics/actions/runs/32037213638), artifact `baseline-calibration-native-amd64` (`9291517227`).
- Superseded score components (0.1.0 physics): baseline mean raw `0.032985375`, identical on all 20
  seeds, sample standard deviation `0.0`; event means 100 m `0.040466`, 400 m `0.004020`, hurdles
  `0.032187`, high `0`, long `0.121240`, triple `0`; baseline ranked above both weaker references on
  every seed (static `0.002628292`, random `0.002610667`).
- Still current — resources are unchanged by the fix, which alters contact parameters rather than
  step counts: full recorded baseline wall time mean `65.705 s`, max `66.2 s`; peak measured
  container memory `927.8 MiB`, below half of the declared 2 GiB envelope.
- Cross-checked on staging under 0.1.0: three submissions of a 14.46 MiB near-cap artifact scored
  `0.032985375` with referee-reported eval times of 354.1/356.4/353.4 s, reproducing the native
  event means digit for digit. Stage is ~5.4x slower than native amd64, so a policy running every
  attempt to its step cap projects to ~632 s against the referee's 840 s internal budget.

A seed-stability gate is no longer meaningful here: the launch meet is seed-neutral by construction,
so 20 seeds return one value and the variance check can only ever pass. If the condition mix or
attempt count changes, that is a new spec version with its own sizing evidence.

## 5. Threat-model questionnaire

1. **Miner-visible surface.** The player gets its own public 104-float observations, reset
   identifier/index, reset seed `0`, and the fixed 500 ms deadline. Observations describe current
   robot/course state only. It receives no platform seed, future conditions, referee filesystem,
   or other submission.
2. **Seed leverage.** The platform round seed remains referee-side but is intentionally
   score-neutral for v0.1. Every round repeats the same public, balanced condition suite, so no
   seed can select an easier meet or move the absolute baseline.
3. **Degenerate submissions.** Static and seeded-untrained ONNX references are measured in the
   candidate suite. Malformed/NaN/wrong-shape responses receive typed invalid outcomes; boundary,
   invalid-contact, and player faults contribute zero.
4. **Baseline resubmission.** The published baseline is a reproducible integration reference. An
   identical artifact reproduces its fixed-round result and cannot exceed itself by the required
   1% takeover margin.
5. **Metric gaming.** Review covered malformed/non-finite actions, timeout and reset faults,
   lateral bypass, hurdle contacts, ducking, bar contact, board skipping, wrong-foot pads,
   side/underside contacts, body-first landings, and floor routing. The referee owns typed action
   validation, 500 Hz route/contact gates, physical obstacles, top-face normal/impulse support
   classification, phase state machines, and bounded event scores.
6. **Copy-plus-epsilon.** The five-day reveal delay is longer than the two-day round. Within a
   round, fixed seed/config means a trivial copy reproduces its score rather than gaining 1%.
7. **Cross-round leakage.** Per-attempt histories are released only after the round and show
   already-observed public physics. The v0.1 meet repeats by design; there is no hidden answer
   corpus, unrevealed condition stream, or other submission to leak.
8. **Error-message hygiene.** Miner-facing outcomes are stable categories such as invalid action,
   player error, hurdle hit, jump foul, or timeout. They contain no stack trace, host path,
   internal seed, secret threshold, or future task information.
9. **Referee state.** For a seed/config/submission the scorer is deterministic. It compiles only
   the current event model, resets simulator/player state per attempt, and writes data only to the
   per-job artifact mount; no submission-keyed persistent cache is used.
10. **Code execution.** The submission type is ONNX, not code/TorchScript. Declarative size and
    weight checks plus player-side exact I/O/type validation constrain it to a CPU graph; no
    custom submission code executes in the referee.
11. **Player-image hygiene.** The player contains only the public HTTP wrapper, vendored protocol,
    pinned ONNX Runtime, and mounted miner artifact. Course geometry, simulation, seed handling,
    scoring, and histories remain in the referee image.
12. **Diagnostics payload.** `result.json` and replay history record event, attempt, public
    condition values, terminal reason, score, metrics, and robot trajectory after a round. They
    do not include another submission, an unrevealed future seed, secret corpus, or hidden judge.

## 6. GPU justification

Not applicable. Both policy inference and referee physics are CPU-only; the candidate workflow
verifies them under the declared 2 CPU / 2 GiB limit.

## 7. Candidate-to-onboarding procedure

1. Push this source. There is no calibration tag to cut: `baseline_raw_score` is 0.0 by choice, so
   nothing about the release depends on re-measuring the baseline. `measure-baseline.yml` remains
   available via `workflow_dispatch` for resource and wall-time evidence, but it gates nothing.
2. If the sizing and resource checks pass, tag `v0.1.0-candidate.1`; release CI publishes and
   keyless-signs both images, then verifies the signature against the identity/issuer in `spec.yaml`.
3. Commit the resulting candidate digests into `spec.yaml` and this handoff, repeat preflight,
   then tag `v0.1.0`. Docker stamps a fresh image manifest on every clean build, so the platform
   continues to pin the already verified candidate digests while the final-tag workflow separately
   re-runs preflight, the two-container meet, and keyless signature verification.
4. Attach this completed manifest, release tag, image refs/digests, and calibration artifact to a
   Macrocosmos onboarding issue. Do not request production activation before the stage round passes.
