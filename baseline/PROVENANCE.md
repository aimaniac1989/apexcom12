# baseline.onnx provenance

`baseline.onnx` is a reproducible integration policy, not an Olympics-trained contender. It must
score above zero through the actual player/referee loop, but it is deliberately not a leaderboard
bar until the native-amd64 calibration evidence in `HANDOFF.md` is accepted at onboarding.

## Immutable inputs

- Unitree source: `https://github.com/unitreerobotics/unitree_rl_gym` at
  `276801e46c5d433564f24658bac64f254b7d2d4b` (BSD-3).
- Source policy: `deploy/pre_train/g1/motion.pt`, SHA-256
  `cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d`.
- Submitted integration artifact: `baseline/baseline.onnx`, SHA-256
  `1d0a88ef2edcd13f9ad0401cc72faaea664951ec0763429ad31bbc907e2954f2`.
- Export equivalence: `tools/make_baseline.py` checks 64 recurrent observations against the
  TorchScript policy. Rebuilding from the pinned revision reproduces the checked-in ONNX byte for
  byte and reports a maximum action difference of `0.0e+00`.

The wrapper preserves the public `[104] + [256] -> [12] + [256]` contract. It maps the stock
47-value Unitree locomotion input from the Olympics observation, supplies a body-frame forward
command, and keeps its LSTM state in `state_in`/`state_out`. It sees no obstacle or jump terrain.

## Rebuild

```bash
git clone --filter=blob:none --sparse https://github.com/unitreerobotics/unitree_rl_gym
cd unitree_rl_gym
git checkout 276801e46c5d433564f24658bac64f254b7d2d4b
git sparse-checkout set deploy/pre_train/g1
cd ..
python tools/make_baseline.py --urlg unitree_rl_gym --out baseline/baseline.onnx
sha256sum baseline/baseline.onnx
```

The exporter rejects any checkout or `motion.pt` whose pinned SHA does not match. Native-amd64
full-meet scores, variance, peak memory, and worst wall time are produced by
`.github/workflows/measure-baseline.yml` and are recorded in the onboarding handoff after its
candidate-release run.
