#!/usr/bin/env bash
# Provision a Linux VPS for Humanoid Olympics training and verify it before spending money on it.
#
#   bash engine/setup_vps.sh              # CPU only (what you want unless the update is batched)
#   bash engine/setup_vps.sh --cuda       # add CUDA torch for the batched update
#
# Versions are pinned to the referee/player images (HANDOFF.md): mujoco 3.11.0 owns the physics
# and therefore the score, so it must match exactly. numpy/onnxruntime affect the score only at
# the ~1e-4 level but are pinned anyway so a local number and a submitted number agree.
set -euo pipefail

CUDA=0
[[ "${1:-}" == "--cuda" ]] && CUDA=1

command -v python3 >/dev/null || { echo "python3 not found"; exit 1; }
cd "$(dirname "$0")/.."
echo "== repo: $(pwd)"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -qU pip wheel

echo "== pinned runtime (must match the referee image)"
pip install -q "mujoco==3.11.0" "numpy==2.3.4" "onnxruntime==1.28.0" onnx psutil

if [[ $CUDA -eq 1 ]]; then
  echo "== torch (CUDA 12.1)"
  pip install -q torch --index-url https://download.pytorch.org/whl/cu121
else
  echo "== torch (CPU)"
  pip install -q torch --index-url https://download.pytorch.org/whl/cpu
fi

export PYTHONPATH="$(pwd)"

echo
echo "== hardware"
python - <<'PY'
import multiprocessing, os
print(f"   logical cpus     {multiprocessing.cpu_count()}")
try:
    print(f"   usable cpus      {len(os.sched_getaffinity(0))}")
except AttributeError:
    pass
with open("/proc/meminfo") as fh:
    for line in fh:
        if line.startswith("MemTotal"):
            print(f"   {line.strip()}")
            break
import torch
print(f"   torch            {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   gpu              {torch.cuda.get_device_name(0)}")
PY

echo
echo "== precompiling scenes + determinism gate"
echo "   (a .mjb worker costs ~102 MB vs ~534 MB compiling from XML -- this is what"
echo "    decides how many workers fit in RAM, and the gate proves it is bit-identical)"
python engine/mjb.py

echo
echo "== environment throughput on THIS box"
python - <<'PY'
import time, numpy as np
from env.sim import OlympicsSim, instance_spec
from engine.mjb import install
install("sprint_100")
sim = OlympicsSim(instance_spec("sprint_100", 0, 1)); sim.reset()
a = np.zeros(12); sim.step(a)
t0 = time.perf_counter(); N = 800
for _ in range(N):
    if sim.step(a).terminal_reason is not None:
        sim.reset()
rate = N / (time.perf_counter() - t0)
print(f"   {rate:.0f} control steps/s per core")
import os
try:
    cores = len(os.sched_getaffinity(0))
except AttributeError:
    import multiprocessing; cores = multiprocessing.cpu_count()
w = max(1, cores - 2)
print(f"   ~{w} workers -> ~{rate*w*0.75:,.0f} steps/s projected (0.75 = measured PPO overhead)")
print(f"   700M steps -> ~{700e6/(rate*w*0.75)/3600:.1f} h")
PY

echo
echo "== submission contract (the gates that zero a round)"
python engine/export.py --out engine/submission.onnx | tail -n 20

echo
echo "== ready. Suggested first run:"
echo "   PYTHONPATH=\$(pwd) python engine/distill.py --iters 12 --episodes 48 --out engine/runs/warm.pt"
echo "   PYTHONPATH=\$(pwd) python engine/ppo.py --init engine/runs/warm.pt \\"
echo "       --workers \$(( \$(nproc) - 2 )) --horizon 128 --minibatch 2048 \\"
echo "       --eval-every 25 --eval-seeds 2 --save-every 10 --updates 20000 \\"
echo "       --out engine/runs/ppo.pt 2>&1 | tee engine/runs/ppo.log"
