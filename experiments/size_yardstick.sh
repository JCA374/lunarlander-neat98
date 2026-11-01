#!/bin/bash
# How big must NEAT's network be? Copy the PPO champion (16->128->128->4) into small MLPs with DAgger (same budget each:
# 10 rounds x 400 flights, CLONE seeds) and fly each on the same 1,000 fresh analysis seeds of layout v3.
# The copies are a yardstick only, never a starting genome.
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
W=${WORKERS:-6}
SEEDS=3000100000      # ANALYSIS + 100,000: 1,000 seeds for every size and for the teacher
T=models/baselines/ppo_M2_i08100.json
.venv/bin/python -m experiments.evaluate $T --start $SEEDS --n 1000 --workers $W --out results/size_teacher_ppo.json
for h in "8" "16" "32" "16 16" "64"; do
  tag=$(echo $h | tr ' ' x)
  .venv/bin/python -m rl.bc --teacher $T --hidden $h --rounds 10 --flights 400 --workers $W --out models/yardstick/bc_$tag.json
  .venv/bin/python -m experiments.evaluate models/yardstick/bc_$tag.json --start $SEEDS --n 1000 --workers $W --out results/size_$tag.json
done
echo DONE
