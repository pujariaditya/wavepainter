#!/usr/bin/env bash
# Score each alpha on all three legs, sequentially.
#
#   ./scripts/run_alpha_sweep.sh 0.70 0.75 0.85 0.90 1.00
#
# Each alpha needs a checkpoint directory `checkpoints/alpha_<a>/` built by
# interpolating the phase-2 child back toward the phase-1 base; see
# docs/RESULTS.md section 3. Paths come from the environment, never from this file --
# WAVEPAINTER_BENCHMARK and SV_CKPT default to the layout `./setup.sh`
# and `scripts/fetch_benchmark.py` produce.
set -uo pipefail
cd "$(dirname "$0")/.."

export PYTHON="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WAVEPAINTER_BENCHMARK="${WAVEPAINTER_BENCHMARK:-data/ming}"
export SV_CKPT="${SV_CKPT:-third_party/wavlm_large_finetune.pth}"
export PYTHONPATH="$PWD"

mkdir -p out/sweep
for A in "$@"; do
  echo "########## alpha $A"
  EXP=alpha_$A OUT=out/sweep/alpha_$A ./scripts/verify_benchmark.sh \
    > out/sweep/alpha_$A.log 2>&1 && echo "  done" || echo "  FAILED (see out/sweep/alpha_$A.log)"
done
echo "SWEEP_DONE"
