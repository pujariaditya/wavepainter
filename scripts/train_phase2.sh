#!/usr/bin/env bash
# Phase 2, end to end, from the released phase-1 base. ~1 GPU-hour.
#
# This is the contribution, and it is the one to run to check the claim: it does
# not depend on rebuilding phase 1, because the base is a released checkpoint.
#
# Note the caveat in the README: a retrain lands on a distribution around the
# reported numbers rather than exactly on them. The released checkpoint is the
# artifact of record.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
EXP="${EXP:-phase2_rerun}"
BASE="${BASE:-checkpoints/phase1-base/model_ckpt_steps_5000.ckpt}"

[ -f "$BASE" ] || { echo "missing $BASE -- run: python scripts/download_weights.py --phase 1" >&2; exit 1; }

echo "==> training the denoiser-only child (1500 updates, lr 2e-5)"
"$PY" -m wavepainter.train \
    --config configs/phase2_hubert_dn_a085.yaml \
    --exp_name "$EXP" --reset \
    --hparams "load_ckpt=$BASE"

echo "==> done: checkpoints/$EXP/model_ckpt_steps_1500.ckpt"
echo "    That is the released model -- phase two as trained. Score it with:"
echo "      EXP=$EXP STEPS=1500 ./scripts/verify_benchmark.sh"
