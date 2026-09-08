#!/usr/bin/env bash
# Phase 1: rebuild the base editor from a cold start. WEEKS of GPU time.
#
# THIS IS A RECONSTRUCTION OF THE METHOD, NOT A REPLAY OF THE LINEAGE. The
# released base came out of a longer and partly unrecorded process; this script
# follows the documented schedule and will land somewhere else on the same curve.
# That is why the base is released as a checkpoint -- phase 2, which is the
# actual contribution, does not depend on rebuilding it.
#
#   ./scripts/train_phase1.sh --dry-run     validate every stage's hparams
#   ./scripts/train_phase1.sh               run it
#
# THE ORDER MATTERS MORE THAN THE VALUES. Train at mask ratio 0.8 FIRST, then
# switch to 0.15. A from-scratch 0.15 run FAILS: two attempts pinned substitution
# accuracy at 1.6%, because 0.15 recalibrates a duration function that already
# works and cannot teach one. The 0.8 -> 0.15 switch is the largest single
# measured win here: a paired control on the same checkpoint moved substitution
# accuracy 15.6% -> 42.2%.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
EXP="${EXP:-phase1}"
DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="1"

# stage | mask ratio | lr | batch | until step | extra hparams
STAGES=(
  "A|0.8 |2e-4|32|11000|warmup_updates=8000"
  "B|0.15|2e-5|16|11500|"
  "C|0.15|2e-5|16|20000|"
  "D|0.15|2e-5|16|25000|lambda_phone_mel_contrast=0.05"
  "E|0.15|2e-5|16|30000|lambda_phone_mel_contrast=0.05"
)

for spec in "${STAGES[@]}"; do
  IFS='|' read -r name ratio lr batch until extra <<<"$spec"
  HP="training_mask_ratio=${ratio// /},lr=$lr,max_sentences=$batch,max_updates=$until"
  [ -n "$extra" ] && HP="$HP,$extra"

  if [ -n "$DRY" ]; then
    # --hparams can only OVERRIDE keys that already exist, so a typo in a late
    # stage would otherwise surface hours in, after the earlier stages have run.
    #
    # This must NOT be piped into grep. `| grep ... || true` reports success
    # whatever happens: set_hparams prints "Unknow hparams: []" unconditionally
    # before it checks anything, and the `|| true` swallows every nonzero exit.
    # --dry-run resolves the config and exits nonzero on a bad key.
    echo "==> stage $name: validating $HP"
    "$PY" -m wavepainter.train --config configs/base.yaml \
        --exp_name "${EXP}_dryrun" --hparams "$HP" --dry-run
    continue
  fi

  echo "==> stage $name -> step $until  (ratio $ratio, lr $lr, batch $batch)"
  if [ "$name" = "D" ] && [ ! -f assets/phone_mel_prototypes.npz ]; then
    echo "    building the phone-mel prototype bank first"
    "$PY" scripts/build_phone_mel_prototypes.py --output assets/phone_mel_prototypes.npz
  fi
  # Stage A starts fresh; every later stage resumes the same work_dir, which is
  # what carries the optimizer state and the warmup schedule position forward.
  RESET=""
  [ "$name" = "A" ] && RESET="--reset"
  "$PY" -m wavepainter.train --config configs/base.yaml \
      --exp_name "$EXP" $RESET --hparams "$HP"
done

echo
echo "Phase 1 done. Select on substitution accuracy, then insertion accuracy,"
echo "then WER -- never on validation loss (step 19500 improved val loss while"
echo "losing 6 substitutions and 9 insertions)."
echo "Then: BASE=checkpoints/$EXP/model_ckpt_steps_30000.ckpt ./scripts/train_phase2.sh"
