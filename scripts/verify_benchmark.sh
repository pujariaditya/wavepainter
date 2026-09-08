#!/usr/bin/env bash
# Reproduce the reported table from the released checkpoint.
#
# Synthesises all three edit types under the frozen protocol, scores each, and
# prints the comparison against Ren et al. The sampler is seeded per item, so
# these match the published numbers exactly rather than approximately.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
EXP="${EXP:-phase2-hubert-child}"
STEPS="${STEPS:-1500}"
SV_CKPT="${SV_CKPT:-third_party/wavlm_large_finetune.pth}"
OUT="${OUT:-out/verify}"

[ -f "$SV_CKPT" ] || { echo "missing $SV_CKPT -- run ./setup.sh" >&2; exit 1; }
mkdir -p "$OUT"

for edit in sub ins del; do
  echo "==> synthesising $edit"
  "$PY" -m wavepainter.evaluate \
      --exp_name "$EXP" --ckpt_steps "$STEPS" --edit_type "$edit" \
      --protocol "assets/protocols/protocol_ming_en_${edit}.json" \
      --out "$OUT/${edit}.json" --skip_arch

  echo "==> scoring $edit"
  "$PY" -m wavepainter.metrics.score_cli \
      --items "$OUT/${edit}.json" --sv_ckpt "$SV_CKPT" \
      --out "$OUT/${edit}_metrics.json"
done

"$PY" -m wavepainter.metrics.report --results "$OUT"

# Full-reference signal quality, reported alongside WER and SIM. Skipped rather
# than fatal when the optional [quality] extra is absent: `pesq` is sdist-only
# and cythonises at build time, so it does not install cleanly everywhere, and
# WER and SIM -- the axes compared against the baseline -- do not need it.
if "$PY" -c "import pesq, pystoi" 2>/dev/null; then
  echo "==> full-reference quality"
  "$PY" scripts/measure_quality.py --results "$OUT" \
      ${COPYSYN:+--copysyn "$COPYSYN"} --out "$OUT/quality.json"
else
  echo "| skipping PESQ/STOI: pip install 'Cython<3.1' && \
pip install -e '.[quality]' --no-build-isolation"
fi
