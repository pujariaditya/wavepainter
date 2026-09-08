#!/usr/bin/env bash
# Edit ONE bundled benchmark item, end to end, and write the result to a wav.
#
# This is a demo of the released checkpoint on a single item, NOT a general
# editing CLI -- this repository does not ship one, and nothing here accepts
# your own audio or your own text.
#
# It needs no aligner install and no full benchmark download. All 655 forced
# alignments ship in assets/alignments/ -- one TextGrid per protocol item across
# all three edit types, tracked in git, 4.8 MB -- and `wavepainter.evaluate`
# defaults --textgrids there. Only this item's source wav is fetched, a few
# hundred kB, from the benchmark's own dataset repository.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
EXP="${EXP:-phase2-hubert-child}"
STEPS="${STEPS:-1500}"
OUT="${OUT:-out/demo}"
ITEM=common_voice_en_23733663   # first item of the frozen substitution protocol

echo "==> item        $ITEM"
echo "==> original    \"...displaying his printing equipment\""
echo "==> edited to   \"...displaying his artistic tools\""

# wavepainter.evaluate runs the diffusion sampler on the GPU. Check that here,
# so a CPU-only machine gets one readable line instead of a stack trace from
# deep inside torch. This is also why hosted CI cannot execute this script end
# to end -- no runner has a CUDA device.
if ! "$PY" -c "import torch" 2>/dev/null; then
  echo "torch is not importable -- run 'pip install -e .' first." >&2
  exit 1
fi
if ! "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "this demo needs a CUDA device: wavepainter.evaluate samples on the GPU." >&2
  echo "the rest of the quick start -- install and the tests -- runs on CPU." >&2
  exit 1
fi

if [ ! -d "checkpoints/$EXP" ]; then
  echo "missing checkpoints/$EXP -- fetching just the released model" >&2
  "$PY" scripts/download_weights.py --phase 2
fi

mkdir -p "$OUT/wavs"
if [ ! -f "$OUT/wavs/$ITEM.wav" ]; then
  echo "==> fetching this one source wav"
  "$PY" - "$ITEM" "$OUT/wavs" <<'PYEOF'
import shutil, sys
from huggingface_hub import hf_hub_download
item, dest = sys.argv[1], sys.argv[2]
p = hf_hub_download("inclusionAI/Ming-Freeform-Audio-Edit-Benchmark",
                    f"wavs/{item}.wav", repo_type="dataset")
shutil.copy(p, f"{dest}/{item}.wav")
print(f"| {dest}/{item}.wav")
PYEOF
fi

export WAVEPAINTER_BENCHMARK="$PWD/$OUT"

# --limit 1 takes the first protocol item, which is $ITEM. The fingerprint check
# still runs against the complete frozen list, so a modified protocol is refused
# here exactly as it is in the full evaluation.
"$PY" -m wavepainter.evaluate \
    --exp_name "$EXP" --ckpt_steps "$STEPS" --edit_type sub --limit 1 \
    --skip_arch --wav_dir "$OUT/edited" --out "$OUT/demo.json"

# Check the demo actually produced audio, rather than announcing that it did.
"$PY" - "$OUT/edited/$ITEM.wav" "$OUT/wavs/$ITEM.wav" <<'PYEOF'
import sys, wave
edited, source = sys.argv[1], sys.argv[2]
with wave.open(edited) as e, wave.open(source) as s:
    ed, sd = e.getnframes() / e.getframerate(), s.getnframes() / s.getframerate()
    print(f"| edited  {ed:5.2f}s  {e.getframerate()} Hz  {e.getnframes()} frames")
    print(f"| source  {sd:5.2f}s  {s.getframerate()} Hz  {s.getnframes()} frames")
    assert e.getnframes() > 0, "the demo wrote an empty wav"
    e.setpos(0); s.setpos(0)
    same = e.readframes(e.getnframes()) == s.readframes(s.getnframes())
    assert not same, "the edited wav is byte-identical to the source"
print("| ok: non-empty, and different from the source")
PYEOF

echo
echo "==> edited audio: $OUT/edited/$ITEM.wav"
echo "==> source audio: $OUT/wavs/$ITEM.wav"
echo "    Outside \"printing equipment\" the source spectrogram is carried over"
echo "    unmodified; the whole utterance is then re-vocoded (see docs/RESULTS.md #5)."
echo
echo "To score the full benchmark instead: ./scripts/verify_benchmark.sh"
