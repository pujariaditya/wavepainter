#!/usr/bin/env bash
# Fetch everything wavepainter needs but does not redistribute.
#
# The editor itself is in this repository -- nothing is cloned. What this fetches
# is the speaker-verification model used by the SIM metric, plus the pretrained
# components the model warm-starts from and decodes with.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python}"

echo "==> speaker-verification sources and checkpoint (SIM metric)"
"$PY" scripts/fetch_sim_model.py

echo "==> pretrained components (Qwen2.5-0.5B trunk donor, BigVGAN v2, HuBERT)"
"$PY" - <<'PYEOF'
from huggingface_hub import snapshot_download
for repo, kw in (
    ("Qwen/Qwen2.5-0.5B", {}),
    # Pinned: the vocoder is part of the measuring instrument.
    ("nvidia/bigvgan_v2_24khz_100band_256x",
     {"revision": "c329ede9e9bbc100ddf5c91e2330a61921262370"}),
    ("facebook/hubert-base-ls960", {}),
):
    print(f"|   {repo}")
    snapshot_download(repo, **kw)
PYEOF

echo
echo "Setup complete. Next:"
echo "  python scripts/download_weights.py   # released checkpoints (~4.4 GB)"
echo "  python scripts/fetch_benchmark.py    # Ming-Freeform-Audio-Edit"
echo "  ./scripts/verify_benchmark.sh        # reproduce the reported table"
