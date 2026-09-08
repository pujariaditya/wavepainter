#!/usr/bin/env python3
"""Publish the two released checkpoints to Hugging Face.

    HF_TOKEN=... python scripts/upload_weights.py \\
        --phase1 /path/to/phase1-base --phase2 /path/to/phase2-hubert-a085

Maintainer tool; end users want ``download_weights.py``.

THE TOKEN IS READ FROM THE ENVIRONMENT ONLY. It is never written to a config, a
committed file, or the upload itself. ``huggingface-cli login`` would persist it
to ``~/.cache/huggingface/token``, which is world-readable by default -- pass it
per-invocation instead.

Every file is hashed before upload and the digest printed. Those are the values
that belong in ``download_weights.py``; the point of pinning them there is that a
reader verifies against a constant in the source, never against a digest served
from the same place as the file.
"""

import argparse
import hashlib
import os
import sys

REPO = os.environ.get("WAVEPAINTER_HF_REPO", "RootAccess4Life/wavepainter")

CARD = """---
license: mit
library_name: pytorch
pipeline_tag: audio-to-audio
tags:
  - speech-editing
  - speech-synthesis
  - text-to-speech
  - diffusion
datasets:
  - inclusionAI/Ming-Freeform-Audio-Edit-Benchmark
metrics:
  - wer
---

# wavepainter

**Multimodal LLM-Guided Diffusion for Text-Based Speech Editing**

Released checkpoints for [wavepainter](https://github.com/pujariaditya/wavepainter),
a masked-span speech editor. Edit a word in the transcript and only that span of
the spectrogram is re-predicted; substitution, insertion and deletion come from
one checkpoint and one decode path.

`phase2-hubert-child` is the released model and the artifact of record: phase two
as trained, and the checkpoint every number in the paper comes from. On the
English `full` split of Ming-Freeform-Audio-Edit, against Ren et al.
([arXiv:2602.00560](https://arxiv.org/abs/2602.00560), the row with GRPO), scored
with edit spans taken from the transcript diff:

| edit type | metric | Ren et al. | wavepainter |
|---|---|---|---|
| substitution | WER | 4.41 | **3.479** |
| insertion | WER | 4.97 | **4.400** |
| deletion | WER | **6.88** | 9.697 |
| substitution | SIM | 0.78 | **0.943** |
| insertion | SIM | 0.82 | **0.961** |
| deletion | SIM | 0.78 | **0.918** |

Five of the six compared numbers improve on theirs: substitution and insertion
WER, and speaker similarity on all three edit types. We do not beat them on
deletion WER -- see Limitations below, which also locates most of that gap.

Signal quality is reported full-reference, since a no-reference MOS estimator
rates the untouched original recordings below synthesised output on this
benchmark. Over the audio the edit does not touch, PESQ is 4.090 / 4.083 / 4.079
for substitution / insertion / deletion, against 4.351 for copy-synthesis with no
edit at all.

## Model

A frame-level conditioning trunk over interleaved text and mel frames, whose
geometry and initial weights come from Qwen2.5-0.5B, driving a diffusion
spectrogram denoiser. Every conditioning signal into the audio stream is
additive, which is what makes the inference-time ablations exact.

| | |
|---|---|
| Trainable parameters | 17.4 M |
| Trunk | DualFFN, 14 layers, warm-started from Qwen2.5-0.5B |
| Decoder | diffusion denoiser, 8 timesteps, L1 |
| Hidden size | 192 |
| Vocoder | BigVGAN v2 (pinned revision) |
| Sample rate | 24 kHz |

Training is two phases. Phase 1 produces the base editor. Phase 2 is the
contribution: a frozen HuBERT encoder supplies a perceptual feature loss over a
matched crop centred on the edit, applied to a denoiser-only trainable surface --
1500 updates at lr 2e-5, seed 1234. Re-running it costs about 1 GPU-hour from
the published phase-1 base.

Outside the edit the source recording's own mel is carried over unchanged, but
the whole utterance is re-vocoded, so the output matches the source
*spectrogram* there rather than its samples. The vocoder pass costs about as
much as the edit itself, which is what the PESQ numbers above measure.

## Checkpoints

Two earlier releases interpolate the phase-two child back toward the phase-1 base,
which moves the diffusion denoiser only. They are published because the paper's
first version reported them and anyone who fetched them should keep working; the
paper no longer reports either.

| | substitution | insertion | deletion |
|---|---|---|---|
| `phase2-hubert-child` (released) | 3.479 | 4.400 | 9.697 |
| `phase2-hubert-a085` | 3.063 | 4.128 | 9.136 |
| `phase2-hubert-a070` | 3.381 | 3.817 | 9.525 |

WER, lower is better. Nothing is bolded because the released checkpoint is not
the best row: interpolating toward the phase-1 base improves every WER column,
a=0.85 most on substitution and deletion and a=0.70 on insertion. We publish the
child as the artifact of record anyway, because it is the model phase two
produced and the coefficient was chosen by looking at these scores. The
coefficient also has no single optimum across edit types, which is why both
variants are kept rather than one being presented as strictly better. Speaker
similarity is 0.943 / 0.961 / 0.918 for all three except insertion at a=0.70,
which is 0.960.

## Files

- `phase1-base/` -- the phase-1 editor. Start here to re-run phase 2 (~1 GPU-hour).
- `phase2-hubert-child/` -- the released model, and the artifact of record.
- `phase2-hubert-a085/` -- an earlier release, interpolated at 0.85.
- `phase2-hubert-a070/` -- the same, at 0.70.

Each directory ships the checkpoint and the `config.yaml` it was trained under.

## Use

```bash
git clone https://github.com/pujariaditya/wavepainter && cd wavepainter
pip install -e . && ./setup.sh
python scripts/download_weights.py     # phase-1 base + released model, ~4.5 GB
./scripts/verify_benchmark.sh          # reproduces the table above, ~50 min on one RTX 8000
```

`download_weights.py` fetches the base and the released model by default. The
interpolated variants are opt-in: `--phase 3` for a=0.70, `--phase 4` for a=0.85.
Every checkpoint is verified by sha256 against a digest compiled into that
script rather than served alongside the file.

To hear one edit instead of scoring the benchmark, `./scripts/demo_edit.sh`
edits a single bundled item and writes a wav. It needs a CUDA device.

## Intended use and limitations

Research artifact. It reproduces a published result on one benchmark, and it is
not a general-purpose editing tool: nothing in the repository accepts your own
audio, and `demo_edit.sh` is a demonstration on a bundled item rather than a CLI.

**Deletion is the weak edit type.** On deletion WER the model is behind two
systems, FluentSpeech as well as Ren et al. A ground-truth waveform splice
isolates where the gap lives: 2.11 of the 2.82 points are modelling headroom
rather than an artefact of the protocol
([detail](https://github.com/pujariaditya/wavepainter/blob/main/docs/RESULTS.md#4-deletion-reference)).
Deletion also looks worse than it is on a null-WER basis, because the reference
text is shorter -- scoring silence against it gives 27.97 for deletion against
15.05 for substitution
([detail](https://github.com/pujariaditya/wavepainter/blob/main/docs/RESULTS.md#7-deletion-error-composition)).

Results are quoted against Ren et al.'s reported table rather than re-scored by
us. Both sides use Whisper for WER and WavLM embeddings for similarity on an
identical, fingerprint-verified item set, but the exact scorer variants on their
side are unstated, so the comparison is quoted rather than scorer-identical.

English only, 24 kHz, and evaluated on read speech. Nothing here has been
measured on spontaneous or noisy audio.

Speech editing can be used to put words a speaker never said into their own
voice. These weights are released for reproducing the reported results. Do not
use them to produce audio attributed to a real person without that person's
consent.

## Citation

```bibtex
@software{pujari_wavepainter_2026,
  author  = {Pujari, Aditya},
  title   = {{wavepainter}: Multimodal LLM-Guided Diffusion for Text-Based Speech Editing},
  year    = {2026},
  version = {1.0.0},
  license = {MIT},
  url     = {https://github.com/pujariaditya/wavepainter}
}
```

Checkpoints are sha256-pinned in `scripts/download_weights.py`.
"""


def sha256_of(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # All optional: a re-upload of one artifact should not require re-hashing
    # and re-pushing 4.5 GB of unchanged checkpoints.
    ap.add_argument("--phase1", default=None, help="dir holding the phase-1 ckpt + config")
    ap.add_argument("--phase2", default=None, help="dir holding the released ckpt + config")
    ap.add_argument("--phase2-alt", default=None,
                    help="dir holding the alpha=0.70 variant, if publishing it")
    ap.add_argument("--private", action="store_true", help="create the repo private")
    ap.add_argument("--card-only", action="store_true",
                    help="refresh the model card and upload no checkpoints")
    ap.add_argument("--dry-run", action="store_true",
                    help="hash and report, upload nothing")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token and not args.dry_run:
        raise SystemExit(
            "set HF_TOKEN in the environment (do not use `huggingface-cli login`, "
            "which writes a world-readable token file)"
        )

    plan = []
    # The remote directory is taken from the local one. Hardcoding it meant
    # --phase2 always published into "phase2-hubert-a085/", so a different
    # checkpoint landed inside the previous release's folder under a name that
    # described neither. Publishing a new checkpoint should not require editing
    # this list.
    targets = [(os.path.basename(os.path.normpath(d)), d)
               for d in (args.phase1, args.phase2, args.phase2_alt) if d]
    # The card is text describing artifacts that are already published; a
    # correction to it should not require re-hashing and re-pushing 4.5 GB of
    # unchanged checkpoints just to satisfy the "something to upload" check.
    if not targets and not args.card_only:
        raise SystemExit("nothing to upload: pass at least one of "
                         "--phase1 / --phase2 / --phase2-alt, or --card-only")
    if targets and args.card_only:
        raise SystemExit("--card-only uploads no checkpoints; drop the phase "
                         "arguments or drop --card-only")
    for prefix, local in targets:
        if not os.path.isdir(local):
            raise SystemExit(f"not a directory: {local}")
        for name in sorted(os.listdir(local)):
            if not name.endswith((".ckpt", ".yaml")):
                continue
            path = os.path.join(local, name)
            digest = sha256_of(path)
            size = os.path.getsize(path) / 1e9
            print(f"| {prefix}/{name}  {size:.2f} GB  sha256 {digest}")
            plan.append((path, f"{prefix}/{name}"))

    if args.dry_run:
        if args.card_only:
            print(CARD)
        print("| dry run: nothing uploaded")
        return 0

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(REPO, repo_type="model", private=args.private, exist_ok=True)

    for path, remote in plan:
        print(f"| uploading {remote}")
        api.upload_file(path_or_fileobj=path, path_in_repo=remote,
                        repo_id=REPO, repo_type="model")

    # The card describes every artifact, so refresh it whenever anything is
    # published -- a partial upload still changes what the repo contains.
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                    repo_id=REPO, repo_type="model")
    print(f"| done: https://huggingface.co/{REPO}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
