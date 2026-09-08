#!/usr/bin/env python3
"""Compare every shipped source file against the upstreams it might derive from.

    python scripts/check_provenance.py            # report
    python scripts/check_provenance.py --max 0.5  # and fail over a threshold

This repository descends from a line of speech-editing codebases. Some of that
upstream is MIT (NATSpeech, DiffSinger); some carries no licence at all
(FluentEditor). The clean-room rewrite replaces the latter with our own
implementations, and this is how that claim is checked rather than asserted.

Similarity is line-level: 1 - (differing lines / total lines) against the
best-matching upstream file at the same path. It is a smoke detector, not a
lawyer -- a low score means the expression differs, which is the thing a rewrite
is supposed to achieve. It says nothing about whether the *design* is original,
which is a citation question and is handled in the paper, not here.

Needs network access. Results are cached under the system temp dir so repeated
runs are cheap.
"""

import argparse
import difflib
import hashlib
import os
import sys
import tempfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(tempfile.gettempdir(), "wavepainter_upstream_cache")

UPSTREAMS = {
    "FluentEditor": "https://raw.githubusercontent.com/AI-S2-Lab/FluentEditor/main/{path}",
    "NATSpeech": "https://raw.githubusercontent.com/NATSpeech/NATSpeech/main/{path}",
    "DiffSinger": "https://raw.githubusercontent.com/MoonInTheRiver/DiffSinger/master/{path}",
}

SKIP_DIRS = {".git", ".venv", "__pycache__", "third_party", "out", "data",
             "checkpoints", "assets", "tests", "runs"}


# The upstream projects still carry the layout this repository was forked from:
# modules/, tasks/, utils/ and data_gen/ at the top level. 329a06a folded all
# four into wavepainter/, so a path from this tree no longer addresses anything
# upstream and every fetch 404s -- silently, because a miss is indistinguishable
# from "this file has no upstream counterpart". Paths are mapped back before the
# URL is built.
#
# Most specific first: the single-file moves have to win over the directory
# prefixes they sit inside.
_TO_UPSTREAM = tuple(sorted((
    ("wavepainter/models/layers.py",           "modules/commons/layers.py"),
    ("wavepainter/models/nar_tts_modules.py",  "modules/commons/nar_tts_modules.py"),
    ("wavepainter/models/align_ops.py",        "modules/tts/commons/align_ops.py"),
    ("wavepainter/models/mel_encoder.py",      "modules/speech_editing/commons/mel_encoder.py"),
    ("wavepainter/models/dualffn/",            "modules/speech_editing/dualffn/"),
    ("wavepainter/models/spec_denoiser/",      "modules/speech_editing/spec_denoiser/"),
    ("wavepainter/tasks/dataset_utils.py",     "tasks/speech_editing/dataset_utils.py"),
    ("wavepainter/tasks/spec_denoiser.py",     "tasks/speech_editing/spec_denoiser.py"),
    ("wavepainter/tasks/speech_editing_base.py", "tasks/speech_editing/speech_editing_base.py"),
    ("wavepainter/tasks/speech_base.py",       "tasks/tts/speech_base.py"),
    ("wavepainter/tasks/tts_utils.py",         "tasks/tts/tts_utils.py"),
    ("wavepainter/tasks/vocoder_infer/",       "tasks/tts/vocoder_infer/"),
    ("wavepainter/losses/ssim.py",             "utils/metrics/ssim.py"),
    ("wavepainter/datasets/time_mask.py",      "utils/spec_aug/time_mask.py"),
    ("wavepainter/nn/gst.py",                  "utils/feature_extractor/gst.py"),
    ("wavepainter/runtime/os_utils.py",        "utils/os_utils.py"),
    ("wavepainter/runtime/plot.py",            "utils/plot/plot.py"),
    ("wavepainter/datasets/",                  "data_gen/tts/"),
    ("wavepainter/runtime/",                   "utils/commons/"),
    ("wavepainter/audio/",                     "utils/audio/"),
    ("wavepainter/nn/",                        "utils/nn/"),
    ("wavepainter/text/",                      "utils/text/"),
), key=lambda kv: -len(kv[0])))


def upstream_path(rel):
    """Map a path in this tree onto the upstream layout it was forked from.

    Files with no entry here -- this project's own code -- are returned
    unchanged and simply find no counterpart, which is the correct answer.
    """
    for ours, theirs in _TO_UPSTREAM:
        if rel.startswith(ours):
            return theirs + rel[len(ours):]
    return rel


def fetch(url):
    """Fetch a URL, cached. Returns None for anything that is not a 200."""
    os.makedirs(CACHE, exist_ok=True)
    key = os.path.join(CACHE, hashlib.sha256(url.encode()).hexdigest())
    if os.path.exists(key):
        data = open(key, "rb").read()
        return None if data == b"__404__" else data.decode("utf-8", "replace")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            body = r.read()
    except Exception:
        body = b"__404__"
    open(key, "wb").write(body)
    return None if body == b"__404__" else body.decode("utf-8", "replace")


def similarity(a, b):
    """Fraction of lines shared, ignoring trailing whitespace."""
    a = [l.rstrip() for l in a.splitlines()]
    b = [l.rstrip() for l in b.splitlines()]
    if not a or not b:
        return 0.0
    same = sum(block.size for block in
               difflib.SequenceMatcher(None, a, b, autojunk=False)
               .get_matching_blocks())
    return same / max(len(a), len(b))


def shipped_files():
    for dirpath, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for n in sorted(names):
            if n.endswith(".py"):
                yield os.path.relpath(os.path.join(dirpath, n), ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max", type=float, default=None,
                    help="exit nonzero if any file exceeds this similarity")
    ap.add_argument("--quiet", action="store_true",
                    help="only show files above the threshold")
    args = ap.parse_args()

    rows, worst = [], 0.0
    for rel in shipped_files():
        local = open(os.path.join(ROOT, rel), encoding="utf-8").read()
        scores = {}
        for name, tmpl in UPSTREAMS.items():
            remote = fetch(tmpl.format(path=upstream_path(rel)))
            if remote is not None:
                scores[name] = similarity(remote, local)
        if scores:
            rows.append((max(scores.values()), rel, scores))
            worst = max(worst, max(scores.values()))

    rows.sort(reverse=True)
    threshold = args.max if args.max is not None else 1.1

    # The FluentEditor-vs-MIT split is the whole point of the exercise, so it is
    # reported per file rather than collapsed into a single "best match".
    # A file that scores high against BOTH is NATSpeech/DiffSinger code that
    # FluentEditor merely carried -- MIT, and legal to keep with the notice.
    # A file that scores high against FluentEditor ALONE is their own work,
    # unlicensed, and has to be rewritten.
    print(f"{'FluentEd':>9} {'NATSpeech':>10} {'DiffSinger':>11}  {'verdict':<18} file")
    print("-" * 96)
    n_over = 0
    for score, rel, scores in rows:
        fe = scores.get("FluentEditor", 0.0)
        nat = max(scores.get("NATSpeech", 0.0), scores.get("DiffSinger", 0.0))
        if score <= 0.5:
            verdict = "rewritten"
        elif nat >= 0.9:
            verdict = "MIT upstream"
        elif fe >= 0.9:
            verdict = "FE-ONLY: rewrite"
        else:
            verdict = "derived"
        if score > threshold:
            n_over += 1
        if args.quiet and score <= threshold:
            continue
        print(f"{fe:>9.1%} {scores.get('NATSpeech', 0.0):>10.1%} "
              f"{scores.get('DiffSinger', 0.0):>11.1%}  {verdict:<18} {rel}")

    print(f"\n{len(rows)} files have an upstream counterpart; "
          f"worst similarity {worst:.1%}")
    fe_only = [r for _, r, s in rows
               if s.get("FluentEditor", 0) >= 0.9
               and max(s.get("NATSpeech", 0), s.get("DiffSinger", 0)) < 0.9]
    print(f"{len(fe_only)} are FluentEditor-only (unlicensed, must be rewritten)")
    if args.max is not None:
        print(f"{n_over} exceed the {args.max:.0%} threshold")
        return 1 if n_over else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
