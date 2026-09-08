#!/usr/bin/env python3
"""Build a deletion reference: cut the deleted words out of the SOURCE waveform.

    python scripts/build_deletion_reference.py --out out/reference/gt_splice

This is the upper bound on what any masked-span editor can achieve on the
deletion protocol. It performs the edit with no model at all: the words the
protocol says to delete are removed from the original recording at their
forced-alignment boundaries, and the two halves are crossfaded. Everything
outside the cut is the untouched source signal -- no vocoder, no generated
frames, no predicted durations.

If this scores near the published target, deletion is a modelling problem and
the gap is closable. If it scores far above, the target is not reachable by
editing quality on this protocol, and the residual is a property of the task or
the recogniser rather than of any system.

The cut boundary comes from the aligner, not from the model, which is what
makes it the reference: it is the correct answer to "where does this word end".
"""

import argparse
import difflib
import json
import os
import re
import sys

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CROSSFADE_MS = 20.0


def read_word_tier(path):
    """Return [(xmin, xmax, text)] for the TextGrid's `words` tier."""
    text = open(path, encoding="utf-8").read()
    # The word tier precedes the phone tier; take everything after its header.
    start = text.find('name = "words"')
    if start < 0:
        raise ValueError(f"no words tier in {path}")
    end = text.find('name = "phones"')
    body = text[start:end if end > 0 else len(text)]

    out = []
    for m in re.finditer(
            r"xmin\s*=\s*([\d.]+)\s*xmax\s*=\s*([\d.]+)\s*text\s*=\s*\"([^\"]*)\"",
            body):
        lo, hi, word = float(m.group(1)), float(m.group(2)), m.group(3).strip()
        if word:
            out.append((lo, hi, word))
    return out


def deleted_intervals(item, words):
    """Seconds intervals covering the words the edit removes.

    The protocol gives the span's words before and after the edit. Which words
    disappear is found by aligning the two sequences, NOT by assuming the
    deletion takes a suffix: the removed word is frequently the first
    ("It was" -> "was") or an interior one ("it can also" -> "it also"), and a
    suffix assumption cuts the wrong audio while still producing a plausible
    file. That failure is silent -- it shows up only as a lower edit-accuracy
    score, which is how it was caught here.

    Returns a list of (start, end) in seconds, latest first, so a caller can
    apply them without invalidating earlier indices.
    """
    old = item["old_words"].split()
    new = item["new_words"].split()

    lo_s, hi_s = float(item["start_s"]), float(item["end_s"])
    inside = [w for w in words if w[0] >= lo_s - 0.05 and w[1] <= hi_s + 0.05]
    if len(inside) < len(old):
        lowered = [o.lower() for o in old]
        inside = [w for w in words if w[2].lower() in lowered]
    if len(inside) < len(old):
        return []

    # Align old -> new; every old index not carried into new is deleted.
    matcher = difflib.SequenceMatcher(
        None, [w.lower() for w in old], [w.lower() for w in new], autojunk=False)
    removed_idx = []
    for tag, i1, i2, _, _ in matcher.get_opcodes():
        if tag in ("delete", "replace"):
            removed_idx.extend(range(i1, i2))
    if not removed_idx:
        return []

    spans, run = [], []
    for i in removed_idx:
        if i >= len(inside):
            continue
        if run and i == run[-1] + 1:
            run.append(i)
        else:
            if run:
                spans.append((inside[run[0]][0], inside[run[-1]][1]))
            run = [i]
    if run:
        spans.append((inside[run[0]][0], inside[run[-1]][1]))
    return sorted(spans, reverse=True)


def cut(wav, sr, t0, t1, fade_ms=CROSSFADE_MS):
    """Remove [t0, t1] and crossfade the join to avoid a click."""
    i0, i1 = int(round(t0 * sr)), int(round(t1 * sr))
    i0 = max(0, min(i0, len(wav)))
    i1 = max(i0, min(i1, len(wav)))
    fade = int(round(fade_ms / 1000.0 * sr))
    fade = min(fade, i0, len(wav) - i1)
    if fade <= 0:
        return np.concatenate([wav[:i0], wav[i1:]])

    head, tail = wav[:i0].copy(), wav[i1:].copy()
    ramp = np.linspace(0.0, 1.0, fade, dtype=wav.dtype)
    # Equal-power crossfade over the last `fade` samples of the head and the
    # first `fade` of the tail; linear amplitude would dip in the middle.
    blend = head[-fade:] * np.sqrt(1 - ramp) + tail[:fade] * np.sqrt(ramp)
    return np.concatenate([head[:-fade], blend, tail[fade:]])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="out/reference/gt_splice")
    ap.add_argument("--protocol", default=os.path.join(
        ROOT, "assets", "protocols", "protocol_ming_en_del.json"))
    ap.add_argument("--alignments", default=os.path.join(ROOT, "assets", "alignments"))
    ap.add_argument("--benchmark", default=os.environ.get(
        "WAVEPAINTER_BENCHMARK", os.path.join(ROOT, "data", "ming")))
    args = ap.parse_args()

    from wavepainter.evaluate import benchmark_path
    from wavepainter.metrics.baselines import verify_protocol

    items = verify_protocol(args.protocol, "del")
    os.makedirs(args.out, exist_ok=True)

    rows, skipped = [], []
    for item in items:
        name = item["item_name"]
        tg = os.path.join(args.alignments, name + ".TextGrid")
        src = benchmark_path(item["wav"])
        if not (os.path.exists(tg) and os.path.exists(src)):
            skipped.append((name, "missing input")); continue

        spans = deleted_intervals(item, read_word_tier(tg))
        if not spans:
            skipped.append((name, "no deletable span")); continue

        wav, sr = sf.read(src)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        edited = wav
        for t0, t1 in spans:            # latest first, so indices stay valid
            edited = cut(edited, sr, t0, t1)

        dest = os.path.join(args.out, name + ".wav")
        sf.write(dest, edited, sr)
        rows.append({"item_name": name, "wav": dest, "original_wav": src,
                     "original_text": item["original_text"],
                     "edited_text": item["edited_text"], "edit_type": "del"})

    json.dump({"items": rows}, open(os.path.join(args.out, "items.json"), "w"))
    print(f"| built {len(rows)}/{len(items)} reference items in {args.out}")
    if skipped:
        print(f"| skipped {len(skipped)}: {skipped[:5]}")
    print(f"| score with: python -m wavepainter.metrics.score_cli "
          f"--items {args.out}/items.json --sv_ckpt <path> --out <path>")


if __name__ == "__main__":
    main()
