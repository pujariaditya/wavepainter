#!/usr/bin/env python3
"""Measure full-reference quality (PESQ, STOI) for an evaluation run.

    python scripts/measure_quality.py --results out/verify

Reports two things per edit type:

  vocoder ceiling   original vs copy-synthesis of the same audio, no edit at
                    all. The best any system decoding through this vocoder can
                    reach on a full-reference metric.

  preservation      original vs system output over the UNEDITED audio -- the
                    region before the edit and the region after it. A
                    masked-span editor is supposed to leave both untouched, so
                    this measures the property that separates it from full
                    resynthesis.

Requires the optional quality extra, which needs a build flag:

    pip install "Cython<3.1"
    pip install -e ".[quality]" --no-build-isolation

Cython 3.1 dropped numpy 1.x support in its bindings, and pesq cythonises at
build time, so a newer Cython emits numpy-2-only symbols against the numpy<2
this stack pins.
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", required=True,
                    help="directory holding <edit>.json from wavepainter.evaluate")
    ap.add_argument("--benchmark", default=os.environ.get(
        "WAVEPAINTER_BENCHMARK", os.path.join(ROOT, "data", "ming")))
    ap.add_argument("--copysyn", default=None,
                    help="directory of copy-synthesis wavs, for the ceiling")
    ap.add_argument("--out", default=None, help="write a JSON summary here")
    args = ap.parse_args()

    import soundfile as sf

    from wavepainter.metrics.baselines import verify_protocol
    from wavepainter.metrics.quality import preservation, vocoder_ceiling

    summary = {}

    if args.copysyn:
        if not os.path.isdir(args.copysyn):
            raise SystemExit(f"--copysyn is not a directory: {args.copysyn}")
        import glob

        synths = sorted(glob.glob(os.path.join(args.copysyn, "*.wav")))
        pairs, missing = [], []
        for synth in synths:
            for src in (os.path.join(args.benchmark, "benchmark", "wavs",
                                     os.path.basename(synth)),
                        os.path.join(args.benchmark, "wavs",
                                     os.path.basename(synth))):
                if os.path.exists(src):
                    pairs.append((src, synth))
                    break
            else:
                missing.append(os.path.basename(synth))
        # Loudly, not silently. Asking for the ceiling and getting nothing back
        # is the failure this guards: --copysyn resolves against --benchmark,
        # which defaults to a path that need not exist, so an unset
        # WAVEPAINTER_BENCHMARK previously produced zero pairs, no ceiling row
        # and no complaint -- a missing reference point that reads as an absent
        # measurement rather than a broken one.
        if not pairs:
            raise SystemExit(
                f"--copysyn matched {len(synths)} wavs but none resolved to a "
                f"source under {args.benchmark}. Set WAVEPAINTER_BENCHMARK or "
                f"pass --benchmark."
            )
        if missing:
            print(f"| warning: {len(missing)} copy-synthesis wavs have no source "
                  f"under {args.benchmark}, e.g. {missing[:2]}")
        summary["vocoder_ceiling"] = vocoder_ceiling(pairs)
        c = summary["vocoder_ceiling"]
        print(f"vocoder ceiling (no edit): PESQ {c['pesq']:.3f}  "
              f"STOI {c['stoi']:.4f}  n={c['n']}")

    print(f"\n{'edit':<6}{'scored':>8}{'skip':>6}{'PESQ':>9}{'STOI':>9}")
    for edit in ("sub", "ins", "del"):
        record = os.path.join(args.results, f"{edit}.json")
        if not os.path.exists(record):
            continue
        proto = {i["item_name"]: i for i in verify_protocol(
            os.path.join(ROOT, "assets", "protocols",
                         f"protocol_ming_en_{edit}.json"), edit)}
        data = json.load(open(record))
        rows = data if isinstance(data, list) else data.get("items", data.get("rows"))

        pairs = []
        for row in rows:
            item = proto.get(row["item_name"])
            if not item or not os.path.exists(row["wav"]):
                continue
            info = sf.info(row["original_wav"])
            pairs.append((row["original_wav"], row["wav"],
                          float(item["start_s"]), float(item["end_s"]),
                          info.frames / info.samplerate))
        res = preservation(pairs)
        summary[edit] = res
        print(f"{edit:<6}{res['n']:>8}{res['n_skipped']:>6}"
              f"{res['pesq']:>9.3f}{res['stoi']:>9.4f}")

    if args.out:
        json.dump(summary, open(args.out, "w"), indent=1)
        print(f"\n| wrote {args.out}")


if __name__ == "__main__":
    main()
