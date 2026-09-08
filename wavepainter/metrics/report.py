"""Print the comparison against Ren et al. from scored results.

    python -m wavepainter.metrics.report --results out/verify

Reads ``<results>/{sub,ins,del}_metrics.json`` as written by ``score_cli`` and
prints one row per (edit type, metric), with the published baseline beside the
measured value. Nothing is aggregated across edit types or across metrics: every
number this project reports is a single per-edit-type WER or SIM, comparable
directly to a column of the baseline paper's Table 1.

Signal quality is not in this table. It is measured full-reference by
`scripts/measure_quality.py` and has no published counterpart to compare against.
"""

import argparse
import json
import os

from wavepainter.metrics.baselines import (CITATION, REN_ET_AL_2602_00560,
                                           WAVEPAINTER)

EDIT_NAMES = {"sub": "substitution", "ins": "insertion", "del": "deletion"}
# (metric, arrow, lower_is_better, decimals)
METRICS = [("wer", "down", True, 3), ("sim", "up", False, 3)]


def load(results_dir):
    out = {}
    for edit in ("sub", "ins", "del"):
        path = os.path.join(results_dir, f"{edit}_metrics.json")
        if os.path.isfile(path):
            out[edit] = json.load(open(path))["values"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", required=True, help="directory of *_metrics.json")
    ap.add_argument("--expected", action="store_true",
                    help="compare against the released model's values too")
    args = ap.parse_args()

    measured = load(args.results)
    if not measured:
        raise SystemExit(f"no *_metrics.json found in {args.results}")

    print(f"\nBaseline: {CITATION}\n")
    header = f"{'edit':<13} {'metric':<7} {'baseline':>9} {'ours':>9} {'delta':>9}   result"
    print(header)
    print("-" * len(header))

    beat = total = 0
    for edit in ("sub", "ins", "del"):
        if edit not in measured:
            continue
        for metric, arrow, lower_better, nd in METRICS:
            ours = measured[edit].get(metric)
            base = REN_ET_AL_2602_00560[edit].get(metric)
            if ours is None or base is None:
                continue
            wins = ours < base if lower_better else ours > base
            delta = ours - base
            total += 1
            beat += wins
            mark = "BEAT" if wins else "below baseline"
            print(f"{EDIT_NAMES[edit]:<13} {metric.upper() + (' v' if lower_better else ' ^'):<7} "
                  f"{base:>9.{nd}f} {ours:>9.{nd}f} {delta:>+9.{nd}f}   {mark}")

    print(f"\n{beat} of {total} reported numbers improve on the baseline.")

    if args.expected:
        print("\nAgainst the released checkpoint's recorded values:")
        for edit in ("sub", "ins", "del"):
            if edit not in measured:
                continue
            for metric in ("wer", "sim"):
                want = WAVEPAINTER[edit].get(metric)
                got = measured[edit].get(metric)
                if want is None or got is None:
                    continue
                # The harness is deterministic, so this should be an exact match,
                # not an approximate one. A drift here means the protocol, the
                # decode path or a pinned component changed.
                status = "exact" if abs(got - want) < 5e-4 else "DRIFTED"
                print(f"  {edit} {metric}: expected {want:.3f}, got {got:.3f}  [{status}]")


if __name__ == "__main__":
    main()
