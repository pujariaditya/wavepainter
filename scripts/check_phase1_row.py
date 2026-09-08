#!/usr/bin/env python3
"""Check the phase-1 ablation row against a regenerated scoring pass.

    EXP=phase1-base STEPS=5000 OUT=out/phase1 ./scripts/verify_benchmark.sh
    python scripts/check_phase1_row.py --results out/phase1

Table 2 of the paper reports the phase-1 base as 4.019 / 4.465 / 10.248 WER.
Those three numbers live as a Python literal in
``wavepainter/metrics/baselines.py:WAVEPAINTER_PHASE1`` and nothing in this
repository connected them to a file until now.

This script is the connection, and it is deliberately one-directional: it reads
the literal, reads the regenerated metrics, and **fails if they differ**. It has
no flag to update the literal. If a rescoring disagrees with the published row,
that is a finding to report, not a number to overwrite -- the same rule the rest
of this repository follows for measured values.

Exit codes: 0 all rows match, 1 a row differs, 2 the results are not there yet.
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

EDITS = ("sub", "ins", "del")
METRICS = ("wer", "sim")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default="out/phase1",
                    help="directory holding {sub,ins,del}_metrics.json")
    a = ap.parse_args()

    from wavepainter.metrics.baselines import WAVEPAINTER_PHASE1

    missing = [e for e in EDITS
               if not os.path.exists(os.path.join(a.results, f"{e}_metrics.json"))]
    if missing:
        print(f"| no regenerated metrics for {', '.join(missing)} in {a.results}")
        print("| the published row is still a compiled-in literal:")
        for e in EDITS:
            print(f"|   {e}: WER {WAVEPAINTER_PHASE1[e]['wer']}  "
                  f"SIM {WAVEPAINTER_PHASE1[e]['sim']}")
        print("| regenerate with:")
        print("|   EXP=phase1-base STEPS=5000 OUT=out/phase1 ./scripts/verify_benchmark.sh")
        return 2

    bad = []
    for e in EDITS:
        got = json.load(open(os.path.join(a.results, f"{e}_metrics.json")))["values"]
        for m in METRICS:
            want = float(WAVEPAINTER_PHASE1[e][m])
            # Exact at the precision the harness writes. The scorer is
            # deterministic and the sampler is seeded per item, so "close" is
            # not the standard here -- a drift of 0.001 means something moved.
            if abs(float(got[m]) - want) > 1e-9:
                bad.append(f"{e}.{m}: regenerated {got[m]} vs published {want}")
            else:
                print(f"| {e}.{m}: {want} reproduces")

    if bad:
        print("\nPHASE-1 ROW DOES NOT REPRODUCE:", file=sys.stderr)
        for b in bad:
            print(f"  {b}", file=sys.stderr)
        print("\nReport this. Do not edit wavepainter/metrics/baselines.py to "
              "match: the literal is the published record, and a rescoring that "
              "disagrees with it is a result about the scoring path.",
              file=sys.stderr)
        return 1

    print("| phase-1 row reproduces exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
