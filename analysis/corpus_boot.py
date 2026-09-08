#!/usr/bin/env python3
"""Corpus-level paired bootstrap on WER between two runs.

Per-item WER on this benchmark is heavily zero-inflated (190 of 256 substitution
items are error-free), so the mean of per-item rates has a very wide sampling
distribution and is badly powered for a half-point difference. Corpus WER, total
errors over total reference words, is the lower-variance estimator of the same
quantity. This resamples ITEMS with replacement and recomputes that ratio, so
the pairing and the item-level dependence are both preserved.
"""
import argparse, json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wavepainter.metrics.ming import wer_one


def counts(run, edit):
    """Per-item (errors, ref_words), in protocol order."""
    items = json.load(open(f"{run}/{edit}.json"))["items"]
    hyps = json.load(open(f"{run}/{edit}_metrics.json"))["hyps"]
    e, n = [], []
    for it, h in zip(items, hyps):
        _, err, words = wer_one(it["edited_text"], h)
        e.append(err); n.append(words)
    return np.array(e, float), np.array(n, float), [i["item_name"] for i in items]


def paired(a_run, b_run, edit, nboot=10000, seed=0):
    ea, na, names_a = counts(a_run, edit)
    eb, nb, names_b = counts(b_run, edit)
    if names_a != names_b:
        raise SystemExit(f"{edit}: item order differs")
    wa, wb = 100 * ea.sum() / na.sum(), 100 * eb.sum() / nb.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ea), size=(nboot, len(ea)))
    da = 100 * ea[idx].sum(1) / na[idx].sum(1)
    db = 100 * eb[idx].sum(1) / nb[idx].sum(1)
    d = da - db
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"a": round(wa, 3), "b": round(wb, 3), "delta": round(float(wa - wb), 3),
            "ci95": [round(float(lo), 3), round(float(hi), 3)],
            "p_improves": round(float((d < 0).mean()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="the run being claimed for")
    ap.add_argument("--b", required=True, help="the run compared against")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rep = {"a": args.a, "b": args.b, "edits": {}}
    for edit in ("sub", "ins", "del"):
        r = paired(args.a, args.b, edit)
        rep["edits"][edit] = r
        star = "separated" if r["ci95"][1] < 0 else "not separated"
        print(f"| {edit}: {r['a']} vs {r['b']}  delta {r['delta']:+.3f} "
              f"CI {r['ci95']}  P(improves)={r['p_improves']}  [{star}]")
    if args.out:
        json.dump(rep, open(args.out, "w"), indent=1)
        print(f"| wrote {args.out}")


if __name__ == "__main__":
    main()
