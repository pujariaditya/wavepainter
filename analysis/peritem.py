#!/usr/bin/env python3
"""Per-item WER, bootstrap CIs and edit-span-duration buckets.

Everything here is recomputed from files already on disk: the `hyps` array in
`*_metrics.json`, the `edited_text` in `*.json`, and the `start_s`/`end_s` in the
frozen protocol. No GPU, no synthesis. The first thing it does is reproduce the
run's published headline WER from the per-item values, so a mismatch is a hard
error rather than a silently different number.
"""
import argparse, json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wavepainter.metrics.ming import wer_one

EDITS = ("sub", "ins", "del")


def load_run(run_dir, edit):
    items = json.load(open(f"{run_dir}/{edit}.json"))["items"]
    met = json.load(open(f"{run_dir}/{edit}_metrics.json"))
    return items, met


def per_item_wer(items, hyps):
    """Per-item WER in percent, in protocol order."""
    return np.array([wer_one(it["edited_text"], h)[0] * 100.0
                     for it, h in zip(items, hyps)])


def spans(edit):
    """Edit-span duration in seconds, keyed by item name."""
    p = json.load(open(f"{ROOT}/assets/protocols/protocol_ming_en_{edit}.json"))
    return {i["item_name"]: i["end_s"] - i["start_s"] for i in p["items"]}


def boot_ci(x, n=10000, seed=0, alpha=0.05):
    """Percentile bootstrap CI on the mean of x."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_boot(a, b, n=10000, seed=0):
    """Paired bootstrap on the mean difference a-b, plus P(a<b)."""
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    dm = d[idx].mean(axis=1)
    lo, hi = np.percentile(dm, [2.5, 97.5])
    return {"delta": float(d.mean()), "ci95": [float(lo), float(hi)],
            "p_improves": float((dm < 0).mean())}


def bucket_by_duration(names, wers, dur, edges, sims=None):
    """WER and SIM by edit-span duration bucket, Ren et al.'s Table 2 axis."""
    d = np.array([dur[n] for n in names])
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi)
        if not m.any():
            rows.append({"lo": lo, "hi": hi, "n": 0})
            continue
        w = wers[m]
        ci = boot_ci(w)
        row = {"lo": lo, "hi": hi, "n": int(m.sum()),
               "wer": round(float(w.mean()), 3),
               "ci95": [round(ci[0], 3), round(ci[1], 3)]}
        if sims is not None:
            row["sim"] = round(float(sims[m].mean()), 4)
        rows.append(row)
    return rows, d


def load_sims(run_dir, edit, names):
    """Per-item SIM written by analysis/peritem_sim.py, in protocol order."""
    path = f"{run_dir}/{edit}_sim_peritem.json"
    if not os.path.isfile(path):
        return None
    blob = json.load(open(path))
    if blob["item_names"] != names:
        raise SystemExit(f"{edit}: sim item order differs from the protocol")
    return np.array(blob["sim"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="results dir, e.g. out/child_verify")
    ap.add_argument("--baseline", default=None, help="dir to pair against, e.g. out/phase1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    edges = [0.0, 0.4, 0.6, 0.8, 1.2, 99.0]
    report = {"run": a.run, "baseline": a.baseline, "edits": {}}

    for edit in EDITS:
        items, met = load_run(a.run, edit)
        names = [it["item_name"] for it in items]
        w = per_item_wer(items, met["hyps"])

        # Reproduce the published headline before reporting anything derived.
        got, want = round(float(w.mean()), 3), met["values"]["wer"]
        if abs(got - want) > 5e-3:
            raise SystemExit(
                f"{edit}: per-item mean {got} != reported {want}; "
                "the per-item recomputation does not match the harness")

        ci = boot_ci(w)
        dur = spans(edit)
        sims = load_sims(a.run, edit, names)
        rows, d = bucket_by_duration(names, w, dur, edges, sims)
        entry = {
            "n": len(w),
            "wer": got,
            "wer_ci95": [round(ci[0], 3), round(ci[1], 3)],
            "span_s": {"mean": round(float(d.mean()), 3),
                       "p25": round(float(np.percentile(d, 25)), 3),
                       "p50": round(float(np.percentile(d, 50)), 3),
                       "p75": round(float(np.percentile(d, 75)), 3)},
            "by_duration": rows,
            "n_zero_error_items": int((w == 0).sum()),
        }

        if a.baseline:
            bitems, bmet = load_run(a.baseline, edit)
            bnames = [it["item_name"] for it in bitems]
            if bnames != names:
                raise SystemExit(f"{edit}: item order differs between runs")
            bw = per_item_wer(bitems, bmet["hyps"])
            entry["baseline_wer"] = round(float(bw.mean()), 3)
            entry["paired"] = paired_boot(w, bw)

        report["edits"][edit] = entry
        print(f"| {edit}: WER {got} CI {entry['wer_ci95']} n={len(w)}", flush=True)
        if a.baseline:
            p = entry["paired"]
            print(f"|   vs baseline {entry['baseline_wer']}: "
                  f"delta {p['delta']:+.3f} CI {p['ci95']} "
                  f"P(improves)={p['p_improves']:.3f}", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)
    print(f"| wrote {a.out}")


if __name__ == "__main__":
    main()
