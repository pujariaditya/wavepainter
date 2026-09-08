#!/usr/bin/env python3
"""Per-item speaker similarity, so SIM can be bucketed like WER.

`score_cli` keeps only the mean and std, so the per-item values are recomputed
here with the same model, the same reader and the same pairing the harness uses.
The mean is checked against the run's published SIM before anything is written.
"""
import argparse, json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wavepainter.metrics import ming


def per_item(items, ckpt):
    import torch
    import torch.nn.functional as F
    from wavepainter.metrics.sim_model import load_sv_model

    model = load_sv_model(ckpt, device="cuda")

    def emb(path):
        if not os.path.isabs(path):
            path = os.path.join(ROOT, path)
        w = ming._read(path, 16000)
        with torch.no_grad():
            return model(torch.tensor(w, dtype=torch.float32)[None].cuda())

    out = []
    for i, it in enumerate(items):
        out.append(float(F.cosine_similarity(
            emb(it["wav"]), emb(it["original_wav"])).item()))
        if i % 50 == 0:
            print(f"|   sim {i}/{len(items)}", flush=True)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--sv_ckpt", default="third_party/wavlm_large_finetune.pth")
    a = ap.parse_args()

    for edit in ("sub", "ins", "del"):
        items = json.load(open(f"{a.run}/{edit}.json"))["items"]
        met = json.load(open(f"{a.run}/{edit}_metrics.json"))
        s = per_item(items, os.path.join(ROOT, a.sv_ckpt))
        got, want = round(float(s.mean()), 3), met["values"]["sim"]
        if abs(got - want) > 5e-4:
            raise SystemExit(f"{edit}: per-item SIM mean {got} != reported {want}")
        path = f"{a.run}/{edit}_sim_peritem.json"
        json.dump({"item_names": [it["item_name"] for it in items],
                   "sim": [round(float(v), 6) for v in s],
                   "mean": got}, open(path, "w"), indent=1)
        print(f"| {edit}: SIM {got} matches reported, wrote {path}", flush=True)


if __name__ == "__main__":
    main()
