#!/usr/bin/env python3
"""Fetch Ming-Freeform-Audio-Edit and lay it out for the evaluator.

    python scripts/fetch_benchmark.py [--dest data/ming]

The benchmark is Apache-2.0 and is fetched from its own repository; this project
redistributes none of it. What *is* shipped, in `assets/protocols/`, is the
frozen protocol: the exact item list, edit spans and edited transcripts the
reported numbers were produced against. Those are ours -- built from the
benchmark, verified by fingerprint at scoring time -- and they are small.

The forced alignments you need for evaluation already ship, in
`assets/alignments/` -- 655 TextGrids, one per protocol item -- and the
evaluator's `--textgrids` defaults to them. No aligner install is required to
reproduce the reported table. `scripts/prepare_data.py --align-benchmark` exists
for rebuilding them from scratch, which produces a *different* set: it depends
on an MFA version and dictionary that were never pinned, so items silently
differ from the ones the reported numbers came from.
"""

import argparse
import os

REPO = "inclusionAI/Ming-Freeform-Audio-Edit-Benchmark"
REPO_TYPE = "dataset"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dest", default=os.environ.get(
        "WAVEPAINTER_BENCHMARK", "data/ming"))
    ap.add_argument("--repo", default=REPO)
    args = ap.parse_args()

    dest = os.path.abspath(args.dest)
    os.makedirs(dest, exist_ok=True)

    from huggingface_hub import snapshot_download

    print(f"| fetching {args.repo}")
    local = snapshot_download(args.repo, repo_type=REPO_TYPE, local_dir=dest)
    print(f"| benchmark at {local}")

    protocols = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "assets", "protocols")
    print(f"| frozen protocols ship in {protocols}")
    print("| next: python scripts/prepare_data.py --align-benchmark")


if __name__ == "__main__":
    main()
