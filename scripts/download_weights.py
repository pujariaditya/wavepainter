#!/usr/bin/env python3
"""Fetch the released checkpoints and verify them by sha256.

    python scripts/download_weights.py              # both phases
    python scripts/download_weights.py --phase 2    # just the released model

Four checkpoints are published. Two mark the phase boundaries, so a reader can
enter at either; the other two are alternative operating points of the second:

  phase1-base          the phase-1 endpoint. `train_phase2.sh` starts here, which
                       is what makes the contribution reproducible in about an
                       hour without rebuilding phase 1.
  phase2-hubert-child  the released model, phase two as trained. This is what
                       `verify_benchmark.sh` scores by default.
                       scores, and the artifact of record for the reported table.
  phase2-hubert-a070   the same child interpolated at a lower coefficient.
                       Better on insertion, worse on substitution and deletion;
                       fetched only with `--phase 3`.
  phase2-hubert-a085   the same, at a higher coefficient. Superseded by a070 and
                       no longer reported; fetched only with `--phase 4`.

The hashes are compiled in rather than fetched alongside the files. A digest
served from the same place as the file it describes attests to nothing.
"""

import argparse
import hashlib
import os
import sys

REPO = os.environ.get("WAVEPAINTER_HF_REPO", "RootAccess4Life/wavepainter")

CHECKPOINTS = {
    1: {
        "path": "phase1-base/model_ckpt_steps_5000.ckpt",
        "sha256": "00819428a66c576fdb260d187e473ff57bdd4d4f21cb6f8c5636672435aea144",
        "note": "phase-1 base -- the starting point for train_phase2.sh",
    },
    2: {
        "path": "phase2-hubert-child/model_ckpt_steps_1500.ckpt",
        "sha256": "bfaa083797d6b9a13f73249a19c8342f05bc42ebeae2add24e5b8b26084a6dcb",
        "note": "phase-two model as trained -- the checkpoint the paper reports",
    },
    # Same phase-1 base and same phase-2 child, interpolated back toward the
    # base. Published as a second operating point: against the released child it
    # is lower on all three WER legs (3.381/3.817/9.525 against
    # 3.479/4.400/9.697). It is not the headline because the alpha sweep was read
    # off the test set, so tuning the coefficient in would be selecting on the
    # data being reported. See docs/RESULTS.md section 3.
    3: {
        "path": "phase2-hubert-a070/model_ckpt_steps_500.ckpt",
        "sha256": "0468d493cd6ebdebb6343bbb540e92da9883e5b9e770d42ac02f6007646791e6",
        "note": "alpha=0.70 variant -- lower WER on all three legs; sweep was read off the test set",
    },
    # Superseded by a070 as the published second operating point, but it stays
    # on the hub for anyone who already fetched it and it is linked from the
    # README, so it is fetchable here rather than only by hand.
    4: {
        "path": "phase2-hubert-a085/model_ckpt_steps_500.ckpt",
        "sha256": "3c0a6fe3e154241cd13560f7fdedf829805dd3961ff4df8b26bf340a90a19cf1",
        "note": "alpha=0.85 variant -- superseded by a070, kept published",
    },
}

CONFIGS = {
    1: "phase1-base/config.yaml",
    2: "phase2-hubert-child/config.yaml",
    3: "phase2-hubert-a070/config.yaml",
    4: "phase2-hubert-a085/config.yaml",
}


def sha256_of(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(phase, dest_root):
    from huggingface_hub import hf_hub_download

    spec = CHECKPOINTS[phase]
    print(f"| phase {phase}: {spec['note']}")

    local = hf_hub_download(repo_id=REPO, filename=spec["path"])
    actual = sha256_of(local)
    if actual != spec["sha256"]:
        raise SystemExit(
            f"sha256 mismatch for {spec['path']}\n"
            f"  expected {spec['sha256']}\n"
            f"  got      {actual}\n"
            "Refusing to use a checkpoint that is not the released one."
        )
    print(f"|   sha256 OK  {actual[:16]}...")

    config = hf_hub_download(repo_id=REPO, filename=CONFIGS[phase])

    # Lay the pair out as the evaluator expects: checkpoints/<exp_name>/.
    exp_dir = os.path.join(dest_root, "checkpoints", os.path.dirname(spec["path"]))
    os.makedirs(exp_dir, exist_ok=True)
    for src, name in ((local, os.path.basename(spec["path"])), (config, "config.yaml")):
        link = os.path.join(exp_dir, name)
        if os.path.lexists(link):
            os.remove(link)
        # Symlink into the HF cache rather than copying 2 GB twice.
        os.symlink(os.path.realpath(src), link)
    print(f"|   ready: {exp_dir}")
    return exp_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", type=int, choices=[1, 2, 3, 4], action="append",
                    help="1 = phase-1 base, 2 = released model, "
                         "3 = the alpha=0.70 variant, 4 = the alpha=0.85 "
                         "variant. Repeatable; default is 1 and 2.")
    ap.add_argument("--dest", default=os.environ.get("WAVEPAINTER_ROOT", "."))
    args = ap.parse_args()

    # 3 and 4 are opt-in: most users want the released model.
    phases = sorted(set(args.phase)) if args.phase else [1, 2]
    print(f"| repo {REPO}")
    for phase in phases:
        fetch(phase, os.path.abspath(args.dest))
    print("| done")


if __name__ == "__main__":
    sys.exit(main())
