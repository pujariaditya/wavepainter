"""Interpolate a trained child back toward its base, over selected prefixes.

    released[t] = base[t] + alpha * (child[t] - base[t])     for t matching a prefix
    released[t] = base[t]                                    otherwise

This is the second half of phase 2. Training a denoiser-only child moves it away
from the base, and some of that movement is the mechanism while some is just
drift. Interpolating back at ``alpha < 1`` keeps the first and bounds the second.
The released model does NOT use this step: it is the child as trained, which is
alpha=1.0. Measured over alpha in {0.70, 0.75, 0.85, 0.90, 1.00}, 0.85 is an
interior optimum for substitution (3.063) and deletion (9.136) but NOT for
insertion, which is best at 0.70 (3.817 against 4.128 at 0.85). Alpha=1.0 is the
worst point on all three legs -- so the interpolation does real work, and it is
still not what ships, because that ranking was read off the test set. Choosing a
coefficient by it would be selecting on the data being reported. The alpha=0.70
checkpoint remains published as a second operating point.

The output keeps the **base** checkpoint's container structure (``epoch``,
``global_step``, ``state_dict['gst']``) so the evaluator's loader finds the
``state_dict['model']`` nesting it expects. ``optimizer_states`` is dropped: it
belongs to the base's own run and refers to a parameter set that no longer
matches.

The written ``config.yaml`` is an **inference** config: it is the child's, with
``lambda_hubert_feature`` set to 0 and ``trainable_only`` emptied. Neither is read
at inference, and leaving them set would make the evaluator construct the frozen
HuBERT and BigVGAN teachers -- a download and a load, to compute a loss nobody
consumes. The released config carries them zeroed for the same reason.

The output checkpoint is always named ``model_ckpt_steps_500.ckpt``. The number is
a label the evaluator is pointed at (``ckpt_steps=500``), not a step count: the
released model interpolates a 5000-step base with a 1500-step child.

    python -m wavepainter.training.interpolate \\
        --base  phase1-base/model_ckpt_steps_5000.ckpt \\
        --child runs/hubert_dn/model_ckpt_steps_1500.ckpt \\
        --child-config runs/hubert_dn/config.yaml \\
        --out   released/ --alpha 0.85 --prefixes denoise_fn.
"""

import argparse
import os

import torch
import yaml

# Training-only keys, zeroed in the inference config the released model ships
# with. Set as literals rather than deleted: the loader expects both to exist.
INFERENCE_OVERRIDES = {"lambda_hubert_feature": 0.0, "trainable_only": []}


def write_inference_config(child_config, dest):
    """Copy the child's config with the phase-2 training keys neutralised."""
    with open(child_config) as handle:
        config = yaml.safe_load(handle)
    for key, value in INFERENCE_OVERRIDES.items():
        if key in config:
            config[key] = value
    with open(dest, "w") as handle:
        yaml.safe_dump(config, handle, default_flow_style=False, sort_keys=True)


def interpolate(base_ckpt, child_ckpt, alpha, prefixes):
    """Return the base checkpoint dict with matching tensors moved toward child."""
    out = torch.load(base_ckpt, map_location="cpu")
    base_model = out["state_dict"]["model"]
    child_model = torch.load(child_ckpt, map_location="cpu")["state_dict"]["model"]

    base_keys = set(base_model.keys())
    n_interp = n_new = 0

    for key, base_value in base_model.items():
        child_value = child_model.get(key)
        if (
            child_value is not None
            and child_value.shape == base_value.shape
            and any(key.startswith(p) for p in prefixes)
        ):
            merged = base_value.float() + alpha * (
                child_value.float() - base_value.float()
            )
            base_model[key] = merged.to(base_value.dtype)
            n_interp += 1

    # Tensors that exist only in the child -- a zero-initialised adapter added
    # onto a frozen base -- have no base vector to interpolate toward. The
    # base's implicit value is absent, not zero, so scaling them by alpha would
    # under-feed a denoiser trained against the full adapter. Copy verbatim.
    for key, child_value in child_model.items():
        if key not in base_keys and any(key.startswith(p) for p in prefixes):
            base_model[key] = child_value.clone()
            n_new += 1

    out["state_dict"]["model"] = base_model
    out.pop("optimizer_states", None)
    return out, n_interp, n_new


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, help="phase-1 base checkpoint")
    ap.add_argument("--child", required=True, help="phase-2 trained child")
    ap.add_argument("--child-config", required=True)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument(
        "--prefixes",
        default="denoise_fn.",
        help="comma-separated parameter-name prefixes to interpolate",
    )
    args = ap.parse_args()

    prefixes = [p.strip() for p in args.prefixes.split(",") if p.strip()]
    out, n_interp, n_new = interpolate(
        args.base, args.child, args.alpha, prefixes
    )

    os.makedirs(args.out, exist_ok=True)
    write_inference_config(args.child_config, os.path.join(args.out, "config.yaml"))
    dest = os.path.join(args.out, "model_ckpt_steps_500.ckpt")
    torch.save(out, dest)

    print(
        f"| interpolated {n_interp} shared tensors at alpha={args.alpha} "
        f"(prefixes {prefixes}); copied {n_new} child-only tensors verbatim"
    )
    print(f"| wrote {dest}")


if __name__ == "__main__":
    main()
