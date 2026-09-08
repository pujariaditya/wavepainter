"""Warm-start a donor-parameterized DualFFN trunk and prove it took effect.

Both FFN branches get the SAME donor MLP, so at init the DualFFN is exactly a
single-FFN donor trunk. Optional attention biases and QK norms are copied only
when they exist in the selected donor.

The verification exists because a silently-skipped warm start is invisible: the
model still trains, the loss still falls, and you discover months later that the
donor weights were overwritten before the first step. Four checks, all cheap, all
printed.
"""

import glob
import json
import os

import torch

# NO DEFAULT DONOR. A fallback here is how a config typo becomes a silently
# different model: the trunk warm-starts from whatever the default names, the
# loss falls, and nothing says the donor you asked for was never loaded. Callers
# pass `trunk_donor` from hparams, and a missing key must raise.
DONOR_DEFAULT = None


def donor_dir(name=DONOR_DEFAULT):
    """Resolve the donor's snapshot dir, fetching it if not already cached.

    Accepts a local directory or a Hugging Face repo id.
    """
    if not name:
        raise ValueError("no donor given; set `trunk_donor` in the config")
    if os.path.isdir(name):
        return name
    from huggingface_hub import snapshot_download

    return snapshot_download(name)


def load_donor_config(name=DONOR_DEFAULT):
    if not name:
        raise ValueError("no donor given; set `trunk_donor` in the config")
    """Read only config.json, without materializing checkpoint tensors."""
    with open(f"{donor_dir(name)}/config.json") as handle:
        return json.load(handle)


def load_donor(name=DONOR_DEFAULT):
    """Return (state_dict, config) for the donor, stripped of the 'model.' prefix."""
    from safetensors.torch import load_file

    d = donor_dir(name)
    with open(f"{d}/config.json") as handle:
        cfg = json.load(handle)
    files = sorted(glob.glob(f"{d}/*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors in {d}")
    sd = {}
    for f in files:
        sd.update(load_file(f))
    sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    return sd, cfg


def layer_sources(n_layers, pick, n_donor=28):
    """Which donor layer feeds each trunk layer.

    'ends' is the default below full depth: keep the donor's early layers (which
    do the low-level work) AND its late layers (which produce the residual stream
    that `model.norm` was calibrated for). A plain 0..n-1 prefix applies
    `model.norm`, fitted for layer-27 output, to a layer-13 stream -- a real
    mismatch that every prefix-style warm start silently carries.
    """
    if n_layers > n_donor:
        # Silently truncating left the extra layers randomly initialised while the
        # write-count check still passed (it used len(src)), and the per-layer
        # verification then IndexError'd on the last layer. Refuse instead.
        raise ValueError(
            f"trunk has {n_layers} layers but the donor only has {n_donor}; "
            f"there is no defined warm start for layers {n_donor}..{n_layers - 1}")
    if n_layers == n_donor:
        return list(range(n_donor))
    if pick == "prefix":
        return list(range(n_layers))
    if pick == "stride":
        stride = n_donor // n_layers
        return [min(i * stride, n_donor - 1) for i in range(n_layers)]
    if pick == "ends":
        n_head = (n_layers + 1) // 2
        n_tail = n_layers - n_head
        return list(range(n_head)) + list(range(n_donor - n_tail, n_donor))
    raise ValueError(f"unknown layer_pick {pick!r}")


# Donor tensor suffix -> list of trunk attribute paths it is copied into.
# Split by ARCHITECTURE, because the norm rows are the one place two families
# use the same tensor names for different things.
_MAP_CORE = [
    ("self_attn.q_proj.weight", ["attn.q_proj.weight"]),
    ("self_attn.k_proj.weight", ["attn.k_proj.weight"]),
    ("self_attn.v_proj.weight", ["attn.v_proj.weight"]),
    ("self_attn.o_proj.weight", ["attn.o_proj.weight"]),
    ("mlp.gate_proj.weight", ["ffn_text.gate_proj.weight", "ffn_audio.gate_proj.weight"]),
    ("mlp.up_proj.weight", ["ffn_text.up_proj.weight", "ffn_audio.up_proj.weight"]),
    ("mlp.down_proj.weight", ["ffn_text.down_proj.weight", "ffn_audio.down_proj.weight"]),
]

_MAP_BIAS = [
    ("self_attn.q_proj.bias", ["attn.q_proj.bias"]),
    ("self_attn.k_proj.bias", ["attn.k_proj.bias"]),
    ("self_attn.v_proj.bias", ["attn.v_proj.bias"]),
]

_MAP_QKNORM = [
    ("self_attn.q_norm.weight", ["attn.q_norm.weight"]),
    ("self_attn.k_norm.weight", ["attn.k_norm.weight"]),
]

# FOUR norms per layer. Gemma sandwiches every sublayer, and
# `post_attention_layernorm` does NOT mean what its name suggests:
#
#   input_layernorm            -> pre-attn    (ln1)
#   post_attention_layernorm   -> POST-attn   (ln1p)
#   pre_feedforward_layernorm  -> pre-FFN     (ln2)
#   post_feedforward_layernorm -> post-FFN    (ln2p)
_MAP_NORMS_SANDWICH = [
    ("input_layernorm.weight", ["ln1_t.weight", "ln1_a.weight"]),
    ("post_attention_layernorm.weight", ["ln1p_t.weight", "ln1p_a.weight"]),
    ("pre_feedforward_layernorm.weight", ["ln2_t.weight", "ln2_a.weight"]),
    ("post_feedforward_layernorm.weight", ["ln2p_t.weight", "ln2p_a.weight"]),
]

# TWO norms per layer -- Qwen and the ordinary decoder-only layout. Here
# `post_attention_layernorm` IS the pre-FFN norm: it is the norm applied to the
# residual stream AFTER the attention block, on the way into the MLP. Loading it
# into ln1p (as the sandwich map does) would put a pre-FFN norm on the attention
# output and leave the FFN unnormalised -- clean shapes, ~1.0 shape-cosine
# provenance, wrong function. This is the single most dangerous line in the file.
_MAP_NORMS_PRE = [
    ("input_layernorm.weight", ["ln1_t.weight", "ln1_a.weight"]),
    ("post_attention_layernorm.weight", ["ln2_t.weight", "ln2_a.weight"]),
]


def _map_for(cfg):
    """The map this donor's tensors actually mean, plus what it MUST provide.

    Derived from the donor config rather than hardcoded, and the required set is
    derived with it -- otherwise `expected_writes` is computed from the same
    filtered list that dropped the missing tensors, and a donor missing half its
    norms warm-starts silently with writes == expected.
    """
    from wavepainter.models.dualffn.trunk import geometry_from_config

    g = geometry_from_config(cfg)
    mapping = list(_MAP_CORE)
    mapping += _MAP_NORMS_SANDWICH if g["post_norms"] else _MAP_NORMS_PRE
    if g["qk_norm"]:
        mapping += _MAP_QKNORM
    if g["attention_bias"]:
        mapping += _MAP_BIAS
    # Everything the selected map names is required. Nothing is optional: the
    # map was chosen FROM this donor's own config, so an absent tensor means the
    # config and the weights disagree.
    return mapping, {suffix for suffix, _ in mapping}


MODEL_WRITES = 3  # norm_t, norm_a, tok_emb


def _available_map(state_dict, source_layer, cfg):
    prefix = f"layers.{source_layer}."
    mapping, required = _map_for(cfg)
    missing = [k for k, _ in mapping
               if k in required and prefix + k not in state_dict]
    if missing:
        raise RuntimeError(
            f"donor layer {source_layer} is missing {missing}, which the map "
            f"selected from its own config ({cfg.get('model_type')}) requires. "
            f"Present: "
            f"{sorted(k[len(prefix):] for k in state_dict if k.startswith(prefix))}")
    return [(k, d) for k, d in mapping if prefix + k in state_dict]


def _set(mod, path, value):
    obj = mod
    parts = path.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    tgt = getattr(obj, parts[-1])
    if tgt.shape != value.shape:
        raise RuntimeError(f"shape mismatch at {path}: {tuple(tgt.shape)} vs {tuple(value.shape)}")
    with torch.no_grad():
        tgt.copy_(value.to(tgt.dtype))


def warm_start(trunk, donor=DONOR_DEFAULT, pick="ends", freeze_tok_emb=True,
               copy_final_norm=True, verbose=True):
    """Copy donor weights into the trunk. Returns a report dict."""
    sd, cfg = load_donor(donor)
    n_donor = cfg["num_hidden_layers"]
    src = layer_sources(len(trunk.layers), pick, n_donor)

    # Local/global identity must follow the WEIGHTS into their new slot: with
    # pick='ends' trunk slot i does not hold donor layer i, and a layer trained
    # at theta=1e4 over a 512 span still wants those after it moves.
    sliding = trunk.set_layer_sources(src) if hasattr(trunk, "set_layer_sources") else None

    n_writes = 0
    for i, s in enumerate(src):
        layer = trunk.layers[i]
        for suffix, dests in _available_map(sd, s, cfg):
            v = sd[f"layers.{s}.{suffix}"]
            for d in dests:
                _set(layer, d, v)
                n_writes += 1

    # `model.norm` is calibrated for the FINAL donor layer's output. Copying it
    # onto a truncated prefix is the mismatch `layer_pick=ends` exists to avoid;
    # if someone asks for a prefix anyway, leave the norms at ones.
    if copy_final_norm and (len(trunk.layers) >= n_donor or pick != "prefix"):
        _set(trunk, "norm_t.weight", sd["norm.weight"])
        _set(trunk, "norm_a.weight", sd["norm.weight"])
        n_writes += 2
    _set(trunk, "tok_emb.weight", sd["embed_tokens.weight"])
    n_writes += 1

    if freeze_tok_emb:
        # 151936 x 1024 = 155.6M. As an nn.Embedding this produces a DENSE
        # gradient, so DDP would all-reduce 622 MB every step and AdamW would
        # carry 1.87 GB of state, for a few thousand ids that ever appear.
        trunk.tok_emb.weight.requires_grad_(False)

    # `expected` must track what was actually attempted: the two norm copies are
    # skipped for a truncated prefix (model.norm is calibrated for the FINAL donor
    # layer, so applying it to a layer-13 stream is wrong). Counting them
    # unconditionally made every `pick='prefix'` run below full depth raise, which
    # meant the ends-vs-prefix ablation -- the whole reason layer_sources exists --
    # could not be run at all.
    copied_norms = copy_final_norm and (len(trunk.layers) >= n_donor or pick != "prefix")
    expected = sum(
        sum(len(destinations) for _, destinations in _available_map(sd, s, cfg))
        for s in src
    ) + (MODEL_WRITES if copied_norms else 1)
    report = {
        "n_writes": n_writes,
        "expected_writes": expected,
        "layer_sources": src,
        "pick": pick,
        "rope_half": trunk.rope_half,
        "frozen_tok_emb": freeze_tok_emb,
        "sliding": sliding,
    }
    if verbose:
        print(f"| WARMSTART: applied | donor={donor} | rope_half={trunk.rope_half} "
              f"| layers={len(src)} pick={pick} | writes={n_writes}/{expected} "
              f"| src={src[:4]}...{src[-2:]}")
    if n_writes != expected:
        raise RuntimeError(f"warm start wrote {n_writes} tensors, expected {expected}")
    return report


def donor_provenance(trunk, donor=DONOR_DEFAULT, n_probe_layers=4, match_at=0.5):
    """Do the trunk's TRAINED weights actually come from a pretrained LLM?

    `warm_start_applied` is a fact about the config, not about the checkpoint.
    The model is reconstructed at eval time, warm-started, and then the trained
    state_dict is loaded straight over the top -- so a trunk trained from random
    init reports `warm_start_applied: True` exactly as loudly as one that really
    started from a donor.

    This measures the weights that were actually trained, WITHOUT assuming the
    donor is Gemma or that anyone kept its tensor names. Every 2-D weight in the
    sampled layers is matched against every donor tensor of the SAME SHAPE and
    scored by its best cosine. That question -- "is this matrix one of the
    donor's, moved a bit?" -- is architecture-agnostic and rename-proof.

    Finetuning moves a weight matrix; it does not move it to a different point
    in a million-dimensional space. Two independent random matrices that size
    sit at cosine ~1e-3, so the two populations are three orders of magnitude
    apart and no threshold between them is delicate.

    Returns None if nothing could be compared (unknown donor, no 2-D weights,
    no shape overlap) -- an unresolvable answer, which is not the same as a
    failing one and must not be scored as one.
    """
    import torch

    try:
        sd, _ = load_donor(donor)
    except Exception as e:  # noqa: BLE001
        return {"donor": donor, "resolved": False, "error": str(e)[:200]}

    n_layers = len(trunk.layers)
    step = max(1, n_layers // n_probe_layers)
    probe_layers = list(range(0, n_layers, step))[:n_probe_layers]

    # Only bucket donor tensors whose shape some probed trunk weight actually
    # has. Otherwise the 262144x1152 embedding is normalised and held in memory
    # on every call to be compared against nothing.
    wanted = {tuple(p.shape) for i in probe_layers
              for p in trunk.layers[i].parameters() if p.ndim == 2}

    # float64 throughout. In fp32 the sum of squares over a 6912x1152 matrix
    # loses enough precision that a tensor compared against ITSELF comes back at
    # 1.0002 -- a cosine above 1, which is impossible and was the only visible
    # symptom. Nothing else about the number looks wrong, and the same class of
    # error once read 1.0403 on a 155M-element embedding.
    def unit(t):
        f = t.detach().flatten().to(torch.float64)
        n = torch.linalg.vector_norm(f)
        return None if n == 0 else f / n

    by_shape = {}
    for v in sd.values():
        if v.ndim == 2 and tuple(v.shape) in wanted:
            u = unit(v)
            if u is not None:
                by_shape.setdefault(tuple(v.shape), []).append(u)
    if not by_shape:
        return {"donor": donor, "resolved": False,
                "error": "no 2-D donor tensor shares a shape with the trunk"}

    best, per = [], {}
    for i in probe_layers:
        for name, p in trunk.layers[i].named_parameters():
            if p.ndim != 2:
                continue
            cands = by_shape.get(tuple(p.shape))
            if not cands:
                continue
            # .cpu(): the trunk is on CUDA at eval time, the donor is read to
            # host memory. Without this every real eval dies on a device mismatch
            # -- which a CPU-only unit test cannot see.
            f = unit(p.cpu())
            if f is None:
                continue
            # Clamp: float64 leaves only rounding at the last bit, but a cosine
            # is bounded by construction and a number outside [-1, 1] downstream
            # is a defect report, not a score.
            c = min(1.0, max(float(torch.dot(f, w)) for w in cands))
            best.append(c)
            per[f"L{i}.{name}"] = round(c, 4)
    if not best:
        return {"donor": donor, "resolved": False,
                "error": "no trunk 2-D weight shares a shape with the donor"}

    matched = [c for c in best if c > match_at]
    return {
        "donor": donor,
        "resolved": True,
        "n_tensors": len(best),
        # The headline number. A trunk that adds fresh random adapters inside
        # its layers still carries the donor in the bulk of its weights, so the
        # FRACTION matched is the honest statistic here and a mean is not.
        "frac_matched": round(len(matched) / len(best), 4),
        "mean_cos_matched": round(sum(matched) / len(matched), 4) if matched else None,
        "max_cos": round(max(best), 4),
        "median_cos": round(sorted(best)[len(best) // 2], 4),
        "per_tensor": dict(sorted(per.items())[:12]),
    }
