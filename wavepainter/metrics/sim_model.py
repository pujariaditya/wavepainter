"""Speaker-verification front end for the SIM metric.

SIM is three of the numbers wavepainter reports, so this path has to reproduce
exactly rather than approximately. Everything below is ours; the two third-party
model files it drives -- UniSpeech's ``ecapa_tdnn.py`` and Microsoft's WavLM --
are fetched by ``scripts/fetch_sim_model.py`` into a gitignored ``third_party/``
and are never redistributed here.

UniSpeech's ``ECAPA_TDNN`` reaches its WavLM encoder one of two ways::

    if config_path is None:
        self.feature_extract = torch.hub.load('s3prl/s3prl', feat_type)
    else:
        self.feature_extract = UpstreamExpert(config_path)

The second branch is the extension point this module fills. We supply the
``UpstreamExpert`` rather than installing s3prl, because the UniSpeech finetune
checkpoint already contains the **entire** front end -- 488 of its 711 tensors
are ``feature_extract.model.*``, including the conv extractor and ``mask_emb``.
Only the module *tree* to load them into is missing. That is all this provides.

The WavLM config is not read from a file, it is asserted by construction: every
field was derived from the checkpoint's own tensor shapes, and ``load_sv_model``
then loads **strictly**, so a wrong config fails loudly at load rather than
silently producing embeddings incomparable to the published SIM column.

    feature_weight            (25,)          -> 24 layers + input
    encoder.layers.*          max index 23   -> encoder_layers 24
    self_attn.q_proj.weight   (1024, 1024)   -> encoder_embed_dim 1024
    fc1.weight                (4096, 1024)   -> encoder_ffn_embed_dim 4096
    relative_attention_bias   present        -> relative_position_embedding True
    post_extract_proj.weight  (1024, 512)    -> conv output 512
    layer_norm.{weight,bias}  present        -> extractor_mode "layer_norm"
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# WavLM-Large. `normalize: True` pairs with `extractor_mode: layer_norm`, and it
# is consumed by the CALLER -- s3prl's expert layer-norms the raw waveform before
# handing it over (see UpstreamExpert.forward below). `WavLM.extract_features`
# never reads it: the `self.layer_norm` inside the model applies to the CONV
# OUTPUT, not the input waveform. Getting this wrong changes every embedding
# without raising anything.
WAVLM_LARGE_CFG = {
    "extractor_mode": "layer_norm",
    "encoder_layers": 24,
    "encoder_embed_dim": 1024,
    "encoder_ffn_embed_dim": 4096,
    "encoder_attention_heads": 16,
    "activation_fn": "gelu",
    "layer_norm_first": True,
    "conv_feature_layers": "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
    "conv_bias": False,
    "feature_grad_mult": 1.0,
    "normalize": True,
    "encoder_layerdrop": 0.0,
    "dropout": 0.0,
    "attention_dropout": 0.0,
    "activation_dropout": 0.0,
    "dropout_input": 0.0,
    "dropout_features": 0.0,
    "relative_position_embedding": True,
    "num_buckets": 320,
    "max_distance": 800,
    "gru_rel_pos": True,
}


def _third_party_dir():
    """Locate the fetched speaker-verification sources."""
    override = os.environ.get("WAVEPAINTER_SV_DIR")
    if override:
        return override
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "third_party", "sv")


def _import_third_party():
    """Import the fetched ``ecapa_tdnn`` and WavLM modules.

    ``scripts/fetch_sim_model.py`` lays these out as a package whose ``utils``
    module re-exports this one, so UniSpeech's unmodified
    ``from .utils import UpstreamExpert`` resolves to the class below.
    """
    directory = _third_party_dir()
    if not os.path.isdir(directory):
        raise RuntimeError(
            f"speaker-verification sources not found at {directory}. "
            "Run `python scripts/fetch_sim_model.py` first."
        )
    parent = os.path.dirname(directory)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from sv.ecapa_tdnn import ECAPA_TDNN_SMALL
    from sv.wavlm.WavLM import WavLM, WavLMConfig

    return ECAPA_TDNN_SMALL, WavLM, WavLMConfig


class UpstreamExpert(nn.Module):
    """Wraps a WavLM so that ``self.model.*`` matches the checkpoint's layout."""

    def __init__(self, config_path=None):
        super().__init__()
        _, WavLM, WavLMConfig = _import_third_party()
        self.model = WavLM(WavLMConfig(WAVLM_LARGE_CFG))

    def forward(self, wavs):
        """``wavs`` is a list of 1-D tensors. Returns {"hidden_states": [B,T,C] x 25}.

        Mirrors s3prl's wavlm expert: pad the batch, build the padding mask from
        true lengths, and take every layer result including the encoder input --
        ``ecapa_tdnn`` softmaxes ``feature_weight`` over exactly these 25.
        """
        # s3prl normalises the WAVEFORM first, per-utterance, over its true
        # length, before any padding:
        #     if self.cfg.normalize:
        #         wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]
        # (s3prl/upstream/wavlm/expert.py, verbatim). Omitting it left SIM biased
        # HIGH in one direction -- ~+0.0035 near the ceiling and ~+0.026 at low
        # similarity -- on a metric whose published bar-to-target span is 0.27.
        #
        # Per-wav and pre-padding both matter: layer-norming the padded batch
        # would fold each utterance's zero-padding into its own mean and
        # variance, making an embedding depend on the longest item beside it.
        if self.model.cfg.normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        device = wavs[0].device
        lengths = torch.tensor([len(w) for w in wavs], device=device)
        padded = pad_sequence(wavs, batch_first=True)
        padding_mask = (
            torch.arange(padded.shape[1], device=device)[None, :] >= lengths[:, None]
        )

        # output_layer MUST be passed. WavLM's encoder only appends to
        # `layer_results` when `tgt_layer is not None`, so the obvious
        # `output_layer=None` silently returns an EMPTY list and `feature_weight`
        # ends up size 0. With output_layer=24 -> tgt_layer=23, the encoder
        # appends its input plus all 24 layers = the 25 states the checkpoint's
        # `feature_weight` is sized for.
        #
        # It also has a side effect that has to be undone below: passing it makes
        # `layer` non-None, which suppresses the encoder's final normalisation.
        _, layer_results = self.model.extract_features(
            padded,
            padding_mask=padding_mask,
            mask=False,
            ret_layer_results=True,
            output_layer=self.model.cfg.encoder_layers,
        )[0]
        # layer_results entries are (T, B, C); ecapa wants B, T, C.
        hidden_states = [h.transpose(0, 1) for h, _ in layer_results]
        # Re-apply the norm that `output_layer` suppressed. s3prl reaches the
        # last state through a HOOK on `self.model.encoder`, i.e. the return of
        # TransformerEncoder.forward, which is
        #     if self.layer_norm_first and layer is None: x = self.layer_norm(x)
        # s3prl passes no output_layer, so `layer is None` holds and its state 24
        # is normalised; ours took the same tensor one step earlier. State 24
        # carries softmax(feature_weight)[24] = 0.0182 of the weighted sum, so
        # this moved SIM by ~-0.0013 -- small, but the same inflating direction
        # as the waveform norm above.
        #
        # LayerNorm is over the channel axis, so applying it after the transpose
        # to [B, T, C] is the same function as before it on [T, B, C].
        enc = self.model.encoder
        if enc.layer_norm_first:
            hidden_states[-1] = enc.layer_norm(hidden_states[-1])
        return {"hidden_states": hidden_states}


def load_sv_model(ckpt_path, device="cpu"):
    """Build ECAPA_TDNN_SMALL over WavLM-Large and load the UniSpeech finetune.

    Strict on the front end: ``feature_extract.*`` must match exactly, which is
    what validates WAVLM_LARGE_CFG. The ECAPA head is loaded strictly too; the
    only key tolerated as missing is ``loss_calculator.*``, which is
    training-only.
    """
    ECAPA_TDNN_SMALL, _, _ = _import_third_party()

    # Any non-None config_path selects the UpstreamExpert branch above; the
    # value itself is unused, since the config is compiled in rather than read.
    model = ECAPA_TDNN_SMALL(
        feat_dim=1024, feat_type="wavlm_large", emb_dim=256, config_path="wavepainter"
    )
    state = torch.load(ckpt_path, map_location="cpu")
    state = state.get("model", state)
    state = {k: v for k, v in state.items() if not k.startswith("loss_calculator")}

    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith("loss_calculator")]
    if missing or unexpected:
        raise RuntimeError(
            f"speaker-verification checkpoint does not match the module tree: "
            f"{len(missing)} missing {missing[:4]}, "
            f"{len(unexpected)} unexpected {unexpected[:4]}. "
            f"WAVLM_LARGE_CFG is wrong, and the SIM numbers would not be "
            f"comparable to the published table."
        )
    return model.eval().to(device)
