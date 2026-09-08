"""Global Style Token encoder.

Summarises a mel spectrogram into a single fixed-width style vector, used by the
prosody-consistency loss to compare the style of a regenerated span against the
style of the utterance it belongs to. Architecture follows the Global Style
Token design of Wang et al., *Style Tokens: Unsupervised Style Modeling, Control
and Transfer in End-to-End Speech Synthesis*, ICML 2018:

    mel -> reference encoder (6x strided conv2d + BN, then a GRU)
        -> attention over a bank of learned style tokens
        -> style embedding (+ a categorical projection over style classes)

THE MODULE TREE IS FIXED BY THE RELEASED CHECKPOINTS. `state_dict['gst']` holds
52 tensors named `encoder.convs.N`, `encoder.bns.N`, `encoder.gru.*`,
`stl.embed`, `stl.attention.W_{query,key,value}.weight` and
`categorical_layer.*`. Attribute names, ModuleList ordering and the absence of
bias on the three attention projections are all part of that contract.

Two numerical details are load-bearing and easy to "fix" wrongly:

  * Attention scores are divided by ``sqrt(key_dim)``, the per-token embedding
    width (32), NOT by the per-head split width. That is what the published
    weights were trained under.
  * The style tokens are passed through ``tanh`` before they are used as keys.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REF_ENC_FILTERS = (32, 32, 64, 64, 128, 128)
REF_ENC_GRU_SIZE = 128
TOKEN_NUM = 10
NUM_HEADS = 8


def _conv_out_width(width, kernel_size, stride, padding, n_convs):
    """Width after ``n_convs`` identical strided convolutions."""
    for _ in range(n_convs):
        width = (width - kernel_size + 2 * padding) // stride + 1
    return width


class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention over a shared key/value source.

    query: ``[B, T_q, query_dim]``  key: ``[B, T_k, key_dim]``
    returns ``[B, T_q, num_units]``.
    """

    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim
        # Bias-free by contract: the checkpoint stores weights only.
        self.W_query = nn.Linear(query_dim, num_units, bias=False)
        self.W_key = nn.Linear(key_dim, num_units, bias=False)
        self.W_value = nn.Linear(key_dim, num_units, bias=False)

    def forward(self, query, key):
        q = self.W_query(query)
        k = self.W_key(key)
        v = self.W_value(key)

        # Split the projection width across heads: [B, T, U] -> [h, B, T, U/h].
        head_width = self.num_units // self.num_heads
        q, k, v = (torch.stack(torch.split(t, head_width, dim=2), dim=0)
                   for t in (q, k, v))

        # Scaled by the TOKEN width, not the head width -- see the module note.
        scores = torch.matmul(q, k.transpose(2, 3)) / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        out = torch.matmul(scores, v)                       # [h, B, T_q, U/h]
        # Concatenate the heads back along the feature axis.
        return torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)


class ReferenceEncoder(nn.Module):
    """Mel spectrogram -> one vector per utterance.

    ``[B, T, n_mels]`` -> ``[B, REF_ENC_GRU_SIZE]``, via six stride-2
    convolutions (each halving both time and frequency) and a GRU whose final
    hidden state is the summary.
    """

    def __init__(self, n_mel_channels=80):
        super().__init__()
        channels = (1,) + REF_ENC_FILTERS
        self.convs = nn.ModuleList([
            nn.Conv2d(channels[i], channels[i + 1], kernel_size=(3, 3),
                      stride=(2, 2), padding=(1, 1))
            for i in range(len(REF_ENC_FILTERS))
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(c) for c in REF_ENC_FILTERS
        ])
        # The GRU's input width follows the mel width, so n_mel_channels must
        # track hparams['audio_num_mel_bins']; hardcoding 80 silently mis-shapes
        # the stack for a 100-band front end.
        freq_out = _conv_out_width(n_mel_channels, 3, 2, 1, len(self.convs))
        self.gru = nn.GRU(input_size=REF_ENC_FILTERS[-1] * freq_out,
                          hidden_size=REF_ENC_GRU_SIZE, batch_first=True)
        self.n_mel_channels = n_mel_channels
        self.ref_enc_gru_size = REF_ENC_GRU_SIZE

    def forward(self, inputs, input_lengths=None):
        assert inputs.dim() == 3, f"expected [B, T, n_mels], got {tuple(inputs.shape)}"
        assert inputs.size(-1) == self.n_mel_channels, \
            f"expected {self.n_mel_channels} mel bins, got {inputs.size(-1)}"

        # unsqueeze(1), not (0): Conv2d wants [B, C=1, T, n_mels]. Unsqueezing
        # dim 0 reads the BATCH as conv channels and is correct only at B == 1.
        out = inputs.unsqueeze(1)
        for conv, bn in zip(self.convs, self.bns):
            out = F.relu(bn(conv(out)))

        # [B, C, T', F'] -> [B, T', C*F'] for the GRU.
        out = out.transpose(1, 2).contiguous()
        out = out.view(out.size(0), out.size(1), -1)

        if input_lengths is not None:
            # Time has been halved once per conv. Clamp per item with
            # np.maximum -- the builtin max() compares an ndarray against a list
            # and returns one whole object, so the clamp silently never happens.
            lengths = input_lengths.cpu().numpy() / 2 ** len(self.convs)
            lengths = np.maximum(lengths.round().astype(int), 1)
            lengths = np.minimum(lengths, out.size(1))
            out = nn.utils.rnn.pack_padded_sequence(
                out, lengths, batch_first=True, enforce_sorted=False)

        self.gru.flatten_parameters()
        _, hidden = self.gru(out)
        return hidden.squeeze(0)


class STL(nn.Module):
    """A bank of learned style tokens, attended to by the reference summary."""

    def __init__(self, token_embedding_size):
        super().__init__()
        self.embed = nn.Parameter(
            torch.FloatTensor(TOKEN_NUM, token_embedding_size // NUM_HEADS))
        self.attention = MultiHeadAttention(
            query_dim=token_embedding_size // 2,
            key_dim=token_embedding_size // NUM_HEADS,
            num_units=token_embedding_size,
            num_heads=NUM_HEADS,
        )
        nn.init.normal_(self.embed, mean=0.0, std=0.5)

    def forward(self, inputs):
        query = inputs.unsqueeze(1)                          # [B, 1, d_q]
        # tanh before use as keys -- part of the trained parameterisation.
        keys = torch.tanh(self.embed).unsqueeze(0).expand(inputs.size(0), -1, -1)
        return self.attention(query, keys)                   # [B, 1, E]


class GST(nn.Module):
    """Reference encoder + style tokens, with a categorical style head.

    Returns ``(style_embed [B, E], class_probs [B, classes_])``.
    """

    def __init__(self, token_embedding_size, classes_, n_mel_channels=80):
        super().__init__()
        self.encoder = ReferenceEncoder(n_mel_channels=n_mel_channels)
        self.stl = STL(token_embedding_size)
        self.categorical_layer = nn.Linear(token_embedding_size, classes_)

    def forward(self, inputs, input_lengths=None):
        enc_out = self.encoder(inputs, input_lengths=input_lengths)
        # squeeze(1), not (0): STL returns [B, 1, E], so squeezing dim 0 only
        # drops anything at B == 1 and makes the output shape batch-dependent.
        style_embed = self.stl(enc_out).squeeze(1)           # [B, E]
        cat_prob = F.softmax(self.categorical_layer(style_embed), dim=-1)
        return style_embed, cat_prob
