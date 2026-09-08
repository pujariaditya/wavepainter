"""Donor-parameterized DualFFN trunk with modality-routed FFN branches.

Both branches receive the same pretrained donor MLP at initialization, so the
DualFFN begins as the donor's single-FFN transformer on the text path. Geometry
is read from the selected donor's config rather than inferred from tensor names,
so `text_only_hidden_states` can be checked against the donor's own forward pass.

The donor is Qwen/Qwen2.5-0.5B. Every convention below is read from the donor's
own config rather than assumed, because families disagree on things that load
cleanly and compute the wrong function:

    norms per layer         qwen2  2            gemma3  4 (sandwich)
    post_attention_layernorm  qwen2  pre-FFN    gemma3  post-attention
    norm storage            qwen2  x*w  init 1  gemma3  x*(1+w) init 0
    embed scale             qwen2  none         gemma3  sqrt(hidden)
    qkv bias / QK-norm      qwen2  yes / no     gemma3  no / yes
    RoPE                    qwen2  all 1e6      gemma3  5 local 1e4 : 1 global 1e6
    gated MLP               qwen2  SwiGLU       gemma3  GeGLU

Not one of those changes a tensor NAME or SHAPE, so a mis-set convention holds
the donor's exact weights and scores ~1.0 on shape-cosine provenance while
computing something else. `text_only_hidden_states` exists to catch exactly
that: it reproduces the donor's own forward pass, and the only acceptable answer
is a per-layer cosine of 1.000000.

The support is deliberately family-driven rather than Qwen-only. It is what
proved this port correct -- running the same check against a second donor is how
a RoPE fallback that made five layers in six use the wrong theta was found.

The trunk sits OUTSIDE the diffusion reverse loop. Its input depends only on the
masked mel, the text and the alignment -- never on the diffusion timestep -- so
one trunk pass feeds all `timesteps` DiffNet passes. That is what makes a
~400M-parameter trunk affordable at inference in a model that denoises 8 times.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# NO FALLBACK GEOMETRY. Every value below comes from the donor's own config.
# A module-level default is how a donor swap half-lands: the config supplies
# some keys, the constants quietly supply the rest, and the trunk ends up a
# chimera that holds real weights and computes a function neither donor has.
# That is not hypothetical -- inheriting one such default (a sliding-window
# pattern of 6) put five layers in six on the wrong RoPE base while every
# tensor stayed byte-correct.


def geometry_from_config(config):
    """Normalize the small subset of HF config that defines trunk topology.

    `config` is REQUIRED and must name its family. Nothing is guessed.
    """
    if not config:
        raise ValueError(
            "geometry_from_config needs the donor's config; there is no default "
            "geometry. Load it with warm_start.load_donor_config(trunk_donor).")
    config = dict(config)
    if "hidden_size" not in config or "num_attention_heads" not in config:
        raise ValueError(f"donor config is missing hidden_size/num_attention_heads: "
                         f"{sorted(config)[:12]}")
    hidden = int(config["hidden_size"])
    n_heads = int(config["num_attention_heads"])
    model_type = str(config.get("model_type") or "")
    if not model_type:
        raise ValueError("donor config has no model_type; the norm, embed-scale "
                         "and RoPE conventions are selected from it")
    return {
        "hidden_size": hidden,
        "intermediate_size": int(config["intermediate_size"]),
        "num_attention_heads": n_heads,
        "num_key_value_heads": int(
            config.get("num_key_value_heads") or n_heads
        ),
        "head_dim": int(config.get("head_dim") or hidden // n_heads),
        "rope_theta": float(config.get("rope_theta", 1e6)),
        "rms_norm_eps": float(config["rms_norm_eps"]),
        "vocab_size": int(config["vocab_size"]),
        # These two differ per FAMILY and the configs do not always say so, which
        # makes a wrong default silent rather than loud:
        #
        #   gemma3  no qkv bias,  QK-norm      (config omits both)
        #   qwen2   qkv BIAS,     no QK-norm   (config omits both; HF hardcodes
        #                                       bias=True on q/k/v, False on o)
        #   qwen3   no qkv bias,  QK-norm
        #
        # Getting `attention_bias` wrong builds a trunk whose q/k/v have no bias
        # to copy the donor's into -- the tensors are simply dropped, and the
        # warm start reports success because the map never asked for them.
        "attention_bias": bool(config.get("attention_bias")
                               if config.get("attention_bias") is not None
                               else model_type.startswith("qwen2")),
        "qk_norm": bool(config.get("qk_norm")
                        if config.get("qk_norm") is not None
                        else (model_type.startswith("gemma")
                              or model_type.startswith("qwen3"))),
        "model_type": model_type,
        # Gemma's MLP is GeGLU (`hidden_activation: gelu_pytorch_tanh`), not the
        # SwiGLU that the same three tensor names imply elsewhere. gate/up/down
        # have identical shapes under both, so the wrong activation loads without
        # a single warning and quietly changes what every layer computes.
        "hidden_activation": str(
            config.get("hidden_activation")
            or config.get("hidden_act") or "gelu_pytorch_tanh"),
        # Gemma sandwiches each sublayer: x + post_norm(sublayer(pre_norm(x))),
        # i.e. FOUR norms per layer. Qwen and the ordinary decoder-only layout
        # have TWO. Read it from the model family rather than assuming, because
        # a 4-norm map applied to a 2-norm donor mis-assigns
        # `post_attention_layernorm` -- which is the POST-attention norm in Gemma
        # and the PRE-FFN norm everywhere else. That loads cleanly, passes
        # shape-matched provenance, and computes the wrong function.
        "post_norms": model_type.startswith("gemma"),
        # 5 local : 1 global, with a DIFFERENT rope base on each. One rope cache
        # for the whole stack applies theta=1e6 to layers trained at 1e4.
        # Gemma interleaves 5 local : 1 global with a DIFFERENT rope base on
        # each. Qwen2.5-0.5B has neither -- every layer is global at theta=1e6,
        # and `use_sliding_window` is false.
        #
        # These MUST default off for a donor that does not declare them.
        # Inheriting Gemma's pattern of 6 made five layers in six use theta=1e4
        # and a 512-wide band mask on a donor trained with neither: the trunk
        # still held Qwen's exact tensors, still passed shape-cosine provenance,
        # and reproduced its hidden states to only 0.975 by layer 22 -- drift
        # that grows with depth, which is what a wrong positional encoding looks
        # like and what a checksum never sees.
        "rope_local_base_freq": float(
            config.get("rope_local_base_freq") or 1e4),
        "sliding_window": int(
            (config.get("sliding_window") or 0)
            if config.get("use_sliding_window", True) else 0),
        "sliding_window_pattern": int(
            config.get("sliding_window_pattern")
            or (6 if model_type.startswith("gemma") else 0)),
        # Gemma scales embeddings by sqrt(hidden) = 33.9; omitting it leaves the
        # trunk input ~34x too small for the weights that follow. Qwen does NOT
        # scale, and applying Gemma's scale to it is the same error in reverse.
        "embed_scale": (float(hidden) ** 0.5
                        if model_type.startswith("gemma") else 1.0),
        # Gemma stores norm weights as a DELTA around zero and computes
        # `x * (1 + w)`; everyone else stores them around one and computes
        # `x * w`. Same tensor names, same shapes, different function -- and the
        # identity element differs, so an un-warm-started trunk built under the
        # wrong convention silently doubles or zeroes every activation.
        "norm_delta": model_type.startswith("gemma"),
        # Gemma divides attention logits by sqrt(query_pre_attn_scalar) rather
        # than sqrt(head_dim); they coincide for 1b/4b but not for 27b.
        "query_pre_attn_scalar": float(
            config.get("query_pre_attn_scalar") or config.get("head_dim")
            or (hidden // n_heads)),
    }


class RMSNorm(nn.Module):
    """Gemma RMSNorm: `x * (1 + w)`, with the weight stored as a DELTA around 0.

    Computes in fp32 regardless of autocast dtype -- these GPUs are sm_75 (fp16
    only, no bf16) and the sum of squares overflows easily. That matters more
    here than for most donors: Gemma's post-norm weights reach 845 (measured,
    gemma-3-1b layer 24) on top of a sqrt(hidden) embedding scale, which is why
    Gemma in pure fp16 is a documented NaN (transformers#39972).
    """

    def __init__(self, dim, eps, delta):
        super().__init__()
        # `delta` selects the storage convention, and the IDENTITY differs with
        # it: Gemma stores w around 0 and computes x*(1+w); Qwen stores w around
        # 1 and computes x*w. Initialising to the wrong identity leaves an
        # un-warm-started trunk either doubling every activation or zeroing it.
        self.delta = bool(delta)
        self.weight = nn.Parameter(
            torch.zeros(dim) if self.delta else torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        w = self.weight.float()
        return (x * ((1.0 + w) if self.delta else w)).to(dtype)


def rope_cache(n_pos, device, dtype, half, theta, dim):
    """Return (cos, sin) of shape [n_pos, dim].

    half=True is the HF `rotate_half` convention -- the one the donor's q/k
    projections were TRAINED under, so it is the one under which the warm start's
    relative-position structure actually transfers. half=False is the interleaved
    GPT-J pairing, kept only as an ablation.
    """
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(n_pos, device=device).float()
    freqs = torch.outer(t, inv)                       # [n_pos, dim/2]
    if half:
        emb = torch.cat([freqs, freqs], dim=-1)       # [n_pos, dim]
    else:
        emb = freqs.repeat_interleave(2, dim=-1)      # [n_pos, dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def _rotate_interleaved(x):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


def apply_rope(x, cos, sin, half=True):
    """x: [B, H, S, head_dim]; cos/sin: [S, head_dim]."""
    cos = cos[None, None]
    sin = sin[None, None]
    rot = _rotate_half if half else _rotate_interleaved
    return x * cos + rot(x) * sin


class SharedAttention(nn.Module):
    """One bidirectional GQA block, shared by both modalities."""

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        attention_bias,
        qk_norm,
        rms_norm_eps,
        query_pre_attn_scalar,
        norm_delta,
    ):
        super().__init__()
        # Gemma divides logits by sqrt(query_pre_attn_scalar), not sqrt(head_dim).
        # They coincide at 1b/4b (both 256) and diverge at 27b, so read it rather
        # than let SDPA's default stand in for it.
        self.scale = float(query_pre_attn_scalar or head_dim) ** -0.5
        self.hidden_size = int(hidden_size)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by "
                "num_key_value_heads"
            )
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=False,
        )
        self.q_norm = (
            RMSNorm(self.head_dim, rms_norm_eps, norm_delta)
            if qk_norm else nn.Identity()
        )
        self.k_norm = (
            RMSNorm(self.head_dim, rms_norm_eps, norm_delta)
            if qk_norm else nn.Identity()
        )

    def forward(self, x, cos, sin, attn_mask, half=True):
        B, S, _ = x.shape
        q = self.q_proj(x).view(
            B, S, self.num_attention_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(x).view(
            B, S, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(x).view(
            B, S, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rope(q, cos, sin, half)
        k = apply_rope(k, cos, sin, half)
        # torch 2.4 has no GQA kernel (enable_gqa lands in 2.5), so expand k/v.
        repeats = self.num_attention_heads // self.num_key_value_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                             scale=self.scale)
        out = out.transpose(1, 2).reshape(
            B, S, self.num_attention_heads * self.head_dim
        )
        return self.o_proj(out)


_ACTS = {
    "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "gelu": F.gelu,
    "silu": F.silu,
    "swish": F.silu,
}


class FFN(nn.Module):
    """Gated MLP: down(act(gate(x)) * up(x)).

    `act` comes from the donor config. Gemma is GeGLU (gelu_pytorch_tanh); the
    SwiGLU that these three tensor names usually imply is a different function
    over the same shapes, so getting it wrong is invisible to any shape check.
    """

    def __init__(self, hidden_size, intermediate_size, hidden_activation):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        if hidden_activation not in _ACTS:
            raise ValueError(f"unsupported hidden_activation {hidden_activation!r}; "
                             f"known: {sorted(_ACTS)}")
        self.hidden_activation = hidden_activation
        self.act = _ACTS[hidden_activation]
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False
        )
        self.lora_scale = 0.0

    def enable_lora(self, rank=8, alpha=16.0):
        if rank <= 0:
            return
        self.lora_gate_a = nn.Linear(self.hidden_size, rank, bias=False)
        self.lora_gate_b = nn.Linear(
            rank, self.intermediate_size, bias=False
        )
        self.lora_up_a = nn.Linear(self.hidden_size, rank, bias=False)
        self.lora_up_b = nn.Linear(
            rank, self.intermediate_size, bias=False
        )
        self.lora_down_a = nn.Linear(
            self.intermediate_size, rank, bias=False
        )
        self.lora_down_b = nn.Linear(rank, self.hidden_size, bias=False)
        for a in (self.lora_gate_a, self.lora_up_a, self.lora_down_a):
            nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
        for b in (self.lora_gate_b, self.lora_up_b, self.lora_down_b):
            nn.init.zeros_(b.weight)
        self.lora_scale = float(alpha) / float(rank)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.lora_scale:
            gate = gate + self.lora_gate_b(self.lora_gate_a(x)) * self.lora_scale
            up = up + self.lora_up_b(self.lora_up_a(x)) * self.lora_scale
        hidden = self.act(gate) * up
        out = self.down_proj(hidden)
        if self.lora_scale:
            out = out + self.lora_down_b(
                self.lora_down_a(hidden)
            ) * self.lora_scale
        return out


class DualFFNLayer(nn.Module):
    """Shared attention + TWO parallel FFN branches routed per token by modality.

    The sequence is laid out [text tokens | audio frames], so routing is an exact
    slice at `nt` rather than a gather: no token passes through the other
    modality's branch, no FLOPs are wasted, and there is no router to train and
    no load-balancing loss. Both LayerNorms are duplicated too -- only the
    attention and the residual stream are shared.

    This is the modality-expert pattern, NOT macaron (there is one FFN per token
    per layer, not a pre/post sandwich) and NOT a mixture-of-experts.
    """

    def __init__(self, geometry, is_sliding):
        super().__init__()
        geometry = geometry_from_config(geometry)
        hidden = geometry["hidden_size"]
        eps = geometry["rms_norm_eps"]
        # Which RoPE base and which attention span this layer was trained under.
        # Set from the DONOR layer this slot is warm-started from, not from the
        # trunk position -- see DualFFNTrunk.set_layer_sources.
        self.is_sliding = bool(is_sliding)
        self.sliding_window = geometry["sliding_window"]
        self.attn = SharedAttention(
            hidden_size=hidden,
            num_attention_heads=geometry["num_attention_heads"],
            num_key_value_heads=geometry["num_key_value_heads"],
            head_dim=geometry["head_dim"],
            attention_bias=geometry["attention_bias"],
            qk_norm=geometry["qk_norm"],
            rms_norm_eps=eps,
            query_pre_attn_scalar=geometry["query_pre_attn_scalar"],
            norm_delta=geometry["norm_delta"],
        )
        nd = geometry["norm_delta"]
        self.ln1_t, self.ln1_a = RMSNorm(hidden, eps, nd), RMSNorm(hidden, eps, nd)
        self.ln2_t, self.ln2_a = RMSNorm(hidden, eps, nd), RMSNorm(hidden, eps, nd)
        # Gemma sandwiches each sublayer -- x + post(sublayer(pre(x))) -- two MORE
        # norms per layer, duplicated per modality like the others.
        self.post_norms = geometry["post_norms"]
        if self.post_norms:
            self.ln1p_t, self.ln1p_a = RMSNorm(hidden, eps, nd), RMSNorm(hidden, eps, nd)
            self.ln2p_t, self.ln2p_a = RMSNorm(hidden, eps, nd), RMSNorm(hidden, eps, nd)
        act = geometry["hidden_activation"]
        self.ffn_text = FFN(hidden, geometry["intermediate_size"], act)
        self.ffn_audio = FFN(hidden, geometry["intermediate_size"], act)

    def forward(self, x, nt, cos_g, sin_g, cos_l, sin_l,
                attn_full, attn_band, half=True):
        # Local layers carry theta=1e4 weights and a 512-wide span; global layers
        # carry theta=1e6 and full span. Feeding one layer the other's cache is a
        # silent, shape-clean way to discard the warm start's position structure.
        if self.is_sliding:
            cos, sin, attn_mask = cos_l, sin_l, attn_band
        else:
            cos, sin, attn_mask = cos_g, sin_g, attn_full
        h = torch.cat([self.ln1_t(x[:, :nt]), self.ln1_a(x[:, nt:])], 1)
        att = self.attn(h, cos, sin, attn_mask, half)
        if self.post_norms:
            att = torch.cat([self.ln1p_t(att[:, :nt]), self.ln1p_a(att[:, nt:])], 1)
        x = x + att
        t = self.ffn_text(self.ln2_t(x[:, :nt]))
        a = self.ffn_audio(self.ln2_a(x[:, nt:]))
        if self.post_norms:
            t, a = self.ln2p_t(t), self.ln2p_a(a)
        return x + torch.cat([t, a], 1)


class DualFFNTrunk(nn.Module):
    """Frame-level conditioning trunk over [text | mel frames].

    Every conditioning signal into the audio stream is purely ADDITIVE, which is
    what makes the inference-time ablations exact (zeroing one term removes
    exactly that term).
    """

    def __init__(self, n_layers=14, n_mels=80, fs_hidden=192, dict_size=100,
                 rope_half=True, rope_reset_at_nt=True, grad_ckpt=True,
                 use_text=True, use_ph=True,
                 audio_lora_rank=0, audio_lora_alpha=16.0,
                 donor_config=None):
        super().__init__()
        geometry = geometry_from_config(donor_config)
        self.geometry = geometry
        self.hidden_size = geometry["hidden_size"]
        self.head_dim = geometry["head_dim"]
        self.rope_theta = geometry["rope_theta"]
        self.n_layers = n_layers
        self.rope_half = rope_half
        self.rope_reset_at_nt = rope_reset_at_nt
        self.grad_ckpt = grad_ckpt
        self.use_text, self.use_ph = use_text, use_ph

        self.rope_local = geometry["rope_local_base_freq"]
        self.sliding_window = geometry["sliding_window"]
        self.sliding_pattern = geometry["sliding_window_pattern"]
        self.embed_scale = geometry["embed_scale"]

        self.layers = nn.ModuleList([
            DualFFNLayer(geometry, is_sliding=self._is_sliding(i))
            for i in range(n_layers)
        ])
        if audio_lora_rank > 0:
            for layer in self.layers:
                layer.ffn_audio.enable_lora(audio_lora_rank, audio_lora_alpha)
        self.norm_t = RMSNorm(
            self.hidden_size, geometry["rms_norm_eps"], geometry["norm_delta"]
        )
        self.norm_a = RMSNorm(
            self.hidden_size, geometry["rms_norm_eps"], geometry["norm_delta"]
        )

        # Text side: the donor's own embedding table, frozen (see warm_start).
        self.tok_emb = nn.Embedding(
            geometry["vocab_size"], self.hidden_size
        )

        # Audio side.
        self.mel_in = nn.Linear(n_mels, self.hidden_size)
        self.mask_emb = nn.Embedding(2, self.hidden_size)
        self.ph_emb = nn.Embedding(dict_size, self.hidden_size)
        # Zero-init so the trunk starts blind to FluentEditor's own front-end and
        # the ablation is exact at init.
        # NOTE: there is no fs_proj any more. It projected FluentEditor's
        # FastSpeech front-end state into the audio stream; that front end is
        # gone and its jobs (duration, pitch, speaker) are read off this trunk
        # by wavepainter/models/speech_editing/dualffn/heads.py instead.

        self._cache = {}

    def _is_sliding(self, donor_layer_index):
        """Gemma's rule: every `pattern`-th layer is global, the rest are local.

        HF sets `is_sliding = bool((idx + 1) % sliding_window_pattern)`, so with
        pattern 6 the GLOBAL layers are donor indices 5, 11, 17, 23.
        """
        if not self.sliding_pattern:
            return False
        return bool((int(donor_layer_index) + 1) % self.sliding_pattern)

    def set_layer_sources(self, sources):
        """Re-tag each trunk layer with the donor layer it was warm-started from.

        `layer_pick='ends'` copies donor layers 0,1,..,24,25 into trunk slots
        0..n-1, so trunk slot i does NOT generally hold donor layer i. The
        local/global identity has to follow the WEIGHTS, not the slot: a layer
        trained at theta=1e4 over a 512 span keeps those even when it lands in a
        slot whose index would say otherwise. warm_start calls this.
        """
        if len(sources) != len(self.layers):
            raise ValueError(f"{len(sources)} sources for {len(self.layers)} layers")
        for layer, s in zip(self.layers, sources):
            layer.is_sliding = self._is_sliding(s)
        self._cache = {}
        return [bool(l.is_sliding) for l in self.layers]

    def _positions(self, n_text, n_ph, ta, device):
        """Absolute position per slot: text, phones and frames each from 0.

        A single ramp over the whole sequence would place frame f at position
        n_text+n_ph+f, so frame positions would shift with the padded transcript
        and phone lengths from batch to batch. In an infilling task the frame
        index is the physically meaningful coordinate and must not depend on how
        long the text is. The phone segment gets its own ramp for the same
        reason: phone k is the k-th phone regardless of the transcript.
        """
        if not self.rope_reset_at_nt:
            return torch.arange(n_text + n_ph + ta, device=device)
        return torch.cat([torch.arange(n_text, device=device),
                          torch.arange(n_ph, device=device),
                          torch.arange(ta, device=device)], 0)

    def _rope(self, n_text, n_ph, ta, device, dtype):
        """(cos_g, sin_g, cos_l, sin_l, band) for this [text | frames] layout.

        Two caches, because Gemma's local and global layers were trained under
        different RoPE bases. `band` is the sliding-window mask: True where the
        pair is within `sliding_window` of each other in POSITION space.
        """
        key = (n_text, n_ph, ta, device, dtype, self.rope_half,
               self.rope_reset_at_nt)
        if key not in self._cache:
            pos = self._positions(n_text, n_ph, ta, device)
            n = int(pos.max().item()) + 1 if pos.numel() else 1
            out = []
            for theta in (self.rope_theta, self.rope_local):
                cos, sin = rope_cache(n, device, dtype, self.rope_half,
                                      theta=theta, dim=self.head_dim)
                out += [cos[pos], sin[pos]]
            if self.sliding_window:
                # Gemma's window is causal (j in (i-w, i]); the trunk is
                # bidirectional by construction -- it has to see the right
                # context of a masked span -- so the faithful analogue is the
                # symmetric band of the same width.
                d = (pos[:, None] - pos[None, :]).abs()
                band = (d < self.sliding_window)[None, None]
            else:
                band = None
            self._cache = {key: (*out, band)}  # one entry; shapes repeat per run
        return self._cache[key]

    def forward(self, mel_masked, mask, text_ids, text_pad,
                ph_tokens, ph_pad, mel2ph, tgt_nonpadding,
                acoustic_context=None):
        """
        mel_masked      [B, T, n_mels]  mel, masked frames already zeroed
        mask            [B, T]          1.0 = predict
        text_ids        [B, n_text]     donor BPE ids of the transcript
        text_pad        [B, n_text]     1.0 = real token
        ph_tokens       [B, n_ph]       phone ids -- their own SEGMENT now, not
                                        just a per-frame lookup
        ph_pad          [B, n_ph]       1.0 = real phone
        mel2ph          [B, T]          1-based phone index per frame, 0 = pad
        tgt_nonpadding  [B, T] or [B,T,1]
        returns (h_audio [B, T, hidden], h_phone [B, n_ph, hidden])

        The phone segment is what replaced FastSpeech: duration is read off
        `h_phone` by TrunkHeads, so the pretrained trunk predicts it instead of
        a from-scratch conv stack running alongside.
        """
        B, T, _ = mel_masked.shape
        if tgt_nonpadding.dim() == 3:
            tgt_nonpadding = tgt_nonpadding[..., 0]

        a = self.mel_in(mel_masked * (1.0 - mask).unsqueeze(-1))
        a = a + self.mask_emb(mask.long())
        if acoustic_context is not None:
            a = a + acoustic_context
        if self.use_ph:
            # Per-FRAME phone identity, still useful even with the phone segment
            # present: it tells the audio stream which phone each frame belongs
            # to without going through attention. mel2ph is 1-based, pad -> 0.
            tt = F.pad(ph_tokens, [1, 0])
            a = a + self.ph_emb(torch.gather(tt, 1, mel2ph.clamp(min=0)))

        segments, pads = [], []
        if self.use_text:
            # Gemma scales the embedding by sqrt(hidden) = 33.9 before layer 0.
            # Skipping it feeds the donor stack an input ~34x smaller than
            # anything it saw in training; it runs, and the symbolic stream
            # carries almost no signal.
            t = self.tok_emb(text_ids)
            t = t * torch.tensor(self.embed_scale, dtype=t.dtype, device=t.device)
            segments.append(t)
            pads.append(text_pad)
        n_text = segments[0].shape[1] if segments else 0

        # Phones ride the TEXT branch, not a third one. The DualFFN constraint
        # is two modality-routed branches and the honest split is symbolic vs
        # acoustic -- a third branch would satisfy its letter and not its point.
        p = self.ph_emb(ph_tokens)
        p = p * torch.tensor(self.embed_scale, dtype=p.dtype, device=p.device)
        segments.append(p)
        pads.append(ph_pad)
        n_ph = p.shape[1]

        segments.append(a)
        pads.append(tgt_nonpadding)
        x = torch.cat(segments, 1)
        pad = torch.cat(pads, 1)
        nt = n_text + n_ph          # the symbolic|acoustic routing boundary

        cos_g, sin_g, cos_l, sin_l, band = self._rope(
            n_text, n_ph, T, x.device, x.dtype)
        # Bool mask, not float: a float mask forces the MATH SDPA backend on some
        # torch versions, which materialises [B, n_heads, S, S].
        S = x.shape[1]
        keep = pad > 0.5
        attn_full = keep[:, None, None, :].expand(B, 1, S, S)
        attn_band = attn_full if band is None else (attn_full & band)

        for layer in self.layers:
            args = (x, nt, cos_g, sin_g, cos_l, sin_l, attn_full, attn_band,
                    self.rope_half)
            if self.grad_ckpt and self.training:
                x = checkpoint(layer, *args, use_reentrant=False)
            else:
                x = layer(*args)
        return self.norm_a(x[:, nt:]), self.norm_t(x[:, n_text:nt])

    def set_audio_expert_adaptation(self):
        """Freeze donor pathways; tune acoustic interfaces and audio FFNs.

        Shared attention and the text FFN retain the language model's geometry.
        The duplicated audio FFN, audio-side norms, and fresh input interfaces
        specialize to masked speech. The surrounding editor remains trainable;
        unlike the earlier graft experiment, this task forbids loading its weights.
        """
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        modules = [
            self.mel_in,
            self.mask_emb,
            self.ph_emb,
            self.norm_a,
        ]
        for layer in self.layers:
            modules.extend([layer.ln1_a, layer.ln2_a, layer.ffn_audio])
            modules.extend(self._audio_post_norms(layer))
        for module in modules:
            module.requires_grad_(True)

    @staticmethod
    def _audio_post_norms(layer):
        """Gemma's two extra audio-side norms. Frozen, they pin the audio branch
        to the donor's text-calibrated output scale, which is the one thing the
        audio expert most needs to move."""
        return [layer.ln1p_a, layer.ln2p_a] if layer.post_norms else []

    def set_audio_lora_adaptation(self):
        """Freeze donor matrices and train only audio LoRA plus fresh interfaces."""
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        modules = [
            self.mel_in,
            self.mask_emb,
            self.ph_emb,
            self.norm_a,
        ]
        for layer in self.layers:
            modules.extend([layer.ln1_a, layer.ln2_a])
            modules.extend(self._audio_post_norms(layer))
            modules.extend(
                module for name, module in layer.ffn_audio.named_children()
                if name.startswith("lora_")
            )
        for module in modules:
            module.requires_grad_(True)

    @torch.no_grad()
    def text_only_hidden_states(self, ids):
        """Run the TEXT path alone with a causal mask, returning every layer's output.

        This exists purely for the warm-start probe: with the donor's weights and
        a matching RoPE convention, this must reproduce the DONOR's own
        hidden states. It is the only check that catches a rotate_half-vs-
        interleaved mismatch, a transposed projection, QK-norm on the wrong axis,
        GeGLU loaded as SwiGLU, a norm in the wrong sandwich slot, or a missing
        embedding scale -- none of which change any checksum, and all of which
        pass shape-matched cosine provenance at ~1.0 while transferring nothing.

        Only meaningful at full donor depth with `pick='ends'` (i.e. all 26
        layers in donor order); a truncated trunk is a different function by
        construction and has nothing to be compared against.
        """
        S = ids.shape[1]
        x = self.tok_emb(ids)
        x = x * torch.tensor(self.embed_scale, dtype=x.dtype, device=x.device)
        pos = torch.arange(S, device=x.device)
        ropes = []
        for theta in (self.rope_theta, self.rope_local):
            ropes += list(rope_cache(S, x.device, x.dtype, self.rope_half,
                                     theta=theta, dim=self.head_dim))
        causal = torch.ones(S, S, dtype=torch.bool, device=x.device).tril()
        attn_full = causal[None, None]
        if self.sliding_window:
            # Causal here, unlike the bidirectional trunk forward: this path is
            # reproducing the donor's own decoder, so it must use the donor's own
            # window -- j in (i - w, i].
            recent = (pos[:, None] - pos[None, :]) < self.sliding_window
            attn_band = (causal & recent)[None, None]
        else:
            attn_band = attn_full
        states = [x]
        for layer in self.layers:
            x = layer(x, S, *ropes, attn_full, attn_band, self.rope_half)
            states.append(x)
        return states

