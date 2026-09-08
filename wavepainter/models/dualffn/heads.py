"""Duration, pitch and speaker heads ON the DualFFN trunk.

This replaces FluentEditor's `FastSpeech` front end entirely. Nothing here is
their topology; what remains from the shared codebase is arithmetic every
non-autoregressive editor needs -- `LengthRegulator` expands a per-phone value
to per-frame, `mel2token_to_dur` inverts an alignment. Those are not a design.

WHY THIS MOVED. The old front end was a from-scratch conv stack that predicted
durations and pitch, and the pretrained trunk merely consumed its output. That
is backwards for a task whose whole claim is composition: the linguistic work --
how long each phone is, what pitch it carries -- is exactly what a pretrained
language model should be good at, and exactly what a randomly-initialised
4-layer conv encoder has to learn from 52 hours.

    old:  phones -> FastSpeech conv encoder -> dur/pitch -> trunk -> diffusion
    new:  phones -> GEMMA TRUNK -> dur/pitch -> diffusion

It also removes the last piece of FluentEditor's architecture from the model, so
"we beat FluentSpeech" stops meaning "we beat it while sharing its front end".

WHERE THE PHONES GO. The trunk's sequence becomes

    [ BPE text | phones | mel frames ]
      \____ symbolic ____/ \_ acoustic _/
           ffn_text            ffn_audio

Phones join the TEXT branch rather than getting a third. The DualFFN constraint
is two modality-routed branches, and the honest split is symbolic vs acoustic --
a third branch for phones would satisfy the letter of the constraint and not its
point. It also means the phone stream is processed by donor weights that already
model symbol sequences, which is the whole reason for warm-starting.
"""

import math

import torch
import torch.nn as nn

from wavepainter.models.nar_tts_modules import LengthRegulator


class TrunkHeads(nn.Module):
    """Duration, pitch and speaker conditioning read off the trunk's own output.

    All three heads are small linear read-outs. The representation they read is
    the trunk's, so the capacity lives in the pretrained weights rather than in
    a predictor trained from scratch beside them.
    """

    def __init__(self, hidden, cond_dim, spk_dim=256):
        super().__init__()
        self.hidden = int(hidden)

        # Per-PHONE, from the phone segment. One scalar: log(duration + 1), the
        # standard parameterisation -- durations are positive and heavy-tailed,
        # and predicting them in log space keeps a 40-frame vowel from dominating
        # the loss of a 2-frame stop.
        self.dur_head = nn.Linear(self.hidden, 1)
        # Zero weight, bias at the corpus prior. Default init is measurably bad
        # here: `norm_t` is the donor's own final RMSNorm, whose weights reach
        # ~800, so a fresh read-out over that produced durations spanning
        # [0, 413] with some phones pinned in the `clamp(min=0)` dead zone where
        # they get no gradient at all. Starting at "every phone is the mean
        # length" took step-0 wdur from 18.6 to 0.42.
        #
        # log1p(4.0): measured over 10,156 LibriTTS phones -- mean 5.87, median
        # 5, and 3.74 is the optimum in log space, which log1p(4)=1.609 is the
        # nearest sane round number to in the parameterisation used here.
        nn.init.zeros_(self.dur_head.weight)
        nn.init.constant_(self.dur_head.bias, math.log1p(4.0))

        # Per-FRAME, from the audio segment. Two outputs: normalised f0 and the
        # unvoiced logit. Pitch inside the masked span must be PREDICTED -- the
        # protocol zeroes it and never hands it over, which is what closed the
        # GT-F0 leak.
        self.pitch_head = nn.Linear(self.hidden, 2)

        # Speaker identity, additive into the audio stream before the trunk.
        # A resemblyzer embedding, not FluentEditor's style module.
        self.spk_proj = nn.Linear(spk_dim, self.hidden)

        # Trunk output -> the diffusion decoder's conditioning width.
        self.cond_proj = nn.Linear(self.hidden, int(cond_dim))

        self.length_regulator = LengthRegulator()

    def durations(self, h_phone, ph_padding=None):
        """[B, n_ph] predicted durations in FRAMES, >= 0."""
        log_dur = self.dur_head(h_phone).squeeze(-1)
        dur = torch.clamp(torch.expm1(log_dur), min=0.0)
        if ph_padding is not None:
            dur = dur * (~ph_padding).float()
        return dur, log_dur

    def pitch(self, h_audio):
        """([B, T] normalised f0, [B, T] uv logit) per frame."""
        out = self.pitch_head(h_audio)
        return out[..., 0], out[..., 1]

    def mel2ph_from_dur(self, dur, ph_padding=None):
        """Expand per-phone durations into a per-frame phone index (1-based).

        This is `LengthRegulator`, a generic NAR primitive: it is the inverse of
        a forced alignment, not an architecture. Predicting durations is what
        lets the edited utterance change LENGTH, which is the only reason
        insertion and deletion are expressible at all.
        """
        return self.length_regulator(dur, ph_padding)

    def condition(self, h_audio):
        """Trunk audio output -> diffusion conditioning."""
        return self.cond_proj(h_audio)
