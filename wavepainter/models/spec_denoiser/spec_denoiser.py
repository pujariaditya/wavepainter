from functools import partial
from wavepainter.models.spec_denoiser.diffusion_utils import (
    default, exists, extract, get_noise_schedule_list)
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from wavepainter.models.layers import Embedding
from wavepainter.models.mel_encoder import MelEncoder
from wavepainter.models.dualffn.heads import TrunkHeads
from wavepainter.models.align_ops import clip_mel2token_to_multiple, expand_states
from wavepainter.audio.pitch.utils import denorm_f0, f0_to_coarse
from wavepainter.runtime.hparams import hparams


class GaussianDiffusion(nn.Module):
    def __init__(self, phone_encoder, out_dims, denoise_fn,
                 timesteps=1000, time_scale=1, loss_type='l1', betas=None, spec_min=None, spec_max=None):
        super().__init__()
        self.denoise_fn = denoise_fn
        # FluentEditor's FastSpeech front end is GONE. Duration, pitch and speaker
        # conditioning are now read off the pretrained trunk by TrunkHeads -- see
        # wavepainter/models/speech_editing/dualffn/heads.py for why that is the
        # right way
        # round. `hidden_size` is the width that used to be `self.fs.hidden_size`;
        # every consumer of it (MelEncoder, cond_dim, DiffNet's conditioner Conv1d)
        # reads the same hparam, so the diffusion decoder is untouched.
        self.hidden_size = hparams['hidden_size']
        # input_dim from the config, not MelEncoder's default of 80. Everything
        # else on this path (DiffNet, spec_min/max) already reads
        # audio_num_mel_bins, so this was the one place a 100-band front end hit
        # an 80-wide Linear.
        self.mel_encoder = MelEncoder(input_dim=hparams['audio_num_mel_bins'],
                                      hidden_size=self.hidden_size)
        self.mel_bins = out_dims

        # ---- DualFFN conditioning trunk (MANDATORY) -------------------------
        # It is no longer "in parallel with FluentEditor's front end": it IS the
        # front end. It still sits outside the reverse diffusion loop -- its
        # input never depends on the diffusion timestep, so one trunk pass feeds
        # all `timesteps` DiffNet passes.
        self.use_trunk = hparams.get('use_dualffn_trunk', False)
        self.trunk_inject = hparams.get('trunk_inject', 'add')
        if not self.use_trunk:
            # Refuse at CONSTRUCTION. Without the trunk nothing predicts duration
            # or pitch at all, so `ret['dur']`/`ret['pitch_pred']` would not exist
            # and the model could not decide how long an edited span is. A build
            # that got that far would either die on a KeyError several minutes
            # into training, or -- if a caller used `.get()` -- train a
            # length-blind editor that reports perfectly finite losses.
            raise ValueError(
                "use_dualffn_trunk is False, but the trunk is not optional any "
                "more: it is the only thing in this model that predicts phone "
                "durations and frame pitch since the FastSpeech front end was "
                "removed. Set use_dualffn_trunk=True.")
        if self.trunk_inject == 'concat':
            # 'concat' concatenated the trunk's conditioning with FastSpeech's.
            # There is no second conditioner left to concatenate with, so a
            # config that still asks for it would build a DiffNet at 2*hidden
            # and feed it a hidden-wide tensor.
            raise ValueError(
                "trunk_inject='concat' concatenated the trunk's conditioning "
                "with FastSpeech's; that second conditioner no longer exists. "
                "Use trunk_inject='add'.")
        # Always taken (the raise above is the only other exit). Kept as a block
        # so the diff against the optional-trunk version stays readable.
        if self.use_trunk:
            from wavepainter.models.dualffn.trunk import DualFFNTrunk
            from wavepainter.models.dualffn.warm_start import (
                load_donor_config,
            )
            trunk_donor = hparams['trunk_donor']
            trunk_donor_config = load_donor_config(trunk_donor)
            self.trunk = DualFFNTrunk(
                n_layers=hparams.get('trunk_layers', 14),
                n_mels=out_dims,
                dict_size=len(phone_encoder),
                rope_half=hparams.get('rope_half', True),
                rope_reset_at_nt=hparams.get('rope_reset_at_nt', True),
                grad_ckpt=hparams.get('trunk_grad_ckpt', True),
                use_text=hparams.get('trunk_use_text', True),
                use_ph=hparams.get('trunk_use_ph', True),
                audio_lora_rank=(
                    hparams.get('trunk_lora_rank', 8)
                    if hparams.get('trunk_adaptation', 'audio_lora') == 'audio_lora'
                    else 0
                ),
                audio_lora_alpha=hparams.get('trunk_lora_alpha', 16.0),
                donor_config=trunk_donor_config,
            )
            trunk_dim = self.trunk.hidden_size
            # WARM START. Without this the trunk is 350M+ RANDOM parameters and
            # the entire pretrained-LLM premise is absent -- the loss still
            # falls, the numbers are self-consistent, and nothing complains. It
            # also leaves `tok_emb` (151936x1024 = 155.6M) trainable, which costs
            # 1.87 GB of AdamW state and a dense 622 MB gradient every step for
            # embeddings that are never meaningfully updated.
            #
            # Runs at construction; a later load_ckpt (resume/init_from) simply
            # overwrites these weights, which is the correct precedence.
            if hparams.get('trunk_warm_start', True):
                from wavepainter.models.dualffn.warm_start import warm_start
                self.trunk_warm_start_report = warm_start(
                    self.trunk,
                    donor=trunk_donor,
                    pick=hparams.get('trunk_layer_pick', 'ends'),
                    freeze_tok_emb=hparams.get('trunk_freeze_tok_emb', True),
                )
            else:
                self.trunk_warm_start_report = None
                print('| WARNING: trunk_warm_start is FALSE -- the DualFFN trunk is '
                      'randomly initialised and the Qwen transfer premise does not hold.')

            adaptation = hparams.get('trunk_adaptation', 'audio_expert')
            if adaptation == 'audio_expert':
                self.trunk.set_audio_expert_adaptation()
            elif adaptation == 'audio_lora':
                self.trunk.set_audio_lora_adaptation()
            elif adaptation != 'full':
                raise ValueError(f"unknown trunk_adaptation {adaptation!r}")

            # ---- the heads that replaced FastSpeech -------------------------
            # cond_dim matches what DiffNet builds its conditioner Conv1d from
            # (`hparams.get('cond_dim') or hparams['hidden_size']`, diffnet.py:119).
            # Reading it differently here is a shape error at best and a silently
            # mis-widened conditioner at worst.
            self.heads = TrunkHeads(
                hidden=trunk_dim,
                cond_dim=hparams.get('cond_dim') or self.hidden_size,
                spk_dim=256,
            )
            # NOT zero-initialised, unlike the `trunk_out` it replaces. That one
            # was zero because the trunk was a residual ADDITION on top of
            # FastSpeech's conditioning, so zero made step 0 bit-exactly
            # FluentEditor. There is no FastSpeech conditioning left to add to:
            # zeroing this would hand DiffNet an all-zero conditioner, i.e. train
            # unconditional mel denoising -- which converges, reports a falling
            # loss, and generates babble.
            #
            # The duration head starts at a CONSTANT prior: zero weight, bias
            # log1p(8). Two reasons, both measured on this trunk.
            #
            # `TrunkHeads.durations` is expm1(log_dur).clamp(min=0), and clamp has
            # ZERO gradient below its bound -- so any phone whose log_dur starts
            # negative is stuck at a duration of exactly 0 with no way back.
            # Default Linear init does exactly that here: `norm_t` is the donor's
            # own final RMSNorm, whose weights reach ~800, so h_phone elements are
            # O(100) and a fresh 1152-wide read-out of them spreads log_dur over
            # several units. Measured before this init: durations spanning
            # [0, 413] frames with a mean of 26, i.e. some phones already dead and
            # the rest ~30x too long, giving wdur ~21 at step 0.
            #
            # 4 frames is the MEASURED corpus prior, not a guess: over 10,156
            # phones from 200 binarised LibriTTS train items the durations have
            # mean 5.87 and median 5, and `add_dur_loss` works in log1p space, so
            # the constant that minimises it is expm1(mean(log1p(d))) = 3.74.
            # Guessing 8 from "85 ms per phone" was 2x too long and cost ~2 units
            # of wdur at step 0. Zero weight does not freeze the head: its
            # gradient is dL/dout (x) h_phone, nonzero from the first step.

            self.use_pitch_embed = hparams['use_pitch_embed']
            if self.use_pitch_embed:
                # Two pitch embeddings at two widths, doing two different jobs.
                #
                # `trunk_pitch_embed` gives the trunk the pitch it is ALLOWED to
                # see: the context f0, zeroed inside the masked span. That is
                # exactly what FastSpeech handed its pitch predictor. Dropping it
                # would leave the trunk's pitch head worse-informed than the
                # module it replaces and confound any comparison between them.
                #
                # `pitch_embed` puts the RESOLVED pitch (ground truth, or the
                # prediction inside the span when use_pred_pitch is on) into the
                # diffusion conditioning. That pathway is the only thing
                # use_pred_pitch controls; without it the flag becomes a no-op and
                # the GT-F0 leak it closed reopens as "f0 never mattered".
                self.trunk_pitch_embed = Embedding(300, trunk_dim, 0)
                self.pitch_embed = Embedding(300, self.hidden_size, 0)
        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = get_noise_schedule_list(
                schedule_mode=hparams['schedule_type'],
                timesteps=timesteps + 1,
                min_beta=0.1,
                max_beta=40,
                s=0.008,
            )

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.time_scale = time_scale
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('timesteps', to_torch(self.num_timesteps))      # beta
        self.register_buffer('timescale', to_torch(self.time_scale))      # beta
        self.register_buffer('betas', to_torch(betas))      # beta
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod)) # alphacum_t
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev)) # alphacum_{t-1}

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        self.register_buffer('spec_min', torch.FloatTensor(spec_min)[None, None, :hparams['keep_bins']])
        self.register_buffer('spec_max', torch.FloatTensor(spec_max)[None, None, :hparams['keep_bins']])


    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_posterior_sample(self, x_start, x_t, t, repeat_noise=False):
        # Use the existing DDPM posterior mean at every transition. The seeded
        # initial latent remains stochastic, but the short reverse chain does
        # not inject a fresh high-variance draw after each denoiser call.
        model_mean, _, _ = self.q_posterior(
            x_start=x_start, x_t=x_t, t=t)
        return model_mean

    @torch.no_grad()
    def p_sample(self, x_t, t, cond, spk_emb=None, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x_t.shape, x_t.device
        x_0_pred = self.denoise_fn(x_t, t, cond)

        return self.q_posterior_sample(x_start=x_0_pred, x_t=x_t, t=t)


    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )


    def diffuse_fn(self, x_start, t, noise=None):
        x_start = self.norm_spec(x_start)
        x_start = x_start.transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]
        zero_idx = t < 0 # for items where t is -1
        t[zero_idx] = 0
        noise = default(noise, lambda: torch.randn_like(x_start))
        out = self.q_sample(x_start=x_start, t=t, noise=noise)
        out[zero_idx] = x_start[zero_idx] # set x_{-1} as the gt mel
        return out

    @property
    def trunk_out(self):
        """The trunk -> diffusion-conditioning projection, under its old name.

        `eval_protocol.py` and `eval_ming.py` read `model.trunk_out.weight` as a
        structural fact about the checkpoint, so the name has to survive. This is
        a PROPERTY, not a second module: assigning `self.trunk_out =
        self.heads.cond_proj` would register the same Linear twice, which
        duplicates it in `state_dict()` and hands AdamW two parameter entries
        pointing at one tensor.
        """
        return self.heads.cond_proj

    def _masked_coarse_pitch(self, f0, uv, m, pitch_padding):
        """Coarse pitch of the CONTEXT only -- the span is zeroed.

        Same arithmetic FastSpeech used to build its pitch predictor's input. The
        span must be zeroed here and not merely down-weighted: at inference the
        protocol never supplies f0 inside the edit, so anything the trunk learns
        to read from it at training is a signal that is absent when it counts.
        """
        use_uv = hparams['pitch_type'] == 'frame' and hparams['use_uv']
        masked_f0 = f0 * (1 - m)
        masked_uv = uv * (1 - m)
        denorm = denorm_f0(masked_f0, masked_uv if use_uv else None,
                           pitch_padding=pitch_padding)
        return f0_to_coarse(denorm)

    def _acoustic_context(self, mel_masked, m, tgt_nonpadding, spk_embed, f0, uv,
                          mel2ph):
        """Everything additive into the trunk's AUDIO stream, at trunk width.

        Speaker identity enters HERE rather than being folded into `mel_in`'s
        output inside trunk.py. `acoustic_context` is the trunk's one documented
        additive slot, and keeping every conditioning term additive is what makes
        the inference-time ablations exact -- zeroing a term removes exactly that
        term and nothing else. The phone segment sees the speaker through
        attention to these frames; FastSpeech instead added its style embedding
        to the phone encoder directly, so this is the one place the two designs
        route the same information differently.
        """
        if spk_embed is None:
            raise RuntimeError(
                "spk_embed is None. The speaker vector is the model's only "
                "route to voice identity now that FastSpeech's style embedding "
                "is gone; running without it would train a speaker-averaged "
                "editor and report entirely normal losses.")
        if not spk_embed.is_floating_point():
            raise RuntimeError(
                f"spk_embed has dtype {spk_embed.dtype}; TrunkHeads.spk_proj "
                "expects a 256-d resemblyzer vector, not speaker IDs. "
                "use_spk_id is not supported by this front end.")
        # [B, 1, HID]: broadcast over frames rather than materialised.
        ctx = self.heads.spk_proj(spk_embed)[:, None, :]
        if self.use_pitch_embed:
            ctx = ctx + self.trunk_pitch_embed(
                self._masked_coarse_pitch(f0, uv, m, mel2ph == 0))
        return ctx

    @torch.no_grad()
    def predict_durations(self, txt_tokens, time_mel_masks, mel2ph, spk_embed,
                          ref_mels, f0=None, uv=None,
                          text_ids=None, text_pad=None):
        """Per-phone durations in FRAMES for `txt_tokens`, from one trunk pass.

        This is the inference-time FIRST pass of the mel2ph chicken-and-egg (see
        `forward`). The caller hands in an alignment it already knows -- typically
        the original one with the edited span zeroed -- and gets back durations
        for the phone sequence it is asking about, which may be longer or shorter
        than the one that alignment came from. `eval_ming.edit_one` then splices
        the predicted span durations into the original timeline and calls
        `forward` with the completed alignment.

        `mel2ph` here is deliberately NOT the frame-validity mask. Padded frames
        and MASKED frames both have mel2ph == 0, and treating a masked frame as
        padding would drop it out of the trunk's attention mask entirely -- the
        model would be reasoning about an utterance with the edit region cut out
        rather than blanked, and would still return plausible durations.
        """
        m = time_mel_masks[..., 0] if time_mel_masks.dim() == 3 else time_mel_masks
        b, T = m.shape
        if f0 is None:
            f0 = m.new_zeros((b, T))
        if uv is None:
            uv = torch.ones_like(f0)
        nonpadding = ((mel2ph > 0) | (m > 0.5)).float()[:, :, None]
        mel_masked = ref_mels * (1 - m).unsqueeze(-1)
        ctx = self._acoustic_context(mel_masked, m, nonpadding, spk_embed, f0, uv,
                                     mel2ph)
        qi, qp = self._text_segment(txt_tokens, text_ids, text_pad, nonpadding)
        ph_padding = txt_tokens == 0
        _, h_phone = self.trunk(mel_masked, m, qi, qp, txt_tokens,
                                (~ph_padding).float(), mel2ph, nonpadding, ctx)
        dur, _ = self.heads.durations(h_phone, ph_padding)
        return dur

    @staticmethod
    def _text_segment(txt_tokens, text_ids, text_pad, tgt_nonpadding):
        if text_ids is None:
            # Running without the BPE segment is legitimate ONLY as a declared
            # ablation (`trunk_use_text: false`). When the trunk was built with
            # the text branch on, a missing `text_ids` is a caller bug, and the
            # zero-width segment below is the most dangerous possible response
            # to it: the model runs, the loss is finite, the audio is plausible,
            # and the entire text half of the DualFFN (tok_emb, ffn_text, ln1_t,
            # ln2_t, ln1p_t, ln2p_t) is silently bypassed.
            #
            # This has now bitten twice. First in `validation_step`'s plotting
            # branch, where the plotted audio came from a text-less model while
            # the logged losses came from a text-ful one. Then -- for the entire
            # life of the Ming protocol -- in `eval_ming.build_inputs`, which
            # never supplied the tokens at all, so EVERY published number
            # described a model with its pretrained language half switched off.
            # Both were invisible precisely because this branch is quiet.
            if (hparams.get('use_dualffn_trunk')
                    and hparams.get('trunk_use_text', True)):
                raise RuntimeError(
                    "text_ids is None but this trunk was built with "
                    "trunk_use_text=True, so the BPE segment is part of the "
                    "model. Running anyway would bypass the text branch and "
                    "report numbers for a different architecture. Pass "
                    "text_ids/text_pad, or set trunk_use_text=False to declare "
                    "the ablation.")
            b = txt_tokens.shape[0]
            return (txt_tokens.new_zeros((b, 0)),
                    tgt_nonpadding.new_zeros((b, 0)))
        return text_ids, text_pad

    def forward(self, txt_tokens, time_mel_masks, mel2ph, spk_embed,
                ref_mels, f0, uv, energy=None,
                infer=False, use_pred_mel2ph=False, use_pred_pitch=False,
                text_ids=None, text_pad=None):
        """Trunk -> duration/pitch/conditioning -> diffusion.

        THE mel2ph CHICKEN-AND-EGG. `mel2ph` is needed to BUILD the trunk input
        (each frame looks up the phone it belongs to) and is also what the
        duration head PREDICTS. Resolved by teacher forcing:

          * training (`use_pred_mel2ph=False`): the trunk gets the ground-truth
            alignment and the duration head is an auxiliary read-out of the same
            pass. One trunk forward per step.
          * inference: the caller runs `predict_durations` first with the
            alignment it knows (span zeroed), builds the edited alignment from
            the result, and calls this with the completed `mel2ph`. That is what
            `eval_ming.edit_one` does, and it is what lets the utterance change
            length -- the only reason insertion and deletion are expressible.
          * `use_pred_mel2ph=True` does both inside one call: predict from the
            handed-in alignment, rebuild, re-run the trunk. The FRAME GRID cannot
            change here (`ref_mels`, `f0`, `uv` and `time_mel_masks` are already
            materialised at length T), so the rebuilt alignment is fitted to T. A
            caller that needs the length to change must do it the eval_ming way.

        Teacher forcing here is on the ALIGNMENT USED TO BUILD THE MEL TARGET, not
        on what the duration head gets to see. Those were once the same thing and
        the difference matters:

        The trunk is handed `mel2ph_in`, the alignment with the masked span
        BLANKED (see the block just above the trunk call). So at training the
        duration head reads a state that did NOT see in-span durations, which is
        exactly what `predict_durations` gives it at inference. That symmetry is
        deliberate and load-bearing -- do not "fix" it by passing the full
        alignment through.

        It used to be asymmetric, and that was the defect: the trunk received the
        true per-frame phone id inside the mask, so in-span durations were
        recoverable by counting frames rather than predictable from text, and the
        head was graded on a task it had never had to learn. Measured cost:
        substitution ACC 3.1% against deletion's 96.9%, deletion being the one
        edit type that needs no in-span durations at all.

        `scripts/test_no_duration_leak.py` asserts the invariant -- `dur` must be
        invariant to in-span alignment and sensitive to context alignment. Reverting
        the one-line trunk argument makes it fail at 8.29e2 against 7.07e1 for the
        context, i.e. the head was leaning on the leak an order of magnitude harder
        than on legitimate signal.
        """
        b, *_, device = *txt_tokens.shape, txt_tokens.device
        ret = {}

        # time_mel_masks arrives as [B, T, 1] from the tasks; the trunk wants
        # [B, T]. Canonicalise both once so no later expression silently
        # broadcasts a [B, T] mask against a [B, T, C] tensor.
        m = time_mel_masks[..., 0] if time_mel_masks.dim() == 3 else time_mel_masks
        time_mel_masks = m.unsqueeze(-1)

        # Frame validity is a property of `ref_mels`, NOT of whatever alignment
        # the duration head predicts, so it is read from the mel2ph that came in
        # and never recomputed from the rebuilt one.
        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        nonpadding = (mel2ph != 0).float().unsqueeze(1).unsqueeze(1)  # [B, 1, 1, T]

        ph_padding = txt_tokens == 0
        ph_nonpadding = (~ph_padding).float()
        qi, qp = self._text_segment(txt_tokens, text_ids, text_pad, tgt_nonpadding)
        mel_masked = ref_mels * (1 - m).unsqueeze(-1)

        if f0 is None:
            # Only reachable from callers that opted out of ground-truth pitch
            # without asking for the prediction. Unvoiced everywhere is the
            # honest reading of "no pitch supplied"; silently skipping the pitch
            # pathway would give a different model with no trace of it.
            f0 = m.new_zeros(m.shape)
        if uv is None:
            uv = torch.ones_like(f0)

        # THE ALIGNMENT THE TRUNK IS ALLOWED TO SEE. Masked frames are blanked
        # here and nowhere else.
        #
        # `trunk.forward` adds `ph_emb[gather(ph_tokens, mel2ph)]` to every frame
        # -- per-frame phone identity, unmasked. The mel is zeroed inside the
        # span, but the phone id was not, so each masked frame still announced
        # which phone it belonged to, and the COUNT of frames sharing a phone id
        # is that phone's duration. Attention is bidirectional, so `h_phone` --
        # which `heads.durations` reads -- could recover in-span durations by
        # counting instead of predicting them.
        #
        # `eval_ming.edit_one` zeroes exactly this (`masked_mel2ph[in_span] = 0`)
        # before `predict_durations`, so the head was trained where the answer
        # was free and tested where it was absent. Measured cost: substitution
        # ACC 3.1% against deletion's 96.9% -- deletion needs no in-span
        # durations at all (its span gets no frames and the head/tail are spliced
        # verbatim), which is why only substitution collapsed.
        #
        # Downstream keeps the TRUE alignment: `decoder_inp` expands `h_phone`
        # onto the frames it owns, and `add_dur_loss` reads `mel2ph` as its
        # target. Teacher forcing there is correct and is not the leak.
        mel2ph_in = mel2ph.masked_fill(m > 0.5, 0)

        # Same alignment into the pitch path, for the same reason. `pitch_padding
        # = (mel2ph == 0)` and `denorm_f0` does `2**f0` then `clamp(min=50)`, so
        # an unpadded masked frame arrives as 50 Hz VOICED while `predict_
        # durations` -- whose caller already zeroed the span -- sees it padded to
        # bin 0. Two different spans for the same frames, train versus infer.
        ctx = self._acoustic_context(mel_masked, m, tgt_nonpadding, spk_embed,
                                     f0, uv, mel2ph_in)

        h_audio, h_phone = self.trunk(mel_masked, m, qi, qp, txt_tokens,
                                      ph_nonpadding, mel2ph_in, tgt_nonpadding,
                                      ctx)
        dur, _ = self.heads.durations(h_phone, ph_padding)
        # Linear frames, matching what FastSpeech's Softplus duration predictor
        # emitted -- `add_dur_loss` takes `(dur_pred + 1).log()` itself, so
        # handing it the log would square the parameterisation.
        ret['dur'] = dur

        if use_pred_mel2ph:
            T = mel2ph.shape[1]
            pred = self.heads.mel2ph_from_dur(dur, ph_padding).detach()
            if pred.shape[1] < T:
                pred = F.pad(pred, [0, T - pred.shape[1]])
            mel2ph = pred[:, :T]
            h_audio, h_phone = self.trunk(mel_masked, m, qi, qp, txt_tokens,
                                          ph_nonpadding, mel2ph, tgt_nonpadding,
                                          ctx)
        ret['mel2ph'] = mel2ph = clip_mel2token_to_multiple(
            mel2ph, hparams['frames_multiple'])

        # Frame conditioning. Two terms from the trunk: what the AUDIO stream
        # made of each frame, plus the PHONE state expanded onto the frames it
        # owns. The second is the direct analogue of FastSpeech's
        # `expand_states(encoder_out, mel2ph)`, which was its primary
        # conditioning signal -- without it the phone segment could only reach
        # the mel through attention, and the duration head would receive no
        # gradient from the mel loss at all. `condition` is linear, so projecting
        # before expanding is the same function at 1/6th the activation memory.
        decoder_inp = (self.heads.condition(h_audio)
                       + expand_states(self.heads.condition(h_phone), mel2ph))
        decoder_inp = decoder_inp * tgt_nonpadding
        decoder_inp = decoder_inp + self.mel_encoder(mel_masked) * tgt_nonpadding

        if self.use_pitch_embed:
            pitch_padding = mel2ph == 0
            use_uv = hparams['pitch_type'] == 'frame' and hparams['use_uv']
            f0_pred, uv_logit = self.heads.pitch(h_audio)
            # [B, T, 2] with f0 at index 0 and the uv LOGIT at index 1, which is
            # what add_pitch_loss reads (speech_editing_base.py:252-262): it
            # feeds [..., 1] to binary_cross_entropy_with_logits, so this must
            # stay a logit and not a probability.
            ret['pitch_pred'] = torch.stack([f0_pred, uv_logit], dim=-1)
            if use_pred_pitch:
                # Pitch inside the span comes from the head, context pitch from
                # the input. This is what closed the ground-truth-F0 leak: the
                # protocol hands over a span whose f0 is ZERO, so any model that
                # reads f0 there at inference is reading a hole, and any model
                # that read it at training was reading the answer.
                pitch_padding = None
                pred_uv = uv_logit > 0
                res_f0 = f0 * (1 - m) + f0_pred * m
                res_uv = uv * (1 - m) + pred_uv * m
            else:
                res_f0, res_uv = f0, uv
            f0_denorm = denorm_f0(res_f0, res_uv if use_uv else None,
                                  pitch_padding=pitch_padding)
            ret['f0_denorm'] = f0_denorm
            ret['f0_denorm_pred'] = denorm_f0(
                f0_pred, (uv_logit > 0) if use_uv else None,
                pitch_padding=pitch_padding)
            decoder_inp = decoder_inp + self.pitch_embed(
                f0_to_coarse(f0_denorm)) * tgt_nonpadding

        cond = decoder_inp.transpose(1, 2)
        if not infer:
            t = torch.randint(0, self.num_timesteps + 1, (b,), device=device).long()
            # Diffusion
            x_t = self.diffuse_fn(ref_mels, t) * nonpadding

            # Predict x_{start}
            x_0_pred = self.denoise_fn(x_t, t, cond) * nonpadding

            ret['mel_out'] = x_0_pred[:, 0].transpose(1, 2) # [B, T, 80]
        else:
            t = self.num_timesteps  # reverse总步数
            shape = (cond.shape[0], 1, self.mel_bins, cond.shape[2])
            x = torch.randn(shape, device=device)  # noise
            # Training samples every index in [0, num_timesteps]. Decode that
            # same complete set instead of skipping the trained terminal index.
            for i in tqdm(reversed(range(0, t + 1)), desc='ProDiff Teacher sample time step', total=t + 1):
                x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), cond)  # x(mel), t, condition(phoneme)
            x = x[:, 0].transpose(1, 2)
            ret['mel_out'] = self.denorm_spec(x)  # 去除norm
        return ret

    def norm_spec(self, x):
        return x

    def denorm_spec(self, x):
        return x

