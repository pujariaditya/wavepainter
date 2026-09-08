import os

import numpy as np
import torch
import torch.optim
import torch.nn.functional as F

from wavepainter.nn.model_utils import num_params
from wavepainter.runtime.tensor_utils import tensors_to_scalars
from wavepainter.runtime.hparams import hparams
from wavepainter.models.spec_denoiser.spec_denoiser import GaussianDiffusion
from wavepainter.models.spec_denoiser.diffnet import DiffNet
from wavepainter.tasks.speech_editing_base import SpeechEditingBaseTask
from wavepainter.tasks.vocoder_infer.base_vocoder import get_vocoder_cls, BaseVocoder
from wavepainter.tasks.dataset_utils import StutterSpeechDataset


DIFF_DECODERS = {
    'wavenet': lambda hp: DiffNet(hp['audio_num_mel_bins']),
}


class SpeechDenoiserTask(SpeechEditingBaseTask):
    def __init__(self):
        super(SpeechDenoiserTask, self).__init__()
        self.dataset_cls = StutterSpeechDataset
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams['vocoder'])()
        self._build_phone_mel_teacher()

    def _build_phone_mel_teacher(self):
        """Load fixed corpus phone prototypes only when their loss is active."""
        weight = float(hparams.get('lambda_phone_mel_contrast', 0.0))
        if weight <= 0:
            self._phone_mel_teacher_active = False
            return
        artifact = hparams.get(
            'phone_mel_prototypes',
            'assets/phone_mel_prototypes.npz')
        if not os.path.isabs(artifact):
            root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            artifact = os.path.join(root, artifact)
        if not os.path.isfile(artifact):
            raise FileNotFoundError(
                f"phone-mel prototype artifact not found: {artifact}")
        blob = np.load(artifact)
        self.register_buffer(
            '_phone_mel_prototypes',
            torch.from_numpy(blob['prototypes']).float(),
            persistent=False)
        self.register_buffer(
            '_phone_mel_feature_mean',
            torch.from_numpy(blob['feature_mean']).float(),
            persistent=False)
        self.register_buffer(
            '_phone_mel_feature_std',
            torch.from_numpy(blob['feature_std']).float().clamp(min=1e-4),
            persistent=False)
        self.register_buffer(
            '_phone_mel_valid_classes',
            torch.from_numpy(blob['valid_classes']).bool(),
            persistent=False)
        self._phone_mel_teacher_active = True
        print(
            "| fixed phone-mel teacher "
            f"| artifact={artifact} "
            f"| classes={int(self._phone_mel_valid_classes.sum())}/"
            f"{len(self._phone_mel_valid_classes)} "
            f"| temperature={float(hparams.get('phone_mel_temperature', 0.1))}"
        )

    def phone_mel_contrast_loss(self, mel_pred, mel2ph, txt_tokens,
                                time_mel_masks):
        """Classify generated masked frames with frozen corpus phone prototypes.

        The alignment is used only to label the generated training frames.  The
        fixed prototypes and feature statistics are corpus aggregates, not a
        trainable recognition model, so the only gradients are into ``mel_pred``.
        Inference topology and the duration grid are unchanged.
        """
        if not self._phone_mel_teacher_active:
            return mel_pred.sum() * 0.0
        mask = (time_mel_masks[..., 0] if time_mel_masks.dim() == 3
                else time_mel_masks) > 0.5
        batch, n_phone = txt_tokens.shape
        phone_pos = mel2ph.clamp(min=1, max=n_phone) - 1
        labels = txt_tokens.gather(1, phone_pos)
        in_vocab = labels < len(self._phone_mel_valid_classes)
        safe_labels = labels.clamp(
            min=0, max=len(self._phone_mel_valid_classes) - 1)
        valid = (mask & (mel2ph > 0) & in_vocab
                 & self._phone_mel_valid_classes[safe_labels])
        if not valid.any():
            return mel_pred.sum() * 0.0

        mean = self._phone_mel_feature_mean
        std = self._phone_mel_feature_std
        frames = F.normalize((mel_pred[valid] - mean) / std, dim=-1)
        prototypes = F.normalize(
            (self._phone_mel_prototypes - mean[None]) / std[None], dim=-1)
        temperature = float(hparams.get('phone_mel_temperature', 0.1))
        logits = frames @ prototypes.transpose(0, 1)
        logits = logits / temperature
        logits = logits.masked_fill(
            ~self._phone_mel_valid_classes[None], -1e4)
        return F.cross_entropy(logits, safe_labels[valid])

    def build_model(self):
        self.build_tts_model()
        self._load_parent_checkpoint()
        num_params(self.model)
        return self.model

    def _load_parent_checkpoint(self):
        """Initialise from a trained parent when `load_ckpt` names one.

        THIS BRANCH USED TO BE MISSING. `wavepainter/tasks/speech_base.py` honours
        `load_ckpt`, but this class overrides `build_model` and dropped it, so
        setting `load_ckpt` in the config did nothing at all -- silently. The
        run started from the donor, the loss fell, and the config said otherwise.

        Model weights only: no optimizer, no `global_step`. That is the
        difference from the trainer's implicit resume, which picks up whatever
        is already in `work_dir` and carries the step counter and LR schedule
        position with it. Use this to START a new arm from a parent; use the
        implicit resume to CONTINUE one.

        `strict=False` is deliberate: the task asks agents to compose extra
        pretrained components onto the trunk, and a strict load would make the
        parent unloadable for exactly the agents doing what was asked. Tensors
        the parent does not have are left at their init.

        But a loader that tolerates everything cannot tell a composed model from
        a typo'd path, so the tolerance is one-directional and floored. The
        invariant is the fraction of the PARENT's tensors that were consumed --
        adding modules to the model cannot lower it, while pointing at the wrong
        checkpoint does. Below the floor this raises rather than warns.
        """
        path = hparams.get('load_ckpt', '')
        if not path:
            return

        from wavepainter.runtime.ckpt_utils import get_last_checkpoint
        if os.path.isfile(path):
            blob = torch.load(path, map_location='cpu')
            resolved = path
        else:
            blob, resolved = get_last_checkpoint(path)
        if blob is None:
            raise FileNotFoundError(f"load_ckpt found no checkpoint at {path!r}")

        parent = blob['state_dict']
        if 'model' in parent and isinstance(parent['model'], dict):
            parent = parent['model']
        mine = self.model.state_dict()

        usable = {k: v for k, v in parent.items()
                  if k in mine and mine[k].shape == v.shape}
        shape_bad = [k for k, v in parent.items()
                     if k in mine and mine[k].shape != v.shape]
        absent = [k for k in parent if k not in mine]
        fresh = [k for k in mine if k not in parent]

        consumed = len(usable) / max(len(parent), 1)
        floor = float(hparams.get('load_ckpt_min_match', 0.9))
        print(f"| load_ckpt {resolved}")
        print(f"|   consumed {len(usable)}/{len(parent)} parent tensors "
              f"({consumed:.1%}, floor {floor:.0%})")
        if fresh:
            print(f"|   {len(fresh)} tensors left at init (new in this model): "
                  f"{', '.join(sorted(fresh)[:4])}"
                  f"{'...' if len(fresh) > 4 else ''}")
        if shape_bad or absent:
            print(f"|   {len(shape_bad)} shape-mismatched, {len(absent)} not in "
                  f"this model -- skipped")

        if consumed < floor:
            raise RuntimeError(
                f"load_ckpt consumed only {consumed:.1%} of {resolved!r} "
                f"({len(usable)}/{len(parent)} tensors), below the "
                f"{floor:.0%} floor. That is a different architecture, not a "
                f"parent. Set load_ckpt_min_match lower only if you know why.")

        self.model.load_state_dict(usable, strict=False)

    def build_tts_model(self):
        self.model = GaussianDiffusion(
            phone_encoder=self.token_encoder,
            out_dims=hparams['audio_num_mel_bins'], denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
            timesteps=hparams['timesteps'], time_scale=hparams['timescale'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
        )

    def build_optimizer(self, model):
        if not hparams.get('use_dualffn_trunk', False):
            return super().build_optimizer(model)

        fresh, donor = [], []
        # Every audio-side tensor that was INITIALISED FROM THE DONOR, so it gets
        # the reduced LR. ln1p_a/ln2p_a are Gemma's post-attention and post-FFN
        # sandwich norms; '.ln1_a.' does not match '.ln1p_a.', so omitting them
        # here silently trains donor weights at the full fresh-parameter rate.
        donor_markers = (
            '.ffn_audio.',
            '.ln1_a.',
            '.ln2_a.',
            '.ln1p_a.',
            '.ln2p_a.',
        )
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if (name.startswith('trunk.layers.')
                    and '.lora_' not in name
                    and any(marker in name for marker in donor_markers)):
                donor.append(parameter)
            elif name.startswith('trunk.norm_a.'):
                donor.append(parameter)
            else:
                fresh.append(parameter)

        donor_scale = float(hparams.get('trunk_pretrained_lr_scale', 0.05))
        groups = [
            {'params': fresh, 'lr': hparams['lr'], 'lr_scale': 1.0},
            {'params': donor, 'lr': hparams['lr'] * donor_scale,
             'lr_scale': donor_scale},
        ]
        print(
            "| DualFFN optimizer "
            f"| fresh={sum(p.numel() for p in fresh):,} "
            f"| donor_audio={sum(p.numel() for p in donor):,} "
            f"| scales=1/{donor_scale}"
        )
        self.optimizer = torch.optim.AdamW(
            groups,
            lr=hparams['lr'],
            betas=(
                hparams['optimizer_adam_beta1'],
                hparams['optimizer_adam_beta2'],
            ),
            weight_decay=hparams['weight_decay'],
        )
        return self.optimizer


    def run_model(self, sample, infer=False, *args, **kwargs):
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        target = sample['mels']  # [B, T_s, 80]
        mel2ph = sample['mel2ph']
        f0 = sample['f0']
        uv = sample['uv']
        energy = None
        time_mel_masks = sample['time_mel_masks'][:,:,None]
        nonpadding = (mel2ph > 0).float()
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        output = self.model(txt_tokens, time_mel_masks, mel2ph=mel2ph, spk_embed=spk_embed,
                       ref_mels=target, f0=f0, uv=uv, energy=energy, infer=infer,
                       text_ids=sample.get('text_ids'), text_pad=sample.get('text_pad'))

        losses = {}
        self.add_mel_loss(output['mel_out']*time_mel_masks, target*time_mel_masks, losses, postfix="_coarse")
        output['mel_out'] = output['mel_out']*time_mel_masks + target*(1-time_mel_masks)
        self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
        # Phase D. Computed on the context-reinserted mel_out (the line above
        # this block reinserted it), but the frame mask keeps only generated
        # frames, so ground-truth context contributes nothing. No `self.training`
        # guard: the teacher-active flag and the weight come from the same
        # hparam, so weight zero already disables it -- and at zero the .npz is
        # never opened, which is what makes the ablation exact.
        phone_mel_weight = float(
            hparams.get('lambda_phone_mel_contrast', 0.0))
        if phone_mel_weight > 0:
            losses['phmel'] = phone_mel_weight * \
                self.phone_mel_contrast_loss(
                    output['mel_out'], mel2ph, txt_tokens, time_mel_masks)
        if hparams['use_pitch_embed']:
            self.add_pitch_loss(output, sample, losses)
        if hparams['use_conn_loss']:
            self.add_conn_loss(output['mel_out'], time_mel_masks, target, sample, losses, postfix="_coarse")
        if hparams['use_perceptual_loss']:
            self.add_perceptual_loss(output['mel_out'], time_mel_masks, target,  losses, postfix="_coarse")
        if not infer:
            return losses, output
        else:
            return output

    def validation_step(self, sample, batch_idx):
        outputs = {}
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        target = sample['mels']

        energy = None
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        mel2ph = sample['mel2ph']
        f0 = sample['f0']
        uv = sample['uv']
        time_mel_masks = sample['time_mel_masks'][:,:,None]

        outputs['losses'] = {}
        outputs['losses'], output = self.run_model(sample, infer=False)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs = tensors_to_scalars(outputs)
        if batch_idx < hparams['num_valid_plots']:
            # text_ids MUST be passed here too. Without them the trunk takes its
            # nt=0 branch and the validation AUDIO bypasses the entire text half of
            # the DualFFN (tok_emb, ffn_text, ln1_t, ln2_t) -- while the validation
            # LOSSES above go through run_model, which does pass them. The plots and
            # the logged numbers would describe two different models.
            #
            model_out = self.model(
                txt_tokens, time_mel_masks, spk_embed=spk_embed, mel2ph=mel2ph, f0=f0, uv=uv, energy=energy, ref_mels=target, infer=True,
                text_ids=sample.get('text_ids'), text_pad=sample.get('text_pad'))
            model_out['mel_out'] = model_out['mel_out']*time_mel_masks + target*(1-time_mel_masks)
            self.plot_wav(batch_idx, sample['mels'], model_out['mel_out'], is_mel=True, gt_f0=None, f0=None)
            self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'])
        return outputs

    ############
    # validation plots
    ############
    def plot_wav(self, batch_idx, gt_wav, wav_out, is_mel=False, gt_f0=None, f0=None, name=None):
        gt_wav = gt_wav[0].cpu().numpy()
        wav_out = wav_out[0].cpu().numpy()
        gt_f0 = None
        f0 = None
        if is_mel:
            gt_wav = self.vocoder.spec2wav(gt_wav, f0=gt_f0)
            wav_out = self.vocoder.spec2wav(wav_out, f0=f0)
        self.logger.add_audio(f'gt_{batch_idx}', gt_wav, sample_rate=hparams['audio_sample_rate'], global_step=self.global_step)
        self.logger.add_audio(f'wav_{batch_idx}', wav_out, sample_rate=hparams['audio_sample_rate'], global_step=self.global_step)
