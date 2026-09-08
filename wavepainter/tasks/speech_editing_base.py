import torch
import torch.distributions
import torch.nn.functional as F
import torch.optim
import torch.utils.data

from wavepainter.tasks.dataset_utils import StutterSpeechDataset
from wavepainter.tasks.speech_base import SpeechBaseTask
from wavepainter.audio.align import mel2token_to_dur
from wavepainter.audio.pitch.utils import denorm_f0
from wavepainter.runtime.hparams import hparams
from wavepainter.nn.gst import GST


def _freeze_norm_layers(module):
    """Put only the normalisation layers into eval mode.

    Calling `.eval()` on the whole GST is wrong twice over. It is too broad --
    the only leak concern is BatchNorm using batch statistics, which for any
    two-view objective would normalise the views by DIFFERENT statistics and
    hand the encoder batch composition as a free signal. And it is fatal: the
    GST contains a GRU, and cuDNN refuses RNN backward in eval mode ("cudnn RNN
    backward can only be called in training mode"), so any loss that backprops
    through it dies.

    Normalisation layers to eval, everything else left alone.
    """
    for m in module.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                          torch.nn.BatchNorm3d, torch.nn.InstanceNorm1d,
                          torch.nn.InstanceNorm2d)):
            m.eval()


class SpeechEditingBaseTask(SpeechBaseTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = StutterSpeechDataset
        self.sil_ph = self.token_encoder.sil_phonemes()

        # Upstream stored the CLASS here (`self.gst = GST`) and then called
        # `self.gst(256, 7)` INSIDE perceptual_loss, i.e. it constructed a fresh
        # ~1.5M-param GST with RANDOM weights, on the CPU, on every single
        # training step. Three consequences, all bad:
        #   1. speed  - a single-threaded CPU forward per sample per step
        #               (tasks/run.py sets OMP_NUM_THREADS=1), plus two .cpu()
        #               syncs per sample. Plausibly the dominant cost of training.
        #   2. correctness - re-randomised weights mean the "prosody consistency"
        #               term measured a DIFFERENT random projection every step, so
        #               it had no consistent gradient direction to descend.
        #   3. BatchNorm ran in training mode on batches of 1, so every sample was
        #      normalised by its own statistics.
        # Build it once, seeded (so runs are reproducible and agents comparable),
        # frozen, in eval mode, on the GPU. Frozen is deliberate: a fixed encoder
        # cannot be gamed by collapsing it, which a jointly-trained one can.
        # gst_trainable defaults to TRUE, which is a deliberate departure from a
        # frozen encoder. Measured on this code: a randomly-initialised GST is
        # DEGENERATE as a feature extractor -- with .eval() the cosine similarity
        # between completely different mel inputs is 1.000000 (batch std 2e-5),
        # because eval-mode BatchNorm on never-trained running stats is an
        # identity and the 6-layer stride-2 conv stack collapses the signal. In
        # train mode it is only marginally better (cos 0.998 +/- 0.0005).
        # So a random GST measures nothing, and any loss built on one is inert.
        # It has to be trained, and the only objective here that can train it
        # without collapsing it is the contrastive one (L_CGPC): an MSE between
        # two views through a TRAINABLE shared encoder is minimised by making the
        # encoder constant. Hence the guard below.
        # Seed the GST reproducibly WITHOUT leaking a global seed. torch.manual_seed
        # seeds the CPU generator AND every CUDA device, while fork_rng(devices=[])
        # only restores the CPU one -- so the naive version silently left the global
        # CUDA RNG deterministically seeded for the rest of the process. That made
        # inference look reproducible across runs when it is in fact stochastic
        # (the reverse diffusion starts from torch.randn), which would have hidden
        # real run-to-run variance from every measurement built on top of it.
        _g = torch.Generator().manual_seed(hparams.get('gst_seed', 1234))
        _cpu_state = torch.get_rng_state()
        # current_device() only: get_rng_state(i) over ALL visible devices calls
        # _lazy_init on each one, creating a CUDA context per GPU per process --
        # which under mp.spawn DDP means N^2 contexts of a few hundred MB each.
        _cuda_dev = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        _cuda_states = [torch.cuda.get_rng_state(i) for i in _cuda_dev]
        try:
            torch.manual_seed(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=_g).item()))
            self.gst = GST(256, 7, n_mel_channels=hparams['audio_num_mel_bins'])
        finally:
            torch.set_rng_state(_cpu_state)
            for _i, _s in zip(_cuda_dev, _cuda_states):
                torch.cuda.set_rng_state(_s, _i)
        self.gst_trainable = hparams.get('gst_trainable', True)
        self.gst.requires_grad_(self.gst_trainable)
        if not self.gst_trainable:
            _freeze_norm_layers(self.gst)
        if self.gst_trainable and hparams.get('use_perceptual_loss', False):
            print('| WARNING: use_perceptual_loss (MSE between two views of a shared '
                  'encoder) with gst_trainable=True has a trivial minimum at a '
                  'constant encoder. Set gst_trainable=false.')
        self._gst_on_device = False


    def build_optimizer(self, model):
        """Include the GST in the optimizer whenever it is trainable.

        GST is a submodule of the TASK, not of `self.model`. `speech_base.py`
        builds `AdamW(model.parameters())`, which therefore MISSES it -- while
        `base_task.py::on_before_optimization` clips `self.parameters()`, which
        INCLUDES it. Left unfixed, GST gradients are never zeroed, their norm
        grows monotonically with step count, and `clip_grad_norm_` scales the
        WHOLE model's gradients by 1/total_norm -- a silent collapse of the
        effective learning rate that looks exactly like an honest plateau.

        (With `gst_trainable: false` the parameters have `requires_grad=False`,
        so `p.grad` stays None and `clip_grad_norm_` skips them -- that path was
        always safe. It is the default `true` that is dangerous.)
        """
        opt = super().build_optimizer(model)
        if self.gst_trainable:
            opt.add_param_group({'params': list(self.gst.parameters()),
                                 'lr': hparams['lr']})
        return opt

    def train(self, mode=True):
        """Keep a frozen GST in eval mode.

        `trainer.py:249` calls `task.train()` after EVERY validation, including
        the startup sanity pass, which would otherwise flip a frozen GST's
        BatchNorm permanently back into training mode -- so its running stats
        would drift and, worse, any two-view objective's separate forwards
        would be normalised by DIFFERENT batch statistics -- the classic
        BatchNorm leak, where batch composition becomes a free signal.
        """
        out = super().train(mode)
        gst = getattr(self, 'gst', None)
        if gst is not None and not getattr(self, 'gst_trainable', False):
            _freeze_norm_layers(gst)
        return out

    def _gst(self, x, lengths=None):
        """Run the prosody encoder, moving it to the data's device exactly once."""
        if not self._gst_on_device:
            self.gst.to(x.device)
            self._gst_on_device = True
        return self.gst(x, input_lengths=lengths)[0]

    @staticmethod
    def _mask_bounds(time_mel_masks, nonpadding=None):
        """First/last masked frame per item, vectorized.

        Replaces the per-item `torch.nonzero(...)[0]` loops, which had two real
        bugs: they raised IndexError on a row with no masked frame, and when the
        mask started at frame 0 the `modi_left - 1` boundary index wrapped to -1,
        silently comparing against the LAST frame of the utterance.

        Returns (L, R, valid) with L/R clamped to be safe to index; `valid`
        marks rows that actually have a masked frame.
        """
        m = time_mel_masks
        if m.dim() == 3:
            m = m[..., 0]
        B, T = m.shape
        idx = torch.arange(T, device=m.device)[None].expand(B, T)
        L = torch.where(m > 0, idx, torch.full_like(idx, T)).amin(1)
        R = torch.where(m > 0, idx, torch.full_like(idx, -1)).amax(1)
        valid = L < T
        # Each item's own final REAL frame. Callers that pass `nonpadding` get the
        # true per-item length; without it we fall back to the padded length, which
        # is only correct for the longest item in the batch.
        if nonpadding is None:
            last = torch.full_like(L, T - 1)
        else:
            np_ = nonpadding[..., 0] if nonpadding.dim() == 3 else nonpadding
            last = torch.where(np_ > 0, idx, torch.full_like(idx, -1)).amax(1).clamp(min=0)
        return L.clamp(0, T - 1), R.clamp(0, T - 1), valid, last


    def run_model(self, sample, infer=False, *args, **kwargs):
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        spk_embed = sample.get('spk_embed')
        spk_id = sample.get('spk_ids')
        if not infer:
            target = sample['mels']  # [B, T_s, 80]
            mel2ph = sample['mel2ph']  # [B, T_s]
            f0 = sample.get('f0')
            uv = sample.get('uv')
            output = self.model(txt_tokens, mel2ph=mel2ph, spk_embed=spk_embed, spk_id=spk_id,
                                f0=f0, uv=uv, infer=False)
            losses = {}
            self.add_mel_loss(output['mel_out'], target, losses)
            self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
            if hparams['use_pitch_embed']:
                self.add_pitch_loss(output, sample, losses)
            return losses, output
        else:
            use_gt_dur = kwargs.get('infer_use_gt_dur', hparams['use_gt_dur'])
            use_gt_f0 = kwargs.get('infer_use_gt_f0', hparams['use_gt_f0'])
            mel2ph, uv, f0 = None, None, None
            if use_gt_dur:
                mel2ph = sample['mel2ph']
            if use_gt_f0:
                f0 = sample['f0']
                uv = sample['uv']
            output = self.model(txt_tokens, mel2ph=mel2ph, spk_embed=spk_embed, spk_id=spk_id,
                                f0=f0, uv=uv, infer=True)
            return output

    def add_dur_loss(self, dur_pred, mel2ph, txt_tokens, losses=None):
        """

        :param dur_pred: [B, T], float, log scale
        :param mel2ph: [B, T]
        :param txt_tokens: [B, T]
        :param losses:
        :return:
        """
        B, T = txt_tokens.shape
        nonpadding = (txt_tokens != 0).float()
        dur_gt = mel2token_to_dur(mel2ph, T).float() * nonpadding
        is_sil = torch.zeros_like(txt_tokens).bool()
        for p in self.sil_ph:
            is_sil = is_sil | (txt_tokens == self.token_encoder.encode(p)[0])
        is_sil = is_sil.float()  # [B, T_txt]
        losses['pdur'] = F.mse_loss((dur_pred + 1).log(), (dur_gt + 1).log(), reduction='none')
        losses['pdur'] = (losses['pdur'] * nonpadding).sum() / nonpadding.sum()
        losses['pdur'] = losses['pdur'] * hparams['lambda_ph_dur']
        # use linear scale for sentence and word duration
        if hparams['lambda_word_dur'] > 0:
            word_id = (is_sil.cumsum(-1) * (1 - is_sil)).long()
            word_dur_p = dur_pred.new_zeros([B, word_id.max() + 1]).scatter_add(1, word_id, dur_pred)[:, 1:]
            word_dur_g = dur_gt.new_zeros([B, word_id.max() + 1]).scatter_add(1, word_id, dur_gt)[:, 1:]
            wdur_loss = F.mse_loss((word_dur_p + 1).log(), (word_dur_g + 1).log(), reduction='none')
            word_nonpadding = (word_dur_g > 0).float()
            wdur_loss = (wdur_loss * word_nonpadding).sum() / word_nonpadding.sum()
            losses['wdur'] = wdur_loss * hparams['lambda_word_dur']
        if hparams['lambda_sent_dur'] > 0:
            sent_dur_p = dur_pred.sum(-1)
            sent_dur_g = dur_gt.sum(-1)
            sdur_loss = F.mse_loss((sent_dur_p + 1).log(), (sent_dur_g + 1).log(), reduction='mean')
            losses['sdur'] = sdur_loss.mean() * hparams['lambda_sent_dur']

    def add_pitch_loss(self, output, sample, losses):
        mel2ph = sample['mel2ph']  # [B, T_s]
        f0 = sample['f0']
        uv = sample['uv']
        nonpadding = (mel2ph != 0).float() if hparams['pitch_type'] == 'frame' \
            else (sample['txt_tokens'] != 0).float()
        p_pred = output['pitch_pred']
        assert p_pred[..., 0].shape == f0.shape
        if hparams['use_uv'] and hparams['pitch_type'] == 'frame':
            assert p_pred[..., 1].shape == uv.shape, (p_pred.shape, uv.shape)
            losses['uv'] = (F.binary_cross_entropy_with_logits(
                p_pred[:, :, 1], uv, reduction='none') * nonpadding).sum() \
                           / nonpadding.sum() * hparams['lambda_uv']
            nonpadding = nonpadding * (uv == 0).float()
        f0_pred = p_pred[:, :, 0]
        losses['f0'] = (F.l1_loss(f0_pred, f0, reduction='none') * nonpadding).sum() \
                       / nonpadding.sum() * hparams['lambda_f0']

    def add_conn_loss(self, output, time_mel_masks, target, sample, losses, postfix=''):
        nonpadding = (sample['mel2ph'] > 0).float() if 'mel2ph' in sample else None
        losses[f'conn'] = getattr(self, f'conn_loss')(
            output, time_mel_masks, target, nonpadding=nonpadding)

    def add_perceptual_loss(self, output, time_mel_masks, target, losses, postfix=''):
        losses[f'perceptual'] = getattr(self, f'perceptual_loss')(output, time_mel_masks, target)

    def _boundary_terms(self, output, target, time_mel_masks, fn, nonpadding=None):
        """Apply `fn` at the left and right mask boundaries and average over items.

        `fn(x, i, j)` receives the batch tensor and two frame-index tensors and
        must return a per-item scalar. The right-hand term is dropped for items
        whose mask runs to the final frame (there is no frame after it), matching
        upstream's `if modi_right != output.size(1) - 1`.

        Note the dropped `torch.tensor(0.005)` / `torch.tensor(1e-6)` seeds that
        upstream added before accumulating: they only added a constant offset of
        0.005/B to the logged value and never affected the gradient.
        """
        L, R, valid, last = self._mask_bounds(time_mel_masks, nonpadding)
        T = output.shape[1]
        Lm = (L - 1).clamp(min=0)      # frame before the mask; == L if mask starts at 0
        Rp = (R + 1).clamp(max=T - 1)  # frame after the mask
        # The right-hand pair is passed as (R, Rp) rather than (Rp, R); for the
        # variance form that negates both sides of the comparison, and MSE is
        # invariant to a shared sign flip.
        left = F.mse_loss(fn(output, Lm, L), fn(target, Lm, L), reduction='none')
        if left.dim() > 1:
            left = left.mean(dim=tuple(range(1, left.dim())))
        right = F.mse_loss(fn(output, R, Rp), fn(target, R, Rp), reduction='none')
        if right.dim() > 1:
            right = right.mean(dim=tuple(range(1, right.dim())))
        # `last` is this item's OWN final real frame, not the batch-padded length.
        # Using T-1 made has_right true for every item that was not the longest in
        # its batch, so Rp indexed PADDING: a zero mel row at frame level, and
        # _pool_units' padding bucket at phone/word level. The term silently became
        # "distance to silence" and depended on batch composition.
        has_right = ((R < last) & valid).to(left.dtype)
        # Symmetric guard. A mask touching frame 0 has Lm == L, so the left term is
        # identically zero for BOTH prediction and target -- it contributed nothing
        # while still counting in the denominator (~10% of items at mask ratio 0.80).
        has_left = ((L > 0) & valid).to(left.dtype)
        total = left * has_left + right * has_right
        # Mean per CONTRIBUTING term rather than per item, so items that supply one
        # boundary instead of two no longer dilute the batch mean.
        return total.sum() / (has_left.sum() + has_right.sum()).clamp(min=1)

    def conn_loss(self, output, time_mel_masks, target, postfix='', nonpadding=None):
        """Acoustic-consistency: match the change in per-frame mel VARIANCE across
        the mask boundary. FluentEditor's original term."""
        b = torch.arange(output.shape[0], device=output.device)

        def delta_var(x, i, j):
            return x[b, i, :].var(dim=-1) - x[b, j, :].var(dim=-1)

        return self._boundary_terms(output, target, time_mel_masks, delta_var, nonpadding)


    def perceptual_loss(self, output, time_mel_masks, target):
        """Prosody consistency: MSE between the GST embedding of the predicted
        masked region and that of the whole ground-truth utterance.

        One batched GST call per side instead of 2*B single-sample CPU calls.
        """
        B, T, C = output.shape
        L, R, valid, _ = self._mask_bounds(time_mel_masks)
        lens = (R - L + 1).clamp(min=1)
        lmax = int(lens.max().item())
        ar = torch.arange(lmax, device=output.device)[None]           # [1, lmax]
        gidx = (L[:, None] + ar).clamp(max=T - 1)                    # [B, lmax]
        keep = (ar < lens[:, None]).to(output.dtype)                  # [B, lmax]
        seg = output.gather(1, gidx[..., None].expand(B, lmax, C)) * keep[..., None]
        zp = self._gst(seg, lens)
        zg = self._gst(target)
        loss = F.mse_loss(zp, zg, reduction='none').mean(-1) * valid.to(zp.dtype)
        return loss.sum() / valid.sum().clamp(min=1)

    def save_valid_result(self, sample, batch_idx, model_out):
        sr = hparams['audio_sample_rate']
        f0_gt = None
        mel_out = model_out['mel_out']
        if sample.get('f0') is not None:
            f0_gt = denorm_f0(sample['f0'][0].cpu(), sample['uv'][0].cpu())
        self.plot_mel(batch_idx, sample['mels'], mel_out, f0s=f0_gt)
        if self.global_step > 0:
            wav_pred = self.vocoder.spec2wav(mel_out[0].cpu(), f0=f0_gt)
            self.logger.add_audio(f'wav_val_{batch_idx}', wav_pred, self.global_step, sr)
            # with gt duration
            model_out = self.run_model(sample, infer=True, infer_use_gt_dur=True)
            dur_info = self.get_plot_dur_info(sample, model_out)
            del dur_info['dur_pred']
            wav_pred = self.vocoder.spec2wav(model_out['mel_out'][0].cpu(), f0=f0_gt)
            self.logger.add_audio(f'wav_gdur_{batch_idx}', wav_pred, self.global_step, sr)
            self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'][0], f'mel_gdur_{batch_idx}',
                          dur_info=dur_info, f0s=f0_gt)

            # with pred duration
            if not hparams['use_gt_dur']:
                model_out = self.run_model(sample, infer=True, infer_use_gt_dur=False)
                dur_info = self.get_plot_dur_info(sample, model_out)
                self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'][0], f'mel_pdur_{batch_idx}',
                              dur_info=dur_info, f0s=f0_gt)
                wav_pred = self.vocoder.spec2wav(model_out['mel_out'][0].cpu(), f0=f0_gt)
                self.logger.add_audio(f'wav_pdur_{batch_idx}', wav_pred, self.global_step, sr)
        # gt wav
        if self.global_step <= hparams['valid_infer_interval']:
            mel_gt = sample['mels'][0].cpu()
            wav_gt = self.vocoder.spec2wav(mel_gt, f0=f0_gt)
            self.logger.add_audio(f'wav_gt_{batch_idx}', wav_gt, self.global_step, sr)

    def get_plot_dur_info(self, sample, model_out):
        T_txt = sample['txt_tokens'].shape[1]
        dur_gt = mel2token_to_dur(sample['mel2ph'], T_txt)[0]
        dur_pred = model_out['dur'] if 'dur' in model_out else dur_gt
        txt = self.token_encoder.decode(sample['txt_tokens'][0].cpu().numpy())
        txt = txt.split(" ")
        return {'dur_gt': dur_gt, 'dur_pred': dur_pred, 'txt': txt}

    def test_step(self, sample, batch_idx):
        assert sample['txt_tokens'].shape[0] == 1, 'only support batch_size=1 in inference'
        outputs = self.run_model(sample, infer=True)
        text = sample['text'][0]
        item_name = sample['item_name'][0]
        tokens = sample['txt_tokens'][0].cpu().numpy()
        mel_gt = sample['mels'][0].cpu().numpy()
        mel_pred = outputs['mel_out'][0].cpu().numpy()
        mel2ph = sample['mel2ph'][0].cpu().numpy()
        time_mel_masks = sample['time_mel_masks'][0].cpu().numpy()
        mel2ph_pred = None
        str_phs = self.token_encoder.decode(tokens, strip_padding=True)
        base_fn = f'[{batch_idx:06d}][{item_name.replace("%", "_")}][%s]'
        if text is not None:
            base_fn += text.replace(":", "$3A")[:80]
        base_fn = base_fn.replace(' ', '_')
        gen_dir = self.gen_dir
        wav_pred = self.vocoder.spec2wav(mel_pred)
        self.saving_result_pool.add_job(self.save_result, args=[
            wav_pred, mel_pred, base_fn % 'P', gen_dir, str_phs, mel2ph_pred])
        mel_pred_seg = mel_pred[time_mel_masks == 1]
        wav_pred_seg = self.vocoder.spec2wav(mel_pred_seg)
        self.saving_result_pool.add_job(self.save_result, args=[
            wav_pred_seg, mel_pred_seg, base_fn % 'P_SEG', gen_dir, None, None])
        if hparams['save_gt']:
            wav_gt = self.vocoder.spec2wav(mel_gt)
            self.saving_result_pool.add_job(self.save_result, args=[
                wav_gt, mel_gt, base_fn % 'G', gen_dir, str_phs, mel2ph])
            mel_gt_seg = mel_gt[time_mel_masks == 1]
            wav_gt_seg = self.vocoder.spec2wav(mel_gt_seg)
            self.saving_result_pool.add_job(self.save_result, args=[
                wav_gt_seg, mel_gt_seg, base_fn % 'G_SEG', gen_dir, None, None])

        return {
            'item_name': item_name,
            'text': text,
            'ph_tokens': self.token_encoder.decode(tokens.tolist()),
            'wav_fn_pred': base_fn % 'P',
            'wav_fn_gt': base_fn % 'G',
            'wav_fn_orig': sample['wav_fn'][0],
        }
