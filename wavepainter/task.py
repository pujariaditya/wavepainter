"""Phase-2 training task: two overrides on the base editor.

The editor lives in this repository (``wavepainter/tasks/``,
``wavepainter/models/``, ``wavepainter/runtime/``, ``wavepainter/datasets/``);
This class extends it
rather than modifying it, so phase 2 is a clean increment over phase 1 and
reduces to the phase-1 task exactly when its two hyperparameters are off.

The two overrides are:

  ``build_model``  freeze everything outside ``trainable_only`` -- for the
                   released model that is ``['denoise_fn.']``, 170 of 531
                   tensors, leaving trunk and heads byte-exact to the phase-1
                   base so the result can be interpolated back toward it.

  ``run_model``    add the frozen-HuBERT feature loss to the loss dict.

Both are pure extensions: ``super()`` runs first and its behaviour is untouched
when ``lambda_hubert_feature`` is 0 and ``trainable_only`` is empty, which is the
exact-ablation property the tests assert.
"""

from wavepainter.tasks.spec_denoiser import SpeechDenoiserTask
from wavepainter.runtime.hparams import hparams

from wavepainter.losses.hubert_feature import FrozenHuBERTFeatureLoss


class WavePainterTask(SpeechDenoiserTask):
    """Denoiser-only training under frozen-HuBERT perceptual supervision."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hubert_loss = None
        if float(hparams.get("lambda_hubert_feature", 0.0)) > 0:
            # A plain object, never an nn.Module: its frozen BigVGAN and HuBERT
            # teachers must not become registered task modules, or they would
            # land in the released checkpoint. They load lazily onto whatever
            # device the first batch arrives on.
            self._hubert_loss = FrozenHuBERTFeatureLoss(
                crop_frames=hparams.get("hubert_feature_crop_frames", 64),
                context_frames=hparams.get("hubert_feature_context_frames", 12),
                items_per_batch=hparams.get("hubert_feature_items_per_batch", 1),
                feature_layer=hparams.get("hubert_feature_layer", "conv"),
            )

    # ------------------------------------------------------------------ model

    def build_model(self):
        """Build as upstream does, then freeze outside ``trainable_only``.

        Order matters: ``super().build_model()`` loads the phase-1 parent, and
        freezing has to happen after those weights are in place.
        """
        model = super().build_model()
        self._apply_trainable_only()
        return model

    def _apply_trainable_only(self):
        """Freeze every parameter whose name lacks a ``trainable_only`` prefix.

        Empty (the default) trains everything, which is exactly upstream
        behaviour -- that is what makes this an exact ablation at defaults.
        """
        prefixes = hparams.get("trainable_only", []) or []
        if not prefixes:
            return
        kept = frozen = 0
        kept_p = frozen_p = 0
        for name, param in self.model.named_parameters():
            keep = any(name.startswith(prefix) for prefix in prefixes)
            param.requires_grad_(keep)
            if keep:
                kept += 1
                kept_p += param.numel()
            else:
                frozen += 1
                frozen_p += param.numel()
        print(
            f"| trainable_only={prefixes} | kept {kept} tensors "
            f"({kept_p:,} params) | frozen {frozen} tensors ({frozen_p:,} params)"
        )

    # ------------------------------------------------------------------- loss

    def run_model(self, sample, infer=False, *args, **kwargs):
        """Upstream forward and losses, plus the HuBERT feature term.

        ``super()`` has already reassigned ``output['mel_out']`` to the
        context-reinserted mel -- generated frames inside the edit mask, real
        audio outside it -- so the loss sees exactly what inference emits.
        """
        losses, output = super().run_model(sample, infer=infer, *args, **kwargs)
        weight = float(hparams.get("lambda_hubert_feature", 0.0))
        if not infer and weight > 0 and self._hubert_loss is not None:
            losses["hubert_fe"] = weight * self._hubert_loss(
                output["mel_out"],
                sample["mels"],
                sample["time_mel_masks"][:, :, None],
            )
        return losses, output
