"""Frozen-HuBERT perceptual feature loss for regenerated mel spans.

This is the phase-2 mechanism. The editor emits BigVGAN log-mels at 24 kHz while
HuBERT consumes 16 kHz waveform, so the loss uses the (already frozen) BigVGAN
decoder as a differentiable bridge and compares HuBERT representations of the
predicted and reference spans.

Two properties are load-bearing:

  * The wrapper is deliberately **not** an ``nn.Module``. When the training task
    holds it as a plain attribute, neither frozen teacher is registered in the
    checkpoint, so the released weights contain only the editor's own 531
    tensors and no teacher leaks into the artifact.

  * Gradients flow through the predicted branch only. The reference branch runs
    under ``no_grad``, so the teachers are a fixed measuring instrument rather
    than a second objective being co-optimised.

The comparison is a **matched crop**: a fixed window centred on each item's real
edit mask, with context frames either side, rather than the whole utterance. The
signal being supervised is local to the edit, and averaging it over an utterance
that is mostly unedited dilutes it.
"""

import os

import torch
import torch.nn.functional as F

HUBERT_REPO = "facebook/hubert-base-ls960"


def _hubert_dir():
    """Resolve the HuBERT snapshot, preferring an explicit override.

    Set ``WAVEPAINTER_HUBERT_DIR`` to pin a local snapshot; otherwise the
    weights are fetched through ``huggingface_hub`` and cached normally.
    """
    override = os.environ.get("WAVEPAINTER_HUBERT_DIR")
    if override:
        return override
    from huggingface_hub import snapshot_download

    return snapshot_download(HUBERT_REPO)


class FrozenHuBERTFeatureLoss:
    """Matched-crop HuBERT feature loss with lazily loaded, unregistered teachers.

    ``feature_layer="conv"`` compares HuBERT's convolutional feature encoder
    output and is what the released model was trained with. A positive integer
    instead keeps that many transformer layers and compares their contextual
    hidden states -- a documented alternative, but **not** the configuration the
    reported numbers come from.

    Either way this stays a continuous matched-audio objective: no vocabulary
    and no CTC head is ever loaded.
    """

    def __init__(
        self,
        crop_frames=64,
        context_frames=12,
        items_per_batch=1,
        feature_layer="conv",
        hubert_dir=None,
        vocoder=None,
        feature_encoder=None,
    ):
        self.crop_frames = int(crop_frames)
        self.context_frames = int(context_frames)
        self.items_per_batch = int(items_per_batch)
        self.feature_layer = feature_layer
        if feature_layer != "conv":
            self.feature_layer = int(feature_layer)
            if not 1 <= self.feature_layer <= 12:
                raise ValueError("HuBERT contextual layer must be in [1, 12]")
        self.hubert_dir = hubert_dir
        if self.crop_frames < 8:
            raise ValueError("crop must contain at least 8 mel frames")
        if self.items_per_batch < 1:
            raise ValueError("need at least one batch item")
        self._vocoder = vocoder
        self._feature_encoder = feature_encoder
        self._device = None
        if vocoder is not None or feature_encoder is not None:
            if vocoder is None or feature_encoder is None:
                raise ValueError("inject both teachers, or neither")
            self._freeze_teachers()

    # ---------------------------------------------------------------- teachers

    def _freeze_teachers(self):
        for module in (self._vocoder, self._feature_encoder):
            module.requires_grad_(False)
            module.eval()

    @staticmethod
    def _load_hubert_feature_encoder(hubert_dir, feature_layer="conv"):
        """Load a convolutional or truncated contextual HuBERT encoder.

        Loaded ``strict=True`` in both branches: a silently partial teacher
        would still produce a plausible-looking loss curve.
        """
        from transformers import HubertConfig, HubertModel
        from transformers.models.hubert.modeling_hubert import HubertFeatureEncoder

        config = HubertConfig.from_pretrained(hubert_dir, local_files_only=True)
        state = torch.load(
            os.path.join(hubert_dir, "pytorch_model.bin"),
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if feature_layer == "conv":
            encoder = HubertFeatureEncoder(config)
            prefix = "feature_extractor."
            feature_state = {
                key[len(prefix):]: value
                for key, value in state.items()
                if key.startswith(prefix)
            }
            incompatible = encoder.load_state_dict(feature_state, strict=True)
        else:
            # Load the complete encoder before truncating, so strict validation
            # still covers the whole checkpoint, then drop the layers the
            # selected loss never consumes.
            encoder = HubertModel(config)
            incompatible = encoder.load_state_dict(state, strict=True)
            encoder.encoder.layers = torch.nn.ModuleList(
                list(encoder.encoder.layers[: int(feature_layer)])
            )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"HuBERT feature load mismatch: {incompatible}")
        return encoder

    @staticmethod
    def _load_bigvgan():
        from wavepainter.tasks.vocoder_infer.bigvgan import _bigvgan_imports, bigvgan_dir

        with _bigvgan_imports():
            from bigvgan import BigVGAN

        model = BigVGAN.from_pretrained(bigvgan_dir(), use_cuda_kernel=False)
        model.remove_weight_norm()
        return model

    def _ensure_loaded(self, device):
        if self._vocoder is None:
            self._vocoder = self._load_bigvgan()
            self._feature_encoder = self._load_hubert_feature_encoder(
                self.hubert_dir or _hubert_dir(), self.feature_layer
            )
            self._freeze_teachers()
        if self._device != device:
            self._vocoder.to(device)
            self._feature_encoder.to(device)
            self._device = device
        self._freeze_teachers()

    # ------------------------------------------------------------------ crops

    def _crop(self, tensor, mask, indices):
        """Extract fixed-width crops centred on each item's actual edit mask."""
        _, total_frames, _ = tensor.shape
        crops, mask_crops = [], []
        for idx in indices.tolist():
            active = torch.nonzero(mask[idx] > 0.5, as_tuple=False).flatten()
            left = int(active[0].item())
            right = int(active[-1].item()) + 1
            wanted_left = left - self.context_frames
            wanted_right = right + self.context_frames
            if wanted_right - wanted_left <= self.crop_frames:
                slack = self.crop_frames - (wanted_right - wanted_left)
                wanted_left -= slack // 2
            start = max(0, min(wanted_left, total_frames - self.crop_frames))
            end = min(total_frames, start + self.crop_frames)

            crop = tensor[idx: idx + 1, start:end]
            mask_crop = mask[idx: idx + 1, start:end]
            missing = self.crop_frames - crop.shape[1]
            if missing > 0:
                crop = F.pad(crop, (0, 0, 0, missing))
                mask_crop = F.pad(mask_crop, (0, missing))
            crops.append(crop)
            mask_crops.append(mask_crop)
        return torch.cat(crops, dim=0), torch.cat(mask_crops, dim=0)

    # ------------------------------------------------------------------ audio

    @staticmethod
    def _resample_24k_to_16k(waveform):
        length = max(1, int(round(waveform.shape[-1] * (2.0 / 3.0))))
        return F.interpolate(
            waveform.unsqueeze(1), size=length, mode="linear", align_corners=False
        )[:, 0]

    @staticmethod
    def _normalise_waveform(waveform):
        mean = waveform.mean(dim=-1, keepdim=True)
        var = waveform.var(dim=-1, keepdim=True, unbiased=False)
        return (waveform - mean) * torch.rsqrt(var + 1e-7)

    def _encode(self, waveform):
        features = self._feature_encoder(waveform)
        if hasattr(features, "last_hidden_state"):
            features = features.last_hidden_state.transpose(1, 2)
        return features.float()

    # ------------------------------------------------------------------- loss

    def __call__(self, predicted, target, time_mask):
        """Masked HuBERT feature MSE. Gradients enter ``predicted`` only."""
        mask = time_mask[..., 0] if time_mask.dim() == 3 else time_mask
        valid = torch.nonzero(mask.gt(0.5).any(dim=1), as_tuple=False).flatten()
        if valid.numel() == 0:
            # Keep the graph alive with an exact zero rather than returning a
            # bare constant, so a batch with no edited item cannot desync the
            # optimiser step.
            return predicted.sum() * 0.0
        valid = valid[: self.items_per_batch]
        pred_crop, crop_mask = self._crop(predicted, mask, valid)
        target_crop, _ = self._crop(target, mask, valid)
        self._ensure_loaded(predicted.device)

        # BigVGAN wants [B, mel, time]. Its parameters are frozen, but the
        # prediction call intentionally retains the input computation graph.
        pred_wave = self._vocoder(pred_crop.transpose(1, 2).float()).squeeze(1)
        with torch.no_grad():
            target_wave = self._vocoder(
                target_crop.transpose(1, 2).float()
            ).squeeze(1)
        pred_wave = self._normalise_waveform(self._resample_24k_to_16k(pred_wave))
        target_wave = self._normalise_waveform(
            self._resample_24k_to_16k(target_wave)
        )

        pred_features = self._encode(pred_wave)
        with torch.no_grad():
            target_features = self._encode(target_wave)
        if pred_features.shape != target_features.shape:
            raise RuntimeError(
                "HuBERT teacher feature shapes disagree: "
                f"{pred_features.shape} vs {target_features.shape}"
            )

        feature_mask = F.interpolate(
            crop_mask[:, None].float(), size=pred_features.shape[-1], mode="nearest"
        )
        # Dilate by one feature either side so the loss sees coarticulation at
        # the join, not only the strictly-edited frames.
        feature_mask = F.max_pool1d(feature_mask, kernel_size=3, stride=1, padding=1)

        squared = (pred_features - target_features).square() * feature_mask
        denom = feature_mask.sum().clamp(min=1.0) * pred_features.shape[1]
        return squared.sum() / denom
