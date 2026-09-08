"""BigVGAN v2 as the output stage, replacing HiFi-GAN.

WHY: the decoder is the binding constraint on this task. Measured copy-synthesis
speaker similarity -- the ceiling a perfect editor could reach through each
vocoder, since copy-synthesis performs no edit at all:

    HiFi-GAN 22.05kHz/80-band   SIM 0.888
    BigVGAN  24kHz/100-band     SIM 0.987

The reported SIM is 0.943 / 0.961 / 0.918, every one of which is ABOVE
HiFi-GAN's ceiling. Through HiFi-GAN those numbers are unreachable no matter how
good the editing is. The vocoder swap is not an optimisation; it is what puts the
result inside reach.

BigVGAN's own cost is measured and reported: copy-synthesis with no edit scores
PESQ 4.351 against a maximum of 4.644, and the edited output sits ~0.27 below
that ceiling (`scripts/measure_quality.py`).

MEL CONVENTION -- the part that silently breaks. BigVGAN's mel is NOT
FluentEditor's:

    FluentEditor  log10, reference-level and min-level-dB normalised
    BigVGAN       torch.log(clamp(x, min=1e-5)), NO normalisation, slaney-norm
                  librosa filterbank, centre=False

Feeding one into the other's generator does not produce degraded audio, it
produces noise. So a model trained to emit FluentEditor mel cannot be pointed at
this vocoder by changing a config key -- the data has to be binarised with
`mel_type: bigvgan` as well. `spec2wav` asserts the band count for that reason;
it is the one mismatch that would otherwise be silent, because 80-band mel fed
to a 100-band generator fails loudly but a WRONGLY SCALED 100-band mel does not.
"""

import numpy as np
import torch

from wavepainter.tasks.vocoder_infer.base_vocoder import BaseVocoder, register_vocoder
from wavepainter.runtime.hparams import hparams

_BIGVGAN_REPO = "nvidia/bigvgan_v2_24khz_100band_256x"
# PINNED, deliberately. The vocoder is part of the measuring instrument: it turns
# mel into the waveform that both the ASR and the speaker model score, and it is
# the differentiable bridge inside the phase-2 HuBERT loss. A different revision
# moves every reported number, so this resolves one specific snapshot rather than
# whatever `main` happens to be.
_BIGVGAN_REVISION = "c329ede9e9bbc100ddf5c91e2330a61921262370"
_BIGVGAN_DIR = None


def bigvgan_dir():
    """Resolve the pinned BigVGAN snapshot, fetching it if not already cached."""
    global _BIGVGAN_DIR
    override = hparams.get("bigvgan_dir")
    if override:
        return override
    if _BIGVGAN_DIR is None:
        from huggingface_hub import snapshot_download

        _BIGVGAN_DIR = snapshot_download(_BIGVGAN_REPO, revision=_BIGVGAN_REVISION)
    return _BIGVGAN_DIR


class _bigvgan_imports:
    """Import BigVGAN's modules without letting them capture the name `utils`.

    BigVGAN's `bigvgan.py` does `from utils import init_weights, get_padding`,
    meaning ITS `utils.py`. This repo no longer has a top-level `utils` package
    to collide with -- it is `wavepainter.runtime` and friends now -- so the
    original ImportError is gone. The eviction still earns its place: BigVGAN's
    `utils`, `env`, `meldataset` and `activations` are generic names, and leaving
    them in sys.modules after the context exits would shadow anything else that
    later imports those names from somewhere other than the checkpoint dir.

    Prepending the checkpoint dir is not enough on its own, because a name
    already in sys.modules means Python never consults sys.path again. So the
    entries are evicted for the duration and put back afterwards.
    """

    _SHADOWED = ("utils", "env", "activations", "meldataset", "alias_free_activation")

    def __enter__(self):
        import sys
        self._path = list(sys.path)
        self._saved = {k: v for k, v in sys.modules.items()
                       if k in self._SHADOWED or any(
                           k.startswith(p + ".") for p in self._SHADOWED)}
        for k in self._saved:
            sys.modules.pop(k, None)
        sys.path.insert(0, bigvgan_dir())
        return self

    def __exit__(self, *exc):
        import sys
        sys.path[:] = self._path
        for k in list(sys.modules):
            if k in self._SHADOWED or any(k.startswith(p + ".") for p in self._SHADOWED):
                sys.modules.pop(k, None)
        sys.modules.update(self._saved)
        return False


_MEL_FN = None


def bigvgan_mel(wav, hp=None):
    """BigVGAN's own mel, for binarization and for eval. [T, num_mels].

    Imported from the checkpoint's shipped `meldataset.py` rather than
    reimplemented: the filterbank normalisation (slaney), the window, and
    centre=False all have to match the generator's training, and a
    reimplementation that differs in any of them is wrong in a way no shape
    check can see.
    """
    # Held across calls. The import context evicts `meldataset` from sys.modules
    # on exit, so re-importing per utterance would also rebuild its
    # mel_basis/hann caches -- a librosa.filters.mel call per utterance, 32k
    # times, for a filterbank that never changes.
    global _MEL_FN
    if _MEL_FN is None:
        with _bigvgan_imports():
            from meldataset import mel_spectrogram
            _MEL_FN = mel_spectrogram
    mel_spectrogram = _MEL_FN

    hp = hp if hp is not None else hparams
    x = torch.as_tensor(wav, dtype=torch.float32)
    if x.dim() == 1:
        x = x[None]
    mel = mel_spectrogram(
        x, n_fft=hp["fft_size"], num_mels=hp["audio_num_mel_bins"],
        sampling_rate=hp["audio_sample_rate"], hop_size=hp["hop_size"],
        win_size=hp["win_size"], fmin=hp["fmin"],
        fmax=(None if hp.get("fmax") in (None, "null", 0) else hp["fmax"]),
        center=False)
    return mel.squeeze(0).T.cpu().numpy()


@register_vocoder("BigVGAN")
class BigVGANVocoder(BaseVocoder):
    def __init__(self):
        with _bigvgan_imports():
            from bigvgan import BigVGAN

        d = bigvgan_dir()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # use_cuda_kernel=False: the fused anti-aliased activation needs a JIT
        # build against the local toolchain. The torch fallback is numerically
        # equivalent and this runs once per eval, not per training step.
        self.model = BigVGAN.from_pretrained(d, use_cuda_kernel=False)
        self.model.remove_weight_norm()
        self.model.to(self.device).eval()
        self.h = self.model.h

    def spec2wav(self, mel, **kwargs):
        mel = np.asarray(mel)
        if mel.shape[-1] != self.h.num_mels:
            raise ValueError(
                f"BigVGAN expects {self.h.num_mels}-band mel, got {mel.shape[-1]}. "
                f"A model trained on FluentEditor's 80-band 22.05kHz mel cannot be "
                f"decoded here -- re-binarise with mel_type: bigvgan.")
        with torch.no_grad():
            c = torch.FloatTensor(mel).unsqueeze(0).to(self.device).transpose(2, 1)
            y = self.model(c).view(-1)
        return y.cpu().numpy()

    @staticmethod
    def wav2spec(wav_fn):
        import librosa
        wav, _ = librosa.core.load(wav_fn, sr=hparams["audio_sample_rate"])
        return wav, bigvgan_mel(wav)
