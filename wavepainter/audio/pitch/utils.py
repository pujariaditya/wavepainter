"""F0 normalisation and quantisation.

Three representations move between the data pipeline and the model:

  * **Hz** -- what the pitch extractor produces. 0 marks an unvoiced frame.
  * **normalised** -- log2 Hz, with unvoiced frames zeroed and interpolated
    across so the model never sees a discontinuity where voicing stops. This is
    what the model predicts.
  * **coarse** -- an integer bin index on the mel scale, for the pitch
    embedding table.

Unvoiced frames are tracked separately, in the ``uv`` mask, because 0 is a
legitimate value in the normalised domain and cannot double as a flag.

Under the default 'log' normalisation both converters build a new array before
zeroing the unvoiced frames, so the caller's input survives. That is worth
knowing rather than assuming: the zeroing itself is an in-place write, and it
would reach the caller if a normalisation mode were added that returned the
input unchanged.
"""

import numpy as np
import torch

F0_BIN = 256
F0_MIN = 50.0
F0_MAX = 900.0


def _hz_to_mel(hz):
    """Mel scale, the 1127*ln(1 + f/700) convention."""
    return 1127 * np.log(1 + hz / 700)


def f0_to_coarse(f0, f0_bin=F0_BIN, f0_max=F0_MAX, f0_min=F0_MIN):
    """Hz -> integer bin in [1, f0_bin-1], for the pitch embedding.

    Bin 1 is reserved for unvoiced and for anything at or below ``f0_min``;
    voiced pitch occupies the remaining ``f0_bin - 2`` bins, spaced linearly on
    the mel scale.
    """
    mel_min, mel_max = _hz_to_mel(f0_min), _hz_to_mel(f0_max)
    is_torch = isinstance(f0, torch.Tensor)

    mel = 1127 * (1 + f0 / 700).log() if is_torch else _hz_to_mel(f0)
    voiced = mel > 0
    mel[voiced] = ((mel[voiced] - mel_min) * (f0_bin - 2)
                   / (mel_max - mel_min) + 1)
    mel[mel <= 1] = 1
    mel[mel > f0_bin - 1] = f0_bin - 1

    # +0.5 then truncate is round-half-up; np.rint is round-half-even. The two
    # disagree on exact .5, which is why each branch keeps its own form.
    coarse = (mel + 0.5).long() if is_torch else np.rint(mel).astype(int)
    assert coarse.max() <= f0_bin - 1 and coarse.min() >= 1, \
        (coarse.max(), coarse.min(), f0.min(), f0.max())
    return coarse


def norm_f0(f0, uv, pitch_norm='log', f0_mean=400, f0_std=100):
    """Hz -> normalised, with unvoiced frames zeroed.

    Returns a new array under 'log' and 'standard'; see the module note.
    """
    is_torch = isinstance(f0, torch.Tensor)
    if pitch_norm == 'standard':
        f0 = (f0 - f0_mean) / f0_std
    if pitch_norm == 'log':
        # The epsilon keeps unvoiced frames (0 Hz) finite; they are zeroed next.
        f0 = torch.log2(f0 + 1e-8) if is_torch else np.log2(f0 + 1e-8)
    if uv is not None:
        f0[uv > 0] = 0
    return f0


def norm_interp_f0(f0, pitch_norm='log', f0_mean=None, f0_std=None):
    """Hz -> (normalised-and-interpolated f0, uv mask).

    Unvoiced frames are filled by linear interpolation between the voiced
    frames either side, so the contour the model sees is continuous. The uv
    mask records where that happened; it is the model's job to predict both.

    Works in numpy internally and restores the input's type and device, because
    ``np.interp`` has no torch equivalent worth the dependency.
    """
    is_torch = isinstance(f0, torch.Tensor)
    if is_torch:
        device = f0.device
        f0 = f0.data.cpu().numpy()

    uv = f0 == 0
    f0 = norm_f0(f0, uv, pitch_norm, f0_mean, f0_std)
    if uv.sum() == len(f0):
        f0[uv] = 0                       # nothing voiced: leave it flat
    elif uv.sum() > 0:
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])

    if is_torch:
        return (torch.FloatTensor(f0).to(device),
                torch.FloatTensor(uv).to(device))
    return f0, uv


def denorm_f0(f0, uv, pitch_norm='log', f0_mean=400, f0_std=100,
              pitch_padding=None, min=F0_MIN, max=F0_MAX):
    """Normalised -> Hz, with unvoiced and padded frames zeroed.

    Clamped to the extractor's range: the model can emit values outside it, and
    an unclamped exp turns a small error in log space into a wild F0. The clamp
    also means a new array is returned rather than the caller's being modified.
    """
    is_torch = isinstance(f0, torch.Tensor)
    if pitch_norm == 'standard':
        f0 = f0 * f0_std + f0_mean
    if pitch_norm == 'log':
        f0 = 2 ** f0
    f0 = f0.clamp(min=min, max=max) if is_torch else np.clip(f0, min, max)
    if uv is not None:
        f0[uv > 0] = 0
    if pitch_padding is not None:
        f0[pitch_padding] = 0
    return f0
