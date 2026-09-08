"""Noise schedules and small helpers for the diffusion decoder.

The schedule fixes how much noise is added at each of the ``timesteps`` steps of
the forward process, and therefore how much the reverse process must remove.
It is chosen by ``schedule_type`` in the config; this model uses ``vpsde``.

Only the pieces the model actually calls live here. The diffusion process
itself -- the forward/reverse maths and the sampler -- is in
``spec_denoiser.py``, next to the model that owns it, rather than split across
two files.

Schedule references:
  linear/cosine  Nichol & Dhariwal, *Improved Denoising Diffusion Probabilistic
                 Models*, 2021
  vpsde          Song et al., *Score-Based Generative Modeling through
                 Stochastic Differential Equations*, ICLR 2021
  logsnr         Kingma et al., *Variational Diffusion Models*, 2021
"""

from inspect import isfunction

import numpy as np


def exists(x):
    return x is not None


def default(val, d):
    """``val`` if it is not None, else ``d`` -- called if it is a function.

    The callable form lets a caller pass an expensive default (a fresh noise
    tensor, say) without paying for it when ``val`` is supplied.
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    """Pick each batch item's schedule value at its own timestep.

    ``a`` is a per-timestep schedule ``[T]`` and ``t`` is a per-item timestep
    ``[B]``; the result is ``[B, 1, 1, ...]``, shaped to broadcast against
    ``x_shape``. Diffusion batches mix timesteps, so this cannot be a plain
    index.
    """
    b = t.shape[0]
    return a.gather(-1, t).reshape(b, *((1,) * (len(x_shape) - 1)))


def vpsde_beta_t(t, T, min_beta, max_beta):
    """Discretised variance-preserving SDE beta at step ``t`` of ``T``."""
    t_coef = (2 * t - 1) / (T ** 2)
    return 1.0 - np.exp(-min_beta / T - 0.5 * (max_beta - min_beta) * t_coef)


def _logsnr_schedule_cosine(t, *, logsnr_min, logsnr_max):
    """Cosine schedule expressed in log signal-to-noise ratio."""
    b = np.arctan(np.exp(-0.5 * logsnr_max))
    a = np.arctan(np.exp(-0.5 * logsnr_min)) - b
    return -2.0 * np.log(np.tan(a * t + b))


def _cosine_betas(timesteps, s=0.008):
    """Betas implied by a cosine cumulative-alpha schedule.

    Derived from the ratio of successive cumulative alphas, then clipped: the
    last few betas approach 1 and an unclipped value makes the reverse step
    numerically unstable.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


def get_noise_schedule_list(schedule_mode, timesteps, min_beta=0.0,
                            max_beta=0.01, s=0.008):
    """Per-step beta schedule of length ``timesteps``."""
    if schedule_mode == "linear":
        return np.linspace(0.000001, 0.01, timesteps)
    if schedule_mode == "cosine":
        return _cosine_betas(timesteps, s)
    if schedule_mode == "vpsde":
        return np.array([vpsde_beta_t(t, timesteps, min_beta, max_beta)
                         for t in range(1, timesteps + 1)])
    if schedule_mode == "logsnr":
        return np.array([
            _logsnr_schedule_cosine(t / timesteps, logsnr_min=-20.0,
                                    logsnr_max=20.0)
            for t in range(1, timesteps + 1)])
    raise NotImplementedError(f"unknown schedule_mode {schedule_mode!r}")
