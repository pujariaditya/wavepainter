"""The diffusion denoiser: predicts the noise added to a mel spectrogram.

A WaveNet-style stack of gated residual blocks, in the DiffWave / DiffSinger
form (Kong et al., *DiffWave*, ICLR 2021; Liu et al., *DiffSinger*, AAAI 2022).
Each block sees three things:

    x               the noisy mel, as a channel stack
    conditioner     per-frame conditioning from the trunk and heads
    diffusion_step  which noise level we are at, as a sinusoidal embedding

and returns a residual (passed to the next block) and a skip connection (summed
across all blocks). The stack output is the predicted x_0 -- the clean mel --
not the noise: both call sites feed it straight in as ``x_start``.

THIS IS THE MODULE PHASE 2 TRAINS -- ``trainable_only: ['denoise_fn.']`` selects
exactly these 170 tensors. The module tree is therefore fixed by the released
checkpoints: ``input_projection``, ``mlp.0``/``mlp.2``, ``residual_layers.N.*``,
``skip_projection`` and ``output_projection`` are all parameter names inside
them. The ``Mish`` at ``mlp.1`` carries no parameters but holds the index, so it
cannot be dropped or reordered either.

Two normalisations keep the stack stable at depth, and both are part of the
trained behaviour rather than free choices: residuals are divided by sqrt(2)
before being passed on, and the summed skips by sqrt(number of blocks).
"""

import math
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from wavepainter.runtime.hparams import hparams


def _conv1d(*args, **kwargs):
    """Conv1d with Kaiming-normal weights.

    The default PyTorch init is Kaiming-uniform with a gain that assumes a
    leaky-ReLU; these convolutions feed tanh/sigmoid gates, and the stack was
    trained under this initialisation.
    """
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class Mish(nn.Module):
    """x * tanh(softplus(x)) -- smooth, non-monotonic activation.

    Parameter-free, but it occupies index 1 of the step-embedding MLP and so is
    part of the checkpoint's naming.
    """

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class SinusoidalPosEmb(nn.Module):
    """Fixed sinusoidal embedding of the diffusion step.

    Same construction as transformer positional encodings: geometrically spaced
    frequencies, sine and cosine concatenated. No parameters -- the step index
    is a known scalar, not something to learn.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        scale = math.log(10000) / (half - 1)
        freqs = torch.exp(torch.arange(half, device=x.device) * -scale)
        angles = x[:, None] * freqs[None, :]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class ResidualBlock(nn.Module):
    """One gated residual block. Returns (residual, skip).

    The step embedding is added to the input before the dilated convolution,
    and the conditioner after it, so conditioning is not squeezed through the
    same gate as the noise level.
    """

    def __init__(self, encoder_hidden, residual_channels, dilation):
        super().__init__()
        # Doubled output width: the halves become the gate and the filter.
        self.dilated_conv = _conv1d(residual_channels, 2 * residual_channels,
                                    3, padding=dilation, dilation=dilation)
        self.diffusion_projection = nn.Linear(residual_channels, residual_channels)
        self.conditioner_projection = _conv1d(encoder_hidden,
                                              2 * residual_channels, 1)
        self.output_projection = _conv1d(residual_channels,
                                         2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step):
        step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = self.dilated_conv(x + step) + self.conditioner_projection(conditioner)

        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)

        residual, skip = torch.chunk(self.output_projection(y), 2, dim=1)
        # sqrt(2) keeps the residual path's variance from growing with depth.
        return (x + residual) / sqrt(2.0), skip


class DiffNet(nn.Module):
    """Noise predictor over mel spectrograms."""

    def __init__(self, in_dims=80):
        super().__init__()
        # cond_dim lets the conditioner be wider than hidden_size, for
        # configurations that concatenate extra state onto the trunk output.
        # Falsy (the default) means "same as hidden_size".
        encoder_hidden = hparams.get('cond_dim') or hparams['hidden_size']
        residual_channels = hparams['residual_channels']
        n_layers = hparams['residual_layers']
        cycle = hparams['dilation_cycle_length']

        self.input_projection = _conv1d(in_dims, residual_channels, 1)
        self.diffusion_embedding = SinusoidalPosEmb(residual_channels)
        self.mlp = nn.Sequential(
            nn.Linear(residual_channels, residual_channels * 4),
            Mish(),
            nn.Linear(residual_channels * 4, residual_channels),
        )
        # Dilation cycles through powers of two with period `cycle`. At
        # cycle == 1 -- this model's setting -- every block has dilation 1 and
        # the receptive field grows linearly rather than exponentially.
        self.residual_layers = nn.ModuleList([
            ResidualBlock(encoder_hidden, residual_channels, 2 ** (i % cycle))
            for i in range(n_layers)
        ])
        self.skip_projection = _conv1d(residual_channels, residual_channels, 1)
        self.output_projection = _conv1d(residual_channels, in_dims, 1)
        # Zero-init: the stack predicts no noise at step 0, so training starts
        # from the identity rather than from random noise injection.
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        """spec ``[B, 1, M, T]``, diffusion_step ``[B]``, cond ``[B, H, T]``
        -> predicted x_0 ``[B, 1, M, T]`` (not noise; see the module docstring).

        ``H`` is the conditioner width, ``cond_dim or hidden_size`` -- not the
        mel-bin count ``M``."""
        x = F.relu(self.input_projection(spec[:, 0]))

        step = self.mlp(self.diffusion_embedding(diffusion_step))

        skips = []
        for layer in self.residual_layers:
            x, skip = layer(x, cond, step)
            skips.append(skip)

        x = torch.sum(torch.stack(skips), dim=0) / sqrt(len(self.residual_layers))
        x = F.relu(self.skip_projection(x))
        return self.output_projection(x)[:, None, :, :]
