"""Projection from mel frames into the decoder's conditioning width.

The masked reference mel is added to the denoiser's conditioning so the model
can see the audio it must blend into. This maps a mel frame to the hidden width
that conditioning is expressed in.

The module tree is fixed by the released checkpoints -- ``encoder.0``,
``encoder.2`` and ``fc_out`` are parameter names inside them, so the two Linear
layers must sit at those positions in the Sequential and the output projection
must keep that attribute name. Reordering, renaming, or dropping either
non-parametric activation would shift the indices and the weights would no
longer load.
"""

import torch.nn as nn


class MelEncoder(nn.Module):
    """Two-layer MLP over mel frames, then a linear projection.

    Applied per frame: input ``[..., input_dim]`` -> output ``[..., hidden_size]``.
    """

    def __init__(self, input_dim=80, hidden_size=192):
        super().__init__()
        # Indices matter: Linear at 0 and 2, activations at 1 and 3.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.fc_out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.fc_out(self.encoder(x))
