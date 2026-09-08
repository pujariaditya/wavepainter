"""Operations on frame-to-token alignments.

``mel2token`` maps each mel frame to the 1-indexed token (phone or word) that
produced it, with 0 meaning "no token" -- padding, or a frame outside the
alignment. That 1-indexing is why the expansion below pads before gathering:
index 0 has to land on something, and it must land on zeros.
"""

import torch
import torch.nn.functional as F


def clip_mel2token_to_multiple(mel2token, frames_multiple):
    """Truncate the time axis to a whole multiple of ``frames_multiple``.

    Downsampling stacks need the sequence length to divide evenly; trimming is
    done here, once, rather than being rediscovered inside each module.
    """
    keep = mel2token.shape[1] // frames_multiple * frames_multiple
    return mel2token[:, :keep]


def expand_states(h, mel2token):
    """Broadcast per-token states out to per-frame states.

    ``h`` is ``[B, N_tokens, H]`` and the result is ``[B, T_frames, H]``, each
    frame carrying a copy of its token's state.

    The left pad is load-bearing: ``mel2token`` is 1-indexed, so padding ``h``
    with a zero row at position 0 gives frames with no token (index 0) a row of
    zeros to gather. Without it every such frame would silently pick up the
    first real token's state.
    """
    h = F.pad(h, [0, 0, 1, 0])
    index = mel2token[..., None].expand(-1, -1, h.shape[-1])
    return torch.gather(h, 1, index)
