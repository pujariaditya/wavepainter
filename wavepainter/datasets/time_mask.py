"""Mask generators for masked-span training.

Four ways to choose which frames the model must regenerate. All return a float
mask that is 1.0 on masked frames and 0.0 elsewhere, shaped like the mel time
axis they were asked about.

Two families:

  * **Frame-level** (``generate_time_mask``) picks a contiguous run of mel
    frames directly.
  * **Alignment-aware** (the other three) picks PHONES and projects the choice
    onto frames through ``mel2ph``, so a masked region always covers whole
    phones. The projection is a gather over a mask padded by one on the left,
    because ``mel2ph`` is 1-indexed -- index 0 means "no phone", and the pad
    gives it a 0.0 entry to land on.

THE RANDOM DRAWS ARE PART OF THE BEHAVIOUR. Each function below uses a specific
generator -- ``torch.randint``, ``np.random.choice`` or ``random.randint`` --
consuming a specific number of values in a specific order. Training runs are
seeded, so swapping a generator or reordering a draw silently produces a
different data stream from the same seed and makes runs incomparable.
"""

import random

import numpy as np
import torch
import torch.nn.functional as F


def _contiguous_span(length, ratio, num_mask, device):
    """Boolean mask over ``length`` positions covering ``num_mask`` runs.

    Each run is ``int(length * ratio)`` wide and starts at a uniformly drawn
    position. Runs may overlap; the result is their union.
    """
    span = int(length * ratio)
    # max(1, ...) keeps randint's range non-empty when the span covers
    # everything, in which case the start is pinned to 0.
    start = torch.randint(0, max(1, length - span), (num_mask, 1), device=device)
    positions = torch.arange(length, device=device)[None, :]
    covered = (start <= positions) * (positions < start + span)
    return covered.any(dim=0).float()


def _phones_to_frames(ph_mask, mel2ph):
    """Project a per-phone mask onto the mel timeline.

    ``mel2ph`` is 1-indexed with 0 meaning "no phone", so the phone mask is
    left-padded by one before the gather: frames with no phone read the pad and
    come back 0.0.
    """
    return torch.gather(F.pad(ph_mask, [1, 0]), 0, mel2ph)


def generate_time_mask(spec, ratio=0.1, num_mask=1, replace_with_zero=True):
    """One contiguous run of mel FRAMES, chosen without reference to alignment.

    ``spec`` is ``[T, F]``; returns ``[T]``.
    """
    return _contiguous_span(spec.shape[0], ratio, num_mask, spec.device)


def generate_alignment_aware_time_mask(spec, mel2ph, ratio=0.1, num_mask=1,
                                       replace_with_zero=True):
    """A random SCATTERED set of phones, projected onto frames.

    The masked phones are drawn without replacement, so the regenerated regions
    are disjoint in phone space but usually not contiguous in time.
    """
    n_phones = mel2ph.max()
    ph_mask = np.zeros((n_phones + 1).item())
    # Population excludes n_phones itself; size follows n_phones + 1.
    population = np.arange(0, n_phones, dtype=float)
    chosen = np.random.choice(
        population, size=int((n_phones + 1) * ratio), replace=False)
    # int64, not uint8: uint8 wraps past 255 phones, so index 256 becomes 0 and
    # silently masks the wrong phone on long LibriTTS items.
    ph_mask[chosen.astype(np.int64)] = 1.0

    return _phones_to_frames(torch.from_numpy(ph_mask).float(), mel2ph)


def generate_continuous_alignment_aware_time_mask(spec, mel2ph, ratio=0.1,
                                                  num_mask=1,
                                                  replace_with_zero=True):
    """One contiguous run of PHONES, projected onto frames.

    This is the training-time mask: a single span of whole phones, which is the
    topology an edit actually has.
    """
    ph_mask = _contiguous_span(mel2ph.max(), ratio, num_mask, spec.device)
    return _phones_to_frames(ph_mask, mel2ph)


def generate_inference_mask(spec, mel2ph, ratio=0.3, num_mask=1,
                            replace_with_zero=True):
    """A contiguous phone span drawn from python's ``random``, for inference.

    Deliberately a different generator from the training masks: this is drawn
    once per utterance at inference rather than per batch during training, and
    keeping it off the torch RNG stops it perturbing a seeded training stream.
    """
    n_phones = mel2ph.max()
    ph_mask = np.zeros((n_phones + 1).item())
    span = int(n_phones * ratio)
    start = random.randint(0, int(n_phones - n_phones * ratio))
    ph_mask[start:start + span] = 1.0

    return _phones_to_frames(torch.from_numpy(ph_mask).float(), mel2ph)
