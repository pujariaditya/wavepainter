"""Published baseline figures and frozen protocol fingerprints.

Constants only -- no scoring function lives here. Every number wavepainter
reports is a per-edit-type WER or SIM, compared directly against the values
below.

BASELINE
    Ren et al., arXiv:2602.00560, "Edit Content, Preserve Acoustics:
    Imperceptible Text-Based Speech Editing via Self-Consistency Rewards",
    Table 1, English, `full` column, the row **with GRPO**.

    256 / 199 / 200 items for substitution / insertion / deletion is the
    complete official English full split, which is exactly what the frozen
    protocols hold -- so the published row and the items we score against are
    the same set.

    Their Table 1 also carries a third column, a no-reference MOS estimator.
    It is deliberately not mirrored here and `comparison_table` does not cover
    it: on this benchmark that estimator ranks the untouched original
    recordings BELOW synthesised output, so it measures recording cleanliness
    rather than fidelity to the source, and a comparison on it would not mean
    what it appears to mean. Signal quality is instead reported
    full-reference, by `scripts/measure_quality.py` (PESQ/STOI against the
    source).
"""

# --------------------------------------------------------------- the baseline

REN_ET_AL_2602_00560 = {
    "sub": {"wer": 4.41, "sim": 0.78},
    "ins": {"wer": 4.97, "sim": 0.82},
    "del": {"wer": 6.88, "sim": 0.78},
}

CITATION = (
    "Ren et al., 'Edit Content, Preserve Acoustics: Imperceptible Text-Based "
    "Speech Editing via Self-Consistency Rewards', arXiv:2602.00560"
)

# ------------------------------------------------------------------ this work

# The released model: phase two as trained, no interpolation. These are the
# values `./scripts/verify_benchmark.sh` reproduces and the paper reports.
WAVEPAINTER = {
    "sub": {"wer": 3.479, "sim": 0.943},
    "ins": {"wer": 4.400, "sim": 0.961},
    "del": {"wer": 9.697, "sim": 0.918},
}

# Full-reference signal quality, over the audio the edit did not touch. The
# vocoder ceiling -- copy-synthesis with no edit at all -- is PESQ 4.351 /
# STOI 0.995, so these sit about 0.27 PESQ below what this decoder can reach.
# Measured by `scripts/measure_quality.py`; no baseline publishes a comparable
# figure, so these are reported, not compared.
WAVEPAINTER_QUALITY = {
    "sub": {"pesq": 4.090, "stoi": 0.9711},
    "ins": {"pesq": 4.083, "stoi": 0.9619},
    "del": {"pesq": 4.079, "stoi": 0.9667},
}
VOCODER_CEILING = {"pesq": 4.351, "stoi": 0.9950}

# The alpha=0.70 interpolation, published as a second operating point. Against
# the released child it is lower on all three WER legs (3.381/3.817/9.525 against
# 3.479/4.400/9.697). It does not ship as the headline because the alpha sweep
# was read off the test set; see docs/RESULTS.md section 3. SIM is unchanged at
# every alpha.
WAVEPAINTER_A070 = {
    "sub": {"wer": 3.381, "sim": 0.943},
    "ins": {"wer": 3.817, "sim": 0.960},
    "del": {"wer": 9.525, "sim": 0.918},
}

# The phase-1 base, released so phase 2 can be reproduced independently.
WAVEPAINTER_PHASE1 = {
    "sub": {"wer": 4.019, "sim": 0.943},
    "ins": {"wer": 4.465, "sim": 0.960},
    "del": {"wer": 10.248, "sim": 0.918},
}

# ----------------------------------------------------------- reference points
# Both measured on this protocol with the same fp32 Whisper and the same
# resampling path used for scoring. Neither is a target; they bound the axis.

# What the UNTOUCHED source transcribes at -- what you score for not editing.
NULL_WER = {"sub": 15.048, "ins": 10.874, "del": 27.974}

# The ground-truth recordings' own headline WER. Nothing can beat it.
FLOOR_WER = 1.705

# ------------------------------------------------------- protocol fingerprints
# Compiled in rather than read from the protocol files: reading a file's own
# self-declared fingerprint proves nothing, since whoever altered the items
# could alter the field. `evaluate` refuses to score against a protocol whose
# item names do not hash to these.

PROTOCOL_FINGERPRINTS = {
    "sub": ("bf5a10028bfb91a5", 256),
    "ins": ("9992974ee907a6b7", 199),
    "del": ("4cee28739664be2a", 200),
}

# Minimum fraction of trunk weights that must match the declared donor for a
# checkpoint to count as warm-started rather than randomly initialised.
PROVENANCE_MIN = 0.4


def verify_protocol(path, edit_type):
    """Raise unless ``path`` is the frozen protocol for ``edit_type``.

    Compares against the constant above, never against the file's own
    ``fingerprint`` field: anyone able to edit the items can edit that field
    too, so a self-declared fingerprint verifies nothing. Returns the items.
    """
    import hashlib
    import json

    want_fp, want_n = PROTOCOL_FINGERPRINTS[edit_type]
    items = json.load(open(path))["items"]
    got = hashlib.sha256(
        json.dumps([i["item_name"] for i in items], sort_keys=True).encode()
    ).hexdigest()[:16]
    if got != want_fp or len(items) != want_n:
        raise RuntimeError(
            f"{edit_type} protocol does not match the frozen one: "
            f"{len(items)} items / {got} vs the expected {want_n} / {want_fp} "
            f"({path}). Refusing to score against an altered protocol."
        )
    return items


def comparison_table():
    """Rows of (edit_type, metric, baseline, ours, beats) for the README table.

    Six rows: WER and SIM for each edit type. We beat five -- substitution and
    insertion WER, and speaker similarity on all three. The one we lose is
    deletion WER.

    This covers both axes on which a like-for-like comparison is meaningful, not
    every column the baseline prints; see the module docstring for the third
    column and why it is excluded rather than lost.
    """
    rows = []
    for edit in ("sub", "ins", "del"):
        for metric, lower_is_better in (("wer", True), ("sim", False)):
            base = REN_ET_AL_2602_00560[edit][metric]
            ours = WAVEPAINTER[edit][metric]
            beats = ours < base if lower_is_better else ours > base
            rows.append((edit, metric, base, ours, beats))
    return rows
