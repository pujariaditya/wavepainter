"""Full-reference quality metrics: PESQ and STOI.

WHY FULL-REFERENCE. These replace the no-reference MOS estimator this project
used to report. A no-reference estimator scores audio on its own, with no
comparison signal, and on this benchmark that cannot separate systems: the one
in question rated copy-synthesis of the unedited audio, and the original
recordings themselves, BELOW the values published for edited output. Trained for
noise suppression, it rewards a signal that sounds cleaner than its source --
the opposite of what a speech editor is asked to do. PESQ compares against a
reference and cannot be won that way. The numbers: it scored the original
recordings 3.060 and copy-synthesis 3.068, both below the 3.09-3.18 published
for edited output, and rated a ground-truth waveform splice 2.995 -- lowest of
all.

WHAT THE REFERENCE IS. The benchmark ships no ground-truth *edited* audio, so
there is nothing to compare a full utterance against. Two measurements are
available and both are reported:

  ``vocoder_ceiling``    original vs copy-synthesis of the same audio, no edit.
                         How much fidelity the architecture loses to the vocoder
                         alone -- the ceiling any system decoding through it can
                         reach.

  ``preservation``       original vs system output, over the region BEFORE the
                         edit only. A masked-span editor is supposed to leave
                         that untouched, so this measures the property that
                         distinguishes it from full resynthesis. The edit span
                         comes from the frozen protocol, so the compared region
                         contains no edited content by construction.

PESQ needs 16 kHz (wideband mode) and signals of equal length; both are handled
here. It also needs roughly a quarter-second of audio, so items whose edit
begins too early are skipped and counted rather than silently scored.
"""

import numpy as np

PESQ_SR = 16000
MIN_SECONDS = 0.30


def _load_mono(path, sr=PESQ_SR):
    import soundfile as sf
    import scipy.signal

    wav, rate = sf.read(path, dtype="float64", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if rate != sr:
        # Fourier resampling, matching the convention used for the WER path so
        # the two metrics see the same signal.
        wav = scipy.signal.resample(wav, int(len(wav) * sr / rate))
    return wav


def _pesq_pair(ref, deg, sr=PESQ_SR):
    """Wideband PESQ over the common prefix of two signals.

    Truncating to the shorter length is safe here because the compared regions
    start at the same instant; PESQ does its own fine alignment within a few
    milliseconds after that.
    """
    from pesq import pesq as _pesq

    n = min(len(ref), len(deg))
    if n < MIN_SECONDS * sr:
        return None
    try:
        return float(_pesq(sr, ref[:n].astype(np.float32),
                           deg[:n].astype(np.float32), "wb"))
    except Exception:
        # PESQ raises when it finds no speech in the compared region -- a
        # pre-edit prefix that is entirely leading silence, for instance. That
        # is a property of the item, not a failure, so it is counted as skipped
        # rather than crashing the run or being scored as zero.
        return None


def _stoi_pair(ref, deg, sr=PESQ_SR):
    from pystoi import stoi as _stoi

    n = min(len(ref), len(deg))
    if n < MIN_SECONDS * sr:
        return None
    return float(_stoi(ref[:n], deg[:n], sr, extended=False))


def vocoder_ceiling(pairs):
    """PESQ/STOI for (original, copy-synthesis) pairs -- no edit involved."""
    return _aggregate(pairs, crop_to=None)


def preservation(pairs):
    """PESQ/STOI over the unedited audio: the region before AND after the edit.

    ``pairs`` is (original_path, system_path, edit_start_s, edit_end_s,
    source_duration_s).

    Both untouched regions are used, and that is not an optimisation -- it is
    what makes the measurement representative. Scoring the prefix alone
    discards every item whose edit begins in the first 0.3 s, and those are not
    a random sample: deletion edits start early (25th percentile 0.18 s against
    1.08 s for substitution), so a prefix-only score silently reports on
    late-edit items only.

    The trailing region is anchored at the END of each signal rather than at an
    absolute time, because the edit changes the utterance's length: an
    insertion pushes the tail later and a deletion pulls it earlier. Anchoring
    at the end makes the compared samples the same content in both.
    """
    return _aggregate(pairs, crop_to="unedited")


def _aggregate(pairs, crop_to):
    pesq_scores, stoi_scores, skipped = [], [], 0
    for entry in pairs:
        if crop_to == "unedited":
            ref_path, deg_path, start_s, end_s, dur_s = entry
        else:
            ref_path, deg_path = entry
            start_s = None

        ref, deg = _load_mono(ref_path), _load_mono(deg_path)

        if start_s is not None:
            head = int(start_s * PESQ_SR)
            # Tail length is the source audio after the edit; take it from the
            # end of each signal so the two carry the same content despite the
            # edit having changed the total duration.
            tail = max(0, int((dur_s - end_s) * PESQ_SR))
            tail = min(tail, len(ref) - head, len(deg) - head)
            parts_ref, parts_deg = [], []
            if head > 0:
                parts_ref.append(ref[:head]); parts_deg.append(deg[:head])
            if tail > 0:
                parts_ref.append(ref[-tail:]); parts_deg.append(deg[-tail:])
            if not parts_ref:
                skipped += 1
                continue
            ref = np.concatenate(parts_ref)
            deg = np.concatenate(parts_deg)

        p = _pesq_pair(ref, deg)
        if p is None:
            skipped += 1
            continue
        pesq_scores.append(p)
        s = _stoi_pair(ref, deg)
        if s is not None:
            stoi_scores.append(s)

    return {
        "pesq": round(float(np.mean(pesq_scores)), 3) if pesq_scores else float("nan"),
        "pesq_std": round(float(np.std(pesq_scores)), 3) if pesq_scores else float("nan"),
        "stoi": round(float(np.mean(stoi_scores)), 4) if stoi_scores else float("nan"),
        "n": len(pesq_scores),
        "n_skipped": skipped,
    }
