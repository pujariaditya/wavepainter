"""Metrics for Ming-Freeform-Audio-Edit, reimplemented from the official harness.

Reimplemented from `github.com/inclusionAI/Ming-Freeform-Audio-Edit` source, not
from its README, because the README and the code disagree in three places that
each move the number:

  * "WER of the edited region" is a misnomer. Nothing in the harness ever slices
    audio; it is whole-utterance ASR against `edited_text`. Ren et al. v1 agrees:
    "WER: Calculated on the full edited utterance."
  * The headline WER is a MEAN OF PER-UTTERANCE WERs (`average_wer.py` does
    `round(np.mean(per_utterance) * 100, 3)`), not the corpus-level
    sum-errors/sum-words that is the statistically better estimator. We match the
    harness, because the entire point is comparability with a printed table, and
    report corpus-level beside it as a labelled diagnostic.
  * The non-edited-region WER excises tokens only when ACC passed; on ACC==0 the
    FULL utterance enters the same corpus sum. That quirk is preserved -- it is
    what the published numbers actually measure.

Baseline, Ren et al. arXiv:2602.00560 Table 1, row "Ours (w. GRPO)", as
WER / SIM:

    substitution   4.41 / 0.78
    insertion      4.97 / 0.82
    deletion       6.88 / 0.78

Table 1's columns are `basic | full`, two subsets of the benchmark -- NOT two
span conditions. The values above are the **full** column, which is the split our
frozen protocols cover (256 / 199 / 200 items). Quoting the `basic` column
instead would understate the substitution and insertion baselines by 0.28 and
0.47 WER. The authoritative copy of these constants is
`wavepainter.metrics.baselines`; this note is orientation, not a second source.

Measured ground-truth WER floor on our frozen substitution protocol: 1.40%.
"""

import difflib
import string
import unicodedata

import numpy as np

# The harness strips `zhon.hanzi.punctuation + string.punctuation` with the
# apostrophe explicitly spared, so "don't" survives as one token. 114 chars.
try:
    import zhon.hanzi
    _PUNCT = zhon.hanzi.punctuation + string.punctuation
except ImportError:  # pragma: no cover - zhon is a hard dependency, fail loudly
    raise ImportError("zhon is required to reproduce the harness punctuation set")


def normalize_text(s, lang="en"):
    """The harness's `process_one` normalisation, exactly.

    Note the single `"  " -> " "` pass: the source does one replace, not a loop,
    so three consecutive spaces collapse to two rather than one. Reproduced
    deliberately -- matching a published number means matching its bugs.
    """
    for ch in _PUNCT:
        if ch == "'":
            continue
        s = s.replace(ch, "")
    s = s.replace("  ", " ")
    if lang == "zh":
        return " ".join(list(s))
    return s.lower()


def _sdi(ref, hyp):
    """(substitutions, deletions, insertions, hits) between two token strings.

    jiwer 4.0.0 removed `compute_measures`, which the harness (jiwer 3.0.1) used.
    `process_words` exposes the same counts, so the arithmetic is unchanged.
    """
    import jiwer

    out = jiwer.process_words(ref if ref.strip() else "<empty>",
                              hyp if hyp.strip() else "<empty>")
    return out.substitutions, out.deletions, out.insertions, out.hits


def wer_one(ref_raw, hyp_raw, lang="en"):
    """Per-utterance WER, exactly the `measures["wer"]` the harness reports.

    `run_wer.process_one` returns jiwer's own `wer`, whose denominator is the
    alignment's reference length S+D+H. It computes `len(ref_list)` separately,
    but only to scale the per-class subs/dele/inse breakdown -- which
    `average_wer.py` never reads. We divided by `len(ref.split(" "))` here, and
    the two disagree exactly when the normalised reference contains an empty
    token: `normalize_text` does a SINGLE `"  " -> " "` pass, faithfully
    reproducing the harness's own bug, so "a   b" becomes "a  b" and splits to
    ['a', '', 'b'] -- three tokens where the aligner sees two, understating WER
    by a third on that utterance.
    """
    ref, hyp = normalize_text(ref_raw, lang), normalize_text(hyp_raw, lang)
    s, d, i, h = _sdi(ref, hyp)
    n = max(1, s + d + h)
    return (s + d + i) / n, (s + d + i), n


def wer_aggregate(pairs, lang="en"):
    """Both aggregations. `headline` is the harness's; `corpus` is the honest one."""
    per, errs, words = [], 0, 0
    for ref, hyp in pairs:
        w, e, n = wer_one(ref, hyp, lang)
        per.append(w)
        errs += e
        words += n
    return {
        "wer_headline_pct": round(float(np.mean(per)) * 100, 3) if per else float("nan"),
        "wer_corpus_pct": round(100 * errs / max(1, words), 3),
        "n_items": len(per),
        "n_words": words,
    }


# `eval_wer.py`'s tokenizer, ported. Only these 18 characters are stripped, and
# only when they START a token -- an ASCII period, hyphen or apostrophe attached
# to a word stays INSIDE it, so "world." and "world" are different tokens. This
# is emphatically not `normalize_text`, which strips all 114 punctuation
# characters wherever they appear; the harness uses the two in different places
# and so must we.
_EVAL_WER_PUNCTS = ["!", ",", "?", "、", "。", "！", "，", "；", "？", "：",
                    "「", "」", "︰", "『", "』", "《", "》", "|"]
_SPACELIST = [" ", "\t", "\r", "\n"]


def characterize(s):
    """`eval_wer.characterize`, verbatim in behaviour.

    CJK (category Lo) splits per character; everything else is scanned forward to
    the next non-ASCII character, whitespace, or separator and emitted whole.
    """
    res, i = [], 0
    while i < len(s):
        char = s[i]
        if char in _EVAL_WER_PUNCTS:
            i += 1
            continue
        cat1 = unicodedata.category(char)
        if cat1 == "Zs" or cat1 == "Cn" or char in _SPACELIST:
            i += 1
            continue
        if cat1 == "Lo":
            res.append(char)
            i += 1
        else:
            sep = ">" if char == "<" else " "
            j = i + 1
            while j < len(s):
                c = s[j]
                if ord(c) >= 128 or (c in _SPACELIST) or (c == sep):
                    break
                j += 1
            if j < len(s) and s[j] == ">":
                j += 1
            res.append(s[i:j])
            i = j
    return res


def _first_edit(a_tokens, b_tokens, want_tag):
    """First non-equal opcode of a word diff, IF it is the expected operation.

    `get_acc.py`'s three validators each open with `if result["type"] == "..."`
    and fall through to `return (0, ref, hyp)` otherwise, so an item filed as an
    insertion whose diff comes back `replace` scores ACC 0 -- it is not re-routed
    to the substitution validator and it is not dropped. Ignoring the tag, as we
    did, silently converted those into scoreable items.

    `autojunk=False` matches `find_edited_region`. It is inert below 200 tokens
    (autojunk only triggers on sequences longer than that) but free to match.
    """
    for op in difflib.SequenceMatcher(None, a_tokens, b_tokens,
                                      autojunk=False).get_opcodes():
        if op[0] != "equal":
            return op if op[0] == want_tag else None
    return None


_WANT_TAG = {"sub": "replace", "ins": "insert", "del": "delete"}


def edit_acc_and_noedit(original_text, edited_text, hyp_raw, edit_type, lang="en"):
    """Edit-operation accuracy plus the reference/hypothesis for noedit WER.

    A port of `get_acc.py`'s `validate_{sub,ins,del}_open`, including its
    normalisation ASYMMETRY, which is not an accident we may tidy away: the
    reference sides (`original_text`, `edited_text`) are `.lower()`ed and nothing
    more, while the hypothesis has already been through `eval_wer.characterize`
    on its way out of the WER file. So a reference token keeps its punctuation
    and so does the hypothesis token it is compared against -- but both differ
    from the `normalize_text` output used for the scored WER.

    We previously ran every side through `normalize_text`. That made "Mr." match
    "mr" and inflated ACC; measured against a GROUND-TRUTH hypothesis it reported
    100/100/100 where a faithful implementation reports sub 99.22 / ins 97.99 /
    del 100.00.

    On ACC==0 the caller folds the returned pair into the noedit corpus sum. For
    sub and ins that pair is the full utterance; for del it is NOT -- see below.
    """
    # `find_edited_region` splits with `.split()`, which drops empty tokens; the
    # scored-WER path splits on a literal single space. Different functions.
    a = original_text.lower().split()
    edited_l = edited_text.lower()
    b = edited_l.split()
    # The hypothesis reaches the official validators already characterized, then
    # lowercased by `process_en`.
    hyp = " ".join(characterize(hyp_raw)).lower()
    hyp_tokens = [t for t in hyp.split(" ") if t]

    op = _first_edit(a, b, _WANT_TAG.get(edit_type, "replace"))
    if op is None:
        return 0, edited_l, hyp

    _tag, i1, i2, _j1, _j2 = op
    # The official code indexes the edited text and the hypothesis with
    # `before_start`, i.e. i1 -- not j1. For the FIRST non-equal opcode the two
    # are always equal (everything before it matched), so this is the same
    # number; using i1 keeps the correspondence to the source obvious.
    start = i1

    if edit_type in ("sub", "ins"):
        inserted = b[start:_j2] if edit_type == "sub" else b[_j1:_j2]
        n = len(inserted)
        if b[start:start + n] == inserted and hyp_tokens[start:start + n] == inserted:
            return (1,
                    " ".join(b[:start] + b[start + n:]),
                    " ".join(hyp_tokens[:start] + hyp_tokens[start + n:]))
        return 0, edited_l, hyp

    if edit_type == "del":
        deleted = a[i1:i2]
        n = len(deleted)
        # The official validator first checks the ORIGINAL really carries those
        # words at that position; only then does it judge the hypothesis.
        if a[start:start + n] == deleted:
            if hyp_tokens[start:start + n] != deleted:
                return 1, edited_l, hyp
            # ACC 0, but NOT the untouched hypothesis: the harness excises the
            # words the model failed to delete before measuring noedit WER, so
            # the failure is charged once (to ACC) rather than twice.
            return (0, edited_l,
                    " ".join(hyp_tokens[:start] + hyp_tokens[start + n:]))
        return 0, edited_l, hyp

    return 0, edited_l, hyp


# --------------------------------------------------------------------- speaker
_SV_MODEL = None


def speaker_sim(pairs, ckpt, device="cuda", sr=16000):
    """Mean cosine of 256-d UniSpeech ECAPA/WavLM-Large embeddings.

    `pairs` is [(edited_wav_path, original_wav_path)]. The reference is the
    ORIGINAL SOURCE audio, whole-to-whole, as the harness does -- not an
    enrolment segment and not the ground-truth edited audio. Verified on this
    checkpoint: self-pair 1.0000, cross-speaker pairs 0.05-0.27.
    """
    global _SV_MODEL
    import torch
    import torch.nn.functional as F

    from .sim_model import load_sv_model

    if _SV_MODEL is None:
        _SV_MODEL = load_sv_model(ckpt, device=device)

    def emb(path):
        w = _read(path, sr)
        with torch.no_grad():
            return _SV_MODEL(torch.tensor(w, dtype=torch.float32)[None].to(device))

    sims = []
    for edited, original in pairs:
        sims.append(float(F.cosine_similarity(emb(edited), emb(original)).item()))
    return {"sim": round(float(np.mean(sims)), 3) if sims else float("nan"),
            "sim_std": round(float(np.std(sims)), 4) if sims else float("nan"),
            "n": len(sims)}


def _read(path, sr):
    """Mono float64 at `sr`, polyphase-resampled. No normalisation, no trimming.

    The harness applies no loudness normalisation, no peak normalisation, no
    silence trimming and no VAD anywhere -- only sample-rate conversion. Adding
    any of them here would silently shift SIM.
    """
    import soundfile as sf
    from scipy.signal import resample_poly

    w, in_sr = sf.read(path, dtype="float64", always_2d=False)
    if w.ndim > 1:
        w = w.mean(1)
    if in_sr != sr:
        g = np.gcd(int(in_sr), sr)
        w = resample_poly(w, sr // g, int(in_sr) // g)
    return w
