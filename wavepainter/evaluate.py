"""Score a checkpoint on the frozen Ming-Freeform-Audio-Edit protocol.

Synthesises the edited audio for one edit type and writes the per-item record::

    python -m wavepainter.evaluate --exp_name released --ckpt_steps 500 \\
        --edit_type sub --protocol assets/protocols/protocol_ming_en_sub.json \\
        --out out/sub.json

The metrics are then computed from that record by
``python -m wavepainter.metrics.score_cli``, which is what needs ``--sv_ckpt``.
``scripts/verify_benchmark.sh`` runs both legs for all three edit types.

The model is built from the editor packages at the repository root; everything
that decides the number lives here or in ``wavepainter.metrics``. The separation
is deliberate: the scoring path must not import anything the model being scored
could have changed.

The editing engine is ``edit_one`` below. Two of its properties are worth stating
outright, because both are easy to get wrong in a way that inflates the score:

  * **No ground-truth F0 leak.** It builds ``edited_f0`` with the masked span
    left at ZERO and passes ``use_pred_pitch=True``, so pitch inside the edit is
    predicted rather than handed over.
  * **Insertion and deletion need no separate code path.** ``forward_dur(...,
    use_pred_mel2ph=True)`` predicts durations for the edited text and
    ``length_edited`` is signed, so the utterance may grow or shrink. One decode
    path serves all three edit types.

Inputs are built from the frozen protocol, not from any binarised pack, so
scoring depends on no artifact of the training data layout.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

_SCORING_TREE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCORING_TREE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Benchmark root: protocols, source audio, and the alignment pack. Fetched by
# scripts/fetch_benchmark.py; never redistributed by this repository.
MING = os.environ.get(
    "WAVEPAINTER_BENCHMARK", os.path.join(_ROOT, "data", "ming")
)


def benchmark_path(path):
    """Resolve a protocol audio path against the benchmark root.

    Paths are stored RELATIVE to the benchmark root, as the published dataset
    lays them out (`wavs/<name>`), so the same frozen item list works on any
    machine. Absolute paths pass through unchanged, which keeps a locally-built
    protocol working.

    Known alternative layouts are tried explicitly before the recursive
    fallback, because that fallback walks the whole tree once per item and
    returns an arbitrary match when a basename appears twice.
    """
    if not path or os.path.isabs(path):
        return path

    base = os.path.basename(path)
    for candidate in (
        os.path.join(MING, path),               # published layout: wavs/<name>
        os.path.join(MING, "benchmark", path),  # locally-built copy
        os.path.join(MING, "wavs", base),
    ):
        if os.path.exists(candidate):
            return candidate

    # Last resort. Deterministic (sorted) so a duplicate basename at least
    # resolves the same way on every run rather than depending on walk order.
    import glob
    hits = sorted(glob.glob(os.path.join(MING, "**", base), recursive=True))
    return hits[0] if hits else os.path.join(MING, path)


def item_seed(item_name):
    """Deterministic per-item RNG seed -- common random numbers across runs.

    The diffusion sampler draws its noise from the global RNG, so without this
    the same checkpoint scored twice produces different audio and any paired
    comparison is measured against a moving target. Sampler noise alone can move
    the score by more than a real effect does, which would credit changes that
    did nothing and hide changes that worked. This is also what makes the
    reported numbers reproducible to four decimal places rather than
    approximately.

    Seeded from the item NAME, not the loop index, so the value a given
    utterance gets does not shift when --limit changes or items are reordered.
    """
    return int(hashlib.sha256(item_name.encode()).hexdigest()[:8], 16)


def spoken_form(text):
    """Transcript -> what the speaker actually said, for the PHONEMISER only.

    THE SINGLE SOURCE. `scripts/build_ming_protocol.py` imports this exact
    function for its MFA `.lab` files, so the phone sequence the TextGrid was
    aligned against and the phone sequence built here cannot drift apart.

    They did drift once, and it cost a full evaluation. mfa-prep substituted
    "&" -> "and" so the item could be aligned at all, while this path phonemised
    the raw transcript, where the text processor drops "&" entirely. The
    TextGrid then carried 38 phones against 35 from the same utterance --
    exactly the three of "AND" -- and `get_mel2ph` asserted. One item in 655,
    which is enough: scoring a partial set is refused, so a single unalignable
    item fails the whole leg.

    Token COUNT is preserved -- "&" becomes one word, not zero -- which is what
    keeps `word_start`/`word_end` meaningful, since they index
    `original_text.split()`.

    Only the phonemiser sees this. WER is scored against the untouched
    `edited_text`, so nothing here can flatter a metric.
    """
    return " ".join("and" if w == "&" else w for w in text.split())


def real_word_ids(words_str):
    """ph2word ids of the REAL words, in order, from txt_to_ph's `words`.

    `txt_to_ph` builds ph2word as `w_id + 1` over every entry of txt_struct --
    and txt_struct is `<BOS>, word, |, word, |, ..., word, <EOS>`. The word
    separators and the sentinels each occupy an id, so the n-th real word is
    ph2word id 2(n+1), NOT n+1.

    The protocol's `word_start`/`word_end` index `original_text.split()`. Feeding
    those to a ph2word comparison masks a region roughly half way to the right
    place: on the first protocol item it masked "department there" instead of
    "printing equipment". Nothing about that is detectable downstream -- the
    shapes are right, the model edits happily, and the audio is simply wrong.

    Derived from the word list rather than hardcoding the doubling, so a change
    to the separator convention cannot silently reintroduce the same offset.

    Punctuation is excluded too, and not as a nicety: the text processor emits
    standalone punctuation as its OWN txt_struct entry, so "Experts of geology
    agree: Yesterday's ..." is 15 entries where `.split()` sees 14 words. The
    protocol indexes `.split()`, so one colon shifts every index after it and the
    span silently lands on the wrong words -- or, when the shift runs off the
    end, resolves to an empty edited span.

    The returned values are still real ph2word IDS, so dropping punctuation from
    the LIST is correct: the id already accounts for the entries being skipped.
    """
    return [i + 1 for i, w in enumerate(words_str.split(" "))
            if w not in ("<BOS>", "<EOS>", "|") and any(c.isalnum() for c in w)]


def token_word_ids(words_str, transcript):
    """One entry per `transcript.split()` token: the ph2word ids it spans.

    `real_word_ids` assumes the preprocessor emits exactly one real word per
    transcript token. For 4 of the 655 protocol items it does not, and the two
    ways it fails are opposite:

        "pre-configured"  -> ['pre', 'configured']   one token, TWO words
        "&"               -> dropped entirely        one token, ZERO words

    Indexing `real_word_ids(...)[word_start]` then either shifts every span
    after the offending token or runs off the end of the list. Both were being
    avoided upstream by dropping the items -- which is how the protocol lost
    them in the first place.

    Aligns by letters rather than by position, so it needs no list of special
    cases and degenerates to `real_word_ids` exactly when the counts agree.
    """
    def letters(s):
        return "".join(c for c in s.lower() if c.isalnum())

    wl = words_str.split(" ")
    real = [(i + 1, wl[i]) for i in range(len(wl))
            if wl[i] not in ("<BOS>", "<EOS>", "|")
            and any(c.isalnum() for c in wl[i])]

    out, k = [], 0
    for tok in transcript.split():
        want, got, ids = letters(tok), "", []
        while k < len(real) and got != want:
            wid, w = real[k]
            got += letters(w)
            ids.append(wid)
            k += 1
            if not want:          # a token with no letters at all, e.g. "&"
                ids = []
                break
        if got != want and want:
            raise RuntimeError(
                f"cannot align transcript token {tok!r} to the preprocessor's "
                f"words (got {got!r}, want {want!r}) in {transcript!r}")
        out.append(ids)
    return out


def build_inputs(item, tg_dir, preprocessor, ph_encoder, spk_encoder, hp):
    """Everything the editing forward pass needs, from one protocol item."""
    from wavepainter.audio import librosa_wav2spec
    from wavepainter.audio.align import get_mel2ph
    from wavepainter.audio.pitch_extractors import extract_pitch
    from wavepainter.audio.pitch.utils import norm_interp_f0

    # The mel comes from the SAME file SIM is scored against. There is no
    # resampled copy any more: source, front end and SIM reference are all
    # 24 kHz. `mel_type` must match what the model was trained on -- BigVGAN's
    # mel and FluentEditor's have the same shape and a different meaning, so
    # picking the wrong one here produces confident nonsense, not an error.
    src = benchmark_path(item.get("wav_mel") or item.get("wav_22k") or item["wav"])
    if hp.get("mel_type") == "bigvgan":
        import librosa
        from wavepainter.tasks.vocoder_infer.bigvgan import bigvgan_mel
        wav, _ = librosa.core.load(src, sr=hp["audio_sample_rate"])
        mel = bigvgan_mel(wav, hp)
        wav = wav[:mel.shape[0] * hp["hop_size"]]
    else:
        spec = librosa_wav2spec(
            src, fft_size=hp["fft_size"], hop_size=hp["hop_size"],
            win_length=hp["win_size"], num_mels=hp["audio_num_mel_bins"],
            fmin=hp["fmin"], fmax=hp["fmax"], sample_rate=hp["audio_sample_rate"])
        mel, wav = spec["mel"], spec["wav"]

    # `spoken_form`, not the raw text: the TextGrid was aligned against it.
    ph, txt, words, ph2word, _ = preprocessor.txt_to_ph(
        preprocessor.txt_processor, spoken_form(item["original_text"]))
    e_ph, _, e_words, e_ph2word, _ = preprocessor.txt_to_ph(
        preprocessor.txt_processor, spoken_form(item["edited_text"]))

    # MFA mirrors its input layout, so with one directory per utterance the
    # TextGrid sits at <dir>/<name>/<name>.TextGrid, not <dir>/<name>.TextGrid.
    # Accept both rather than depend on how a given alignment run was laid out.
    name = item["item_name"]
    tg_fn = os.path.join(tg_dir, name + ".TextGrid")
    if not os.path.exists(tg_fn):
        tg_fn = os.path.join(tg_dir, name, name + ".TextGrid")
    if not os.path.exists(tg_fn):
        raise FileNotFoundError(
            f"no TextGrid for {name} under {tg_dir} (tried flat and nested)")
    # The 6th argument is `min_sil_duration`, and it defaults to 0 -- i.e. no
    # short-silence merging. Binarisation used 0.1 (`_T2M_DEFAULTS` in
    # wavepainter/datasets/base_binarizer.py:44, consumed at :290), so omitting it here
    # gave the model an alignment built under a different rule than the one it
    # trained on: measured, 178 of 245 substitution items (72.7%) got a different
    # mel2ph, shifting a mean 2.18% of frames and up to 6.84%.
    #
    # `.get` with an explicit default, not a subscript: the saved checkpoint
    # config carries no `mfa_min_sil_duration` key, so `hp[...]` would KeyError
    # at eval time on every existing checkpoint.
    mel2ph, dur = get_mel2ph(tg_fn, ph, mel, hp["hop_size"], hp["audio_sample_rate"],
                             hp.get("mfa_min_sil_duration", 0.1))
    mel2word = [ph2word[p - 1] for p in mel2ph]

    # Exactly the call `BaseBinarizer.process_pitch` makes, argument for
    # argument. The extractor takes (wav, hop, sr, f0_min, f0_max) and returns
    # a bare array -- not (wav, mel, hparams), and not a tuple. Extracting f0
    # differently here than at binarization would feed the model a pitch signal
    # unlike anything it trained on, for every item.
    f0 = extract_pitch(hp["pitch_extractor"], wav, hp["hop_size"],
                       hp["audio_sample_rate"], f0_min=hp["f0_min"],
                       f0_max=hp["f0_max"])
    f0 = np.asarray(f0)
    if len(f0) != len(mel):
        raise RuntimeError(f"f0/mel length mismatch: {len(f0)} vs {len(mel)}")
    # NORMALISE, exactly as the dataset does (dataset_utils.py:205). Extracting
    # f0 identically to the binarizer is only half the contract: the binarizer's
    # output goes through `norm_interp_f0` before the model ever sees it, so raw
    # Hz here is a different quantity under the same name.
    #
    # This was not a subtle skew. `denorm_f0` inverts the LOG2 normalisation with
    # `2 ** f0`, so a 200 Hz frame arriving as 200.0 instead of log2(200)=7.64
    # became 2**200, which the [50, 900] clamp turned into 900 Hz. Measured on a
    # 7-frame example: every voiced context frame landed in coarse bin 255
    # instead of 31/62/55/84/43. The model's entire pitch conditioning at
    # inference was a constant, maximal, meaningless 900 Hz -- while at training
    # it was the real contour. `uv` comes from the same call for the same
    # reason: it must mark the frames the interpolation filled.
    f0, uv = norm_interp_f0(f0)
    f0, uv = np.asarray(f0), np.asarray(uv)

    # The protocol stores 0-based Python slice bounds over `text.split()`; the
    # forward pass compares mel2word against an INCLUSIVE [lo, hi] in ph2word
    # ids. Translate through the real-word map -- see real_word_ids.
    # Indexed by TRANSCRIPT TOKEN, which is what word_start/word_end count, and
    # each entry is the ph2word ids that token occupies. Identical to
    # `real_word_ids` whenever the preprocessor emits one word per token, which
    # is 651 of the 655 items; the other 4 are hyphenated compounds and "&".
    src_tok = token_word_ids(words, spoken_form(item["original_text"]))
    dst_tok = token_word_ids(e_words, spoken_form(item["edited_text"]))
    ws, we = item["word_start"], item["word_end"]
    if not (0 <= ws < we <= len(src_tok)):
        raise RuntimeError(
            f"protocol span [{ws},{we}) outside the {len(src_tok)} tokens of "
            f"the original text for {item['item_name']}")
    # Widen past any token that maps to no word at all ("&"), so a span that
    # starts or ends on one still resolves to real ph2word ids.
    lo_ids = next((src_tok[k] for k in range(ws, we) if src_tok[k]), None)
    hi_ids = next((src_tok[k] for k in range(we - 1, ws - 1, -1) if src_tok[k]), None)
    if lo_ids is None or hi_ids is None:
        raise RuntimeError(
            f"protocol span [{ws},{we}) covers no pronounceable word for "
            f"{item['item_name']}")
    lo, hi = lo_ids[0], hi_ids[-1]
    src_ids, dst_ids = real_word_ids(words), real_word_ids(e_words)
    # The edited-side span is found from the TAIL, not by counting new_words.
    # For a single contiguous replace the prefix and suffix are unchanged, so the
    # number of words after the edit is identical on both sides -- whereas
    # `new_words.split()` disagrees with the preprocessor whenever a replacement
    # is hyphenated ("in-depth essays" is 2 tokens by split() and 3 words after
    # the text processor), which put the edited span one word short on exactly
    # those items.
    e_ws = ws
    e_we = len(dst_tok) - (len(src_tok) - we)
    if not (e_ws < e_we <= len(dst_tok)):
        raise RuntimeError(
            f"edited span [{e_ws},{e_we}) is empty or outside the "
            f"{len(dst_tok)} tokens of the edited text for {item['item_name']}")
    e_lo_ids = next((dst_tok[k] for k in range(e_ws, e_we) if dst_tok[k]), None)
    e_hi_ids = next((dst_tok[k] for k in range(e_we - 1, e_ws - 1, -1) if dst_tok[k]), None)
    if e_lo_ids is None or e_hi_ids is None:
        raise RuntimeError(
            f"edited span [{e_ws},{e_we}) covers no pronounceable word for "
            f"{item['item_name']}")
    e_lo, e_hi = e_lo_ids[0], e_hi_ids[-1]

    # The trunk's sequence is [BPE text | phones | mel frames]. Training always
    # supplies the BPE segment (dataset_utils.py:160-170); this path used to
    # supply nothing, and `_text_segment` turns a missing `text_ids` into a
    # ZERO-WIDTH segment rather than an error -- correct as a failure mode,
    # invisible as a skew. Every evaluation therefore ran a sequence layout the
    # model had never seen, and the one component whose whole justification is
    # "a pretrained language model should do the linguistic work" was switched
    # off exactly when it was being measured.
    #
    # The text is the EDITED transcript, not the original. At training the two
    # coincide -- masked-span reconstruction of one utterance -- so `item['txt']`
    # is unambiguous there. Here they differ, and the edited text is both what
    # the model is asked to realise and what WER is scored against.
    #
    # Same tokenizer, same cap, same helper as training. Re-tokenising with a
    # locally-constructed AutoTokenizer would reintroduce the class of bug this
    # is fixing.
    text_ids = text_pad = None
    if hp.get("use_dualffn_trunk") and hp.get("trunk_use_text", True):
        from wavepainter.tasks.dataset_utils import get_text_tokenizer
        ids = get_text_tokenizer()(
            item["edited_text"], add_special_tokens=False).input_ids
        ids = ids[:int(hp.get("trunk_max_text", 48))] or [0]
        text_ids = torch.LongTensor(ids)[None].cuda()
        text_pad = torch.ones_like(text_ids).float()

    # resemblyzer's 3x256 LSTM over ~10 short windows is far too small to
    # amortise oneDNN's 16-thread fan-out: measured 314 ms at 16 threads vs
    # 112 ms at 1, over six protocol wavs, with the embedding BIT-identical
    # (md5 of .tobytes() equal on all six).
    #
    # Scoped and restored in `finally`. A process-wide change is NOT safe: it
    # also moves the torch STFT in `bigvgan_mel` above (maxabsdiff 9.5e-07) and
    # 11 of 24 end-to-end wavs came back byte-different.
    _prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        spk = spk_encoder.embed_utterance(wav.astype(float))
    finally:
        torch.set_num_threads(_prev_threads)

    return {
        "item_name": item["item_name"],
        "text_ids": text_ids,
        "text_pad": text_pad,
        "mel": torch.FloatTensor(mel)[None].cuda(),
        "wav": wav,
        "mel2ph": torch.LongTensor(mel2ph)[None].cuda(),
        "mel2word": torch.LongTensor(mel2word)[None].cuda(),
        "dur": torch.LongTensor(dur)[None].cuda(),
        "ph2word": torch.LongTensor(ph2word)[None].cuda(),
        "edited_ph2word": torch.LongTensor(e_ph2word)[None].cuda(),
        "txt_tokens": torch.LongTensor(ph_encoder.encode(ph))[None].cuda(),
        "edited_txt_tokens": torch.LongTensor(ph_encoder.encode(e_ph))[None].cuda(),
        "f0": torch.FloatTensor(f0)[None].cuda(),
        "uv": torch.FloatTensor(uv)[None].cuda(),
        "spk_embed": torch.FloatTensor(
            spk[None]).cuda(),
        "words_region": [lo, hi],
        "edited_words_region": [e_lo, e_hi],
    }


def edit_one(model, s, seed=None):
    """FluentEditor's editing forward pass, verbatim in structure.

    What changed is only WHO predicts the durations. FastSpeech's conv duration
    predictor is gone; the Gemma trunk's duration head stands in its place, via
    `GaussianDiffusion.predict_durations`. Everything else here -- the span
    mapped through `real_word_ids`, `masks_orig` living on the ORIGINAL timeline
    while the output lives on the edited one, the head/tail splice taken from
    `*2word` rather than from which phones received frames, and the masked-span
    F0 left at zero under `use_pred_pitch=True` -- is untouched. Each of those
    encodes a bug that produced confident, wrong audio rather than an error.
    """
    mel, mel2ph, mel2word = s["mel"], s["mel2ph"], s["mel2word"]
    ph2word, e_ph2word = s["ph2word"], s["edited_ph2word"]
    e_tokens = s["edited_txt_tokens"]
    lo, hi = s["words_region"]
    e_lo, e_hi = s["edited_words_region"]

    in_span = (mel2word >= lo) & (mel2word <= hi)
    masked_mel2ph = mel2ph.clone()
    masked_mel2ph[in_span] = 0
    masks_orig = torch.zeros_like(mel2ph).float()
    masks_orig[in_span] = 1.0

    # The alignment the duration pass is allowed to see: the ORIGINAL timeline
    # with the edited span blanked, RE-INDEXED into the edited phone sequence.
    # The re-indexing is not cosmetic -- the trunk looks each frame's phone up in
    # `e_tokens`, so leaving the original indices in the tail would hand the
    # context frames phones from a transcript nobody is editing, and the
    # durations would come back plausible and wrong. FastSpeech got the same
    # information as `masked_dur`, a per-phone vector; the trunk has no phone-side
    # injection slot, so it arrives per frame instead.
    known_mel2ph = masked_mel2ph.clone()
    src_tail = (ph2word[0] > hi).nonzero()
    dst_tail = (e_ph2word[0] > e_hi).nonzero()
    if mel2word.max() > hi and src_tail.numel() and dst_tail.numel():
        # +1 on both sides cancels; mel2ph is 1-based into the phone sequence.
        shift = int(dst_tail[0]) - int(src_tail[0])
        tail_frames = mel2word > hi
        known_mel2ph[tail_frames] = mel2ph[tail_frames] + shift
    if int(known_mel2ph.max()) > e_tokens.shape[1] or int(known_mel2ph.min()) < 0:
        raise RuntimeError(
            f"context mel2ph out of range for the edited text: max "
            f"{int(known_mel2ph.max())}, min {int(known_mel2ph.min())}, but the "
            f"edited sequence has {e_tokens.shape[1]} phones")

    # Predicted durations for the EDITED phone sequence. This is what lets the
    # utterance change length, so the same call serves substitution, insertion
    # and deletion.
    with torch.no_grad():
        dur = model.predict_durations(
            e_tokens, masks_orig[:, :, None], known_mel2ph, s["spk_embed"], mel,
            f0=s["f0"], uv=s["uv"],
            text_ids=s.get("text_ids"), text_pad=s.get("text_pad"))
    e_mel2ph = model.heads.mel2ph_from_dur(dur, e_tokens == 0).detach()
    if e_mel2ph.shape[1] == 0 or int(e_mel2ph.min()) < 1:
        raise RuntimeError(
            "the duration head gave the edited utterance no frames at all "
            f"(shape {tuple(e_mel2ph.shape)}); nothing downstream can splice a "
            "zero-length span")
    # One gather. `for p in e_mel2ph[0]` iterated a CUDA tensor and re-ran
    # `.cpu().numpy()` inside the comprehension, so every mel frame paid a host
    # round trip and a device sync: measured 21.1 ms vs 0.044 ms at T=478 on an
    # idle card, and ~50x worse when the card is contended -- which is exactly
    # when an evaluation runs. `e_mel2ph.min() >= 1` is enforced by the raise
    # above, so `- 1` cannot go negative. torch.equal verified at T=271/424/478.
    e_mel2word = e_ph2word[0][e_mel2ph[0] - 1][None]
    e_in_span = (e_mel2word >= e_lo) & (e_mel2word <= e_hi)

    delta = int(e_mel2word[e_in_span].size(0) - mel2word[in_span].size(0))
    head = int(mel2word[mel2word < lo].size(0))
    tail = int(mel2word[mel2word <= hi].size(0)) + delta

    out_T = mel2ph.size(1) + delta
    out_T = mel2ph.size(1) + delta
    new_mel2ph = torch.zeros((1, out_T), device=mel2ph.device)
    new_mel2ph[:, :head] = mel2ph[:, :head]
    new_mel2ph[:, head:tail] = e_mel2ph[e_in_span]
    if mel2word.max() > hi:
        after = mel2ph[mel2word > hi]
        # The tail text is unchanged, so its phones exist verbatim in the edited
        # sequence -- take where they START from `e_ph2word`, which is a fact
        # about the text, and shift the original tail onto them.
        #
        # This was `e_mel2ph[e_in_span].max() + 2`, i.e. "one past the last span
        # phone that received frames". When the duration predictor gives a span
        # phone zero frames that max falls short, every tail index shifts up, and
        # the result indexes PAST the edited phone sequence -- a CUDA
        # device-side assert inside the trunk's ph_emb gather, on 3 of the first
        # 6 protocol items. It cannot be caught by any shape check because the
        # tensor is the right shape; only its values are out of range.
        # BOTH ends come from *2word, never from which phones happened to get
        # frames. `after.min()` has the same defect as the old right-hand side:
        # if the first tail phone is given zero duration it is not the minimum,
        # and the whole tail shifts.
        src_tail = (ph2word[0] > hi).nonzero()
        dst_tail = (e_ph2word[0] > e_hi).nonzero()
        if src_tail.numel() and dst_tail.numel():
            # +1 because mel2ph is 1-based into the phone sequence.
            new_mel2ph[:, tail:] = (after - (int(src_tail[0]) + 1)
                                    + (int(dst_tail[0]) + 1))
    e_mel2ph = new_mel2ph.long()

    # The trunk gathers ph_emb with these indices; an out-of-range value is a
    # device-side assert with no usable traceback. Fail here, on the host, saying
    # which item and by how much.
    n_edit_ph = e_tokens.shape[1]
    if int(e_mel2ph.max()) > n_edit_ph or int(e_mel2ph.min()) < 0:
        raise RuntimeError(
            f"mel2ph out of range for the edited text: max {int(e_mel2ph.max())}, "
            f"min {int(e_mel2ph.min())}, but the edited sequence has {n_edit_ph} "
            f"phones")

    ref = torch.zeros((1, out_T, mel.size(2)), device=mel.device)
    ref[:, :head] = mel[:, :head]
    if mel2word.max() > hi:
        ref[:, tail:] = mel[mel2word > hi]

    # f0 inside the span stays ZERO and use_pred_pitch=True predicts it. This is
    # the GT-F0 pathway the old protocol had to forbid by rule.
    e_f0 = torch.zeros((1, out_T), device=mel.device)
    e_uv = torch.zeros((1, out_T), device=mel.device)
    e_f0[:, :head] = s["f0"][:, :head]
    e_uv[:, :head] = s["uv"][:, :head]
    if mel2word.max() > hi:
        e_f0[:, tail:] = s["f0"][mel2word > hi]
        e_uv[:, tail:] = s["uv"][mel2word > hi]

    masks = torch.zeros((1, out_T, 1), device=mel.device)
    masks[:, head:tail] = 1.0

    # Seed immediately before the sampler, not once per run: a failed item would
    # otherwise shift the RNG for every item after it, so one crash changes the
    # audio of utterances that did not fail.
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    with torch.no_grad():
        # `masks_orig` is on the ORIGINAL timeline, `masks` on the edited one.
        # The features come from the original audio, so the zeroing mask must be
        # the original-time one -- otherwise every insertion and deletion, which
        # by definition change duration, would zero the wrong feature frames.
        # `n_out` inside the encoder is taken from the edited mel, so the output
        # still lands on the edited timeline.
        out = model(e_tokens, time_mel_masks=masks, mel2ph=e_mel2ph,
                    spk_embed=s["spk_embed"], ref_mels=ref, f0=e_f0, uv=e_uv,
                    energy=None, infer=True, use_pred_pitch=True,
                    text_ids=s.get("text_ids"), text_pad=s.get("text_pad"))
        return out["mel_out"] * masks + ref * (1 - masks)


def _arch_facts(model, hp, probe=True):
    """Structural facts read from the LOADED model.

    Reported alongside the metrics so a reader can confirm the checkpoint being
    scored is the one described: that it has a DualFFN trunk, and that the trunk
    really is warm-started from the declared donor rather than randomly
    initialised. ``wavepainter.metrics.baselines.PROVENANCE_MIN`` is the fraction
    of trunk weights that must match for the latter to count.

    ``probe=False`` keeps the three free facts and skips the float64 provenance
    cosines, which cost ~130 s.
    """
    trunk = getattr(model, "trunk", None)
    ws = getattr(model, "trunk_warm_start_report", None)
    arch = {
        "has_trunk": trunk is not None,
        "warm_start_applied": bool(ws),
        "trunk_out_absmax": (float(model.trunk_out.weight.abs().max())
                             if hasattr(model, "trunk_out") else None),
    }
    if not probe:
        return arch

    from wavepainter.models.dualffn.warm_start import donor_provenance

    if trunk is not None:
        # warm_start_applied above is a fact about the CONFIG -- the model is
        # rebuilt here, warm-started, then the trained state_dict is loaded over
        # the top. This asks the checkpoint itself.
        arch["provenance"] = donor_provenance(
            trunk, donor=hp["trunk_donor"])
    return arch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", required=True)
    ap.add_argument("--ckpt_steps", type=int, required=True)
    ap.add_argument("--dir", default=os.environ.get("WAVEPAINTER_ROOT", "."),
                    help="root holding checkpoints/<exp_name>/")
    # Defaults to the protocol matching --edit_type. Pinning it to `sub`
    # meant `--edit_type del` silently pointed at the substitution items; the
    # fingerprint check caught it, but a default that cannot be wrong is better
    # than a guard that reports it.
    ap.add_argument("--protocol", default=None)
    # The alignments SHIP, in assets/alignments/ -- 655 TextGrids, one per
    # protocol item across all three edit types.
    #
    # They are shipped rather than regenerated because they cannot be
    # regenerated: they depend on an MFA version and dictionary that were never
    # pinned, and on orthography fixes (`&` -> `and`, and the recovery of 25
    # items that failed to align) applied by a generator step that is not part
    # of this repository. An independently re-aligned set is a different set --
    # the earlier partial sets here had 630 and 245 items -- and scoring against
    # one silently drops items rather than failing.
    ap.add_argument("--textgrids",
                    default=os.path.join(_ROOT, "assets", "alignments"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--wav_dir", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip_arch", action="store_true",
                    help="skip the trunk provenance probe (~130s)")
    ap.add_argument("--edit_type", default="sub", choices=["sub", "ins", "del"])
    a = ap.parse_args()
    if a.protocol is None:
        a.protocol = os.path.join(_ROOT, "assets", "protocols",
                                  f"protocol_ming_en_{a.edit_type}.json")

    from wavepainter.datasets.base_preprocess import BasePreprocessor
    from resemblyzer import VoiceEncoder
    from wavepainter.audio.io import save_wav
    from wavepainter.runtime.ckpt_utils import load_ckpt
    from wavepainter.runtime.hparams import hparams, set_hparams, vocab_root

    work_dir = os.path.join(a.dir, "checkpoints", a.exp_name)
    # `dir` must be passed through -- set_hparams derives work_dir from it, and
    # without it every run tries to mkdir `/checkpoints` at the filesystem root.
    # It also asserts `os.path.exists(dir + config)`, so `config` is RELATIVE to
    # `dir`; handing it the absolute path concatenates the two into nonsense.
    set_hparams(config=f"checkpoints/{a.exp_name}/config.yaml",
                exp_name=a.exp_name, dir=a.dir, print_hparams=False)

    from wavepainter.tasks.spec_denoiser import SpeechDenoiserTask

    task = SpeechDenoiserTask()
    task.build_model()
    model = task.model.eval().cuda()
    ckpt = f"{work_dir}/model_ckpt_steps_{a.ckpt_steps}.ckpt"
    load_ckpt(model, ckpt, "model")
    vocoder = task.vocoder

    pre = BasePreprocessor()
    ph_encoder, _ = pre.load_dict(vocab_root())
    spk_encoder = VoiceEncoder(device="cpu")

    arch = _arch_facts(model, hparams, probe=not a.skip_arch)
    print(f"| arch: has_trunk={arch['has_trunk']} "
          f"warm_start={arch['warm_start_applied']} "
          f"donor_frac={(arch.get('provenance') or {}).get('frac_matched')}")

    # Refuse an altered protocol outright. Checked against a constant compiled
    # into wavepainter.metrics.baselines, not the file's own fingerprint field.
    from wavepainter.metrics.baselines import verify_protocol

    items = verify_protocol(a.protocol, a.edit_type)
    if a.limit:
        items = items[: a.limit]
    # UNIQUE PER PROCESS. Keyed on exp_name + ckpt_steps alone, two concurrent
    # evaluations of the same checkpoint -- two runs that picked the same
    # exp_name, or a probe and a full evaluation -- write the SAME
    # `<item_name>.wav` paths. One run's save_wav then overwrites the file the
    # other's scorer is mid-read, and nothing raises: `soundfile.read` returns a
    # torn buffer and it is scored as a metric. The result is a plausible WER
    # that describes neither model.
    #
    # `eval_protocol.py` (the retired evaluator) keyed on pid for exactly this
    # reason and the note was lost in the port. `edit_type` is in the key too, so
    # a single evaluation's three legs cannot alias each other either.
    wav_dir = a.wav_dir or os.path.join(
        work_dir, f"ming_eval_{a.ckpt_steps}_{a.edit_type}_{os.getpid()}")
    os.makedirs(wav_dir, exist_ok=True)

    rows, failed = [], {}
    for i, it in enumerate(items, 1):
        try:
            s = build_inputs(it, a.textgrids, pre, ph_encoder, spk_encoder, hparams)
            mel_out = edit_one(model, s, seed=item_seed(it["item_name"]))
            wav = vocoder.spec2wav(mel_out[0].cpu().numpy())
            path = os.path.join(wav_dir, it["item_name"] + ".wav")
            save_wav(wav, path, hparams["audio_sample_rate"])
            rows.append({"item_name": it["item_name"], "wav": path,
                         "original_wav": benchmark_path(it["wav"]),
                         "original_text": it["original_text"],
                         "edited_text": it["edited_text"],
                         "edit_type": it.get("edit_type", a.edit_type)})
        except Exception as e:  # noqa: BLE001
            k = type(e).__name__
            failed[k] = failed.get(k, 0) + 1
            if len(failed) <= 3 and failed[k] == 1:
                print(f"|   {it['item_name']}: {k}: {str(e)[:120]}", flush=True)
        if i % 50 == 0:
            print(f"|   {i}/{len(items)} synthesised, {sum(failed.values())} failed",
                  flush=True)

    print(f"| synthesised {len(rows)}/{len(items)}; failures {failed}")
    json.dump({"exp_name": a.exp_name, "ckpt_steps": a.ckpt_steps,
               "protocol": a.protocol, "n_ok": len(rows), "failed": failed,
               "arch": arch, "items": rows}, open(a.out, "w"), indent=1)
    print(f"| wrote {a.out}")
    # A partial synthesis must not be scored as if it were complete: the missing
    # items would silently leave the denominator instead of counting as errors.
    return 0 if len(rows) == len(items) else 1


if __name__ == "__main__":
    sys.exit(main())
