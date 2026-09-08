"""Compute WER / SIM over a `wavepainter.evaluate` result file.

    python -m wavepainter.metrics.score_cli --items out/sub.json \
        --sv_ckpt third_party/wavlm_large_finetune.pth --out out/sub_metrics.json

Scoring is a separate step from synthesis, and deliberately so: the measuring
instruments here -- Whisper for WER, WavLM for speaker similarity -- must not be
reachable from the model being scored.

Signal quality is measured separately, by `scripts/measure_quality.py`, which is
full-reference (PESQ/STOI against the source recording). No no-reference quality
estimator is used: on this benchmark they rank the untouched original recordings
below synthesised output, so they measure recording cleanliness rather than
fidelity to the source.

`scripts/verify_benchmark.sh` runs this for all three edit types and then
`wavepainter.metrics.report` prints the comparison against the baseline.
"""

import argparse
import json
import sys

from .ming import edit_acc_and_noedit, speaker_sim, wer_aggregate


def transcribe(paths, asr_model):
    import scipy.signal
    import soundfile as sf
    import torch
    from transformers import pipeline

    # fp32, matching the harness. run_wer.py does
    #     WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    # with no dtype argument, i.e. full precision. We ran fp16, which is worth
    # ~0.1 WER points of noise in an unpredictable direction -- tolerable when
    # the substitution bar-to-target span was thought to be 0.53, but that span
    # is 0.24 against the `full` column reported here, so fp16
    # rounding alone was ~40% of the distance we are trying to measure.
    asr = pipeline("automatic-speech-recognition", model=asr_model, device=0,
                   torch_dtype=torch.float32)
    # Pinned: left to itself a multilingual Whisper runs language detection per
    # clip, and one mis-detection yields a WER of 1.0 that reads as a model
    # failure rather than an ASR one. Equivalent to the harness's
    # `processor.get_decoder_prompt_ids(language="english", task="transcribe")`
    # forced_decoder_ids.
    #
    # `pipeline` and the harness's bare `model.generate` agree only for clips
    # under 30 s -- past that the harness's WhisperFeatureExtractor hard-
    # truncates while the pipeline switches to long-form sequential decoding.
    # Measured over all 630 protocol items: longest is 8.81 s.
    kw = {"generate_kwargs": {"language": "english", "task": "transcribe"}}
    out = []
    for i, p in enumerate(paths, 1):
        w, sr = sf.read(p, dtype="float64", always_2d=False)
        if w.ndim > 1:
            w = w.mean(1)
        if sr != 16000:
            # `scipy.signal.resample` (Fourier), not `resample_poly`. The harness
            # uses the former, including its truncating length computation; the
            # two differ in their treatment of band edges by more than the WER
            # margin we are claiming.
            w = scipy.signal.resample(w, int(len(w) * 16000 / sr))
        out.append(asr(w.astype("float32"), **kw)["text"])
        if i % 50 == 0:
            print(f"|   transcribed {i}/{len(paths)}", flush=True)
    del asr
    torch.cuda.empty_cache()
    return out


def score(items, sv_ckpt, asr_model):
    hyps = transcribe([it["wav"] for it in items], asr_model)
    w = wer_aggregate([(it["edited_text"], h) for it, h in zip(items, hyps)])

    acc_n, no_ref, no_hyp = 0, [], []
    for it, h in zip(items, hyps):
        a, r, y = edit_acc_and_noedit(it["original_text"], it["edited_text"],
                                      h, it.get("edit_type", "sub"))
        acc_n += a
        no_ref.append(r)
        no_hyp.append(y)
    # Named for the normaliser it uses. `wer_aggregate` -> `normalize_text` is
    # `run_wer.process_one`, which is right for the SCORED WER; the harness
    # computes noedit-WER through `eval_wer.py`'s `characterize` instead, which
    # keeps word-attached punctuation. Whisper-large-v3 ends essentially every
    # utterance with a period while `edited_text` carries none, so the official
    # number reads roughly 12 points higher on this protocol (measured: 1.567 vs
    # 13.283 over 245 real hypotheses). Do not compare this field to a published
    # noedit figure under this name.
    noedit = wer_aggregate(list(zip(no_ref, no_hyp)))

    sim = speaker_sim([(it["wav"], it["original_wav"]) for it in items], sv_ckpt)

    # `values` carries the two comparable columns of Ren et al.'s Table 1 -- WER
    # and SIM -- plus ACC.
    #
    # ACC and noedit-WER are two of the four metrics the benchmark README defines.
    # They are surfaced here rather than buried in `detail`, so a run that never
    # applies its edits cannot look healthy on the scored axes while ACC sits at
    # zero out of sight.
    acc_pct = round(100.0 * acc_n / max(1, len(items)), 3)
    # ACC sits in `values` as a sanity floor, not as a headline metric: a model
    # that edits nothing keeps a free perfect SIM, and ACC is what makes that
    # visible. No published ACC column exists to compare it against.
    return {
        "values": {"wer": w["wer_headline_pct"], "sim": sim["sim"],
                   "acc": acc_pct},
        "reported": {"noedit_wer_processone": noedit["wer_corpus_pct"]},
        "detail": {"n": w["n_items"], "wer_corpus": w["wer_corpus_pct"],
                   "sim_std": sim["sim_std"], "acc": acc_pct},
        "hyps": hyps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, help="eval_ming --out json")
    ap.add_argument("--sv_ckpt", required=True)
    ap.add_argument("--asr_model", default="openai/whisper-large-v3")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    items = json.load(open(a.items))["items"]
    if not items:
        print("evaluator produced no audio", file=sys.stderr)
        return 2
    res = score(items, a.sv_ckpt, a.asr_model)
    json.dump(res, open(a.out, "w"), indent=1)
    v = res["values"]
    print(f"| WER {v['wer']:.3f}%  SIM {v['sim']:.3f}  ACC {v['acc']:.1f}%"
          f"  n={res['detail']['n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
