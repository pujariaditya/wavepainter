# Measured results

Every number here was measured in this repository against the frozen protocol,
with the command that produced it. Nothing is quoted from an earlier record.

The scorer is deterministic and the diffusion sampler is seeded per item, so
these reproduce to the precision printed rather than approximately. Where a
number is an estimate or rests on a subset, that is said.

**Baseline** throughout is Ren et al., *"Edit Content, Preserve Acoustics"*
([arXiv:2602.00560](https://arxiv.org/abs/2602.00560)), Table 1, English, the
`full` column, row "Ours (w. GRPO)". Their columns are `basic | full`, two
benchmark subsets — quoting `basic` instead understates the substitution and
insertion baselines by 0.28 and 0.47 WER.

---

## 1. Main comparison

Protocol: 256 substitution, 199 insertion, 200 deletion items.

```bash
./scripts/verify_benchmark.sh              # ~50 min on one RTX 8000
```

| edit | system | WER ↓ | SIM ↑ | ACC | PESQ ↑ |
|---|---|---|---|---|---|
| sub | Ren et al. | 4.41 | 0.78 | — | — |
| | **wavepainter** (released) | **3.479** | **0.943** | 66.41 | 4.090 |
| | ours, α=0.85 | 3.063 | 0.943 | 67.19 | 4.089 |
| | ours, α=0.70 | 3.381 | 0.943 | — | — |
| ins | Ren et al. | 4.97 | 0.82 | — | — |
| | **wavepainter** (released) | 4.400 | **0.961** | 70.35 | 4.083 |
| | ours, α=0.85 | 4.128 | **0.961** | 73.87 | 4.083 |
| | **ours, α=0.70** | **3.817** | 0.960 | — | — |
| del | **Ren et al.** | **6.88** | 0.78 | — | — |
| | **wavepainter** (released) | 9.697 | **0.918** | 94.00 | 4.079 |
| | ours, α=0.85 | 9.136 | **0.918** | 94.50 | 4.076 |
| | ours, α=0.70 | 9.525 | 0.918 | — | — |

Five of the six compared numbers improve on theirs: substitution and insertion
WER, and speaker similarity on all three edit types. WER and SIM are the two
axes on which a like-for-like comparison exists; PESQ has no published
counterpart here and is reported against this system's own vocoder ceiling
(§5), not against prior work.

### The other systems Ren et al. re-ran

Their Table I also carries FluentSpeech and VoiceCraft, re-run on this split
under the same forced-alignment masks — so those rows are comparable with ours
and are quoted here. They are quoted, not measured by us.

**How comparable, exactly.** Read from Ren et al. §3.1 verbatim:

> Ground-truth editing intervals are obtained via forced alignment using
> WhisperX [28].

> We report both objective and subjective metrics: WER (Whisper [31]) for
> intelligibility and boundary consistency; SIM (WavLM [32]) for speaker
> preservation; DNSMOS [33] for perceptual quality […]

So both sides localise edits by forced alignment, score WER with Whisper and
score similarity with WavLM embeddings, on an item set that is identical and
fingerprint-verified. What is *not* pinned on their side is the Whisper variant
and the similarity checkpoint — ours are Whisper large-v3 in fp32 and the WavLM
large speaker encoder `setup.sh` fetches. That is why these figures are quoted
rather than presented as scorer-identical; the harness that produced our column
ships, so anyone can rescore.

| | FluentSpeech | VoiceCraft | Ren et al. | **wavepainter** |
|---|---|---|---|---|
| sub WER ↓ | 4.65 | 12.73 | 4.41 | **3.479** |
| ins WER ↓ | 11.91 | 12.94 | 4.97 | **4.400** |
| del WER ↓ | 8.78 | 17.88 | **6.88** | 9.697 |
| sub SIM ↑ | 0.51 | 0.59 | 0.78 | **0.943** |
| ins SIM ↑ | 0.60 | 0.67 | 0.82 | **0.961** |
| del SIM ↑ | 0.52 | 0.62 | 0.78 | **0.918** |

On deletion WER we are behind **two** systems, FluentSpeech as well as Ren et
al. On speaker similarity we are ahead of all three by at least 0.12.

**Ming-UniAudio is not in this table.** It created the benchmark, and Ren et al.
mark its rows as *cited* — quoted from Ming-UniAudio's own free-form evaluation
rather than re-run under forced alignment. Its protocol supplies no timestamp,
so a system must find the edit span as well as realise it. Those figures answer
a harder question and are not interchangeable with the rows above; the caveat
applies to that system alone.

## 2. Phase 1 → phase 2

```bash
BASE=<phase1-base> ./scripts/train_phase2.sh   # ~1 GPU-hour
```

| | sub | ins | del |
|---|---|---|---|
| phase 1 only | 4.019 | 4.465 | 10.248 |
| + phase 2 | **3.479** | **4.400** | **9.697** |
| Δ | −0.540 | −0.065 | −0.551 |

SIM is unchanged on substitution and deletion and 0.001 higher on insertion (0.943 / 0.960→0.961 / 0.918). Phase 2
improves every edit type, so the gain is not a trade between them.

**Read this table against §1, not instead of it.** Phase 1 alone already beats
Ren et al. on substitution (4.019 vs 4.41), on insertion (4.465 vs 4.97) and on
all three similarities. Every comparison in §1 is therefore settled before phase
2 runs, and what phase 2 contributes is the Δ row above. It is a real gain on
every edit type for about 1 GPU-hour; it is not where the margin over the
baselines comes from.

**What this isolates, and what it does not.** The row is the phase-2 *package* —
the frozen-HuBERT feature loss and 1500 denoiser-only updates — measured against
phase 1. It establishes that the package helps and by how much. It does **not**
separate the feature loss from the extra updates: no matched λ_feat=0 control was
run at the same step count and seed. Neither of the two ingredients is isolated.
(Interpolation was a third ingredient once; §3 measures it, and it is not in the
released model.) Attributing the gain to perceptual supervision specifically is
motivated, not verified, and the paper says so.

## 3. Interpolation coefficient

```bash
python -m wavepainter.training.interpolate --base <base> --child <child> \
    --alpha <A> --prefixes denoise_fn. --out checkpoints/alpha_<A>
EXP=alpha_<A> OUT=out/sweep/alpha_<A> ./scripts/verify_benchmark.sh
```

WER; SIM was scored at α=0.70 and α=0.85 only, and differs by 0.001 between
them (0.943 / 0.960–0.961 / 0.918); it is not measured at the intermediate α.
We expect little movement because α
moves only the 170 denoiser tensors and speaker identity lives in the frozen
trunk and heads.

| α | sub | ins | del |
|---|---|---|---|
| 0.70 | 3.381 | **3.817** | 9.525 |
| 0.75 | 3.301 | 3.976 | 9.512 |
| **0.85** | **3.063** | 4.128 | **9.136** |
| 0.90 | 3.327 | 4.053 | 9.557 |
| 1.00 | 3.479 | 4.400 | 9.697 |

- α=0.85 is an interior optimum for **substitution and deletion only**.
  Insertion is best at 0.70 and pays 0.311 WER at 0.85. An earlier claim that
  0.85 was optimal on every leg was inherited from a lost record and is false.
- **α=1.00 — the trained child with no interpolation — is the worst setting on
  every leg.** It is nonetheless what ships: the coefficient was chosen by
  reading this table, which is the test set, so the interpolated points are
  tuned on the numbers they are then reported against. The child is the model
  phase 2 produced, and that is the one the paper reports.
- Adjacent points differ by as little as 0.013 WER (0.70 vs 0.75 on deletion),
  which is within single-seed variation. Only the 0.85 minimum and the 1.00
  maximum are separated by margins worth trusting.

## 4. Deletion reference

The deletion protocol's edits performed on the **source waveform**: the deleted
words are cut at their forced-alignment boundaries and the join crossfaded. No
model, no vocoder, no predicted durations.

```bash
python scripts/build_deletion_reference.py --out out/reference/gt_splice
python -m wavepainter.metrics.score_cli --items out/reference/gt_splice/items.json \
    --sv_ckpt <wavlm_large_finetune.pth> --out out/reference/gt_metrics.json
```

| | WER | SIM | ACC |
|---|---|---|---|
| ground-truth waveform splice | 7.589 | 0.960 | 91.00 |
| wavepainter (released) | 9.697 | 0.918 | 94.00 |
| Ren et al. target | 6.880 | 0.780 | — |

**Most of the deletion gap is modelling headroom**: 2.11 of the 2.82 points lie
between our model and a perfect cut, and 0.71 separates a perfect cut from the
published target.

*Caveat:* the splice's own edit accuracy is 91.0 %, below the model's 94.0 %. A
hard cut at a word boundary removes coarticulation and is not always heard as
the intended edit, so 7.589 is an approximate bound, not an exact one.

## 5. Full-reference quality

```bash
export WAVEPAINTER_BENCHMARK=<benchmark root holding wavs/>
python scripts/measure_quality.py --results out/child_verify \
    --copysyn <copysyn_bigvgan> --out out/child_verify/quality.json
```

`verify_benchmark.sh` runs this too, and skips it with a message if the optional
`[quality]` extra is not installed. `--copysyn` resolves each synthesised file
against `WAVEPAINTER_BENCHMARK`; if none resolve it exits rather than reporting
no ceiling.

| | PESQ ↑ | STOI ↑ | n |
|---|---|---|---|
| PESQ maximum (identical signals) | 4.644 | 1.000 | — |
| vocoder ceiling: copy-synthesis, no edit | **4.351** | **0.995** | 256 |
| preservation, substitution | 4.090 | 0.9711 | 256 |
| preservation, insertion | 4.083 | 0.9619 | 199 |
| preservation, deletion | 4.079 | 0.9667 | 200 |

"Preservation" compares the original against the system output over the
**unedited** audio only — before and after the edit. It measures the property
that distinguishes masked-span editing from full resynthesis.

The vocoder costs ~0.29 PESQ from the theoretical maximum; editing costs a
further ~0.27. All three edit types land within 0.011 PESQ of each other, which
is what you expect if the cost is dominated by re-vocoding rather than by the
edit.

## 6. Reference points

| | value | what it is |
|---|---|---|
| WER floor | 1.705 | ground-truth recordings scored against their own transcript |
| SIM, copy-synthesis | 0.978 | ceiling for a system that decodes through the vocoder |
| PESQ, copy-synthesis | 4.351 | same, on the full-reference axis (max is 4.644) |
| null WER: sub / ins / del | 15.048 / 10.874 / 27.974 | unedited audio scored against the *edited* transcript |

Quality is measured full-reference and is **not** compared against prior work.
The no-reference estimator the baseline reports was dropped rather than matched:
on this benchmark it scored the original recordings 3.060 and copy-synthesis
3.068, both below the 3.09-3.18 published for edited output, and rated a
ground-truth waveform splice 2.995 - lowest of all. It measures recording
cleanliness, not fidelity to the source.

The null-WER row explains why deletion looks worse than the other edit types on
WER: deleting a word leaves an ungrammatical reference, and scoring untouched
audio against it gives 27.97 for deletion against 15.05 for substitution.

## 7. Deletion error composition

Released model, 200 deletion items:

- 179 word errors in total
- **76 items are entirely error-free**
- **12 items leave a deleted word in the transcript** — which is what the
  94.0 % edit accuracy implies (200 items, 188 correct)

So the deletion gap is mostly *not* failure to perform the edit.

> **Corrected twice.** This row first read **13** word-survival failures, which
> nothing produced; recomputing with the repository's own scorer gave **11**.
> That recomputation ran against `out/verify/del_metrics.json` — the α=0.85
> checkpoint, which the paper no longer reports. All three figures here are now
> recomputed from `out/child_verify/del_metrics.json`, the released model, and
> all three moved: 171→179 errors, 81→76 error-free, 11→12 survivals, against
> 94.0 % rather than 94.5 % accuracy. The lesson worth carrying: a derived count
> belongs to a *checkpoint*, not to a claim. Changing which model ships
> invalidates every hand-computed figure downstream of it, and no test catches
> that, because each stale number is still a true statement about some run that
> was really performed.
