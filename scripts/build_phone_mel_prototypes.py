#!/usr/bin/env python3
"""Build a fixed phone-discriminative teacher from forced-aligned training mels.

The output contains no model weights: it is a corpus statistic with one mean
100-bin mel vector per phone token plus global feature normalisation.  Phone
labels are recovered by mapping each 1-indexed mel2ph position through the
utterance's ph_token sequence.
"""

import argparse
import json
import os
import sys

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wavepainter.runtime.indexed_datasets import IndexedDataset


def iter_labeled_frames(dataset, limit=0):
    total = len(dataset) if limit <= 0 else min(len(dataset), limit)
    for index in range(total):
        item = dataset[index]
        mel = np.asarray(item["mel"], dtype=np.float64)
        mel2ph = np.asarray(item["mel2ph"], dtype=np.int64)[: len(mel)]
        phones = np.asarray(item["ph_token"], dtype=np.int64)
        valid = (mel2ph > 0) & (mel2ph <= len(phones))
        if not np.any(valid):
            continue
        yield mel[valid], phones[mel2ph[valid] - 1]


def accumulate(dataset, vocab_size, mel_bins, limit=0):
    counts = np.zeros(vocab_size, dtype=np.int64)
    sums = np.zeros((vocab_size, mel_bins), dtype=np.float64)
    global_count = 0
    global_sum = np.zeros(mel_bins, dtype=np.float64)
    global_sumsq = np.zeros(mel_bins, dtype=np.float64)

    for mel, labels in iter_labeled_frames(dataset, limit=limit):
        keep = (labels > 0) & (labels < vocab_size)
        mel, labels = mel[keep], labels[keep]
        if not len(labels):
            continue
        counts += np.bincount(labels, minlength=vocab_size)
        for phone in np.unique(labels):
            sums[phone] += mel[labels == phone].sum(axis=0)
        global_count += len(labels)
        global_sum += mel.sum(axis=0)
        global_sumsq += np.square(mel).sum(axis=0)

    mean = global_sum / max(global_count, 1)
    variance = global_sumsq / max(global_count, 1) - np.square(mean)
    std = np.sqrt(np.maximum(variance, 1e-6))
    prototypes = sums / np.maximum(counts[:, None], 1)
    return counts, prototypes, mean, std


def accuracy(dataset, prototypes, mean, std, valid_classes, limit=0):
    proto = (prototypes - mean[None]) / std[None]
    proto /= np.maximum(np.linalg.norm(proto, axis=1, keepdims=True), 1e-8)
    proto[~valid_classes] = 0.0
    n_correct = 0
    n_total = 0
    per_class_correct = np.zeros(len(prototypes), dtype=np.int64)
    per_class_total = np.zeros(len(prototypes), dtype=np.int64)
    invalid_logit = -1e9

    for mel, labels in iter_labeled_frames(dataset, limit=limit):
        keep = ((labels > 0) & (labels < len(prototypes))
                & valid_classes[np.clip(labels, 0, len(prototypes) - 1)])
        mel, labels = mel[keep], labels[keep]
        if not len(labels):
            continue
        x = (mel - mean[None]) / std[None]
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
        logits = x @ proto.T
        logits[:, ~valid_classes] = invalid_logit
        pred = logits.argmax(axis=1)
        correct = pred == labels
        n_correct += int(correct.sum())
        n_total += len(labels)
        per_class_total += np.bincount(labels, minlength=len(prototypes))
        per_class_correct += np.bincount(
            labels, weights=correct.astype(np.int64),
            minlength=len(prototypes)).astype(np.int64)

    seen = per_class_total > 0
    macro = np.mean(
        per_class_correct[seen] / np.maximum(per_class_total[seen], 1))
    return n_correct / max(n_total, 1), float(macro), n_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("WAVEPAINTER_DATA_BINARY",
                               "data/binary/libritts_bigvgan"),
    )
    parser.add_argument(
        "--output", default="assets/phone_mel_prototypes.npz")
    parser.add_argument("--max-train-items", type=int, default=0)
    parser.add_argument("--max-eval-items", type=int, default=2000)
    parser.add_argument("--min-count", type=int, default=100)
    args = parser.parse_args()

    phone_set = json.load(open(os.path.join(args.data_dir, "phone_set.json")))
    vocab_size = len(phone_set) + 2  # padding plus the separator token at id 79
    train = IndexedDataset(os.path.join(args.data_dir, "train"))
    first = train[0]
    mel_bins = np.asarray(first["mel"]).shape[-1]
    counts, prototypes, mean, std = accumulate(
        train, vocab_size, mel_bins, limit=args.max_train_items)
    valid_classes = counts >= args.min_count
    valid_classes[0] = False

    train_micro, train_macro, train_frames = accuracy(
        train, prototypes, mean, std, valid_classes,
        limit=args.max_eval_items)
    valid = IndexedDataset(os.path.join(args.data_dir, "valid"))
    valid_micro, valid_macro, valid_frames = accuracy(
        valid, prototypes, mean, std, valid_classes,
        limit=args.max_eval_items)

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    np.savez_compressed(
        output,
        prototypes=prototypes.astype(np.float32),
        feature_mean=mean.astype(np.float32),
        feature_std=std.astype(np.float32),
        counts=counts,
        valid_classes=valid_classes,
        phone_set=np.asarray(["<PAD>"] + phone_set + ["<EXTRA>"]),
    )
    print(f"| wrote {output}")
    print(
        f"| classes {int(valid_classes.sum())}/{vocab_size} "
        f"| counts min/median/max "
        f"{int(counts[valid_classes].min())}/"
        f"{int(np.median(counts[valid_classes]))}/"
        f"{int(counts[valid_classes].max())}"
    )
    print(
        f"| train prototype accuracy micro={train_micro:.4f} "
        f"macro={train_macro:.4f} n={train_frames}"
    )
    print(
        f"| valid prototype accuracy micro={valid_micro:.4f} "
        f"macro={valid_macro:.4f} n={valid_frames}"
    )


if __name__ == "__main__":
    main()
