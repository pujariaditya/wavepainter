"""Spectrogram figures for the training log.

Diagnostic only -- nothing here affects a reported number. The figure is written
to TensorBoard during validation so a run can be watched without decoding audio.
"""

import matplotlib

# Must precede the pyplot import: training runs headless, and the default
# interactive backend fails without a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

# Distinct colours for overlaid F0 contours, ordered for contrast on a dark
# spectrogram rather than by any convention.
LINE_COLORS = ["w", "r", "orange", "k", "cyan", "m", "b", "lime", "g", "brown", "navy"]


def _numpy(x):
    return x.cpu().numpy() if isinstance(x, torch.Tensor) else x


def _draw_boundaries(cumulative, labels, baseline, height, colour):
    """Vertical token boundaries with staggered text labels.

    Labels are stepped through eight vertical offsets so adjacent short tokens
    do not overprint each other.
    """
    for i, edge in enumerate(cumulative):
        plt.text(edge, (i % 8 + 1) * 4 + baseline, labels[i])
        plt.vlines(edge, baseline, height, colors=colour)


def spec_to_figure(spec, vmin=None, vmax=None, title="", f0s=None, dur_info=None):
    """Render a mel spectrogram, optionally with durations and F0 overlaid.

    ``dur_info`` may carry ``dur_gt`` and ``dur_pred``; ground truth is drawn in
    blue along the lower half and predictions in red along the upper half, so
    the two alignments can be compared at a glance.
    """
    spec = _numpy(spec)
    half = spec.shape[1] // 2

    fig = plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.pcolor(spec.T, vmin=vmin, vmax=vmax)

    if dur_info is not None:
        labels = dur_info["txt"]
        gt = np.cumsum(_numpy(dur_info["dur_gt"])).astype(int)
        _draw_boundaries(gt, labels, 0, half // 2, "b")
        right = gt[-1]
        if "dur_pred" in dur_info:
            pred = np.cumsum(_numpy(dur_info["dur_pred"])).astype(int)
            _draw_boundaries(pred, labels, half, half * 1.5, "r")
            right = max(right, pred[-1])
        plt.xlim(0, right)

    if f0s is not None:
        if not isinstance(f0s, dict):
            f0s = {"f0": f0s}
        axis = plt.gca().twinx()
        for i, (name, f0) in enumerate(f0s.items()):
            axis.plot(_numpy(f0), label=name, c=LINE_COLORS[i % len(LINE_COLORS)],
                      linewidth=1, alpha=0.5)
        axis.set_ylim(0, 1000)
        axis.legend()

    return fig
