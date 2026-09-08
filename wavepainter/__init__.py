"""Masked-span speech editing with frozen-HuBERT perceptual supervision.

Edit a word in the transcript and only that span of the spectrogram is
re-predicted; outside it the source mel is carried over unchanged.

Entry points, all driven through ``python -m``:

``wavepainter.train``
    Phase-1 and phase-2 training. Reads ``task_cls`` from the config and starts
    the named task.
``wavepainter.evaluate``
    Scores a checkpoint against the benchmark protocol, or edits a single item.
``wavepainter.metrics.score_cli`` / ``wavepainter.metrics.report``
    Scoring and table generation over an evaluation run's outputs.

The shell wrappers in ``scripts/`` are the documented way in --
``scripts/demo_edit.sh`` for one item, ``scripts/verify_benchmark.sh`` for the
reported table.

``WavePainterTask`` is resolved lazily: importing it eagerly here would pull
torch and the whole model stack into every ``import wavepainter``, including the
ones that only want ``__version__``.
"""

__all__ = ["WavePainterTask", "__version__"]

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("wavepainter")
except (ImportError, PackageNotFoundError):  # not installed, e.g. a source checkout
    __version__ = "unknown"


def __getattr__(name):
    """PEP 562 lazy attribute access, so the heavy import is opt-in."""
    if name == "WavePainterTask":
        from wavepainter.task import WavePainterTask

        return WavePainterTask
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
