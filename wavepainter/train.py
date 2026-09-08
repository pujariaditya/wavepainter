"""Training entry point.

    python -m wavepainter.train --config configs/phase2_hubert_dn_a085.yaml \\
        --exp_name my_run --reset \\
        --hparams "load_ckpt=checkpoints/phase1-base/model_ckpt_steps_5000.ckpt"

Pins exactly one GPU, then hands off to the task named by ``task_cls`` --
``wavepainter.task.WavePainterTask`` for both phases.

EXACTLY ONE DEVICE, never "0,1", and the pin must happen before anything imports
torch. Two reasons, both silent if you get them wrong:

  * ``wavepainter/tasks/speech_base.py`` scales ``max_tokens`` and ``max_sentences`` by
    ``device_count()``. With two cards visible, the configured batch of 16
    becomes 32 and the run is no longer the one the config describes.
  * the multi-GPU path this stack has never been run on. It has since been
    removed outright, so a second device would now fail rather than diverge.

``CUDA_VISIBLE_DEVICES`` is read once, at CUDA init, so setting it after torch
loads is a no-op and every job piles onto card 0. Hence the ordering here.
"""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")


def _pin_single_device():
    """Ensure exactly one visible CUDA device, before torch is imported."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        return
    devices = [d for d in visible.split(",") if d.strip() != ""]
    if len(devices) > 1:
        raise SystemExit(
            f"CUDA_VISIBLE_DEVICES={visible!r} exposes {len(devices)} devices. "
            "This stack requires exactly one: with more than one, the batch "
            "size is silently multiplied by the device count and the trainer "
            "switches to an untested DDP path. Set a single device id."
        )


def _repo_root_on_path():
    """Return the repository root, and make sure it is importable.

    The insertion is belt and braces since 329a06a folded the four top-level
    packages into ``wavepainter``: reaching this module at all means
    ``wavepainter`` already imported, so the root is already reachable. It costs
    nothing and keeps ``task_cls``, which names its class absolutely, resolvable
    if this is ever driven from somewhere other than ``python -m``.

    The return value is the point of the call -- ``_resolve_data_root`` rewrites
    the config's relative data dirs against it.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def _resolve_data_root(root):
    """Rewrite the config's relative data dirs against ``--data-root``.

    Appends to ``--hparams`` rather than editing the config, so the committed
    config stays portable and the override is visible in the run's own record.
    """
    data_root = None
    for i, arg in enumerate(sys.argv):
        if arg == "--data-root" and i + 1 < len(sys.argv):
            data_root = sys.argv[i + 1]
            del sys.argv[i : i + 2]
            break
        if arg.startswith("--data-root="):
            data_root = arg.split("=", 1)[1]
            del sys.argv[i]
            break
    if data_root is None:
        data_root = os.environ.get("WAVEPAINTER_DATA")
    if data_root is None:
        return

    data_root = os.path.abspath(data_root)
    override = (
        f"processed_data_dir={data_root}/processed/libritts,"
        f"binary_data_dir={data_root}/binary/libritts_bigvgan"
    )
    for i, arg in enumerate(sys.argv):
        if arg == "--hparams" and i + 1 < len(sys.argv):
            sys.argv[i + 1] = f"{sys.argv[i + 1]},{override}".strip(",")
            return
    sys.argv += ["--hparams", override]


def _take_flag(name):
    """Remove a bare flag from argv and report whether it was present.

    set_hparams parses argv itself and rejects anything it does not know, so
    wrapper flags have to be consumed before it runs.
    """
    if name in sys.argv:
        sys.argv.remove(name)
        return True
    return False


def main():
    dry_run = _take_flag("--dry-run")

    _pin_single_device()
    root = _repo_root_on_path()
    _resolve_data_root(root)

    import importlib

    from wavepainter.runtime.hparams import hparams, set_hparams

    # This is the check itself: `--hparams` can only OVERRIDE keys that already
    # exist, so set_hparams raises KeyError on a typo'd or removed key. Running
    # it is what validates a stage's overrides -- before any data is touched and
    # before a GPU is claimed.
    set_hparams()
    if not hparams.get("task_cls"):
        raise SystemExit("config sets no task_cls")

    # A checkpoint is written only when `global_step % val_check_interval == 0`,
    # so a run whose interval exceeds its update budget finishes having saved
    # nothing and the next stage fails on a missing file. Cheap to check here,
    # expensive to discover an hour in.
    interval = int(hparams.get("val_check_interval") or 0)
    updates = int(hparams.get("max_updates") or 0)
    if interval and updates and interval > updates:
        raise SystemExit(
            f"val_check_interval={interval} exceeds max_updates={updates}: this "
            f"run would write no checkpoint. Set val_check_interval <= "
            f"max_updates (the released phase-2 recipe uses 1500/1500)."
        )

    if dry_run:
        print(f"| dry run OK | {len(hparams)} hparams resolved "
              f"| max_updates={updates} val_check_interval={interval} "
              f"| task_cls={hparams['task_cls']}")
        return

    task_cls = _import_cls(hparams["task_cls"])
    task_cls.start()


# The published checkpoints each ship a config.yaml snapshot, and set_hparams
# lets that snapshot win over the repo config. Those files name the module
# layout as it was before the packages were folded into wavepainter/ -- e.g.
# `tasks.speech_editing.spec_denoiser.SpeechDenoiserTask`. They are on the hub,
# sha256-pinned and immutable, so the rename is absorbed here instead: warm
# starting phase two from the released phase-1 base is the documented reproduce
# path, and it reads exactly such a snapshot.
_MOVED = tuple(sorted((
    ("tasks.speech_editing.",           "wavepainter.tasks."),
    ("tasks.tts.vocoder_infer.",        "wavepainter.tasks.vocoder_infer."),
    ("tasks.tts.",                      "wavepainter.tasks."),
    ("modules.commons.",                "wavepainter.models."),
    ("modules.tts.commons.",            "wavepainter.models."),
    ("modules.speech_editing.commons.", "wavepainter.models."),
    ("modules.speech_editing.",         "wavepainter.models."),
    ("modules.",                        "wavepainter.models."),
    ("data_gen.tts.",                   "wavepainter.datasets."),
    ("utils.commons.",                  "wavepainter.runtime."),
    ("utils.metrics.ssim",              "wavepainter.losses.ssim"),
    ("utils.spec_aug.time_mask",        "wavepainter.datasets.time_mask"),
), key=lambda kv: -len(kv[0])))


def _import_cls(path):
    import importlib

    for old, new in _MOVED:
        if path.startswith(old):
            path = new + path[len(old):]
            break
    module_name, _, class_name = path.rpartition(".")
    return getattr(importlib.import_module(module_name), class_name)


if __name__ == "__main__":
    main()
