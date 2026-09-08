# Importing a vocoder module registers it under its `vocoder` config name.
# Only BigVGAN is registered: the HiFi-GAN path is unused here (`vocoder:
# BigVGAN`), its checkpoint has no recorded provenance, and a model trained on
# BigVGAN's 24 kHz 100-band mel cannot be decoded by it anyway.
from . import bigvgan  # noqa: F401
