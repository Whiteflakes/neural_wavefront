"""Neural Wavefront: JWST Phase Retrieval using Deep Learning."""

__version__ = "0.1.0"

# Expose main components at package level
from neural_wavefront.optics import pupil, zernike, propagation
from neural_wavefront.utils import config, visualization

__all__ = [
    "__version__",
    "pupil",
    "zernike",
    "propagation",
    "config",
    "visualization",
]
