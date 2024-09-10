from .hsi import HSI

from .lime import Lime, HSIAttributes, HSISpatialAttributes, HSISpectralAttributes

from . import utils
from . import visualize

__all__ = [
    "HSI",
    "Lime",
    "HSIAttributes",
    "HSISpatialAttributes",
    "HSISpectralAttributes",
    "utils",
    "visualize",
]


def __dir__():
    """IPython tab completion seems to respect this."""
    return __all__ + [
        "__all__",
        "__builtins__",
        "__cached__",
        "__doc__",
        "__file__",
        "__loader__",
        "__name__",
        "__package__",
        "__path__",
        "__spec__",
        "__version__",
    ]
