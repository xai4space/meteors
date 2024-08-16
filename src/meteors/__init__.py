from .image import Image

from .lime import Lime, ImageAttributes, ImageSpatialAttributes, ImageSpectralAttributes

from . import utils
from . import visualize

__all__ = [
    "Image",
    "Lime",
    "ImageAttributes",
    "ImageSpatialAttributes",
    "ImageSpectralAttributes",
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
