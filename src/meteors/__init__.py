from .image import Image

from . import utils
from . import visualize
from . import attr

__all__ = [
    "Image",
    "utils",
    "visualize",
    "attr",
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
