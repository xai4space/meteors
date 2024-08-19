from .attributes import ImageAttributes, ImageSpatialAttributes, ImageSpectralAttributes
from .explainer import Explainer


__all__ = [
    "ImageAttributes",
    "ImageSpatialAttributes",
    "ImageSpectralAttributes",
    "Explainer",
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
