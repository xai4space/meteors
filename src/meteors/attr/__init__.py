from .attributes import ImageAttributes, ImageLimeSpatialAttributes, ImageLimeSpectralAttributes
from .explainer import Explainer

from .lime import Lime
from .integrated_gradients import IntegratedGradients
from .saliency import Saliency


__all__ = [
    "ImageAttributes",
    "ImageLimeSpatialAttributes",
    "ImageLimeSpectralAttributes",
    "Explainer",
    "IntegratedGradients",
    "Lime",
    "Saliency",
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
