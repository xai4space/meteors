from .hyper_image import Image

from .hyper_lime import Lime, ImageAttributes, ImageSpatialAttributes, ImageSpectralAttributes



__all__ = [
    "Image",
    "Lime",
    "ImageAttributes",
    "ImageSpatialAttributes",
    "ImageSpectralAttributes",
    "models",
    "visualise",
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