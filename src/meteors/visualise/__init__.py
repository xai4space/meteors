from .lime_visualise import (
    visualize_spectral_attributes_by_waveband,
    visualize_spectral_attributes_by_magnitude,
)
from .lime_visualise import visualize_spectral_attributes, visualize_spatial_attributes


__all__ = [
    "visualize_spectral_attributes_by_waveband",
    "visualize_spectral_attributes_by_magnitude",
    "visualize_spectral_attributes",
    "visualize_spatial_attributes",
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
