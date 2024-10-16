class MaskCreationError(Exception):
    """Exception raised in case there a problem with creating a mask."""

    def __init__(self, message):
        self.message = message


class InvalidSegmentError(Exception):
    """Exception raised in case there a problem with the segment selected for the band mask"""

    def __init__(self, message):
        self.message = message


class BandSelectionError(InvalidSegmentError):
    """Exception raised in case there a problem with selecting bands from spyndex library."""

    def __init__(self, message):
        self.message = message


class ExplainerInitializationError(Exception):
    """Exception raised in case there a problem with initializing the explainer."""

    def __init__(self, message):
        self.message = message


class ExplanationError(Exception):
    """Exception raised in case there a problem with explaining the model."""

    def __init__(self, message):
        self.message = message


class HSIError(Exception):
    """Exception raised in case there a problem with the HSI object."""

    def __init__(self, message):
        self.message = message


class HSIAttributesError(Exception):
    """Exception raised in case there a problem with a HSIAttributes object."""

    def __init__(self, message):
        self.message = message


class ShapeMismatchError(Exception):
    """Exception raised for errors in shape mismatch between two tensors."""

    def __init__(self, message):
        self.message = message


class OrientationError(Exception):
    """Exception raised in case the orientation passed is invalid."""

    def __init__(self, orientation):
        self.orientation = orientation

    def __str__(self):
        return (
            f"Invalid orientation '{self.orientation}'. Orientation must be a tuple of 'H', 'W', and 'C' in any order."
        )
