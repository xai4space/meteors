class MaskCreationError(Exception):
    """Exception raised in case there a problem with creating a mask."""

    def __init__(self, message):
        self.message = message


class BandSelectionError(Exception):
    """Exception raised in case there a problem with selecting bands from spyndex library."""

    def __init__(self, message):
        self.message = message


class HSIAttributesError(Exception):
    """Exception raised in case there a problem with a HSIAttributes object."""

    def __init__(self, message):
        self.message = message


class ShapeMismatchError(ValueError):
    """Exception raised for errors in shape mismatch between two tensors."""

    def __init__(self, message):
        self.message = message
