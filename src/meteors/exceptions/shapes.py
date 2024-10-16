class ShapeMismatchError(TypeError):
    def __init__(self, first_array, second_array, expected_shape, actual_shape, additional_message=None):
        self.first_array = first_array
        self.second_array = second_array
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape
        self.additional_message = additional_message

    def __str__(self):
        return (
            f"Shape mismatch between {self.first_array} and {self.second_array}. Expected shape {self.expected_shape}, got {self.actual_shape}."
            + (f" {self.additional_message}" if self.additional_message else "")
        )


class OrientationError(ValueError):
    def __init__(self, orientation):
        self.orientation = orientation

    def __str__(self):
        return (
            f"Invalid orientation '{self.orientation}'. Orientation must be a tuple of 'H', 'W', and 'C' in any order."
        )
