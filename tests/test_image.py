import unittest


import torch

import meteors as mt


class TestImageMethods(unittest.TestCase):
    def test_image(self) -> None:
        sample = torch.tensor([[[0]]])
        wavelengths = [0]
        mt.Image(image=sample, wavelengths=wavelengths, binary_mask="artificial")

    def test_wavelengths(self) -> None:
        sample = torch.tensor([[[0]]])
        with self.assertRaises(
            ValueError,
            msg="Improper length of wavelengths - it should correspond to the number of channels",
        ):
            wavelengths = [0, 1]
            mt.Image(image=sample, wavelengths=wavelengths)

    def test_artificial_mask(self) -> None:
        sample = torch.tensor([[[0]]])
        wavelengths = [0]
        image = mt.Image(
            image=sample, wavelengths=wavelengths, binary_mask="artificial"
        )
        self.assertEqual(
            image.binary_mask,
            torch.tensor([[[0]]]),
            "The simplest mask with no data should be created",
        )

    def test_incorrect_shape_mask(self) -> None:
        sample = torch.tensor([[[0]]])
        wavelengths = [0]
        with self.assertRaises(ValueError):
            binary_mask = torch.tensor([[[0, 0]]])

            mt.Image(image=sample, wavelengths=wavelengths, binary_mask=binary_mask)

        with self.assertRaises(
            ValueError,
            msg="Mask should be a tensor, numpy ndarray or a string 'artificial' which will create an automatic mask",
        ):
            binary_mask = "very bad mask"
            mt.Image(image=sample, wavelengths=wavelengths, binary_mask=binary_mask)

    def test_rgb_image(self) -> None:
        pass


if __name__ == "__main__":
    unittest.main()
