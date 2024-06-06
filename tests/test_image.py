import unittest

import pandas as pd
import numpy as np

import torch

import meteors as mt

class TestImageMethods(unittest.TestCase):
    
    def test_image(seld):
        sample = torch.tensor([[[0]]])
        wavelengths = [0]
        image = mt.Image(image=sample, wavelengths=wavelengths, binary_mask="artificial")
        
    
    def test_wavelengths(self):
        sample = torch.tensor([[[0]]])
        with self.assertRaises(ValueError):
                wavelengths = [0, 1]

                image = mt.Image(image=sample, wavelengths=wavelengths)
                    
    
    
    def test_artificial_mask(self):
        sample = torch.tensor([[[0]]])
        wavelengths = [0]
        image = mt.Image(image=sample, wavelengths=wavelengths, binary_mask="artificial")
        self.assertEqual(image.binary_mask, torch.tensor([[[0]]]))
        
    def test_incorrect_shape_mask(self):
        sample = torch.tensor([[[0]]])
        wavelengths = [0]
        with self.assertRaises(ValueError):
            binary_mask = torch.tensor([[[0, 0]]])

            image = mt.Image(image=sample, wavelengths=wavelengths, binary_mask=binary_mask)
        
        with self.assertRaises(ValueError):
            binary_mask = "very bad mask"
            image = mt.Image(image=sample, wavelengths=wavelengths, binary_mask=binary_mask)
            

    def test_rgb_image(self):
        
        pass        
        
        
        

if __name__ == "__main__":
    unittest.main()