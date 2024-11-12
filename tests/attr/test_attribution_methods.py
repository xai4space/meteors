import pytest

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from meteors.models import ExplainableModel
from meteors.utils import agg_segmentation_postprocessing
from meteors.attr import (
    IntegratedGradients,
    Saliency,
    InputXGradient,
    Occlusion,
    HSISpatialAttributes,
    HSISpectralAttributes,
)
from meteors import HSI


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sum(F.relu(input), dim=(1, 2, 3))


class SegmentationToyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        adjusted_input = F.relu(input)
        return torch.round(adjusted_input)


@pytest.fixture
def explainable_toy_model():
    return ExplainableModel(problem_type="regression", forward_func=ToyModel())


def explainable_segmentation_toy_model(postprocessing_segmentation_output):
    return ExplainableModel(
        problem_type="segmentation",
        forward_func=SegmentationToyModel(),
        postprocessing_output=postprocessing_segmentation_output,
    )


def test_integrated_gradients(explainable_toy_model):
    ig = IntegratedGradients(explainable_toy_model)

    assert ig is not None
    assert ig._attribution_method is not None

    image = HSI(image=torch.rand(3, 224, 224), wavelengths=[0, 100, 200])
    attributions = ig.attribute(image)
    assert attributions.attributes.shape == image.image.shape

    assert ig.has_convergence_delta()

    attributions = ig.attribute(image, return_convergence_delta=True)
    assert attributions.score is not None

    # Test multiple images
    attributions = ig.attribute([image, image], return_convergence_delta=True)
    assert len(attributions) == 2
    assert attributions[0].score is not None
    assert attributions[0].attributes.shape == image.image.shape
    assert attributions[1].score is not None
    assert attributions[1].attributes.shape == image.image.shape

    # Test step size
    start_time = time.time()
    ig.attribute(image, return_convergence_delta=True, n_steps=100)
    longer_time = time.time() - start_time
    start_time = time.time()
    ig.attribute(image, return_convergence_delta=True, n_steps=1)
    shorter_time = time.time() - start_time
    assert longer_time > shorter_time

    # Test segmentation postprocessing
    postprocessing_segmentation_output = agg_segmentation_postprocessing(classes_numb=3)
    explainable_segmentation_model = explainable_segmentation_toy_model(postprocessing_segmentation_output)
    ig = IntegratedGradients(explainable_segmentation_model)

    image = HSI(image=torch.rand(3, 224, 224), wavelengths=[0, 100, 200])
    attributions = ig.attribute(image, target=0)
    assert attributions.attributes.shape == image.image.shape

    ig = IntegratedGradients(explainable_segmentation_model)
    similar_attributions = ig.attribute(image, target=0)
    assert torch.allclose(attributions.attributes, similar_attributions.attributes)

    attributions = ig.attribute([image, image], target=0)
    assert len(attributions) == 2
    assert attributions[0].attributes.shape == image.image.shape
    assert attributions[1].attributes.shape == image.image.shape

    attributions = ig.attribute([image, image], target=[0, 0])
    assert len(attributions) == 2
    assert attributions[0].attributes.shape == image.image.shape
    assert attributions[1].attributes.shape == image.image.shape

    with pytest.raises(RuntimeError):
        ig._attribution_method = None
        ig.attribute(image)


def test_saliency(explainable_toy_model):
    saliency = Saliency(explainable_toy_model)

    assert saliency is not None

    image = HSI(image=torch.rand(3, 224, 224), wavelengths=[0, 100, 200])
    attributions = saliency.attribute(image)
    assert attributions.attributes.shape == image.image.shape

    assert not saliency.has_convergence_delta()
    assert not saliency.multiplies_by_inputs

    # Test multiple images
    attributions = saliency.attribute([image, image])
    assert len(attributions) == 2
    assert attributions[0].attributes.shape == image.image.shape
    assert attributions[1].attributes.shape == image.image.shape

    # Test segmentation postprocessing
    postprocessing_segmentation_output = agg_segmentation_postprocessing(classes_numb=3)
    explainable_segmentation_model = explainable_segmentation_toy_model(postprocessing_segmentation_output)
    saliency = Saliency(explainable_segmentation_model)

    image = HSI(image=torch.rand(3, 224, 224), wavelengths=[0, 100, 200])
    attributions = saliency.attribute(image, target=0)
    assert attributions.attributes.shape == image.image.shape

    saliency = Saliency(explainable_segmentation_model)
    similar_attributions = saliency.attribute(image, target=0)
    assert torch.allclose(attributions.attributes, similar_attributions.attributes)

    attributions = saliency.attribute([image, image], target=0)
    assert len(attributions) == 2
    assert attributions[0].attributes.shape == image.image.shape
    assert attributions[1].attributes.shape == image.image.shape

    attributions = saliency.attribute([image, image], target=[0, 0])
    assert len(attributions) == 2
    assert attributions[0].attributes.shape == image.image.shape
    assert attributions[1].attributes.shape == image.image.shape

    with pytest.raises(RuntimeError):
        saliency._attribution_method = None
        saliency.attribute(image)


def test_input_x_gradient(explainable_toy_model):
    input_x_gradient = InputXGradient(explainable_toy_model)

    assert input_x_gradient is not None

    image = HSI(image=torch.rand(3, 224, 224), wavelengths=[0, 100, 200])
    attributions = input_x_gradient.attribute(image)
    assert attributions.attributes.shape == image.image.shape

    assert not input_x_gradient.has_convergence_delta()

    # Test multiple images
    attributions = input_x_gradient.attribute([image, image])
    assert len(attributions) == 2
    assert attributions[0].attributes.shape == image.image.shape
    assert attributions[1].attributes.shape == image.image.shape

    # Test segmentation postprocessing
    postprocessing_segmentation_output = agg_segmentation_postprocessing(classes_numb=3)
    explainable_segmentation_model = explainable_segmentation_toy_model(postprocessing_segmentation_output)
    input_x_gradient = InputXGradient(explainable_segmentation_model)

    image = HSI(image=torch.rand(3, 224, 224), wavelengths=[0, 100, 200])
    attributions = input_x_gradient.attribute(image, target=0)
    assert attributions.attributes.shape == image.image.shape

    input_x_gradient = InputXGradient(explainable_segmentation_model)
    similar_attributions = input_x_gradient.attribute(image, target=0)
    assert torch.allclose(attributions.attributes, similar_attributions.attributes)

    attributions = input_x_gradient.attribute([image, image], target=0)
    assert len(attributions) == 2
    assert attributions[0].attributes.shape == image.image.shape
    assert attributions[1].attributes.shape == image.image.shape

    attributions = input_x_gradient.attribute([image, image], target=[0, 0])
    assert len(attributions) == 2
    assert attributions[0].attributes.shape == image.image.shape
    assert attributions[1].attributes.shape == image.image.shape

    with pytest.raises(RuntimeError):
        input_x_gradient._attribution_method = None
        input_x_gradient.attribute(image)


def test_occlusion(explainable_toy_model):
    occlusion = Occlusion(explainable_toy_model)
    assert occlusion is not None and occlusion._attribution_method is not None

    image = HSI(image=torch.rand(3, 10, 10), wavelengths=[0, 100, 200])
    attributions = occlusion.attribute(image, sliding_window_shapes=(1, 2, 2), strides=(1, 2, 2))
    assert attributions.attributes.shape == image.image.shape

    attributions = occlusion.attribute(image, sliding_window_shapes=1, strides=1)
    assert attributions.attributes.shape == image.image.shape

    attributions = occlusion.attribute([image, image], sliding_window_shapes=1, strides=1)
    assert len(attributions) == 2
    assert attributions[0].attributes.shape == image.image.shape
    assert attributions[1].attributes.shape == image.image.shape

    postprocessing_segmentation_output = agg_segmentation_postprocessing(classes_numb=3)
    explainable_segmentation_model = explainable_segmentation_toy_model(postprocessing_segmentation_output)
    segment_occlusion = Occlusion(explainable_segmentation_model)

    attributions = segment_occlusion.attribute(
        image,
        target=0,
        sliding_window_shapes=(1, 2, 2),
        strides=(1, 2, 2),
    )
    assert attributions.attributes.shape == image.image.shape

    segment_occlusion = Occlusion(explainable_segmentation_model)
    similar_attributions = segment_occlusion.attribute(
        image,
        target=0,
        sliding_window_shapes=(1, 2, 2),
        strides=(1, 2, 2),
    )
    assert torch.allclose(attributions.attributes, similar_attributions.attributes)

    attributions = segment_occlusion.attribute(
        [image, image],
        target=0,
        sliding_window_shapes=(1, 2, 2),
        strides=(1, 2, 2),
    )
    assert len(attributions) == 2
    assert attributions[0].attributes.shape == image.image.shape
    assert attributions[1].attributes.shape == image.image.shape

    attributions = segment_occlusion.attribute(
        [image, image],
        target=[0, 0],
        sliding_window_shapes=(1, 2, 2),
        strides=(1, 2, 2),
    )
    assert len(attributions) == 2
    assert attributions[0].attributes.shape == image.image.shape
    assert attributions[1].attributes.shape == image.image.shape

    with pytest.raises(ValueError):
        occlusion.attribute(image, sliding_window_shapes=(2, 2), strides=(2, 2, 2))

    with pytest.raises(ValueError):
        occlusion.attribute(image, sliding_window_shapes=(2, 2, 2), strides=(2, 2))

    attributions = occlusion.get_spatial_attributes(image, sliding_window_shapes=(2, 2), strides=(2, 2))
    assert attributions.attributes.shape == image.image.shape
    assert isinstance(attributions, HSISpatialAttributes)

    attributions = occlusion.get_spatial_attributes(image, sliding_window_shapes=2, strides=2)
    assert attributions.attributes.shape == image.image.shape

    attributions = occlusion.get_spatial_attributes([image, image], sliding_window_shapes=2, strides=2)
    assert len(attributions) == 2
    assert attributions[0].attributes.shape == image.image.shape
    assert attributions[1].attributes.shape == image.image.shape

    postprocessing_segmentation_output = agg_segmentation_postprocessing(classes_numb=3)
    explainable_segmentation_model = explainable_segmentation_toy_model(postprocessing_segmentation_output)
    segment_occlusion = Occlusion(explainable_segmentation_model)
    attributions = segment_occlusion.get_spatial_attributes(
        image,
        target=0,
        sliding_window_shapes=2,
        strides=2,
    )
    assert attributions.attributes.shape == image.image.shape

    attributions = segment_occlusion.get_spatial_attributes(
        [image, image],
        target=0,
        sliding_window_shapes=2,
        strides=2,
    )
    assert len(attributions) == 2
    assert attributions[0].attributes.shape == image.image.shape
    assert attributions[1].attributes.shape == image.image.shape

    attributions = segment_occlusion.get_spatial_attributes(
        [image, image],
        target=[0, 0],
        sliding_window_shapes=2,
        strides=2,
    )
    assert len(attributions) == 2
    assert attributions[0].attributes.shape == image.image.shape
    assert attributions[1].attributes.shape == image.image.shape

    with pytest.raises(ValueError):
        occlusion.get_spatial_attributes(image, sliding_window_shapes=(2, 2), strides=(2, 2, 2))

    with pytest.raises(ValueError):
        occlusion.get_spatial_attributes(image, sliding_window_shapes=(2, 2, 2), strides=(2, 2))

    attributions = occlusion.get_spectral_attributes(image, sliding_window_shapes=1, strides=1)
    assert attributions.attributes.shape == image.image.shape
    assert isinstance(attributions, HSISpectralAttributes)

    attributions = occlusion.get_spectral_attributes(image, sliding_window_shapes=(1,), strides=(1,))
    assert attributions.attributes.shape == image.image.shape

    attributions = occlusion.get_spectral_attributes([image, image], sliding_window_shapes=1, strides=1)
    assert len(attributions) == 2
    assert attributions[0].attributes.shape == image.image.shape
    assert attributions[1].attributes.shape == image.image.shape

    postprocessing_segmentation_output = agg_segmentation_postprocessing(classes_numb=3)
    explainable_segmentation_model = explainable_segmentation_toy_model(postprocessing_segmentation_output)
    segment_occlusion = Occlusion(explainable_segmentation_model)
    attributions = segment_occlusion.get_spectral_attributes(
        image,
        target=0,
        sliding_window_shapes=1,
        strides=1,
    )
    assert attributions.attributes.shape == image.image.shape

    attributions = segment_occlusion.get_spectral_attributes(
        [image, image],
        target=0,
        sliding_window_shapes=1,
        strides=1,
    )
    assert len(attributions) == 2
    assert attributions[0].attributes.shape == image.image.shape
    assert attributions[1].attributes.shape == image.image.shape

    attributions = segment_occlusion.get_spectral_attributes(
        [image, image],
        target=[0, 0],
        sliding_window_shapes=1,
        strides=1,
    )
    assert len(attributions) == 2
    assert attributions[0].attributes.shape == image.image.shape
    assert attributions[1].attributes.shape == image.image.shape

    with pytest.raises(ValueError):
        occlusion.get_spectral_attributes(image, sliding_window_shapes=2, strides=(2, 2))

    with pytest.raises(ValueError):
        occlusion.get_spectral_attributes(image, sliding_window_shapes=(2, 2), strides=2)

    with pytest.raises(ValueError):
        occlusion.get_spectral_attributes(image, sliding_window_shapes=(2, 2), strides=(2, 2))

    with pytest.raises(ValueError):
        occlusion.get_spectral_attributes(image, sliding_window_shapes=(2, 2), strides=(2, 2, 2))

    with pytest.raises(ValueError):
        occlusion.get_spectral_attributes(image, sliding_window_shapes=(2, 2, 2), strides=(2, 2))

    with pytest.raises(ValueError):
        occlusion.get_spectral_attributes(image, sliding_window_shapes=(2, 2, 2), strides=(2, 2, 2))

    with pytest.raises(TypeError):
        occlusion.get_spectral_attributes(image, sliding_window_shapes=(1,), strides="0")

    with pytest.raises(RuntimeError):
        occlusion._attribution_method = None
        occlusion.attribute(image, sliding_window_shapes=(2, 2), strides=(2, 2))

    with pytest.raises(RuntimeError):
        occlusion._attribution_method = None
        occlusion.get_spatial_attributes(image, sliding_window_shapes=(2, 2), strides=(2, 2))

    with pytest.raises(RuntimeError):
        occlusion._attribution_method = None
        occlusion.get_spectral_attributes(image, sliding_window_shapes=(2, 2), strides=(2, 2))
