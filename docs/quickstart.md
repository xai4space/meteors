# 🚀 Quickstart

Welcome to the Quickstart Guide for Meteors! This guide will walk you through the basic steps to get started with using Meteors for explaining your hyperspectral and multispectral image models.

## 📋 Prerequisites

Before you begin, make sure you have the following installed:

- Python >= 3.9
- PyTorch >= 1.10
- Captum >= 0.7.0
- shap >= 0.46.0

## 📥 Installation

To install Meteors, simply run the following command:

```bash
pip install meteors
```

For conda users, we will provide a conda installation method in the near future. We promise!🤞

## 🌟 Basic Hyperspectral or Multispectral Data Object

Meteors provide an easy-to-use object for handling hyperspectral and multispectral images. The `HSI` object is a simple way to process and manipulate your data.

```python
from meteors import HSI
```

**Remember**, when providing data to the model, make sure it is in the final format that the model expects, without the batch dimension. The `HSI` object will handle the rest.
We also recommend providing the image orientation, e.g., `'HWC'` for height, width, and channels, or `'CHW'` for channels, height, and width.


## 🔍 Explanation Methods

Meteors provides several explanation methods for hyperspectral and multispectral images, including:

- **LIME**: [Local Interpretable Model-agnostic Explanations](https://christophm.github.io/interpretable-ml-book/lime.html)
- **SHAP** [SHapley Additive exPlanations](https://papers.nips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)
- **Saliency Maps**: [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901)
- **Integrated Gradients**: [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365)
- **SmoothGrad**: [SmoothGrad: removing noise by adding noise](https://arxiv.org/abs/1706.03825)
- **HyperNoiseTunnel**: our own modification of SmoothGrad for hyperspectral and multispectral images, more details can be found in the [docs](https://xai4space.github.io/meteors/latest/tutorials/attr_showcase/#hypernoisetunnel)
- **Occlusion**: [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901)

and more methods are on the way!
To use a specific explanation method in [tutorials](tutorials/introduction.md) we provide for each method, example code.

## 🎨 Visualization Options

Meteors offers various visualization options to help you understand and interpret the explanations in package `meteors.visualize`.

```python
from meteors.visualize import visualize_spectral_attributes, visualize_spatial_attributes
```

## 📚 Tutorials

We have several tutorials to help get you off the ground with Meteors. The tutorials are Jupyter notebooks and cover the basics along with demonstrating usage of Meteors .

View the tutorials page [here](tutorials/introduction.md).

## 📖 API Reference

For an in-depth reference of the various Meteors internals, see our [API Reference](reference.md).

## 🙌 Contributing

We welcome contributions to Meteors! Please refer to our [Contributing Guide](how-to-guides.md) for more information on how to get involved.
