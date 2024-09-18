# ☄️🛰️ Meteors <img src="assets/logo.png" align="right" width="150"/>

[![PyPI](https://img.shields.io/pypi/v/meteors.svg)](https://github.com/xai4space/meteors/blob/main/LICENSE)
[![PyPI - License](https://img.shields.io/pypi/l/meteors?style=flat-square)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/meteors?style=flat-square)](https://pypistats.org/packages/meteors)
[![GitHub star chart](https://img.shields.io/github/stars/xai4space/meteors?style=flat-square)](https://star-history.com/#xai4space/meteors)
[![Open Issues](https://img.shields.io/github/issues-raw/xai4space/meteors?style=flat-square)](https://github.com/xai4space/meteors/issues)
[![Docs - GitHub.io](https://img.shields.io/static/v1?logo=meteors&style=flat&color=pink&label=docs&message=meteors)](https://xai4space.github.io/meteors/latest)

## 🛰️ Introduction

Meteors is an open-source package for creating explanations of hyperspectral and multispectral images. Developed primarily for [Pytorch](https://pytorch.org) models, Meteors was inspired by the [Captum](https://captum.ai/) library. Our goal is to provide not only the ability to create explanations for hyperspectral images but also to visualize them in a user-friendly way.

_Please note that this package is still in the development phase, and we welcome any feedback and suggestions to help improve the library._

Meteors emerged from a research grant project between the Warsaw University of Technology research group [MI2.ai](https://www.mi2.ai/index.html) and [Kp Labs](https://kplabs.space), financially supported by the [European Space Agency (ESA)](https://www.esa.int).

## 🎯 Target Audience

Meteors is designed for:

- Researchers, data scientists, and developers who work with hyperspectral and multispectral images and want to understand the decisions made by their models.
- Engineers who build models for production and want to troubleshoot through improved model interpretability.
- Developers seeking to deliver better explanations to end users on why they're seeing specific content.

## 📦 Installation

**Requirements**

- Python >= 3.9
- PyTorch >= 1.10
- Captum >= 0.7.0

Install with `pip`:

```bash
pip install meteors
```

With conda:
_Coming soon_

## 📚 Documentation

Please refer to the [documentation](https://xai4space.github.io/meteors/latest) for more information on how to use Meteors.

## 🤝 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

We use [rye](https://rye.astral.sh) as our project and package management tool. To start developing, follow these steps:

```bash
curl -sSf https://rye.astral.sh/get | bash # Install Rye
rye pin <python version >=3.9> # Pin the python version
rye sync # Sync the environment
```

Before pushing your changes, please run the tests and the linter:

```bash
rye test
rye run pre-commit run --all-files
```

For more information on how to contribute, please refer to our [Contributing Guide](https://xai4space.github.io/meteors/latest/how-to-guides/).

Thank you for considering contributing to Meteors!

## 💫 Contributors

[![Meteors contributors](https://contrib.rocks/image?repo=xai4space/meteors&max=100)](https://github.com/xai4space/meteors/graphs/contributors)
