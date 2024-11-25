# Changelog

## v0.1.2 (2024-11-21)

### 🛎️ Chores

- update the package version ([#142](https://github.com/xai4space/meteors/pull/142))

### 🩺 Bug Fixes

- documentation version alias update fixed ([#139](https://github.com/xai4space/meteors/pull/139))

### 📚 Documentation

- tutorials fixes ([#129](https://github.com/xai4space/meteors/pull/129))
- fixed documentation versioning ([#135](https://github.com/xai4space/meteors/pull/135))
- Documentation template update ([#134](https://github.com/xai4space/meteors/pull/134))
- fix reference docs for visualizer modules ([#133](https://github.com/xai4space/meteors/pull/133))
- fix navbar for new tutorials ([#132](https://github.com/xai4space/meteors/pull/132))
- update changelog.md for v0.1.1 [skip ci] ([#131](https://github.com/xai4space/meteors/pull/131))

### 🔨 Features

- Added Hyperlinks to the PR in changelog ([#136](https://github.com/xai4space/meteors/pull/136))

## v0.1.1 (2024-11-18)

### 📚 Documentation

- refine the tutorial for lime and add tutorial for attributions methods (#128)
- refine the example for segmentation problem type (#124)

### 🩺 Bug Fixes

- Moved the postprocessing to the ExplainableModel (#123)
- Corrected visualisers (#118)
- Corrected Attributes Functionalities (#117)
- changelog.md and release notes do not contain all commits due to the commit processor not parsing multi-line commit messages. (#116)

### 🔨 Features

- Updated release and docs GitHub actions to trigger when GitHub tag is pushed (#119)

## v0.1.0 (2024-10-29)

### 🛎️ Chores

- update the project version (#112)
- reduced coverage threshold (#94)

### 🔨 Features

- 96 feat segmentation support for attribution methods (#103)
- refactored the package structure (#111)
- Custom errors (#101)

### 🩺 Bug Fixes

- corrected visualizations (#106)
- Updated occlusion (#93)
- 91 docs the lime illustration image is missing in the docs (#92)

### 📚 Documentation

- update changelog.md for 0.0.4 [skip ci] (#90)

## v0.0.4 (2024-09-25)

### 🩺 Bug Fixes

- infinite loop in segmentation (#87)

### 🔨 Features

- HyperNoiseTunnel and captum attribution methods (#51)

## v0.0.3 (2024-09-23)

### 🩺 Bug Fixes

- github action release workflow to pypi (#83)

## meteors 0.0.2 (2024-08-11)

- No release
- Refined package structure - simple modules for models and visualisation, installation using toml file
- Spectral attributions using LIME
- CUDA compatibility of LIME

## meteors 0.0.1 (2024-06-02)

- No release
- Prepared a simple draft of package along with some ideas and sample files for implementation of LIME for hyperspectral images.
- Segmentation mask for LIME using slic
- Spatial attributions using LIME
