# API Reference

::: src.meteors.hsi.HSI
options:
show_bases: false
show_root_heading: true
show_root_full_path: false

## Visualizations

::: src.meteors.visualize.lime_visualize
options:
heading_level: 3
show_bases: false
show_root_heading: true
show_root_full_path: false
members: - "visualize_spectral_attributes_by_waveband" - "visualize_spectral_attributes_by_magnitude" - "visualize_spectral_attributes" - "visualize_spatial_attributes"

## Methods

### LIME

::: src.meteors.lime.HSIAttributes
options:
heading_level: 4
show_bases: false
show_root_heading: true
show_root_full_path: false

::: src.meteors.lime.HSISpatialAttributes
options:
heading_level: 4
show_bases: true
show_root_heading: true
show_root_full_path: false

::: src.meteors.lime.HSISpectralAttributes
options:
heading_level: 4
show_bases: true
show_root_heading: true
show_root_full_path: false

::: src.meteors.lime.Explainer
options:
heading_level: 4
show_bases: false
show_root_heading: true
show_root_full_path: false

::: src.meteors.lime.Lime
options:
heading_level: 4
show_bases: true
show_root_heading: true
show_root_full_path: false

### Lime Base

The Lime Base class was adapted from the Captum Lime implementation. This adaptation builds upon the original work, extending and customizing it for specific use cases within this project. To see the original implementation, please refer to the [Captum repository](https://captum.ai/api/_modules/captum/attr/_core/lime.html#LimeBase).
