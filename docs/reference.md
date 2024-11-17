# API Reference

The architecture of the package can be seen on the UML diagram:
![UML diagram of package structure](assets/UML-attribution-methods.png)

## HyperSpectral Image

::: src.meteors.hsi.HSI
    options:
      heading_level: 3
      show_bases: false
      show_root_heading: true
      show_root_full_path: false
      members:
        - spatial_binary_mask
        - change_orientation
        - spectral_axis
        - extract_band_by_name
        - get_image
        - get_rgb_image
        - to

## Visualizations

::: src.meteors.visualize.hsi_visualize
    options:
      heading_level: 3
      show_bases: false
      show_root_heading: false
      show_root_full_path: false

::: src.meteors.visualize.attr_visualize
    options:
      heading_level: 3
      show_bases: false
      show_root_heading: false
      show_root_full_path: false

::: src.meteors.visualize.lime_visualize.visualize_spectral_attributes_by_waveband
    options:
      heading_level: 3
      show_bases: false
      show_root_heading: true
      show_root_full_path: false

::: src.meteors.visualize.lime_visualize.visualize_spectral_attributes_by_magnitude
    options:
      heading_level: 3
      show_bases: false
      show_root_heading: true
      show_root_full_path: false

::: src.meteors.visualize.lime_visualize.visualize_spectral_attributes
    options:
      heading_level: 3
      show_bases: false
      show_root_heading: true
      show_root_full_path: false

::: src.meteors.visualize.lime_visualize.visualize_spatial_attributes
    options:
      heading_level: 3
      show_bases: false
      show_root_heading: true
      show_root_full_path: false

## Attribution Methods

::: src.meteors.attr.attributes.HSIAttributes
    options:
      heading_level: 3
      show_bases: true
      show_root_heading: true
      show_root_full_path: false
      members:
        - flattened_attributes
        - orientation
        - change_orientation
        - to

::: src.meteors.attr.attributes.HSISpatialAttributes
    options:
      heading_level: 3
      show_bases: true
      show_root_heading: true
      show_root_full_path: false

::: src.meteors.attr.attributes.HSISpectralAttributes
    options:
      heading_level: 3
      show_bases: true
      show_root_heading: true
      show_root_full_path: false

::: src.meteors.attr.lime.Lime
    options:
      heading_level: 3
      show_bases: true
      show_root_heading: true
      show_root_full_path: false

#### Lime Base

The Lime Base class was adapted from the Captum Lime implementation. This adaptation builds upon the original work, extending and customizing it for specific use cases within this project. To see the original implementation, please refer to the [Captum repository](https://captum.ai/api/_modules/captum/attr/_core/lime.html#LimeBase).

::: src.meteors.attr.integrated_gradients.IntegratedGradients
    options:
      heading_level: 3
      show_bases: true
      show_root_heading: true
      show_root_full_path: false
      members:
        - attribute

::: src.meteors.attr.input_x_gradients.InputXGradient
    options:
      heading_level: 3
      show_bases: true
      show_root_heading: true
      show_root_full_path: false
      members:
        - attribute

::: src.meteors.attr.occlusion.Occlusion
    options:
      heading_level: 3
      show_bases: true
      show_root_heading: true
      show_root_full_path: false
      members:
        - attribute
        - get_spatial_attributes
        - get_spectral_attributes

::: src.meteors.attr.saliency.Saliency
    options:
      heading_level: 3
      show_bases: true
      show_root_heading: true
      show_root_full_path: false
      members:
        - attribute

::: src.meteors.attr.noise_tunnel.NoiseTunnel
    options:
      heading_level: 3
      show_bases: true
      show_root_heading: true
      show_root_full_path: false
      members:
        - attribute
        - perturb_input

::: src.meteors.attr.noise_tunnel.HyperNoiseTunnel
    options:
      heading_level: 3
      show_bases: true
      show_root_heading: true
      show_root_full_path: false
      members:
        - attribute
        - perturb_input
