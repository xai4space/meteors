from __future__ import annotations


from captum.attr import NoiseTunnel as CaptumNoiseTunnel

from meteors import HSI
from meteors.attr import HSIAttributes, Explainer

from meteors.attr.explainer import validate_attribution_method_initialization


class NoiseTunnel(Explainer):
    """
    NoiseTunnel explainer class for generating attributions using the Noise Tunnel method.
    This attribution method works on top of a different one to better approximate its explanations.
    The Noise Tunnel (Smooth Grad) adds Gaussian noise to each input in the batch and applies the given attribution algorithm to each modified sample.
    This method is based on the [`captum` implementation](https://captum.ai/api/noise_tunnel.html)

    Attributes:
        _attribution_method (CaptumNoiseTunnel): The Noise Tunnel method from the `captum` library.
    """

    def __init__(self, attribution_method: Explainer):
        super().__init__(attribution_method)
        validate_attribution_method_initialization(attribution_method)
        self._attribution_method = CaptumNoiseTunnel(attribution_method._attribution_method)

    def attribute(
        self,
        hsi: HSI,
        target: int | None = None,
        nt_type="smoothgrad",
        nt_samples=5,
        nt_samples_batch_size=None,
        stdevs=1.0,
        draw_baseline_from_distrib=False,
    ) -> HSIAttributes:
        if self._attribution_method is None:
            raise ValueError("NoiseTunnel explainer is not initialized")
        if self.chained_explainer is None:
            raise ValueError(
                f"The attribution method {self.chained_explainer.__class__.__name__} is not properly initialized"
            )

        noise_tunnel_attributes = self._attribution_method.attribute(
            hsi.get_image().unsqueeze(0),
            target=target,
            nt_type=nt_type,
            nt_samples=nt_samples,
            nt_samples_batch_size=nt_samples_batch_size,
            stdevs=stdevs,
            draw_baseline_from_distrib=draw_baseline_from_distrib,
        )

        attributes = HSIAttributes(
            hsi=hsi, attributes=noise_tunnel_attributes.squeeze(0), attribution_method=self.get_name()
        )

        return attributes
