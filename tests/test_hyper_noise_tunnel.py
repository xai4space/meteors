import pytest


from meteors import HyperNoiseTunnel


def test_hyper_noise_tunnel():
    with pytest.raises(TypeError):
        HyperNoiseTunnel()
