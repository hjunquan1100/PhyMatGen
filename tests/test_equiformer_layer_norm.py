import math

import torch

from src.models.components.equiformer_v2.layer_norm import EquivariantDegreeLayerScale


def test_degree_layer_scale_initializes_expected_weights() -> None:
    layer = EquivariantDegreeLayerScale(lmax=2, num_channels=1, scale_factor=2.0)

    expected = torch.tensor([1.0, 1.0 / math.sqrt(2.0), 1.0 / math.sqrt(4.0)])
    torch.testing.assert_close(layer.affine_weight.detach().flatten(), expected)
