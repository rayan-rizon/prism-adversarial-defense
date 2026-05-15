import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import BACKBONE_INPUT_SIZE, BACKBONE_MEAN, BACKBONE_STD
from src.prism import PRISM


def test_dct_feature_image_round_trips_to_pixel_space():
    pixel = torch.rand(1, 3, BACKBONE_INPUT_SIZE, BACKBONE_INPUT_SIZE)
    mean = torch.tensor(BACKBONE_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(BACKBONE_STD).view(1, 3, 1, 1)
    x_norm = (pixel - mean) / std

    recovered = PRISM._normalised_to_pixel_numpy(x_norm)

    assert recovered.shape == tuple(pixel.shape[1:])
    assert np.allclose(recovered, pixel.squeeze(0).numpy(), atol=1e-6)


def test_explicit_pixel_image_takes_precedence():
    pixel = torch.full((1, 3, BACKBONE_INPUT_SIZE, BACKBONE_INPUT_SIZE), 0.25)
    unrelated_norm = torch.randn_like(pixel)

    recovered = PRISM._normalised_to_pixel_numpy(unrelated_norm, pixel_image=pixel)

    assert np.allclose(recovered, pixel.squeeze(0).numpy(), atol=1e-6)
