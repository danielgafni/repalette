import numpy as np
import torch
from typing import Tuple

from repalette.constants import L_RANGE, A_RANGE, B_RANGE


def normalize_lab_img(img):
    """Normalizes image in LAB format to be in [-1, 1] range by a linear transformation."""
    img = torch.stack([
        normalize_img_component(img[..., 0, :, :], L_RANGE),
        normalize_img_component(img[..., 1, :, :], A_RANGE),
        normalize_img_component(img[..., 2, :, :], B_RANGE),
    ], dim=-3)
    return img


def restore_lab_img(img):
    img = torch.stack([
        normalize_img_component(img[..., 0, :, :], (-1, 1), L_RANGE),
        normalize_img_component(img[..., 1, :, :], (-1, 1), A_RANGE),
        normalize_img_component(img[..., 2, :, :], (-1, 1), B_RANGE),
    ], dim=-3)
    return img


def normalize_img_component(component, old_range: Tuple[float, float],
                            new_range: Tuple[float, float] = (-1, 1)):
    new_component = component - np.mean(old_range)
    new_component *= (new_range[1] - new_range[0]) / (old_range[1] - old_range[0])
    new_component += np.mean(new_range)
    return new_component
