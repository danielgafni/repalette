import argparse
from matplotlib import pyplot as plt
import colorgram
from PIL import Image
import numpy as np


def extract_palette(filepath: str, num_colors: int = 6, backend="colorgram") -> np.ndarray:
    """
    Extracts palette from image.
    :param filepath: path to image
    :type filepath: str
    :param num_colors: number of colors in the palette
    :type num_colors: int
    :param backend: supported backends: `colorgram`
    :type backend: str
    :return: RGB color palette with shape [1, num_colors, 3]
    :rtype: np.ndarray
    """
    if backend == "colorgram":
        colors = colorgram.extract(filepath, num_colors)
        channels = "rgb"
        palette = np.expand_dims(
            np.array(
                [
                    [getattr(color.rgb, channel) for channel in channels] for color in colors
                ]
            ), 0
        )
        return palette
    else:
        raise NotImplemented(f"Backend {backend} is not implemented.")


def viz_image(filepath: str, num_colors: int = 6, backend="colorgram") -> None:
    """
    Plots image along with its color palette
    :param filepath: path to image
    :type filepath: str
    :param num_colors: number of colors in the palette
    :type num_colors: int
    :param backend: backend for `repalette.utils.extract_palette`
    :type backend: str
    """
    plt.imshow(Image.open(filepath))
    plt.show()
    plt.imshow(extract_palette(filepath, num_colors=num_colors, backend=backend))
    plt.show()


def main(filepath=None, directorypath=None, num_colors=6, destination_path=None):
    if filepath and directorypath:
        raise ValueError("Only one of \"filepath\" or \"directorypath\" can be specified.")

    pass
