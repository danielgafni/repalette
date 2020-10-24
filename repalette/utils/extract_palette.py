import argparse
from matplotlib import pyplot as plt
import colorgram
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import cv2
import pywal


def extract_palette(filepath: str, num_colors: int = 6, backend="colorgram", *args, **kwargs) -> np.ndarray:
    """
    Extracts palette from image.
    :param filepath: path to image
    :type filepath: str
    :param num_colors: number of colors in the palette
    :type num_colors: int
    :param backend: supported backends: `colorgram`, `kmeans`, `pywal/{pywal_backend}`.
    :type backend: str
    :param args: args go into backend
    :param kwargs: kwargs go into backend
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
    elif backend == "kmeans":
        cl = KMeans(n_clusters=num_colors, *args, **kwargs)
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).reshape(-1, 3)
        cl.fit(image)
        return cl.cluster_centers_.reshape(1, 6, 3).astype(np.int64)
    elif "pywal" in backend:
        backend = backend.split("/")[1]

        if backend not in ["wal", "colorthief", "colorz", "haishoku", "schemer"]:
            raise NotImplemented(f"Backend pywal/{backend} is not implemented.")

        color_dict = pywal.colors.get(filepath, color_count=num_colors, backend=backend, *args, **kwargs)
        palette = []
        for key in color_dict:
            palette.append(color_dict["key"])
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
