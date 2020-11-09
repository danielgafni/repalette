from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import torch
from torchvision.transforms import Resize
import numpy as np
from pandas import DataFrame
from itertools import permutations
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from repalette.constants import ROOT_DIR, IMAGE_SIZE, DATABASE_PATH
from repalette.utils.color import smart_hue_adjust
from repalette.utils.models import RawImage
from multiprocessing import Pool
from tqdm import tqdm
import os
from repalette.constants import DATA_DIR
from PIL import ImageColor
from repalette.utils.data import RawDataset


def find_edges(array, edge_size=10):
    edges = []
    for i, idx in enumerate(array):
        for delta in range(1, len(array) - i):
            if not array[i + delta] == idx + delta:
                break
            edges.append(idx)
    return edges


def cut_numpy_image(np_image):
    np_image = np_image[5:, 5:, :]
    white_x = np.argwhere((np_image.mean(axis=2).mean(axis=1) == 255))
    white_y = np.argwhere((np_image.mean(axis=2).mean(axis=0) == 255))

    edges_x = find_edges(white_x, 10)
    edge_x = int(min(edges_x)) if edges_x else None

    edges_y = find_edges(white_y, 10)
    edge_y = int(min(edges_y)) if edges_y else None

    assert edge_x or edge_y, (edge_x, edge_y)

    if edge_x:
        np_image = np_image[: int(edge_x), :, :]
    if edge_y:
        np_image = np_image[:, : int(edge_y), :]

    return np_image


def validate_image(np_image):
    if np.prod(np_image.shape) > 160000:
        return True
    else:
        return False


def process_image_info(image, palette):
    np_image = np.array(image)
    np_image = cut_numpy_image(np_image)
    if validate_image(np_image):
        np_palette = np.array(
            [ImageColor.getcolor(color, "RGB") for color in reversed(palette)]
        ).reshape(1, 6, 3)
        return np_image, np_palette
    else:
        return None, None


def main():
    raw_dataset = RawDataset()

    for (image, palette), raw_image in tqdm(
        raw_dataset, desc="Processing", total=len(raw_dataset)
    ):
        np_image, np_palette = process_image_info(image, palette)
        if np_image is not None:
            processed_image = Image.fromarray(np_image)
            path = os.path.join(DATA_DIR, raw_image.name)
            processed_image.save(path, "PNG")
        else:
            print(f"Dropping image {raw_image.path}...")


if __name__ == "__main__":
    main()
