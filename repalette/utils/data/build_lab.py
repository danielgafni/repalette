import torch
import argparse
from skimage.color import rgb2lab
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from multiprocessing import Pool
from tqdm import tqdm

from repalette.datasets import RGBDataset
from repalette.db import LABTensor, Base
from repalette.constants import LAB_IMAGES_DIR, LAB_PALETTES_DIR, DEFAULT_DATABASE


def main():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", default=8, type=int)
    args = parser.parse_args()

    rgb_dataset = RGBDataset()

    if not os.path.exists(LAB_IMAGES_DIR):
        os.makedirs(LAB_IMAGES_DIR)

    if not os.path.exists(LAB_PALETTES_DIR):
        os.makedirs(LAB_PALETTES_DIR)

    engine = create_engine(DEFAULT_DATABASE)
    # create a configured "Session" class
    Session = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)

    indices = list(range(len(rgb_dataset)))

    session = Session()

    def download_lab_image_and_palette_to_database(index):
        (image, palette), rgb_image = rgb_dataset[index]
        try:
            name = "".join(rgb_image.name.split(".")[:-1])
            image_path = os.path.join(LAB_IMAGES_DIR, name + ".pt")
            palette_path = os.path.join(LAB_PALETTES_DIR, name + ".pt")
            lab_tensor = LABTensor(
                image_path=rgb_image.path,
                palette_path=palette_path,
                url=rgb_image.url,
                name=name,
                height=rgb_image.height,
                width=rgb_image.width,
            )

            session.add(lab_tensor)

            image_lab = rgb2lab(image)
            palette_lab = rgb2lab(palette)

            image_lab = torch.from_numpy(image_lab)
            palette_lab = torch.from_numpy(palette_lab)

            torch.save(image_lab, image_path)
            torch.save(palette_lab, palette_path)
        except IntegrityError:  # image already in database
            pass

    with Pool(args.num_workers) as pool:
        with tqdm(desc="Building LAB", total=len(indices)) as bar:
            for _ in pool.imap_unordered(
                download_lab_image_and_palette_to_database, indices
            ):
                bar.update(n=1)

    session.commit()
