import requests
import argparse
from bs4 import BeautifulSoup
from tqdm import tqdm
import io
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

from multiprocessing import Pool
from PIL import Image

from repalette.constants import (
    BASE_DATA_DIR,
    RAW_DATA_DIR,
    RGB_IMAGES_DIR,
    DEFAULT_RAW_DATABASE,
)
from repalette.db import image_url_to_name
from repalette.db.raw import RawImage, RAWBase


DESIGN_SEEDS_PAGES_ROOT = r"https://www.design-seeds.com/blog/page/"


def get_image_urls_and_palettes():
    image_urls = []
    palettes = []

    i = 0
    skipped = 0
    bar = tqdm(desc=f"Parsing... skipped: [{skipped}]")
    response = requests.get(DESIGN_SEEDS_PAGES_ROOT + str(i))
    bar.update(n=1)
    while response.status_code != 404:
        bs = BeautifulSoup(response.content, "html.parser")
        posts = bs.find_all(class_="entry-content")

        for post in posts:
            image_url = post.find_all(class_="attachment-full")[0]["src"]
            palette = [header.text for header in post.find_all("h5") if "#" in header.text]

            if (
                len(palette) != 6
            ):  # this happens on "Anniversary posts" - they are duplicates anyway
                skipped += 1
                bar.desc = f"Parsing... skipped: [{skipped}]"
                break

            image_urls.append(image_url)
            palettes.append(palette)

        bar.update(n=1)

        i += 1
        response = requests.get(DESIGN_SEEDS_PAGES_ROOT + str(i))

    return image_urls, palettes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(BASE_DATA_DIR, exist_ok=True)
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(RGB_IMAGES_DIR, exist_ok=True)

    engine = create_engine(DEFAULT_RAW_DATABASE)
    # create a configured "Session" class
    Session = sessionmaker(bind=engine)
    RAWBase.metadata.create_all(engine)

    def download_image_to_database(image_data):
        url, palette = image_data
        name = image_url_to_name(url)

        # create a database Session
        session = Session()

        raw_image = RawImage(palette=palette, url=url, name=name)
        try:
            session.add(raw_image)
            # if add successful (new image) - download image
            image_data = requests.get(url).content
            image = Image.open(io.BytesIO(image_data))
            raw_image.height = image.height
            raw_image.width = image.width
            session.commit()
            # save image on disk
            image.save(raw_image.path, "PNG")

        except IntegrityError:  # image already in the database
            pass

    (
        image_urls,
        palettes,
    ) = get_image_urls_and_palettes()

    with Pool(args.num_workers) as pool:
        with tqdm(
            desc="Downloading",
            total=len(image_urls),
        ) as bar:
            for _ in pool.imap_unordered(
                download_image_to_database,
                list(zip(image_urls, palettes)),
            ):
                bar.update(n=1)
