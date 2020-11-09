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

from repalette.constants import BASE_DATA_DIR, RAW_DATA_DIR, DATA_DIR, DATABASE_PATH
from repalette.utils.models import RawImage
from repalette.utils.models import Base


DESIGN_SEEDS_PAGES_ROOT = r"https://www.design-seeds.com/blog/page/"


def get_image_urls_and_palettes():
    image_urls = []
    palettes = []

    i = 0
    bar = tqdm(desc="Parsing")
    response = requests.get(DESIGN_SEEDS_PAGES_ROOT + str(i))
    bar.update(n=1)
    while response.status_code != 404:
        bs = BeautifulSoup(response.content, "html.parser")
        posts = bs.find_all(class_="entry-content")

        for post in posts:
            image_url = post.find_all(class_="attachment-full")[0]["src"]
            palette = [header.text for header in post.find_all("h5")[1:]]

            image_urls.append(image_url)
            palettes.append(palette)

        bar.update(n=1)

        i += 1
        response = requests.get(DESIGN_SEEDS_PAGES_ROOT + str(i))

    return image_urls, palettes


def get_image_name(image_url):
    return image_url.split("/")[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    if not os.path.exists(BASE_DATA_DIR):
        os.mkdir(BASE_DATA_DIR)
    if not os.path.exists(RAW_DATA_DIR):
        os.mkdir(RAW_DATA_DIR)
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)

    engine = create_engine(f"sqlite:///{DATABASE_PATH}")
    # create a configured "Session" class
    Session = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)

    def download_image_to_database(image_data):
        url, palette = image_data
        name = get_image_name(url)
        path = os.path.join(RAW_DATA_DIR, name)

        # create a database Session
        session = Session()

        raw_image = RawImage(
            path=path,
            palette=palette,
            url=url,
            name=name,
        )
        try:
            session.add(raw_image)
            # if add successful (new image) - download image
            image_data = requests.get(url).content
            image = Image.open(io.BytesIO(image_data))
            raw_image.height = image.height
            raw_image.width = image.width
            session.commit()
            # save image on disk
            image.save(path, "PNG")

        except IntegrityError:  # image already in the database
            pass

    image_urls, palettes = get_image_urls_and_palettes()

    with Pool(args.num_workers) as pool:
        with tqdm(desc="Downloading", total=len(image_urls)) as bar:
            for _ in pool.imap_unordered(
                download_image_to_database, list(zip(image_urls, palettes))
            ):
                bar.update(n=1)
