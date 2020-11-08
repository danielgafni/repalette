import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import os
import sqlalchemy

from repalette.constants import BASE_DATA_DIR, RAW_DATA_DIR, DATA_DIR, DATABASE_PATH

DESIGN_SEEDS_PAGES_ROOT = r"https://www.design-seeds.com/blog/page/"


def get_image_links():
    image_links = []
    i = 0
    response = requests.get(DESIGN_SEEDS_PAGES_ROOT + str(i))
    bar = tqdm()
    while response.status_code != 404:
        soup = BeautifulSoup(response.content, "html.parser")
        image_links.extend(
            [img["src"] for img in soup.find_all(class_="attachment-full")]
        )
        bar.update(n=1)

        i += 1
        response = requests.get(DESIGN_SEEDS_PAGES_ROOT + str(i))

    return image_links


def download_images(image_links):
    if not os.path.exists(BASE_DATA_DIR):
        os.mkdir(BASE_DATA_DIR)
    if not os.path.exists(RAW_DATA_DIR):
        os.mkdir(RAW_DATA_DIR)
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)

    engine = sqlalchemy.create_engine(f"sqlite:////{DATABASE_PATH}")

    for image_link in image_links:
        # TODO: multiprocessing
        pass
