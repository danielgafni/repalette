import io
import os

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

from repalette.constants import BASE_DATA_DIR, PRETRAINED_MODEL_CHECKPOINT_PATH, ROOT_DIR
from repalette.inference import fit_image_to_max_size, html2numpy, recolor_image
from repalette.lightning.systems import PreTrainSystem

# directories
STATIC_FILES_DIR = os.path.join(BASE_DATA_DIR, "static")
USER_IMAGES_DIR = os.path.join(STATIC_FILES_DIR, "user-images")
TEMPLATES_DIR = os.path.join(ROOT_DIR, "app", "templates")


def setup():
    os.makedirs(STATIC_FILES_DIR, exist_ok=True)
    os.makedirs(USER_IMAGES_DIR, exist_ok=True)
    load_dotenv()
    max_image_size = int(os.getenv("MAX_IMAGE_SIZE"))

    return {"max_image_size": max_image_size}


setup_config = setup()

app = FastAPI()
app.mount(STATIC_FILES_DIR, StaticFiles(directory=STATIC_FILES_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

generator = PreTrainSystem.load_from_checkpoint(PRETRAINED_MODEL_CHECKPOINT_PATH).generator
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")
generator.to(device)


@app.get("/")
@app.get("/index")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/")
async def recolor(
    request: Request,
    file: UploadFile = File(...),
    color1: str = Form(...),
    color2: str = Form(...),
    color3: str = Form(...),
    color4: str = Form(...),
    color5: str = Form(...),
    color6: str = Form(...),
):
    colors = [color1, color2, color3, color4, color5, color6]
    numpy_palette = html2numpy(colors)
    image = Image.open(io.BytesIO((await file.read()))).convert("RGB")

    image = fit_image_to_max_size(image=image, max_image_size=setup_config["max_image_size"])

    recolored_image = recolor_image(image=image, palette=numpy_palette, generator=generator, device=device)
    image_path = os.path.join(USER_IMAGES_DIR, "recolored_image.png")

    recolored_image.save(image_path)

    return templates.TemplateResponse("index.html", {"request": request, "image_path": image_path, "colors": colors})
