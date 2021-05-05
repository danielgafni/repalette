# Usage
1. Install [Docker](https://docs.docker.com/engine/install/)
2. Pull the web app container:
```bash
docker pull danielgafni/repalette:app
```
3. Run the container:
```bash
docker run -p 8000:8000 danielgafni/repalette:app
```
4. Open [localhost:8000](localhost:8000) in your browser
5. In the web interface:
- Upload the image
- Select the desired color palette
- Press the "recolor" button

# Screenshots

![image](screenshots/flowers.jpg)
![image](screenshots/flowers_recolored.png)

# Development
## Installation
```bash
poetry install
poetry run pre-commit install  # for development
```
To activate the virtual environment run `poetry shell`

### Configuration
```bash
cp ./.env.example ./.env
```
Fill it with the correct values.

### Install Jupyter Lab plugins

```bash
# Avoid "JavaScript heap out of memory" errors during extension installation
# (OS X/Linux)
export NODE_OPTIONS=--max-old-space-size=4096
# (Windows)
set NODE_OPTIONS=--max-old-space-size=4096
# Jupyter widgets extension
jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build

# jupyterlab plotly renderer support
jupyter labextension install jupyterlab-plotly --no-build

# Plotly FigureWidget support
jupyter labextension install plotlywidget --no-build

# Build extensions (must be done to activate extensions since --no-build is used above)
jupyter lab build

# Unset NODE_OPTIONS environment variable
# (OS X/Linux)
unset NODE_OPTIONS
# (Windows)
set NODE_OPTIONS=
```
## Download data
### Option 1: scrap data from www.design-seeds.com
```bash
python repalette/db/utils/download_raw.py --num_workers 8  # adjust num_workers
python repalette/db/utils/build_rgb.py
```
### Option 2: download prepared data from S3
This data might be a little outdated comparing to the #1 option, but will be downloaded much faster.
```bash
python repalette/db/utils/download_rgb_from_s3.py
```
## Update S3 dataset
```bash
python repalette/db/utils/upload_rgb_to_s3.py
```
## Download the pre-trained model checkpoint:
```bash
python repalette/db/utils/download_pretrain_checkpoint_from_s3.py
```

## Training
The model can be trained on the data downloaded from www.design-seeds.com. After running the training script the logs (losses, images, etc) will be available at localhost:6006.
### Pre-training
```bash
python repalette/training/pretrain.py
```
### GAN training
```bash
python repalette/training/gan.py
```

## Project structure
### data
* `data` - root data directory
* `data/raw.sqlite`, `data/rgb.sqlite` - databases
* `data/raw` - raw images downloaded from [Design Seeds](https://www.design-seeds.com/blog/page/")
* `data/rgb` - cropped images without palettes in RGB
* `data/model-checkpoints` - `pytorch-lightning` models checkpoints
* `data/lightning-logs` - `pytorch-lightning` `tensorboard` (or other logger) logs
### Code
* `repalette` - main python package
* `scripts` - misc scripts
* `app` - web app
