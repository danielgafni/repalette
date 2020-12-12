# Installation
```bash
poetry install
```

## Configure
```bash
cp ./.env.example ./.env
```
Fill it with the correct values.

## Install Jupyter Lab plugins

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

# Dataset
## Download data
### Option 1: scrap data from www.design-seeds.com
```bash
python repalette/utils/download_raw.py --num_workers 8  # adjust num_workers
python repalette/utils/build_rgb.py
```
### Option 2: download prepared data from S3
```bash
python repalette/db/utils/download_rgb_from_s3.py
```
## Update S3 dataset
```bash
python repalette/db/utils/upload_rgb_to_s3.py

```

# Project structure
## data
* `data` - root data directory
* `data/raw.sqlite`, `data/rgb.sqlite` - databases
* `data/raw` - raw images downloaded from [Design Seeds](https://www.design-seeds.com/blog/page/")
* `data/rgb` - cropped images without palettes in RGB
* `data/model-checkpoints` - `pytorch-lightning` models checkpoints
* `data/lightning-logs` - `pytorch-lightning` `tensorboard` (or other logger) logs
## Code
* `repalette` - main python package
* `scripts` - training scripts