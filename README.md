# Installation
```bash
conda create -n repalette --file conda-requirements -c conda-forge -c pytorch-nightly
conda activate repalette
python setup.py sdist
python setup.py bdist_wheel
python setup.py build
python setup.py install
python setup.py develop
```

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

# Download dataset
```bash
python repalette/utils/download_data.py --num_workers 8  # adjust num_workers
python repaletet/utils/build_data.py
```

# Project structure
## data
* `data` - root data directory
* `data/sqlite.db` - database file
* `data/raw` - raw images downloaded from [Design Seeds](https://www.design-seeds.com/blog/page/")
* `data/rgb` - cropped images without palettes in RGB
* `data/models` - `pytorch-lightning` models checkpoints
* `data/pl_logs` - `pytorch-lightning` logs to use with `tensorboard`
## Code
* `repalette` - main python package

# TODO
* Use image augmentations (`torchvision.transforms.RandomWhatever`)
