# Installation
```bash
conda create -n repalette --file conda-requirements -c conda-forge -c pytorch-nightly
conda activate repalette
pip install colorgram.py git+https://github.com/dylanaraps/pywal
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

# jupyterlab renderer support
jupyter labextension install jupyterlab-plotly --no-build

# FigureWidget support
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

# TODO

* Use image augmentations (`torchvision.transforms.RandomWhatever`)
