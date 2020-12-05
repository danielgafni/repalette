# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['repalette',
 'repalette.datasets',
 'repalette.model_common',
 'repalette.models',
 'repalette.utils',
 'repalette.utils.models']

package_data = \
{'': ['*']}

install_requires = \
['SQLAlchemy>=1.3.20,<2.0.0',
 'beautifulsoup4>=4.9.3,<5.0.0',
 'pytorch-lightning>=1.0.8,<2.0.0',
 'requests>=2.25.0,<3.0.0',
 'scikit-image>=0.17.2,<0.18.0',
 'scikit-learn>=0.23.2,<0.24.0',
 'torch>=1.7.0,<2.0.0',
 'torchvision>=0.8.1,<0.9.0',
 'tqdm>=4.54.1,<5.0.0']

setup_kwargs = {
    'name': 'repalette',
    'version': '0.1.0',
    'description': '',
    'long_description': None,
    'author': 'Your Name',
    'author_email': 'you@example.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.8,<4.0',
}


setup(**setup_kwargs)
