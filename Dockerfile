FROM nvidia/cuda:10.2-base
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y python3.8 python3-pip awscli

WORKDIR ./repalette

RUN python3.8 -m pip install poetry==1.1.4  # install specific version of poetry with pip, not the official script

COPY pyproject.toml poetry.lock ./

RUN poetry install --no-dev

COPY ./repalette ./scripts ./