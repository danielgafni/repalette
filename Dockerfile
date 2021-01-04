FROM nvidia/cuda:10.2-base
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y python3.8 python3-pip python3.8-dev \
       awscli build-essential mysql-client libmysqlclient-dev \
       && rm -rf /var/lib/apt/lists/*

WORKDIR ./repalette

RUN python3.8 -m pip install poetry==1.1.4  # install specific version of poetry with pip, not the official install script

COPY pyproject.toml poetry.lock ./

ARG PIP_NO_CACHE_DIR=1

RUN poetry install --no-dev  # this will install all production dependencies

COPY ./repalette ./repalette

RUN poetry install --no-dev  # this will only install the actual `repalette` project copied above

#RUN rm -r ~/.cache

COPY ./scripts/pretrain.py ./scripts/pretrain.py

ADD ./scripts/aws_docker_setup.sh ./scripts/aws_docker_setup.sh

ENTRYPOINT ["./scripts/aws_docker_setup.sh"]
