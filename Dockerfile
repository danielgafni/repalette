FROM pytorch/pytorch

WORKDIR ./repalette

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

COPY ./pyproject.toml .
COPY ./poetry.lock .

RUN poetry install

COPY ./repalette .
COPY ./scripts .