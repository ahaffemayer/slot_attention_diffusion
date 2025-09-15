FROM python:3.13-slim-trixie

EXPOSE 7000

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

RUN apt update \
 && apt install -y unzip \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ADD ressources.zip /app
RUN unzip ressources.zip

ADD . /app
RUN uv sync --locked \
 && uv cache clean

ENTRYPOINT ["/app/entrypoint.sh"]
