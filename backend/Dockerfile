FROM ghcr.io/astral-sh/uv:0.13.7 AS uv

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    HF_HOME=/home/scoutx/.cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates default-jre-headless ffmpeg libmagic1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 scoutx \
    && useradd --uid 10001 --gid scoutx --create-home --shell /usr/sbin/nologin scoutx \
    && mkdir -p /app/backend /home/scoutx/.cache/huggingface \
    && chown -R scoutx:scoutx /app /home/scoutx

COPY --from=uv /uv /uvx /bin/
WORKDIR /app/backend

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra extraction --extra embedding --no-install-project

COPY app ./app
COPY migrations ./migrations
COPY alembic.ini ./alembic.ini
COPY evaluation ./evaluation
RUN chown -R scoutx:scoutx /app/backend

USER 10001:10001
EXPOSE 8000
