FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PTB_DATA_ROOT=/data

WORKDIR /app

# System deps (build + runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc curl && \
    rm -rf /var/lib/apt/lists/*

# Copy only project metadata first to leverage Docker layer caching
COPY pyproject.toml ./
COPY src ./src
COPY examples ./examples
COPY docs ./docs

RUN pip install --upgrade pip && pip install -e ".[dev]"

EXPOSE 8000

# Create data dir for artifacts
RUN mkdir -p /data

CMD ["uvicorn", "potatobacon.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
