# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Copy pyproject first (for caching)
COPY pyproject.toml .

# Copy the source + web files
COPY src/ ./src/
COPY web/ ./web/

# Verify that web files exist (debug)
RUN ls -R /app/web || (echo "❌ web directory missing!" && exit 1)

# Install package in editable mode
RUN pip install --no-cache-dir .

# Expose dynamic port for Railway
EXPOSE 8000

# Launch FastAPI (Railway injects $PORT)
CMD uvicorn potatobacon.api.app:app --host 0.0.0.0 --port ${PORT:-8000}
