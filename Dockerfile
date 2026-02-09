# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Copy pyproject first (for caching)
COPY pyproject.toml .

# Copy the source + web files
COPY src/ ./src/
COPY web/ ./web/
COPY data/ ./data/

# Verify that web files exist (debug)
RUN ls -R /app/web || echo "Warning: web directory not found, UI will not be served."

# Install package (no PyTorch needed for TEaaS)
RUN pip install --no-cache-dir .

# Expose dynamic port for Railway
EXPOSE 8000

# Launch FastAPI (Railway injects $PORT)
CMD uvicorn potatobacon.api.app:app --host 0.0.0.0 --port ${PORT:-8000}
