# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Copy project metadata first
COPY pyproject.toml .

# Copy the actual source so pip can build from src/
COPY src/ ./src/

# Install project dependencies (uses pyproject.toml)
RUN pip install --no-cache-dir .

# Copy the web UI after install
COPY web/ ./web/

# Verify that /web exists
RUN ls -R /app/web

EXPOSE 8000

# Start FastAPI (Railway injects $PORT automatically)
CMD ["uvicorn", "potatobacon.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
