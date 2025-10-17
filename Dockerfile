# Use a lightweight Python base image
FROM python:3.11-slim

# Environment setup
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PTB_DATA_ROOT=/data

# Set working directory
WORKDIR /app

# System dependencies (build + runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc curl bash && \
    rm -rf /var/lib/apt/lists/*

# Copy project metadata and source code
COPY pyproject.toml ./
COPY src ./src
COPY web ./web
COPY examples ./examples
COPY docs ./docs

# Install dependencies (editable for dev mode)
RUN pip install --upgrade pip && pip install -e ".[dev]"

# Expose port (for local testing)
EXPOSE 8000

# Create data directory for runtime artifacts
RUN mkdir -p /data

# --- Final startup command ---
# Use bash -c so that ${PORT} is expanded correctly on Railway
# Fallback to 8000 if no $PORT provided (for local runs)
CMD ["bash", "-c", "uvicorn potatobacon.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
