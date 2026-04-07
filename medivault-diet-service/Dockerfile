# ================================
# MediVault Diet Service — Dockerfile
# Optimised for Railway (small image, no sentence-transformers)
#
# Why a Dockerfile instead of nixpacks.toml:
# Nixpacks built a Nix-based Python 3.11.10 that failed during
# `pip install -r requirements.txt` on Railway (exit 1 with no error body).
# The standard `python:3.11-slim-bookworm` image has all the C/SSL headers
# needed for compiled wheels (cryptography, grpcio, etc.) and builds cleanly.
# ================================

# Stage 1: Build dependencies
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# System build deps — required when pip has to build any wheel from source
# (rare with modern manylinux wheels, but safer to have them for cryptography,
# grpcio, and any transitive source-only packages). Cleaned up in stage 2.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      gcc \
      libffi-dev \
      libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install deps into a virtual env so stage 2 can copy just the site-packages
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    # Strip unused test dirs + byte-compile artefacts to shrink the image
    find /opt/venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null; \
    find /opt/venv -type d -name "test" -exec rm -rf {} + 2>/dev/null; \
    find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
    find /opt/venv -name "*.pyc" -delete 2>/dev/null; \
    find /opt/venv -name "*.pyo" -delete 2>/dev/null; \
    true

# Stage 2: Slim runtime image
FROM python:3.11-slim-bookworm

# Create non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy only the virtual env from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY . .

# Give the runtime user ownership of /app so any local-writable paths (caches,
# logs) work without permission errors. /opt/venv stays root-owned and is
# world-readable which is the standard pattern for site-packages.
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Railway injects PORT env var at runtime
ENV PORT=5001
EXPOSE 5001

# Start FastAPI via gunicorn — $PORT is set by Railway at runtime
CMD exec gunicorn app.main:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120
