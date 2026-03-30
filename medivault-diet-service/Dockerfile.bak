# ================================
# MediVault Diet Service — Dockerfile
# Optimised for Railway (small image, no sentence-transformers)
# ================================

# Stage 1: Build dependencies
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

COPY requirements.txt .

# Install deps into a virtual env so we can copy just the site-packages
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    # Remove unnecessary large files from installed packages
    find /opt/venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null; \
    find /opt/venv -type d -name "test" -exec rm -rf {} + 2>/dev/null; \
    find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
    find /opt/venv -name "*.pyc" -delete 2>/dev/null; \
    find /opt/venv -name "*.pyo" -delete 2>/dev/null; \
    find /opt/venv -name "*.dist-info" -exec rm -rf {} + 2>/dev/null; \
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

# Switch to non-root user
USER appuser

# Railway injects PORT env var
ENV PORT=5001
EXPOSE 5001

# Start FastAPI via gunicorn — $PORT is set by Railway at runtime
CMD exec gunicorn app.main:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120
