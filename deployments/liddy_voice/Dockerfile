# Optimized Dockerfile for both local development and GCP deployment
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS base

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Performance optimizations
ENV PYTHONUNBUFFERED=1
ENV PYTHONFAULTHANDLER=1
ENV PYTHONHASHSEED=random

# Create non-privileged user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/appuser" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/appuser/.cache
RUN chown -R appuser:appuser /home/appuser

USER appuser
WORKDIR /home/appuser

# =============================================================================
# STAGE 1: Dependencies (rarely changes - good caching)
# =============================================================================
FROM base AS dependencies

# Install torch first (biggest dependency, specify CPU version)
RUN --mount=type=cache,target=/home/appuser/.cache/pip,uid=10001 \
    python -m pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy and install all requirements at once
COPY --chown=appuser:appuser requirements.txt requirements.txt
COPY --chown=appuser:appuser packages/liddy_voice/requirements-external.txt packages/liddy_voice/requirements-external.txt

# Install base requirements
RUN --mount=type=cache,target=/home/appuser/.cache/pip,uid=10001 \
    python -m pip install --user -r requirements.txt

# Install all livekit requirements from requirements-external.txt
RUN --mount=type=cache,target=/home/appuser/.cache/pip,uid=10001 \
    python -m pip install --user -r packages/liddy_voice/requirements-external.txt

# =============================================================================
# STAGE 2: Application (changes frequently)
# =============================================================================
FROM dependencies AS app

# Copy package source code
COPY --chown=appuser:appuser packages/liddy/ packages/liddy/
COPY --chown=appuser:appuser packages/liddy_intelligence/ packages/liddy_intelligence/
COPY --chown=appuser:appuser packages/liddy_voice/ packages/liddy_voice/

# Install packages in editable mode
RUN --mount=type=cache,target=/home/appuser/.cache/pip,uid=10001 \
    pip install --user -e packages/liddy && \
    pip install --user -e packages/liddy_intelligence && \
    pip install --user -e packages/liddy_voice && \
    python -m pip uninstall -y pinecone-client 2>/dev/null || true

# Create directories that the app expects
RUN mkdir -p response_logs requests logs KMS/logs

# Set Python path to include packages
ENV PYTHONPATH=/home/appuser/packages:$PYTHONPATH

# Note: ENV_TYPE is set by docker-compose or Cloud Run, not here

# Expose healthcheck port
EXPOSE 8081

# Default command - can be overridden by docker-compose
CMD ["sh", "-c", "python packages/liddy_voice/voice_agent.py download-files && python packages/liddy_voice/voice_agent.py start"]