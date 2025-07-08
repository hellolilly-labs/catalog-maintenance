ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Performance optimizations
ENV PYTHONUNBUFFERED=1
ENV PYTHONFAULTHANDLER=1
ENV PYTHONHASHSEED=random
ENV PYTHONOPTIMIZE=1

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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/appuser/.cache
RUN chown -R appuser:appuser /home/appuser

USER appuser

WORKDIR /home/appuser

# Copy and install base liddy package first (rarely changes)
COPY --chown=appuser:appuser packages/liddy/setup.py packages/liddy/setup.py
COPY --chown=appuser:appuser packages/liddy/ packages/liddy/
RUN --mount=type=cache,target=/home/appuser/.cache/pip,uid=10001 \
    python -m pip install --user --no-cache-dir -e packages/liddy

# Copy and install liddy_intelligence package (changes less frequently)
COPY --chown=appuser:appuser packages/liddy_intelligence/setup.py packages/liddy_intelligence/setup.py
COPY --chown=appuser:appuser packages/liddy_intelligence/ packages/liddy_intelligence/
RUN --mount=type=cache,target=/home/appuser/.cache/pip,uid=10001 \
    python -m pip install --user --no-cache-dir -e packages/liddy_intelligence

# Copy liddy_voice requirements and install external dependencies
COPY --chown=appuser:appuser packages/liddy_voice/requirements-external.txt packages/liddy_voice/requirements-external.txt
RUN --mount=type=cache,target=/home/appuser/.cache/pip,uid=10001 \
    python -m pip install --user --no-cache-dir -r packages/liddy_voice/requirements-external.txt

# Install torch separately for CPU
RUN --mount=type=cache,target=/home/appuser/.cache/pip,uid=10001 \
    python -m pip install --user --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Create directories that the app expects
RUN mkdir -p response_logs requests logs KMS/logs

# Copy voice service package and install
COPY --chown=appuser:appuser packages/liddy_voice/pyproject.toml packages/liddy_voice/pyproject.toml
COPY --chown=appuser:appuser packages/liddy_voice/setup.py packages/liddy_voice/setup.py
COPY --chown=appuser:appuser packages/liddy_voice/ packages/liddy_voice/
RUN --mount=type=cache,target=/home/appuser/.cache/pip,uid=10001 \
    python -m pip install --user --no-cache-dir -e packages/liddy_voice && \
    python -m pip uninstall -y pinecone-client 2>/dev/null || true

# Set environment type for GCP deployment
ENV ENV_TYPE=gcp

# Set Python path to include packages
ENV PYTHONPATH=/home/appuser/packages:$PYTHONPATH

# Note: Account data (account.json, products.json) is loaded from GCS at runtime
# No need to copy data/config/prompts directories - they don't exist in monorepo

# Pre-download required model files during build (if needed)
# This step is optional - models can be downloaded at runtime if preferred
# RUN cd packages/liddy_voice && python voice_agent.py download-files

# Expose healthcheck port
EXPOSE 8081

# Run the application (use "dev" for development mode)
CMD ["python", "packages/liddy_voice/voice_agent.py", "dev"]