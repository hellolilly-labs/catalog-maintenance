#!/bin/bash
set -e

# Load environment variables from config file if it exists
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -f "$SCRIPT_DIR/.env.gcp" ]; then
  source "$SCRIPT_DIR/.env.gcp"
  echo "Loaded configuration from $SCRIPT_DIR/.env.gcp"
elif [ -f "../../config/.env.gcp" ]; then
  source ../../config/.env.gcp
  echo "Loaded configuration from config/.env.gcp"
elif [ -f "config/.env.gcp" ]; then
  source config/.env.gcp
  echo "Loaded configuration from config/.env.gcp"
fi

# Script parameters
PROJECT_ID="laure-430512"
REGION=${REGION:-"us-central1"}
SERVICE_NAME=${SERVICE_NAME:-"voice-service-livekit"}
REPO_NAME=${REPO_NAME:-"voice-services"}
IMAGE_NAME=${IMAGE_NAME:-"voice-service-livekit"}
IMAGE_TAG=${IMAGE_TAG:-"latest"}
VPC_CONNECTOR_NAME="redis-connector"

# Redis configuration
REDIS_HOST="10.210.164.91"
REDIS_PORT="6379"
REDIS_PREFIX=""
REDIS_TTL=3600

# LiveKit and API settings - these will be referenced from Secret Manager
LIVEKIT_URL=${LIVEKIT_URL:-"wss://spence-vde671ld.livekit.cloud"}
LIVEKIT_PORT=${LIVEKIT_PORT:-"8081"}

# Model settings
MODEL_NAME=${MODEL_NAME:-"gpt-4.1"}
FALLBACK_MODEL_NAME=${FALLBACK_MODEL_NAME:-"gemini-2.5-pro-preview-05-06"}
DEFAULT_ANALYSIS_MODEL=${DEFAULT_ANALYSIS_MODEL:-"gpt-4.1"}
GENERAL_ANALYSIS_MODEL=${GENERAL_ANALYSIS_MODEL:-"gpt-4.1"}
PROMPTS_DIR=${PROMPTS_DIR:-"prompts"}

# Bucket that stores prompt files
PROMPTS_BUCKET="liddy-account-documents"

# Voice settings
VOICE_STABILITY=${VOICE_STABILITY:-0.6}
VOICE_SIMILARITY_BOOST=${VOICE_SIMILARITY_BOOST:-0.8}
VOICE_MODEL=${VOICE_MODEL:-"eleven_flash_v2_5"}
USE_NOISE_CANCELLATION=${USE_NOISE_CANCELLATION:-"false"}

# Verify we're using the correct GCP account
CURRENT_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
if [[ "$CURRENT_ACCOUNT" != "crb@liddy.ai" ]]; then
  echo "ERROR: You are currently authenticated as '$CURRENT_ACCOUNT'"
  echo "Please authenticate with the correct account using:"
  echo "gcloud auth login crb@liddy.ai"
  exit 1
fi
echo "Authenticated as $CURRENT_ACCOUNT. Proceeding with deployment."

# Navigate to the monorepo root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MONOREPO_ROOT="$( cd "$SCRIPT_DIR/../../../" && pwd )"
cd "$MONOREPO_ROOT"

echo "Building from monorepo root: $MONOREPO_ROOT"

# Verify the version of livekit-agents and other important dependencies
echo "Checking package versions..."
echo "LiveKit agents version requirement (from requirements.txt):"
grep "livekit-agents" packages/liddy_voice/requirements-external.txt || echo "Not found in requirements-external.txt"

echo "Installed LiveKit agents version:"
pip list | grep livekit-agents || echo "Not installed"

echo "Installed LiveKit plugins:"
pip list | grep livekit-plugins || echo "Not installed"

# Version validation
LIVEKIT_VERSION=$(pip list | grep livekit-agents | awk '{print $2}' || echo "")
if [[ -z "$LIVEKIT_VERSION" ]]; then
  echo "WARNING: livekit-agents package not found! This may cause deployment issues."
elif [[ $(echo "$LIVEKIT_VERSION" | awk -F. '{ printf("%d%03d%03d\n", $1,$2,$3); }') -lt $(echo "1.0.0" | awk -F. '{ printf("%d%03d%03d\n", $1,$2,$3); }') ]]; then
  echo "ERROR: livekit-agents version $LIVEKIT_VERSION is less than the required version 1.0.0"
  echo "Please update livekit-agents to version 1.0.0 or higher before deploying:"
  echo "pip install livekit-agents>=1.0.0"
  exit 1
else
  echo "✅ Using livekit-agents version $LIVEKIT_VERSION (meets minimum requirement of 1.0.0)"
fi

# Make sure we're using the correct project
echo "Setting project to ${PROJECT_ID}..."
gcloud config set project "${PROJECT_ID}"

echo "Deploying LiveKit service to Cloud Run in project ${PROJECT_ID}, region ${REGION}..."

# Check if service exists
echo "Checking if service already exists..."
SERVICE_EXISTS=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format="value(name)" 2>/dev/null || echo "")
if [ -z "$SERVICE_EXISTS" ]; then
  echo "Service ${SERVICE_NAME} doesn't exist yet. It will be created."
else
  echo "Service ${SERVICE_NAME} already exists. It will be updated with the new configuration."
fi

# Check if bucket exists before creating it
if ! gcloud storage buckets describe gs://liddy-conversations &>/dev/null; then
  echo "Creating storage bucket gs://liddy-conversations..."
  gcloud storage buckets create gs://liddy-conversations \
    --project=laure-430512 \
    --location=us-central1 \
    --uniform-bucket-level-access
else
  echo "Storage bucket gs://liddy-conversations already exists."
fi

# -------------------------------------------------------------
# Ensure Cloud Run service account can read prompt bucket
# -------------------------------------------------------------
echo "Ensuring Cloud Run service account has viewer access to gs://${PROMPTS_BUCKET}..."

# Grant roles/storage.objectViewer to the service account for prompt bucket
gcloud storage buckets add-iam-policy-binding gs://${PROMPTS_BUCKET} \
  --member="serviceAccount:${SERVICE_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/storage.objectViewer" || \
  echo "⚠️  Could not add IAM policy binding for ${PROMPTS_BUCKET} (it may already exist or the bucket is missing)."

# Authenticate Docker to Artifact Registry
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Create Artifact Registry repository if it doesn't exist
if ! gcloud artifacts repositories describe "${REPO_NAME}" --location=${REGION} &>/dev/null; then
  echo "Creating Artifact Registry repository..."
  gcloud artifacts repositories create "${REPO_NAME}" \
    --repository-format=docker \
    --location=${REGION} \
    --description="Repository for voice services"
fi

# Build the Docker image for Cloud Run (x86) with caching
echo "Building Docker image with cache optimization..."

# Set the full image name for easier reference
FULL_IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}"

# Try to pull the latest image to use as cache
echo "Pulling latest image for cache (this may fail if it doesn't exist yet)..."
docker pull ${FULL_IMAGE_NAME}:latest || echo "Latest image not found, building without cache"

# Build with cache optimization using the monorepo Dockerfile
docker buildx build --platform linux/amd64 \
  --cache-from=${FULL_IMAGE_NAME}:latest \
  --cache-from=${FULL_IMAGE_NAME}:${IMAGE_TAG} \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  -t ${FULL_IMAGE_NAME}:${IMAGE_TAG} \
  -t ${FULL_IMAGE_NAME}:latest \
  -f deployments/liddy_voice/Dockerfile .

# Push the Docker image
echo "Pushing Docker image to Artifact Registry..."
docker push ${FULL_IMAGE_NAME}:${IMAGE_TAG}
docker push ${FULL_IMAGE_NAME}:latest

# Check if CUDA is requested (for development, Cloud Run doesn't support GPU)
USE_GPU=${USE_GPU:-"false"}
if [ "$USE_GPU" == "true" ]; then
  echo "Warning: Cloud Run doesn't support GPU acceleration."
  echo "The deployment will use CPU-only FAISS implementation."
fi

# Create non-sensitive environment variables string
ENV_VARS="ENV_TYPE=gcp,\
GOOGLE_CLOUD_PROJECT=${PROJECT_ID},\
REDIS_HOST=${REDIS_HOST},\
REDIS_PORT=${REDIS_PORT},\
REDIS_PREFIX=${REDIS_PREFIX},\
REDIS_TTL=${REDIS_TTL},\
LIVEKIT_URL=${LIVEKIT_URL},\
LIVEKIT_PORT=${LIVEKIT_PORT},\
MODEL_NAME=${MODEL_NAME},\
FALLBACK_MODEL_NAME=${FALLBACK_MODEL_NAME},\
DEFAULT_ANALYSIS_MODEL=${DEFAULT_ANALYSIS_MODEL},\
GENERAL_ANALYSIS_MODEL=${GENERAL_ANALYSIS_MODEL},\
PROMPTS_DIR=${PROMPTS_DIR},\
PROMPTS_BUCKET=${PROMPTS_BUCKET},\
VOICE_STABILITY=${VOICE_STABILITY},\
VOICE_SIMILARITY_BOOST=${VOICE_SIMILARITY_BOOST},\
VOICE_MODEL=${VOICE_MODEL},\
USE_NOISE_CANCELLATION=${USE_NOISE_CANCELLATION},\
STORAGE_PROVIDER=gcp,\
STORAGE_BUCKET=liddy-conversations,\
PYTHONFAULTHANDLER=1,\
PYTHONUNBUFFERED=1,\
PYTHONOPTIMIZE=1,\
PYTHONHASHSEED=random,\
OPENAI_MAX_RETRIES=3,\
ANTHROPIC_MAX_RETRIES=3,\
GOOGLE_MAX_RETRIES=3,\
GROQ_MAX_RETRIES=3,\
HTTP_TIMEOUT=30,\
REDIS_SOCKET_KEEPALIVE=1,\
REDIS_CONNECTION_POOL_SIZE=20,\
LANGFUSE_HOST=https://us.cloud.langfuse.com,\
PINECONE_INDEX_NAME=specialized-llama-2048,\
PINECONE_ENVIRONMENT=gcp-starter"

echo "Deploying to Cloud Run with WebSocket support..."
gcloud run deploy ${SERVICE_NAME} \
  --image ${FULL_IMAGE_NAME}:${IMAGE_TAG} \
  --region ${REGION} \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars=${ENV_VARS} \
  --set-secrets="ELEVENLABS_API_KEY=ELEVENLABS_API_KEY:latest,\
ELEVEN_API_KEY=ELEVENLABS_API_KEY:latest,\
OPENAI_API_KEY=OPENAI_API_KEY:latest,\
GOOGLE_API_KEY=GOOGLE_API_KEY:latest,\
LIVEKIT_API_KEY=LIVEKIT_API_KEY:latest,\
LIVEKIT_API_SECRET=LIVEKIT_API_SECRET:latest,\
GROQ_API_KEY=GROQ_API_KEY:latest,\
DEEPGRAM_API_KEY=DEEPGRAM_API_KEY:latest,\
ADMIN_KEY=ADMIN_KEY:latest,\
LANGFUSE_PUBLIC_KEY=LANGFUSE_PUBLIC_KEY:latest,\
LANGFUSE_SECRET_KEY=LANGFUSE_SECRET_KEY:latest,\
ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest,\
ASSEMBLYAI_API_KEY=ASSEMBLYAI_API_KEY:latest,\
PINECONE_API_KEY=PINECONE_API_KEY:latest,\
TAVILY_API_KEY=TAVILY_API_KEY:latest" \
  --vpc-connector=${VPC_CONNECTOR_NAME} \
  --vpc-egress=private-ranges-only \
  --cpu=4 \
  --memory=8Gi \
  --concurrency=13 \
  --min-instances=1 \
  --max-instances=2 \
  --use-http2 \
  --port=${LIVEKIT_PORT} \
  --timeout=3600 \
  --no-cpu-throttling \
  --execution-environment=gen2 \
  --service-account="${SERVICE_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# Explicitly allow public invocation
gcloud run services add-iam-policy-binding ${SERVICE_NAME} \
  --member="allUsers" \
  --role="roles/run.invoker" \
  --region=${REGION}

# Attach Cloud Armor security policy
echo "Attaching Cloud Armor security policy to the Cloud Run service..."
gcloud run services update ${SERVICE_NAME} \
  --region=${REGION} \
  --cloud-armor-security-policy=voice-service-policy

echo "LiveKit service deployed successfully!"
echo "Service URL: $(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format='value(status.url)')"
echo "To view the logs for version verification, run:"
echo "gcloud logging read 'resource.type=cloud_run_revision AND resource.labels.service_name=${SERVICE_NAME}' --project=${PROJECT_ID} --limit=50"