#!/bin/bash
set -e

# Load environment variables from config file if it exists
if [ -f "../../config/.env.gcp" ]; then
  source ../../config/.env.gcp
elif [ -f "config/.env.gcp" ]; then
  source config/.env.gcp
fi

# Script parameters - explicitly set PROJECT_ID to the correct format
PROJECT_ID="laure-430512"
REGION=${REGION:-"us-central1"}
SERVICE_NAME=${SERVICE_NAME:-"voice-service-livekit"}
VPC_CONNECTOR=${VPC_CONNECTOR:-"redis-connector"}
VPC_NETWORK=${VPC_NETWORK:-"default"}
REPO_NAME="voice-services"

# Verify we're using the correct GCP account
CURRENT_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
if [[ "$CURRENT_ACCOUNT" != "crb@liddy.ai" ]]; then
  echo "ERROR: You are currently authenticated as '$CURRENT_ACCOUNT'"
  echo "Please authenticate with the correct account using:"
  echo "gcloud auth login crb@liddy.ai"
  exit 1
fi
echo "Authenticated as $CURRENT_ACCOUNT. Proceeding with setup."

# Make sure we're using the correct project
echo "Setting project to ${PROJECT_ID}..."
gcloud config set project "${PROJECT_ID}"

echo "Setting up GCP infrastructure for LiveKit Service in project ${PROJECT_ID}, region ${REGION}..."

# Check if VPC network exists, if not create it
if ! gcloud compute networks describe "${VPC_NETWORK}" &>/dev/null; then
  echo "Creating VPC network ${VPC_NETWORK}..."
  gcloud compute networks create "${VPC_NETWORK}" --subnet-mode=auto
fi

# Check if VPC connector exists, if not create it
if ! gcloud compute networks vpc-access connectors describe "${VPC_CONNECTOR}" --region="${REGION}" &>/dev/null; then
  echo "Creating VPC connector ${VPC_CONNECTOR}..."
  gcloud compute networks vpc-access connectors create "${VPC_CONNECTOR}" \
    --network="${VPC_NETWORK}" \
    --region="${REGION}" \
    --range="10.8.0.0/28"
fi

# Create a service account for the LiveKit service if it doesn't exist
SA_EMAIL="${SERVICE_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
if ! gcloud iam service-accounts describe "${SA_EMAIL}" &>/dev/null; then
  echo "Creating service account ${SA_EMAIL}..."
  gcloud iam service-accounts create "${SERVICE_NAME}" \
    --display-name="LiveKit Voice Service Account"
fi

# Grant necessary permissions to the service account
echo "Granting IAM permissions to service account..."
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/secretmanager.secretAccessor"

# Grant Storage Object Admin role to the service account
echo "Granting Storage permissions to service account..."
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.objectAdmin"

# Grant Speech Client role to the service account
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/speech.client"

# Give current user permissions to act as the service account
echo "Granting permissions for $CURRENT_ACCOUNT to act as the service account..."
gcloud iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
  --member="user:${CURRENT_ACCOUNT}" \
  --role="roles/iam.serviceAccountUser"

# For Cloud Run admin permissions
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="user:${CURRENT_ACCOUNT}" \
  --role="roles/run.admin"

# Create Artifact Registry repository if it doesn't exist
if ! gcloud artifacts repositories describe "${REPO_NAME}" --location=${REGION} &>/dev/null; then
  echo "Creating Artifact Registry repository..."
  gcloud artifacts repositories create "${REPO_NAME}" \
    --repository-format=docker \
    --location=${REGION} \
    --description="Repository for voice services"
fi

# Create secrets if they don't exist
echo "Setting up Secret Manager secrets..."

# Function to create a secret if it doesn't exist
create_secret_if_not_exists() {
  local secret_name=$1
  local secret_value=$2
  
  if ! gcloud secrets describe "${secret_name}" &>/dev/null; then
    echo "Creating secret ${secret_name}..."
    echo -n "${secret_value}" | gcloud secrets create "${secret_name}" --data-file=-
  else
    echo "Secret ${secret_name} already exists."
  fi
}

# Note: These are placeholder values. Real secrets should be stored in Secret Manager
# and not hardcoded in scripts
echo "WARNING: Using placeholder secret values. Please update these in Secret Manager!"

create_secret_if_not_exists "LIVEKIT_API_KEY" "${LIVEKIT_API_KEY:-PLACEHOLDER_KEY}"
create_secret_if_not_exists "LIVEKIT_API_SECRET" "${LIVEKIT_API_SECRET:-PLACEHOLDER_SECRET}"
create_secret_if_not_exists "LIVEKIT_URL" "${LIVEKIT_URL:-wss://spence-vde671ld.livekit.cloud}"
create_secret_if_not_exists "OPENAI_API_KEY" "${OPENAI_API_KEY:-PLACEHOLDER_KEY}"
create_secret_if_not_exists "GOOGLE_API_KEY" "${GOOGLE_API_KEY:-PLACEHOLDER_KEY}"
create_secret_if_not_exists "ELEVENLABS_API_KEY" "${ELEVENLABS_API_KEY:-PLACEHOLDER_KEY}"
create_secret_if_not_exists "DEEPGRAM_API_KEY" "${DEEPGRAM_API_KEY:-PLACEHOLDER_KEY}"
create_secret_if_not_exists "GROQ_API_KEY" "${GROQ_API_KEY:-PLACEHOLDER_KEY}"
create_secret_if_not_exists "REDIS_URL" "${REDIS_URL:-10.210.164.91}"
create_secret_if_not_exists "REDIS_PASSWORD" "${REDIS_PASSWORD:-}"
create_secret_if_not_exists "ADMIN_KEY" "${ADMIN_KEY:-PLACEHOLDER_KEY}"

# Set up Cloud Armor
echo "Setting up Cloud Armor WAF protection..."
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -f "$SCRIPT_DIR/setup-cloud-armor.sh" ]; then
  bash "$SCRIPT_DIR/setup-cloud-armor.sh"
else
  echo "Cloud Armor setup script not found. Please run setup-cloud-armor.sh manually after this script completes."
fi

echo "GCP infrastructure setup completed!"
echo "Next steps:"
echo "1. Review the secrets in Secret Manager to ensure they have valid values"
echo "2. Run the deploy script to deploy the LiveKit service with enhanced security"