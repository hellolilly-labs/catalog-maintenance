# Liddy Voice Service Deployment

This directory contains deployment configurations and scripts for the Liddy Voice Service.

## Directory Structure

```
liddy_voice/
├── Dockerfile              # Docker container definition
├── gcp/                    # Google Cloud Platform deployment
│   ├── deploy-cloudrun.sh  # Deploy to Cloud Run
│   ├── setup-gcp-infra.sh  # Setup GCP infrastructure
│   └── setup-cloud-armor.sh # Setup Cloud Armor WAF
└── README.md              # This file
```

## Prerequisites

1. **GCP Authentication**: Ensure you're authenticated with the correct account:
   ```bash
   gcloud auth login crb@liddy.ai
   ```

2. **Environment Variables**: The scripts will look for configuration in:
   - `../../config/.env.gcp` (from deploy directory)
   - `config/.env.gcp` (from monorepo root)

3. **Docker**: Make sure Docker is installed and running

## Deployment Steps

### 1. Initial Infrastructure Setup

Run this once to set up the GCP infrastructure:

```bash
cd deployments/liddy_voice/gcp
./setup-gcp-infra.sh
```

This will:
- Create VPC network and connector
- Set up service accounts with proper permissions
- Create Artifact Registry repository
- Initialize Secret Manager secrets
- Set up Cloud Armor WAF protection

### 2. Deploy the Service

To deploy or update the voice service:

```bash
cd deployments/liddy_voice/gcp
./deploy-cloudrun.sh
```

This will:
- Build the Docker image from the monorepo root
- Push to Artifact Registry
- Deploy to Cloud Run with proper configuration
- Attach Cloud Armor security policy

## Important Notes

1. **Secrets**: The setup script creates placeholder secrets. You must update these in Secret Manager with real values before the service will work properly.

2. **Monorepo Structure**: The deployment scripts are designed to work with the monorepo structure. They expect to be run from `deployments/liddy_voice/gcp/` and will navigate to the monorepo root for building.

3. **Docker Build**: The Dockerfile at `deployments/liddy_voice/Dockerfile` is configured to:
   - Install packages in the correct order (liddy → liddy_intelligence → liddy_voice)
   - Use build caching for faster deployments
   - Run as a non-root user for security

## Configuration

Key environment variables configured in the deployment:

- `LIVEKIT_URL`: LiveKit server URL
- `REDIS_HOST`: Redis server for caching
- `MODEL_NAME`: Primary LLM model
- `PROMPTS_BUCKET`: GCS bucket for prompt files
- Various API keys stored in Secret Manager

## Monitoring

View logs after deployment:

```bash
gcloud logging read 'resource.type=cloud_run_revision AND resource.labels.service_name=voice-service-livekit' --project=laure-430512 --limit=50
```

## Updating

To update the deployment:

1. Make your code changes
2. Run `./deploy-cloudrun.sh` again
3. The script will build and deploy the new version

The deployment uses Docker layer caching, so unchanged dependencies won't be rebuilt.