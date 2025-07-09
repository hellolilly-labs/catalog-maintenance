#!/bin/bash
set -e

# Script to create or update Google Cloud secrets from .env file

# Function to create or update a secret
create_or_update_secret() {
    local SECRET_NAME=$1
    local SECRET_VALUE=$2
    
    # Check if secret exists
    if gcloud secrets describe "$SECRET_NAME" &>/dev/null; then
        echo "Updating existing secret: $SECRET_NAME"
        echo -n "$SECRET_VALUE" | gcloud secrets versions add "$SECRET_NAME" --data-file=-
    else
        echo "Creating new secret: $SECRET_NAME"
        echo -n "$SECRET_VALUE" | gcloud secrets create "$SECRET_NAME" --data-file=-
    fi
}

# Load the main .env file from catalog-maintenance root
ENV_FILE="../../../.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: .env file not found at $ENV_FILE"
    exit 1
fi

echo "Loading environment variables from $ENV_FILE"

# Source the .env file in a subshell and create secrets
while IFS='=' read -r key value; do
    # Skip comments and empty lines
    [[ $key =~ ^#.*$ ]] && continue
    [[ -z "$key" ]] && continue
    
    # Remove leading/trailing whitespace
    key=$(echo "$key" | xargs)
    
    # Remove quotes from value if present
    value=$(echo "$value" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")
    
    # Only process specific secrets that are used in the deployment
    case "$key" in
        ELEVENLABS_API_KEY|ELEVEN_API_KEY|OPENAI_API_KEY|GOOGLE_API_KEY|LIVEKIT_API_KEY|LIVEKIT_API_SECRET|GROQ_API_KEY|DEEPGRAM_API_KEY|ADMIN_KEY|LANGFUSE_PUBLIC_KEY|LANGFUSE_SECRET_KEY|ANTHROPIC_API_KEY|ASSEMBLYAI_API_KEY|PINECONE_API_KEY|TAVILY_API_KEY)
            if [ -n "$value" ]; then
                create_or_update_secret "$key" "$value"
            else
                echo "Warning: $key has no value, skipping"
            fi
            ;;
        *)
            # Skip other variables
            ;;
    esac
done < "$ENV_FILE"

echo "Secret creation/update complete!"
echo ""
echo "To verify secrets were created:"
echo "gcloud secrets list"
echo ""
echo "To see a specific secret's value:"
echo "gcloud secrets versions access latest --secret=SECRET_NAME"