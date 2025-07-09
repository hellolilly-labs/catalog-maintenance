---
description: Start the voice agent with Docker Compose
allowed-tools:
  - Bash
---

# Start Voice Agent

Start the voice agent using Docker Compose with the appropriate configuration.

## Pre-flight Checks
```bash
!echo "=== Pre-flight Checks ==="
!cd /Users/collinbrown/Dropbox/Development/Lily/code/catalog-maintenance/deployments/liddy_voice

# Check Docker
!docker ps >/dev/null 2>&1 && echo "✅ Docker daemon is running" || { echo "❌ Docker daemon is not running - please start Docker Desktop"; exit 1; }

# Check for .env file
![ -f ../../.env ] && echo "✅ Root .env file found" || { echo "❌ Root .env file missing"; exit 1; }

# Check Redis
!nc -zv localhost 6379 2>&1 | grep -q succeeded && echo "✅ Redis is running" || echo "⚠️  Redis not detected - will use bundled Redis"

# Check gcloud auth
![ -f ~/.config/gcloud/application_default_credentials.json ] && echo "✅ gcloud credentials found" || echo "⚠️  No gcloud credentials - run: gcloud auth application-default login"

!echo ""
```

## Starting Voice Agent
```bash
!echo "=== Starting Voice Agent ==="
!cd /Users/collinbrown/Dropbox/Development/Lily/code/catalog-maintenance/deployments/liddy_voice

# Determine which compose file to use based on Redis availability
!if nc -zv localhost 6379 2>&1 | grep -q succeeded; then
  echo "Using docker-compose-external-redis.yml (external Redis detected)"
  COMPOSE_FILE="docker-compose-external-redis.yml"
else
  echo "Using docker-compose.yml (will start bundled Redis)"
  COMPOSE_FILE="docker-compose.yml"
fi

# Enable BuildKit for better caching
!export DOCKER_BUILDKIT=1
!export COMPOSE_DOCKER_CLI_BUILD=1

# Clear conflicting environment variables
!unset GOOGLE_APPLICATION_CREDENTIALS

# Start the services
!echo ""
!echo "🚀 Starting voice agent..."
!echo "   Compose file: $COMPOSE_FILE"
!echo ""
!echo "To view logs: docker compose -f $COMPOSE_FILE logs -f"
!echo "To stop: docker compose -f $COMPOSE_FILE down"
!echo ""

# Start in detached mode
!docker compose -f $COMPOSE_FILE up -d --build
```

## Check Status
```bash
!echo ""
!echo "=== Container Status ==="
!sleep 3  # Give containers time to start
!docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(NAMES|voice-agent|redis)"
!echo ""
!echo "Voice agent should be available at: http://localhost:8081"
```

## View Logs
To view the logs after starting:
```bash
cd /Users/collinbrown/Dropbox/Development/Lily/code/catalog-maintenance/deployments/liddy_voice
docker compose -f docker-compose-external-redis.yml logs -f
```