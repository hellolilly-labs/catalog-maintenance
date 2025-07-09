---
description: Check the status of the voice agent deployment and related services
allowed-tools:
  - Bash
  - Read
---

# Voice Agent Status Check

Check the current status of the voice agent deployment, including Docker, Redis, authentication, and environment configuration.

## 1. Docker Status
```bash
!echo "=== Docker Status ==="
!docker --version 2>/dev/null && echo "✅ Docker is installed" || echo "❌ Docker is not installed"
!docker ps >/dev/null 2>&1 && echo "✅ Docker daemon is running" || echo "❌ Docker daemon is not running"
!echo ""
```

## 2. Redis Status
```bash
!echo "=== Redis Status ==="
!nc -zv localhost 6379 2>&1 | grep -q succeeded && echo "✅ Redis is running on localhost:6379" || echo "❌ Redis is not running on localhost:6379"
!docker ps | grep -q redis && echo "✅ Redis container is running" || echo "⚠️  No Redis container found"
!echo ""
```

## 3. Google Cloud Authentication
```bash
!echo "=== Google Cloud Authentication ==="
!if [ -f ~/.config/gcloud/application_default_credentials.json ]; then echo "✅ gcloud application default credentials found"; else echo "❌ No gcloud application default credentials"; fi
!gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | head -1 | xargs -I {} echo "✅ Authenticated as: {}"
!echo "GOOGLE_APPLICATION_CREDENTIALS: ${GOOGLE_APPLICATION_CREDENTIALS:-Not set}"
!echo ""
```

## 4. Voice Agent Container Status
```bash
!echo "=== Voice Agent Container Status ==="
!docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.State}}" | grep -E "(NAMES|voice-agent)" || echo "No voice agent containers found"
!echo ""
```

## 5. Environment Configuration
```bash
!echo "=== Environment Configuration ==="
!cd /Users/collinbrown/Dropbox/Development/Lily/code/catalog-maintenance && [ -f .env ] && echo "✅ Root .env file exists" || echo "❌ Root .env file missing"
!cd /Users/collinbrown/Dropbox/Development/Lily/code/catalog-maintenance && [ -L deployments/liddy_voice/.env ] && echo "✅ Voice deployment .env symlink exists" || echo "⚠️  Voice deployment .env symlink missing"
!echo ""
```

## 6. Required Environment Variables
```bash
!echo "=== Required Environment Variables Check ==="
!cd /Users/collinbrown/Dropbox/Development/Lily/code/catalog-maintenance && source .env 2>/dev/null && {
  [ -n "$LIVEKIT_API_KEY" ] && echo "✅ LIVEKIT_API_KEY is set" || echo "❌ LIVEKIT_API_KEY is missing"
  [ -n "$LIVEKIT_API_SECRET" ] && echo "✅ LIVEKIT_API_SECRET is set" || echo "❌ LIVEKIT_API_SECRET is missing"
  [ -n "$LIVEKIT_URL" ] && echo "✅ LIVEKIT_URL is set" || echo "❌ LIVEKIT_URL is missing"
  [ -n "$OPENAI_API_KEY" ] && echo "✅ OPENAI_API_KEY is set" || echo "❌ OPENAI_API_KEY is missing"
  [ -n "$GOOGLE_API_KEY" ] && echo "✅ GOOGLE_API_KEY is set" || echo "❌ GOOGLE_API_KEY is missing"
  [ -n "$ENV_TYPE" ] && echo "ℹ️  ENV_TYPE: $ENV_TYPE" || echo "ℹ️  ENV_TYPE: Not set (defaults to development)"
}
!echo ""
```

## 7. Recent Voice Agent Logs
```bash
!echo "=== Recent Voice Agent Logs (last 10 lines) ==="
!docker logs liddy_voice-voice-agent-1 2>&1 | tail -10 || echo "No logs available"
```

## Quick Actions

To fix common issues:
- **Docker not running**: Start Docker Desktop
- **Redis not running**: `docker run -d -p 6379:6379 redis:7-alpine`
- **No gcloud auth**: `gcloud auth application-default login`
- **Start voice agent**: `cd deployments/liddy_voice && ./test-docker-compose.sh`