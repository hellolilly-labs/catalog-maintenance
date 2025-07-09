# Redis Configuration for Voice Agent

The voice agent uses Redis for caching account configurations. Here's how Redis connectivity works in different environments:

## 1. Local IDE Development (Running Python directly)
Your `.env` file should have:
```env
REDIS_HOST=localhost
REDIS_PORT=6379
```

Make sure Redis is running locally:
```bash
# Using homebrew on macOS
brew services start redis

# Or run Redis in Docker
docker run -d -p 6379:6379 redis:7-alpine
```

## 2. Docker Local Testing (docker-compose)
The `docker-compose.yml` automatically:
- Starts a Redis container
- Overrides `REDIS_HOST=redis` to point to the container
- Links the containers together

Just run:
```bash
docker compose up
```

## 3. Cloud Run Deployment
For production, you'll need to:
- Use Google Cloud Memorystore (managed Redis)
- Or deploy Redis separately and configure the connection

Update your Cloud Run environment variables:
```bash
gcloud run services update voice-agent \
  --update-env-vars REDIS_HOST=your-redis-ip,REDIS_PORT=6379
```

## Troubleshooting

### Check Redis connectivity in Docker:
```bash
# From host machine
docker compose exec voice-agent sh -c "nc -zv redis 6379"

# Check Redis is working
docker compose exec redis redis-cli ping
```

### If you already have Redis running on host:
You can use the host's Redis instead of the Docker Redis:

1. Comment out the Redis service in docker-compose.yml
2. Change REDIS_HOST to use host network:
   ```yaml
   environment:
     - REDIS_HOST=host.docker.internal  # macOS/Windows
     # or
     - REDIS_HOST=172.17.0.1  # Linux default bridge
   ```

### Redis connection errors:
If you see Redis connection errors, it might be because:
1. Redis is not running
2. Wrong host/port configuration
3. Network connectivity issues between containers

The voice agent should continue to work even without Redis, but performance may be slower as it won't cache account configurations.