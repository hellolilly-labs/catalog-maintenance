---
description: Clean up Docker resources to free disk space
allowed-tools:
  - Bash
---

# Docker Cleanup

Clean up Docker resources including stopped containers, unused images, volumes, and build cache.

## Current Disk Usage
```bash
!echo "=== Current Docker Disk Usage ==="
!docker system df
!echo ""
```

## Cleanup Operations

### 1. Remove stopped containers
```bash
!echo "🧹 Removing stopped containers..."
!docker container prune -f
!echo ""
```

### 2. Remove unused images
```bash
!echo "🧹 Removing unused images..."
!docker image prune -a -f
!echo ""
```

### 3. Remove unused volumes
```bash
!echo "🧹 Removing unused volumes..."
!docker volume prune -f
!echo ""
```

### 4. Remove build cache
```bash
!echo "🧹 Removing build cache..."
!docker builder prune -a -f
!echo ""
```

## Final Disk Usage
```bash
!echo "=== Final Docker Disk Usage ==="
!docker system df
!echo ""
!echo "✅ Docker cleanup complete!"
```

## Additional Options

For more aggressive cleanup (removes ALL containers and images):
```bash
# Stop all containers
docker stop $(docker ps -aq)

# Remove all containers
docker rm $(docker ps -aq)

# Remove all images
docker rmi $(docker images -q) -f

# Full system prune
docker system prune -a --volumes -f
```