#!/bin/bash

echo "======================================================"
echo "Docker Cleanup Script"
echo "======================================================"
echo ""

# Show current disk usage
echo "Current Docker disk usage:"
docker system df
echo ""

# Clean up stopped containers
echo "Removing stopped containers..."
docker container prune -f

# Clean up unused images
echo "Removing unused images..."
docker image prune -a -f

# Clean up unused volumes
echo "Removing unused volumes..."
docker volume prune -f

# Clean up unused networks
echo "Removing unused networks..."
docker network prune -f

# Clean up build cache
echo "Removing build cache..."
docker builder prune -a -f

# Full system prune (aggressive)
echo ""
echo "Running full system prune..."
docker system prune -a --volumes -f

echo ""
echo "Cleanup complete! New disk usage:"
docker system df