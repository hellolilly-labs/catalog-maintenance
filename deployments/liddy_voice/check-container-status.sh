#!/bin/bash

echo "===================================================="
echo "Checking Voice Agent Container Status"
echo "===================================================="
echo ""

# Check if container is running
CONTAINER_STATUS=$(docker ps --filter "name=voice-agent" --format "table {{.Names}}\t{{.Status}}" | tail -n +2)

if [ -z "$CONTAINER_STATUS" ]; then
    echo "❌ No voice-agent container is running"
    echo ""
    echo "Recent container logs (if any):"
    docker logs voice-agent-1 2>&1 | tail -20 || echo "No logs available"
else
    echo "✅ Container is running:"
    echo "$CONTAINER_STATUS"
    echo ""
    echo "Recent logs:"
    docker logs voice-agent-1 2>&1 | tail -20
fi

echo ""
echo "To view full logs: docker logs -f voice-agent-1"
echo "To stop: docker compose down"