#!/bin/bash
# Docker entrypoint script that handles product loading and app startup

set -e

echo "🚀 Starting Liddy Voice Agent..."

# Default to using Redis for products if not explicitly set
: ${USE_REDIS_PRODUCTS:=true}
export USE_REDIS_PRODUCTS

# Function to check if Redis is available
check_redis() {
    if [ -n "$REDIS_URL" ]; then
        echo "Checking Redis connection..."
        python -c "
import redis
import sys
try:
    r = redis.from_url('$REDIS_URL')
    r.ping()
    print('✅ Redis is available')
    sys.exit(0)
except:
    print('❌ Redis is not available')
    sys.exit(1)
" || return 1
    else
        echo "❌ REDIS_URL not set"
        return 1
    fi
}

# If USE_REDIS_PRODUCTS is enabled, load data first
if [ "$USE_REDIS_PRODUCTS" = "true" ]; then
    echo "📦 Redis data loading enabled"
    
    # Check Redis availability
    if check_redis; then
        echo "Loading products and accounts into Redis..."
        
        # Load data for all accounts (or specified via VOICE_ACCOUNTS)
        python packages/liddy_voice/voice_agent.py load-data
        
        if [ $? -eq 0 ]; then
            echo "✅ Data loaded successfully"
        else
            echo "❌ Failed to load data into Redis"
            # Decide whether to exit or continue without data
            if [ "$REQUIRE_PRODUCTS" = "true" ]; then
                exit 1
            fi
        fi
    else
        echo "❌ Redis not available, falling back to in-memory storage"
        export USE_REDIS_PRODUCTS=false
    fi
fi

# Run any other initialization steps
if [ -f "download-files.sh" ]; then
    echo "📥 Downloading files..."
    ./download-files.sh
fi

# Start the application
echo "🎤 Starting voice agent..."
exec "$@"