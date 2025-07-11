"""
Redis Client Module

Provides utility functions for interacting with Redis.
"""

import logging
import os
import json
import time
import re
from typing import Dict, Optional, Any, List

# Import Redis library
import redis

# Get logger for this module
logger = logging.getLogger(__name__)


# Load Redis configuration from environment variables with fallbacks
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)
REDIS_DB = int(os.getenv('REDIS_DB', 0))
REDIS_PREFIX = os.getenv('REDIS_PREFIX', '')
# REDIS_PREFIX = 'user_state:'
REDIS_TTL = int(os.getenv('REDIS_TTL', 3600))  # 1 hour default TTL
MEMORY_CACHE_TTL = int(os.getenv('MEMORY_CACHE_TTL', 600))  # 10 minute default TTL

# Cloud provider-specific Redis configuration
def setup_redis_connection():
    """Setup Redis connection based on environment variables for different cloud providers"""
    # Check for GCP Memorystore environment variables
    if os.getenv('REDIS_HOST'):
        return  # Already configured
    
    # GCP VPC connector with private IP
    if os.getenv('MEMORYSTORE_REDIS_ENDPOINT'):
        endpoint = os.getenv('MEMORYSTORE_REDIS_ENDPOINT')
        if ':' in endpoint:
            host, port = endpoint.split(':')
            os.environ['REDIS_HOST'] = host
            os.environ['REDIS_PORT'] = port
        else:
            os.environ['REDIS_HOST'] = endpoint
            
    # AWS ElastiCache
    elif os.getenv('ELASTICACHE_ENDPOINT'):
        os.environ['REDIS_HOST'] = os.getenv('ELASTICACHE_ENDPOINT')
        
    # Azure Cache for Redis
    elif os.getenv('AZURE_REDIS_HOST'):
        os.environ['REDIS_HOST'] = os.getenv('AZURE_REDIS_HOST')
        os.environ['REDIS_PASSWORD'] = os.getenv('AZURE_REDIS_KEY')


# Initialize Redis connection (with connection pooling)
redis_client = None

def get_redis_client() -> Optional[redis.Redis]:
    # Call this function during startup
    setup_redis_connection()
    global redis_client
    if redis_client:
        return redis_client
    try:
        logger.info(f"Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}")
        # Try to connect to Redis
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            db=REDIS_DB,
            decode_responses=True,  # Automatically decode responses to strings
            socket_timeout=5,  # 5 second timeout
            socket_connect_timeout=5
        )
        # Test the connection
        redis_client.ping()
        logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    except (redis.ConnectionError, redis.TimeoutError) as e:
        logger.warning(f"Redis connection failed: {e}. Falling back to in-memory cache only.")
        redis_client = None

    return redis_client

