"""
Redis Client Module

Provides utility functions for interacting with Redis.
"""

import logging
import os
import json
import time
from typing import Dict, Optional, Any, List

# Import Redis library
import redis
from liddy.model import UserState

# Configure logging first, before any functions try to use it
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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


# User state management functions
def get_user_state(user_id: str) -> Optional[UserState]:
    """Get user state from Redis"""
    client = get_redis_client()
    if not client:
        return None
    
    try:
        key = f"{REDIS_PREFIX}user_state:{user_id}"
        data = client.get(key)
        if data:
            return UserState.from_json(json.loads(data))
    except Exception as e:
        logger.error(f"Error getting user state: {e}")
    
    return None


def save_user_state(state: UserState, ttl: Optional[int] = None) -> bool:
    """Save user state to Redis
    
    Args:
        state: UserState object to save
        ttl: Optional TTL in seconds (defaults to REDIS_TTL)
    """
    client = get_redis_client()
    if not client:
        return False
    
    try:
        key = f"{REDIS_PREFIX}user_state:{state.user_id}"
        # Convert UserState to JSON string using its to_json() method
        state_json = state.to_json()
        client.set(key, state_json, ex=ttl or REDIS_TTL)
        logger.debug(f"Saved user state for {state.user_id}")
        return True
    except Exception as e:
        logger.error(f"Error saving user state: {e}")
        return False


def update_user_room_reconnect_time(user_id: str, room_id: str, account: Optional[str] = None, 
                                   auto_reconnect_until: Optional[float] = None, 
                                   resumption_message: Optional[str] = None) -> bool:
    """Update user's room reconnect time with reconnect metadata
    
    Args:
        user_id: User identifier
        room_id: Room identifier
        account: Account/domain (optional)
        auto_reconnect_until: Timestamp until which auto-reconnect is allowed (optional)
        resumption_message: Message to show on reconnect (optional)
    """
    client = get_redis_client()
    if not client:
        return False
    
    try:
        key = f"{REDIS_PREFIX}user_room:{user_id}:{room_id}"
        
        # Get existing data if updating partially
        existing_data = {}
        if auto_reconnect_until is None or resumption_message is None:
            existing_json = client.get(key)
            if existing_json:
                existing_data = json.loads(existing_json)
        
        # Build reconnect data, using existing values as defaults
        reconnect_data = {
            "account": account or existing_data.get("account", ""),
            "auto_reconnect_until": auto_reconnect_until or existing_data.get("auto_reconnect_until", time.time() + 300),
            "resumption_message": resumption_message or existing_data.get("resumption_message", ""),
            "updated_at": time.time()
        }
        client.set(key, json.dumps(reconnect_data), ex=3600)  # 1 hour TTL
        return True
    except Exception as e:
        logger.error(f"Error updating room reconnect time: {e}")
        return False


def save_user_latest_conversation(user_id: str, conversation: Dict[str, Any]) -> bool:
    """Save user's latest conversation"""
    client = get_redis_client()
    if not client:
        return False
    
    try:
        key = f"{REDIS_PREFIX}user_conversation:{user_id}"
        client.set(key, json.dumps(conversation), ex=86400)  # 24 hour TTL
        return True
    except Exception as e:
        logger.error(f"Error saving user conversation: {e}")
        return False


def get_user_latest_conversation(user_id: str) -> Optional[Dict[str, Any]]:
    """Get user's latest conversation"""
    client = get_redis_client()
    if not client:
        return None
    
    try:
        key = f"{REDIS_PREFIX}user_conversation:{user_id}"
        data = client.get(key)
        if data:
            return json.loads(data)
    except Exception as e:
        logger.error(f"Error getting user conversation: {e}")
    
    return None


def get_user_recent_history(user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Get user's recent conversation history"""
    client = get_redis_client()
    if not client:
        return []
    
    try:
        key = f"{REDIS_PREFIX}user_history:{user_id}"
        history = client.lrange(key, 0, limit - 1)
        return [json.loads(item) for item in history]
    except Exception as e:
        logger.error(f"Error getting user history: {e}")
        return []


class UserManager:
    def __init__(self):
        pass

    @staticmethod
    def get_user_state(self, user_id: str) -> Optional[UserState]:
        return get_user_state(user_id)
    
    @staticmethod
    def save_user_state(self, state: UserState, ttl: Optional[int] = None) -> bool:
        return save_user_state(state, ttl)
    
    @staticmethod
    def update_user_room_reconnect_time(user_id: str, room_id: str, account: Optional[str] = None, 
                                   auto_reconnect_until: Optional[float] = None, 
                                   resumption_message: Optional[str] = None) -> bool:
        return update_user_room_reconnect_time(user_id, room_id, account, auto_reconnect_until, resumption_message)
    
    @staticmethod
    def save_user_latest_conversation(user_id: str, conversation: Dict[str, Any]) -> bool:
        return save_user_latest_conversation(user_id, conversation)
    
    @staticmethod
    def get_user_latest_conversation(user_id: str) -> Optional[Dict[str, Any]]:
        return get_user_latest_conversation(user_id)
    
    @staticmethod
    def get_user_recent_history(user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        return get_user_recent_history(user_id, limit)