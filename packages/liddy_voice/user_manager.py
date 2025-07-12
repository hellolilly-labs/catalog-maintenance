"""
User Manager Module

Provides utility functions for managing user state in Redis.
"""

import logging
import os
import json
import time
from typing import Dict, Optional, Any, List

# Import shared Redis client
from liddy.redis_client import get_redis_client
from liddy.model import UserState, UrlTracking

# Configure logging
logger = logging.getLogger(__name__)

# Configuration from environment variables
REDIS_PREFIX = os.getenv('REDIS_PREFIX', '')
REDIS_TTL = int(os.getenv('REDIS_TTL', 3600))  # 1 hour default TTL
MEMORY_CACHE_TTL = int(os.getenv('MEMORY_CACHE_TTL', 600))  # 10 minute default TTL


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


#   async getUserUrlTrackings(userId: string): Promise<UrlTrackingDto[]> {
#     if (this.inMemoryMode) {
#       const key = `user_state:${userId}:recent_history`;
#       const url_trackings = this.memoryStorage[key] || [];
#       return url_trackings || [];
#     }
#     if (!this.client) {
#       this.logger.warn('Redis not available, returning empty URL tracking');
#       return [];
#     }
#     try {
#       const key = `user_state:${userId}:recent_history`;
#       const data = await this.client.get(key);
#       if (data) {
#         return JSON.parse(data);
#       }
#       return [];
#     } catch (error) {
#       this.logger.error(`Error getting user URL tracking: ${error.message}`);
#       return [];
#     }
#   }
def get_user_recent_history(user_id: str, limit: int = 10) -> List[UrlTracking]:
    """Get user's recent conversation history"""
    client = get_redis_client()
    if not client:
        return []
    
    try:
        key = f"{REDIS_PREFIX}user_state:{user_id}:recent_history"
        history = client.lrange(key, 0, limit - 1)
        return [UrlTracking.from_json(json.loads(item)) for item in history]
    except Exception as e:
        logger.error(f"Error getting user history: {e}")
        return []


class UserManager:
    def __init__(self):
        pass

    @staticmethod
    def get_user_state(user_id: str) -> Optional[UserState]:
        return get_user_state(user_id)
    
    @staticmethod
    def save_user_state(state: UserState, ttl: Optional[int] = None) -> bool:
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