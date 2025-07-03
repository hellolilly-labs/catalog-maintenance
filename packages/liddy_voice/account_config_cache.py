"""
Account Configuration Cache

Redis-based caching for account configurations with TTL support.
"""

import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class AccountConfigCache:
    """Redis-based cache for account configurations"""
    
    def __init__(self, redis_client=None, key_prefix="account_config"):
        """
        Initialize the cache
        
        Args:
            redis_client: Redis client instance. If None, will get from redis_client module
            key_prefix: Prefix for cache keys
        """
        self.key_prefix = key_prefix
        
        if redis_client:
            self.redis = redis_client
        else:
            try:
                from redis_client import get_redis_client
                self.redis = get_redis_client()
            except ImportError as e:
                logger.warning(f"Redis client not available: {e}")
                self.redis = None
    
    def get_config(self, account: str) -> Optional[Dict[str, Any]]:
        """
        Get account config from cache
        
        Args:
            account: Account domain name
            
        Returns:
            Account configuration dict or None if not found/error
        """
        if not self.redis:
            return None
            
        try:
            cache_key = f"{self.key_prefix}:{account}"
            cached_data = self.redis.get(cache_key)
            if cached_data:
                logger.debug(f"Cache hit for account: {account}")
                return json.loads(cached_data)
            else:
                logger.debug(f"Cache miss for account: {account}")
                return None
        except Exception as e:
            logger.warning(f"Error getting cached config for {account}: {e}")
            return None
    
    def set_config(self, account: str, config: Dict[str, Any], ttl: int = 300):
        """
        Set account config in cache with TTL
        
        Args:
            account: Account domain name
            config: Configuration dictionary to cache
            ttl: Time-to-live in seconds (default: 5 minutes)
        """
        if not self.redis:
            return
            
        try:
            cache_key = f"{self.key_prefix}:{account}"
            self.redis.setex(
                cache_key, 
                ttl, 
                json.dumps(config)
            )
            logger.debug(f"Cached config for {account} (TTL: {ttl}s)")
        except Exception as e:
            logger.warning(f"Error caching config for {account}: {e}")
    
    def invalidate_config(self, account: str):
        """
        Remove account config from cache
        
        Args:
            account: Account domain name
        """
        if not self.redis:
            return
            
        try:
            cache_key = f"{self.key_prefix}:{account}"
            self.redis.delete(cache_key)
            logger.info(f"Invalidated cache for account: {account}")
        except Exception as e:
            logger.warning(f"Error invalidating cache for {account}: {e}")
    
    def clear_all(self):
        """Clear all account configuration cache entries"""
        if not self.redis:
            return
            
        try:
            pattern = f"{self.key_prefix}:*"
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)
                logger.info(f"Cleared {len(keys)} cached account configs")
        except Exception as e:
            logger.warning(f"Error clearing cache: {e}")
    
    def is_available(self) -> bool:
        """Check if Redis cache is available"""
        return self.redis is not None 