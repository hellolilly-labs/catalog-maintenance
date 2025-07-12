"""
Account Configuration Cache

Redis-based caching for account configurations.
Account configs are stored permanently in Redis (no expiration) and
are refreshed via the load-data command when needed.
"""

import json
import logging
import asyncio
from typing import Optional, Dict, Any, List

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
                from .redis_client import get_redis_client
                self.redis = get_redis_client()
            except ImportError as e:
                logger.warning(f"Redis client not available: {e}")
                self.redis = None
    
    async def get_accounts_async(self) -> List[str]:
        """
        Get all account names from cache asynchronously
        """
        if not self.redis:
            return []
        
        try:
            cache_key = f"{self.key_prefix}:*"
            keys = await asyncio.to_thread(self.redis.keys, cache_key)
            if keys:
                account_keys = [key.decode('utf-8').split(':')[1] for key in keys]
                # deduplicate
                account_keys = list(set(account_keys))
                return account_keys
            return []
        except Exception as e:
            logger.warning(f"Error getting accounts: {e}")
            return []
            
    
    async def get_config_async(self, account: str) -> Optional[Dict[str, Any]]:
        """
        Get account config from cache asynchronously
        
        Args:
            account: Account domain name
            
        Returns:
            Account configuration dict or None if not found/error
        """
        if not self.redis:
            return None
            
        try:
            cache_key = f"{self.key_prefix}:{account}"
            # Run synchronous Redis operation in thread pool to avoid blocking
            cached_data = await asyncio.to_thread(self.redis.get, cache_key)
            if cached_data:
                logger.debug(f"Cache hit for account: {account}")
                account_config = json.loads(cached_data)
                if "agents" in account_config and isinstance(account_config["agents"][0], str):
                    raise ValueError(f"Invalid account config for {account}: agents must be a list of dicts")
                elif "agent" in account_config and isinstance(account_config["agent"], str):
                    raise ValueError(f"Invalid account config for {account}: agent must be a dict")
                return account_config
            else:
                logger.debug(f"Cache miss for account: {account}")
                return None
        except Exception as e:
            logger.warning(f"Error getting cached config for {account}: {e}")
            return None
    
    def get_config(self, account: str) -> Optional[Dict[str, Any]]:
        """
        Get account config from cache (synchronous version for backward compatibility)
        
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
                account_config = json.loads(cached_data)
                if "agents" in account_config and isinstance(account_config["agents"][0], str):
                    raise ValueError(f"Invalid account config for {account}: agents must be a list of dicts")
                elif "agent" in account_config and isinstance(account_config["agent"], str):
                    raise ValueError(f"Invalid account config for {account}: agent must be a dict")
                return account_config
            else:
                logger.debug(f"Cache miss for account: {account}")
                return None
        except Exception as e:
            logger.warning(f"Error getting cached config for {account}: {e}")
            return None
    
    async def set_config_async(self, account: str, config: Dict[str, Any], ttl: int = None):
        """
        Set account config in cache permanently (no expiration)
        
        Args:
            account: Account domain name
            config: Configuration dictionary to cache
            ttl: Deprecated parameter, kept for backwards compatibility
        """
        if not self.redis:
            return
            
        try:
            cache_key = f"{self.key_prefix}:{account}"
            
            if "agents" in config and isinstance(config["agents"][0], str):
                raise ValueError(f"Invalid account config for {account}: agents must be a list of dicts")
            elif "agent" in config and isinstance(config["agent"], str):
                raise ValueError(f"Invalid account config for {account}: agent must be a dict")
            
            # Set without expiration - accounts should persist
            await asyncio.to_thread(
                self.redis.set,
                cache_key, 
                json.dumps(config)
            )
            logger.debug(f"Cached config for {account} (permanent)")
        except Exception as e:
            logger.warning(f"Error caching config for {account}: {e}")
    
    def set_config(self, account: str, config: Dict[str, Any], ttl: int = None):
        """
        Set account config in cache permanently (no expiration)
        
        Args:
            account: Account domain name
            config: Configuration dictionary to cache
            ttl: Deprecated parameter, kept for backwards compatibility
        """
        if not self.redis:
            return
            
        try:
            cache_key = f"{self.key_prefix}:{account}"
            
            if "agents" in config and isinstance(config["agents"][0], str):
                raise ValueError(f"Invalid account config for {account}: agents must be a list of dicts")
            elif "agent" in config and isinstance(config["agent"], str):
                raise ValueError(f"Invalid account config for {account}: agent must be a dict")
            
            # Set without expiration - accounts should persist
            self.redis.set(
                cache_key, 
                json.dumps(config)
            )
            logger.debug(f"Cached config for {account} (permanent)")
        except Exception as e:
            logger.warning(f"Error caching config for {account}: {e}")
    
    async def invalidate_config_async(self, account: str):
        """
        Remove account config from cache asynchronously
        
        Args:
            account: Account domain name
        """
        if not self.redis:
            return
            
        try:
            cache_key = f"{self.key_prefix}:{account}"
            # Run synchronous Redis operation in thread pool to avoid blocking
            await asyncio.to_thread(self.redis.delete, cache_key)
            logger.debug(f"Invalidated cache for {account}")
        except Exception as e:
            logger.warning(f"Error invalidating cache for {account}: {e}")
    
    def invalidate_config(self, account: str):
        """
        Remove account config from cache (synchronous version for backward compatibility)
        
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