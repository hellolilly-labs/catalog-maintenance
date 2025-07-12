"""
Account Configuration Loader

Orchestrates loading account configurations from storage with Redis caching support.
Provides fallback to hardcoded defaults for graceful degradation.
"""

import logging
from typing import Optional, Dict, Any, List

from liddy.storage import get_account_storage_provider, AccountStorageProvider
from .account_config_cache import AccountConfigCache

logger = logging.getLogger(__name__)

class AccountConfigLoader:
    """Loads account configurations with caching and fallback support"""
    
    def __init__(self, storage_provider: AccountStorageProvider = None, cache: AccountConfigCache = None):
        """
        Initialize the configuration loader
        
        Args:
            storage_provider: Account storage provider (auto-detected if None)
            cache: Redis cache instance (auto-created if None)
        """
        self._storage_provider = storage_provider or None
        self.cache = cache or AccountConfigCache()
        self.cached_accounts = {}
        self.fallback_configs = {}  # Loaded from hardcoded defaults
    
    async def get_storage_provider(self) -> AccountStorageProvider:
        if self._storage_provider is None:
            self._storage_provider = get_account_storage_provider()
        return self._storage_provider
    
    async def get_accounts(self) -> List[str]:
        """Get all accounts from storage provider"""
        storage_provider = await self.get_storage_provider()
        if storage_provider:
            return await storage_provider.get_accounts()
        else:
            # Return empty list if storage provider is skipped
            logger.warning("Storage provider not available, returning empty accounts list")
            return []
    
    async def get_account_config(self, account: str) -> Optional[Dict[str, Any]]:
        """
        Get account configuration with caching and fallback support
        
        Loading priority:
        1. Redis cache (if available)
        2. Storage provider (GCP/Local)
        3. Hardcoded fallback (if registered)
        
        Args:
            account: Account domain name (e.g., "specialized.com")
            
        Returns:
            Account configuration dictionary or None
        """
        
        if account in self.cached_accounts:
            logger.info(f"Using cached config for {account}")
            return self.cached_accounts[account]
        
        # Try cache first
        try:
            logger.info(f"Loading configuration for account: {account}")
            cached_config = await self.cache.get_config_async(account)
            if cached_config:
                logger.info(f"Loaded {account} config from cache")
                self.cached_accounts[account] = cached_config
                return cached_config
        except Exception as e:
            logger.warning(f"Cache error for {account}: {e}")
        
        # Try storage provider
        storage_provider = await self.get_storage_provider()
        if storage_provider:
            try:
                config = await storage_provider.get_account_config(account)
                if config:
                    self.cached_accounts[account] = config
                    # Cache the config for future requests
                    try:
                        await self.cache.set_config_async(account, config)
                    except Exception as e:
                        logger.warning(f"Failed to cache config for {account}: {e}")
                    
                    logger.info(f"Loaded {account} config from storage")
                    return config
            except Exception as e:
                logger.error(f"Storage error for {account}: {e}")
        else:
            logger.debug(f"Storage provider skipped for {account}")
        
        # Fallback to hardcoded config
        if account in self.fallback_configs:
            logger.warning(f"Using fallback config for {account}")
            self.cached_accounts[account] = self.fallback_configs[account]
            return self.fallback_configs[account]
        
        logger.error(f"No configuration found for account: {account}")
        return None
    
    def register_fallback_config(self, account: str, config: Dict[str, Any]):
        """
        Register a hardcoded fallback configuration
        
        Args:
            account: Account domain name
            config: Configuration dictionary
        """
        self.fallback_configs[account] = config
        logger.info(f"Registered fallback config for {account}")
    
    async def save_account_config(self, account: str, config: Dict[str, Any]) -> bool:
        """
        Save account configuration and invalidate cache
        
        Args:
            account: Account domain name
            config: Configuration dictionary to save
            
        Returns:
            Success status
        """
        try:
            storage_provider = await self.get_storage_provider()
            if storage_provider:
                # Save to storage
                success = await storage_provider.save_account_config(account, config)
                
                if success:
                    # Invalidate cache so next request loads fresh data
                    await self.cache.invalidate_config_async(account)
                    logger.info(f"Saved and invalidated cache for {account}")
                
                return success
            else:
                logger.warning(f"Storage provider not available, cannot save config for {account}")
                return False
        except Exception as e:
            logger.error(f"Error saving config for {account}: {e}")
            return False
    
    async def reload_config(self, account: str) -> Optional[Dict[str, Any]]:
        """
        Force reload configuration from storage (bypass cache)
        
        Args:
            account: Account domain name
            
        Returns:
            Reloaded configuration or None
        """
        logger.info(f"Force reloading config for {account}")
        
        # Invalidate cache first
        self.cache.invalidate_config(account)
        
        # Load fresh from storage
        return await self.get_account_config(account)
    
    async def validate_config(self, account: str, config: Dict[str, Any]) -> bool:
        """
        Validate account configuration structure
        
        Args:
            account: Account domain name
            config: Configuration to validate
            
        Returns:
            Validation success
        """
        try:
            # Basic required fields check
            required_fields = [
                'account', 'agent', 'branding', 'security'
            ]
            
            for field in required_fields:
                if field not in config:
                    logger.error(f"Missing required field '{field}' in config for {account}")
                    return False
            
            # Agent configuration checks
            agent_config = config.get('agent', {})
            if 'persona' not in agent_config:
                logger.error(f"Missing agent.persona in config for {account}")
                return False
            
            # Voice configuration checks
            if 'voice' in agent_config:
                voice_config = agent_config['voice']
                if voice_config.get('provider') == 'elevenlabs':
                    if not voice_config.get('voice_id'):
                        logger.error(f"Missing voice_id for ElevenLabs config in {account}")
                        return False
            
            logger.info(f"Configuration validation passed for {account}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating config for {account}: {e}")
            return False
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get cache status information"""
        return {
            'cache_available': self.cache.is_available(),
            'fallback_configs': list(self.fallback_configs.keys()),
            'storage_provider': type(self._storage_provider).__name__ if self._storage_provider else 'None'
        }

# Global instance for use throughout the application
_account_config_loader = None

def get_account_config_loader() -> AccountConfigLoader:
    """Get global account configuration loader instance"""
    global _account_config_loader
    if _account_config_loader is None:
        _account_config_loader = AccountConfigLoader()
    return _account_config_loader 