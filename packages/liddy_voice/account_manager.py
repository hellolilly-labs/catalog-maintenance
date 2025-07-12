"""
Account Manager for RAG Configuration - Backward Compatibility Layer

This module provides backward compatibility for the legacy voice AccountManager.
All functionality has been merged into liddy.AccountManager.

DEPRECATED: Use liddy.account_manager instead.
"""

import os
import logging
from typing import Tuple, Dict, Any, Optional, List
import asyncio
import warnings

from liddy import (
    AccountManager as BaseAccountManager,
    get_account_manager as base_get_account_manager,
    clear_account_managers as base_clear_account_managers,
    TtsProviderSettings,
    ElevenLabsTtsProviderSettings,
    AccountTTSSettings,
    AccountVariantType,
)
from liddy.account_config_loader import get_account_config_loader

logger = logging.getLogger(__name__)


class AccountManager(BaseAccountManager):
    """
    Manager for account-specific configurations
    Supports dynamic loading from GCP storage with fallback to hardcoded values
    """
    _cached_account_managers = {}
    @staticmethod
    def get_account_manager(account: str = None):
        if account in AccountManager._cached_account_managers:
            return AccountManager._cached_account_managers[account]
        else:
            account_manager = AccountManager(account)
            AccountManager._cached_account_managers[account] = account_manager
            return account_manager
    
    def __init__(self, account: str = None):
        """
        DEPRECATED: Initialize the account manager.
        
        This is a backward compatibility wrapper. Use liddy.AccountManager instead.
        
        Args:
            account: Account name (domain) to use for configuration
        """
        warnings.warn(
            "liddy_voice.AccountManager is deprecated. Use liddy.AccountManager instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(account)
        
        # Legacy attributes for compatibility
        self.config_loader = get_account_config_loader()
        self.dynamic_config = None
        self.tts_settings = None
    
    async def _ensure_voice_loaded(self):
        """
        Ensure voice settings are loaded for backward compatibility.
        """
        if self.tts_settings is None:
            self.tts_settings = await self.get_tts_settings()
            
        if self.dynamic_config is None:
            # For backward compatibility with code that accesses dynamic_config directly
            await self._load_account_config()
            self.dynamic_config = self._account_config
    
    def get_tts_settings(self) -> AccountTTSSettings:
        """
        Get TTS settings synchronously (legacy compatibility).
        
        DEPRECATED: Use async get_tts_settings() instead.
        """
        warnings.warn(
            "Synchronous get_tts_settings() is deprecated. Use async version instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # Try to return cached value or hardcoded settings
        if self.tts_settings:
            return self.tts_settings
        return self._get_hardcoded_tts_settings() or AccountTTSSettings([
            ElevenLabsTtsProviderSettings(
                voice_id="8sGzMkj2HZn6rYwGx6G0",
                voice_name="Tomas"
            ),
            TtsProviderSettings(
                voice_provider="google",
                voice_id="en-US-Chirp3-HD-Orus",
                voice_name="Spence"
            )
        ])
    
    def get_variant_types(self) -> List[AccountVariantType]:
        """
        Get variant types (legacy compatibility).
        
        DEPRECATED: This method shadows the parent implementation.
        """
        return self.get_voice_variant_types()
    
    def get_rag_details(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Get RAG details (legacy compatibility).
        
        Returns tuple for backward compatibility with old signature.
        """
        # Delegate to parent and ensure we return a tuple
        result = super().get_rag_details()
        if isinstance(result, tuple):
            return result
        # If parent returns something else, convert to expected format
        return (result, "llama-text-embed-v2") if result else ("", "llama-text-embed-v2")
    
    # Legacy method names for backward compatibility
    def get_account(self) -> str:
        """DEPRECATED: Use self.account property instead."""
        return self.account
    
    def set_account(self, account: str) -> str:
        """DEPRECATED: Account should not be changed after initialization."""
        warnings.warn(
            "set_account() is deprecated and has no effect.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.account
    


# =============================================================================
# SINGLETON FACTORY FUNCTIONS - Delegate to liddy.account_manager
# =============================================================================

async def get_account_manager(account: str = None) -> AccountManager:
    """
    DEPRECATED: Get singleton AccountManager for account.
    
    This function now delegates to liddy.get_account_manager().
    
    Args:
        account: Account name (domain) to use for configuration
        
    Returns:
        AccountManager instance (singleton for this account)
    """
    warnings.warn(
        "liddy_voice.get_account_manager() is deprecated. Use liddy.get_account_manager() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Use the base implementation which returns the correct type
    return await base_get_account_manager(account)


def clear_account_managers():
    """DEPRECATED: Clear all AccountManager instances from memory"""
    warnings.warn(
        "liddy_voice.clear_account_managers() is deprecated. Use liddy.clear_account_managers() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    base_clear_account_managers()


# Export all the classes for backward compatibility
__all__ = [
    "AccountManager",
    "get_account_manager", 
    "clear_account_managers",
    "TtsProviderSettings",
    "ElevenLabsTtsProviderSettings",
    "AccountTTSSettings",
    "AccountVariantType",
]
