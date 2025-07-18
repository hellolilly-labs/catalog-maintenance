"""
Liddy Core Platform

Shared functionality for all Liddy services including:
- Data models
- Storage providers
- Search infrastructure
- Account and product management
"""

__version__ = "0.3.0"

# Re-export key modules
from .storage import get_account_storage_provider, AccountStorageProvider
from .config import Settings, get_settings
from .models.product import Product
from .account_manager import (
    AccountManager,
    get_account_manager,
    clear_account_managers,
)
from .account_manager.models import (
    SearchConfig,
    AccountVariantType,
    TtsProviderSettings,
    ElevenLabsTtsProviderSettings,
    AccountTTSSettings,
)
from .auth_utils import setup_google_auth, get_google_credentials

# Backward compatibility aliases
Config = Settings
get_config = get_settings

__all__ = [
    "get_account_storage_provider",
    "AccountStorageProvider",
    "Settings",
    "get_settings",
    "Config",  # Backward compatibility
    "get_config",  # Backward compatibility
    "Product",
    "AccountManager",
    "get_account_manager",
    "clear_account_managers",
    "SearchConfig",
    "AccountVariantType",
    "TtsProviderSettings",
    "ElevenLabsTtsProviderSettings",
    "AccountTTSSettings",
    "setup_google_auth",
    "get_google_credentials",
]