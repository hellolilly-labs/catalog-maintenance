"""
Configuration package for catalog maintenance system.

Provides centralized configuration management with environment-aware settings,
secure API key handling, and performance tuning parameters.

Usage:
    from configs import settings
    
    # Access configuration
    api_key = settings.OPENAI_API_KEY
    bucket = settings.current_bucket
    
    # Get provider-specific config
    openai_config = settings.get_llm_config("openai")
"""

from .settings import settings, get_settings, reload_settings, create_test_settings

__all__ = [
    "settings",
    "get_settings", 
    "reload_settings",
    "create_test_settings"
] 