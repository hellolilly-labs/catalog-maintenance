"""
Search Service Compatibility Module

This module provides backward compatibility for code expecting the old search_service module.
New code should use voice_search_wrapper instead.
"""

import warnings
from .voice_search_wrapper import VoiceSearchService as SearchService, VoiceSearchWrapper

warnings.warn(
    "liddy_voice.search_service is deprecated. Use liddy_voice.voice_search_wrapper instead.",
    DeprecationWarning,
    stacklevel=2
)

# For backward compatibility
VoiceOptimizedSearchService = SearchService

__all__ = ['SearchService', 'VoiceOptimizedSearchService', 'VoiceSearchWrapper']