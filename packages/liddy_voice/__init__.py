"""
Liddy Voice Assistant Package

Real-time voice-enabled AI assistant with brand-aware customer service capabilities.
"""

__version__ = "0.3.0"

# Core exports
from .voice_search_wrapper import VoiceSearchWrapper, VoiceSearchService
from .session_state_manager import SessionStateManager

# Backward compatibility
try:
    from .search_service import VoiceOptimizedSearchService
except ImportError:
    VoiceOptimizedSearchService = VoiceSearchService

__all__ = [
    '__version__',
    'VoiceSearchWrapper',
    'VoiceSearchService',
    'VoiceOptimizedSearchService',  # Backward compatibility
    'SessionStateManager',
]