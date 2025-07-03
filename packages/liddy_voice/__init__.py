"""
Liddy Voice Assistant Package

Real-time voice-enabled AI assistant with brand-aware customer service capabilities.
"""

__version__ = "0.1.0"

# Core exports
from .search_service import VoiceOptimizedSearchService
from .session_state_manager import SessionStateManager

__all__ = [
    '__version__',
    'VoiceOptimizedSearchService',
    'SessionStateManager',
]