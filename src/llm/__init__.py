"""
LLM Services Package

Multi-provider LLM strategy with intelligent routing following Decision #1
in COPILOT_NOTES.md:
- OpenAI: Creative writing and general purpose  
- Anthropic: Superior reasoning and analytical tasks
- Gemini: Multimodal and specialized tasks

Usage:
    from src.llm import LLMRouter, OpenAIService, AnthropicService, GeminiService
    
    # Create router with all providers
    router = create_default_router()
    
    # Or use specific services
    openai = OpenAIService()
    anthropic = AnthropicService()  # Requires anthropic package
    gemini = GeminiService()        # Requires google-generativeai package
"""

from .base import LLMModelService
from .errors import (
    LLMError, RateLimitError, TokenLimitError, ModelNotFoundError,
    AuthenticationError, NetworkError, ServiceError
)
from .router import LLMRouter, create_default_router
from .openai_service import OpenAIService, create_openai_service

# Optional imports with graceful fallbacks
try:
    from .anthropic_service import AnthropicService, create_anthropic_service
except ImportError:
    AnthropicService = None
    create_anthropic_service = None

try:
    from .gemini_service import GeminiService, create_gemini_service
except ImportError:
    GeminiService = None
    create_gemini_service = None

__all__ = [
    # Base and router
    'LLMModelService',
    'LLMRouter', 
    'create_default_router',
    
    # OpenAI (always available)
    'OpenAIService',
    'create_openai_service',
    
    # Optional services
    'AnthropicService',      # None if package not installed
    'create_anthropic_service',
    'GeminiService',         # None if package not installed
    'create_gemini_service',
    
    # Errors
    'LLMError',
    'RateLimitError', 
    'TokenLimitError',
    'ModelNotFoundError',
    'AuthenticationError',
    'NetworkError',
    'ServiceError'
] 