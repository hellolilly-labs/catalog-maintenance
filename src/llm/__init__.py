"""
LLM Services Package

Simple, direct LLM factory following KISS principle.
Use LLMFactory for all LLM operations - no complex routing needed.

Usage:
    # Simple direct usage
    response = await LLMFactory.chat_completion(
        task="descriptor_generation",
        system="You are a helpful assistant",
        messages=[{"role": "user", "content": "Hello"}]
    )
    
    # Or get service directly  
    service = LLMFactory.get_service("openai/o3")
    response = await service.chat_completion(...)
"""

from .base import LLMModelService
from .errors import (
    LLMError, RateLimitError, TokenLimitError, ModelNotFoundError,
    AuthenticationError, NetworkError, ServiceError
)
from .simple_factory import LLMFactory
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
    # Primary interface (recommended)
    'LLMFactory',
    
    # Base classes
    'LLMModelService',
    
    # Individual services (for direct use)
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