"""
Anthropic Claude LLM Service Implementation

Provides Claude model access for superior reasoning tasks following Decision #1
in COPILOT_NOTES.md. Specialized for analytical and reasoning-heavy tasks.

Key Features:
- Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku support
- Large context windows (200K+ tokens)
- Superior reasoning capabilities
- Configuration integration
- Comprehensive error handling
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional

from .base import LLMModelService
from .errors import (
    LLMError, RateLimitError, TokenLimitError, ModelNotFoundError,
    AuthenticationError, NetworkError, ServiceError
)
from configs.settings import get_settings

logger = logging.getLogger(__name__)

# Try to import Anthropic, but make it optional for now
try:
    import anthropic
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic package not installed. Install with: pip install anthropic")


class AnthropicService(LLMModelService):
    """Anthropic Claude service implementation"""
    
    # Supported models with context windows
    SUPPORTED_MODELS = {
        "claude-3-5-sonnet-20241022": {
            "max_tokens": 200000,
            "max_output_tokens": 8192,
            "description": "Most capable model for complex reasoning"
        },
        "claude-3-opus-20240229": {
            "max_tokens": 200000,
            "max_output_tokens": 4096,
            "description": "Most powerful model for highly complex tasks"
        },
        "claude-3-sonnet-20240229": {
            "max_tokens": 200000,
            "max_output_tokens": 4096,
            "description": "Balanced model for most tasks"
        },
        "claude-3-haiku-20240307": {
            "max_tokens": 200000,
            "max_output_tokens": 4096,
            "description": "Fastest model for simpler tasks"
        },
        "claude-4-0-sonnet": {
            "max_tokens": 200000,
            "max_output_tokens": 8192,
            "description": "Most capable model for complex reasoning"
        },
        "claude-sonnet-4-20250514": {
            "max_tokens": 200000,
            "max_output_tokens": 64000,
            "description": "Most capable model for complex reasoning"
        }
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Anthropic service
        
        Args:
            api_key: Anthropic API key (optional, reads from settings if not provided)
        """
        super().__init__(provider_name="anthropic")
        
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package not installed. Install with: pip install anthropic")
        
        # Get configuration
        self.settings = get_settings()
        
        # Set API key
        self.api_key = api_key or self.settings.ANTHROPIC_API_KEY
        if not self.api_key:
            raise AuthenticationError("Anthropic API key not provided")
        
        # Initialize client
        self.client = AsyncAnthropic(api_key=self.api_key)
        
        # Default model
        self.default_model = "claude-3-5-sonnet-20241022"
        
        logger.info(f"Anthropic service initialized with default model: {self.default_model}")
    
    def list_supported_models(self) -> List[str]:
        """Return list of supported Anthropic models"""
        return list(self.SUPPORTED_MODELS.keys())
    
    def get_max_tokens(self, model: str) -> int:
        """Get maximum context tokens for a model"""
        model_info = self.SUPPORTED_MODELS.get(model)
        return model_info["max_tokens"] if model_info else 200000
    
    def get_max_output_tokens(self, model: str) -> int:
        """Get maximum output tokens for a model"""
        model_info = self.SUPPORTED_MODELS.get(model)
        return model_info["max_output_tokens"] if model_info else 4096
    
    async def chat_completion(self, system: str = None, messages: List[Dict[str, str]] = None,
                            model: str = None, max_tokens: int = None, temperature: float = 0.7,
                            **kwargs) -> Dict[str, Any]:
        """
        Generate chat completion using Anthropic Claude
        
        Args:
            system: System message/prompt
            messages: List of conversation messages
            model: Claude model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            **kwargs: Additional parameters
            
        Returns:
            Standardized response dictionary
        """
        try:
            # Validate and prepare inputs
            model = model or self.default_model
            
            # if model name starts with "anthropic/" remove the prefix
            if model.startswith("anthropic/"):
                model = model[len("anthropic/"):]
            
            if model not in self.SUPPORTED_MODELS:
                raise ModelNotFoundError(f"Model {model} not supported by Anthropic service")
            
            messages = messages or []
            if not messages:
                raise ValueError("At least one message is required")
            
            # Set max output tokens
            if max_tokens is None:
                max_tokens = min(4096, self.get_max_output_tokens(model))
            else:
                max_tokens = min(max_tokens, self.get_max_output_tokens(model))
            
            # Validate context length
            total_tokens = self._estimate_tokens(system, messages)
            max_context = self.get_max_tokens(model)
            
            if total_tokens > max_context - max_tokens:
                # Try to truncate conversation
                messages = self._truncate_conversation(messages, max_context - max_tokens - 1000)
                logger.warning(f"Truncated conversation to fit {model} context window")
            
            # Prepare API call
            api_messages = []
            
            # Add messages (Claude doesn't use system in messages array)
            for msg in messages:
                if msg["role"] in ["user", "assistant"]:
                    api_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Make API call
            response = await self.client.messages.create(
                model=model,
                system=system or "You are a helpful AI assistant.",
                messages=api_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            # Parse response
            content = response.content[0].text if response.content else ""
            
            # Calculate token usage
            input_tokens = getattr(response.usage, 'input_tokens', 0)
            output_tokens = getattr(response.usage, 'output_tokens', 0)
            
            return {
                "content": content,
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                },
                "model": model,
                "finish_reason": response.stop_reason or "stop",
                "provider": self.provider_name
            }
            
        except Exception as e:
            if ANTHROPIC_AVAILABLE:
                if isinstance(e, anthropic.RateLimitError):
                    logger.warning(f"Anthropic rate limit exceeded: {e}")
                    raise RateLimitError(f"Rate limit exceeded: {str(e)}", original_error=e)
                elif isinstance(e, anthropic.AuthenticationError):
                    logger.error(f"Anthropic authentication failed: {e}")
                    raise AuthenticationError(f"Authentication failed: {str(e)}", original_error=e)
                elif isinstance(e, anthropic.BadRequestError):
                    if "maximum context length" in str(e).lower():
                        logger.error(f"Anthropic token limit exceeded: {e}")
                        raise TokenLimitError(f"Token limit exceeded: {str(e)}", original_error=e)
                    else:
                        logger.error(f"Anthropic bad request: {e}")
                        raise ServiceError(f"Bad request: {str(e)}", original_error=e)
                elif isinstance(e, anthropic.APIError):
                    logger.error(f"Anthropic API error: {e}")
                    raise ServiceError(f"API error: {str(e)}", original_error=e)
                elif isinstance(e, anthropic.APIConnectionError):
                    logger.error(f"Anthropic connection error: {e}")
                    raise NetworkError(f"Connection error: {str(e)}", original_error=e)
            
            logger.error(f"Unexpected Anthropic error: {e}")
            raise LLMError(f"Unexpected error: {str(e)}", original_error=e)
    
    def _estimate_tokens(self, system: str, messages: List[Dict[str, str]]) -> int:
        """
        Estimate token count for Anthropic models
        Claude uses different tokenization than OpenAI, but this provides a rough estimate
        """
        text = (system or "") + " "
        for msg in messages:
            text += msg.get("content", "") + " "
        
        # Rough estimation: ~4 characters per token for Claude
        return len(text) // 4
    
    def _truncate_conversation(self, messages: List[Dict[str, str]], max_tokens: int) -> List[Dict[str, str]]:
        """
        Truncate conversation to fit within token limit
        Keeps the most recent messages
        """
        if not messages:
            return messages
        
        # Always keep the first message (usually contains important context)
        result = [messages[0]] if messages else []
        
        # Add messages from the end until we approach the limit
        current_tokens = self._estimate_tokens("", result)
        
        for msg in reversed(messages[1:]):
            msg_tokens = self._estimate_tokens("", [msg])
            if current_tokens + msg_tokens < max_tokens:
                result.insert(-1 if len(result) > 1 else 0, msg)
                current_tokens += msg_tokens
            else:
                break
        
        return result
    
    async def test_connection(self) -> bool:
        """Test connection to Anthropic API"""
        try:
            response = await self.chat_completion(
                system="Test connection",
                messages=[{"role": "user", "content": "Say 'OK' if you can hear me."}],
                max_tokens=5
            )
            return response is not None and response.get("content")
            
        except Exception as e:
            logger.error(f"Anthropic connection test failed: {e}")
            return False


def create_anthropic_service(api_key: str = None) -> AnthropicService:
    """
    Factory function to create Anthropic service
    
    Args:
        api_key: Optional API key (uses settings if not provided)
        
    Returns:
        Configured Anthropic service
    """
    return AnthropicService(api_key=api_key) 