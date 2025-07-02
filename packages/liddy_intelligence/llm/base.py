"""
Base LLM Service Interface

Provides the standardized interface for all LLM providers with:
- chat_completion method following documented patterns
- Error handling with exponential backoff
- Token counting and conversation truncation
- Retry logic for reliable operations
"""

import asyncio
import logging
import tiktoken
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from liddy_intelligence.llm.errors import LLMError, RateLimitError, TokenLimitError, NetworkError

logger = logging.getLogger(__name__)


class LLMModelService(ABC):
    """
    Base class for all LLM service providers.
    
    Provides standardized interface following COPILOT_NOTES.md patterns:
    - chat_completion method with system/messages/model parameters
    - Error handling with exponential backoff 
    - Token counting and conversation truncation
    - Retry logic for reliability
    """
    
    def __init__(self, provider_name: str, default_model: str = None):
        self.provider_name = provider_name
        self.default_model = default_model
        self._tokenizer = None
        
    @abstractmethod
    async def chat_completion(self, system: str = None, messages: List[Dict[str, str]] = None,
                            model: str = None, **kwargs) -> Dict[str, Any]:
        """
        Standard chat completion interface for all providers.
        
        Args:
            system: System message/prompt
            messages: List of conversation messages
            model: Model name to use (falls back to default_model)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Dict containing response with standardized keys:
            - content: Response text
            - usage: Token usage information
            - model: Model used
            - finish_reason: Completion reason
            
        Raises:
            LLMError: For any LLM service errors
            RateLimitError: When rate limits exceeded
            TokenLimitError: When token limits exceeded
        """
        pass
    
    @staticmethod
    def count_tokens(text: str, model: str = "o3") -> int:
        """
        Count tokens in text using tiktoken.
        
        Args:
            text: Text to count tokens for
            model: Model to use for tokenization (defaults to o3)
            
        Returns:
            Number of tokens
        """
        try:
            # Map newer models to their tokenizer equivalents
            model_mapping = {
                "gpt-4.1": "gpt-4",
                "gpt-4.1-mini": "gpt-4",
                "o3": "gpt-4",
                "o3-mini": "gpt-4",
                "o4-mini": "gpt-4"
            }
            
            # Get the mapped model or use original
            tokenizer_model = model_mapping.get(model, model)
            
            # Try to get model-specific encoding
            try:
                encoding = tiktoken.encoding_for_model(tokenizer_model)
            except KeyError:
                # Fallback to cl100k_base encoding (used by GPT-4 family)
                encoding = tiktoken.get_encoding("cl100k_base")
                
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed for model {model}: {e}")
            # Rough fallback estimation (4 chars per token)
            return len(text) // 4
    
    def count_conversation_tokens(self, system: str = None, messages: List[Dict[str, str]] = None, 
                                model: str = None) -> int:
        """
        Count total tokens in a conversation including system message.
        
        Args:
            system: System message
            messages: Conversation messages
            model: Model to use for tokenization
            
        Returns:
            Total token count
        """
        total_tokens = 0
        model_name = model or self.default_model or "o3"
        
        # Count system message tokens
        if system:
            total_tokens += self.count_tokens(system, model_name)
            
        # Count message tokens
        if messages:
            for message in messages:
                content = message.get('content', '')
                total_tokens += self.count_tokens(content, model_name)
                # Add overhead for message formatting (role, etc.)
                total_tokens += 4
                
        return total_tokens
    
    def truncate_conversation(self, system: str = None, messages: List[Dict[str, str]] = None,
                            max_tokens: int = 8000, model: str = None) -> tuple[str, List[Dict[str, str]]]:
        """
        Truncate conversation to fit within token limits.
        
        Keeps system message and most recent messages that fit within limit.
        
        Args:
            system: System message (always preserved)
            messages: Conversation messages
            max_tokens: Maximum token limit
            model: Model for token counting
            
        Returns:
            Tuple of (system, truncated_messages)
        """
        if not messages:
            return system, []
            
        model_name = model or self.default_model or "o3"
        
        # Always preserve system message
        system_tokens = self.count_tokens(system or "", model_name)
        available_tokens = max_tokens - system_tokens - 100  # Reserve 100 tokens for safety
        
        # Work backwards from most recent messages
        truncated_messages = []
        current_tokens = 0
        
        for message in reversed(messages):
            message_tokens = self.count_tokens(message.get('content', ''), model_name) + 4
            
            if current_tokens + message_tokens <= available_tokens:
                truncated_messages.insert(0, message)
                current_tokens += message_tokens
            else:
                # If we can't fit this message, we're done
                break
                
        logger.info(f"Truncated conversation: {len(messages)} -> {len(truncated_messages)} messages, "
                   f"{current_tokens + system_tokens} tokens")
        
        return system, truncated_messages
    
    async def retry_with_backoff(self, operation, max_retries: int = 3, 
                               base_delay: float = 1.0) -> Any:
        """
        Execute operation with exponential backoff retry logic.
        
        Args:
            operation: Async operation to retry
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds (will be exponentially increased)
            
        Returns:
            Result of successful operation
            
        Raises:
            LLMError: If all retries exhausted
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                return await operation()
            except RateLimitError as e:
                last_error = e
                if attempt >= max_retries:
                    break
                    
                # Use retry_after if provided, otherwise exponential backoff
                delay = e.retry_after or (base_delay * (2 ** attempt))
                logger.warning(f"Rate limited, retrying in {delay}s (attempt {attempt + 1}/{max_retries + 1})")
                await asyncio.sleep(delay)
                
            except NetworkError as e:
                last_error = e
                if attempt >= max_retries:
                    break
                    
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Network error, retrying in {delay}s (attempt {attempt + 1}/{max_retries + 1})")
                await asyncio.sleep(delay)
                
            except LLMError as e:
                # Don't retry other LLM errors (auth, invalid request, etc.)
                raise e
        
        # All retries exhausted
        raise last_error or LLMError(f"Operation failed after {max_retries + 1} attempts")
    
    def validate_messages(self, messages: List[Dict[str, str]]) -> None:
        """
        Validate message format and content.
        
        Args:
            messages: List of messages to validate
            
        Raises:
            ValueError: If messages are invalid
        """
        if not isinstance(messages, list):
            raise ValueError("Messages must be a list")
            
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(f"Message {i} must be a dictionary")
                
            if 'role' not in message:
                raise ValueError(f"Message {i} missing required 'role' field")
                
            if message['role'] not in ['user', 'assistant', 'system', 'tool']:
                raise ValueError(f"Message {i} has invalid role: {message['role']}")
                
            # Tool messages require tool_call_id instead of content being mandatory
            if message['role'] == 'tool' and 'tool_call_id' not in message:
                raise ValueError(f"Tool message {i} missing required 'tool_call_id' field")
            elif message['role'] != 'tool' and 'content' not in message:
                raise ValueError(f"Message {i} missing required 'content' field")
    
    def format_response(self, raw_response: Any, model: str = None) -> Dict[str, Any]:
        """
        Format provider-specific response into standardized format.
        
        Args:
            raw_response: Provider-specific response
            model: Model used for the request
            
        Returns:
            Standardized response dictionary with keys:
            - content: Response text
            - usage: Token usage info
            - model: Model used
            - finish_reason: Completion reason
        """
        # Default implementation - providers should override
        return {
            'content': str(raw_response),
            'usage': {'total_tokens': 0},
            'model': model or self.default_model,
            'finish_reason': 'stop'
        } 