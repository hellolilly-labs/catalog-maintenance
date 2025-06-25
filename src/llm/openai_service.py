"""
OpenAI Service Implementation

Provides OpenAI GPT model integration following the standardized LLMModelService interface.
Supports GPT-4, GPT-4-turbo, GPT-3.5-turbo models with comprehensive error handling,
token management, and retry logic.
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional
import openai
from openai import AsyncOpenAI

from .base import LLMModelService
from .errors import (
    LLMError, RateLimitError, TokenLimitError, ModelNotFoundError,
    AuthenticationError, NetworkError, InvalidRequestError, ServiceUnavailableError
)

logger = logging.getLogger(__name__)


class OpenAIService(LLMModelService):
    """
    OpenAI service implementation following LLMModelService interface.
    
    Supports models:
    - gpt-4: High-quality reasoning and analysis
    - gpt-4-turbo: Improved performance and cost efficiency
    - gpt-3.5-turbo: Fast responses for simple tasks
    - gpt-4o: Latest optimized model
    """
    
    # Supported models with their characteristics
    SUPPORTED_MODELS = {
        'gpt-4': {
            'max_tokens': 8192,
            'context_window': 8192,
            'description': 'High-quality reasoning and analysis'
        },
        'gpt-4-turbo': {
            'max_tokens': 4096,
            'context_window': 128000,
            'description': 'Improved performance and cost efficiency'
        },
        'gpt-4-turbo-preview': {
            'max_tokens': 4096,
            'context_window': 128000,
            'description': 'Preview version of GPT-4 Turbo'
        },
        'gpt-3.5-turbo': {
            'max_tokens': 4096,
            'context_window': 16385,
            'description': 'Fast responses for simple tasks'
        },
        'gpt-4o': {
            'max_tokens': 4096,
            'context_window': 128000,
            'description': 'Latest optimized model'
        }
    }
    
    def __init__(self, api_key: str = None, default_model: str = "gpt-4-turbo"):
        """
        Initialize OpenAI service.
        
        Args:
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
            default_model: Default model to use for requests
        """
        super().__init__(provider_name="openai", default_model=default_model)
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise AuthenticationError("OpenAI API key not provided", provider="openai")
        
        # Initialize async client
        self.client = AsyncOpenAI(api_key=self.api_key)
        
        logger.info(f"OpenAI service initialized with default model: {default_model}")
    
    async def chat_completion(self, system: str = None, messages: List[Dict[str, str]] = None,
                            model: str = None, **kwargs) -> Dict[str, Any]:
        """
        OpenAI chat completion implementation.
        
        Args:
            system: System message/prompt
            messages: List of conversation messages
            model: Model name (defaults to default_model)
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Standardized response dictionary
            
        Raises:
            LLMError: For any OpenAI service errors
        """
        # Use default model if not specified
        model_name = model or self.default_model
        
        # Validate model is supported
        if model_name not in self.SUPPORTED_MODELS:
            raise ModelNotFoundError(f"Model {model_name} not supported", 
                                   model=model_name, provider="openai")
        
        # Validate and prepare messages
        formatted_messages = self._prepare_messages(system, messages)
        
        # Check token limits and truncate if necessary
        max_context = self.SUPPORTED_MODELS[model_name]['context_window']
        max_tokens = kwargs.get('max_tokens', 1000)
        available_tokens = max_context - max_tokens - 100  # Reserve space for response
        
        total_tokens = self.count_conversation_tokens(system, messages, model_name)
        if total_tokens > available_tokens:
            logger.warning(f"Conversation too long ({total_tokens} tokens), truncating to fit {available_tokens}")
            system, truncated_messages = self.truncate_conversation(
                system, messages, available_tokens, model_name
            )
            formatted_messages = self._prepare_messages(system, truncated_messages)
        
        # Prepare request parameters
        request_params = {
            'model': model_name,
            'messages': formatted_messages,
            'temperature': kwargs.get('temperature', 0.7),
            'max_tokens': max_tokens,
            'top_p': kwargs.get('top_p', 1.0),
            'frequency_penalty': kwargs.get('frequency_penalty', 0.0),
            'presence_penalty': kwargs.get('presence_penalty', 0.0),
        }
        
        # Remove None values
        request_params = {k: v for k, v in request_params.items() if v is not None}
        
        # Execute with retry logic
        async def make_request():
            return await self._make_openai_request(request_params)
        
        try:
            response = await self.retry_with_backoff(make_request, max_retries=3)
            return self._format_openai_response(response, model_name)
            
        except Exception as e:
            if isinstance(e, LLMError):
                raise e
            else:
                raise LLMError(f"Unexpected error in OpenAI service: {str(e)}", 
                             original_error=e, provider="openai")
    
    async def _make_openai_request(self, request_params: Dict[str, Any]) -> Any:
        """
        Make the actual OpenAI API request with proper error handling.
        
        Args:
            request_params: Request parameters for OpenAI API
            
        Returns:
            OpenAI response object
            
        Raises:
            Various LLMError subclasses based on the error type
        """
        try:
            response = await self.client.chat.completions.create(**request_params)
            return response
            
        except openai.RateLimitError as e:
            # Extract retry-after if available
            retry_after = None
            if hasattr(e, 'response') and e.response:
                retry_after = e.response.headers.get('retry-after')
                if retry_after:
                    retry_after = int(retry_after)
            
            raise RateLimitError(
                f"OpenAI rate limit exceeded: {str(e)}", 
                retry_after=retry_after,
                original_error=e,
                provider="openai"
            )
            
        except openai.AuthenticationError as e:
            raise AuthenticationError(
                f"OpenAI authentication failed: {str(e)}", 
                original_error=e,
                provider="openai"
            )
            
        except openai.BadRequestError as e:
            # Check if it's a token limit error
            error_message = str(e).lower()
            if 'token' in error_message and ('limit' in error_message or 'maximum' in error_message):
                raise TokenLimitError(
                    f"OpenAI token limit exceeded: {str(e)}", 
                    original_error=e,
                    provider="openai"
                )
            else:
                raise InvalidRequestError(
                    f"OpenAI request invalid: {str(e)}", 
                    original_error=e,
                    provider="openai"
                )
                
        except openai.NotFoundError as e:
            raise ModelNotFoundError(
                f"OpenAI model not found: {str(e)}", 
                original_error=e,
                provider="openai"
            )
            
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            raise NetworkError(
                f"OpenAI network error: {str(e)}", 
                original_error=e,
                provider="openai"
            )
            
        except openai.InternalServerError as e:
            raise ServiceUnavailableError(
                f"OpenAI service unavailable: {str(e)}", 
                original_error=e,
                provider="openai"
            )
            
        except Exception as e:
            raise LLMError(
                f"Unexpected OpenAI error: {str(e)}", 
                original_error=e,
                provider="openai"
            )
    
    def _prepare_messages(self, system: str = None, messages: List[Dict[str, str]] = None) -> List[Dict[str, str]]:
        """
        Prepare messages in OpenAI format.
        
        Args:
            system: System message
            messages: Conversation messages
            
        Returns:
            List of messages in OpenAI format
        """
        formatted_messages = []
        
        # Add system message if provided
        if system:
            formatted_messages.append({
                'role': 'system',
                'content': system
            })
        
        # Add conversation messages
        if messages:
            # Validate messages format
            self.validate_messages(messages)
            
            for message in messages:
                formatted_messages.append({
                    'role': message['role'],
                    'content': message['content']
                })
        
        return formatted_messages
    
    def _format_openai_response(self, response: Any, model: str) -> Dict[str, Any]:
        """
        Format OpenAI response into standardized format.
        
        Args:
            response: OpenAI response object
            model: Model used for the request
            
        Returns:
            Standardized response dictionary
        """
        try:
            choice = response.choices[0]
            message = choice.message
            
            return {
                'content': message.content,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                'model': model,
                'finish_reason': choice.finish_reason,
                'provider': 'openai'
            }
            
        except (AttributeError, IndexError, KeyError) as e:
            logger.error(f"Failed to format OpenAI response: {e}")
            raise LLMError(f"Invalid response format from OpenAI: {str(e)}", 
                         original_error=e, provider="openai")
    
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model: Model name
            
        Returns:
            Model information dictionary
            
        Raises:
            ModelNotFoundError: If model not supported
        """
        if model not in self.SUPPORTED_MODELS:
            raise ModelNotFoundError(f"Model {model} not supported", 
                                   model=model, provider="openai")
        
        return self.SUPPORTED_MODELS[model].copy()
    
    def list_supported_models(self) -> List[str]:
        """
        Get list of supported model names.
        
        Returns:
            List of supported model names
        """
        return list(self.SUPPORTED_MODELS.keys())
    
    async def test_connection(self) -> bool:
        """
        Test the OpenAI connection with a simple request.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = await self.chat_completion(
                system="You are a helpful assistant.",
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-3.5-turbo",
                max_tokens=10
            )
            return response is not None and 'content' in response
            
        except Exception as e:
            logger.error(f"OpenAI connection test failed: {e}")
            return False 