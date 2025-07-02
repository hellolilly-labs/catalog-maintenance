"""
OpenAI Service Implementation

Provides OpenAI GPT model integration following the standardized LLMModelService interface.
Supports GPT-4.1 and o3 models with comprehensive error handling,
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
from configs import settings

logger = logging.getLogger(__name__)


class OpenAIService(LLMModelService):
    """
    OpenAI service implementation following LLMModelService interface.
    
    Supports models:
    - gpt-4o: Latest optimized model
    - gpt-4.1: Latest optimized model
    - o3: Advanced reasoning model optimized for research and analysis
    """
    
    # Supported models with their characteristics
    SUPPORTED_MODELS = {
        'gpt-4o': {
            'max_tokens': 16384,
            'context_window': 128000,
            'description': 'Latest optimized model',
            'uses_max_completion_tokens': False
        },
        'gpt-4.1': {
            'max_tokens': 32768,
            'context_window': 1047576,
            'description': 'Latest optimized model',
            'uses_max_completion_tokens': False,
            'speed': 'medium'
        },
        'gpt-4.1-mini': {
            'max_tokens': 32768,
            'context_window': 1047576,
            'description': 'Smaller optimized model',
            'uses_max_completion_tokens': False,
            'speed': 'fast'
        },
        'o3': {
            'max_tokens': 100000,
            'context_window': 200000,
            'description': 'Advanced reasoning model optimized for research and analysis',
            'uses_max_completion_tokens': True,
            'temperature_fixed': 1.0,
            'supported_params': ['model', 'messages', 'max_completion_tokens'],
            'speed': 'slow'
        },
        'o3-mini': {
            'max_tokens': 100000,
            'context_window': 200000,
            'description': 'Advanced reasoning model optimized for research and analysis',
            'uses_max_completion_tokens': True,
            'temperature_fixed': 1.0,
            'supported_params': ['model', 'messages', 'max_completion_tokens'],
            'speed': 'medium'
        },
        'o4-mini': {
            'max_tokens': 100000,
            'context_window': 200000,
            'description': 'Faster reasoning model optimized for research and analysis',
            'uses_max_completion_tokens': True,
            'temperature_fixed': 1.0,
            'supported_params': ['model', 'messages', 'max_completion_tokens'],
            'speed': 'medium'
        }
    }
    
    def __init__(self, api_key: str = None, default_model: str = None):
        """
        Initialize OpenAI service.
        
        Args:
            api_key: OpenAI API key (if None, uses settings.OPENAI_API_KEY)
            default_model: Default model to use (if None, uses settings.OPENAI_DEFAULT_MODEL)
        """
        # Get default model from settings if not provided
        if default_model is None:
            default_model = settings.OPENAI_DEFAULT_MODEL
            
        super().__init__(provider_name="openai", default_model=default_model)
        
        # Get API key from parameter, settings, or environment (in that order)
        self.api_key = api_key or settings.OPENAI_API_KEY or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise AuthenticationError("OpenAI API key not provided in settings or environment", provider="openai")
        
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
        
        # if model name starts with "openai/" remove the prefix
        if model_name.startswith("openai/"):
            model_name = model_name[len("openai/"):]
        
        # Validate model is supported
        if model_name not in self.SUPPORTED_MODELS:
            raise ModelNotFoundError(f"Model {model_name} not supported", 
                                   model=model_name, provider="openai")
        
        # Validate and prepare messages
        formatted_messages = self._prepare_messages(system, messages)
        
        # Check token limits and truncate if necessary
        max_context = self.SUPPORTED_MODELS[model_name]['context_window']
        max_tokens = kwargs.get('max_tokens', settings.OPENAI_MAX_TOKENS)
        available_tokens = max_context - max_tokens - 100  # Reserve space for response
        
        total_tokens = self.count_conversation_tokens(system, messages, model_name)
        if total_tokens > available_tokens:
            logger.warning(f"Conversation too long ({total_tokens} tokens), truncating to fit {available_tokens}")
            system, truncated_messages = self.truncate_conversation(
                system, messages, available_tokens, model_name
            )
            formatted_messages = self._prepare_messages(system, truncated_messages)
        
        # Prepare request parameters
        model_config = self.SUPPORTED_MODELS[model_name]
        
        request_params = {
            'model': model_name,
            'messages': formatted_messages,
        }
        
        # O3 and reasoning models have specific parameter restrictions
        if model_config.get('uses_max_completion_tokens', False):
            request_params['max_completion_tokens'] = max_tokens
            logger.debug(f"Using max_completion_tokens={max_tokens} for reasoning model {model_name}")
            
            # O3 has fixed temperature and limited parameter support
            if 'temperature_fixed' in model_config:
                # O3 only supports temperature=1.0 (default)
                logger.debug(f"Using fixed temperature={model_config['temperature_fixed']} for {model_name}")
                # Don't set temperature parameter - let it use default
            else:
                request_params['temperature'] = kwargs.get('temperature', 0.7)
                
            # Only add other parameters if supported by the model
            supported_params = model_config.get('supported_params', [])
            if 'top_p' in supported_params:
                request_params['top_p'] = kwargs.get('top_p', 1.0)
            if 'frequency_penalty' in supported_params:
                request_params['frequency_penalty'] = kwargs.get('frequency_penalty', 0.0)
            if 'presence_penalty' in supported_params:
                request_params['presence_penalty'] = kwargs.get('presence_penalty', 0.0)
        else:
            # Standard models support all parameters
            request_params.update({
                'max_tokens': max_tokens,
                'temperature': kwargs.get('temperature', 0.7),
                'top_p': kwargs.get('top_p', 1.0),
                'frequency_penalty': kwargs.get('frequency_penalty', 0.0),
                'presence_penalty': kwargs.get('presence_penalty', 0.0),
            })
        
        # Add tool support for standard models (not O3/reasoning models)
        if not model_config.get('uses_max_completion_tokens', False):
            if 'tools' in kwargs and kwargs['tools']:
                request_params['tools'] = kwargs['tools']
                if 'tool_choice' in kwargs:
                    request_params['tool_choice'] = kwargs['tool_choice']
        
        # Execute with retry logic
        async def make_request():
            return await self._make_openai_request(request_params)
        
        try:
            response = await self.retry_with_backoff(make_request, max_retries=settings.RETRY_ATTEMPTS)
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
            
            # Base response structure
            formatted_response = {
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
            
            # Add tool calls if present
            if hasattr(message, 'tool_calls') and message.tool_calls:
                formatted_response['tool_calls'] = []
                for tool_call in message.tool_calls:
                    formatted_response['tool_calls'].append({
                        'id': tool_call.id,
                        'type': tool_call.type,
                        'function': {
                            'name': tool_call.function.name,
                            'arguments': tool_call.function.arguments
                        }
                    })
            
            # Also include the raw choices structure for backward compatibility
            formatted_response['choices'] = [{
                'message': {
                    'content': message.content,
                    'role': message.role
                }
            }]
            
            # Add tool calls to choices format as well for compatibility
            if hasattr(message, 'tool_calls') and message.tool_calls:
                formatted_response['choices'][0]['message']['tool_calls'] = []
                for tool_call in message.tool_calls:
                    formatted_response['choices'][0]['message']['tool_calls'].append({
                        'id': tool_call.id,
                        'type': tool_call.type,
                        'function': {
                            'name': tool_call.function.name,
                            'arguments': tool_call.function.arguments
                        }
                    })
            
            return formatted_response
            
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
                model="gpt-4.1",
                max_tokens=10
            )
            return response is not None and 'content' in response
            
        except Exception as e:
            logger.error(f"OpenAI connection test failed: {e}")
            return False 


def create_openai_service(api_key: str = None) -> OpenAIService:
    """
    Factory function to create OpenAI service
    
    Args:
        api_key: Optional API key (uses settings if not provided)
        
    Returns:
        Configured OpenAI service
    """
    return OpenAIService(api_key=api_key)
