"""
Simple LLM Factory

Direct, simple factory for creating LLM services based on model names.
Follows KISS principle - no complex routing or registration needed.

Usage:
    service = LLMFactory.get_service("openai/o3")
    response = await service.chat_completion(...)
"""

import os
import logging
from typing import Dict, Any, List, Optional
import asyncio

from .base import LLMModelService
from .openai_service import OpenAIService
from .errors import ModelNotFoundError

logger = logging.getLogger(__name__)

# Optional imports with graceful fallbacks
try:
    from .anthropic_service import AnthropicService
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from .gemini_service import GeminiService
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class LLMFactory:
    """Simple factory for creating LLM services"""
    
    # Default model routing (configurable via environment)
    DEFAULT_MODELS = {
        'descriptor_generation': 'openai/o3',
        'sizing_analysis': 'openai/o3', 
        'brand_research': 'openai/o3',
        'foundation': 'openai/o3',
        'market': 'openai/o3',
        'product': 'openai/o3',
        'customer': 'openai/o3',
        'voice': 'openai/o3',
        'interview': 'openai/o3',
        'synthesis': 'openai/o3',
        'quality_evaluation': 'openai/o3',
        'summarization': 'openai/o3',
        'conversation': 'openai/gpt-4.1',
        'default': 'openai/o3'
    }
    
    @staticmethod
    def get_service(model_name: str) -> LLMModelService:
        """
        Get an LLM service instance for the specified model.
        
        Args:
            model_name: Model name in format "provider/model" (e.g., "openai/o3")
            
        Returns:
            LLMModelService instance
            
        Raises:
            ModelNotFoundError: If provider not supported or API key missing
        """
        if model_name.startswith('openai/'):
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ModelNotFoundError(f"OPENAI_API_KEY not set for model {model_name}")
            return OpenAIService(api_key=api_key)
            
        elif model_name.startswith('anthropic/'):
            if not ANTHROPIC_AVAILABLE:
                raise ModelNotFoundError(f"Anthropic service not available. Install with: pip install anthropic")
            api_key = os.getenv('ANTHROPIC_API_KEY') 
            if not api_key:
                raise ModelNotFoundError(f"ANTHROPIC_API_KEY not set for model {model_name}")
            return AnthropicService(api_key=api_key)
            
        elif model_name.startswith('gemini/'):
            if not GEMINI_AVAILABLE:
                raise ModelNotFoundError(f"Gemini service not available. Install with: pip install google-generativeai")
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ModelNotFoundError(f"GEMINI_API_KEY not set for model {model_name}")
            return GeminiService(api_key=api_key)
            
        else:
            raise ModelNotFoundError(f"Unsupported model format: {model_name}. Use 'provider/model' format.")
    
    @staticmethod
    def get_model_for_task(task: str) -> str:
        """
        Get the appropriate model for a task (configurable via environment).
        
        Args:
            task: Task name (e.g., 'descriptor_generation', 'sizing_analysis')
            
        Returns:
            Model name in "provider/model" format
        """
        # Check environment variables first (allows runtime configuration)
        env_var_map = {
            'sizing_analysis': 'SIZING_MODEL',
            'brand_research': 'BRAND_MODEL', 
            'quality_evaluation': 'QUALITY_MODEL',
            'descriptor_generation': 'DESCRIPTOR_MODEL',
            'conversation': 'CONVERSATION_MODEL'
        }
        
        if task in env_var_map:
            env_model = os.getenv(env_var_map[task])
            if env_model:
                return env_model
        
        # Fall back to defaults
        return LLMFactory.DEFAULT_MODELS.get(task, LLMFactory.DEFAULT_MODELS['default'])
    
    @staticmethod
    async def chat_completion(task: str = None, model: str = None, 
                            system: str = None, messages: List[Dict[str, str]] = None,
                            **kwargs) -> Dict[str, Any]:
        """
        Convenient method to get service and make chat completion in one call.
        
        Args:
            task: Task type for model selection (optional if model specified)
            model: Specific model to use (overrides task-based selection)
            system: System message
            messages: Conversation messages
            **kwargs: Additional parameters
            
        Returns:
            Response from the LLM service
        """
        # Determine model to use
        if model:
            selected_model = model
        elif task:
            selected_model = LLMFactory.get_model_for_task(task)
        else:
            selected_model = LLMFactory.DEFAULT_MODELS['default']
        
        # Get service and make request
        service = LLMFactory.get_service(selected_model)
        
        # Extract just the model name (remove provider prefix)
        model_name = selected_model.split('/', 1)[1] if '/' in selected_model else selected_model
        
        return await service.chat_completion(
            system=system,
            messages=messages, 
            model=model_name,
            **kwargs
        )
    
    @staticmethod
    def list_available_providers() -> List[str]:
        """List available LLM providers"""
        providers = ['openai']  # Always available
        
        if ANTHROPIC_AVAILABLE and os.getenv('ANTHROPIC_API_KEY'):
            providers.append('anthropic')
            
        if GEMINI_AVAILABLE and os.getenv('GEMINI_API_KEY'):
            providers.append('gemini')
            
        return providers
    
    @staticmethod  
    def get_configuration() -> Dict[str, Any]:
        """Get current factory configuration for debugging"""
        return {
            'available_providers': LLMFactory.list_available_providers(),
            'task_models': {
                task: LLMFactory.get_model_for_task(task) 
                for task in LLMFactory.DEFAULT_MODELS.keys()
            },
            'environment_overrides': {
                'SIZING_MODEL': os.getenv('SIZING_MODEL'),
                'BRAND_MODEL': os.getenv('BRAND_MODEL'),
                'QUALITY_MODEL': os.getenv('QUALITY_MODEL'),
                'DESCRIPTOR_MODEL': os.getenv('DESCRIPTOR_MODEL'),
                'CONVERSATION_MODEL': os.getenv('CONVERSATION_MODEL')
            }
        } 

# Example usage
if __name__ == "__main__":
    service = LLMFactory.get_service("openai/o3")
    result = asyncio.run(service.complete_chat([
        {"role": "user", "content": "Hello, world!"}
    ]))
    print(result)


def get_service(model_name: str):
    """
    Get LLM service for the specified model.
    
    Args:
        model_name: Model name in format "provider/model" (e.g., "openai/o3")
        
    Returns:
        LLM service instance
    """
    return LLMFactory.get_service(model_name) 