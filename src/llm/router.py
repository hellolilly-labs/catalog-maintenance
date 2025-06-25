"""
LLM Router Implementation

Implements multi-provider LLM strategy following Decision #1 in COPILOT_NOTES.md:
- Different models excel at different tasks
- Router allows optimal model selection per use case
- Fallback strategies for reliability
"""

import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

from .base import LLMModelService
from .openai_service import OpenAIService
from .errors import LLMError, ModelNotFoundError

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Configuration for a provider."""
    service: LLMModelService
    models: List[str]
    priority: int = 0
    enabled: bool = True


class LLMRouter:
    """
    Multi-provider LLM router for optimal model selection.
    
    Provides intelligent routing based on:
    - Task-specific model optimization
    - Provider availability and reliability
    - Fallback strategies for resilience
    """
    
    def __init__(self):
        self.providers: Dict[str, ProviderConfig] = {}
        self.task_routing: Dict[str, str] = {}
        
        # Default task-specific routing based on COPILOT_NOTES.md Decision #1
        self.default_task_routing = {
            'descriptor_generation': 'openai/gpt-4-turbo',  # Creative writing excellence
            'sizing_analysis': 'openai/gpt-4',              # Superior reasoning 
            'brand_research': 'openai/gpt-4o',              # Latest capabilities
            'quality_evaluation': 'openai/gpt-4',           # Analytical tasks
            'conversation': 'openai/gpt-3.5-turbo',         # Fast responses
            'default': 'openai/gpt-4-turbo'                 # General purpose
        }
    
    def register_provider(self, name: str, service: LLMModelService, 
                         models: List[str] = None, priority: int = 0) -> None:
        """
        Register an LLM provider with the router.
        
        Args:
            name: Provider name (e.g., 'openai', 'anthropic', 'gemini')
            service: LLM service instance
            models: List of supported models (if None, uses service defaults)
            priority: Provider priority (higher = preferred)
        """
        if models is None:
            # Try to get models from service if available
            if hasattr(service, 'list_supported_models'):
                models = service.list_supported_models()
            else:
                models = []
        
        self.providers[name] = ProviderConfig(
            service=service,
            models=models,
            priority=priority,
            enabled=True
        )
        
        logger.info(f"Registered LLM provider: {name} with {len(models)} models")
    
    def set_task_routing(self, task: str, provider_model: str) -> None:
        """
        Set routing for a specific task.
        
        Args:
            task: Task name (e.g., 'descriptor_generation', 'sizing_analysis')
            provider_model: Provider and model in format 'provider/model'
        """
        self.task_routing[task] = provider_model
        logger.info(f"Set task routing: {task} -> {provider_model}")
    
    def get_optimal_provider(self, task: str = None, model: str = None) -> tuple[str, LLMModelService, str]:
        """
        Get optimal provider for a task or specific model.
        
        Args:
            task: Task type for intelligent routing
            model: Specific model requested (overrides task routing)
            
        Returns:
            Tuple of (provider_name, service_instance, model_name)
            
        Raises:
            ModelNotFoundError: If no suitable provider found
        """
        # If specific model requested, find provider that supports it
        if model:
            provider_name, model_name = self._parse_provider_model(model)
            
            if provider_name:
                # Specific provider requested
                if provider_name not in self.providers:
                    raise ModelNotFoundError(f"Provider {provider_name} not registered")
                
                provider_config = self.providers[provider_name]
                if not provider_config.enabled:
                    raise ModelNotFoundError(f"Provider {provider_name} is disabled")
                
                return provider_name, provider_config.service, model_name
            
            else:
                # Model only specified, find best provider
                for provider_name, config in sorted(self.providers.items(), 
                                                  key=lambda x: x[1].priority, reverse=True):
                    if config.enabled and model in config.models:
                        return provider_name, config.service, model
                
                raise ModelNotFoundError(f"Model {model} not found in any provider")
        
        # Task-based routing
        if task:
            route = self.task_routing.get(task) or self.default_task_routing.get(task)
            if route:
                provider_name, model_name = self._parse_provider_model(route)
                if provider_name in self.providers and self.providers[provider_name].enabled:
                    return provider_name, self.providers[provider_name].service, model_name
        
        # Fallback to default
        default_route = self.default_task_routing.get('default')
        if default_route:
            provider_name, model_name = self._parse_provider_model(default_route)
            if provider_name in self.providers and self.providers[provider_name].enabled:
                return provider_name, self.providers[provider_name].service, model_name
        
        # Last resort: any available provider
        for provider_name, config in sorted(self.providers.items(), 
                                          key=lambda x: x[1].priority, reverse=True):
            if config.enabled:
                default_model = getattr(config.service, 'default_model', None)
                if default_model:
                    return provider_name, config.service, default_model
        
        raise ModelNotFoundError("No available LLM providers")
    
    async def chat_completion(self, system: str = None, messages: List[Dict[str, str]] = None,
                            model: str = None, task: str = None, **kwargs) -> Dict[str, Any]:
        """
        Route chat completion to optimal provider.
        
        Args:
            system: System message/prompt
            messages: Conversation messages
            model: Specific model (overrides task routing)
            task: Task type for intelligent routing
            **kwargs: Additional parameters
            
        Returns:
            Standardized response dictionary
            
        Raises:
            LLMError: For any routing or LLM service errors
        """
        try:
            provider_name, service, model_name = self.get_optimal_provider(task, model)
            
            logger.debug(f"Routing {task or 'chat'} to {provider_name}/{model_name}")
            
            # Add provider info to response
            response = await service.chat_completion(
                system=system, 
                messages=messages, 
                model=model_name, 
                **kwargs
            )
            
            # Ensure provider info is in response
            response['provider'] = provider_name
            response['routed_model'] = model_name
            
            return response
            
        except Exception as e:
            if isinstance(e, LLMError):
                raise e
            else:
                raise LLMError(f"Router error: {str(e)}", original_error=e)
    
    async def test_provider(self, provider_name: str) -> bool:
        """
        Test a specific provider connection.
        
        Args:
            provider_name: Name of provider to test
            
        Returns:
            True if provider is working, False otherwise
        """
        if provider_name not in self.providers:
            return False
        
        provider = self.providers[provider_name]
        if not provider.enabled:
            return False
        
        try:
            if hasattr(provider.service, 'test_connection'):
                return await provider.service.test_connection()
            else:
                # Fallback test
                response = await provider.service.chat_completion(
                    system="Test",
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=5
                )
                return response is not None
                
        except Exception as e:
            logger.error(f"Provider {provider_name} test failed: {e}")
            return False
    
    async def test_all_providers(self) -> Dict[str, bool]:
        """
        Test all registered providers.
        
        Returns:
            Dict mapping provider names to their test results
        """
        results = {}
        for provider_name in self.providers.keys():
            results[provider_name] = await self.test_provider(provider_name)
        
        return results
    
    def _parse_provider_model(self, provider_model: str) -> tuple[Optional[str], str]:
        """
        Parse provider/model string.
        
        Args:
            provider_model: String in format 'provider/model' or just 'model'
            
        Returns:
            Tuple of (provider_name, model_name)
        """
        if '/' in provider_model:
            parts = provider_model.split('/', 1)
            return parts[0], parts[1]
        else:
            return None, provider_model
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """
        List all available models by provider.
        
        Returns:
            Dict mapping provider names to their available models
        """
        available = {}
        for provider_name, config in self.providers.items():
            if config.enabled:
                available[provider_name] = config.models.copy()
        
        return available
    
    def get_routing_info(self) -> Dict[str, Any]:
        """
        Get current routing configuration information.
        
        Returns:
            Dict with routing configuration details
        """
        return {
            'providers': {
                name: {
                    'enabled': config.enabled,
                    'priority': config.priority,
                    'models': config.models,
                    'provider_name': config.service.provider_name
                }
                for name, config in self.providers.items()
            },
            'task_routing': {**self.default_task_routing, **self.task_routing},
            'active_providers': [name for name, config in self.providers.items() if config.enabled]
        }


def create_default_router(openai_api_key: str = None, anthropic_api_key: str = None, 
                         gemini_api_key: str = None) -> LLMRouter:
    """
    Create a default LLM router with all available services.
    
    Args:
        openai_api_key: OpenAI API key (optional, reads from env if not provided)
        anthropic_api_key: Anthropic API key (optional, reads from env if not provided)
        gemini_api_key: Gemini API key (optional, reads from env if not provided)
        
    Returns:
        Configured LLM router with all available providers
    """
    router = LLMRouter()
    
    # Register OpenAI service (always available)
    try:
        from .openai_service import OpenAIService
        openai_service = OpenAIService(api_key=openai_api_key)
        router.register_provider(
            'openai', 
            openai_service, 
            openai_service.list_supported_models(),
            priority=100
        )
        logger.info("Registered OpenAI service with router")
        
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI service: {e}")
    
    # Register Anthropic service (optional)
    try:
        from .anthropic_service import AnthropicService
        anthropic_service = AnthropicService(api_key=anthropic_api_key)
        router.register_provider(
            'anthropic',
            anthropic_service,
            anthropic_service.list_supported_models(),
            priority=110  # Higher priority for reasoning tasks
        )
        logger.info("Registered Anthropic service with router")
        
        # Update task routing to use Anthropic for reasoning
        router.set_task_routing('sizing_analysis', 'anthropic/claude-3-5-sonnet-20241022')
        router.set_task_routing('brand_research', 'anthropic/claude-3-5-sonnet-20241022')
        router.set_task_routing('quality_evaluation', 'anthropic/claude-3-sonnet-20240229')
        
    except Exception as e:
        logger.warning(f"Anthropic service not available: {e}")
    
    # Register Gemini service (optional)
    try:
        from .gemini_service import GeminiService
        gemini_service = GeminiService(api_key=gemini_api_key)
        router.register_provider(
            'gemini',
            gemini_service, 
            gemini_service.list_supported_models(),
            priority=90
        )
        logger.info("Registered Gemini service with router")
        
        # Gemini can be good for certain specialized tasks
        # router.set_task_routing('multimodal_analysis', 'gemini/gemini-1.5-pro')
        
    except Exception as e:
        logger.warning(f"Gemini service not available: {e}")
    
    # Ensure we have at least one provider
    if not router.providers:
        raise LLMError("No LLM providers available. Check API keys and package installations.")
    
    logger.info(f"Router initialized with {len(router.providers)} providers")
    return router 