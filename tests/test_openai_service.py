"""
Test OpenAI Service Implementation

Validates Issue #10: Complete LLM Provider Suite - Add OpenAI Service

Tests:
- OpenAI service initialization and configuration
- Chat completion functionality
- Error handling with retry logic
- Token counting and conversation truncation
- Multi-provider routing via LLMRouter
"""

import pytest
import asyncio
import os
from unittest.mock import Mock, patch, AsyncMock

from src.llm import OpenAIService, LLMRouter, create_default_router
from src.llm.errors import LLMError, AuthenticationError, ModelNotFoundError


@pytest.fixture
def mock_api_key():
    """Provide a mock API key for testing."""
    return "sk-test-api-key-for-testing"


@pytest.fixture
def openai_service(mock_api_key):
    """Create OpenAI service instance for testing."""
    with patch.dict(os.environ, {'OPENAI_API_KEY': mock_api_key}):
        return OpenAIService(api_key=mock_api_key)


class TestOpenAIService:
    """Test OpenAI service implementation."""
    
    def test_initialization_with_api_key(self, mock_api_key):
        """Test OpenAI service initializes correctly with API key."""
        service = OpenAIService(api_key=mock_api_key)
        
        assert service.provider_name == "openai"
        assert service.default_model == "gpt-4-turbo"
        assert service.api_key == mock_api_key
        assert service.client is not None
    
    def test_initialization_from_environment(self, mock_api_key):
        """Test OpenAI service reads API key from environment."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': mock_api_key}):
            service = OpenAIService()
            assert service.api_key == mock_api_key
    
    def test_initialization_without_api_key(self):
        """Test OpenAI service fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(AuthenticationError):
                OpenAIService()
    
    def test_supported_models(self, openai_service):
        """Test that all documented models are supported."""
        expected_models = ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo', 'gpt-4o']
        supported_models = openai_service.list_supported_models()
        
        for model in expected_models:
            assert model in supported_models
    
    def test_model_info(self, openai_service):
        """Test model information retrieval."""
        info = openai_service.get_model_info('gpt-4-turbo')
        
        assert 'max_tokens' in info
        assert 'context_window' in info
        assert 'description' in info
        assert info['context_window'] == 128000
    
    def test_model_info_invalid_model(self, openai_service):
        """Test model info fails for unsupported model."""
        with pytest.raises(ModelNotFoundError):
            openai_service.get_model_info('invalid-model')
    
    def test_message_preparation(self, openai_service):
        """Test message formatting for OpenAI API."""
        system = "You are a helpful assistant."
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        formatted = openai_service._prepare_messages(system, messages)
        
        assert len(formatted) == 3
        assert formatted[0] == {"role": "system", "content": system}
        assert formatted[1] == {"role": "user", "content": "Hello"}
        assert formatted[2] == {"role": "assistant", "content": "Hi there!"}
    
    def test_message_validation(self, openai_service):
        """Test message validation catches invalid formats."""
        # Test invalid message format
        invalid_messages = [
            {"content": "Missing role"},  # Missing role
            {"role": "invalid", "content": "Invalid role"},  # Invalid role
            {"role": "user"}  # Missing content
        ]
        
        for invalid_msg in invalid_messages:
            with pytest.raises(ValueError):
                openai_service.validate_messages([invalid_msg])
    
    @patch('src.llm.openai_service.AsyncOpenAI')
    async def test_chat_completion_success(self, mock_openai_client, openai_service):
        """Test successful chat completion."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Hello! How can I help you?"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 8
        mock_response.usage.total_tokens = 18
        
        mock_client_instance = AsyncMock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai_client.return_value = mock_client_instance
        
        # Create service with mocked client
        service = OpenAIService(api_key="test-key")
        service.client = mock_client_instance
        
        # Test chat completion
        response = await service.chat_completion(
            system="You are helpful",
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-3.5-turbo"
        )
        
        assert response['content'] == "Hello! How can I help you?"
        assert response['provider'] == 'openai'
        assert response['model'] == 'gpt-3.5-turbo'
        assert response['usage']['total_tokens'] == 18
        assert response['finish_reason'] == 'stop'
    
    def test_token_counting(self, openai_service):
        """Test token counting functionality."""
        text = "Hello, how are you doing today?"
        
        token_count = openai_service.count_tokens(text)
        
        # Token count should be reasonable (not exact due to encoding differences)
        assert isinstance(token_count, int)
        assert token_count > 0
        assert token_count < 50  # Should be much less for this simple text
    
    def test_conversation_token_counting(self, openai_service):
        """Test conversation token counting."""
        system = "You are a helpful assistant."
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        total_tokens = openai_service.count_conversation_tokens(system, messages)
        
        assert isinstance(total_tokens, int)
        assert total_tokens > 0
    
    def test_conversation_truncation(self, openai_service):
        """Test conversation truncation when too long."""
        system = "You are a helpful assistant."
        
        # Create a long conversation
        long_messages = []
        for i in range(100):
            long_messages.append({"role": "user", "content": f"Message {i} " * 50})
            long_messages.append({"role": "assistant", "content": f"Response {i} " * 50})
        
        truncated_system, truncated_messages = openai_service.truncate_conversation(
            system, long_messages, max_tokens=1000
        )
        
        assert truncated_system == system
        assert len(truncated_messages) < len(long_messages)
        
        # Verify truncated conversation fits within token limit
        total_tokens = openai_service.count_conversation_tokens(
            truncated_system, truncated_messages
        )
        assert total_tokens <= 1000


class TestLLMRouter:
    """Test LLM router functionality."""
    
    def test_router_initialization(self):
        """Test router initializes with correct default routing."""
        router = LLMRouter()
        
        assert 'descriptor_generation' in router.default_task_routing
        assert 'sizing_analysis' in router.default_task_routing
        assert router.default_task_routing['descriptor_generation'] == 'openai/gpt-4-turbo'
    
    def test_provider_registration(self, mock_api_key):
        """Test provider registration in router."""
        router = LLMRouter()
        service = OpenAIService(api_key=mock_api_key)
        
        router.register_provider('openai', service, ['gpt-4', 'gpt-3.5-turbo'], priority=100)
        
        assert 'openai' in router.providers
        assert router.providers['openai'].priority == 100
        assert router.providers['openai'].enabled == True
        assert 'gpt-4' in router.providers['openai'].models
    
    def test_optimal_provider_selection_by_task(self, mock_api_key):
        """Test router selects correct provider for specific tasks."""
        router = LLMRouter()
        service = OpenAIService(api_key=mock_api_key)
        router.register_provider('openai', service, service.list_supported_models())
        
        # Test task-based routing
        provider_name, service_instance, model = router.get_optimal_provider(task='descriptor_generation')
        
        assert provider_name == 'openai'
        assert model == 'gpt-4-turbo'
        assert isinstance(service_instance, OpenAIService)
    
    def test_optimal_provider_selection_by_model(self, mock_api_key):
        """Test router selects correct provider for specific model."""
        router = LLMRouter()
        service = OpenAIService(api_key=mock_api_key)
        router.register_provider('openai', service, service.list_supported_models())
        
        # Test model-based routing
        provider_name, service_instance, model = router.get_optimal_provider(model='gpt-3.5-turbo')
        
        assert provider_name == 'openai'
        assert model == 'gpt-3.5-turbo'
    
    def test_provider_model_parsing(self):
        """Test provider/model string parsing."""
        router = LLMRouter()
        
        # Test with provider prefix
        provider, model = router._parse_provider_model('openai/gpt-4')
        assert provider == 'openai'
        assert model == 'gpt-4'
        
        # Test without provider prefix
        provider, model = router._parse_provider_model('gpt-4')
        assert provider is None
        assert model == 'gpt-4'
    
    def test_available_models_listing(self, mock_api_key):
        """Test listing available models from router."""
        router = LLMRouter()
        service = OpenAIService(api_key=mock_api_key)
        router.register_provider('openai', service, service.list_supported_models())
        
        available = router.list_available_models()
        
        assert 'openai' in available
        assert 'gpt-4' in available['openai']
        assert 'gpt-4-turbo' in available['openai']
    
    def test_create_default_router(self, mock_api_key):
        """Test creating default router with OpenAI service."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': mock_api_key}):
            router = create_default_router()
            
            assert 'openai' in router.providers
            assert router.providers['openai'].enabled == True
            
            available_models = router.list_available_models()
            assert 'openai' in available_models
            assert len(available_models['openai']) > 0


# Integration tests (require actual API key)
@pytest.mark.integration
class TestOpenAIIntegration:
    """Integration tests requiring actual OpenAI API key."""
    
    @pytest.mark.skipif(not os.getenv('OPENAI_API_KEY'), reason="No OpenAI API key provided")
    async def test_real_openai_connection(self):
        """Test actual OpenAI API connection."""
        service = OpenAIService()
        
        # Test connection
        connection_ok = await service.test_connection()
        assert connection_ok == True
    
    @pytest.mark.skipif(not os.getenv('OPENAI_API_KEY'), reason="No OpenAI API key provided")
    async def test_real_chat_completion(self):
        """Test actual chat completion with OpenAI."""
        service = OpenAIService()
        
        response = await service.chat_completion(
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": "Say 'Hello World' and nothing else."}],
            model="gpt-3.5-turbo",
            max_tokens=10,
            temperature=0.1
        )
        
        assert response is not None
        assert 'content' in response
        assert 'Hello World' in response['content']
        assert response['provider'] == 'openai'
    
    @pytest.mark.skipif(not os.getenv('OPENAI_API_KEY'), reason="No OpenAI API key provided")
    async def test_router_integration(self):
        """Test router with actual OpenAI service."""
        router = create_default_router()
        
        # Test provider test
        test_results = await router.test_all_providers()
        assert test_results['openai'] == True
        
        # Test routed chat completion
        response = await router.chat_completion(
            system="You are helpful.",
            messages=[{"role": "user", "content": "Say 'test' and nothing else."}],
            task="conversation",
            max_tokens=5,
            temperature=0.1
        )
        
        assert response is not None
        assert 'content' in response
        assert response['provider'] == 'openai'
        assert response['routed_model'] == 'gpt-3.5-turbo'


if __name__ == "__main__":
    # Run basic unit tests
    pytest.main([__file__, "-v", "--tb=short"]) 