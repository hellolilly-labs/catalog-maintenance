"""
Tests for Configuration Management System

Tests the centralized configuration system including settings validation,
environment handling, provider configuration, and cache duration management.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from configs.settings import Settings, get_settings, reload_settings, create_test_settings
from pydantic import ValidationError


class TestSettings:
    """Test the Settings class and its various configurations."""
    
    def test_default_settings(self):
        """Test default settings are properly loaded."""
        settings = Settings()
        
        # Test default values
        assert settings.ENV == "dev"
        assert settings.DEBUG is True
        assert settings.LOG_LEVEL == "INFO"
        assert settings.OPENAI_DEFAULT_MODEL == "gpt-4.1"
        assert settings.PINECONE_DEFAULT_DIMENSION == 1536
        assert settings.INGESTION_BATCH_SIZE == 20
    
    def test_environment_validation(self):
        """Test environment validation."""
        # Valid environments
        for env in ['dev', 'staging', 'prod']:
            settings = Settings(ENV=env)
            assert settings.ENV == env
        
        # Invalid environment
        with pytest.raises(ValidationError):
            Settings(ENV="invalid")
    
    def test_log_level_validation(self):
        """Test log level validation."""
        # Valid log levels (case insensitive)
        for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            settings = Settings(LOG_LEVEL=level)
            assert settings.LOG_LEVEL == level
            
            settings = Settings(LOG_LEVEL=level.lower())
            assert settings.LOG_LEVEL == level
        
        # Invalid log level
        with pytest.raises(ValidationError):
            Settings(LOG_LEVEL="INVALID")
    
    def test_quality_threshold_validation(self):
        """Test quality threshold validation."""
        # Valid thresholds
        for threshold in [1.0, 5.5, 10.0]:
            settings = Settings(DEFAULT_QUALITY_THRESHOLD=threshold)
            assert settings.DEFAULT_QUALITY_THRESHOLD == threshold
        
        # Invalid thresholds
        for threshold in [0.0, 0.5, 10.5, -1.0]:
            with pytest.raises(ValidationError):
                Settings(DEFAULT_QUALITY_THRESHOLD=threshold)
    
    def test_pinecone_metric_validation(self):
        """Test Pinecone metric validation."""
        # Valid metrics
        for metric in ['cosine', 'euclidean', 'dotproduct']:
            settings = Settings(PINECONE_INDEX_METRIC=metric)
            assert settings.PINECONE_INDEX_METRIC == metric
        
        # Invalid metric
        with pytest.raises(ValidationError):
            Settings(PINECONE_INDEX_METRIC="invalid")
    
    def test_computed_properties(self):
        """Test computed properties."""
        # Development settings
        dev_settings = Settings(ENV="dev")
        assert dev_settings.current_bucket == dev_settings.GCP_BUCKET_DEV
        assert dev_settings.is_development is True
        assert dev_settings.is_production is False
        
        # Production settings
        prod_settings = Settings(ENV="prod")
        assert prod_settings.current_bucket == prod_settings.GCP_BUCKET_PROD
        assert prod_settings.is_development is False
        assert prod_settings.is_production is True
    
    def test_cache_duration_for_phase(self):
        """Test cache duration calculation for different phases."""
        settings = Settings()
        
        # Test known phases
        assert settings.cache_duration_for_phase("foundation") == 180
        assert settings.cache_duration_for_phase("product_style") == 60
        assert settings.cache_duration_for_phase("voice_messaging") == 30
        assert settings.cache_duration_for_phase("ai_persona_generation") == 0
        
        # Test unknown phase (default)
        assert settings.cache_duration_for_phase("unknown_phase") == 90
    
    def test_get_llm_config(self):
        """Test LLM configuration retrieval."""
        settings = Settings(
            OPENAI_API_KEY="test-openai-key",
            ANTHROPIC_API_KEY="test-anthropic-key",
            GEMINI_API_KEY="test-gemini-key"
        )
        
        # Test OpenAI config
        openai_config = settings.get_llm_config("openai")
        assert openai_config["api_key"] == "test-openai-key"
        assert openai_config["default_model"] == "gpt-4.1"
        assert openai_config["max_tokens"] == 4000
        
        # Test Anthropic config
        anthropic_config = settings.get_llm_config("anthropic")
        assert anthropic_config["api_key"] == "test-anthropic-key"
        assert anthropic_config["default_model"] == "claude-3-sonnet-20240229"
        
        # Test Gemini config
        gemini_config = settings.get_llm_config("gemini")
        assert gemini_config["api_key"] == "test-gemini-key"
        assert gemini_config["default_model"] == "gemini-pro"
        
        # Test unknown provider
        unknown_config = settings.get_llm_config("unknown")
        assert unknown_config == {}
    
    def test_validate_required_settings_dev(self):
        """Test validation in development environment."""
        # Valid development settings
        settings = Settings(ENV="dev", OPENAI_API_KEY="test-key")
        missing = settings.validate_required_settings()
        assert missing == []
        
        # Missing LLM provider
        settings = Settings(ENV="dev")
        missing = settings.validate_required_settings()
        assert len(missing) == 1
        assert "LLM provider" in missing[0]
    
    def test_validate_required_settings_prod(self):
        """Test validation in production environment."""
        # Valid production settings
        settings = Settings(
            ENV="prod", 
            OPENAI_API_KEY="test-key",
            GCP_PROJECT_ID="test-project"
        )
        missing = settings.validate_required_settings()
        assert missing == []
        
        # Missing GCP project in production
        settings = Settings(ENV="prod", OPENAI_API_KEY="test-key")
        missing = settings.validate_required_settings()
        assert len(missing) == 1
        assert "GCP_PROJECT_ID" in missing[0]


class TestEnvironmentIntegration:
    """Test environment variable integration."""
    
    def test_env_file_loading(self):
        """Test .env file loading (mocked)."""
        with patch.dict(os.environ, {
            'ENV': 'staging',
            'OPENAI_API_KEY': 'env-openai-key',
            'DEBUG': 'false'
        }):
            settings = Settings()
            assert settings.ENV == 'staging'
            assert settings.OPENAI_API_KEY == 'env-openai-key'
            assert settings.DEBUG is False
    
    def test_environment_override(self):
        """Test environment variables override defaults."""
        with patch.dict(os.environ, {
            'OPENAI_DEFAULT_MODEL': 'gpt-3.5-turbo',
            'INGESTION_BATCH_SIZE': '50',
            'DEFAULT_QUALITY_THRESHOLD': '9.0'
        }):
            settings = Settings()
            assert settings.OPENAI_DEFAULT_MODEL == 'gpt-3.5-turbo'
            assert settings.INGESTION_BATCH_SIZE == 50
            assert settings.DEFAULT_QUALITY_THRESHOLD == 9.0


class TestHelperFunctions:
    """Test helper functions for settings management."""
    
    def test_get_settings(self):
        """Test get_settings function."""
        settings = get_settings()
        assert isinstance(settings, Settings)
        assert settings.ENV in ['dev', 'staging', 'prod']
    
    @patch('configs.settings.Settings')
    def test_reload_settings(self, mock_settings):
        """Test reload_settings function."""
        mock_instance = MagicMock()
        mock_settings.return_value = mock_instance
        
        result = reload_settings()
        
        mock_settings.assert_called_once()
        assert result == mock_instance
    
    def test_create_test_settings(self):
        """Test create_test_settings function."""
        # Default test settings
        settings = create_test_settings()
        assert settings.ENV == "dev"
        assert settings.DEBUG is True
        assert settings.OPENAI_API_KEY == "test-key"
        
        # Test settings with overrides
        settings = create_test_settings(
            ENV="staging",
            OPENAI_DEFAULT_MODEL="o3",
            CUSTOM_SETTING="custom-value"
        )
        assert settings.ENV == "staging"
        assert settings.OPENAI_DEFAULT_MODEL == "o3"


class TestProductionReadiness:
    """Test production-specific configuration requirements."""
    
    def test_production_validation_success(self):
        """Test successful production configuration."""
        settings = Settings(
            ENV="prod",
            DEBUG=False,
            LOG_LEVEL="WARNING",
            OPENAI_API_KEY="prod-key",
            GCP_PROJECT_ID="prod-project",
            GCP_BUCKET_PROD="prod-bucket"
        )
        
        missing = settings.validate_required_settings()
        assert missing == []
        assert settings.is_production is True
        assert settings.DEBUG is False
    
    def test_production_validation_failure(self):
        """Test production configuration validation failure."""
        # Missing required production settings
        with pytest.raises(ValueError, match="Missing required settings in production"):
            with patch('configs.settings.logger'):
                Settings(ENV="prod")


class TestPerformanceConfiguration:
    """Test performance-related configuration options."""
    
    def test_batch_size_settings(self):
        """Test batch size configurations."""
        settings = Settings(
            INGESTION_BATCH_SIZE=30,
            MAX_BATCH_SIZE=200,
            MAX_CONCURRENT_BATCHES=10
        )
        
        assert settings.INGESTION_BATCH_SIZE == 30
        assert settings.MAX_BATCH_SIZE == 200
        assert settings.MAX_CONCURRENT_BATCHES == 10
    
    def test_timeout_settings(self):
        """Test timeout configurations."""
        settings = Settings(
            DEFAULT_TIMEOUT=60,
            INGESTION_TIMEOUT=3600,
            INDEX_CREATION_TIMEOUT=600
        )
        
        assert settings.DEFAULT_TIMEOUT == 60
        assert settings.INGESTION_TIMEOUT == 3600
        assert settings.INDEX_CREATION_TIMEOUT == 600
    
    def test_retry_settings(self):
        """Test retry configurations."""
        settings = Settings(
            RETRY_ATTEMPTS=5,
            UPSERT_RETRY_ATTEMPTS=7
        )
        
        assert settings.RETRY_ATTEMPTS == 5
        assert settings.UPSERT_RETRY_ATTEMPTS == 7


class TestCacheConfiguration:
    """Test cache duration and research configuration."""
    
    def test_cache_durations(self):
        """Test cache duration settings."""
        settings = Settings(
            CACHE_DURATION_FOUNDATION=365,  # 1 year
            CACHE_DURATION_PRODUCT_STYLE=30,  # 1 month
            CACHE_DURATION_VOICE_MESSAGING=7   # 1 week
        )
        
        assert settings.CACHE_DURATION_FOUNDATION == 365
        assert settings.CACHE_DURATION_PRODUCT_STYLE == 30
        assert settings.CACHE_DURATION_VOICE_MESSAGING == 7
        
        # Test computed durations
        assert settings.cache_duration_for_phase("foundation") == 365
        assert settings.cache_duration_for_phase("product_style") == 30
        assert settings.cache_duration_for_phase("voice_messaging") == 7
    
    def test_research_quality_settings(self):
        """Test research quality and timing configurations."""
        settings = Settings(
            MIN_RESEARCH_TIME=600,  # 10 minutes
            MAX_RESEARCH_TIME=1800, # 30 minutes
            MIN_SOURCES=20,
            MIN_LLM_ROUNDS=10,
            DEFAULT_QUALITY_THRESHOLD=9.5
        )
        
        assert settings.MIN_RESEARCH_TIME == 600
        assert settings.MAX_RESEARCH_TIME == 1800
        assert settings.MIN_SOURCES == 20
        assert settings.MIN_LLM_ROUNDS == 10
        assert settings.DEFAULT_QUALITY_THRESHOLD == 9.5


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_api_keys(self):
        """Test handling of empty/None API keys."""
        settings = Settings()
        
        # None values should be preserved
        assert settings.OPENAI_API_KEY is None
        assert settings.ANTHROPIC_API_KEY is None
        assert settings.GEMINI_API_KEY is None
    
    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed for extensibility."""
        # This should not raise an error due to extra="allow"
        settings = Settings(FUTURE_SETTING="future-value")
        
        # The setting should be accessible
        assert hasattr(settings, 'FUTURE_SETTING')
        assert settings.FUTURE_SETTING == "future-value"
    
    def test_case_sensitivity(self):
        """Test case sensitivity of environment variables."""
        with patch.dict(os.environ, {
            'openai_api_key': 'lowercase-key',  # Wrong case
            'OPENAI_API_KEY': 'correct-key'     # Correct case
        }):
            settings = Settings()
            # Should use the correctly cased version
            assert settings.OPENAI_API_KEY == 'correct-key'


if __name__ == "__main__":
    pytest.main([__file__]) 