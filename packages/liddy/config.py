"""
Configuration management for Liddy platform.

Provides centralized settings using Pydantic for validation.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Keys
    openai_api_key: Optional[str] = Field(None, env='OPENAI_API_KEY')
    OPENAI_API_KEY: Optional[str] = Field(None, env='OPENAI_API_KEY')  # Backward compatibility
    anthropic_api_key: Optional[str] = Field(None, env='ANTHROPIC_API_KEY')
    gemini_api_key: Optional[str] = Field(None, env='GEMINI_API_KEY')
    pinecone_api_key: Optional[str] = Field(None, env='PINECONE_API_KEY')
    tavily_api_key: Optional[str] = Field(None, env='TAVILY_API_KEY')
    TAVILY_API_KEY: Optional[str] = Field(None, env='TAVILY_API_KEY')  # Backward compatibility
    GOOGLE_SEARCH_API_KEY: Optional[str] = Field(None, env='GOOGLE_SEARCH_API_KEY')  # Backward compatibility
    GOOGLE_SEARCH_CX: Optional[str] = Field(None, env='GOOGLE_SEARCH_CX')  # Backward compatibility
    OPENAI_DEFAULT_MODEL: Optional[str] = Field('gpt-4o', env='OPENAI_DEFAULT_MODEL')  # Backward compatibility
    OPENAI_MAX_TOKENS: Optional[int] = Field(16000, env='OPENAI_MAX_TOKENS')  # Backward compatibility
    assemblyai_api_key: Optional[str] = Field(None, env='ASSEMBLYAI_API_KEY')
    
    # Langfuse (optional)
    langfuse_public_key: Optional[str] = Field(None, env='LANGFUSE_PUBLIC_KEY')
    langfuse_secret_key: Optional[str] = Field(None, env='LANGFUSE_SECRET_KEY')
    langfuse_host: Optional[str] = Field('https://cloud.langfuse.com', env='LANGFUSE_HOST')
    
    # Google Cloud
    google_cloud_project: Optional[str] = Field(None, env='GOOGLE_CLOUD_PROJECT')
    google_application_credentials: Optional[str] = Field(None, env='GOOGLE_APPLICATION_CREDENTIALS')
    
    # Storage
    storage_type: str = Field('local', env='STORAGE_TYPE')  # 'local' or 'gcs'
    gcs_bucket: Optional[str] = Field(None, env='GCS_BUCKET')
    
    # Pinecone
    pinecone_environment: str = Field('us-east1-gcp', env='PINECONE_ENVIRONMENT')
    
    # Model defaults
    default_model: str = Field('gpt-4o', env='DEFAULT_MODEL')
    default_temperature: float = Field(0.7, env='DEFAULT_TEMPERATURE')
    
    # Quality thresholds
    quality_threshold: float = Field(8.0, env='QUALITY_THRESHOLD')
    max_quality_retries: int = Field(3, env='MAX_QUALITY_RETRIES')
    
    # Development
    debug: bool = Field(False, env='DEBUG')
    log_level: str = Field('INFO', env='LOG_LEVEL')
    
    # Retry settings (backward compatibility)
    RETRY_ATTEMPTS: int = Field(3, env='RETRY_ATTEMPTS')
    RETRY_DELAY: float = Field(1.0, env='RETRY_DELAY')
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = False
        extra = 'ignore'  # Ignore extra fields


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# Backward compatibility
settings = get_settings()