"""
Configuration Management System

Centralized configuration management with environment-aware settings following
ROADMAP section 8.4. Provides secure API key handling, storage configuration,
and performance tuning parameters for the catalog maintenance system.
"""

import os
import logging
from typing import Dict, Optional, List, Any
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from pathlib import Path

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Centralized configuration management for catalog maintenance system.
    
    Features:
    - Environment-aware settings (dev, staging, prod)
    - Secure API key management
    - Storage bucket configuration
    - Performance tuning parameters
    - Future brand research configuration
    """
    
    # ============================================================================
    # Environment Configuration
    # ============================================================================
    
    ENV: str = Field(
        default="dev", 
        description="Environment: dev, staging, prod",
        env="ENV"
    )
    
    DEBUG: bool = Field(
        default=True,
        description="Enable debug mode",
        env="DEBUG"
    )
    
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level",
        env="LOG_LEVEL"
    )
    
    # ============================================================================
    # Storage Configuration
    # ============================================================================
    
    GCP_BUCKET_PROD: str = Field(
        default="liddy-account-documents",
        description="Production GCP storage bucket",
        env="GCP_BUCKET_PROD"
    )
    
    GCP_BUCKET_DEV: str = Field(
        default="liddy-account-documents-dev", 
        description="Development GCP storage bucket",
        env="GCP_BUCKET_DEV"
    )
    
    GCP_PROJECT_ID: Optional[str] = Field(
        default=None,
        description="GCP Project ID",
        env="GCP_PROJECT_ID"
    )
    
    GCP_CREDENTIALS_PATH: Optional[str] = Field(
        default=None,
        description="Path to GCP service account credentials",
        env="GCP_CREDENTIALS_PATH"
    )
    
    # ============================================================================
    # LLM Service Configuration
    # ============================================================================
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = Field(None, description="OpenAI API key")
    OPENAI_DEFAULT_MODEL: str = Field(
        default="gpt-4-turbo",
        description="Default OpenAI model",
        env="OPENAI_DEFAULT_MODEL"
    )
    OPENAI_MAX_TOKENS: int = Field(4000, description="Maximum tokens for OpenAI requests")
    OPENAI_RETRY_ATTEMPTS: int = Field(3, description="Number of retry attempts for failed requests")
    
    # Anthropic Configuration  
    ANTHROPIC_API_KEY: Optional[str] = Field(None, description="Anthropic Claude API key")
    ANTHROPIC_MAX_TOKENS: int = Field(4000, description="Maximum tokens for Anthropic requests")
    
    # Gemini Configuration
    GEMINI_API_KEY: Optional[str] = Field(None, description="Google Gemini API key")
    GEMINI_MAX_TOKENS: int = Field(2000, description="Maximum tokens for Gemini requests")
    
    # ============================================================================
    # Langfuse Configuration
    # ============================================================================
    
    LANGFUSE_PUBLIC_KEY: Optional[str] = Field(
        default=None,
        description="Langfuse public key for prompt management",
        env="LANGFUSE_PUBLIC_KEY"
    )
    
    LANGFUSE_SECRET_KEY: Optional[str] = Field(
        default=None,
        description="Langfuse secret key for prompt management", 
        env="LANGFUSE_SECRET_KEY"
    )
    
    LANGFUSE_HOST: str = Field(
        default="https://cloud.langfuse.com",
        description="Langfuse host URL",
        env="LANGFUSE_HOST"
    )
    
    # ============================================================================
    # Web Search Configuration (Future Brand Research)
    # ============================================================================
    
    TAVILY_API_KEY: Optional[str] = Field(
        default=None,
        description="Tavily API key for web search",
        env="TAVILY_API_KEY"
    )
    
    GOOGLE_SEARCH_API_KEY: Optional[str] = Field(
        default=None,
        description="Google Search API key",
        env="GOOGLE_SEARCH_API_KEY"
    )
    
    GOOGLE_SEARCH_ENGINE_ID: Optional[str] = Field(
        default=None,
        description="Google Custom Search Engine ID",
        env="GOOGLE_SEARCH_ENGINE_ID"
    )
    
    # ============================================================================
    # Research Configuration (Future Brand Research)
    # ============================================================================
    
    MIN_RESEARCH_TIME: int = Field(
        default=300,
        description="Minimum research time in seconds (5 minutes)",
        env="MIN_RESEARCH_TIME"
    )
    
    MAX_RESEARCH_TIME: int = Field(
        default=900,
        description="Maximum research time in seconds (15 minutes)",
        env="MAX_RESEARCH_TIME"
    )
    
    MIN_SOURCES: int = Field(
        default=15,
        description="Minimum sources per research phase",
        env="MIN_SOURCES"
    )
    
    MIN_LLM_ROUNDS: int = Field(
        default=5,
        description="Minimum LLM analysis rounds per phase",
        env="MIN_LLM_ROUNDS"
    )
    
    DEFAULT_QUALITY_THRESHOLD: float = Field(
        default=8.0,
        description="Default quality threshold for research phases",
        env="DEFAULT_QUALITY_THRESHOLD"
    )
    
    # ============================================================================
    # Phase Cache Durations (in days)
    # ============================================================================
    
    CACHE_DURATION_FOUNDATION: int = Field(
        default=180,
        description="Foundation phase cache duration in days (6 months)",
        env="CACHE_DURATION_FOUNDATION"
    )
    
    CACHE_DURATION_MARKET_POSITIONING: int = Field(
        default=120,
        description="Market positioning cache duration in days (4 months)",
        env="CACHE_DURATION_MARKET_POSITIONING"
    )
    
    CACHE_DURATION_PRODUCT_STYLE: int = Field(
        default=60,
        description="Product style cache duration in days (2 months)",
        env="CACHE_DURATION_PRODUCT_STYLE"
    )
    
    CACHE_DURATION_CUSTOMER_CULTURAL: int = Field(
        default=90,
        description="Customer cultural cache duration in days (3 months)",
        env="CACHE_DURATION_CUSTOMER_CULTURAL"
    )
    
    CACHE_DURATION_VOICE_MESSAGING: int = Field(
        default=30,
        description="Voice messaging cache duration in days (1 month)",
        env="CACHE_DURATION_VOICE_MESSAGING"
    )
    
    CACHE_DURATION_LINEARITY_ANALYSIS: int = Field(
        default=120,
        description="Linearity analysis cache duration in days (4 months)",
        env="CACHE_DURATION_LINEARITY_ANALYSIS"
    )
    
    CACHE_DURATION_RAG_OPTIMIZATION: int = Field(
        default=180,
        description="RAG optimization cache duration in days (6 months)",
        env="CACHE_DURATION_RAG_OPTIMIZATION"
    )
    
    # AI persona generation has no auto-refresh (manual only)
    
    # ============================================================================
    # Pinecone Configuration
    # ============================================================================
    
    PINECONE_API_KEY: Optional[str] = Field(
        default=None,
        description="Pinecone API key for vector storage",
        env="PINECONE_API_KEY"
    )
    
    PINECONE_ENVIRONMENT: str = Field(
        default="us-west1-gcp",
        description="Pinecone environment region",
        env="PINECONE_ENVIRONMENT"
    )
    
    PINECONE_DEFAULT_DIMENSION: int = Field(
        default=1536,
        description="Default vector dimension (OpenAI embeddings)",
        env="PINECONE_DEFAULT_DIMENSION"
    )
    
    PINECONE_INDEX_METRIC: str = Field(
        default="cosine",
        description="Vector similarity metric",
        env="PINECONE_INDEX_METRIC"
    )
    
    # ============================================================================
    # Performance Configuration
    # ============================================================================
    
    MAX_PARALLEL_BRANDS: int = Field(
        default=5,
        description="Maximum brands to process in parallel",
        env="MAX_PARALLEL_BRANDS"
    )
    
    DEFAULT_TIMEOUT: int = Field(
        default=120,
        description="Default operation timeout in seconds",
        env="DEFAULT_TIMEOUT"
    )
    
    MAX_BATCH_SIZE: int = Field(
        default=100,
        description="Maximum batch size for operations",
        env="MAX_BATCH_SIZE"
    )
    
    UPSERT_RETRY_ATTEMPTS: int = Field(
        default=3,
        description="Number of retry attempts for upsert operations",
        env="UPSERT_RETRY_ATTEMPTS"
    )
    
    INDEX_CREATION_TIMEOUT: int = Field(
        default=300,
        description="Timeout for index creation in seconds",
        env="INDEX_CREATION_TIMEOUT"
    )
    
    # ============================================================================
    # Ingestion Configuration
    # ============================================================================
    
    INGESTION_BATCH_SIZE: int = Field(
        default=20,
        description="Batch size for product ingestion",
        env="INGESTION_BATCH_SIZE"
    )
    
    MAX_CONCURRENT_BATCHES: int = Field(
        default=5,
        description="Maximum concurrent ingestion batches",
        env="MAX_CONCURRENT_BATCHES"
    )
    
    INGESTION_TIMEOUT: int = Field(
        default=1800,
        description="Ingestion timeout in seconds (30 minutes)",
        env="INGESTION_TIMEOUT"
    )
    
    RETRY_ATTEMPTS: int = Field(
        default=3,
        description="Number of retry attempts for failed operations",
        env="RETRY_ATTEMPTS"
    )
    
    ENABLE_PROGRESS_LOGGING: bool = Field(
        default=True,
        description="Enable progress logging for long-running operations",
        env="ENABLE_PROGRESS_LOGGING"
    )
    
    # ============================================================================
    # Replicate Configuration (Future AI Persona Generation)
    # ============================================================================
    
    REPLICATE_API_TOKEN: Optional[str] = Field(
        default=None,
        description="Replicate API token for avatar generation",
        env="REPLICATE_API_TOKEN"
    )
    
    # ============================================================================
    # Computed Properties
    # ============================================================================
    
    @property
    def current_bucket(self) -> str:
        """Get bucket name for current environment."""
        return self.GCP_BUCKET_PROD if self.ENV == "prod" else self.GCP_BUCKET_DEV
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENV == "prod"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENV == "dev"
    
    def cache_duration_for_phase(self, phase_name: str) -> int:
        """
        Get cache duration for specific research phase.
        
        Args:
            phase_name: Name of the research phase
            
        Returns:
            Cache duration in days
        """
        phase_durations = {
            "foundation": self.CACHE_DURATION_FOUNDATION,
            "market_positioning": self.CACHE_DURATION_MARKET_POSITIONING,
            "product_style": self.CACHE_DURATION_PRODUCT_STYLE,
            "customer_cultural": self.CACHE_DURATION_CUSTOMER_CULTURAL,
            "voice_messaging": self.CACHE_DURATION_VOICE_MESSAGING,
            "linearity_analysis": self.CACHE_DURATION_LINEARITY_ANALYSIS,
            "rag_optimization": self.CACHE_DURATION_RAG_OPTIMIZATION,
            "ai_persona_generation": 0  # Manual only - never auto-refresh
        }
        
        return phase_durations.get(phase_name, 90)  # Default 3 months
    
    def get_llm_config(self, provider: str) -> Dict[str, Any]:
        """
        Get LLM configuration for specific provider.
        
        Args:
            provider: LLM provider name (openai, anthropic, gemini)
            
        Returns:
            Configuration dictionary for the provider
        """
        configs = {
            "openai": {
                "api_key": self.OPENAI_API_KEY,
                "default_model": self.OPENAI_DEFAULT_MODEL,
                "max_tokens": self.OPENAI_MAX_TOKENS
            },
            "anthropic": {
                "api_key": self.ANTHROPIC_API_KEY,
                "default_model": "claude-3-sonnet-20240229"
            },
            "gemini": {
                "api_key": self.GEMINI_API_KEY,
                "default_model": "gemini-pro"
            }
        }
        
        return configs.get(provider, {})
    
    def validate_required_settings(self) -> List[str]:
        """
        Validate that required settings are present.
        
        Returns:
            List of missing required settings
        """
        missing = []
        
        # Check for at least one LLM provider
        if not any([self.OPENAI_API_KEY, self.ANTHROPIC_API_KEY, self.GEMINI_API_KEY]):
            missing.append("At least one LLM provider API key (OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY)")
        
        # Check for storage configuration in production
        if self.is_production and not self.GCP_PROJECT_ID:
            missing.append("GCP_PROJECT_ID (required in production)")
        
        return missing
    
    # ============================================================================
    # Validators
    # ============================================================================
    
    @validator('ENV')
    def validate_environment(cls, v):
        """Validate environment setting."""
        valid_envs = ['dev', 'staging', 'prod']
        if v not in valid_envs:
            raise ValueError(f"ENV must be one of {valid_envs}")
        return v
    
    @validator('LOG_LEVEL')
    def validate_log_level(cls, v):
        """Validate log level setting."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()
    
    @validator('DEFAULT_QUALITY_THRESHOLD')
    def validate_quality_threshold(cls, v):
        """Validate quality threshold is between 1 and 10."""
        if not 1.0 <= v <= 10.0:
            raise ValueError("DEFAULT_QUALITY_THRESHOLD must be between 1.0 and 10.0")
        return v
    
    @validator('PINECONE_INDEX_METRIC')
    def validate_pinecone_metric(cls, v):
        """Validate Pinecone similarity metric."""
        valid_metrics = ['cosine', 'euclidean', 'dotproduct']
        if v not in valid_metrics:
            raise ValueError(f"PINECONE_INDEX_METRIC must be one of {valid_metrics}")
        return v
    
    # ============================================================================
    # Pydantic Configuration
    # ============================================================================
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True
        validate_assignment = True
        
        # Allow extra fields for future extensibility
        extra = "allow"


# ============================================================================
# Global Settings Instance
# ============================================================================

# Global settings instance - will be imported by other modules
settings = Settings()

# Validate settings on import
missing_settings = settings.validate_required_settings()
if missing_settings:
    logger.warning(f"Missing required settings: {', '.join(missing_settings)}")
    if settings.is_production:
        raise ValueError(f"Missing required settings in production: {', '.join(missing_settings)}")

logger.info(f"Configuration loaded for environment: {settings.ENV}")


# ============================================================================
# Helper Functions
# ============================================================================

def get_settings() -> Settings:
    """
    Get the global settings instance.
    
    Returns:
        Global settings instance
    """
    return settings


def reload_settings() -> Settings:
    """
    Reload settings from environment variables.
    
    Returns:
        Reloaded settings instance
    """
    global settings
    settings = Settings()
    return settings


def create_test_settings(**overrides) -> Settings:
    """
    Create settings instance for testing with overrides.
    
    Args:
        **overrides: Settings to override
        
    Returns:
        Settings instance with test configuration
    """
    test_env = {
        "ENV": "dev",
        "DEBUG": True,
        "OPENAI_API_KEY": "test-key",
        "GCP_BUCKET_DEV": "test-bucket",
        **overrides
    }
    
    return Settings(**test_env) 