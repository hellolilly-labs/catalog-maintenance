"""
LLM Service Error Classes

Provides comprehensive error handling for LLM service operations including
rate limiting, token limits, model errors, and network issues.
"""

class LLMError(Exception):
    """Base exception for all LLM service errors"""
    
    def __init__(self, message: str, original_error: Exception = None, provider: str = None):
        super().__init__(message)
        self.original_error = original_error
        self.provider = provider
        self.message = message

    def __str__(self):
        if self.provider:
            return f"[{self.provider}] {self.message}"
        return self.message


class RateLimitError(LLMError):
    """Raised when API rate limits are exceeded"""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class TokenLimitError(LLMError):
    """Raised when token limits are exceeded"""
    
    def __init__(self, message: str = "Token limit exceeded", 
                 token_count: int = None, max_tokens: int = None, **kwargs):
        super().__init__(message, **kwargs)
        self.token_count = token_count
        self.max_tokens = max_tokens


class ModelNotFoundError(LLMError):
    """Raised when requested model is not available"""
    
    def __init__(self, message: str = "Model not found", model: str = None, **kwargs):
        super().__init__(message, **kwargs)
        self.model = model


class AuthenticationError(LLMError):
    """Raised when API authentication fails"""
    pass


class NetworkError(LLMError):
    """Raised when network operations fail"""
    
    def __init__(self, message: str = "Network error", status_code: int = None, **kwargs):
        super().__init__(message, **kwargs)
        self.status_code = status_code


class InvalidRequestError(LLMError):
    """Raised when the request is invalid or malformed"""
    pass


class ServiceUnavailableError(LLMError):
    """Raised when the LLM service is temporarily unavailable"""
    
    def __init__(self, message: str = "Service unavailable", retry_after: int = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after 