"""
Liddy Brand Intelligence Engine

Comprehensive brand research, catalog ingestion, and knowledge base creation
to power AI-driven customer experiences.
"""

import logging
import re
import hashlib
from typing import Set

__version__ = "0.3.0"


class RedactingFilter(logging.Filter):
    """
    Enhanced logging filter that redacts sensitive information.
    
    Features:
    - Case-insensitive pattern matching
    - 32-character hex fallback for unknown patterns
    - Idempotent installation (safe to call multiple times)
    """
    
    def __init__(self):
        super().__init__()
        self._redacted_patterns = {
            # API keys and tokens
            re.compile(r'\b[a-f0-9]{32,}\b', re.IGNORECASE),  # API keys (32+ hex chars)
            re.compile(r'\bsk-[a-zA-Z0-9]{32,}\b', re.IGNORECASE),  # OpenAI style keys
            re.compile(r'\bBearer\s+[a-zA-Z0-9+/=]+\b', re.IGNORECASE),  # Bearer tokens
            
            # Email addresses
            re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', re.IGNORECASE),
            
            # Phone numbers (various formats)
            re.compile(r'\b(?:\+?1[-.\s]?)?(?:\([0-9]{3}\)|[0-9]{3})[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),
            
            # Credit card numbers (basic pattern)
            re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
            
            # SSN pattern
            re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        }
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log record by redacting sensitive information.
        
        Args:
            record: The log record to filter
            
        Returns:
            True (always process the record, but redact sensitive data)
        """
        if hasattr(record, 'msg') and record.msg:
            record.msg = self._redact_text(str(record.msg))
        
        # Also redact any args that might contain sensitive data
        if hasattr(record, 'args') and record.args:
            if isinstance(record.args, (list, tuple)):
                record.args = tuple(
                    self._redact_text(str(arg)) if isinstance(arg, str) else arg
                    for arg in record.args
                )
            elif isinstance(record.args, dict):
                record.args = {
                    k: self._redact_text(str(v)) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
        
        return True
    
    def _redact_text(self, text: str) -> str:
        """
        Redact sensitive information from text.
        
        Args:
            text: The text to redact
            
        Returns:
            Text with sensitive information redacted
        """
        redacted_text = text
        
        for pattern in self._redacted_patterns:
            redacted_text = pattern.sub(self._generate_replacement, redacted_text)
        
        return redacted_text
    
    def _generate_replacement(self, match: re.Match) -> str:
        """
        Generate a consistent replacement for matched sensitive data.
        
        Uses the first 8 characters of SHA-256 hash as a 32-char hex identifier
        for consistent redaction of the same sensitive values.
        
        Args:
            match: The regex match object
            
        Returns:
            Redacted replacement string
        """
        sensitive_value = match.group(0)
        
        # Generate consistent hash-based identifier
        hash_obj = hashlib.sha256(sensitive_value.encode('utf-8'))
        hex_hash = hash_obj.hexdigest()[:8]  # First 8 chars of hash
        
        # Create 32-char hex string by repeating the 8-char hash
        redacted_id = (hex_hash * 4)[:32]
        
        return f"[REDACTED-{redacted_id}]"


def install_redacting_filter() -> None:
    """
    Install the RedactingFilter on the root logger.
    
    This function is idempotent - safe to call multiple times.
    It will only install the filter once per logger.
    """
    root_logger = logging.getLogger()
    
    # Check if filter is already installed (idempotence)
    filter_sentinel = '_liddy_redacting_filter_installed'
    if hasattr(root_logger, filter_sentinel):
        return
    
    # Install the filter
    redacting_filter = RedactingFilter()
    root_logger.addFilter(redacting_filter)
    
    # Mark as installed
    setattr(root_logger, filter_sentinel, True)
    
    # Configure basic logging if not already configured
    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )