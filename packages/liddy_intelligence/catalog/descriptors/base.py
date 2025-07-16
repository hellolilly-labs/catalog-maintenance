"""Base class for descriptor enhancement modules."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from liddy.models.product import Product


class BaseDescriptorModule(ABC):
    """Abstract base class for descriptor enhancement modules.
    
    Each module can modify the descriptor text, add search keywords,
    and provide metadata for a product.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the module with optional configuration."""
        self.config = config or {}
    
    @abstractmethod
    def enhance_descriptor(self, descriptor: str, product: Product, **kwargs) -> str:
        """Enhance the descriptor text.
        
        Args:
            descriptor: Current descriptor text
            product: Product being processed
            **kwargs: Additional context (e.g., price_statistics)
            
        Returns:
            Enhanced descriptor text
        """
        pass
    
    @abstractmethod
    def enhance_search_keywords(self, keywords: List[str], product: Product, **kwargs) -> List[str]:
        """Add module-specific search keywords.
        
        Args:
            keywords: Current list of keywords
            product: Product being processed
            **kwargs: Additional context
            
        Returns:
            Enhanced list of keywords
        """
        pass
    
    @abstractmethod
    def get_metadata(self, product: Product, **kwargs) -> Dict[str, Any]:
        """Return module-specific metadata.
        
        Args:
            product: Product being processed
            **kwargs: Additional context
            
        Returns:
            Dictionary of metadata
        """
        pass
    
    @abstractmethod
    def is_applicable(self, product: Product) -> bool:
        """Check if this module should process the given product.
        
        Args:
            product: Product to check
            
        Returns:
            True if module should process this product
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Module name for identification."""
        pass
    
    @property
    def priority(self) -> int:
        """Module execution priority (lower runs first).
        
        Default is 100. Override in subclasses to control execution order.
        """
        return 100