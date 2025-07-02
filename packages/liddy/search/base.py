"""
Base Search Interface for RAG Applications

Defines the abstract interface for search implementations, allowing
different backends (Pinecone, Elasticsearch, etc.) to be used interchangeably.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Standardized search result across all implementations."""
    id: str
    score: float
    metadata: Dict[str, Any]
    source: str  # 'dense', 'sparse', or 'hybrid'
    text: Optional[str] = None
    debug_info: Optional[Dict[str, Any]] = None


class BaseRAG(ABC):
    """
    Abstract base class for RAG search implementations.
    
    This interface defines the core search functionality that any
    search backend must implement.
    """
    
    def __init__(
        self,
        brand_domain: str,
        namespace: str = "products",
        **kwargs
    ):
        """
        Initialize the search interface.
        
        Args:
            brand_domain: Brand/account identifier
            namespace: Default namespace for searches
            **kwargs: Implementation-specific parameters
        """
        self.brand_domain = brand_domain
        self.namespace = namespace
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the search backend (create indexes, connections, etc.)."""
        pass
    
    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        search_mode: str = "hybrid",
        namespace: Optional[str] = None,
        **kwargs
    ) -> List[SearchResult]:
        """
        Perform a search operation.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filters: Metadata filters to apply
            search_mode: 'dense', 'sparse', or 'hybrid'
            namespace: Override default namespace
            **kwargs: Implementation-specific parameters
            
        Returns:
            List of SearchResult objects
        """
        pass
    
    async def search_products(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Convenience method for product search with legacy format.
        
        Returns results in dictionary format for compatibility.
        """
        results = await self.search(
            query=query,
            filters=filters,
            top_k=top_k,
            namespace="products",
            **kwargs
        )
        
        # Convert to legacy format
        return [
            {
                'id': r.id,
                'score': r.score,
                'metadata': r.metadata,
                'text': r.text or r.metadata.get('descriptor', ''),
                'source': r.source
            }
            for r in results
        ]
    
    async def search_knowledge(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Convenience method for knowledge base search.
        
        Returns results in dictionary format for compatibility.
        """
        results = await self.search(
            query=query,
            filters=filters,
            top_k=top_k,
            namespace="information",
            **kwargs
        )
        
        # Convert to legacy format
        return [
            {
                'id': r.id,
                'score': r.score,
                'metadata': r.metadata,
                'text': r.text or '',
                'type': 'knowledge'
            }
            for r in results
        ]
    
    def determine_search_weights(self, query: str) -> Tuple[float, float]:
        """
        Determine dense vs sparse weights based on query characteristics.
        
        Can be overridden by implementations for custom logic.
        
        Returns:
            Tuple of (dense_weight, sparse_weight)
        """
        query_lower = query.lower()
        
        # Check for exact match indicators
        exact_indicators = [
            'exact', 'specifically', 'model', 'sku', '"', 
            'item number', 'part number'
        ]
        
        if any(indicator in query_lower for indicator in exact_indicators):
            # Favor sparse for exact searches
            return 0.3, 0.7
        
        # Check for semantic indicators
        semantic_indicators = [
            'like', 'similar', 'comfortable', 'quality', 'best', 
            'recommend', 'looking for', 'need', 'want', 'suggest'
        ]
        
        if any(indicator in query_lower for indicator in semantic_indicators):
            # Favor dense for semantic searches
            return 0.8, 0.2
        
        # Default balanced approach
        return 0.7, 0.3