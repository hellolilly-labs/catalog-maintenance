"""
Unified Search Service that integrates with liddy.search components
while maintaining backward compatibility with existing voice assistant code.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from liddy.search.service import SearchService as LiddySearchService
from liddy.search.pinecone import PineconeRAG as LiddyPineconeRAG
from liddy.models.product import Product as LiddyProduct

logger = logging.getLogger(__name__)


class UnifiedSearchService:
    """
    Adapter class that provides backward compatibility for voice assistant
    while using the new unified search components from liddy package.
    """
    
    def __init__(self, account: str, index_name: Optional[str] = None):
        """
        Initialize the unified search service.
        
        Args:
            account: Brand account name (e.g., "specialized.com")
            index_name: Optional explicit index name override
        """
        self.account = account
        self.index_name = index_name
        self._search_service = None
        self._pinecone_rag = None
        
    async def initialize(self):
        """Initialize the underlying search components."""
        # Determine the index name
        if not self.index_name:
            # Extract brand name from account domain
            brand_name = self.account.split('.')[0]
            
            # Try new index format first, then fall back to legacy format
            self.index_name = f"{brand_name}-llama-2048"
            
        # Initialize the PineconeRAG instance
        self._pinecone_rag = LiddyPineconeRAG(
            pinecone_api_key=None,  # Will use env var
            index_name=self.index_name,
            namespace=self.account.split('.')[0]  # Use brand name as namespace
        )
        
        # Initialize the search service
        self._search_service = LiddySearchService(
            rag_backend=self._pinecone_rag,
            default_namespace=self.account.split('.')[0]
        )
        
        logger.info(f"Initialized UnifiedSearchService for {self.account} with index {self.index_name}")
        
    async def search_products(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search for products using the unified search service.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional metadata filters
            **kwargs: Additional arguments passed to search
            
        Returns:
            List of product dictionaries
        """
        if not self._search_service:
            await self.initialize()
            
        try:
            # Use the unified search service
            results = await self._search_service.search(
                query=query,
                top_k=top_k,
                filters=filters,
                search_mode="hybrid",  # Use hybrid search by default
                **kwargs
            )
            
            # Convert results to expected format for voice assistant
            products = []
            for result in results:
                # Extract product data from metadata
                product_data = result.metadata.copy()
                product_data['score'] = result.score
                product_data['relevance_score'] = result.score
                
                # Ensure required fields exist
                if 'name' not in product_data and 'title' in product_data:
                    product_data['name'] = product_data['title']
                if 'product_id' not in product_data and 'id' in product_data:
                    product_data['product_id'] = product_data['id']
                    
                products.append(product_data)
                
            return products
            
        except Exception as e:
            logger.error(f"Error searching products: {e}")
            return []
            
    async def get_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific product by ID."""
        results = await self.search_products(
            query=f"product_id:{product_id}",
            top_k=1,
            filters={"product_id": product_id}
        )
        return results[0] if results else None
        
    async def close(self):
        """Close any open connections."""
        # PineconeRAG doesn't currently have a close method
        pass


# Backward compatibility class that mimics the old PineconeRAG interface
class PineconeRAG:
    """
    Backward compatibility wrapper for the old PineconeRAG interface.
    This allows existing voice assistant code to work without modifications.
    """
    
    def __init__(
        self,
        account: str,
        index_name: str,
        model_name: str = "llama-text-embed-v2",
        namespace: str = "specialized",
        connection_timeout: float = 10.0,
        debug: bool = False
    ):
        """Initialize with backward compatible interface."""
        self.account = account
        self.index_name = index_name
        self.namespace = namespace
        self.model_name = model_name
        self._unified_search = UnifiedSearchService(account, index_name)
        
    async def init(self):
        """Initialize the search service."""
        await self._unified_search.initialize()
        
    async def search(
        self,
        query: str,
        top_k: int = 10,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search with backward compatible interface."""
        return await self._unified_search.search_products(
            query=query,
            top_k=top_k,
            filters=filter,
            namespace=namespace or self.namespace,
            **kwargs
        )
        
    async def search_products(self, *args, **kwargs):
        """Alias for search method."""
        return await self.search(*args, **kwargs)