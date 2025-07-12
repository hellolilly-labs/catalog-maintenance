"""
Unified RAG implementation that uses the liddy.search components
while maintaining backward compatibility with existing voice assistant code.
"""

import os
import asyncio
import logging
import time
from datetime import datetime
from typing import List, Dict, Optional, Union, Any

from liddy.search.pinecone import PineconeRAG as LiddyPineconeRAG
from liddy.search.service import SearchService as LiddySearchService
from liddy_voice.account_manager import get_account_manager

logger = logging.getLogger(__name__)


class PineconeRAG:
    """
    Backward-compatible RAG implementation that uses the unified search components.
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
        """
        Initialize the RAG system with backward-compatible interface.
        
        Args:
            account: Brand account (e.g., "specialized.com")
            index_name: Name of the Pinecone index
            model_name: Embedding model name (for compatibility)
            namespace: Default namespace
            connection_timeout: Connection timeout in seconds
            debug: Debug mode flag
        """
        self.account = account
        self.index_name = index_name
        self.namespace = namespace
        self.model = model_name
        self.connection_timeout = connection_timeout
        self.debug = debug
        
        # Initialize unified components
        self._pinecone_rag = None
        self._search_service = None
        
        # Stats for backward compatibility
        self.stats = {
            "total_vectors": 0,
            "namespaces": [],
            "connection_time": 0
        }
        self.available_namespaces = []
        
        # Request tracking
        self._requests_processed = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._total_query_time = 0.0
        
        # Asyncio lock
        self._lock = asyncio.Lock()
        
        # Start initialization
        asyncio.create_task(self.init())
        
    async def init(self):
        """Initialize the RAG system."""
        try:
            # Get account configuration
            account_manager = await get_account_manager(account=self.account)
            configured_index_name, embedding_model = account_manager.get_rag_details()
            
            # Use configured index name if available, otherwise use provided
            if configured_index_name:
                self.index_name = configured_index_name
                logger.info(f"Using configured index name: {self.index_name}")
            else:
                # Try legacy index format as fallback
                brand_name = self.account.split('.')[0]
                legacy_index = f"{brand_name}-llama-2048"
                logger.info(f"No configured index, trying legacy format: {legacy_index}")
                self.index_name = legacy_index
            
            # Initialize PineconeRAG
            start_time = time.time()
            self._pinecone_rag = LiddyPineconeRAG(
                pinecone_api_key=os.getenv('PINECONE_API_KEY'),
                index_name=self.index_name,
                namespace=self.namespace
            )
            
            # Initialize search service
            self._search_service = LiddySearchService(
                rag_backend=self._pinecone_rag,
                default_namespace=self.namespace
            )
            
            # Update stats
            conn_time = time.time() - start_time
            self.stats["connection_time"] = conn_time
            self.connection_time = datetime.now().isoformat()
            self.initialization_timestamp = int(time.time())
            
            # Try to get index stats (if the index exists)
            try:
                # This might fail if index doesn't exist yet
                self.available_namespaces = [self.namespace]
                self.stats["namespaces"] = self.available_namespaces
                logger.info(f"Connected to index '{self.index_name}' in {conn_time:.2f}s")
            except Exception as e:
                logger.warning(f"Could not get index stats: {e}")
                
        except Exception as e:
            logger.error(f"Failed to initialize RAG: {e}")
            raise
            
    async def search(
        self,
        query: str,
        intent: str = "product",
        namespace: Optional[str] = None,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search with backward-compatible interface.
        
        Args:
            query: Search query
            intent: Search intent (product, information, etc.)
            namespace: Override namespace
            top_k: Number of results
            filter: Metadata filters
            include_metadata: Include metadata in results
            **kwargs: Additional arguments
            
        Returns:
            List of search results in backward-compatible format
        """
        if not self._search_service:
            # Wait for initialization
            await asyncio.sleep(0.5)
            if not self._search_service:
                raise RuntimeError("Search service not initialized")
                
        async with self._lock:
            self._requests_processed += 1
            start_time = time.time()
            
            try:
                # Use the unified search service
                results = await self._search_service.search(
                    query=query,
                    top_k=top_k,
                    filters=filter,
                    namespace=namespace or self.namespace,
                    search_mode="hybrid",  # Always use hybrid for better results
                    **kwargs
                )
                
                # Convert to backward-compatible format
                formatted_results = []
                for result in results:
                    formatted_result = {
                        "id": result.id,
                        "score": result.score,
                        "values": [],  # Not used in voice assistant
                        "metadata": result.metadata if include_metadata else {}
                    }
                    
                    # Add backward compatibility fields
                    if result.metadata:
                        # Ensure expected fields exist
                        metadata = result.metadata.copy()
                        if 'name' not in metadata and 'title' in metadata:
                            metadata['name'] = metadata['title']
                        if 'product_id' not in metadata and 'id' in metadata:
                            metadata['product_id'] = metadata['id']
                        formatted_result["metadata"] = metadata
                        
                    formatted_results.append(formatted_result)
                
                # Update stats
                self._successful_requests += 1
                query_time = time.time() - start_time
                self._total_query_time += query_time
                
                if self.debug:
                    logger.debug(f"Search completed in {query_time:.2f}s, found {len(formatted_results)} results")
                    
                return formatted_results
                
            except Exception as e:
                self._failed_requests += 1
                logger.error(f"Search failed: {e}")
                raise
                
    async def search_products(
        self,
        query: str,
        namespace: Optional[str] = None,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search specifically for products."""
        return await self.search(
            query=query,
            intent="product",
            namespace=namespace,
            top_k=top_k,
            filter=filter,
            **kwargs
        )
        
    async def search_information(
        self,
        query: str,
        namespace: Optional[str] = None,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for informational content."""
        return await self.search(
            query=query,
            intent="information",
            namespace=namespace,
            top_k=top_k,
            **kwargs
        )
        
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG statistics."""
        avg_query_time = (
            self._total_query_time / self._successful_requests
            if self._successful_requests > 0
            else 0
        )
        
        return {
            "requests_processed": self._requests_processed,
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
            "average_query_time": avg_query_time,
            "index_name": self.index_name,
            "namespace": self.namespace,
            "initialization_time": self.stats.get("connection_time", 0)
        }