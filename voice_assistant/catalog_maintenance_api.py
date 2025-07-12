"""
Catalog Maintenance API Wrapper

This module provides a clean interface for the voice assistant to interact
with the catalog-maintenance RAG service. It handles API calls for hybrid
search, filter extraction, and catalog synchronization.
"""

import os
import logging
import aiohttp
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


class CatalogMaintenanceAPI:
    """
    API client for interacting with the catalog-maintenance service.
    
    This wrapper provides methods for:
    - Hybrid search (dense + sparse embeddings)
    - Filter dictionary retrieval
    - Sync status monitoring
    - Health checks
    """
    
    def __init__(self, base_url: str = None, api_key: str = None, timeout: int = 30):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the catalog-maintenance service
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url or os.getenv("CATALOG_MAINTENANCE_API_URL", "http://localhost:8000")
        self.api_key = api_key or os.getenv("CATALOG_MAINTENANCE_API_KEY", "")
        self.timeout = timeout
        self._session = None
        
    @property
    def session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session
    
    async def close(self):
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def health_check(self) -> bool:
        """
        Check if the catalog-maintenance service is healthy.
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            url = urljoin(self.base_url, "/health")
            async with self.session.get(url) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def hybrid_search(
        self,
        query: str,
        account: str,
        filters: Optional[Dict[str, Any]] = None,
        dense_weight: Optional[float] = None,
        sparse_weight: Optional[float] = None,
        user_context: Optional[Dict[str, Any]] = None,
        top_k: int = 35,
        top_n: int = 10,
        min_score: float = 0.15
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using dense and sparse embeddings.
        
        Each brand has its own dedicated index, so no brand filtering is needed.
        The account parameter is used only to select the correct index.
        
        Args:
            query: Search query
            account: Brand domain (e.g., "specialized.com") - determines index
            filters: Product filters to apply (category, price, etc.)
            dense_weight: Weight for dense embeddings (0.0-1.0)
            sparse_weight: Weight for sparse embeddings (0.0-1.0)
            user_context: User preferences and context
            top_k: Number of candidates to retrieve
            top_n: Number of results to return after reranking
            min_score: Minimum score threshold
            
        Returns:
            List of search results with metadata and scores
        """
        try:
            url = urljoin(self.base_url, "/api/search")
            
            # Build request payload
            payload = {
                "query": query,
                "brand_domain": account,
                "filters": filters or {},
                "top_k": top_k,
                "top_n": top_n,
                "min_score": min_score
            }
            
            # Add optional parameters
            if dense_weight is not None:
                payload["dense_weight"] = dense_weight
            if sparse_weight is not None:
                payload["sparse_weight"] = sparse_weight
            if user_context:
                payload["user_context"] = user_context
            
            logger.info(f"Hybrid search request: account={account}, query='{query}', weights=({dense_weight}, {sparse_weight})")
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    results = data.get("results", [])
                    logger.info(f"Hybrid search returned {len(results)} results")
                    return results
                else:
                    error_text = await response.text()
                    logger.error(f"Hybrid search failed: {response.status} - {error_text}")
                    return []
                    
        except asyncio.TimeoutError:
            logger.error(f"Hybrid search timed out for query: {query}")
            return []
        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            return []
    
    async def get_filter_dictionary(self, account: str) -> Dict[str, List[str]]:
        """
        Get the available filters for a brand's catalog.
        
        Args:
            account: Brand domain (e.g., "specialized.com")
            
        Returns:
            Dictionary of filter names to possible values
        """
        try:
            url = urljoin(self.base_url, f"/api/filters/{account}")
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("filters", {})
                else:
                    logger.error(f"Failed to get filters for {account}: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error getting filter dictionary: {e}")
            return {}
    
    async def get_sync_status(self, account: str) -> Dict[str, Any]:
        """
        Get the synchronization status for a brand's catalog.
        
        Args:
            account: Brand domain (e.g., "specialized.com")
            
        Returns:
            Dictionary with sync status information
        """
        try:
            url = urljoin(self.base_url, f"/api/sync/status/{account}")
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get sync status for {account}: {response.status}")
                    return {"status": "unknown", "error": f"HTTP {response.status}"}
                    
        except Exception as e:
            logger.error(f"Error getting sync status: {e}")
            return {"status": "error", "error": str(e)}
    
    async def trigger_sync(self, account: str, catalog_path: str) -> bool:
        """
        Trigger a catalog synchronization.
        
        Args:
            account: Brand domain (e.g., "specialized.com")
            catalog_path: Path to the catalog file
            
        Returns:
            True if sync was triggered successfully
        """
        try:
            url = urljoin(self.base_url, "/api/sync/trigger")
            
            payload = {
                "brand_domain": account,
                "catalog_path": catalog_path
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    logger.info(f"Sync triggered successfully for {account}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to trigger sync: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error triggering sync: {e}")
            return False


# Singleton instance
_api_client: Optional[CatalogMaintenanceAPI] = None


def get_catalog_api() -> CatalogMaintenanceAPI:
    """
    Get the singleton catalog maintenance API client.
    
    Returns:
        CatalogMaintenanceAPI instance
    """
    global _api_client
    if _api_client is None:
        _api_client = CatalogMaintenanceAPI()
    return _api_client


async def search_products_hybrid(
    query: str,
    account: str,
    filters: Optional[Dict[str, Any]] = None,
    dense_weight: Optional[float] = None,
    sparse_weight: Optional[float] = None,
    user_context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Convenience function for hybrid product search.
    
    This function uses the singleton API client to perform hybrid search
    and can be easily integrated into existing search flows.
    """
    api = get_catalog_api()
    
    # Check if service is available
    if not await api.health_check():
        logger.warning("Catalog maintenance service unavailable, returning empty results")
        return []
    
    return await api.hybrid_search(
        query=query,
        account=account,
        filters=filters,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        user_context=user_context,
        **kwargs
    )