"""
Unified Search Service

A minimal-dependency search service that works across both catalog-maintenance
and voice-assistant projects. Uses AccountManager for configuration and
SearchRAG interface for search operations.
"""

import logging
import time
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from liddy.account_manager import get_account_manager
from .base import BaseRAG
from .pinecone import PineconeRAG

logger = logging.getLogger(__name__)


@dataclass
class SearchMetrics:
    """Metrics for search operations."""
    query: str
    enhanced_query: str
    search_time: float
    total_results: int
    filters_applied: Dict[str, Any]
    search_backend: str
    enhancements_used: List[str]


class SearchService:
    """
    Unified search service with minimal dependencies.
    
    Features:
    - Query enhancement using AccountManager intelligence
    - Filter extraction from natural language
    - Multiple search backend support (via SearchRAG interface)
    - Metrics and performance tracking
    """
    
    # Cache for search instances per account
    _search_instances: Dict[str, BaseRAG] = {}
    
    @classmethod
    async def search_products(
        cls,
        query: str,
        account: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        enable_enhancement: bool = True,
        enable_filter_extraction: bool = True,
        search_mode: str = "hybrid",
        user_context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], SearchMetrics]:
        """
        Search for products with intelligent enhancements.
        
        Args:
            query: Search query
            account: Account/brand domain
            filters: Pre-extracted filters (optional)
            top_k: Number of results to return
            enable_enhancement: Whether to enhance query with catalog intelligence
            enable_filter_extraction: Whether to extract filters from query
            search_mode: 'dense', 'sparse', or 'hybrid'
            user_context: Optional user context for personalization
            **kwargs: Additional search parameters
            
        Returns:
            Tuple of (search results, metrics)
        """
        start_time = time.time()
        enhancements_used = []
        
        # Get account manager
        account_manager = await get_account_manager(account)
        
        # Original query for metrics
        original_query = query
        
        # Extract filters if enabled and not provided
        if enable_filter_extraction and not filters:
            filters = cls._extract_filters(query, account_manager)
            if filters:
                enhancements_used.append("filter_extraction")
        
        # Enhance query if enabled
        if enable_enhancement:
            query = await cls._enhance_query(
                query, 
                account_manager,
                user_context
            )
            if query != original_query:
                enhancements_used.append("query_enhancement")
        
        # Get or create search instance
        search_instance = await cls._get_search_instance(account_manager)
        
        # Perform search
        results = await search_instance.search_products(
            query=query,
            filters=filters,
            top_k=top_k,
            search_mode=search_mode,
            **kwargs
        )
        
        # Create metrics
        metrics = SearchMetrics(
            query=original_query,
            enhanced_query=query,
            search_time=time.time() - start_time,
            total_results=len(results),
            filters_applied=filters or {},
            search_backend=search_instance.__class__.__name__,
            enhancements_used=enhancements_used
        )
        
        logger.info(
            f"Search completed for {account}: {len(results)} results in {metrics.search_time:.3f}s "
            f"(enhancements: {', '.join(enhancements_used) or 'none'})"
        )
        
        return results, metrics
    
    @classmethod
    async def search_knowledge(
        cls,
        query: str,
        account: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        enable_enhancement: bool = True,
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], SearchMetrics]:
        """
        Search knowledge base articles.
        
        Similar to search_products but for knowledge/information namespace.
        """
        start_time = time.time()
        enhancements_used = []
        
        # Get account manager
        account_manager = await get_account_manager(account)
        
        # Original query for metrics
        original_query = query
        
        # Enhance query if enabled
        if enable_enhancement:
            query = await cls._enhance_query(
                query,
                account_manager,
                context_type="knowledge"
            )
            if query != original_query:
                enhancements_used.append("query_enhancement")
        
        # Get or create search instance
        search_instance = await cls._get_search_instance(account_manager)
        
        # Perform search
        results = await search_instance.search_knowledge(
            query=query,
            filters=filters,
            top_k=top_k,
            **kwargs
        )
        
        # Create metrics
        metrics = SearchMetrics(
            query=original_query,
            enhanced_query=query,
            search_time=time.time() - start_time,
            total_results=len(results),
            filters_applied=filters or {},
            search_backend=search_instance.__class__.__name__,
            enhancements_used=enhancements_used
        )
        
        return results, metrics
    
    @classmethod
    async def _get_search_instance(cls, account_manager) -> BaseRAG:
        """Get or create search instance based on account configuration."""
        account = account_manager.account
        
        if account not in cls._search_instances:
            search_config = account_manager.get_search_config()
            
            if search_config.backend == "pinecone":
                # Create Pinecone search instance
                cls._search_instances[account] = PineconeRAG(
                    brand_domain=account,
                    dense_index_name=search_config.dense_index,
                    sparse_index_name=search_config.sparse_index,
                    **search_config.config
                )
                await cls._search_instances[account].initialize()
                logger.info(f"Created SearchPinecone instance for {account}")
            else:
                raise ValueError(f"Unsupported search backend: {search_config.backend}")
        
        return cls._search_instances[account]
    
    @classmethod
    async def _enhance_query(
        cls,
        query: str,
        account_manager,
        user_context: Optional[Any] = None,
        context_type: str = "product"
    ) -> str:
        """Enhance query using catalog intelligence."""
        try:
            if context_type == "product":
                enhancement_context = await account_manager.get_query_enhancement_context()
            else:
                # For knowledge queries, use brand insights
                intelligence = await account_manager.get_catalog_intelligence()
                enhancement_context = intelligence.get('brand_insights', '')
            
            if not enhancement_context:
                return query
            
            # Simple enhancement: Add brand context to ambiguous queries
            query_lower = query.lower()
            
            # Check if query is very short or generic
            if len(query.split()) <= 2 or any(
                generic in query_lower 
                for generic in ['best', 'good', 'nice', 'product', 'item']
            ):
                # Extract key terms from context
                context_terms = []
                if 'specializes in' in enhancement_context:
                    match = re.search(r'specializes in ([^.]+)', enhancement_context)
                    if match:
                        context_terms.append(match.group(1).strip())
                
                if context_terms and not any(term in query_lower for term in context_terms):
                    # Enhance with context
                    return f"{query} {' '.join(context_terms[:1])}"
            
            return query
            
        except Exception as e:
            logger.error(f"Error enhancing query: {e}")
            return query
    
    @classmethod
    def _extract_filters(
        cls,
        query: str,
        account_manager
    ) -> Dict[str, Any]:
        """Extract filters from natural language query."""
        filters = {}
        query_lower = query.lower()
        
        # Price extraction
        price_patterns = [
            (r'under \$?(\d+(?:,\d{3})*(?:\.\d{2})?)', 'max'),
            (r'below \$?(\d+(?:,\d{3})*(?:\.\d{2})?)', 'max'),
            (r'less than \$?(\d+(?:,\d{3})*(?:\.\d{2})?)', 'max'),
            (r'over \$?(\d+(?:,\d{3})*(?:\.\d{2})?)', 'min'),
            (r'above \$?(\d+(?:,\d{3})*(?:\.\d{2})?)', 'min'),
            (r'more than \$?(\d+(?:,\d{3})*(?:\.\d{2})?)', 'min'),
            (r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?) ?- ?\$?(\d+(?:,\d{3})*(?:\.\d{2})?)', 'range')
        ]
        
        for pattern, price_type in price_patterns:
            match = re.search(pattern, query_lower)
            if match:
                if price_type == 'range':
                    min_price = float(match.group(1).replace(',', ''))
                    max_price = float(match.group(2).replace(',', ''))
                    filters['price'] = {'$gte': min_price, '$lte': max_price}
                elif price_type == 'max':
                    filters['price'] = {'$lte': float(match.group(1).replace(',', ''))}
                else:  # min
                    filters['price'] = {'$gte': float(match.group(1).replace(',', ''))}
                break
        
        # Category extraction based on account
        brand_config = account_manager.get_brand_config()
        if account_manager.account == "specialized.com":
            categories = {
                'mountain': 'Mountain',
                'road': 'Road',
                'gravel': 'Gravel',
                'electric': 'E-Bike',
                'kids': 'Kids'
            }
        elif account_manager.account == "sundayriley.com":
            categories = {
                'serum': 'Serum',
                'moisturizer': 'Moisturizer',
                'cleanser': 'Cleanser',
                'mask': 'Mask',
                'oil': 'Oil'
            }
        else:
            categories = {}
        
        for keyword, category in categories.items():
            if keyword in query_lower:
                filters['category'] = category
                break
        
        # Variant extraction
        variants = account_manager.get_variant_types()
        for variant in variants:
            variant_name = variant['name'].lower()
            if variant_name in query_lower:
                # Extract variant value (simplified)
                words = query_lower.split()
                for i, word in enumerate(words):
                    if word == variant_name and i + 1 < len(words):
                        filters[f"variant_{variant_name}"] = words[i + 1]
                        break
        
        return filters
    
    @classmethod
    def determine_search_weights(
        cls,
        query: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, float]:
        """
        Determine dense vs sparse weights for hybrid search.
        
        Args:
            query: Search query
            filters: Extracted filters
            
        Returns:
            Tuple of (dense_weight, sparse_weight)
        """
        # If filters are present, slightly favor sparse
        if filters:
            return 0.6, 0.4
        
        # Otherwise use query characteristics
        query_lower = query.lower()
        
        # Exact match indicators
        if any(indicator in query_lower for indicator in ['exact', 'model', 'sku']):
            return 0.3, 0.7
        
        # Semantic indicators
        if any(indicator in query_lower for indicator in ['comfortable', 'quality', 'best']):
            return 0.8, 0.2
        
        # Default balanced
        return 0.7, 0.3


# Convenience functions for backward compatibility
async def search_products(query: str, account: str, **kwargs) -> List[Dict[str, Any]]:
    """Simple product search without metrics."""
    results, _ = await SearchService.search_products(query, account, **kwargs)
    return results


async def search_knowledge(query: str, account: str, **kwargs) -> List[Dict[str, Any]]:
    """Simple knowledge search without metrics."""
    results, _ = await SearchService.search_knowledge(query, account, **kwargs)
    return results