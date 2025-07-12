"""
SearchPinecone - Pinecone Implementation of SearchRAG Interface

A Pinecone-specific implementation supporting both dense and sparse embeddings
with server-side embedding models and neural reranking.
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

from pinecone import Pinecone, ServerlessSpec

from .base import BaseRAG, SearchResult

logger = logging.getLogger(__name__)


class PineconeRAG(BaseRAG):
    """
    Unified search interface for RAG applications with hybrid search capabilities.
    
    Features:
    - Support for separate dense and sparse indexes
    - Server-side embeddings (no local model dependencies)
    - Configurable search modes (dense, sparse, hybrid)
    - Neural reranking support
    - Minimal external dependencies
    """
    
    def __init__(
        self,
        brand_domain: str,
        dense_index_name: Optional[str] = None,
        sparse_index_name: Optional[str] = None,
        api_key: Optional[str] = None,
        namespace: str = "products"
    ):
        """
        Initialize SearchRAG with flexible index configuration.
        
        Args:
            brand_domain: Brand/account identifier (e.g., "specialized.com")
            dense_index_name: Name of dense embedding index (auto-generated if not provided)
            sparse_index_name: Name of sparse embedding index (auto-generated if not provided)
            api_key: Pinecone API key
            namespace: Default namespace for searches
        """
        self.brand_domain = brand_domain
        self.namespace = namespace
        
        # Auto-generate index names if not provided
        brand_name = brand_domain.replace('.', '-')
        self.dense_index_name = dense_index_name or f"{brand_name}-dense"
        self.sparse_index_name = sparse_index_name or f"{brand_name}-sparse"
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=api_key or os.getenv("PINECONE_API_KEY"))
        
        # Index references (will be set in initialize method)
        self.dense_index = None
        self.sparse_index = None
        
        # Default search weights
        self.default_dense_weight = 0.7
        self.default_sparse_weight = 0.3
        
        # Configuration
        self.config = {
            "dense_model": "llama-text-embed-v2",
            "sparse_model": "pinecone-sparse-english-v0",
            "rerank_model": "bge-reranker-v2-m3",
            "cloud": "gcp",
            "region": "us-central1"
        }
        
        logger.info(f"🔍 Initialized SearchRAG for {brand_domain}")
        logger.info(f"  - Dense index: {self.dense_index_name}")
        logger.info(f"  - Sparse index: {self.sparse_index_name}")
    
    async def initialize(self) -> None:
        """Initialize indexes, creating them if they don't exist."""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        # Create dense index if needed
        if self.dense_index_name not in existing_indexes:
            logger.info(f"Creating dense index: {self.dense_index_name}")
            self.pc.create_index_for_model(
                name=self.dense_index_name,
                cloud=self.config["cloud"],
                region=self.config["region"],
                embed={
                    "model": self.config["dense_model"],
                    "field_map": {"text": "text"}
                }
            )
            logger.info(f"✅ Created dense index: {self.dense_index_name}")
        
        # Create sparse index if needed
        if self.sparse_index_name not in existing_indexes:
            logger.info(f"Creating sparse index: {self.sparse_index_name}")
            self.pc.create_index_for_model(
                name=self.sparse_index_name,
                cloud=self.config["cloud"],
                region=self.config["region"],
                embed={
                    "model": self.config["sparse_model"],
                    "field_map": {"text": "text"}
                }
            )
            logger.info(f"✅ Created sparse index: {self.sparse_index_name}")
        
        # Get index references
        self.dense_index = self.pc.Index(self.dense_index_name)
        self.sparse_index = self.pc.Index(self.sparse_index_name)
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        dense_weight: Optional[float] = None,
        sparse_weight: Optional[float] = None,
        search_mode: str = "hybrid",  # 'hybrid', 'dense', 'sparse'
        rerank: bool = True,
        namespace: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Perform search with flexible configuration.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Metadata filters
            dense_weight: Weight for dense results (auto-determined if None)
            sparse_weight: Weight for sparse results (auto-determined if None)
            search_mode: 'hybrid', 'dense', or 'sparse'
            rerank: Whether to apply neural reranking
            namespace: Override default namespace
            
        Returns:
            List of SearchResult objects
        """
        # Ensure indexes are initialized
        if self.dense_index is None or self.sparse_index is None:
            await self.initialize()
        
        # Use provided namespace or default
        namespace = namespace or self.namespace
        
        # Determine weights if not provided
        if dense_weight is None or sparse_weight is None:
            dense_weight, sparse_weight = self._determine_weights(query)
        
        # Normalize weights
        total_weight = dense_weight + sparse_weight
        if total_weight > 0:
            dense_weight /= total_weight
            sparse_weight /= total_weight
        
        logger.info(f"🔎 Search mode: {search_mode}, query: '{query[:50]}...'")
        logger.info(f"   Weights - dense: {dense_weight:.2f}, sparse: {sparse_weight:.2f}")
        
        # Execute searches based on mode
        results = []
        
        if search_mode in ["hybrid", "dense"] and dense_weight > 0:
            dense_results = await self._search_dense(
                query, 
                filters, 
                namespace,
                top_k * 2 if search_mode == "hybrid" else top_k,
                rerank and search_mode == "dense"
            )
            results.extend(dense_results)
        
        if search_mode in ["hybrid", "sparse"] and sparse_weight > 0:
            sparse_results = await self._search_sparse(
                query, 
                filters, 
                namespace,
                top_k * 2 if search_mode == "hybrid" else top_k,
                rerank and search_mode == "sparse"
            )
            results.extend(sparse_results)
        
        # Combine and rerank for hybrid search
        if search_mode == "hybrid":
            results = self._combine_results(
                results, 
                dense_weight, 
                sparse_weight,
                top_k
            )
            
            # Apply neural reranking if requested
            if rerank and len(results) > 0:
                results = await self._rerank_results(query, results, top_k)
        
        return results[:top_k]
    
    async def _search_dense(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]], 
        namespace: str,
        top_k: int,
        rerank: bool
    ) -> List[SearchResult]:
        """Search dense index with server-side embeddings."""
        try:
            from pinecone import SearchQuery
            
            # Build search query
            search_query = SearchQuery(
                inputs={"text": query},
                top_k=top_k * 2 if filters else top_k  # Get more if filtering locally
            )
            
            # Search with server-side embeddings
            results = self.dense_index.search_records(
                namespace=namespace,
                query=search_query,
                fields=["metadata", "text"]
            )
            
            # Process results
            search_results = []
            for hit in results.result.hits:
                metadata = self._parse_metadata(hit.fields.get('metadata', {}))
                
                # Apply local filtering if needed
                if filters and not self._matches_filters(metadata, filters):
                    continue
                
                # Add text to metadata if available
                text = hit.fields.get('text', '')
                if text and 'descriptor' not in metadata:
                    metadata['descriptor'] = text
                
                search_results.append(SearchResult(
                    id=hit._id,
                    score=hit._score,
                    metadata=metadata,
                    source="dense",
                    text=text,
                    debug_info={"reranked": False}
                ))
            
            logger.info(f"Dense search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []
    
    async def _search_sparse(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]], 
        namespace: str,
        top_k: int,
        rerank: bool
    ) -> List[SearchResult]:
        """Search sparse index with server-side embeddings."""
        try:
            from pinecone import SearchQuery
            
            # Build search query
            search_query = SearchQuery(
                inputs={"text": query},
                top_k=top_k * 2 if filters else top_k  # Get more if filtering locally
            )
            
            # Search with server-side embeddings
            results = self.sparse_index.search_records(
                namespace=namespace,
                query=search_query,
                fields=["metadata", "text"]
            )
            
            # Process results
            search_results = []
            for hit in results.result.hits:
                metadata = self._parse_metadata(hit.fields.get('metadata', {}))
                
                # Apply local filtering if needed
                if filters and not self._matches_filters(metadata, filters):
                    continue
                
                # Add text to metadata if available
                text = hit.fields.get('text', '')
                if text and 'descriptor' not in metadata:
                    metadata['descriptor'] = text
                
                search_results.append(SearchResult(
                    id=hit._id,
                    score=hit._score,
                    metadata=metadata,
                    source="sparse",
                    text=text,
                    debug_info={"reranked": False}
                ))
            
            logger.info(f"Sparse search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Sparse search failed: {e}")
            return []
    
    def _combine_results(
        self,
        results: List[SearchResult],
        dense_weight: float,
        sparse_weight: float,
        top_k: int
    ) -> List[SearchResult]:
        """Combine results from dense and sparse searches with weighted scoring."""
        # Group results by ID
        results_by_id: Dict[str, List[SearchResult]] = defaultdict(list)
        for result in results:
            results_by_id[result.id].append(result)
        
        # Combine scores for each unique ID
        combined_results = []
        for product_id, product_results in results_by_id.items():
            # Calculate weighted score
            weighted_score = 0.0
            dense_score = 0.0
            sparse_score = 0.0
            
            for result in product_results:
                if result.source == "dense":
                    dense_score = result.score
                    weighted_score += dense_weight * result.score
                elif result.source == "sparse":
                    sparse_score = result.score
                    weighted_score += sparse_weight * result.score
            
            # Use metadata from first result
            base_result = product_results[0]
            
            combined_results.append(SearchResult(
                id=product_id,
                score=weighted_score,
                metadata=base_result.metadata,
                source="hybrid",
                text=base_result.text,
                debug_info={
                    "dense_score": dense_score,
                    "sparse_score": sparse_score,
                    "dense_weight": dense_weight,
                    "sparse_weight": sparse_weight
                }
            ))
        
        # Sort by combined score
        combined_results.sort(key=lambda r: r.score, reverse=True)
        
        logger.info(f"Combined {len(results)} results into {len(combined_results)} unique items")
        
        return combined_results
    
    async def _rerank_results(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        """Apply neural reranking to results."""
        # Prepare documents for reranking
        documents = []
        for result in results:
            # Construct document text
            text_parts = []
            
            # Add product name and description
            if result.metadata.get('name'):
                text_parts.append(result.metadata['name'])
            if result.metadata.get('description'):
                text_parts.append(result.metadata['description'])
            if result.text:
                text_parts.append(result.text)
            elif result.metadata.get('descriptor'):
                text_parts.append(result.metadata['descriptor'])
            
            documents.append({
                "id": result.id,
                "text": " ".join(text_parts)
            })
        
        try:
            # Use Pinecone's inference API for reranking
            from pinecone import RerankResult
            reranked: RerankResult = self.pc.inference.rerank(
                model=self.config["rerank_model"],
                query=query,
                documents=documents,
                top_n=min(top_k, len(documents)),
                return_documents=False
            )
            
            # Update scores based on reranking
            reranked_results = []
            for item in reranked.rerank_result.data:
                # Find original result
                document = documents[item.index]
                result = next((r for r in results if r.id == document['id']), None)
                if result:
                    result.score = item.score
                    result.debug_info["reranked"] = True
                    result.debug_info["rerank_model"] = self.config["rerank_model"]
                    reranked_results.append(result)
            
            logger.info(f"Reranked {len(results)} results, returning top {len(reranked_results)}")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}, returning original order")
            return results
    
    def _determine_weights(self, query: str) -> Tuple[float, float]:
        """Determine weights using base class logic."""
        # Use base class implementation
        return self.determine_search_weights(query)
    
    def _parse_metadata(self, metadata: Any) -> Dict[str, Any]:
        """Parse metadata which might be a string or dict."""
        if isinstance(metadata, str):
            try:
                return json.loads(metadata)
            except:
                return {"raw": metadata}
        return metadata or {}
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches the given filters."""
        for key, filter_value in filters.items():
            metadata_value = metadata.get(key)
            
            # Handle missing metadata
            if metadata_value is None:
                return False
            
            # Handle different filter operators
            if isinstance(filter_value, dict):
                if '$in' in filter_value:
                    if metadata_value not in filter_value['$in']:
                        return False
                elif '$gte' in filter_value:
                    if not (isinstance(metadata_value, (int, float)) and metadata_value >= filter_value['$gte']):
                        return False
                elif '$lte' in filter_value:
                    if not (isinstance(metadata_value, (int, float)) and metadata_value <= filter_value['$lte']):
                        return False
                elif '$gt' in filter_value:
                    if not (isinstance(metadata_value, (int, float)) and metadata_value > filter_value['$gt']):
                        return False
                elif '$lt' in filter_value:
                    if not (isinstance(metadata_value, (int, float)) and metadata_value < filter_value['$lt']):
                        return False
            else:
                # Simple equality check
                if metadata_value != filter_value:
                    return False
        
        return True
    


# Singleton instance cache for performance
_search_instances: Dict[str, PineconeRAG] = {}


async def get_search_pinecone(
    brand_domain: str,
    namespace: str = "products",
    **kwargs
) -> PineconeRAG:
    """
    Get or create a SearchPinecone instance for a brand.
    
    Uses singleton pattern to avoid recreating indexes.
    """
    cache_key = f"{brand_domain}_{namespace}"
    
    if cache_key not in _search_instances:
        search_pinecone = PineconeRAG(
            brand_domain=brand_domain,
            namespace=namespace,
            **kwargs
        )
        await search_pinecone.initialize()
        _search_instances[cache_key] = search_pinecone
    
    return _search_instances[cache_key]