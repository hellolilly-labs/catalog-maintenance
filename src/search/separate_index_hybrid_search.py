"""
Separate Index Hybrid Search Implementation

This module implements Pinecone's recommended best practice of using
separate indexes for dense and sparse embeddings, providing maximum
flexibility and control.
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

from pinecone import Pinecone, SearchQuery, SearchRerank, RerankModel
import numpy as np

from ..ingestion.sparse_embeddings import SparseEmbeddingGenerator

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Structured search result."""
    id: str
    score: float
    metadata: Dict[str, Any]
    source: str  # 'dense', 'sparse', or 'hybrid'
    debug_info: Optional[Dict[str, Any]] = None


class SeparateIndexHybridSearch:
    """
    Hybrid search using separate dense and sparse indexes.
    
    This implementation follows Pinecone's best practices:
    - Separate indexes for dense and sparse embeddings
    - Consistent ID linkage between indexes
    - Support for sparse-only queries
    - Multi-stage reranking capability
    """
    
    def __init__(
        self,
        brand_domain: str,
        dense_index_name: str,
        sparse_index_name: str,
        api_key: Optional[str] = None
    ):
        """
        Initialize hybrid search with separate indexes.
        
        Args:
            brand_domain: Brand domain (e.g., "specialized.com")
            dense_index_name: Name of dense embedding index
            sparse_index_name: Name of sparse embedding index
            api_key: Pinecone API key
        """
        self.brand_domain = brand_domain
        self.dense_index_name = dense_index_name
        self.sparse_index_name = sparse_index_name
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=api_key or os.getenv("PINECONE_API_KEY"))
        
        # Get separate indexes
        self.dense_index = self.pc.Index(dense_index_name)
        self.sparse_index = self.pc.Index(sparse_index_name)
        
        # Initialize sparse embedding generator
        self.sparse_generator = SparseEmbeddingGenerator(brand_domain)
        
        # Default namespace
        self.namespace = "products"
        
        # Default weights
        self.default_dense_weight = 0.7
        self.default_sparse_weight = 0.3
        
        logger.info(f"🔍 Initialized Separate Index Hybrid Search for {brand_domain}")
        logger.info(f"  - Dense index: {dense_index_name}")
        logger.info(f"  - Sparse index: {sparse_index_name}")
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        dense_weight: Optional[float] = None,
        sparse_weight: Optional[float] = None,
        search_mode: str = "hybrid",  # 'hybrid', 'dense', 'sparse'
        rerank: bool = True
    ) -> List[SearchResult]:
        """
        Perform search using separate indexes with neural reranking.
        
        Args:
            query: Search query
            top_k: Number of results to return after reranking
            filters: Metadata filters to apply
            dense_weight: Weight for dense results (0-1)
            sparse_weight: Weight for sparse results (0-1)
            search_mode: Search mode - 'hybrid', 'dense', or 'sparse'
            rerank: Whether to use neural reranking
            
        Returns:
            List of search results, reranked if requested
        """
        # Normalize weights
        if dense_weight is None or sparse_weight is None:
            dense_weight, sparse_weight = self._determine_weights(query)
        
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
                top_k * 2 if search_mode == "hybrid" else top_k,
                rerank and search_mode == "dense"
            )
            results.extend(dense_results)
        
        if search_mode in ["hybrid", "sparse"] and sparse_weight > 0:
            sparse_results = await self._search_sparse(
                query, 
                filters, 
                top_k * 2 if search_mode == "hybrid" else top_k,
                rerank and search_mode == "sparse"
            )
            results.extend(sparse_results)
        
        # Combine results for hybrid search
        if search_mode == "hybrid":
            results = self._combine_results(
                results, 
                dense_weight, 
                sparse_weight,
                top_k
            )
            
            # Apply neural reranking to combined results
            if rerank and len(results) > 0:
                results = await self._rerank_combined_results(query, results, top_k)
        
        return results[:top_k]
    
    async def _search_dense(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]], 
        top_k: int,
        rerank: bool
    ) -> List[SearchResult]:
        """Search dense index with optional reranking."""
        
        # Build search query
        search_query = SearchQuery(
            inputs={"text": query},
            top_k=top_k
        )
        
        # Configure reranking if requested
        rerank_config = None
        if rerank:
            rerank_config = SearchRerank(
                model=RerankModel.Cohere_Rerank_3_5,
                rank_fields=["text"],
                top_n=top_k
            )
        
        # Convert filters
        filter_dict = self._build_pinecone_filter(filters) if filters else None
        
        try:
            # Search dense index
            results = self.dense_index.search_records(
                namespace=self.namespace,
                query=search_query,
                rerank=rerank_config,
                filter=filter_dict,
                fields=["metadata"]
            )
            
            # Process results
            search_results = []
            for hit in results.result.hits:
                metadata = hit.fields.get('metadata', {})
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                
                search_results.append(SearchResult(
                    id=hit._id,
                    score=hit._score,
                    metadata=metadata,
                    source="dense",
                    debug_info={"reranked": rerank}
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
        top_k: int,
        rerank: bool
    ) -> List[SearchResult]:
        """Search sparse index with optional reranking."""
        
        # Generate sparse representation for query
        sparse_values = self._generate_query_sparse_values(query)
        
        if not sparse_values or not sparse_values.get('indices'):
            logger.warning("No sparse values generated for query")
            return []
        
        # Build search query with sparse values
        search_query = SearchQuery(
            sparse_values=sparse_values,
            top_k=top_k
        )
        
        # Configure reranking if requested
        rerank_config = None
        if rerank:
            rerank_config = SearchRerank(
                model=RerankModel.Cohere_Rerank_3_5,
                rank_fields=["text"],
                top_n=top_k
            )
        
        # Convert filters
        filter_dict = self._build_pinecone_filter(filters) if filters else None
        
        try:
            # Search sparse index
            results = self.sparse_index.search_records(
                namespace=self.namespace,
                query=search_query,
                rerank=rerank_config,
                filter=filter_dict,
                fields=["metadata"]
            )
            
            # Process results
            search_results = []
            for hit in results.result.hits:
                metadata = hit.fields.get('metadata', {})
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                
                search_results.append(SearchResult(
                    id=hit._id,
                    score=hit._score,
                    metadata=metadata,
                    source="sparse",
                    debug_info={"reranked": rerank}
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
        """
        Combine results from dense and sparse searches.
        
        Uses consistent ID linkage to merge results and compute
        weighted scores.
        """
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
            
            # Use metadata from first result (should be same across indexes)
            metadata = product_results[0].metadata
            
            combined_results.append(SearchResult(
                id=product_id,
                score=weighted_score,
                metadata=metadata,
                source="hybrid",
                debug_info={
                    "dense_score": dense_score,
                    "sparse_score": sparse_score,
                    "dense_weight": dense_weight,
                    "sparse_weight": sparse_weight
                }
            ))
        
        # Sort by combined score
        combined_results.sort(key=lambda r: r.score, reverse=True)
        
        logger.info(f"Combined {len(results)} results into {len(combined_results)} unique products")
        
        return combined_results
    
    async def _rerank_combined_results(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        """
        Apply neural reranking to combined results.
        
        This is a second-stage reranking after combining dense and sparse results.
        """
        # Prepare documents for reranking
        documents = []
        for result in results:
            # Construct document text from metadata
            text_parts = []
            
            # Add product name and description
            if result.metadata.get('name'):
                text_parts.append(result.metadata['name'])
            if result.metadata.get('description'):
                text_parts.append(result.metadata['description'])
            if result.metadata.get('enhanced_descriptor'):
                text_parts.append(result.metadata['enhanced_descriptor'])
            
            documents.append({
                "id": result.id,
                "text": " ".join(text_parts)
            })
        
        try:
            # Use Pinecone's inference API for reranking
            reranked = self.pc.inference.rerank(
                model="bge-reranker-v2-m3",
                query=query,
                documents=documents,
                top_n=min(top_k, len(documents)),
                return_documents=False
            )
            
            # Update scores based on reranking
            reranked_results = []
            for item in reranked.data:
                # Find original result
                for result in results:
                    if result.id == item.id:
                        # Update with reranked score
                        result.score = item.score
                        result.debug_info["reranked"] = True
                        result.debug_info["rerank_model"] = "bge-reranker-v2-m3"
                        reranked_results.append(result)
                        break
            
            logger.info(f"Reranked {len(results)} results, returning top {len(reranked_results)}")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}, returning original order")
            return results
    
    def _determine_weights(self, query: str) -> Tuple[float, float]:
        """Dynamically determine dense vs sparse weights based on query."""
        query_lower = query.lower()
        
        # Check for exact match indicators
        exact_indicators = [
            'exact', 'specifically', 'model', 'sku', '"'
        ]
        
        if any(indicator in query_lower for indicator in exact_indicators):
            # Favor sparse for exact searches
            return 0.3, 0.7
        
        # Check for semantic indicators
        semantic_indicators = [
            'like', 'similar', 'comfortable', 'quality', 'best'
        ]
        
        if any(indicator in query_lower for indicator in semantic_indicators):
            # Favor dense for semantic searches
            return 0.8, 0.2
        
        # Default balanced approach
        return self.default_dense_weight, self.default_sparse_weight
    
    def _generate_query_sparse_values(self, query: str) -> Dict[str, Any]:
        """Generate sparse values for a query."""
        if not hasattr(self.sparse_generator, 'vocabulary') or not self.sparse_generator.vocabulary:
            return {}
        
        # Create pseudo-product for query
        pseudo_product = {
            'universal_fields': {
                'name': query,
                'description': query
            },
            'search_keywords': query.lower().split(),
            'filter_metadata': {}
        }
        
        # Extract key terms from query
        key_terms = []
        import re
        
        # Extract quoted phrases
        quoted = re.findall(r'"([^"]+)"', query)
        key_terms.extend(quoted)
        
        # Extract potential model numbers
        models = re.findall(r'\b[A-Z0-9]{2,}[\-_]?[A-Z0-9]+\b', query, re.IGNORECASE)
        key_terms.extend(models)
        
        # Generate sparse embedding
        sparse_data = self.sparse_generator.generate_sparse_embedding(
            pseudo_product,
            {'key_features': key_terms}
        )
        
        return sparse_data
    
    def _build_pinecone_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Convert filters to Pinecone format."""
        pinecone_filter = {}
        
        for key, value in filters.items():
            if isinstance(value, dict) and '$in' in value:
                # Handle $in operator
                pinecone_filter[key] = {"$in": value['$in']}
            elif isinstance(value, dict) and any(k in value for k in ['$gte', '$lte', '$gt', '$lt']):
                # Handle range operators
                pinecone_filter[key] = value
            else:
                # Simple equality
                pinecone_filter[key] = value
        
        return pinecone_filter