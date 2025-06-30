"""
Hybrid Search Implementation for Pinecone RAG

Combines dense and sparse embeddings for optimal search accuracy.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from pinecone import Pinecone, SearchQuery, SearchRerank, RerankModel
import numpy as np

from ..ingestion.sparse_embeddings import SparseEmbeddingGenerator
from ..catalog.enhanced_descriptor_generator import EnhancedDescriptorGenerator

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Structured search result."""
    id: str
    score: float
    metadata: Dict[str, Any]
    debug_info: Optional[Dict[str, Any]] = None


class HybridSearchEngine:
    """
    Implements hybrid search combining dense and sparse embeddings.
    
    This approach provides:
    - Semantic understanding through dense embeddings
    - Keyword precision through sparse embeddings
    - Metadata filtering for hard constraints
    - Dynamic weight adjustment based on query type
    """
    
    def __init__(
        self,
        brand_domain: str,
        index_name: str,
        namespace: str = "products",
        embedding_model: str = "llama-text-embed-v2"
    ):
        self.brand_domain = brand_domain
        self.index_name = index_name
        self.namespace = namespace
        self.embedding_model = embedding_model
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        self.index = self.pc.Index(index_name)
        
        # Initialize components
        self.sparse_generator = SparseEmbeddingGenerator(brand_domain)
        self.descriptor_generator = EnhancedDescriptorGenerator(brand_domain)
        
        # Load sparse vocabulary
        if not self.sparse_generator.load_vocabulary():
            logger.warning(f"⚠️ Sparse vocabulary not found for {brand_domain}")
        
        # Default weights
        self.default_dense_weight = 0.8
        self.default_sparse_weight = 0.2
        
        logger.info(f"🔍 Initialized Hybrid Search Engine for {brand_domain}")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        dense_weight: Optional[float] = None,
        sparse_weight: Optional[float] = None,
        rerank: bool = True
    ) -> List[SearchResult]:
        """
        Perform hybrid search with neural reranking.
        
        This method combines dense and sparse embeddings for initial retrieval,
        then uses Pinecone's integrated Cohere reranking model for neural reranking.
        
        Neural reranking uses a cross-encoder architecture that jointly encodes
        the query and each candidate document, providing much better relevance
        scoring than the initial vector similarity scores.
        
        Args:
            query: Search query
            top_k: Number of results to return after reranking
            filters: Metadata filters to apply
            dense_weight: Weight for dense embeddings (0-1)
            sparse_weight: Weight for sparse embeddings (0-1)
            rerank: Whether to use neural reranking (Cohere model)
            
        Returns:
            List of search results, reranked if requested
        """
        # Adjust weights based on query characteristics
        if dense_weight is None or sparse_weight is None:
            dense_weight, sparse_weight = self._determine_weights(query)
        
        # Ensure weights sum to 1
        total_weight = dense_weight + sparse_weight
        if total_weight > 0:
            dense_weight /= total_weight
            sparse_weight /= total_weight
        
        logger.info(f"🔎 Hybrid search: '{query}' (dense: {dense_weight:.2f}, sparse: {sparse_weight:.2f})")
        
        # Generate query representations
        query_data = self._prepare_query(query)
        
        # Build search query
        search_query_params = {
            "inputs": {
                "text": query_data["enhanced_query"]
            },
            "top_k": top_k * 2 if rerank else top_k  # Get more candidates for reranking
        }
        
        # Add sparse values if available
        if sparse_weight > 0 and query_data.get("sparse_values"):
            search_query_params["sparse_values"] = query_data["sparse_values"]
        
        # Configure neural reranking if requested
        rerank_config = None
        if rerank:
            rerank_config = SearchRerank(
                model=RerankModel.Cohere_Rerank_3_5,  # Using Cohere's reranking model
                rank_fields=["text"],  # Fields to use for reranking
                top_n=top_k  # Final number of results after reranking
            )
        
        # Add filters
        filter_dict = None
        if filters:
            filter_dict = self._build_pinecone_filter(filters)
        
        # Execute search with neural reranking
        try:
            results = self.index.search_records(
                namespace=self.namespace,
                query=SearchQuery(**search_query_params),
                rerank=rerank_config,
                filter=filter_dict,
                fields=["metadata"]
            )
            
            # Process results
            search_results = []
            for hit in results.result.hits:
                # Extract metadata from fields
                metadata = hit.fields.get('metadata', {})
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                
                result = SearchResult(
                    id=hit._id,
                    score=hit._score,
                    metadata=metadata,
                    debug_info={
                        "reranked": rerank,
                        "model": "Cohere_Rerank_3_5" if rerank else None
                    }
                )
                search_results.append(result)
            
            # Results are already reranked by Pinecone if rerank=True
            # No need for additional reranking
            
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _prepare_query(self, query: str) -> Dict[str, Any]:
        """
        Prepare query for hybrid search.
        """
        # Enhanced query for dense embedding
        enhanced_query = query
        
        # Extract key terms for emphasis
        key_terms = self._extract_key_terms(query)
        if key_terms:
            enhanced_query = f"{query} {' '.join(key_terms)}"
        
        # Generate sparse representation
        sparse_values = None
        if hasattr(self.sparse_generator, 'vocabulary') and self.sparse_generator.vocabulary:
            # Create a pseudo-product for sparse generation
            pseudo_product = {
                'universal_fields': {
                    'name': query,
                    'description': query
                },
                'search_keywords': key_terms,
                'filter_metadata': {}
            }
            
            sparse_data = self.sparse_generator.generate_sparse_embedding(
                pseudo_product,
                {'key_features': key_terms}
            )
            
            if sparse_data['indices']:
                sparse_values = sparse_data
        
        return {
            "original_query": query,
            "enhanced_query": enhanced_query,
            "key_terms": key_terms,
            "sparse_values": sparse_values
        }
    
    def _determine_weights(self, query: str) -> Tuple[float, float]:
        """
        Dynamically determine dense vs sparse weights based on query.
        """
        query_lower = query.lower()
        
        # High sparse weight for:
        # - Brand names
        # - Model numbers
        # - Exact phrases in quotes
        # - Technical specifications
        
        sparse_boost_patterns = [
            # Brand names (customize per brand)
            self.brand_domain.split('.')[0].lower(),
            # Model number patterns
            r'\b[A-Z0-9]{2,}[\-_]?[A-Z0-9]+\b',
            # Quoted phrases
            r'"[^"]+"',
            # Price queries
            r'\$\d+',
            r'under \d+',
            r'below \d+',
            # Specific technical terms
            r'\b\d+mm\b',
            r'\b\d+gb\b',
            r'\b\d+mph\b',
        ]
        
        # Check for sparse-favoring patterns
        sparse_score = 0
        for pattern in sparse_boost_patterns:
            if isinstance(pattern, str) and pattern in query_lower:
                sparse_score += 1
            elif hasattr(pattern, 'search'):  # Regex pattern
                import re
                if re.search(pattern, query, re.IGNORECASE):
                    sparse_score += 1
        
        # Adjust weights based on sparse score
        if sparse_score >= 2:
            # Heavy sparse weight for very specific queries
            return 0.4, 0.6
        elif sparse_score == 1:
            # Balanced for semi-specific queries
            return 0.6, 0.4
        else:
            # Default: favor dense for general queries
            return self.default_dense_weight, self.default_sparse_weight
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """
        Extract important terms from query.
        """
        import re
        
        key_terms = []
        
        # Extract quoted phrases
        quoted = re.findall(r'"([^"]+)"', query)
        key_terms.extend(quoted)
        
        # Extract model numbers
        models = re.findall(r'\b[A-Z0-9]{2,}[\-_]?[A-Z0-9]+\b', query, re.IGNORECASE)
        key_terms.extend([m.lower() for m in models])
        
        # Extract price constraints
        prices = re.findall(r'(?:under|below|above|over)\s*\$?\d+', query, re.IGNORECASE)
        key_terms.extend(prices)
        
        # Brand-specific important terms
        brand_terms = {
            "specialized": ["turbo", "s-works", "tarmac", "roubaix", "diverge"],
            "balenciaga": ["cagole", "hourglass", "track", "triple s"],
            "sundayriley": ["good genes", "luna", "ceo", "ice"],
        }
        
        brand_key = self.brand_domain.split('.')[0].lower()
        if brand_key in brand_terms:
            for term in brand_terms[brand_key]:
                if term in query.lower():
                    key_terms.append(term)
        
        return list(set(key_terms))
    
    def _build_pinecone_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build Pinecone-compatible filter from simplified format.
        """
        pinecone_filter = {}
        
        for key, value in filters.items():
            if isinstance(value, list):
                # Multiple values: OR condition
                pinecone_filter[key] = {"$in": value}
            elif isinstance(value, dict):
                # Range or complex filter
                if "min" in value or "max" in value:
                    range_filter = {}
                    if "min" in value:
                        range_filter["$gte"] = value["min"]
                    if "max" in value:
                        range_filter["$lte"] = value["max"]
                    pinecone_filter[key] = range_filter
                else:
                    pinecone_filter[key] = value
            else:
                # Single value: exact match
                pinecone_filter[key] = value
        
        return pinecone_filter
    
    def _rerank_results(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Rerank results using additional signals.
        """
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        # Score adjustments
        for result in results:
            adjustment = 0.0
            metadata = result.metadata
            
            # Boost exact name matches
            if metadata.get('name', '').lower() == query_lower:
                adjustment += 0.3
            
            # Boost brand matches
            brand = metadata.get('brand', '').lower()
            if brand and brand in query_lower:
                adjustment += 0.2
            
            # Boost if all query terms appear in name
            name_lower = metadata.get('name', '').lower()
            if all(term in name_lower for term in query_terms):
                adjustment += 0.1
            
            # Boost popular/featured items
            if metadata.get('is_featured') or metadata.get('best_seller'):
                adjustment += 0.05
            
            # Apply adjustment
            result.score *= (1 + adjustment)
        
        # Sort by adjusted score
        results.sort(key=lambda r: r.score, reverse=True)
        
        return results


class HybridQueryOptimizer:
    """
    Optimizes queries for hybrid search based on context and intent.
    """
    
    def __init__(self, brand_domain: str):
        self.brand_domain = brand_domain
        self.query_patterns = self._load_query_patterns()
    
    def optimize_query(
        self,
        query: str,
        context: Optional[List[str]] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize query for hybrid search.
        
        Returns:
            Dictionary with optimized query and search parameters
        """
        optimization = {
            "original_query": query,
            "optimized_query": query,
            "filters": {},
            "dense_weight": None,
            "sparse_weight": None,
            "search_strategy": "hybrid"
        }
        
        # Extract intent and entities
        intent = self._detect_intent(query, context)
        entities = self._extract_entities(query)
        
        # Apply intent-based optimization
        if intent == "specific_product":
            # Looking for specific item: boost sparse
            optimization["dense_weight"] = 0.3
            optimization["sparse_weight"] = 0.7
            optimization["search_strategy"] = "sparse_focused"
            
        elif intent == "browse_category":
            # Browsing: balanced approach
            optimization["dense_weight"] = 0.6
            optimization["sparse_weight"] = 0.4
            
            # Add category filter if detected
            if entities.get("category"):
                optimization["filters"]["category"] = entities["category"]
        
        elif intent == "price_conscious":
            # Price-focused: use filters
            if entities.get("price_range"):
                optimization["filters"]["price"] = entities["price_range"]
            
            # Emphasize value in query
            optimization["optimized_query"] = f"{query} best value"
        
        # Add user preference filters
        if user_preferences:
            if user_preferences.get("preferred_brands"):
                optimization["filters"]["brand"] = {"$in": user_preferences["preferred_brands"]}
            
            if user_preferences.get("size"):
                optimization["filters"]["size"] = user_preferences["size"]
        
        return optimization
    
    def _detect_intent(self, query: str, context: Optional[List[str]]) -> str:
        """
        Detect search intent from query and context.
        """
        query_lower = query.lower()
        
        # Specific product indicators
        specific_indicators = [
            "model", "sku", "item number", "product code",
            "exactly", "specific"
        ]
        if any(ind in query_lower for ind in specific_indicators):
            return "specific_product"
        
        # Category browsing indicators
        browse_indicators = [
            "show me", "what do you have", "options for",
            "types of", "browse", "collection"
        ]
        if any(ind in query_lower for ind in browse_indicators):
            return "browse_category"
        
        # Price-conscious indicators
        price_indicators = [
            "under", "below", "budget", "cheap", "affordable",
            "less than", "sale", "discount"
        ]
        if any(ind in query_lower for ind in price_indicators):
            return "price_conscious"
        
        # Check context for additional clues
        if context:
            recent_context = " ".join(context[-3:]).lower()
            if "price" in recent_context or "cost" in recent_context:
                return "price_conscious"
        
        return "general"
    
    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """
        Extract entities from query.
        """
        import re
        
        entities = {}
        query_lower = query.lower()
        
        # Extract price range
        price_match = re.search(r'(?:under|below)\s*\$?(\d+)', query_lower)
        if price_match:
            entities["price_range"] = {"max": float(price_match.group(1))}
        
        # Extract categories (brand-specific)
        categories = {
            "specialized": ["bikes", "helmets", "shoes", "accessories"],
            "balenciaga": ["bags", "shoes", "clothing", "accessories"],
            "sundayriley": ["serums", "moisturizers", "treatments", "cleansers"]
        }
        
        brand_key = self.brand_domain.split('.')[0].lower()
        if brand_key in categories:
            for category in categories[brand_key]:
                if category in query_lower:
                    entities["category"] = category
                    break
        
        return entities
    
    def _load_query_patterns(self) -> Dict[str, List[str]]:
        """
        Load brand-specific query patterns.
        """
        # This could be loaded from a file or database
        return {
            "product_qualifiers": ["best", "top", "premium", "professional"],
            "comparison_terms": ["vs", "versus", "compared to", "better than"],
            "technical_specs": ["watts", "grams", "mm", "carbon", "aluminum"]
        }