"""
Cache Management for RAG System

Manages caching for query results, embeddings, and filter summaries.
"""

import os
import json
import time
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from collections import OrderedDict
import pickle
import threading

import numpy as np

logger = logging.getLogger(__name__)


class RAGCacheManager:
    """
    Manages multiple cache layers for the RAG system.
    
    Features:
    - Query result caching
    - Embedding caching
    - Filter summary caching
    - TTL-based expiration
    - LRU eviction
    - Persistent storage option
    """
    
    def __init__(
        self,
        brand_domain: str,
        cache_dir: Optional[str] = None,
        max_memory_items: int = 1000,
        default_ttl: int = 3600
    ):
        self.brand_domain = brand_domain
        self.max_memory_items = max_memory_items
        self.default_ttl = default_ttl
        
        # Cache directories
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(f"accounts/{brand_domain}/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory caches
        self.query_cache = OrderedDict()
        self.embedding_cache = OrderedDict()
        self.filter_cache = OrderedDict()
        
        # Cache statistics
        self.stats = {
            'query_hits': 0,
            'query_misses': 0,
            'embedding_hits': 0,
            'embedding_misses': 0,
            'evictions': 0
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Load persistent caches
        self._load_persistent_caches()
        
        logger.info(f"📦 Initialized cache manager for {brand_domain}")
    
    def get_query_result(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        search_type: str = "hybrid"
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached query results.
        
        Args:
            query: Search query
            filters: Applied filters
            search_type: Type of search performed
            
        Returns:
            Cached results or None
        """
        cache_key = self._generate_query_key(query, filters, search_type)
        
        with self.lock:
            if cache_key in self.query_cache:
                entry = self.query_cache[cache_key]
                
                # Check TTL
                if time.time() < entry['expires_at']:
                    # Move to end (LRU)
                    self.query_cache.move_to_end(cache_key)
                    self.stats['query_hits'] += 1
                    
                    logger.debug(f"Cache hit for query: {query[:50]}...")
                    return entry['results']
                else:
                    # Expired
                    del self.query_cache[cache_key]
            
            self.stats['query_misses'] += 1
            return None
    
    def set_query_result(
        self,
        query: str,
        results: List[Dict[str, Any]],
        filters: Optional[Dict[str, Any]] = None,
        search_type: str = "hybrid",
        ttl: Optional[int] = None
    ) -> None:
        """
        Cache query results.
        """
        cache_key = self._generate_query_key(query, filters, search_type)
        
        with self.lock:
            # Evict if necessary
            self._evict_if_needed(self.query_cache)
            
            # Store result
            self.query_cache[cache_key] = {
                'query': query,
                'filters': filters,
                'search_type': search_type,
                'results': results,
                'cached_at': time.time(),
                'expires_at': time.time() + (ttl or self.default_ttl)
            }
            
            logger.debug(f"Cached results for query: {query[:50]}...")
    
    def get_embedding(self, text: str, model: str = "default") -> Optional[np.ndarray]:
        """
        Get cached embedding.
        """
        cache_key = self._generate_embedding_key(text, model)
        
        with self.lock:
            if cache_key in self.embedding_cache:
                entry = self.embedding_cache[cache_key]
                
                # Embeddings don't expire as quickly
                if time.time() < entry['expires_at']:
                    self.embedding_cache.move_to_end(cache_key)
                    self.stats['embedding_hits'] += 1
                    return entry['embedding']
                else:
                    del self.embedding_cache[cache_key]
            
            self.stats['embedding_misses'] += 1
            return None
    
    def set_embedding(
        self,
        text: str,
        embedding: np.ndarray,
        model: str = "default",
        ttl: Optional[int] = None
    ) -> None:
        """
        Cache embedding.
        """
        cache_key = self._generate_embedding_key(text, model)
        
        with self.lock:
            self._evict_if_needed(self.embedding_cache)
            
            # Use longer TTL for embeddings
            embedding_ttl = ttl or (self.default_ttl * 24)  # 24 hours default
            
            self.embedding_cache[cache_key] = {
                'text': text[:100],  # Store prefix for debugging
                'model': model,
                'embedding': embedding,
                'cached_at': time.time(),
                'expires_at': time.time() + embedding_ttl
            }
    
    def get_filter_summary(self, brand_domain: str) -> Optional[Dict[str, Any]]:
        """
        Get cached filter summary for a brand.
        """
        with self.lock:
            if brand_domain in self.filter_cache:
                entry = self.filter_cache[brand_domain]
                
                if time.time() < entry['expires_at']:
                    self.filter_cache.move_to_end(brand_domain)
                    return entry['summary']
                else:
                    del self.filter_cache[brand_domain]
        
        return None
    
    def set_filter_summary(
        self,
        brand_domain: str,
        summary: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> None:
        """
        Cache filter summary.
        """
        with self.lock:
            self._evict_if_needed(self.filter_cache)
            
            self.filter_cache[brand_domain] = {
                'summary': summary,
                'cached_at': time.time(),
                'expires_at': time.time() + (ttl or self.default_ttl * 2)
            }
    
    def invalidate_query_cache(self, pattern: Optional[str] = None) -> int:
        """
        Invalidate query cache entries.
        
        Args:
            pattern: Optional pattern to match queries
            
        Returns:
            Number of entries invalidated
        """
        with self.lock:
            if pattern:
                # Invalidate matching entries
                to_remove = []
                for key, entry in self.query_cache.items():
                    if pattern.lower() in entry['query'].lower():
                        to_remove.append(key)
                
                for key in to_remove:
                    del self.query_cache[key]
                
                logger.info(f"🗑️ Invalidated {len(to_remove)} query cache entries matching '{pattern}'")
                return len(to_remove)
            else:
                # Clear all
                count = len(self.query_cache)
                self.query_cache.clear()
                logger.info(f"🗑️ Cleared all {count} query cache entries")
                return count
    
    def invalidate_filter_cache(self) -> None:
        """
        Invalidate filter cache (typically after catalog sync).
        """
        with self.lock:
            self.filter_cache.clear()
            logger.info("🗑️ Cleared filter cache")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        """
        with self.lock:
            total_queries = self.stats['query_hits'] + self.stats['query_misses']
            query_hit_rate = (
                self.stats['query_hits'] / total_queries if total_queries > 0 else 0
            )
            
            total_embeddings = self.stats['embedding_hits'] + self.stats['embedding_misses']
            embedding_hit_rate = (
                self.stats['embedding_hits'] / total_embeddings if total_embeddings > 0 else 0
            )
            
            return {
                'query_cache_size': len(self.query_cache),
                'embedding_cache_size': len(self.embedding_cache),
                'filter_cache_size': len(self.filter_cache),
                'query_hit_rate': query_hit_rate,
                'embedding_hit_rate': embedding_hit_rate,
                'total_evictions': self.stats['evictions'],
                'stats': self.stats.copy()
            }
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries from all caches.
        
        Returns:
            Number of entries removed
        """
        removed = 0
        current_time = time.time()
        
        with self.lock:
            # Clean query cache
            expired = [
                k for k, v in self.query_cache.items()
                if current_time >= v['expires_at']
            ]
            for k in expired:
                del self.query_cache[k]
            removed += len(expired)
            
            # Clean embedding cache
            expired = [
                k for k, v in self.embedding_cache.items()
                if current_time >= v['expires_at']
            ]
            for k in expired:
                del self.embedding_cache[k]
            removed += len(expired)
            
            # Clean filter cache
            expired = [
                k for k, v in self.filter_cache.items()
                if current_time >= v['expires_at']
            ]
            for k in expired:
                del self.filter_cache[k]
            removed += len(expired)
        
        if removed > 0:
            logger.info(f"🧹 Cleaned up {removed} expired cache entries")
        
        return removed
    
    def persist_to_disk(self) -> None:
        """
        Save important cache entries to disk.
        """
        with self.lock:
            # Save filter cache (most important)
            filter_path = self.cache_dir / "filter_cache.pkl"
            with open(filter_path, 'wb') as f:
                pickle.dump(dict(self.filter_cache), f)
            
            # Save cache statistics
            stats_path = self.cache_dir / "cache_stats.json"
            with open(stats_path, 'w') as f:
                json.dump({
                    'stats': self.stats,
                    'saved_at': datetime.now().isoformat(),
                    'cache_sizes': {
                        'query': len(self.query_cache),
                        'embedding': len(self.embedding_cache),
                        'filter': len(self.filter_cache)
                    }
                }, f, indent=2)
            
            logger.info("💾 Persisted cache to disk")
    
    def _generate_query_key(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        search_type: str
    ) -> str:
        """Generate cache key for query."""
        # Normalize and hash
        key_parts = [
            query.lower().strip(),
            json.dumps(filters or {}, sort_keys=True),
            search_type
        ]
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _generate_embedding_key(self, text: str, model: str) -> str:
        """Generate cache key for embedding."""
        key_string = f"{model}|{text}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _evict_if_needed(self, cache: OrderedDict) -> None:
        """Evict oldest entries if cache is full."""
        while len(cache) >= self.max_memory_items:
            # Remove oldest (first) item
            cache.popitem(last=False)
            self.stats['evictions'] += 1
    
    def _load_persistent_caches(self) -> None:
        """Load caches from disk if available."""
        # Load filter cache
        filter_path = self.cache_dir / "filter_cache.pkl"
        if filter_path.exists():
            try:
                with open(filter_path, 'rb') as f:
                    saved_filters = pickle.load(f)
                
                # Only load non-expired entries
                current_time = time.time()
                for k, v in saved_filters.items():
                    if current_time < v.get('expires_at', 0):
                        self.filter_cache[k] = v
                
                logger.info(f"📂 Loaded {len(self.filter_cache)} filter cache entries from disk")
            except Exception as e:
                logger.warning(f"Failed to load filter cache: {e}")


class QueryOptimizationCache:
    """
    Specialized cache for query optimization results.
    """
    
    def __init__(self, cache_manager: RAGCacheManager):
        self.cache_manager = cache_manager
        self.optimization_cache = OrderedDict()
        self.max_items = 500
    
    def get_optimization(self, query: str, context: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Get cached query optimization.
        """
        cache_key = self._generate_key(query, context)
        
        if cache_key in self.optimization_cache:
            # Move to end (LRU)
            self.optimization_cache.move_to_end(cache_key)
            return self.optimization_cache[cache_key]
        
        return None
    
    def set_optimization(
        self,
        query: str,
        optimization: Dict[str, Any],
        context: Optional[List[str]] = None
    ) -> None:
        """
        Cache query optimization result.
        """
        cache_key = self._generate_key(query, context)
        
        # Evict if needed
        while len(self.optimization_cache) >= self.max_items:
            self.optimization_cache.popitem(last=False)
        
        self.optimization_cache[cache_key] = optimization
    
    def _generate_key(self, query: str, context: Optional[List[str]]) -> str:
        """Generate cache key."""
        context_str = "|".join(context[-3:]) if context else ""
        key_string = f"{query.lower()}|{context_str}"
        return hashlib.md5(key_string.encode()).hexdigest()