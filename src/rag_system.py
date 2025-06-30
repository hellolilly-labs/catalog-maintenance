"""
Integrated RAG System

Combines all components into a unified RAG system with:
- Universal product processing
- Hybrid search
- Automatic synchronization
- Langfuse integration
- Caching
- Monitoring
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

from .ingestion import PineconeIngestion, UniversalProductProcessor
from .search.hybrid_search import HybridSearchEngine, HybridQueryOptimizer
from .sync import SyncOrchestrator
from .integration import (
    LangfuseRAGManager,
    RAGConfigManager,
    RAGCacheManager,
    QueryOptimizationCache,
    RAGMonitor
)

logger = logging.getLogger(__name__)


class IntegratedRAGSystem:
    """
    Complete RAG system with all features integrated.
    
    This is the main class for production use, combining:
    - Product ingestion and processing
    - Hybrid search capabilities
    - Automatic synchronization
    - Prompt management via Langfuse
    - Intelligent caching
    - Comprehensive monitoring
    """
    
    def __init__(
        self,
        brand_domain: str,
        catalog_path: str,
        index_name: str,
        namespace: str = "products",
        enable_monitoring: bool = True,
        enable_caching: bool = True,
        auto_sync: bool = False
    ):
        self.brand_domain = brand_domain
        self.catalog_path = catalog_path
        self.index_name = index_name
        self.namespace = namespace
        
        logger.info(f"🚀 Initializing Integrated RAG System for {brand_domain}")
        
        # Core components
        self.processor = UniversalProductProcessor(brand_domain)
        self.ingestion = PineconeIngestion(brand_domain, index_name, namespace)
        self.search_engine = HybridSearchEngine(brand_domain, index_name, namespace)
        self.query_optimizer = HybridQueryOptimizer(brand_domain)
        
        # Synchronization
        self.sync_orchestrator = SyncOrchestrator(
            brand_domain=brand_domain,
            catalog_path=catalog_path,
            index_name=index_name,
            namespace=namespace,
            auto_start=auto_sync
        )
        
        # Integration components
        self.langfuse_manager = LangfuseRAGManager(brand_domain)
        self.config_manager = RAGConfigManager(brand_domain)
        
        # Caching
        self.cache_enabled = enable_caching
        if enable_caching:
            self.cache_manager = RAGCacheManager(brand_domain)
            self.query_cache = QueryOptimizationCache(self.cache_manager)
        
        # Monitoring
        self.monitoring_enabled = enable_monitoring
        if enable_monitoring:
            self.monitor = RAGMonitor(brand_domain)
            
            # Register alert handlers
            self.monitor.add_alert_handler(self._handle_alert)
        
        # System state
        self.is_initialized = False
        self._initialize_system()
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        user_context: Optional[Dict[str, Any]] = None,
        use_cache: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform an intelligent product search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional filters to apply
            user_context: Optional user context (preferences, history)
            use_cache: Override cache setting
            
        Returns:
            List of search results with metadata
        """
        use_cache = use_cache if use_cache is not None else self.cache_enabled
        
        # Check cache first
        if use_cache:
            cached_results = self.cache_manager.get_query_result(query, filters)
            if cached_results:
                logger.debug(f"Cache hit for query: {query}")
                return cached_results[:top_k]
        
        # Track search performance
        with self.monitor.track_search(
            query,
            cache_hit=False,
            filters_used=len(filters) if filters else 0
        ) as tracker:
            
            # Optimize query
            optimization = self._optimize_query(query, user_context)
            
            # Apply user preferences to filters
            enhanced_filters = self._enhance_filters(filters, user_context)
            
            # Perform search
            results = self.search_engine.search(
                query=optimization['optimized_query'],
                top_k=top_k,
                filters=enhanced_filters,
                dense_weight=optimization.get('dense_weight'),
                sparse_weight=optimization.get('sparse_weight'),
                rerank=True
            )
            
            # Convert results to dictionaries
            result_dicts = []
            for result in results:
                result_dict = {
                    'id': result.id,
                    'score': result.score,
                    'name': result.metadata.get('name', 'Unknown'),
                    'brand': result.metadata.get('brand', ''),
                    'price': result.metadata.get('price', 0),
                    'description': result.metadata.get('description', ''),
                    'enhanced_descriptor': result.metadata.get('enhanced_descriptor', ''),
                    'key_selling_points': json.loads(result.metadata.get('key_selling_points', '[]')),
                    'metadata': result.metadata
                }
                result_dicts.append(result_dict)
            
            # Track result count
            tracker.set_result_count(len(result_dicts))
            
            # Cache results
            if use_cache and result_dicts:
                self.cache_manager.set_query_result(
                    query=query,
                    results=result_dicts,
                    filters=filters,
                    ttl=300  # 5 minutes for search results
                )
            
            return result_dicts
    
    def ingest_catalog(self, force_update: bool = False) -> Dict[str, Any]:
        """
        Ingest or update the product catalog.
        
        Args:
            force_update: Force update all products
            
        Returns:
            Ingestion statistics
        """
        logger.info(f"📦 Starting catalog ingestion for {self.brand_domain}")
        
        # Load catalog
        with open(self.catalog_path) as f:
            catalog_data = json.load(f)
        
        # Extract products
        if isinstance(catalog_data, list):
            products = catalog_data
        else:
            # Find products in dict
            for key in ['products', 'items', 'catalog', 'data']:
                if key in catalog_data and isinstance(catalog_data[key], list):
                    products = catalog_data[key]
                    break
            else:
                raise ValueError("Could not find products in catalog")
        
        # Perform ingestion
        start_time = time.time()
        
        try:
            results = self.ingestion.ingest_products(
                products=products,
                force_update=force_update,
                update_prompts=True
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Track metrics
            if self.monitoring_enabled:
                self.monitor.track_ingestion(
                    operation='full_sync' if force_update else 'incremental',
                    product_count=len(products),
                    duration_ms=duration_ms,
                    success=True
                )
            
            # Invalidate caches
            if self.cache_enabled:
                self.cache_manager.invalidate_query_cache()
                self.cache_manager.invalidate_filter_cache()
            
            logger.info(f"✅ Ingestion completed: {results}")
            return results
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            if self.monitoring_enabled:
                self.monitor.track_ingestion(
                    operation='full_sync' if force_update else 'incremental',
                    product_count=len(products),
                    duration_ms=duration_ms,
                    success=False,
                    error=str(e)
                )
            
            logger.error(f"❌ Ingestion failed: {e}")
            raise
    
    def sync_changes(self) -> bool:
        """
        Manually trigger synchronization of catalog changes.
        
        Returns:
            True if sync successful
        """
        return self.sync_orchestrator.sync_changes()
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.
        
        Returns:
            System status including all components
        """
        status = {
            'brand': self.brand_domain,
            'index': self.index_name,
            'initialized': self.is_initialized,
            'components': {
                'search': 'ready',
                'ingestion': 'ready',
                'sync': 'running' if self.sync_orchestrator.is_running else 'stopped',
                'cache': 'enabled' if self.cache_enabled else 'disabled',
                'monitoring': 'enabled' if self.monitoring_enabled else 'disabled'
            }
        }
        
        # Add sync statistics
        if hasattr(self.sync_orchestrator, 'get_sync_stats'):
            status['sync_stats'] = self.sync_orchestrator.get_sync_stats()
        
        # Add cache statistics
        if self.cache_enabled:
            status['cache_stats'] = self.cache_manager.get_statistics()
        
        # Add monitoring statistics
        if self.monitoring_enabled:
            status['search_stats'] = self.monitor.get_search_statistics()
            status['ingestion_stats'] = self.monitor.get_ingestion_statistics()
        
        # Add index statistics
        try:
            status['index_stats'] = self.ingestion.get_index_stats()
        except:
            status['index_stats'] = {'error': 'Could not fetch index stats'}
        
        return status
    
    def _initialize_system(self) -> None:
        """
        Initialize system components.
        """
        # Load configuration
        search_config = self.config_manager.get_search_weights()
        
        # Apply configuration
        if 'default' in search_config:
            self.search_engine.default_dense_weight = search_config['default']['dense']
            self.search_engine.default_sparse_weight = search_config['default']['sparse']
        
        # Load filter dictionary
        filter_dict = self.langfuse_manager.get_filter_dictionary()
        if filter_dict:
            logger.info(f"📚 Loaded filter dictionary with {len(filter_dict)} filters")
        
        self.is_initialized = True
        logger.info("✅ RAG system initialized")
    
    def _optimize_query(
        self,
        query: str,
        user_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Optimize query using context and caching.
        """
        # Check optimization cache
        if self.cache_enabled:
            context_messages = user_context.get('messages', []) if user_context else []
            cached_optimization = self.query_cache.get_optimization(query, context_messages)
            if cached_optimization:
                return cached_optimization
        
        # Perform optimization
        optimization = self.query_optimizer.optimize_query(
            query=query,
            context=user_context.get('messages', []) if user_context else None,
            user_preferences=user_context.get('preferences') if user_context else None
        )
        
        # Cache optimization
        if self.cache_enabled:
            self.query_cache.set_optimization(
                query=query,
                optimization=optimization,
                context=user_context.get('messages', []) if user_context else None
            )
        
        return optimization
    
    def _enhance_filters(
        self,
        filters: Optional[Dict[str, Any]],
        user_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Enhance filters with user preferences.
        """
        enhanced = filters.copy() if filters else {}
        
        if user_context and 'preferences' in user_context:
            prefs = user_context['preferences']
            
            # Add preference-based filters
            if 'preferred_brands' in prefs and 'brand' not in enhanced:
                enhanced['brand'] = {'$in': prefs['preferred_brands']}
            
            if 'max_price' in prefs and 'price' not in enhanced:
                enhanced['price'] = {'$lte': prefs['max_price']}
        
        return enhanced
    
    def _handle_alert(self, alert_type: str, message: str, severity: str) -> None:
        """
        Handle monitoring alerts.
        """
        logger.warning(f"🚨 System Alert [{severity}] {alert_type}: {message}")
        
        # Could send to external monitoring system
        # Could trigger automatic remediation
        # Could notify administrators


def create_rag_system(
    brand_domain: str,
    catalog_path: str,
    index_name: str,
    **kwargs
) -> IntegratedRAGSystem:
    """
    Factory function to create a configured RAG system.
    
    Args:
        brand_domain: Brand domain (e.g., 'specialized.com')
        catalog_path: Path to product catalog JSON
        index_name: Pinecone index name
        **kwargs: Additional configuration options
        
    Returns:
        Configured RAG system ready for use
    """
    # Set defaults
    config = {
        'namespace': 'products',
        'enable_monitoring': True,
        'enable_caching': True,
        'auto_sync': False
    }
    config.update(kwargs)
    
    # Create system
    system = IntegratedRAGSystem(
        brand_domain=brand_domain,
        catalog_path=catalog_path,
        index_name=index_name,
        **config
    )
    
    return system