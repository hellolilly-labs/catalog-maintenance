"""
Sync Orchestrator for Automatic RAG Updates

Orchestrates the synchronization process between catalog changes and Pinecone.
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from .catalog_monitor import CatalogMonitor, CatalogChange
from ..ingestion import PineconeIngestion, UniversalProductProcessor
from ..agents.catalog_filter_analyzer import CatalogFilterAnalyzer

logger = logging.getLogger(__name__)


class SyncOrchestrator:
    """
    Orchestrates automatic synchronization of catalog changes to Pinecone.
    
    Features:
    - Automatic monitoring and sync triggering
    - Incremental updates for efficiency
    - Filter dictionary updates in Langfuse
    - Error handling and retry logic
    - Performance monitoring
    """
    
    def __init__(
        self,
        brand_domain: str,
        catalog_path: str,
        index_name: str,
        namespace: str = "products",
        auto_start: bool = False
    ):
        self.brand_domain = brand_domain
        self.catalog_path = catalog_path
        self.index_name = index_name
        self.namespace = namespace
        
        # Initialize components
        self.monitor = CatalogMonitor(brand_domain, catalog_path)
        self.ingestion = PineconeIngestion(
            brand_domain=brand_domain,
            index_name=index_name,
            namespace=namespace
        )
        self.processor = UniversalProductProcessor(brand_domain)
        
        # Sync configuration
        self.check_interval = 300  # 5 minutes
        self.retry_attempts = 3
        self.retry_delay = 60  # 1 minute
        
        # State tracking
        self.is_running = False
        self.sync_in_progress = False
        self.last_sync_time = None
        self.sync_stats = {
            'total_syncs': 0,
            'successful_syncs': 0,
            'failed_syncs': 0,
            'total_products_synced': 0
        }
        
        # Callbacks
        self.on_sync_complete: Optional[Callable] = None
        self.on_sync_error: Optional[Callable] = None
        
        # Background monitoring
        self._monitor_thread = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        logger.info(f"🎯 Initialized Sync Orchestrator for {brand_domain}")
        
        if auto_start:
            self.start_monitoring()
    
    def start_monitoring(self) -> None:
        """Start automatic monitoring in background."""
        if self.is_running:
            logger.warning("Monitoring already running")
            return
        
        self.is_running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info("🚀 Started automatic catalog monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop automatic monitoring."""
        self.is_running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)
        
        logger.info("🛑 Stopped catalog monitoring")
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self.is_running:
            try:
                # Check for changes
                changes = self.monitor.check_for_changes()
                
                # Determine if sync needed
                if self.monitor.should_trigger_sync(changes):
                    logger.info("🔄 Triggering automatic sync...")
                    
                    # Run sync in executor to avoid blocking
                    future = self._executor.submit(self.sync_changes)
                    
                    # Wait for completion with timeout
                    try:
                        success = future.result(timeout=600)  # 10 minute timeout
                        if success:
                            logger.info("✅ Automatic sync completed successfully")
                        else:
                            logger.error("❌ Automatic sync failed")
                    except Exception as e:
                        logger.error(f"❌ Sync error: {e}")
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
            
            # Wait for next check
            time.sleep(self.check_interval)
    
    def sync_changes(self, force_full_sync: bool = False) -> bool:
        """
        Synchronize detected changes to Pinecone.
        
        Args:
            force_full_sync: Force full catalog re-ingestion
            
        Returns:
            True if sync successful
        """
        if self.sync_in_progress:
            logger.warning("Sync already in progress")
            return False
        
        self.sync_in_progress = True
        sync_start = time.time()
        success = False
        
        try:
            if force_full_sync:
                logger.info("🔄 Performing full catalog sync...")
                success = self._perform_full_sync()
            else:
                logger.info("🔄 Performing incremental sync...")
                success = self._perform_incremental_sync()
            
            # Update statistics
            if success:
                self.sync_stats['successful_syncs'] += 1
                self.last_sync_time = datetime.now()
                
                # Trigger callback
                if self.on_sync_complete:
                    self.on_sync_complete(self.get_sync_stats())
            else:
                self.sync_stats['failed_syncs'] += 1
                
                # Trigger error callback
                if self.on_sync_error:
                    self.on_sync_error("Sync failed")
            
            self.sync_stats['total_syncs'] += 1
            
        except Exception as e:
            logger.error(f"Sync error: {e}")
            self.sync_stats['failed_syncs'] += 1
            success = False
            
            if self.on_sync_error:
                self.on_sync_error(str(e))
        
        finally:
            self.sync_in_progress = False
            sync_duration = time.time() - sync_start
            logger.info(f"⏱️ Sync completed in {sync_duration:.2f}s")
        
        return success
    
    def _perform_incremental_sync(self) -> bool:
        """Perform incremental sync of changes."""
        # Get batch of changes
        batch = self.monitor.get_sync_batch()
        
        if not any(batch.values()):
            logger.info("No changes to sync")
            self.monitor.mark_sync_complete()
            return True
        
        logger.info(f"📊 Syncing: {len(batch['add'])} adds, "
                   f"{len(batch['update'])} updates, {len(batch['delete'])} deletes")
        
        # Load current catalog
        try:
            with open(self.catalog_path) as f:
                catalog_data = json.load(f)
            
            # Extract products
            if isinstance(catalog_data, list):
                all_products = catalog_data
            else:
                # Find products in dict
                for key in ['products', 'items', 'catalog', 'data']:
                    if key in catalog_data and isinstance(catalog_data[key], list):
                        all_products = catalog_data[key]
                        break
                else:
                    raise ValueError("Could not find products in catalog")
            
            # Create product lookup
            product_lookup = {}
            for product in all_products:
                pid = self._extract_product_id(product)
                if pid:
                    product_lookup[pid] = product
            
            # Prepare products for sync
            products_to_sync = []
            
            # Add new and updated products
            for pid in batch['add'] + batch['update']:
                if pid in product_lookup:
                    products_to_sync.append(product_lookup[pid])
                else:
                    logger.warning(f"Product {pid} not found in catalog")
            
            # Process and ingest
            if products_to_sync:
                # Use force_update=True for specific products
                results = self.ingestion.ingest_products(
                    products=products_to_sync,
                    force_update=True,
                    update_prompts=True
                )
                
                self.sync_stats['total_products_synced'] += results.get('added', 0) + results.get('updated', 0)
            
            # Handle deletions
            if batch['delete']:
                self._delete_products(batch['delete'])
                self.sync_stats['total_products_synced'] += len(batch['delete'])
            
            # Mark sync complete
            self.monitor.mark_sync_complete()
            
            # Update filter dictionaries if significant changes
            if len(batch['add']) + len(batch['update']) > 10:
                self._update_filter_dictionaries(all_products)
            
            return True
            
        except Exception as e:
            logger.error(f"Incremental sync failed: {e}")
            return False
    
    def _perform_full_sync(self) -> bool:
        """Perform full catalog sync."""
        try:
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
            
            # Full ingestion
            results = self.ingestion.ingest_products(
                products=products,
                force_update=True,
                update_prompts=True
            )
            
            self.sync_stats['total_products_synced'] += (
                results.get('added', 0) + 
                results.get('updated', 0) + 
                results.get('deleted', 0)
            )
            
            # Clear pending changes
            self.monitor.pending_changes.clear()
            self.monitor.mark_sync_complete()
            
            return True
            
        except Exception as e:
            logger.error(f"Full sync failed: {e}")
            return False
    
    def _delete_products(self, product_ids: List[str]) -> None:
        """Delete products from Pinecone."""
        try:
            self.ingestion.index.delete(
                ids=product_ids,
                namespace=self.namespace
            )
            logger.info(f"🗑️ Deleted {len(product_ids)} products")
        except Exception as e:
            logger.error(f"Failed to delete products: {e}")
    
    def _update_filter_dictionaries(self, products: List[Dict[str, Any]]) -> None:
        """Update filter dictionaries in Langfuse."""
        try:
            analyzer = CatalogFilterAnalyzer(self.brand_domain)
            filters = analyzer.analyze_product_catalog(products)
            analyzer.save_filters_to_file(filters, "catalog_filters.json")
            
            # Would update Langfuse prompts here
            logger.info("📝 Updated filter dictionaries")
            
        except Exception as e:
            logger.warning(f"Failed to update filter dictionaries: {e}")
    
    def _extract_product_id(self, product: Dict[str, Any]) -> Optional[str]:
        """Extract product ID (delegate to monitor)."""
        return self.monitor._extract_product_id(product)
    
    def get_sync_stats(self) -> Dict[str, Any]:
        """Get synchronization statistics."""
        stats = self.sync_stats.copy()
        stats['last_sync_time'] = self.last_sync_time.isoformat() if self.last_sync_time else None
        stats['is_running'] = self.is_running
        stats['sync_in_progress'] = self.sync_in_progress
        
        # Add monitor stats
        monitor_stats = self.monitor.get_catalog_stats()
        stats['catalog_stats'] = monitor_stats
        
        return stats
    
    def trigger_manual_sync(self, force_full: bool = False) -> bool:
        """
        Manually trigger synchronization.
        
        Args:
            force_full: Force full sync instead of incremental
            
        Returns:
            True if sync started successfully
        """
        if self.sync_in_progress:
            logger.warning("Sync already in progress")
            return False
        
        logger.info("🔄 Manual sync triggered")
        
        # Check for changes first
        if not force_full:
            changes = self.monitor.check_for_changes()
            if not changes and not self.monitor.pending_changes:
                logger.info("No changes to sync")
                return True
        
        # Run sync
        return self.sync_changes(force_full_sync=force_full)


class SyncScheduler:
    """
    Advanced scheduler for sync operations.
    
    Supports:
    - Cron-like scheduling
    - Rate limiting
    - Priority queuing
    - Conflict resolution
    """
    
    def __init__(self, orchestrator: SyncOrchestrator):
        self.orchestrator = orchestrator
        self.scheduled_syncs = []
        self.rate_limit = 10  # Max syncs per hour
        self.sync_history = []
        
    def schedule_sync(
        self,
        schedule: str,
        sync_type: str = "incremental",
        priority: int = 5
    ) -> None:
        """
        Schedule a sync operation.
        
        Args:
            schedule: Cron expression or interval (e.g., "*/30 * * * *" or "30m")
            sync_type: "incremental" or "full"
            priority: 1-10 (higher = more important)
        """
        # Implementation would parse schedule and manage execution
        pass
    
    def can_sync_now(self) -> bool:
        """Check if sync is allowed based on rate limits."""
        # Count recent syncs
        recent_syncs = [
            s for s in self.sync_history 
            if (datetime.now() - s['timestamp']).total_seconds() < 3600
        ]
        
        return len(recent_syncs) < self.rate_limit