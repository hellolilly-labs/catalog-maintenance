"""
Catalog Monitor for Automatic Synchronization

Monitors product catalogs for changes and triggers incremental updates.
"""

import os
import json
import hashlib
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
from dataclasses import dataclass, asdict
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class CatalogChange:
    """Represents a change in the catalog."""
    change_type: str  # 'added', 'modified', 'deleted'
    product_id: str
    timestamp: datetime
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None
    fields_changed: Optional[List[str]] = None


class CatalogMonitor:
    """
    Monitors product catalogs for changes and manages synchronization.
    
    Features:
    - Content-based change detection using hashing
    - Field-level change tracking
    - Batch change aggregation
    - Sync state persistence
    - Automatic trigger scheduling
    """
    
    def __init__(self, brand_domain: str, catalog_path: str):
        self.brand_domain = brand_domain
        self.catalog_path = Path(catalog_path)
        
        # Paths for state tracking
        self.brand_path = Path(f"accounts/{brand_domain}")
        self.monitor_state_path = self.brand_path / "catalog_monitor_state.json"
        self.change_log_path = self.brand_path / "catalog_changes.jsonl"
        
        # Load monitor state
        self.state = self._load_state()
        
        # Change tracking
        self.pending_changes: List[CatalogChange] = []
        self.change_threshold = 10  # Trigger sync after N changes
        self.time_threshold = timedelta(minutes=5)  # Or after N minutes
        
        logger.info(f"📊 Initialized Catalog Monitor for {brand_domain}")
    
    def check_for_changes(self) -> List[CatalogChange]:
        """
        Check catalog for changes since last sync.
        
        Returns:
            List of detected changes
        """
        logger.info(f"🔍 Checking catalog for changes: {self.catalog_path}")
        
        # Load current catalog
        try:
            with open(self.catalog_path) as f:
                catalog_data = json.load(f)
            
            # Extract products (handle different formats)
            if isinstance(catalog_data, list):
                current_products = catalog_data
            elif isinstance(catalog_data, dict):
                # Try common keys
                for key in ['products', 'items', 'catalog', 'data']:
                    if key in catalog_data and isinstance(catalog_data[key], list):
                        current_products = catalog_data[key]
                        break
                else:
                    logger.error("Could not find product list in catalog")
                    return []
            else:
                logger.error("Invalid catalog format")
                return []
            
        except Exception as e:
            logger.error(f"Failed to load catalog: {e}")
            return []
        
        # Build current state
        current_state = {}
        for product in current_products:
            product_id = self._extract_product_id(product)
            if product_id:
                product_hash = self._hash_product(product)
                current_state[product_id] = {
                    'hash': product_hash,
                    'data': product
                }
        
        # Compare with previous state
        previous_state = self.state.get('product_hashes', {})
        changes = []
        
        # Check for additions and modifications
        for product_id, current_info in current_state.items():
            if product_id not in previous_state:
                # New product
                changes.append(CatalogChange(
                    change_type='added',
                    product_id=product_id,
                    timestamp=datetime.now(),
                    new_hash=current_info['hash']
                ))
            elif previous_state[product_id] != current_info['hash']:
                # Modified product
                fields_changed = self._detect_changed_fields(
                    self.state.get('product_data', {}).get(product_id, {}),
                    current_info['data']
                )
                changes.append(CatalogChange(
                    change_type='modified',
                    product_id=product_id,
                    timestamp=datetime.now(),
                    old_hash=previous_state[product_id],
                    new_hash=current_info['hash'],
                    fields_changed=fields_changed
                ))
        
        # Check for deletions
        for product_id in previous_state:
            if product_id not in current_state:
                changes.append(CatalogChange(
                    change_type='deleted',
                    product_id=product_id,
                    timestamp=datetime.now(),
                    old_hash=previous_state[product_id]
                ))
        
        # Update state for next comparison
        self.state['product_hashes'] = {pid: info['hash'] for pid, info in current_state.items()}
        self.state['product_data'] = {pid: info['data'] for pid, info in current_state.items()}
        self.state['last_check'] = datetime.now().isoformat()
        self._save_state()
        
        # Log changes
        if changes:
            self._log_changes(changes)
            logger.info(f"✅ Detected {len(changes)} changes")
        else:
            logger.info("✅ No changes detected")
        
        return changes
    
    def should_trigger_sync(self, changes: Optional[List[CatalogChange]] = None) -> bool:
        """
        Determine if synchronization should be triggered.
        
        Args:
            changes: Recent changes to consider
            
        Returns:
            True if sync should be triggered
        """
        if changes:
            self.pending_changes.extend(changes)
        
        # Check change count threshold
        if len(self.pending_changes) >= self.change_threshold:
            logger.info(f"📈 Change threshold reached: {len(self.pending_changes)} changes")
            return True
        
        # Check time threshold
        if self.pending_changes:
            oldest_change = min(self.pending_changes, key=lambda c: c.timestamp)
            age = datetime.now() - oldest_change.timestamp
            if age >= self.time_threshold:
                logger.info(f"⏰ Time threshold reached: {age} since first change")
                return True
        
        # Check for high-priority changes
        priority_changes = [
            c for c in self.pending_changes 
            if c.change_type == 'deleted' or 
            (c.fields_changed and 'price' in c.fields_changed)
        ]
        if priority_changes:
            logger.info(f"🚨 Priority changes detected: {len(priority_changes)} changes")
            return True
        
        return False
    
    def get_sync_batch(self) -> Dict[str, List[str]]:
        """
        Get batch of changes for synchronization.
        
        Returns:
            Dictionary with 'add', 'update', and 'delete' lists
        """
        batch = {
            'add': [],
            'update': [],
            'delete': []
        }
        
        for change in self.pending_changes:
            if change.change_type == 'added':
                batch['add'].append(change.product_id)
            elif change.change_type == 'modified':
                batch['update'].append(change.product_id)
            elif change.change_type == 'deleted':
                batch['delete'].append(change.product_id)
        
        return batch
    
    def mark_sync_complete(self, success: bool = True) -> None:
        """
        Mark synchronization as complete and clear pending changes.
        """
        if success:
            self.state['last_sync'] = datetime.now().isoformat()
            self.state['sync_count'] = self.state.get('sync_count', 0) + 1
            self.pending_changes.clear()
            logger.info("✅ Sync marked as complete")
        else:
            # Keep changes for retry
            logger.warning("⚠️ Sync failed, keeping changes for retry")
        
        self._save_state()
    
    def get_catalog_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the catalog and monitoring.
        """
        stats = {
            'brand': self.brand_domain,
            'catalog_path': str(self.catalog_path),
            'total_products': len(self.state.get('product_hashes', {})),
            'pending_changes': len(self.pending_changes),
            'last_check': self.state.get('last_check'),
            'last_sync': self.state.get('last_sync'),
            'sync_count': self.state.get('sync_count', 0)
        }
        
        # Change breakdown
        change_types = defaultdict(int)
        for change in self.pending_changes:
            change_types[change.change_type] += 1
        stats['pending_by_type'] = dict(change_types)
        
        return stats
    
    def _extract_product_id(self, product: Dict[str, Any]) -> Optional[str]:
        """Extract product ID from various formats."""
        # Try common ID fields
        for field in ['id', 'productId', 'product_id', 'sku', 'item_id', '_id']:
            if field in product:
                return str(product[field])
        
        # Generate ID from other fields
        if 'name' in product:
            # Use name + brand as fallback
            brand = product.get('brand', '')
            return hashlib.md5(f"{brand}:{product['name']}".encode()).hexdigest()[:12]
        
        return None
    
    def _hash_product(self, product: Dict[str, Any]) -> str:
        """Generate hash for change detection."""
        # Create normalized representation
        normalized = json.dumps(product, sort_keys=True, default=str)
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def _detect_changed_fields(self, old_product: Dict[str, Any], new_product: Dict[str, Any]) -> List[str]:
        """Detect which fields changed between versions."""
        changed_fields = []
        
        # Get all keys
        all_keys = set(old_product.keys()) | set(new_product.keys())
        
        for key in all_keys:
            old_value = old_product.get(key)
            new_value = new_product.get(key)
            
            # Simple comparison (could be enhanced for nested structures)
            if old_value != new_value:
                changed_fields.append(key)
        
        return changed_fields
    
    def _log_changes(self, changes: List[CatalogChange]) -> None:
        """Log changes to file for audit trail."""
        self.change_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.change_log_path, 'a') as f:
            for change in changes:
                log_entry = {
                    'timestamp': change.timestamp.isoformat(),
                    'change_type': change.change_type,
                    'product_id': change.product_id,
                    'old_hash': change.old_hash,
                    'new_hash': change.new_hash,
                    'fields_changed': change.fields_changed
                }
                f.write(json.dumps(log_entry) + '\n')
    
    def _load_state(self) -> Dict[str, Any]:
        """Load monitor state from disk."""
        if self.monitor_state_path.exists():
            try:
                with open(self.monitor_state_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
        
        return {
            'product_hashes': {},
            'product_data': {},
            'last_check': None,
            'last_sync': None,
            'sync_count': 0
        }
    
    def _save_state(self) -> None:
        """Save monitor state to disk."""
        self.monitor_state_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Don't save full product data to keep file size reasonable
        save_state = {
            'product_hashes': self.state.get('product_hashes', {}),
            'last_check': self.state.get('last_check'),
            'last_sync': self.state.get('last_sync'),
            'sync_count': self.state.get('sync_count', 0)
        }
        
        with open(self.monitor_state_path, 'w') as f:
            json.dump(save_state, f, indent=2)