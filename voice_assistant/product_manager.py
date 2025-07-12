"""
Product Manager - KISS Approach

Simple product catalog manager for stateful voice agents. Loads product catalogs
into memory at startup and provides instant access. No Redis complexity - just
clean, simple, fast memory access perfect for long-running voice agents.
"""

import asyncio
import json
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from liddy.storage import get_account_storage_provider

logger = logging.getLogger(__name__)


class ProductManager:
    """
    Simple product catalog manager for stateful voice agents (KISS approach).
    
    Loads product catalog into memory at startup and provides instant access:
    - Direct memory access (sub-1ms vs ~10ms Redis calls)
    - Single load at voice agent startup
    - Graceful fallback to existing data/{account}/products.json files
    - Zero Redis dependencies - perfect for stateful applications
    """
    
    def __init__(self, account: str):
        """
        Initialize ProductManager for an account.
        
        Args:
            account: Account domain (e.g., "specialized.com")
        """
        self.account = account
        self._products: Optional[List[Dict[str, Any]]] = None
        self._storage_provider = None
        self._last_loaded: Optional[datetime] = None
        self._loading_lock = asyncio.Lock()
        self._memory_size: int = 0
    
    @property
    def storage_provider(self):
        """Lazy initialization of storage provider"""
        if self._storage_provider is None:
            self._storage_provider = get_account_storage_provider()
        return self._storage_provider
    
    @classmethod
    async def create_for_startup(cls, account: str) -> 'ProductManager':
        """
        Create ProductManager and load catalog for voice agent startup.
        
        Args:
            account: Account domain
            
        Returns:
            ProductManager instance with products loaded in memory
        """
        manager = cls(account)
        await manager.load_at_startup()
        return manager
    
    async def get_products(self) -> List[Dict[str, Any]]:
        """
        Get product catalog from memory (waits for loading if in progress).
        
        If products are still loading, this method will wait for loading to complete
        before returning the results. This prevents race conditions.
        
        Returns:
            List of product dictionaries from memory
        """
        # If products are already loaded, return immediately
        if self._products is not None:
            return self._products
        
        # If loading is in progress, wait for it to complete
        async with self._loading_lock:
            # Check again after acquiring lock (loading might have completed)
            if self._products is not None:
                return self._products
            
            # If we reach here, loading hasn't been started yet
            # Trigger loading now
            await self.load_at_startup()
            return self._products or []
    
    def get_products_if_loaded(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get products immediately if already loaded, None if still loading.
        
        This is a synchronous method for performance-critical paths where
        you want to avoid the async overhead if products are already available.
        
        Returns:
            List of products if loaded, None if loading in progress
        """
        return self._products
    
    async def find_product_by_id(self, product_id: str):
        """
        Find a product by ID.
        
        Args:
            product_id: Product ID to search for
            
        Returns:
            Product object if found, None otherwise
        """
        from .product import Product
        
        products_data = await self.get_products()
        for product_data in products_data:
            if str(product_data.get('id', '')) == str(product_id):
                return Product.from_metadata(product_data)
        return None
    
    async def find_product_by_url(self, product_url: str):
        """
        Find a product by URL.
        
        Args:
            product_url: Product URL to search for (base URL without query params)
            
        Returns:
            Product object if found, None otherwise
        """
        from .product import Product
        
        # Get the base URL
        base_url = product_url.split('?')[0]
        
        products_data = await self.get_products()
        for product_data in products_data:
            if product_data.get('productUrl', '') == base_url:
                return Product.from_metadata(product_data)
        return None
    
    async def get_product_objects(self):
        """
        Get products as Product objects.
        
        Returns:
            List of Product objects
        """
        from .product import Product
        
        products_data = await self.get_products()
        return [Product.from_metadata(metadata=item) for item in products_data]
    
    async def get_memory_info(self) -> Dict[str, Any]:
        """
        Get memory usage information for this account.
        
        Returns:
            Dictionary with memory usage information
        """
        return {
            "account": self.account,
            "product_count": len(self._products) if self._products else 0,
            "memory_bytes": self._memory_size,
            "memory_mb": round(self._memory_size / 1024 / 1024, 2),
            "last_loaded": self._last_loaded.isoformat() if self._last_loaded else None,
            "loaded": self._products is not None
        }
    
    async def load_at_startup(self) -> bool:
        """
        Load product catalog once at voice agent startup.
        
        Simple loading order:
        1. GCP storage 
        2. Local files
        3. Empty list
        
        No Redis caching - direct to memory for stateful voice agent.
        
        Returns:
            True if products loaded successfully
        """
        async with self._loading_lock:
            try:
                # Try GCP storage first
                storage_products = await self._load_from_storage()
                if storage_products is not None:
                    self._products = storage_products
                    self._last_loaded = datetime.now()
                    self._memory_size = len(json.dumps(storage_products).encode('utf-8'))
                    
                    logger.info(f"Loaded {len(storage_products)} products for {self.account} from GCP storage ({self._memory_size:,} bytes)")
                    return True
                
                # Fallback to local files
                local_products = await self._load_from_local_files()
                if local_products is not None:
                    self._products = local_products
                    self._last_loaded = datetime.now()
                    self._memory_size = len(json.dumps(local_products).encode('utf-8'))
                    
                    logger.info(f"Loaded {len(local_products)} products for {self.account} from local files ({self._memory_size:,} bytes)")
                    return True
                
                # Empty list if all fails
                logger.warning(f"No product catalog found for {self.account}, using empty list")
                self._products = []
                self._last_loaded = datetime.now()
                self._memory_size = 0
                return False
                
            except Exception as e:
                logger.error(f"Error loading products for {self.account}: {e}")
                self._products = []
                self._last_loaded = datetime.now()
                self._memory_size = 0
                return False
    
    async def refresh_from_storage(self) -> bool:
        """
        Optional: Refresh product catalog from storage without restart.
        
        Useful for updating product catalogs while voice agent is running.
        
        Returns:
            True if refresh was successful
        """
        return await self.load_at_startup()
    
    async def _load_from_storage(self) -> Optional[List[Dict[str, Any]]]:
        """Load products from GCP storage"""
        try:
            return await self.storage_provider.get_product_catalog(self.account)
        except Exception as e:
            logger.debug(f"GCP storage load failed for {self.account}: {e}")
            return None
    
    async def _load_from_local_files(self) -> Optional[List[Dict[str, Any]]]:
        """Load products from local files as fallback"""
        try:
            import os
            
            # Try account-specific file first
            fallback_path = f"data/{self.account}/products.json"
            if os.path.exists(fallback_path):
                with open(fallback_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            
            # Try global file as last resort
            global_path = "data/products.json"
            if os.path.exists(global_path):
                with open(global_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            
            return None
            
        except Exception as e:
            logger.debug(f"Fallback load failed for {self.account}: {e}")
            return None
    
    async def save_products(self, products: List[Dict[str, Any]]) -> bool:
        """
        Save product catalog to GCP storage and update memory.
        
        Args:
            products: List of product dictionaries to save
            
        Returns:
            True if save was successful
        """
        try:
            success = await self.storage_provider.save_product_catalog(self.account, products)
            if success:
                # Update memory cache
                self._products = products
                self._last_loaded = datetime.now()
                self._memory_size = len(json.dumps(products).encode('utf-8'))
                
                logger.info(f"Saved {len(products)} products for {self.account} ({self._memory_size:,} bytes in memory)")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error saving products for {self.account}: {e}")
            return False
    
    def get_product_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Get metadata about the product catalog in memory.
        
        Returns:
            Dictionary with catalog metadata or None
        """
        try:
            if self._products is not None:
                return {
                    "account": self.account,
                    "product_count": len(self._products),
                    "memory_size_bytes": self._memory_size,
                    "memory_size_mb": round(self._memory_size / 1024 / 1024, 2),
                    "last_loaded": self._last_loaded.isoformat() if self._last_loaded else None,
                    "source": "memory",
                    "loaded": True
                }
            else:
                return {
                    "account": self.account,
                    "product_count": 0,
                    "memory_size_bytes": 0,
                    "memory_size_mb": 0,
                    "last_loaded": None,
                    "source": "none",
                    "loaded": False
                }
            
        except Exception as e:
            logger.error(f"Error getting product metadata for {self.account}: {e}")
            return None
    
    def clear_memory(self) -> bool:
        """
        Clear product catalog from memory.
        
        Returns:
            True if memory was cleared
        """
        try:
            self._products = None
            self._last_loaded = None
            self._memory_size = 0
            logger.info(f"Cleared product catalog from memory for {self.account}")
            return True
        except Exception as e:
            logger.error(f"Error clearing memory for {self.account}: {e}")
            return False


# Global ProductManager memory storage for voice agents
_product_managers: Dict[str, ProductManager] = {}


async def get_product_manager(account: str) -> ProductManager:
    """
    Get singleton ProductManager for account.
    
    This is the main entry point for all product operations. Uses singleton pattern
    to ensure one ProductManager instance per account across the application.
    
    Args:
        account: Account domain
        
    Returns:
        ProductManager instance (singleton for this account)
    """
    if account not in _product_managers:
        _product_managers[account] = await ProductManager.create_for_startup(account)
    return _product_managers[account]


async def load_product_catalog_for_assistant(account: str) -> ProductManager:
    """
    Load product catalog for a single account when Assistant is instantiated.
    
    KISS approach for voice agents: Load only the catalog needed for this specific account.
    Voice agents serve one account at a time, so we only load what we need.
    
    Args:
        account: Account domain for this voice agent instance
        
    Returns:
        ProductManager instance with products loaded for this account
    """
    try:
        # Check if ProductManager already exists for this account
        if account in _product_managers:
            logger.info(f"Using existing ProductManager for voice agent: {account}")
            return _product_managers[account]
        
        # Create new ProductManager and load catalog
        manager = await ProductManager.create_for_startup(account)
        _product_managers[account] = manager
        
        # Get product count from manager (no circular dependency)
        products = await manager.get_products()
        product_count = len(products)
        
        logger.info(f"Loaded product catalog for voice agent: {account} "
                   f"({product_count} products, {manager._memory_size:,} bytes)")
        
        return manager
        
    except Exception as e:
        logger.error(f"Failed to load product catalog for Assistant account {account}: {e}")
        # Return empty manager as fallback
        manager = ProductManager(account)
        manager._products = []
        manager._memory_size = 0
        manager._last_loaded = datetime.now()
        _product_managers[account] = manager
        return manager


def get_assistant_memory_usage(account: str) -> Dict[str, Any]:
    """
    Get memory usage for a specific Assistant's product catalog.
    
    Args:
        account: Account domain for this Assistant
        
    Returns:
        Dictionary with memory usage for this specific account
    """
    if account in _product_managers:
        mgr = _product_managers[account]
        return {
            "account": account,
            "product_count": len(mgr._products) if mgr._products else 0,
            "memory_bytes": mgr._memory_size,
            "memory_mb": round(mgr._memory_size / 1024 / 1024, 2),
            "last_loaded": mgr._last_loaded.isoformat() if mgr._last_loaded else None,
            "loaded": mgr._products is not None
        }
    else:
        return {
            "account": account,
            "product_count": 0,
            "memory_bytes": 0,
            "memory_mb": 0,
            "last_loaded": None,
            "loaded": False
        }


def get_memory_usage() -> Dict[str, Any]:
    """
    Get memory usage statistics for all loaded product catalogs.
    
    Note: In typical voice agent usage, this will usually show one account
    since each Assistant instance serves a single account.
    
    Returns:
        Dictionary with memory usage statistics
    """
    total_size = sum(mgr._memory_size for mgr in _product_managers.values())
    total_products = sum(len(mgr._products) if mgr._products else 0 for mgr in _product_managers.values())
    
    return {
        "total_catalogs": len(_product_managers),
        "total_products": total_products,
        "total_memory_bytes": total_size,
        "total_memory_mb": round(total_size / 1024 / 1024, 2),
        "accounts": {
            account: {
                "product_count": len(mgr._products) if mgr._products else 0,
                "memory_bytes": mgr._memory_size,
                "memory_mb": round(mgr._memory_size / 1024 / 1024, 2)
            }
            for account, mgr in _product_managers.items()
        }
    }


def clear_all_product_managers():
    """Clear all ProductManager instances from memory"""
    global _product_managers
    total_cleared = sum(mgr._memory_size for mgr in _product_managers.values())
    _product_managers.clear()
    logger.info(f"Cleared all product catalogs from memory ({total_cleared:,} bytes freed)")


# =============================================================================
# CONVENIENCE FUNCTIONS - Clean API using singleton ProductManager pattern
# =============================================================================

async def find_product_by_id(account: str, product_id: str):
    """
    Find a product by ID using singleton ProductManager.
    
    Args:
        account: Account domain
        product_id: Product ID to search for
        
    Returns:
        Product object if found, None otherwise
    """
    manager = await get_product_manager(account)
    return await manager.find_product_by_id(product_id)


async def find_product_by_url(account: str, product_url: str):
    """
    Find a product by URL using singleton ProductManager.
    
    Args:
        account: Account domain
        product_url: Product URL to search for
        
    Returns:
        Product object if found, None otherwise
    """
    manager = await get_product_manager(account)
    return await manager.find_product_by_url(product_url)


async def get_products_for_account(account: str):
    """
    Get all products for an account as Product objects.
    
    Args:
        account: Account domain
        
    Returns:
        List of Product objects
    """
    manager = await get_product_manager(account)
    return await manager.get_product_objects()


async def save_products_for_account(account: str, products) -> bool:
    """
    Save products for an account.
    
    Args:
        account: Account domain
        products: List of Product objects or dictionaries
        
    Returns:
        True if save was successful
    """
    manager = await get_product_manager(account)
    
    # Convert Product objects to dictionaries if needed
    if products and hasattr(products[0], 'to_dict'):
        product_data = [p.to_dict() for p in products]
    else:
        product_data = products
    
    return await manager.save_products(product_data)


async def refresh_products_for_account(account: str) -> bool:
    """
    Refresh product catalog from storage for an account.
    
    Args:
        account: Account domain
        
    Returns:
        True if refresh was successful
    """
    manager = await get_product_manager(account)
    return await manager.refresh_from_storage()


async def get_product_metadata_for_account(account: str):
    """
    Get product catalog metadata for an account.
    
    Args:
        account: Account domain
        
    Returns:
        Dictionary with catalog metadata or None
    """
    manager = await get_product_manager(account)
    return manager.get_product_metadata()


async def get_memory_info_for_account(account: str):
    """
    Get memory usage information for an account.
    
    Args:
        account: Account domain
        
    Returns:
        Dictionary with memory information
    """
    manager = await get_product_manager(account)
    return await manager.get_memory_info()


def clear_products_for_account(account: str) -> bool:
    """
    Clear product catalog from memory for an account.
    
    Args:
        account: Account domain
        
    Returns:
        True if memory was cleared
    """
    if account in _product_managers:
        manager = _product_managers[account]
        success = manager.clear_memory()
        if success:
            del _product_managers[account]
        return success
    return True 