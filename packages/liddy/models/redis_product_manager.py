"""
Redis-backed Product Manager for memory-efficient product access.

This implementation uses Redis as a shared cache for product data,
allowing multiple containers to access products without loading them
into memory. Perfect for memory-constrained environments like Cloud Run.
"""

import asyncio
import json
import logging
import os
from typing import List, Optional, Dict, Any
from datetime import datetime
import redis.asyncio as redis

from .product import Product

logger = logging.getLogger(__name__)


class RedisProductManager:
    """
    Redis-backed product catalog manager for memory-efficient access.
    
    Instead of loading 10MB+ of products into each container's memory,
    this manager fetches products on-demand from Redis, keeping memory
    usage minimal while maintaining fast access times.
    """
    
    def __init__(self, account: str, redis_url: Optional[str] = None):
        """
        Initialize RedisProductManager for an account.
        
        Args:
            account: Account domain (e.g., "specialized.com")
            redis_url: Redis connection URL (defaults to env vars or localhost)
        """
        self.account = account
        if redis_url:
            self.redis_url = redis_url
        else:
            # Construct URL from REDIS_HOST and REDIS_PORT if available
            redis_host = os.getenv('REDIS_HOST', 'localhost')
            redis_port = os.getenv('REDIS_PORT', '6379')
            redis_password = os.getenv('REDIS_PASSWORD', '')
            
            if redis_password:
                self.redis_url = f"redis://:{redis_password}@{redis_host}:{redis_port}"
            else:
                self.redis_url = f"redis://{redis_host}:{redis_port}"
        self._redis_client = None
        self._connection_lock = asyncio.Lock()
        self._products_available = None  # Cache availability check
        self._product_count = None
        
    async def _get_redis_client(self) -> redis.Redis:
        """Get or create Redis client connection."""
        if self._redis_client is None:
            async with self._connection_lock:
                if self._redis_client is None:
                    logger.debug(f"Connecting to Redis for {self.account}")
                    self._redis_client = await redis.from_url(
                        self.redis_url,
                        encoding="utf-8",
                        decode_responses=True,
                        socket_connect_timeout=5,
                        socket_timeout=5
                    )
                    # Test connection
                    await self._redis_client.ping()
                    logger.info(f"Connected to Redis for {self.account}")
        return self._redis_client
    
    async def get_products(self) -> List[Product]:
        """
        Get all products for the account from Redis.
        
        Note: This method loads ALL products and should be avoided in 
        memory-constrained environments. Use find_product_by_id() or
        search methods instead.
        
        Returns:
            List of Product objects
        """
        try:
            client = await self._get_redis_client()
            
            # Get all product IDs for the account
            product_ids = await client.smembers(f"products:{self.account}")
            if not product_ids:
                logger.warning(f"No products found in Redis for {self.account}")
                return []
            
            # Fetch products in batches using pipeline
            products = []
            batch_size = 100
            product_id_list = list(product_ids)
            
            for i in range(0, len(product_id_list), batch_size):
                batch_ids = product_id_list[i:i + batch_size]
                pipe = client.pipeline()
                
                for product_id in batch_ids:
                    pipe.get(f"product:{self.account}:{product_id}")
                
                results = await pipe.execute()
                
                for product_data in results:
                    if product_data:
                        product_dict = json.loads(product_data)
                        products.append(Product.from_dict(product=product_dict))
            
            logger.debug(f"Loaded {len(products)} products from Redis for {self.account}")
            return products
            
        except Exception as e:
            logger.error(f"Error loading products from Redis for {self.account}: {e}")
            return []
    
    async def find_product_by_id(self, product_id: str) -> Optional[Product]:
        """
        Find a single product by ID.
        
        This is the preferred method for memory-efficient access.
        
        Args:
            product_id: Product ID to search for
            
        Returns:
            Product object if found, None otherwise
        """
        try:
            client = await self._get_redis_client()
            
            # Fetch the specific product
            product_data = await client.get(f"product:{self.account}:{product_id}")
            if product_data:
                product_dict = json.loads(product_data)
                return Product.from_dict(product=product_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding product {product_id} in Redis: {e}")
            return None
    
    async def find_products_by_ids(self, product_ids: List[str]) -> List[Product]:
        """
        Find multiple products by their IDs efficiently.
        
        Args:
            product_ids: List of product IDs to fetch
            
        Returns:
            List of Product objects (only found products)
        """
        if not product_ids:
            return []
            
        try:
            client = await self._get_redis_client()
            
            # Use pipeline for efficient batch fetching
            pipe = client.pipeline()
            for product_id in product_ids:
                pipe.get(f"product:{self.account}:{product_id}")
            
            results = await pipe.execute()
            
            products = []
            for product_data in results:
                if product_data:
                    product_dict = json.loads(product_data)
                    products.append(Product.from_dict(product=product_dict))
            
            return products
            
        except Exception as e:
            logger.error(f"Error finding products by IDs in Redis: {e}")
            return []
    
    async def find_product_by_url(self, product_url: str) -> Optional[Product]:
        """
        Find a product by URL.
        
        Note: This requires iterating through products, so it's less efficient
        than find_by_id. Consider adding a URL index in Redis if this is
        frequently used.
        
        Args:
            product_url: Product URL to search for
            
        Returns:
            Product object if found, None otherwise
        """
        # Get the base URL
        base_url = product_url.split('?')[0]
        
        # For now, we need to load all products and search
        # TODO: Add URL indexing in Redis for efficient lookup
        products = await self.get_products()
        for product in products:
            if product.productUrl == base_url:
                return product
        return None
    
    async def find_product_from_url_smart(self, url: str, fallback_to_url_lookup: bool = False) -> Optional[Product]:
        """
        Find a product by URL using smart extraction.
        
        Args:
            url: Product URL to search for
            fallback_to_url_lookup: If True and ID extraction fails, fallback to find_product_by_url
            
        Returns:
            Product if found, None otherwise
        """
        import re
        from liddy.account_config_loader import get_account_config_loader
        
        # Get account config for URL pattern
        account_config_loader = get_account_config_loader()
        config_data = await account_config_loader.get_account_config(account=self.account)
        
        # Default generic pattern - extracts product_id from last URL segment
        # Examples:
        # - https://www.specialized.com/us/en/diverge-comp-e5/p/4223497?color=5381902-4223497
        # - https://www.brand.com/category/product-name/12345
        # - https://example.com/products/some-item/98765?variant=abc
        default_pattern = r"^https?://[^/]+(?:/[^/?]*)*?/(?P<product_id>\d+)(?:\?.*)?$"
        product_url_pattern = config_data.get("product_url_pattern", default_pattern) if config_data else default_pattern
        
        # Try regex extraction first
        match = re.match(product_url_pattern, url)
        if match:
            # Extract product_id from named group or first group
            product_id = None
            if 'product_id' in match.groupdict():
                product_id = match.group('product_id')
            elif match.groups():
                product_id = match.group(1)
            
            if product_id:
                logger.debug(f"Extracted product ID '{product_id}' from URL using pattern")
                product = await self.find_product_by_id(product_id)
                if product:
                    return product
                else:
                    logger.debug(f"Product ID '{product_id}' not found in catalog")
        
        # Fallback to URL lookup if requested
        if fallback_to_url_lookup:
            logger.debug(f"Falling back to URL lookup for: {url}")
            return await self.find_product_by_url(url)
        
        return None
    
    async def get_product_count(self) -> int:
        """
        Get the number of products for this account.
        
        Returns:
            Number of products in the catalog
        """
        if self._product_count is not None:
            return self._product_count
            
        try:
            client = await self._get_redis_client()
            count = await client.scard(f"products:{self.account}")
            self._product_count = count
            return count
        except Exception as e:
            logger.error(f"Error getting product count from Redis: {e}")
            return 0
    
    async def is_available(self) -> bool:
        """
        Check if products are available in Redis for this account.
        
        Returns:
            True if products are loaded in Redis
        """
        if self._products_available is not None:
            return self._products_available
            
        try:
            count = await self.get_product_count()
            self._products_available = count > 0
            return self._products_available
        except Exception:
            return False
    
    async def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the product catalog in Redis.
        
        Returns:
            Dictionary with catalog metadata
        """
        try:
            client = await self._get_redis_client()
            metadata = await client.hgetall(f"catalog_metadata:{self.account}")
            
            if metadata:
                # Convert string values to appropriate types
                if 'product_count' in metadata:
                    metadata['product_count'] = int(metadata['product_count'])
                if 'loaded_at' in metadata:
                    metadata['loaded_at'] = float(metadata['loaded_at'])
                if 'size_bytes' in metadata:
                    metadata['size_bytes'] = int(metadata['size_bytes'])
                
                return metadata
            
            # Fallback: compute basic metadata
            count = await self.get_product_count()
            return {
                "product_count": count,
                "loaded_at": None,
                "size_bytes": None,
                "source": "redis"
            }
            
        except Exception as e:
            logger.error(f"Error getting metadata from Redis: {e}")
            return {
                "product_count": 0,
                "error": str(e)
            }
    
    async def close(self):
        """Close Redis connection."""
        if self._redis_client:
            await self._redis_client.aclose()
            self._redis_client = None
            logger.debug(f"Closed Redis connection for {self.account}")
    
    # Compatibility methods
    async def get_product_objects(self) -> List[Product]:
        """Compatibility method - same as get_products()."""
        return await self.get_products()
    
    def get_products_if_loaded(self) -> Optional[List[Product]]:
        """
        Compatibility method - returns None since we don't keep products in memory.
        
        Use is_available() to check if products are in Redis.
        """
        return None
    
    @property
    def memory_usage_mb(self) -> float:
        """Return memory usage - minimal for Redis-backed manager."""
        return 0.1  # Approximate memory for connection and minimal caching
    
    def get_product_metadata(self) -> Optional[Dict[str, Any]]:
        """Sync wrapper for get_metadata() - returns None to indicate async needed."""
        return None


# Factory function to match existing interface
async def get_redis_product_manager(account: str) -> RedisProductManager:
    """
    Get or create a RedisProductManager for an account.
    
    Args:
        account: Account domain
        
    Returns:
        RedisProductManager instance
    """
    # For now, create a new instance each time
    # Could add caching if needed
    manager = RedisProductManager(account)
    
    # Verify products are available
    if not await manager.is_available():
        logger.warning(f"No products in Redis for {account} - run load_products_to_redis.py first")
    
    return manager