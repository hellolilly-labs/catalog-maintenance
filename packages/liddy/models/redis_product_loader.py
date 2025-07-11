"""
Redis Product Loader - Loads product catalogs from GCS into Redis.

This module provides functionality to load product catalogs from Google Cloud Storage
into Redis for efficient, memory-constrained access by voice agents.
"""

import json
import logging
import os
import time
from typing import List, Optional
import redis.asyncio as redis

from liddy.storage import get_account_storage_provider
from liddy.models.product import Product

logger = logging.getLogger(__name__)


class RedisProductLoader:
    """Load and manage product catalogs in Redis."""
    
    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize the Redis product loader.
        
        Args:
            redis_url: Redis connection URL (defaults to env vars or localhost)
        """
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
        self.redis_client = None
        self.storage_provider = get_account_storage_provider()
        
    async def connect(self):
        """Connect to Redis."""
        logger.info(f"Connecting to Redis for product loading at {self.redis_url}")
        self.redis_client = await redis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True
        )
        # Test connection
        await self.redis_client.ping()
        logger.info("✅ Connected to Redis")
        
    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.aclose()
            logger.info("Disconnected from Redis")
    
    async def load_account_products(self, account: str) -> int:
        """
        Load products for a single account into Redis.
        
        This method completely refreshes the product catalog:
        - Loads all products from the source of truth (GCS)
        - Removes any products that no longer exist
        - Updates all product data
        
        Args:
            account: Account domain (e.g., "specialized.com")
            
        Returns:
            Number of products loaded
        """
        start_time = time.time()
        logger.info(f"Loading products for {account}...")
        
        try:
            # Get products from storage (source of truth)
            products_data = await self.storage_provider.get_product_catalog(account)
            if not products_data:
                logger.warning(f"No products found for {account}")
                # Clear all products for this account since none exist
                await self._clear_account_products(account)
                return 0
            
            # Get existing product IDs to identify stale ones
            existing_product_ids = await self.redis_client.smembers(f"products:{account}")
            new_product_ids = set()
            
            # Prepare pipeline for efficient bulk loading
            pipe = self.redis_client.pipeline()
            
            # Store each product individually for efficient access
            product_count = 0
            for product_dict in products_data:
                product = Product.from_dict(product=product_dict)
                new_product_ids.add(str(product.id))
                
                # Store full product as JSON
                product_key = f"product:{account}:{product.id}"
                pipe.set(product_key, json.dumps(product.to_dict()))
                
                # Add to account's product set
                pipe.sadd(f"products:{account}", product.id)
                
                # Create search indexes
                # Index by name (for text search)
                if product.name:
                    name_key = f"product_name:{account}:{product.name.lower()}"
                    pipe.sadd(name_key, product.id)
                
                # Index by categories
                if product.categories:
                    for category in product.categories:
                        category_key = f"product_category:{account}:{category.lower()}"
                        pipe.sadd(category_key, product.id)
                
                product_count += 1
                
                # Execute pipeline every 100 products to avoid memory issues
                if product_count % 100 == 0:
                    await pipe.execute()
                    pipe = self.redis_client.pipeline()
                    logger.debug(f"  Loaded {product_count} products...")
            
            # Execute remaining commands
            if product_count % 100 != 0:
                await pipe.execute()
            
            # Remove stale products that no longer exist
            stale_product_ids = existing_product_ids - new_product_ids
            if stale_product_ids:
                logger.info(f"Removing {len(stale_product_ids)} stale products for {account}")
                await self._remove_stale_products(account, stale_product_ids)
            
            # Store metadata about the catalog
            metadata = {
                "product_count": product_count,
                "loaded_at": time.time(),
                "size_bytes": len(json.dumps(products_data).encode('utf-8')),
                "removed_count": len(stale_product_ids)
            }
            await self.redis_client.hset(
                f"catalog_metadata:{account}",
                mapping=metadata
            )
            
            # Set expiration (24 hours by default)
            ttl = int(os.getenv('REDIS_PRODUCT_TTL', 86400))
            if ttl > 0:
                # Set TTL on all keys for this account
                cursor = 0
                while True:
                    cursor, keys = await self.redis_client.scan(
                        cursor, 
                        match=f"*:{account}:*",
                        count=100
                    )
                    if keys:
                        pipe = self.redis_client.pipeline()
                        for key in keys:
                            pipe.expire(key, ttl)
                        await pipe.execute()
                    if cursor == 0:
                        break
            
            load_time = time.time() - start_time
            logger.info(f"✅ Loaded {product_count} products for {account} in {load_time:.2f}s" + 
                       (f" (removed {len(stale_product_ids)} stale)" if stale_product_ids else ""))
            
            # Log memory usage estimate
            memory_mb = metadata['size_bytes'] / (1024 * 1024)
            logger.info(f"   Memory usage: ~{memory_mb:.1f} MB in Redis")
            
            return product_count
            
        except Exception as e:
            logger.error(f"❌ Failed to load products for {account}: {e}")
            return 0
    
    async def load_all_accounts(self, accounts: Optional[List[str]] = None) -> dict:
        """
        Load products for all specified accounts.
        
        Args:
            accounts: List of account domains. If None, loads from environment.
            
        Returns:
            Dictionary mapping account to product count
        """
        if not accounts:
            # Get accounts from environment variable
            env_accounts = os.getenv('VOICE_ACCOUNTS', '')
            accounts = [a.strip() for a in env_accounts.split(',') if a.strip()]
            
            if not accounts:
                # Default to all available accounts
                logger.info("No accounts specified - loading all available accounts")
                try:
                    accounts = await self.storage_provider.get_accounts()
                    logger.info(f"Found {len(accounts)} accounts in storage")
                except Exception as e:
                    logger.error(f"Failed to get accounts from storage: {e}")
                    accounts = []
        
        if not accounts:
            logger.warning("No accounts specified to load")
            return {}
        
        logger.info(f"Loading products for {len(accounts)} accounts: {accounts}")
        
        results = {}
        total_start = time.time()
        
        for account in accounts:
            try:
                count = await self.load_account_products(account)
                results[account] = count
            except Exception as e:
                logger.error(f"Failed to load {account}: {e}")
                results[account] = 0
        
        total_time = time.time() - total_start
        total_products = sum(results.values())
        
        logger.info(f"\n{'='*50}")
        logger.info(f"✅ LOADING COMPLETE")
        logger.info(f"{'='*50}")
        logger.info(f"Total accounts: {len(results)}")
        logger.info(f"Total products: {total_products}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"{'='*50}\n")
        
        return results
    
    async def verify_loading(self, account: str) -> bool:
        """
        Verify that products were loaded correctly for an account.
        
        Args:
            account: Account domain to verify
            
        Returns:
            True if products are accessible
        """
        try:
            # Check if we have products
            product_ids = await self.redis_client.smembers(f"products:{account}")
            if not product_ids:
                return False
            
            # Try to load a sample product
            sample_id = list(product_ids)[0]
            product_data = await self.redis_client.get(f"product:{account}:{sample_id}")
            
            if product_data:
                product = json.loads(product_data)
                logger.info(f"✅ Verified: Found product '{product.get('name', 'Unknown')}' for {account}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Verification failed for {account}: {e}")
            return False
    
    async def _clear_account_products(self, account: str) -> None:
        """
        Clear all products for an account from Redis.
        
        Args:
            account: Account domain to clear
        """
        try:
            # Get all product IDs for the account
            product_ids = await self.redis_client.smembers(f"products:{account}")
            
            if product_ids:
                pipe = self.redis_client.pipeline()
                
                # Delete each product
                for product_id in product_ids:
                    pipe.delete(f"product:{account}:{product_id}")
                
                # Clear the product set
                pipe.delete(f"products:{account}")
                
                # Clear search indexes
                cursor = 0
                while True:
                    cursor, keys = await self.redis_client.scan(
                        cursor, 
                        match=f"product_*:{account}:*",
                        count=100
                    )
                    if keys:
                        for key in keys:
                            pipe.delete(key)
                    if cursor == 0:
                        break
                
                await pipe.execute()
                logger.info(f"Cleared {len(product_ids)} products for {account}")
                
        except Exception as e:
            logger.error(f"Failed to clear products for {account}: {e}")
    
    async def _remove_stale_products(self, account: str, stale_product_ids: set) -> None:
        """
        Remove stale products that no longer exist in the source.
        
        Args:
            account: Account domain
            stale_product_ids: Set of product IDs to remove
        """
        try:
            pipe = self.redis_client.pipeline()
            
            for product_id in stale_product_ids:
                # Remove product data
                pipe.delete(f"product:{account}:{product_id}")
                
                # Remove from product set
                pipe.srem(f"products:{account}", product_id)
                
                # TODO: Clean up search indexes for this product
                # This would require loading the product to get its name/categories
                # For now, indexes will be cleaned up on next full reload
            
            await pipe.execute()
            logger.debug(f"Removed {len(stale_product_ids)} stale products for {account}")
            
        except Exception as e:
            logger.error(f"Failed to remove stale products for {account}: {e}")


# Convenience function for loading a single account
async def load_products_to_redis(account: str, redis_url: Optional[str] = None) -> bool:
    """
    Load products for a single account into Redis.
    
    Args:
        account: Account domain (e.g., "specialized.com")
        redis_url: Optional Redis URL override
        
    Returns:
        True if successful
    """
    loader = RedisProductLoader(redis_url)
    try:
        await loader.connect()
        count = await loader.load_account_products(account)
        if count > 0:
            verified = await loader.verify_loading(account)
            return verified
        return False
    finally:
        await loader.disconnect()