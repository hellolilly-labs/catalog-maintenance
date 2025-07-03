"""
Product Catalog Cache

Size-aware Redis caching system for product catalogs with intelligent TTL management.
Optimized for different catalog sizes:
- Large catalogs (>1MB): 15-minute TTL
- Medium catalogs (>50KB): 10-minute TTL  
- Small catalogs (≤50KB): 5-minute TTL
"""

import json
import gzip
import hashlib
import base64
from typing import List, Dict, Any, Optional
import logging

from redis_client import get_redis_client

logger = logging.getLogger(__name__)


class ProductCatalogCache:
    """Size-aware caching system for product catalogs"""
    
    # TTL mapping based on catalog size
    SIZE_TTL_MAP = {
        "large": 900,    # 15 minutes for >1MB
        "medium": 600,   # 10 minutes for >50KB  
        "small": 300     # 5 minutes for ≤50KB
    }
    
    def __init__(self):
        self._redis_client = None
    
    @property
    def redis_client(self):
        """Lazy initialization of Redis client"""
        if self._redis_client is None:
            self._redis_client = get_redis_client()
        return self._redis_client
    
    def get_cache_key(self, account: str, catalog_type: str = "products") -> str:
        """Generate cache key for product catalog"""
        return f"product_catalog:{account}:{catalog_type}"
    
    def get_metadata_key(self, account: str, catalog_type: str = "products") -> str:
        """Generate cache key for catalog metadata"""
        return f"product_catalog_meta:{account}:{catalog_type}"
    
    def get_size_category(self, catalog_size: int) -> str:
        """Determine size category for TTL selection"""
        if catalog_size > 1024 * 1024:  # >1MB
            return "large"
        elif catalog_size > 50 * 1024:  # >50KB (adjusted threshold)
            return "medium"
        else:
            return "small"
    
    def get_ttl_for_size(self, catalog_size: int) -> int:
        """Return appropriate TTL based on catalog size"""
        size_category = self.get_size_category(catalog_size)
        return self.SIZE_TTL_MAP[size_category]
    
    def _should_compress_for_cache(self, catalog_size: int) -> bool:
        """Determine if catalog should be compressed for cache storage"""
        return catalog_size > 500 * 1024  # >500KB
    
    def _compress_for_cache(self, products: List[dict]) -> str:
        """Compress products for cache storage and encode as base64 string"""
        json_data = json.dumps(products, separators=(',', ':'))
        compressed_bytes = gzip.compress(json_data.encode('utf-8'))
        # Encode as base64 string to store safely in Redis
        return base64.b64encode(compressed_bytes).decode('ascii')
    
    def _decompress_from_cache(self, compressed_data_str: str) -> List[dict]:
        """Decompress products from cache (base64 encoded compressed data)"""
        # Decode from base64 first
        compressed_bytes = base64.b64decode(compressed_data_str.encode('ascii'))
        json_data = gzip.decompress(compressed_bytes).decode('utf-8')
        return json.loads(json_data)
    
    def _get_catalog_hash(self, products: List[dict]) -> str:
        """Generate hash for catalog content verification"""
        content = json.dumps(products, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(content.encode()).hexdigest()[:12]
    
    async def get_catalog_cache(self, account: str) -> Optional[List[dict]]:
        """Get cached product catalog for account"""
        try:
            cache_key = self.get_cache_key(account)
            metadata_key = self.get_metadata_key(account)
            
            # Get metadata first to know if compressed
            metadata_data = self.redis_client.get(metadata_key)
            if not metadata_data:
                logger.debug(f"No cached metadata found for product catalog: {account}")
                return None
            
            # Handle bytes from Redis
            if isinstance(metadata_data, bytes):
                metadata_data = metadata_data.decode('utf-8')
            metadata = json.loads(metadata_data)
            
            # Get catalog data
            cached_data = self.redis_client.get(cache_key)
            if not cached_data:
                logger.debug(f"No cached product catalog found for: {account}")
                return None
            
            # Handle bytes from Redis
            if isinstance(cached_data, bytes):
                cached_data = cached_data.decode('utf-8')
            
            # Decompress if needed
            if metadata.get("compressed", False):
                products = self._decompress_from_cache(cached_data)
            else:
                products = json.loads(cached_data)
            
            # Verify integrity
            expected_hash = metadata.get("content_hash")
            if expected_hash:
                actual_hash = self._get_catalog_hash(products)
                if actual_hash != expected_hash:
                    logger.warning(f"Product catalog cache integrity check failed for {account}")
                    # Remove corrupted cache
                    self.invalidate_catalog_cache(account)
                    return None
            
            logger.debug(f"Retrieved cached product catalog for {account}: {len(products)} products")
            return products
            
        except Exception as e:
            logger.error(f"Error retrieving cached product catalog for {account}: {e}")
            return None
    
    async def set_catalog_cache(self, account: str, products: List[dict], 
                               metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Cache product catalog with size-appropriate TTL"""
        try:
            cache_key = self.get_cache_key(account)
            metadata_key = self.get_metadata_key(account)
            
            # Prepare catalog data
            json_content = json.dumps(products, separators=(',', ':'))
            catalog_size = len(json_content.encode('utf-8'))
            
            # Determine TTL based on size
            ttl = self.get_ttl_for_size(catalog_size)
            size_category = self.get_size_category(catalog_size)
            
            # Prepare metadata
            cache_metadata = {
                "product_count": len(products),
                "original_size": catalog_size,
                "size_category": size_category,
                "ttl": ttl,
                "content_hash": self._get_catalog_hash(products),
                "compressed": False,
                "cached_at": "now"  # Redis will add timestamp
            }
            
            # Add external metadata if provided
            if metadata:
                cache_metadata.update(metadata)
            
            # Compress for cache if needed
            if self._should_compress_for_cache(catalog_size):
                compressed_data = self._compress_for_cache(products)
                compressed_size = len(compressed_data.encode('utf-8'))
                
                cache_metadata.update({
                    "compressed": True,
                    "compressed_size": compressed_size,
                    "compression_ratio": round(compressed_size / catalog_size, 3)
                })
                
                # Store compressed data (as base64 string)
                self.redis_client.setex(cache_key, ttl, compressed_data)
                
                logger.info(f"Cached compressed product catalog for {account}: "
                           f"{len(products)} products, {catalog_size} -> {compressed_size} bytes, "
                           f"TTL: {ttl}s ({size_category})")
            else:
                # Store uncompressed
                self.redis_client.setex(cache_key, ttl, json_content)
                
                logger.info(f"Cached product catalog for {account}: "
                           f"{len(products)} products, {catalog_size} bytes, "
                           f"TTL: {ttl}s ({size_category})")
            
            # Store metadata
            self.redis_client.setex(
                metadata_key, 
                ttl, 
                json.dumps(cache_metadata, separators=(',', ':'))
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error caching product catalog for {account}: {e}")
            return False
    
    def invalidate_catalog_cache(self, account: str) -> bool:
        """Remove cached product catalog for account"""
        try:
            cache_key = self.get_cache_key(account)
            metadata_key = self.get_metadata_key(account)
            
            # Delete both keys
            deleted_count = self.redis_client.delete(cache_key, metadata_key)
            
            if deleted_count > 0:
                logger.info(f"Invalidated product catalog cache for {account}")
                return True
            else:
                logger.debug(f"No cached product catalog to invalidate for {account}")
                return False
                
        except Exception as e:
            logger.error(f"Error invalidating product catalog cache for {account}: {e}")
            return False
    
    def get_cache_info(self, account: str) -> Optional[Dict[str, Any]]:
        """Get cache information for account without loading full catalog"""
        try:
            metadata_key = self.get_metadata_key(account)
            metadata_data = self.redis_client.get(metadata_key)
            
            if not metadata_data:
                return None
            
            # Handle bytes from Redis
            if isinstance(metadata_data, bytes):
                metadata_data = metadata_data.decode('utf-8')
            metadata = json.loads(metadata_data)
            
            # Add TTL info
            cache_key = self.get_cache_key(account)
            ttl = self.redis_client.ttl(cache_key)
            
            metadata.update({
                "cache_ttl_remaining": ttl if ttl > 0 else None,
                "cache_exists": ttl > 0
            })
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting cache info for {account}: {e}")
            return None
    
    def get_all_cached_accounts(self) -> List[str]:
        """Get list of all accounts with cached product catalogs"""
        try:
            pattern = "product_catalog:*:products"
            keys = self.redis_client.keys(pattern)
            
            # Extract account names from keys
            accounts = []
            for key in keys:
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                
                # Extract account from key: product_catalog:{account}:products
                parts = key.split(':')
                if len(parts) >= 3:
                    account = parts[1]
                    accounts.append(account)
            
            return sorted(list(set(accounts)))
            
        except Exception as e:
            logger.error(f"Error getting cached accounts: {e}")
            return []
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get overall cache statistics"""
        try:
            accounts = self.get_all_cached_accounts()
            stats = {
                "total_cached_accounts": len(accounts),
                "by_size_category": {"large": 0, "medium": 0, "small": 0},
                "total_products": 0,
                "total_cache_size": 0,
                "compression_stats": {"compressed": 0, "uncompressed": 0}
            }
            
            for account in accounts:
                info = self.get_cache_info(account)
                if info:
                    size_category = info.get("size_category", "unknown")
                    if size_category in stats["by_size_category"]:
                        stats["by_size_category"][size_category] += 1
                    
                    stats["total_products"] += info.get("product_count", 0)
                    
                    cache_size = info.get("compressed_size") or info.get("original_size", 0)
                    stats["total_cache_size"] += cache_size
                    
                    if info.get("compressed", False):
                        stats["compression_stats"]["compressed"] += 1
                    else:
                        stats["compression_stats"]["uncompressed"] += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)} 