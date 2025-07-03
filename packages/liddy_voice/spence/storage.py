"""
Account Configuration Storage

Provides storage providers for account configurations with support for GCP and local storage.
Conversation storage is now handled by Langfuse integration.
"""

import json
import os
import gzip
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


# =====================================================
# ACCOUNT CONFIGURATION STORAGE PROVIDERS
# =====================================================

class AccountStorageProvider(ABC):
    """Extended storage provider for account configurations and product catalogs"""
    
    # Account Configuration Methods
    @abstractmethod
    async def get_account_config(self, account: str) -> Optional[Dict[str, Any]]:
        """Get account configuration by domain"""
        pass
    
    @abstractmethod
    async def save_account_config(self, account: str, config: Dict[str, Any]) -> bool:
        """Save account configuration with backup"""
        pass
    
    @abstractmethod
    async def backup_account_config(self, account: str) -> Optional[str]:
        """Create backup of existing config before update"""
        pass
    
    # Product Catalog Methods
    @abstractmethod
    async def get_product_catalog(self, account: str) -> Optional[List[dict]]:
        """Get product catalog for account"""
        pass
    
    @abstractmethod
    async def save_product_catalog(self, account: str, products: List[dict]) -> bool:
        """Save product catalog with compression and backup"""
        pass
    
    @abstractmethod
    async def get_product_catalog_metadata(self, account: str) -> Optional[Dict[str, Any]]:
        """Get product catalog metadata (size, count, last_updated)"""
        pass
    
    # Utility Methods
    def _should_compress_catalog(self, catalog_size: int) -> bool:
        """Determine if catalog should be compressed (>500KB)"""
        return catalog_size > 500 * 1024
    
    def _get_catalog_hash(self, products: List[dict]) -> str:
        """Generate hash for catalog content"""
        content = json.dumps(products, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:12]
    
    def _compress_catalog(self, catalog_data: List[dict]) -> bytes:
        """Compress catalog data using gzip"""
        json_data = json.dumps(catalog_data, separators=(',', ':'))
        return gzip.compress(json_data.encode('utf-8'))
    
    def _decompress_catalog(self, compressed_data: bytes) -> List[dict]:
        """Decompress catalog data"""
        json_data = gzip.decompress(compressed_data).decode('utf-8')
        return json.loads(json_data)

class GCPAccountStorageProvider(AccountStorageProvider):
    """GCP Storage provider with account configuration support"""
    
    def __init__(self, bucket_name: str):
        from google.cloud import storage
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
    
    async def get_account_config(self, account: str) -> Optional[Dict[str, Any]]:
        """Get account config from accounts/{domain}/account.json"""
        try:
            blob = self.bucket.blob(f"accounts/{account}/account.json")
            if not blob.exists():
                logger.info(f"No account config found for {account}")
                return None
            content = blob.download_as_text()
            return json.loads(content)
        except Exception as e:
            logger.error(f"Error loading account config for {account}: {e}")
            return None
    
    async def save_account_config(self, account: str, config: Dict[str, Any]) -> bool:
        """Save account config with backup"""
        try:
            # Create backup first
            backup_path = await self.backup_account_config(account)
            if backup_path:
                logger.info(f"Created backup for {account}: {backup_path}")
            
            # Save new config
            blob = self.bucket.blob(f"accounts/{account}/account.json")
            config['updatedAt'] = datetime.now().isoformat() + "Z"
            blob.upload_from_string(
                json.dumps(config, indent=2), 
                content_type="application/json"
            )
            
            logger.info(f"Saved account config for {account}")
            return True
        except Exception as e:
            logger.error(f"Error saving account config for {account}: {e}")
            return False
    
    async def backup_account_config(self, account: str) -> Optional[str]:
        """Create timestamped backup in accounts/{domain}/backup/"""
        try:
            source_blob = self.bucket.blob(f"accounts/{account}/account.json")
            if not source_blob.exists():
                return None
            
            # Create backup with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            backup_path = f"accounts/{account}/backup/account-{timestamp}.json"
            backup_blob = self.bucket.blob(backup_path)
            
            # Copy current config to backup
            backup_blob.upload_from_string(
                source_blob.download_as_text(),
                content_type="application/json"
            )
            
            return backup_path
        except Exception as e:
            logger.error(f"Error creating backup for {account}: {e}")
            return None
    
    # Product Catalog Methods
    async def get_product_catalog(self, account: str) -> Optional[List[dict]]:
        """Get product catalog from accounts/{domain}/products.json"""
        try:
            blob = self.bucket.blob(f"accounts/{account}/products.json")
            if not blob.exists():
                logger.info(f"No product catalog found for {account}")
                return None
            
            content = blob.download_as_text()
            products = json.loads(content)
            
            logger.info(f"Loaded product catalog for {account}: {len(products)} products")
            return products
        except Exception as e:
            logger.error(f"Error loading product catalog for {account}: {e}")
            return None
    
    async def save_product_catalog(self, account: str, products: List[dict]) -> bool:
        """Save product catalog with compression and backup"""
        try:
            # Create backup first
            backup_path = await self._backup_product_catalog(account)
            if backup_path:
                logger.info(f"Created product catalog backup for {account}: {backup_path}")
            
            # Determine if compression is needed
            json_content = json.dumps(products, indent=2)
            catalog_size = len(json_content.encode('utf-8'))
            
            blob = self.bucket.blob(f"accounts/{account}/products.json")
            
            if self._should_compress_catalog(catalog_size):
                # Save compressed version with metadata
                compressed_data = self._compress_catalog(products)
                blob.upload_from_string(
                    compressed_data,
                    content_type="application/octet-stream"
                )
                # Also save metadata
                metadata_blob = self.bucket.blob(f"accounts/{account}/products.metadata.json")
                metadata = {
                    "compressed": True,
                    "original_size": catalog_size,
                    "compressed_size": len(compressed_data),
                    "product_count": len(products),
                    "last_updated": datetime.now().isoformat() + "Z",
                    "compression_ratio": round(len(compressed_data) / catalog_size, 3)
                }
                metadata_blob.upload_from_string(
                    json.dumps(metadata, indent=2),
                    content_type="application/json"
                )
                logger.info(f"Saved compressed product catalog for {account}: {len(products)} products, {catalog_size} -> {len(compressed_data)} bytes")
            else:
                # Save uncompressed
                blob.upload_from_string(
                    json_content,
                    content_type="application/json"
                )
                logger.info(f"Saved product catalog for {account}: {len(products)} products, {catalog_size} bytes")
            
            return True
        except Exception as e:
            logger.error(f"Error saving product catalog for {account}: {e}")
            return False
    
    async def get_product_catalog_metadata(self, account: str) -> Optional[Dict[str, Any]]:
        """Get product catalog metadata"""
        try:
            # Check if metadata file exists (for compressed catalogs)
            metadata_blob = self.bucket.blob(f"accounts/{account}/products.metadata.json")
            if metadata_blob.exists():
                content = metadata_blob.download_as_text()
                return json.loads(content)
            
            # For uncompressed catalogs, generate metadata on demand
            catalog_blob = self.bucket.blob(f"accounts/{account}/products.json")
            if not catalog_blob.exists():
                return None
            
            catalog_blob.reload()  # Get updated metadata
            
            # Try to parse to get product count
            try:
                content = catalog_blob.download_as_text()
                products = json.loads(content)
                product_count = len(products)
            except Exception:
                product_count = None
            
            return {
                "compressed": False,
                "size": catalog_blob.size,
                "product_count": product_count,
                "last_updated": catalog_blob.updated.isoformat() if catalog_blob.updated else None
            }
        except Exception as e:
            logger.error(f"Error getting product catalog metadata for {account}: {e}")
            return None
    
    async def _backup_product_catalog(self, account: str) -> Optional[str]:
        """Create timestamped backup of product catalog"""
        try:
            source_blob = self.bucket.blob(f"accounts/{account}/products.json")
            if not source_blob.exists():
                return None
            
            # Create backup with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            backup_path = f"accounts/{account}/backup/products-{timestamp}.json"
            backup_blob = self.bucket.blob(backup_path)
            
            # Copy current catalog to backup
            backup_blob.upload_from_string(
                source_blob.download_as_text(),
                content_type="application/json"
            )
            
            return backup_path
        except Exception as e:
            logger.error(f"Error creating product catalog backup for {account}: {e}")
            return None

class LocalAccountStorageProvider(AccountStorageProvider):
    """Local storage provider with account configuration support (for development)"""
    
    def __init__(self, base_dir: str = "storage"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    async def get_account_config(self, account: str) -> Optional[Dict[str, Any]]:
        """Get account config from local storage"""
        try:
            filepath = os.path.join(self.base_dir, "accounts", account, "account.json")
            if not os.path.exists(filepath):
                logger.info(f"No account config found for {account}")
                return None
            
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading account config for {account}: {e}")
            return None
    
    async def save_account_config(self, account: str, config: Dict[str, Any]) -> bool:
        """Save account config with backup to local storage"""
        try:
            # Create backup first
            backup_path = await self.backup_account_config(account)
            if backup_path:
                logger.info(f"Created backup for {account}: {backup_path}")
            
            # Ensure directory exists
            account_dir = os.path.join(self.base_dir, "accounts", account)
            os.makedirs(account_dir, exist_ok=True)
            
            # Save new config
            filepath = os.path.join(account_dir, "account.json")
            config['updatedAt'] = datetime.now().isoformat() + "Z"
            
            with open(filepath, "w") as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Saved account config for {account}")
            return True
        except Exception as e:
            logger.error(f"Error saving account config for {account}: {e}")
            return False
    
    async def backup_account_config(self, account: str) -> Optional[str]:
        """Create timestamped backup in local storage"""
        try:
            source_path = os.path.join(self.base_dir, "accounts", account, "account.json")
            if not os.path.exists(source_path):
                return None
            
            # Create backup directory
            backup_dir = os.path.join(self.base_dir, "accounts", account, "backup")
            os.makedirs(backup_dir, exist_ok=True)
            
            # Create backup with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            backup_path = os.path.join(backup_dir, f"account-{timestamp}.json")
            
            # Copy current config to backup
            import shutil
            shutil.copy2(source_path, backup_path)
            
            return backup_path
        except Exception as e:
            logger.error(f"Error creating backup for {account}: {e}")
            return None
    
    # Product Catalog Methods
    async def get_product_catalog(self, account: str) -> Optional[List[dict]]:
        """Get product catalog from local storage or fallback to data/{account}/products.json"""
        try:
            # First try local storage
            filepath = os.path.join(self.base_dir, "accounts", account, "products.json")
            
            # Fallback to data/{account}/products.json if not in storage
            if not os.path.exists(filepath):
                fallback_path = f"data/{account}/products.json"
                if os.path.exists(fallback_path):
                    logger.info(f"Using fallback product catalog for {account}: {fallback_path}")
                    with open(fallback_path, "r") as f:
                        return json.load(f)
                
                logger.info(f"No product catalog found for {account}")
                return None
            
            with open(filepath, "r") as f:
                products = json.load(f)
                logger.info(f"Loaded product catalog for {account}: {len(products)} products")
                return products
        except Exception as e:
            logger.error(f"Error loading product catalog for {account}: {e}")
            return None
    
    async def save_product_catalog(self, account: str, products: List[dict]) -> bool:
        """Save product catalog to local storage"""
        try:
            # Create backup first
            backup_path = await self._backup_product_catalog(account)
            if backup_path:
                logger.info(f"Created product catalog backup for {account}: {backup_path}")
            
            # Ensure directory exists
            account_dir = os.path.join(self.base_dir, "accounts", account)
            os.makedirs(account_dir, exist_ok=True)
            
            # Save catalog
            filepath = os.path.join(account_dir, "products.json")
            json_content = json.dumps(products, indent=2)
            catalog_size = len(json_content.encode('utf-8'))
            
            with open(filepath, "w") as f:
                f.write(json_content)
            
            # Save metadata
            metadata_path = os.path.join(account_dir, "products.metadata.json")
            metadata = {
                "compressed": False,
                "size": catalog_size,
                "product_count": len(products),
                "last_updated": datetime.now().isoformat() + "Z"
            }
            
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved product catalog for {account}: {len(products)} products, {catalog_size} bytes")
            return True
        except Exception as e:
            logger.error(f"Error saving product catalog for {account}: {e}")
            return False
    
    async def get_product_catalog_metadata(self, account: str) -> Optional[Dict[str, Any]]:
        """Get product catalog metadata from local storage"""
        try:
            # Check for metadata file
            metadata_path = os.path.join(self.base_dir, "accounts", account, "products.metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    return json.load(f)
            
            # Generate metadata from catalog file
            catalog_path = os.path.join(self.base_dir, "accounts", account, "products.json")
            
            # Try fallback path if not in storage
            if not os.path.exists(catalog_path):
                catalog_path = f"data/{account}/products.json"
            
            if not os.path.exists(catalog_path):
                return None
            
            stat = os.stat(catalog_path)
            
            # Try to get product count
            try:
                with open(catalog_path, "r") as f:
                    products = json.load(f)
                    product_count = len(products)
            except Exception:
                product_count = None
            
            return {
                "compressed": False,
                "size": stat.st_size,
                "product_count": product_count,
                "last_updated": datetime.fromtimestamp(stat.st_mtime).isoformat() + "Z"
            }
        except Exception as e:
            logger.error(f"Error getting product catalog metadata for {account}: {e}")
            return None
    
    async def _backup_product_catalog(self, account: str) -> Optional[str]:
        """Create timestamped backup of product catalog in local storage"""
        try:
            source_path = os.path.join(self.base_dir, "accounts", account, "products.json")
            if not os.path.exists(source_path):
                return None
            
            # Create backup directory
            backup_dir = os.path.join(self.base_dir, "accounts", account, "backup")
            os.makedirs(backup_dir, exist_ok=True)
            
            # Create backup with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            backup_path = os.path.join(backup_dir, f"products-{timestamp}.json")
            
            # Copy current catalog to backup
            import shutil
            shutil.copy2(source_path, backup_path)
            
            return backup_path
        except Exception as e:
            logger.error(f"Error creating product catalog backup for {account}: {e}")
            return None

def get_account_storage_provider() -> AccountStorageProvider:
    """
    Get account storage provider based on environment variables.
    
    Environment Variables:
        STORAGE_PROVIDER: "gcp" or "local" (default: "local")
        ENV_TYPE: "production" or "development" (default: "development")
        ACCOUNT_STORAGE_BUCKET: Override bucket name (optional)
        ACCOUNT_STORAGE_DIR: Local storage directory (default: "local/account_storage")
    
    GCP Bucket Selection:
        Production: liddy-account-documents
        Development: liddy-account-documents-dev
    
    Returns:
        Configured storage provider instance
    """
    provider_type = os.environ.get("STORAGE_PROVIDER", "local")
    
    if provider_type.lower() == "gcp":
        # Determine bucket based on environment
        env_type = os.environ.get("ENV_TYPE", "development")
        
        if env_type.lower() == "production":
            default_bucket = "liddy-account-documents"
        else:
            default_bucket = "liddy-account-documents-dev" 
        
        bucket_name = os.environ.get("ACCOUNT_STORAGE_BUCKET", default_bucket)
        
        logger.info(f"Using GCP account storage: {bucket_name} (env_type={env_type})")
        return GCPAccountStorageProvider(bucket_name)
    else:
        # Local storage for development
        storage_dir = os.environ.get("ACCOUNT_STORAGE_DIR", "local/account_storage")
        logger.info(f"Using local account storage: {storage_dir}")
        return LocalAccountStorageProvider(base_dir=storage_dir)