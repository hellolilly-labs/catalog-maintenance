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
    
    @abstractmethod
    async def get_accounts(self) -> List[str]:
        """Get all accounts"""
        pass
    
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
    
    @abstractmethod
    async def get_research_data(self, account: str, research_type: str) -> Optional[str]:
        """Get research data for an account"""
        pass
    
    @abstractmethod
    async def get_research_data_metadata(self, account: str, research_type: str) -> Optional[Dict[str, Any]]:
        """Get research data metadata for an account"""
        pass
    
    
    # File System Methods for Workflow State Management
    @abstractmethod
    async def list_files(self, account: str, directory: str) -> List[str]:
        """List files in a directory for an account"""
        pass
    
    @abstractmethod
    async def file_exists(self, account: str, file_path: str) -> bool:
        """Check if a file exists for an account"""
        pass
    
    @abstractmethod
    async def read_file(self, account: str, file_path: str) -> Optional[str]:
        """Read file content as string for an account"""
        pass
    
    @abstractmethod
    async def write_file(self, account: str, file_path: str, content: str, content_type: str = "text/plain") -> bool:
        """Write file content as string for an account"""
        pass
    
    @abstractmethod
    async def get_file_metadata(self, account: str, file_path: str) -> Optional[Dict[str, Any]]:
        """Get file metadata (size, modified_time, etc.)"""
        pass
    
    # Utility Methods
    def _should_compress_catalog(self, catalog_size: int) -> bool:
        """Determine if catalog should be compressed (>500KB)"""
        return catalog_size > 500 * 1024 and False
    
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
        from ..auth_utils import setup_google_auth, get_google_credentials
        
        # Set up authentication
        auth_success = setup_google_auth()
        if not auth_success:
            logger.warning("Google Cloud authentication may not be properly configured")
        
        # Get credentials
        credentials = get_google_credentials()
        
        # Create client with explicit credentials if available
        if credentials:
            self.client = storage.Client(credentials=credentials)
        else:
            # Try to create client anyway - it might work with default auth
            try:
                self.client = storage.Client()
            except Exception as e:
                logger.error(f"Failed to create GCS client: {e}")
                raise
        
        self.bucket = self.client.bucket(bucket_name)
    
    async def get_accounts(self) -> List[str]:
        """Get all accounts"""
        try:
            blobs = self.client.list_blobs(self.bucket, prefix="accounts/")
            account_names = [blob.name.split("/")[1] for blob in blobs]
            # deduplicate
            account_names = list(set(account_names))
            return account_names
        except Exception as e:
            logger.error(f"Error getting accounts: {e}")
            return []
    
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
            if "agents" in config and isinstance(config["agents"][0], str):
                raise ValueError(f"Invalid account config for {account}: agents must be a list of dicts")
            elif "agent" in config and isinstance(config["agent"], str):
                raise ValueError(f"Invalid account config for {account}: agent must be a dict")
            
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
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
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
    
    async def get_research_data(self, account: str, research_type: str) -> Optional[str]:
        """Get research data for an account"""
        try:
            blob = self.bucket.blob(f"accounts/{account}/research/{research_type}/research.md")
            if not blob.exists():
                return None
            return blob.download_as_text()
        except Exception as e:
            logger.error(f"Error getting research data for {account}: {e}")
            return None
    
    async def get_research_data_metadata(self, account: str, research_type: str) -> Optional[Dict[str, Any]]:
        """Get research data metadata for an account"""
        try:
            blob = self.bucket.blob(f"accounts/{account}/research/{research_type}/research_metadata.json")
            if not blob.exists():
                return None
            return json.loads(blob.download_as_text())
        except Exception as e:
            logger.error(f"Error getting research data metadata for {account}: {e}")
            return None

    # File System Methods for Workflow State Management
    async def list_files(self, account: str, directory: str) -> List[str]:
        """List files in a directory for an account"""
        try:
            prefix = f"accounts/{account}/{directory}/"
            blobs = self.client.list_blobs(self.bucket, prefix=prefix)
            
            files = []
            for blob in blobs:
                # Remove the prefix to get just the filename
                filename = blob.name[len(prefix):]
                # Skip subdirectories (contains /)
                if '/' not in filename and filename:
                    files.append(filename)
            
            return files
        except Exception as e:
            logger.error(f"Error listing files in {directory} for {account}: {e}")
            return []
    
    async def file_exists(self, account: str, file_path: str) -> bool:
        """Check if a file exists for an account"""
        try:
            blob = self.bucket.blob(f"accounts/{account}/{file_path}")
            return blob.exists()
        except Exception as e:
            logger.error(f"Error checking file existence {file_path} for {account}: {e}")
            return False
    
    async def read_file(self, account: str, file_path: str) -> Optional[str]:
        """Read file content as string for an account"""
        try:
            blob = self.bucket.blob(f"accounts/{account}/{file_path}")
            if not blob.exists():
                return None
            return blob.download_as_text()
        except Exception as e:
            logger.error(f"Error reading file {file_path} for {account}: {e}")
            return None
    
    async def write_file(self, account: str, file_path: str, content: str, content_type: str = "text/plain") -> bool:
        """Write file content as string for an account"""
        try:
            blob = self.bucket.blob(f"accounts/{account}/{file_path}")
            blob.upload_from_string(
                content,
                content_type=content_type
            )
            return True
        except Exception as e:
            logger.error(f"Error writing file {file_path} for {account}: {e}")
            return False
    
    async def get_file_metadata(self, account: str, file_path: str) -> Optional[Dict[str, Any]]:
        """Get file metadata (size, modified_time, etc.)"""
        try:
            blob = self.bucket.blob(f"accounts/{account}/{file_path}")
            if not blob.exists():
                return None
            
            blob.reload()  # Refresh metadata
            
            return {
                "size": blob.size,
                "modified_time": blob.updated.isoformat() if blob.updated else None,
                "created_time": blob.time_created.isoformat() if blob.time_created else None,
                "content_type": blob.content_type
            }
        except Exception as e:
            logger.error(f"Error getting file metadata {file_path} for {account}: {e}")
            return None

class LocalAccountStorageProvider(AccountStorageProvider):
    """Local storage provider with account configuration support (for development)"""
    
    def __init__(self, base_dir: str = "storage"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    async def get_accounts(self) -> List[str]:
        """Get all accounts"""
        try:
            return [d.rstrip('/') for d in os.listdir(os.path.join(self.base_dir, "accounts")) if d.endswith('/')]
        except Exception as e:
            logger.error(f"Error getting accounts: {e}")
            return []
    
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
        import gzip
        
        try:
            # First try local storage - check for both compressed and uncompressed files
            base_filepath = os.path.join(self.base_dir, "accounts", account, "products.json")
            gzip_filepath = base_filepath + ".gz"
            
            # Try compressed file first
            if os.path.exists(gzip_filepath):
                logger.info(f"Loading compressed product catalog for {account}: {gzip_filepath}")
                with gzip.open(gzip_filepath, 'rt', encoding='utf-8') as f:
                    products = json.load(f)
                    logger.info(f"Loaded compressed product catalog for {account}: {len(products)} products")
                    return products
            
            # Try uncompressed file
            if os.path.exists(base_filepath):
                logger.info(f"Loading product catalog for {account}: {base_filepath}")
                with open(base_filepath, "r") as f:
                    products = json.load(f)
                    logger.info(f"Loaded product catalog for {account}: {len(products)} products")
                    return products
            
            # Fallback to data/{account}/products.json if not in storage
            fallback_path = f"data/{account}/products.json"
            fallback_gzip_path = fallback_path + ".gz"
            
            # Try compressed fallback
            if os.path.exists(fallback_gzip_path):
                logger.info(f"Using compressed fallback product catalog for {account}: {fallback_gzip_path}")
                with gzip.open(fallback_gzip_path, 'rt', encoding='utf-8') as f:
                    return json.load(f)
            
            # Try uncompressed fallback
            if os.path.exists(fallback_path):
                logger.info(f"Using fallback product catalog for {account}: {fallback_path}")
                with open(fallback_path, "r") as f:
                    return json.load(f)
            
            logger.info(f"No product catalog found for {account}")
            return None
            
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
    
    async def get_research_data(self, account: str, research_type: str) -> Optional[str]:
        """Get research data for an account"""
        try:
            filepath = os.path.join(self.base_dir, "accounts", account, "research", research_type + "research.md")
            if not os.path.exists(filepath):
                return None
            with open(filepath, "r") as f:  
                return f.read()
        except Exception as e:
            logger.error(f"Error getting research data for {account}: {e}")
            return None
    
    async def get_research_data_metadata(self, account: str, research_type: str) -> Optional[Dict[str, Any]]:
        """Get research data metadata for an account"""
        try:
            filepath = os.path.join(self.base_dir, "accounts", account, "research", research_type + "research_metadata.json")
            if not os.path.exists(filepath):
                return None
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error getting research data metadata for {account}: {e}")
            return None
    

    # File System Methods for Workflow State Management
    async def list_files(self, account: str, directory: str) -> List[str]:
        """List files in a directory for an account"""
        try:
            dir_path = os.path.join(self.base_dir, "accounts", account, directory)
            if not os.path.exists(dir_path):
                return []
            
            files = []
            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)
                if os.path.isfile(item_path):
                    files.append(item)
            
            return files
        except Exception as e:
            logger.error(f"Error listing files in {directory} for {account}: {e}")
            return []
    
    async def file_exists(self, account: str, file_path: str) -> bool:
        """Check if a file exists for an account"""
        try:
            full_path = os.path.join(self.base_dir, "accounts", account, file_path)
            return os.path.exists(full_path)
        except Exception as e:
            logger.error(f"Error checking file existence {file_path} for {account}: {e}")
            return False
    
    async def read_file(self, account: str, file_path: str) -> Optional[str]:
        """Read file content as string for an account"""
        try:
            full_path = os.path.join(self.base_dir, "accounts", account, file_path)
            if not os.path.exists(full_path):
                return None
            
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path} for {account}: {e}")
            return None
    
    async def write_file(self, account: str, file_path: str, content: str, content_type: str = "text/plain") -> bool:
        """Write file content as string for an account"""
        try:
            full_path = os.path.join(self.base_dir, "accounts", account, file_path)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True
        except Exception as e:
            logger.error(f"Error writing file {file_path} for {account}: {e}")
            return False
    
    async def get_file_metadata(self, account: str, file_path: str) -> Optional[Dict[str, Any]]:
        """Get file metadata (size, modified_time, etc.)"""
        try:
            full_path = os.path.join(self.base_dir, "accounts", account, file_path)
            if not os.path.exists(full_path):
                return None
            
            stat = os.stat(full_path)
            
            return {
                "size": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "content_type": "text/plain"  # Could be enhanced to detect MIME type
            }
        except Exception as e:
            logger.error(f"Error getting file metadata {file_path} for {account}: {e}")
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

class BrandDataManager:
    """
    Brand data management utilities with restart/cleanup functionality.
    
    ⚠️  WARNING: These operations are IRREVERSIBLE and will permanently delete data!
    """
    
    def __init__(self, storage_provider: AccountStorageProvider):
        self.storage = storage_provider
    
    async def restart_brand(self, brand_domain: str, force: bool = False) -> bool:
        """
        🚨 DANGER: Completely removes ALL brand data and starts fresh.
        
        This will permanently delete:
        - Account configuration
        - Product catalogs
        - All backups
        - Any cached data
        
        Args:
            brand_domain: Brand domain (e.g., "specialized.com")
            force: Skip safety checks (USE WITH EXTREME CAUTION)
            
        Returns:
            True if successful, False if cancelled or failed
            
        ⚠️  THIS OPERATION IS IRREVERSIBLE ⚠️
        """
        if not force:
            # Safety checks and user confirmation
            print(f"\n🚨 DANGER: BRAND DATA RESTART FOR '{brand_domain}' 🚨")
            print("=" * 70)
            print("This operation will PERMANENTLY DELETE ALL DATA for this brand:")
            print(f"  • Account configuration (accounts/{brand_domain}/account.json)")
            print(f"  • Product catalog (accounts/{brand_domain}/products.json)")  
            print(f"  • All backups (accounts/{brand_domain}/backup/)")
            print(f"  • Cached vertical detection data")
            print(f"  • Any generated descriptors/sizing data")
            print("\n⚠️  THIS CANNOT BE UNDONE! ⚠️")
            print("\nReasons you might want to do this:")
            print("  • Testing new implementation")
            print("  • Brand has completely changed their product line")
            print("  • Starting fresh after major data corruption")
            
            # Check if data exists first
            account_config = await self.storage.get_account_config(brand_domain)
            product_catalog = await self.storage.get_product_catalog(brand_domain)
            
            if not account_config and not product_catalog:
                print(f"\n✅ No existing data found for '{brand_domain}' - nothing to delete.")
                return True
            
            if account_config:
                print(f"\n📊 Current account config: {len(account_config)} settings")
                
            if product_catalog:
                print(f"📦 Current product catalog: {len(product_catalog)} products")
            
            # Require explicit confirmation
            print(f"\nTo confirm deletion, type exactly: DELETE {brand_domain}")
            confirmation = input("Confirmation: ").strip()
            
            if confirmation != f"DELETE {brand_domain}":
                print("❌ Confirmation failed. Brand restart cancelled.")
                return False
            
            print("\n⚠️  LAST CHANCE: Are you absolutely sure? (yes/no)")
            final_confirm = input("Final confirmation: ").strip().lower()
            
            if final_confirm != "yes":
                print("❌ Brand restart cancelled.")
                return False
        
        # Proceed with deletion
        print(f"\n🗑️  Starting brand data deletion for '{brand_domain}'...")
        
        deleted_items = []
        errors = []
        
        try:
            # Delete account configuration
            if await self._delete_account_config(brand_domain):
                deleted_items.append("Account configuration")
            else:
                errors.append("Failed to delete account configuration")
                
            # Delete product catalog  
            if await self._delete_product_catalog(brand_domain):
                deleted_items.append("Product catalog")
            else:
                errors.append("Failed to delete product catalog")
                
            # Delete all backups
            backup_count = await self._delete_all_backups(brand_domain)
            if backup_count > 0:
                deleted_items.append(f"{backup_count} backup files")
            
            # Clear any cached data (brand vertical cache, etc.)
            await self._clear_brand_cache(brand_domain)
            deleted_items.append("Cached data")
            
            # Report results
            if deleted_items:
                print(f"\n✅ Successfully deleted:")
                for item in deleted_items:
                    print(f"   • {item}")
            
            if errors:
                print(f"\n❌ Errors encountered:")
                for error in errors:
                    print(f"   • {error}")
                return False
            else:
                print(f"\n🎉 Brand restart complete for '{brand_domain}'")
                print("   You can now start fresh with this brand.")
                return True
                
        except Exception as e:
            logger.error(f"Error during brand restart for {brand_domain}: {e}")
            print(f"\n❌ Unexpected error during restart: {e}")
            return False
    
    async def _delete_account_config(self, brand_domain: str) -> bool:
        """Delete account configuration"""
        try:
            if isinstance(self.storage, GCPAccountStorageProvider):
                # Delete from GCP
                blob = self.storage.bucket.blob(f"accounts/{brand_domain}/account.json")
                if blob.exists():
                    blob.delete()
                    return True
                    
            elif isinstance(self.storage, LocalAccountStorageProvider):
                # Delete from local storage
                filepath = os.path.join(self.storage.base_dir, "accounts", brand_domain, "account.json")
                if os.path.exists(filepath):
                    os.remove(filepath)
                    return True
                    
            return True  # Nothing to delete
        except Exception as e:
            logger.error(f"Error deleting account config for {brand_domain}: {e}")
            return False
    
    async def _delete_product_catalog(self, brand_domain: str) -> bool:
        """Delete product catalog and metadata"""
        try:
            if isinstance(self.storage, GCPAccountStorageProvider):
                # Delete from GCP
                catalog_blob = self.storage.bucket.blob(f"accounts/{brand_domain}/products.json")
                metadata_blob = self.storage.bucket.blob(f"accounts/{brand_domain}/products.metadata.json")
                
                if catalog_blob.exists():
                    catalog_blob.delete()
                if metadata_blob.exists():
                    metadata_blob.delete()
                    
            elif isinstance(self.storage, LocalAccountStorageProvider):
                # Delete from local storage
                catalog_path = os.path.join(self.storage.base_dir, "accounts", brand_domain, "products.json")
                metadata_path = os.path.join(self.storage.base_dir, "accounts", brand_domain, "products.metadata.json")
                
                if os.path.exists(catalog_path):
                    os.remove(catalog_path)
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                
                # Also check fallback location
                fallback_path = f"data/{brand_domain}/products.json"
                if os.path.exists(fallback_path):
                    os.remove(fallback_path)
                    
            return True
        except Exception as e:
            logger.error(f"Error deleting product catalog for {brand_domain}: {e}")
            return False
    
    async def _delete_all_backups(self, brand_domain: str) -> int:
        """Delete all backup files and return count of deleted files"""
        deleted_count = 0
        
        try:
            if isinstance(self.storage, GCPAccountStorageProvider):
                # Delete from GCP
                backup_prefix = f"accounts/{brand_domain}/backup/"
                blobs = self.storage.bucket.list_blobs(prefix=backup_prefix)
                
                for blob in blobs:
                    blob.delete()
                    deleted_count += 1
                    
            elif isinstance(self.storage, LocalAccountStorageProvider):
                # Delete from local storage
                backup_dir = os.path.join(self.storage.base_dir, "accounts", brand_domain, "backup")
                
                if os.path.exists(backup_dir):
                    import shutil
                    for filename in os.listdir(backup_dir):
                        filepath = os.path.join(backup_dir, filename)
                        if os.path.isfile(filepath):
                            os.remove(filepath)
                            deleted_count += 1
                    
                    # Remove empty backup directory
                    os.rmdir(backup_dir)
                    
        except Exception as e:
            logger.error(f"Error deleting backups for {brand_domain}: {e}")
            
        return deleted_count
    
    async def _clear_brand_cache(self, brand_domain: str) -> None:
        """Clear any cached data for the brand"""
        try:
            # Clear brand vertical cache if it exists
            # This would integrate with the descriptor.py caching mechanism
            # Legacy descriptor support - needs migration
            # from liddy_intelligence.descriptor import DescriptorGenerator
            pass
            generator = DescriptorGenerator()
            
            # Clear the brand from cache if present
            if hasattr(generator, '_brand_vertical_cache'):
                generator._brand_vertical_cache.pop(brand_domain, None)
                logger.info(f"Cleared brand vertical cache for {brand_domain}")
                
        except Exception as e:
            logger.debug(f"Note: Could not clear cache for {brand_domain}: {e}")
    
    async def list_brand_data(self, brand_domain: str) -> Dict[str, Any]:
        """
        List all data associated with a brand (for inspection before deletion).
        
        Returns:
            Dictionary with brand data summary
        """
        try:
            summary = {
                "brand_domain": brand_domain,
                "account_config": None,
                "product_catalog": None,
                "backups": [],
                "total_size_estimate": 0
            }
            
            # Get account config
            account_config = await self.storage.get_account_config(brand_domain)
            if account_config:
                summary["account_config"] = {
                    "exists": True,
                    "settings_count": len(account_config),
                    "last_updated": account_config.get("updatedAt", "unknown")
                }
            
            # Get product catalog metadata
            catalog_metadata = await self.storage.get_product_catalog_metadata(brand_domain)
            if catalog_metadata:
                summary["product_catalog"] = catalog_metadata
                summary["total_size_estimate"] += catalog_metadata.get("size", 0)
            
            # List backups (implementation depends on storage type)
            if isinstance(self.storage, LocalAccountStorageProvider):
                backup_dir = os.path.join(self.storage.base_dir, "accounts", brand_domain, "backup")
                if os.path.exists(backup_dir):
                    for filename in os.listdir(backup_dir):
                        filepath = os.path.join(backup_dir, filename)
                        if os.path.isfile(filepath):
                            stat = os.stat(filepath)
                            summary["backups"].append({
                                "filename": filename,
                                "size": stat.st_size,
                                "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
                            })
                            summary["total_size_estimate"] += stat.st_size
            
            return summary
            
        except Exception as e:
            logger.error(f"Error listing brand data for {brand_domain}: {e}")
            return {"error": str(e)}

# Factory function
def get_brand_data_manager() -> BrandDataManager:
    """Get brand data manager with configured storage provider"""
    storage = get_account_storage_provider()
    return BrandDataManager(storage)