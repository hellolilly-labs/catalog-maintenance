"""
Separate Index Ingestion for Hybrid Search

This module handles ingestion into separate dense and sparse indexes,
following Pinecone's best practices for maximum flexibility.
"""

import os
import json
import logging
import hashlib
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from pinecone import Pinecone, ServerlessSpec
import numpy as np

from .universal_product_processor import UniversalProductProcessor
from .sparse_embeddings import SparseEmbeddingGenerator
from ..catalog.enhanced_descriptor_generator import EnhancedDescriptorGenerator

logger = logging.getLogger(__name__)


class SeparateIndexIngestion:
    """
    Handles product ingestion into separate dense and sparse indexes.
    
    Key features:
    - Separate indexes for dense and sparse embeddings
    - Consistent ID linkage between indexes
    - Change detection and incremental updates
    - Dynamic metadata schema generation
    """
    
    def __init__(
        self,
        brand_domain: str,
        dense_index_name: str,
        sparse_index_name: str,
        api_key: Optional[str] = None
    ):
        """
        Initialize ingestion system.
        
        Args:
            brand_domain: Brand domain (e.g., "specialized.com")
            dense_index_name: Name for dense embedding index
            sparse_index_name: Name for sparse embedding index
            api_key: Pinecone API key
        """
        self.brand_domain = brand_domain
        self.dense_index_name = dense_index_name
        self.sparse_index_name = sparse_index_name
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=api_key or os.getenv("PINECONE_API_KEY"))
        
        # Initialize processors
        self.product_processor = UniversalProductProcessor(brand_domain)
        self.sparse_generator = SparseEmbeddingGenerator(brand_domain)
        self.descriptor_generator = EnhancedDescriptorGenerator(brand_domain)
        
        # State management
        self.state_file = Path(f"data/sync_state/{brand_domain}_separate_indexes.json")
        self.sync_state = self._load_sync_state()
        
        logger.info(f"📦 Initialized Separate Index Ingestion for {brand_domain}")
    
    async def create_indexes(self, dimension: int = 2048, metric: str = "cosine"):
        """
        Create separate dense and sparse indexes if they don't exist.
        
        Args:
            dimension: Dimension for dense embeddings
            metric: Distance metric for dense index
        """
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        # Create dense index
        if self.dense_index_name not in existing_indexes:
            logger.info(f"Creating dense index: {self.dense_index_name}")
            self.pc.create_index(
                name=self.dense_index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            logger.info(f"✅ Created dense index: {self.dense_index_name}")
        
        # Create sparse index
        if self.sparse_index_name not in existing_indexes:
            logger.info(f"Creating sparse index: {self.sparse_index_name}")
            self.pc.create_index(
                name=self.sparse_index_name,
                dimension=50000,  # Large dimension for sparse vectors
                metric="dotproduct",  # Best for sparse vectors
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                ),
                # Enable sparse values
                sparse_values=True
            )
            logger.info(f"✅ Created sparse index: {self.sparse_index_name}")
    
    async def ingest_catalog(
        self,
        catalog_path: str,
        namespace: str = "products",
        batch_size: int = 100,
        force_update: bool = False
    ) -> Dict[str, Any]:
        """
        Ingest product catalog into separate indexes.
        
        Args:
            catalog_path: Path to product catalog
            namespace: Namespace for products
            batch_size: Batch size for upserts
            force_update: Force update all products
            
        Returns:
            Ingestion statistics
        """
        # Load and process catalog
        logger.info(f"Loading catalog from {catalog_path}")
        with open(catalog_path, 'r') as f:
            catalog_data = json.load(f)
        
        if isinstance(catalog_data, dict):
            catalog_data = catalog_data.get('products', [catalog_data])
        
        # Process products
        processed_products = self.product_processor.process_catalog(catalog_data)
        
        # Generate enhanced descriptors and extract filters
        enhanced_products, filter_labels = self.descriptor_generator.process_catalog(
            processed_products,
            descriptor_style="voice_optimized"
        )
        
        # Build vocabulary if not already loaded
        if not self.sparse_generator.vocabulary:
            logger.info("Building sparse embedding vocabulary...")
            self.sparse_generator.build_vocabulary(enhanced_products)
        
        # Save filter labels for the brand
        self._save_filter_labels(filter_labels)
        
        # Detect changes
        changes = self._detect_changes(enhanced_products, force_update)
        
        logger.info(f"📊 Changes detected: {len(changes['add'])} new, "
                   f"{len(changes['update'])} updated, {len(changes['delete'])} deleted")
        
        # Get indexes
        dense_index = self.pc.Index(self.dense_index_name)
        sparse_index = self.pc.Index(self.sparse_index_name)
        
        # Process additions and updates
        products_to_upsert = []
        for product_id in changes['add'] + changes['update']:
            product = next(p for p in enhanced_products if p['id'] == product_id)
            products_to_upsert.append(product)
        
        # Prepare batches
        dense_vectors = []
        sparse_vectors = []
        
        for i in range(0, len(products_to_upsert), batch_size):
            batch = products_to_upsert[i:i + batch_size]
            
            # Prepare dense vectors
            for product in batch:
                dense_data = self._prepare_dense_vector(product)
                dense_vectors.append({
                    'id': dense_data['id'],
                    'text': dense_data['text'],
                    'metadata': dense_data['metadata']
                })
            
            # Prepare sparse vectors
            for product in batch:
                sparse_data = self._prepare_sparse_vector(product)
                if sparse_data['sparse_values'].get('indices'):
                    sparse_vectors.append({
                        'id': sparse_data['id'],
                        'sparse_values': sparse_data['sparse_values'],
                        'metadata': sparse_data['metadata']
                    })
        
        # Upsert to dense index
        if dense_vectors:
            logger.info(f"Upserting {len(dense_vectors)} vectors to dense index")
            for i in range(0, len(dense_vectors), batch_size):
                batch = dense_vectors[i:i + batch_size]
                dense_index.upsert(
                    vectors=batch,
                    namespace=namespace
                )
                logger.info(f"  Upserted batch {i//batch_size + 1}/{(len(dense_vectors)-1)//batch_size + 1}")
        
        # Upsert to sparse index
        if sparse_vectors:
            logger.info(f"Upserting {len(sparse_vectors)} vectors to sparse index")
            for i in range(0, len(sparse_vectors), batch_size):
                batch = sparse_vectors[i:i + batch_size]
                sparse_index.upsert(
                    vectors=batch,
                    namespace=namespace
                )
                logger.info(f"  Upserted batch {i//batch_size + 1}/{(len(sparse_vectors)-1)//batch_size + 1}")
        
        # Handle deletions
        if changes['delete']:
            logger.info(f"Deleting {len(changes['delete'])} products")
            dense_index.delete(
                ids=changes['delete'],
                namespace=namespace
            )
            sparse_index.delete(
                ids=changes['delete'],
                namespace=namespace
            )
        
        # Update sync state
        self._update_sync_state(enhanced_products)
        
        # Get index statistics
        dense_stats = dense_index.describe_index_stats()
        sparse_stats = sparse_index.describe_index_stats()
        
        return {
            'products_processed': len(processed_products),
            'added': len(changes['add']),
            'updated': len(changes['update']),
            'deleted': len(changes['delete']),
            'dense_vectors': dense_stats.total_vector_count,
            'sparse_vectors': sparse_stats.total_vector_count,
            'filter_labels': len(filter_labels) - 1  # Exclude metadata
        }
    
    def _prepare_dense_vector(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare product for dense index."""
        
        # Combine descriptors for embedding
        embedding_text = self._build_embedding_text(product)
        
        # Build metadata - same structure in both indexes
        metadata = self._build_metadata(product)
        
        return {
            'id': product['id'],
            'text': embedding_text,  # For server-side embedding
            'metadata': metadata
        }
    
    def _prepare_sparse_vector(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare product for sparse index."""
        
        # Generate sparse embeddings
        sparse_data = self.sparse_generator.generate_sparse_embedding(
            product,
            {'key_features': product.get('search_keywords', [])}
        )
        
        # Use same metadata structure
        metadata = self._build_metadata(product)
        
        return {
            'id': product['id'],
            'sparse_values': sparse_data,
            'metadata': metadata
        }
    
    def _build_embedding_text(self, product: Dict[str, Any]) -> str:
        """Build text for dense embedding."""
        parts = []
        
        # Enhanced descriptor (primary)
        if product.get('enhanced_description'):
            parts.append(product['enhanced_description'])
        
        # Voice summary
        if product.get('voice_summary'):
            parts.append(product['voice_summary'])
        
        # Original descriptor
        if product.get('enhanced_descriptor'):
            parts.append(product['enhanced_descriptor'])
        
        # Key selling points
        if product.get('key_selling_points'):
            parts.append(' '.join(product['key_selling_points']))
        
        return ' '.join(parts)
    
    def _build_metadata(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build metadata dynamically based on product data.
        
        This ensures consistent metadata between dense and sparse indexes.
        """
        # Start with universal fields
        metadata = {
            'id': product['id'],
            'name': product['universal_fields'].get('name', ''),
            'price': product['universal_fields'].get('price', 0),
            'category': product['universal_fields'].get('category', ['general'])[0],
            'brand': product['universal_fields'].get('brand', self.brand_domain.split('.')[0])
        }
        
        # Add filter metadata dynamically
        if product.get('filter_metadata'):
            for key, value in product['filter_metadata'].items():
                # Skip if already in metadata
                if key not in metadata:
                    metadata[key] = value
        
        # Add content fields
        content_fields = [
            'description',
            'enhanced_descriptor', 
            'voice_summary',
            'key_selling_points',
            'search_keywords'
        ]
        
        for field in content_fields:
            if field in product['universal_fields']:
                value = product['universal_fields'][field]
            elif field in product:
                value = product[field]
            else:
                continue
            
            # Convert lists to JSON strings for storage
            if isinstance(value, list):
                metadata[field] = json.dumps(value)
            else:
                metadata[field] = value
        
        # Add image URL if available
        if 'images' in product['universal_fields']:
            images = product['universal_fields']['images']
            if isinstance(images, list) and images:
                metadata['image_url'] = images[0]
        
        # Add metadata fields
        metadata['content_type'] = 'product'
        metadata['last_updated'] = datetime.now().isoformat()
        metadata['text'] = self._build_embedding_text(product)  # For reranking
        
        return metadata
    
    def _detect_changes(
        self,
        products: List[Dict[str, Any]],
        force_update: bool
    ) -> Dict[str, List[str]]:
        """Detect which products need to be added, updated, or deleted."""
        
        if force_update:
            return {
                'add': [p['id'] for p in products],
                'update': [],
                'delete': []
            }
        
        current_products = {p['id']: self._get_product_hash(p) for p in products}
        previous_products = self.sync_state.get('product_hashes', {})
        
        current_ids = set(current_products.keys())
        previous_ids = set(previous_products.keys())
        
        # Detect changes
        add_ids = current_ids - previous_ids
        delete_ids = previous_ids - current_ids
        
        update_ids = []
        for product_id in current_ids & previous_ids:
            if current_products[product_id] != previous_products[product_id]:
                update_ids.append(product_id)
        
        return {
            'add': list(add_ids),
            'update': update_ids,
            'delete': list(delete_ids)
        }
    
    def _get_product_hash(self, product: Dict[str, Any]) -> str:
        """Generate hash for change detection."""
        
        # Create consistent representation
        content = json.dumps({
            'universal_fields': product.get('universal_fields', {}),
            'filter_metadata': product.get('filter_metadata', {}),
            'enhanced_descriptor': product.get('enhanced_descriptor', ''),
            'key_selling_points': product.get('key_selling_points', [])
        }, sort_keys=True)
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def _save_filter_labels(self, filter_labels: Dict[str, Any]):
        """Save filter labels for the brand."""
        
        filter_file = Path(f"accounts/{self.brand_domain}/filter_dictionary.json")
        filter_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filter_file, 'w') as f:
            json.dump(filter_labels, f, indent=2)
        
        logger.info(f"💾 Saved filter labels to {filter_file}")
    
    def _load_sync_state(self) -> Dict[str, Any]:
        """Load sync state from file."""
        
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _update_sync_state(self, products: List[Dict[str, Any]]):
        """Update sync state with current product hashes."""
        
        self.sync_state['product_hashes'] = {
            p['id']: self._get_product_hash(p) for p in products
        }
        self.sync_state['last_sync'] = datetime.now().isoformat()
        
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.sync_state, f, indent=2)
        
        logger.info(f"💾 Updated sync state for {len(products)} products")