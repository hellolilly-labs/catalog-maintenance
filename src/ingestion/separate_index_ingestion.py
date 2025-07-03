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

from liddy_intelligence.ingestion.universal_product_processor import UniversalProductProcessor
# Removed custom sparse embeddings - using Pinecone's sparse model instead
from liddy_intelligence.ingestion.stt_vocabulary_extractor import STTVocabularyExtractor
from liddy_intelligence.catalog.unified_descriptor_generator import UnifiedDescriptorGenerator
from liddy.storage import get_account_storage_provider
from liddy.models.product import Product

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
        
        # Use unified descriptor generator (has research integration built-in)
        self.descriptor_generator = UnifiedDescriptorGenerator(brand_domain)
        
        # Initialize STT vocabulary extractor
        self.stt_extractor = STTVocabularyExtractor(brand_domain)
        
        # Storage provider for consistent data management
        self.storage = get_account_storage_provider()
        
        # State management using storage provider
        self.sync_state_path = "sync_state/separate_indexes.json"
        self.sync_state = None  # Will be loaded on first use
        
        logger.info(f"📦 Initialized Separate Index Ingestion for {brand_domain}")
    
    async def create_indexes(self, dimension: int = 1024, metric: str = "cosine"):
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
            self.pc.create_index_for_model(
                name=self.dense_index_name,
                cloud="gcp",
                region="us-central1",
                # Configure server-side embeddings
                embed={
                    "model": "llama-text-embed-v2",
                    "field_map": {"text": "text"},
                    "dimension": dimension
                }
            )
            logger.info(f"✅ Created dense index: {self.dense_index_name}")
        
        # Create sparse index
        if self.sparse_index_name not in existing_indexes:
            logger.info(f"Creating sparse index: {self.sparse_index_name}")
            self.pc.create_index_for_model(
                name=self.sparse_index_name,
                cloud="gcp",
                region="us-central1",
                embed={
                    "model": "pinecone-sparse-english-v0",
                    "field_map": {"text": "text"}
                }
            )
            logger.info(f"✅ Created sparse index: {self.sparse_index_name}")
    
    async def ingest_catalog(
        self,
        namespace: str = "products",
        batch_size: int = 100,
        force_update: bool = False
    ) -> Dict[str, Any]:
        """
        Ingest product catalog into separate indexes.
        
        Args:
            namespace: Namespace for products
            batch_size: Batch size for upserts
            force_update: Force update all products
            
        Returns:
            Ingestion statistics
        """
        logger.info(f"Processing catalog with UnifiedDescriptorGenerator")
        
        # Generate enhanced descriptors and extract filters using UnifiedDescriptorGenerator
        enhanced_products, filter_labels = await self.descriptor_generator.process_catalog(
            force_regenerate=False
        )
        
        logger.info(f"📝 Generated descriptors for {len(enhanced_products)} products")
        
        # No need to build vocabulary - Pinecone's sparse model handles it
        
        # Extract and save STT vocabulary
        logger.info("🎤 Extracting STT vocabulary for speech recognition optimization...")
        self.stt_extractor.extract_from_catalog(enhanced_products)
        await self.stt_extractor.save_vocabulary()
        
        # Log STT vocabulary stats
        stats = self.stt_extractor.get_stats()
        logger.info(f"   STT Vocabulary: {stats['term_count']} terms, "
                   f"{stats['character_count']} chars ({stats['coverage_percentage']:.1f}% of limit)")
        logger.info(f"   Top terms: {', '.join(stats['top_terms'][:10])}")
        
        # Save filter labels for the brand
        await self._save_filter_labels(filter_labels)
        
        # Detect changes
        changes = await self._detect_changes(enhanced_products, force_update)
        
        logger.info(f"📊 Changes detected: {len(changes['add'])} new, "
                   f"{len(changes['update'])} updated, {len(changes['delete'])} deleted")
        
        # Ensure indexes exist
        await self.create_indexes()
        
        # Get indexes
        dense_index = self.pc.Index(self.dense_index_name)
        sparse_index = self.pc.Index(self.sparse_index_name)
        
        # Process additions and updates
        products_to_upsert = []
        for product_id in changes['add'] + changes['update']:
            product = next(p for p in enhanced_products if getattr(p, 'id', '') == product_id)
            products_to_upsert.append(product)
        
        # Prepare batches
        dense_vectors = []
        sparse_vectors = []
        
        for i in range(0, len(products_to_upsert), batch_size):
            batch = products_to_upsert[i:i + batch_size]
            
            # Prepare dense vectors
            for product in batch:
                dense_data = self._prepare_dense_vector(product)
                dense_vectors.append(dense_data)
            
            # Prepare sparse vectors
            for product in batch:
                sparse_data = self._prepare_sparse_vector(product)
                sparse_vectors.append(sparse_data)
        
        # Upsert to dense index
        if dense_vectors:
            logger.info(f"Upserting {len(dense_vectors)} vectors to dense index")
            for i in range(0, len(dense_vectors), batch_size):
                batch = dense_vectors[i:i + batch_size]
                dense_index.upsert_records(
                    namespace=namespace,
                    records=batch
                )
                logger.info(f"  Upserted batch {i//batch_size + 1}/{(len(dense_vectors)-1)//batch_size + 1}")
        
        # Upsert to sparse index
        if sparse_vectors:
            logger.info(f"Upserting {len(sparse_vectors)} vectors to sparse index")
            try:
                for i in range(0, len(sparse_vectors), batch_size):
                    batch = sparse_vectors[i:i + batch_size]
                    sparse_index.upsert_records(
                        namespace=namespace,
                        records=batch,
                    )
                    logger.info(f"  Upserted batch {i//batch_size + 1}/{(len(sparse_vectors)-1)//batch_size + 1}")
            except Exception as e:
                logger.error(f"Error upserting sparse vectors: {e}")
                logger.error(f"Batch: {batch}")
        
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
        await self._update_sync_state(enhanced_products)
        
        # Get index statistics
        dense_stats = dense_index.describe_index_stats()
        sparse_stats = sparse_index.describe_index_stats()
        
        return {
            'products_processed': len(products_to_upsert),
            'added': len(changes['add']),
            'updated': len(changes['update']),
            'deleted': len(changes['delete']),
            'dense_vectors': dense_stats.total_vector_count,
            'sparse_vectors': sparse_stats.total_vector_count,
            'filter_labels': len(filter_labels) - 1  # Exclude metadata
        }
    
    def _prepare_dense_vector(self, product: Product) -> Dict[str, Any]:
        """Prepare product for dense index with server-side embeddings."""
        
        # Get embedding text
        embedding_text = self._build_embedding_text(product)
        
        # Build metadata - same structure in both indexes
        metadata = self._build_metadata(product)
        
        # For upsert_records with server-side embeddings,
        # metadata must be stored as a JSON string in the metadata field
        record = {
            '_id': f"{product.id}",
            'text': embedding_text,  # This will be embedded server-side
            'name': product.name,
            # 'descriptor': embedding_text,  # This will be embedded server-side
            'metadata': json.dumps(metadata)  # Store metadata as JSON string
        }
        
        return record
    
    def _prepare_sparse_vector(self, product: Product) -> Dict[str, Any]:
        """Prepare product for sparse index with server-side sparse embeddings."""
        
        # Get embedding text for the sparse index's server-side model
        embedding_text = self._build_embedding_text(product)
        
        # Use same metadata structure
        metadata = self._build_metadata(product)
        
        # For server-side sparse embeddings, we only need text
        # Pinecone will generate sparse embeddings automatically
        record = {
            '_id': f"{product.id}",
            'text': embedding_text,  # Pinecone-sparse-english-v0 will process this
            'name': product.name,
            # 'descriptor': embedding_text,  # This will be embedded server-side
            'metadata': json.dumps(metadata)  # Store metadata as JSON string
        }
        
        return record
    
    def _build_embedding_text(self, product: Product) -> str:
        """
        Build text for dense embedding.
        
        The descriptor field is specifically generated for dense embeddings
        and already includes all relevant semantic information.
        """
        # Use the enhanced descriptor - it's specifically optimized for dense embeddings
        if product.descriptor:
            return product.descriptor
        
        # Fallback to voice summary if no descriptor
        if product.voice_summary:
            return product.voice_summary
        
        # Final fallback to original description
        if product.description:
            return product.description
        
        # Last resort - use product name
        if product.name:
            return product.name
        
        return ""
    
    def _build_metadata(self, product: Product) -> Dict[str, Any]:
        """
        Build minimal metadata for RAG lookup + filtering.
        
        Since we lookup full product from database using the ID,
        we only need essential fields for search, filtering, and reranking.
        """
        
        # Extract price for filtering
        price_str = product.salePrice or product.originalPrice or getattr(product, 'price', None)
        extracted_price = 0.0
        if price_str:
            try:
                # Remove $ and commas, convert to float
                extracted_price = float(str(price_str).replace('$', '').replace(',', ''))
            except (ValueError, TypeError):
                extracted_price = 0.0
        
        # Get primary category for basic filtering
        primary_category = product.categories[0] if product.categories else 'general'
        
        # Build minimal metadata
        metadata = {
            # Essential fields
            'product_id': f"{product.id}",  # For database lookup (renamed to avoid conflict with _id)
            'name': product.name,  # Helpful for debugging/monitoring
            
            # Basic filtering fields
            'price': extracted_price,
            'category': primary_category,
            
            # System fields
            'content_type': 'product',
            'last_updated': datetime.now().isoformat()
        }
        
        # Optional: Add flattened product labels for Pinecone-side filtering
        # Only include if you need to filter by these in Pinecone
        if product.product_labels:
            # Flatten labels for Pinecone's filtering syntax
            for label_type, values in product.product_labels.items():
                if isinstance(values, list) and values:
                    # Use first value as primary
                    metadata[label_type] = values[0].lower()
                elif isinstance(values, str):
                    metadata[label_type] = values.lower()
        
        return metadata
    
    async def _detect_changes(
        self,
        products: List,
        force_update: bool
    ) -> Dict[str, List[str]]:
        """Detect which products need to be added, updated, or deleted."""
        
        if force_update:
            return {
                'add': [getattr(p, 'id', '') for p in products],
                'update': [],
                'delete': []
            }
        
        # Load sync state if not already loaded
        if self.sync_state is None:
            self.sync_state = await self._load_sync_state()
        
        current_products = {getattr(p, 'id', ''): self._get_product_hash(p) for p in products}
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
    
    def _get_product_hash(self, product) -> str:
        """Generate hash for change detection based on fields we actually store."""

        try:
            # Only hash fields that affect what we store in Pinecone
            price = product.salePrice or product.originalPrice or 0.0
            if not price:
                price = 0.0
            
            content = json.dumps({
                'id': product.id,
                'name': product.name if product.name else '',
                'price': price,
                'inventory': 0,
                'categories': product.categories if product.categories else [],
                'descriptor': product.descriptor if product.descriptor else '',  # Affects embedding text
                'product_labels': product.product_labels if product.product_labels else {}
            }, sort_keys=True)
            
            return hashlib.md5(content.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error generating product hash: {e}")
            raise e
    
    async def _save_filter_labels(self, filter_labels: Dict[str, Any]):
        """Save filter labels using storage provider."""
        
        try:
            # Convert filter labels to JSON string
            filter_json = json.dumps(filter_labels, indent=2)
            
            # Save using storage provider
            success = await self.storage.write_file(
                account=self.brand_domain,
                file_path="filter_dictionary.json",
                content=filter_json,
                content_type="application/json"
            )
            
            if success:
                logger.info(f"💾 Saved filter labels via storage provider")
            else:
                logger.error(f"❌ Failed to save filter labels")
                
        except Exception as e:
            logger.error(f"❌ Error saving filter labels: {e}")
    
    async def _load_sync_state(self) -> Dict[str, Any]:
        """Load sync state using storage provider."""
        
        try:
            content = await self.storage.read_file(
                account=self.brand_domain,
                file_path=self.sync_state_path
            )
            
            if content:
                return json.loads(content)
            else:
                logger.info("No existing sync state found - starting fresh")
                return {}
                
        except Exception as e:
            logger.warning(f"Could not load sync state: {e}")
            return {}
    
    async def _update_sync_state(self, products: List):
        """Update sync state with current product hashes using storage provider."""
        
        try:
            self.sync_state['product_hashes'] = {
                p.id: self._get_product_hash(p) for p in products
            }
            self.sync_state['last_sync'] = datetime.now().isoformat()
        
            # Save using storage provider
            success = await self.storage.write_file(
                account=self.brand_domain,
                file_path=self.sync_state_path,
                content=json.dumps(self.sync_state, indent=2),
                content_type="application/json"
            )
            
            if success:
                logger.info(f"💾 Updated sync state for {len(products)} products via storage provider")
            else:
                logger.error("❌ Failed to save sync state")
                
        except Exception as e:
            logger.error(f"❌ Error saving sync state: {e}")