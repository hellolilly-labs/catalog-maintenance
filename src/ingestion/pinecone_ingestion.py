"""
Enhanced Pinecone Ingestion System

Handles ingestion of products into Pinecone with:
- Dense embeddings for semantic search
- Sparse embeddings for keyword precision
- Rich metadata for filtering
- Automatic synchronization
"""

import os
import json
import time
import hashlib
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from pinecone import Pinecone
from tqdm import tqdm

from .universal_product_processor import UniversalProductProcessor
from .sparse_embeddings import SparseEmbeddingGenerator
from ..agents.catalog_filter_analyzer import CatalogFilterAnalyzer
from ..catalog.unified_descriptor_generator import UnifiedDescriptorGenerator

logger = logging.getLogger(__name__)


class PineconeIngestion:
    """
    Manages product ingestion into Pinecone with hybrid search support.
    """
    
    def __init__(
        self,
        brand_domain: str,
        index_name: str,
        namespace: str = "products",
        embedding_model: str = "llama-text-embed-v2",
        batch_size: int = 45
    ):
        self.brand_domain = brand_domain
        self.index_name = index_name
        self.namespace = namespace
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        self.index = self.pc.Index(index_name)
        
        # Initialize processors
        self.product_processor = UniversalProductProcessor(brand_domain)
        self.descriptor_generator = UnifiedDescriptorGenerator(brand_domain)
        self.sparse_generator = SparseEmbeddingGenerator(brand_domain)
        
        # Paths for tracking
        self.brand_path = Path(f"accounts/{brand_domain}")
        self.sync_state_path = self.brand_path / "pinecone_sync_state.json"
        
        # Load sync state
        self.sync_state = self._load_sync_state()
        
        logger.info(f"🚀 Initialized Pinecone Ingestion for {brand_domain}")
    
    def ingest_products(
        self,
        products: List[Dict[str, Any]],
        force_update: bool = False,
        update_prompts: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest products into Pinecone with change detection.
        
        Args:
            products: List of raw products to ingest
            force_update: Force update all products regardless of changes
            update_prompts: Update Langfuse prompts with filter dictionaries
            
        Returns:
            Ingestion statistics
        """
        
        logger.info(f"📦 Starting ingestion of {len(products)} products")
        
        # Analyze catalog for filters
        filter_analyzer = CatalogFilterAnalyzer(self.brand_domain)
        catalog_filters = filter_analyzer.analyze_product_catalog(products)
        filter_analyzer.save_filters_to_file(catalog_filters, "catalog_filters.json")
        
        # Process products
        processed_products = self.product_processor.process_catalog(products)
        
        # Build or load sparse vocabulary
        if not self.sparse_generator.load_vocabulary() or force_update:
            logger.info("🔨 Building sparse embedding vocabulary...")
            self.sparse_generator.build_vocabulary(processed_products)
        
        # Detect changes
        changes = self._detect_changes(processed_products, force_update)
        
        logger.info(f"📊 Changes detected - Add: {len(changes['add'])}, "
                   f"Update: {len(changes['update'])}, Delete: {len(changes['delete'])}")
        
        # Prepare records for Pinecone
        records_to_upsert = []
        
        # Process additions and updates
        for product_id in changes['add'] + changes['update']:
            product = next(p for p in processed_products if p['id'] == product_id)
            record = self._prepare_pinecone_record(product)
            records_to_upsert.append(record)
        
        # Upsert to Pinecone
        stats = {
            'added': 0,
            'updated': 0,
            'deleted': 0,
            'errors': 0,
            'start_time': time.time()
        }
        
        if records_to_upsert:
            stats.update(self._upsert_to_pinecone(records_to_upsert))
        
        # Handle deletions
        if changes['delete']:
            stats['deleted'] = self._delete_from_pinecone(changes['delete'])
        
        # Update sync state
        self._update_sync_state(processed_products)
        
        # Update prompts if requested
        if update_prompts:
            self._update_langfuse_prompts(catalog_filters)
        
        stats['duration'] = time.time() - stats['start_time']
        
        logger.info(f"✅ Ingestion complete - Added: {stats['added']}, "
                   f"Updated: {stats['updated']}, Deleted: {stats['deleted']}, "
                   f"Errors: {stats['errors']}, Time: {stats['duration']:.2f}s")
        
        return stats
    
    def _prepare_pinecone_record(self, processed_product: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare a product for Pinecone ingestion with dense and sparse embeddings.
        """
        
        # Generate enhanced descriptor
        enhanced = self.descriptor_generator.generate_unified_descriptor(
            processed_product['original_data']
        )
        
        # Combine descriptors for embedding
        embedding_text = f"{enhanced['enhanced_descriptor']} {processed_product['enhanced_descriptor']}"
        
        # Generate sparse embeddings using enhanced generator
        sparse_data = self.sparse_generator.generate_sparse_embedding(
            processed_product,
            enhanced
        )
        
        # Build metadata
        metadata = {
            # Universal fields
            'id': processed_product['id'],
            'brand': processed_product['universal_fields'].get('brand', ''),
            'name': processed_product['universal_fields'].get('name', ''),
            'price': processed_product['universal_fields'].get('price', 0),
            'category': processed_product['universal_fields'].get('category', ['general'])[0],
            
            # Filter metadata
            **processed_product['filter_metadata'],
            
            # Content for response
            'description': processed_product['universal_fields'].get('description', ''),
            'enhanced_descriptor': enhanced['enhanced_descriptor'],
            'voice_summary': processed_product['voice_summary'],
            'key_selling_points': json.dumps(processed_product['key_selling_points']),
            'search_keywords': json.dumps(processed_product['search_keywords']),
            
            # Additional metadata
            'content_type': 'product',
            'last_updated': datetime.now().isoformat()
        }
        
        # Handle image URLs
        if 'images' in processed_product['universal_fields']:
            images = processed_product['universal_fields']['images']
            if isinstance(images, list) and images:
                metadata['image_url'] = images[0]
        
        return {
            'id': processed_product['id'],
            'text': embedding_text,  # For server-side embedding
            'sparse_values': sparse_data,
            'metadata': metadata
        }
    
    def _detect_changes(
        self,
        processed_products: List[Dict[str, Any]],
        force_update: bool
    ) -> Dict[str, List[str]]:
        """
        Detect which products need to be added, updated, or deleted.
        """
        
        if force_update:
            return {
                'add': [p['id'] for p in processed_products],
                'update': [],
                'delete': []
            }
        
        current_products = {p['id']: self._get_product_hash(p) for p in processed_products}
        previous_products = self.sync_state.get('product_hashes', {})
        
        # Detect changes
        current_ids = set(current_products.keys())
        previous_ids = set(previous_products.keys())
        
        # New products
        add_ids = current_ids - previous_ids
        
        # Deleted products
        delete_ids = previous_ids - current_ids
        
        # Updated products (hash changed)
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
        """Generate a hash for change detection."""
        
        # Create a consistent representation
        content = json.dumps({
            'universal_fields': product['universal_fields'],
            'filter_metadata': product['filter_metadata'],
            'key_selling_points': product['key_selling_points']
        }, sort_keys=True)
        
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _upsert_to_pinecone(self, records: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Upsert records to Pinecone in batches.
        """
        
        stats = {'added': 0, 'updated': 0, 'errors': 0}
        
        # Process in batches
        batches = [records[i:i + self.batch_size] for i in range(0, len(records), self.batch_size)]
        
        with tqdm(total=len(batches), desc="Upserting to Pinecone") as pbar:
            for batch in batches:
                try:
                    # Format for Pinecone upsert
                    upsert_records = []
                    for record in batch:
                        upsert_record = {
                            "_id": record['id'],
                            "text": record['text'],
                            "metadata": json.dumps(record['metadata'])
                        }
                        
                        # Add sparse values if available
                        if 'sparse_values' in record and record['sparse_values']:
                            upsert_record['sparse_values'] = record['sparse_values']
                        
                        upsert_records.append(upsert_record)
                    
                    # Upsert to Pinecone
                    self.index.upsert_records(
                        namespace=self.namespace,
                        records=upsert_records
                    )
                    
                    # Update stats (simplified - all are considered updates in Pinecone)
                    stats['updated'] += len(batch)
                    
                except Exception as e:
                    logger.error(f"Error upserting batch: {e}")
                    stats['errors'] += len(batch)
                
                pbar.update(1)
                
                # Rate limiting
                time.sleep(1)
        
        return stats
    
    def _delete_from_pinecone(self, product_ids: List[str]) -> int:
        """Delete products from Pinecone."""
        
        try:
            self.index.delete(
                ids=product_ids,
                namespace=self.namespace
            )
            return len(product_ids)
        except Exception as e:
            logger.error(f"Error deleting products: {e}")
            return 0
    
    def _load_sync_state(self) -> Dict[str, Any]:
        """Load the sync state from disk."""
        
        if self.sync_state_path.exists():
            try:
                with open(self.sync_state_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load sync state: {e}")
        
        return {
            'product_hashes': {},
            'last_sync': None,
            'version': '1.0'
        }
    
    def _update_sync_state(self, processed_products: List[Dict[str, Any]]) -> None:
        """Update and save the sync state."""
        
        self.sync_state['product_hashes'] = {
            p['id']: self._get_product_hash(p) for p in processed_products
        }
        self.sync_state['last_sync'] = datetime.now().isoformat()
        self.sync_state['total_products'] = len(processed_products)
        
        # Save to disk
        self.sync_state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.sync_state_path, 'w') as f:
            json.dump(self.sync_state, f, indent=2)
    
    def _update_langfuse_prompts(self, catalog_filters: Dict[str, Any]) -> None:
        """Update Langfuse prompts with filter dictionaries."""
        
        try:
            from langfuse import Langfuse
            langfuse = Langfuse()
            
            # Prepare filter dictionary for prompts
            filter_dict = {
                'brand': self.brand_domain,
                'filters': {},
                'categories': [],
                'attributes': []
            }
            
            # Extract key information
            for filter_name, filter_config in catalog_filters.items():
                if filter_name.startswith('_'):
                    continue
                
                filter_type = filter_config.get('type')
                if filter_type == 'categorical' and filter_name == 'category':
                    filter_dict['categories'] = filter_config.get('values', [])
                elif filter_type == 'categorical':
                    filter_dict['filters'][filter_name] = filter_config.get('values', [])
                elif filter_type == 'multi_select':
                    filter_dict['attributes'].extend(filter_config.get('values', []))
            
            # Update prompt (this would need to be configured in Langfuse)
            prompt_name = f"liddy/catalog/{self.brand_domain}/filter_dictionary"
            logger.info(f"📝 Would update Langfuse prompt: {prompt_name}")
            # In practice: langfuse.update_prompt(prompt_name, filter_dict)
            
        except Exception as e:
            logger.warning(f"Failed to update Langfuse prompts: {e}")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get current index statistics."""
        
        try:
            stats = self.index.describe_index_stats()
            namespace_stats = stats.namespaces.get(self.namespace, {})
            
            return {
                'total_vectors': namespace_stats.vector_count,
                'index_fullness': stats.index_fullness,
                'dimension': stats.dimension,
                'namespaces': list(stats.namespaces.keys())
            }
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {}