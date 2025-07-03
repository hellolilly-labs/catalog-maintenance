#!/usr/bin/env python3
"""
Product Catalog Ingestor

A professional class-based ingestion system for product catalogs with comprehensive
timestamp tracking and history management. Products are loaded via ProductManager 
from the unified storage system.

Features:
- Class-based architecture (ProductCatalogIngestor)
- Unique ingestion ID generation with timestamps
- Complete ingestion history tracking via storage provider
- Pinecone metadata includes ingestion timestamps for data freshness
- ProductManager integration for unified data loading
- Filter analysis and summary generation
- Preview and analysis-only modes

Usage:
    python ingest_product_catalog.py specialized.com --preview
    python ingest_product_catalog.py balenciaga.com --force
    python ingest_product_catalog.py sundayriley.com --filters-only
    python ingest_product_catalog.py specialized.com --history
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime, timezone

from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..core.separate_index_ingestion import SeparateIndexIngestion
from ...catalog.unified_descriptor_generator import UnifiedDescriptorGenerator
from ...agents.catalog_filter_analyzer import CatalogFilterAnalyzer
from liddy.models.product_manager import get_product_manager
from liddy.storage import get_account_storage_provider

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductCatalogIngestor:
    """
    Professional product catalog ingestion system.
    
    Features:
    - ProductManager integration for unified data loading
    - Timestamp tracking for ingestion history
    - Storage provider integration
    - Filter analysis and summary generation
    - Preview and analysis-only modes
    """
    
    def __init__(
        self,
        brand_domain: str,
        dense_index_name: Optional[str] = None,
        sparse_index_name: Optional[str] = None
    ):
        """
        Initialize ProductCatalogIngestor.
        
        Args:
            brand_domain: Brand domain (e.g., "specialized.com")
            dense_index_name: Override dense index name
            sparse_index_name: Override sparse index name
        """
        self.brand_domain = brand_domain
        self.brand_name = brand_domain.replace('.', '-')
        
        # Index names
        self.dense_index_name = dense_index_name or f"{self.brand_name}-dense"
        self.sparse_index_name = sparse_index_name or f"{self.brand_name}-sparse"
        
        # Components
        self.product_manager = None
        self.storage = get_account_storage_provider()
        self.filter_analyzer = CatalogFilterAnalyzer(brand_domain)
        self.descriptor_generator = UnifiedDescriptorGenerator(brand_domain)
        self.ingestion_system = None
        
        # Ingestion metadata
        self.ingestion_id = self._generate_ingestion_id()
        self.ingestion_timestamp = datetime.now(timezone.utc)
        
        logger.info(f"🏭 Initialized ProductCatalogIngestor for {brand_domain}")
        logger.info(f"   Dense Index: {self.dense_index_name}")
        logger.info(f"   Sparse Index: {self.sparse_index_name}")
        logger.info(f"   Ingestion ID: {self.ingestion_id}")
    
    def _generate_ingestion_id(self) -> str:
        """Generate unique ingestion ID for tracking"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{self.brand_name}_{timestamp}"
    
    async def _initialize_components(self):
        """Initialize async components"""
        if self.product_manager is None:
            self.product_manager = await get_product_manager(self.brand_domain)
        
        if self.ingestion_system is None:
            self.ingestion_system = SeparateIndexIngestion(
                brand_domain=self.brand_domain,
                dense_index_name=self.dense_index_name,
                sparse_index_name=self.sparse_index_name
            )
    
    async def load_products(self) -> List[Dict[str, Any]]:
        """Load product catalog using ProductManager."""
        
        logger.info(f"📂 Loading catalog for {self.brand_domain} via ProductManager")
        
        await self._initialize_components()
        
        try:
            product_objects = await self.product_manager.get_product_objects()
            
            # Convert Product objects to dictionaries with ingestion metadata
            products = []
            for product in product_objects:
                product_dict = product.to_dict()
                
                # Add ingestion metadata
                product_dict['_ingestion_metadata'] = {
                    'ingestion_id': self.ingestion_id,
                    'ingestion_timestamp': self.ingestion_timestamp.isoformat(),
                    'last_updated': datetime.now(timezone.utc).isoformat(),
                    'brand_domain': self.brand_domain,
                    'version': '1.0'
                }
                
                products.append(product_dict)
            
            logger.info(f"✅ Loaded {len(products)} products with ingestion metadata")
            return products
            
        except Exception as e:
            logger.error(f"Failed to load products via ProductManager: {e}")
            raise
    
    async def analyze_filters_only(self) -> Dict[str, Any]:
        """Analyze catalog for filters only (no ingestion)"""
        
        logger.info(f"🔍 Starting filter analysis for {self.brand_domain}")
        
        # Load products
        products = await self.load_products()
        
        # Analyze filters
        filters = await self.filter_analyzer.analyze_product_catalog(products)
        
        # Save filter data
        await self.filter_analyzer.save_filters_to_file(filters, "catalog_filters.json")
        await self._save_filter_summary(filters)
        
        logger.info(f"✅ Filter analysis complete: {len([k for k in filters.keys() if not k.startswith('_')])} filter types")
    
        return filters

    async def preview_processing(self, sample_size: int = 3) -> Dict[str, Any]:
        """Preview how products will be processed."""
        
        logger.info(f"👀 Starting preview processing for {self.brand_domain}")
        
        # Load products
        products = await self.load_products()
        
        # Process using the unified generator
        enhanced_samples, filter_labels = await self.descriptor_generator.process_catalog(force_regenerate=False)
        
        # Take sample
        sample_products = enhanced_samples[:sample_size]
        
        # Save filter summary
        await self._save_filter_summary(filter_labels)
        
        logger.info(f"✅ Preview complete: processed {len(sample_products)} sample products")
        
        return {
            'sample_products': sample_products,
            'filter_labels': filter_labels,
            'total_products': len(products),
            'preview_size': sample_size,
            'ingestion_id': self.ingestion_id
        }
    
    async def create_indexes(self):
        """Create Pinecone indexes if they don't exist"""
        
        logger.info(f"🔨 Creating indexes for {self.brand_domain}")
        
        await self._initialize_components()
        await self.ingestion_system.create_indexes()
        
        logger.info(f"✅ Indexes created successfully")
    
    async def ingest_catalog(
        self,
        namespace: str = "products",
        batch_size: int = 100,
        force_update: bool = False
    ) -> Dict[str, Any]:
        """
        Perform full catalog ingestion with timestamp tracking.
        
        Args:
            namespace: Namespace for vectors
            batch_size: Batch size for ingestion
            force_update: Force update all products
        
    Returns:
            Ingestion statistics with timestamp information
        """
        
        logger.info(f"🚀 Starting catalog ingestion for {self.brand_domain}")
        logger.info(f"   Ingestion ID: {self.ingestion_id}")
        logger.info(f"   Timestamp: {self.ingestion_timestamp.isoformat()}")
        logger.info(f"   Force Update: {force_update}")
        
        await self._initialize_components()
        
        # Get product count for display
        products = await self.product_manager.get_products()
        
        # Perform ingestion
        stats = await self.ingestion_system.ingest_catalog(
            namespace=namespace,
            batch_size=batch_size,
            force_update=force_update
        )
        
        # Add ingestion metadata to stats
        stats.update({
            'ingestion_id': self.ingestion_id,
            'ingestion_timestamp': self.ingestion_timestamp.isoformat(),
            'brand_domain': self.brand_domain,
            'namespace': namespace,
            'force_update': force_update,
            'batch_size': batch_size
        })
        
        # Save ingestion history
        await self._save_ingestion_history(stats)
        
        logger.info(f"✅ Catalog ingestion complete for {self.brand_domain}")
        logger.info(f"   Products processed: {stats['products_processed']}")
        logger.info(f"   Ingestion ID: {self.ingestion_id}")
        
        return stats
    
    async def _save_filter_summary(self, filters: Dict[str, Any]) -> None:
        """Save a human-readable summary of extracted filters"""
        
        try:
            summary_content = f"# Filter Summary for {self.brand_domain}\n\n"
            summary_content += f"**Ingestion ID:** {self.ingestion_id}\n"
            summary_content += f"**Generated:** {filters.get('_metadata', {}).get('generated_at', 'unknown')}\n"
            summary_content += f"**Total Products Analyzed:** {filters.get('_metadata', {}).get('total_products', 'unknown')}\n\n"
        
            for filter_name, filter_config in filters.items():
                if filter_name.startswith('_'):
                    continue
                    
                filter_type = filter_config.get('type', 'unknown')
                summary_content += f"## {filter_name} ({filter_type})\n\n"
                
                if filter_type == "categorical":
                    values = filter_config.get('values', [])
                    summary_content += f"**Values:** {', '.join(values)}\n\n"
                    
                    aliases = filter_config.get('aliases', {})
                    if aliases:
                        summary_content += "**Aliases:**\n"
                        for value, alias_list in aliases.items():
                            summary_content += f"- {value}: {', '.join(alias_list)}\n"
                            summary_content += "\n"
                        
                elif filter_type == "numeric_range":
                    min_val = filter_config.get('min')
                    max_val = filter_config.get('max')
                    unit = filter_config.get('unit', '')
                    summary_content += f"**Range:** {min_val} to {max_val} {unit}\n\n"
                    
                    ranges = filter_config.get('common_ranges', [])
                    if ranges:
                        summary_content += "**Common Ranges:**\n"
                        for range_def in ranges:
                            label = range_def['label']
                            range_vals = range_def['range']
                            summary_content += f"- {label}: {range_vals[0]} to {range_vals[1]}\n"
                            summary_content += "\n"
                        
                elif filter_type == "multi_select":
                    values = filter_config.get('values', [])
                    summary_content += f"**Options:** {', '.join(values)}\n\n"
                    
                    frequency = filter_config.get('frequency', {})
                    if frequency:
                        sorted_freq = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
                        summary_content += "**Most Common:**\n"
                        for value, count in sorted_freq[:5]:
                            summary_content += f"- {value}: {count} products\n"
                            summary_content += "\n"
                
                # Save using storage provider
                success = await self.storage.write_file(
                    account=self.brand_domain,
                    file_path="filter_summary.md",
                    content=summary_content,
                    content_type="text/markdown"
                )
                
                if success:
                    logger.info(f"📄 Filter summary saved via storage provider")
                else:
                    logger.error(f"❌ Failed to save filter summary")
                
        except Exception as e:
            logger.error(f"❌ Error saving filter summary: {e}")
    
    async def _save_ingestion_history(self, stats: Dict[str, Any]) -> None:
        """Save ingestion history for tracking"""
        
        try:
            # Create ingestion record
            ingestion_record = {
                'ingestion_id': self.ingestion_id,
                'brand_domain': self.brand_domain,
                'timestamp': self.ingestion_timestamp.isoformat(),
                'stats': stats,
                'indexes': {
                    'dense': self.dense_index_name,
                    'sparse': self.sparse_index_name
                }
            }
            
            # Save to storage
            success = await self.storage.write_file(
                account=self.brand_domain,
                file_path=f"ingestion_history/{self.ingestion_id}.json",
                content=json.dumps(ingestion_record, indent=2),
                content_type="application/json"
            )
            
            if success:
                logger.info(f"📝 Ingestion history saved: {self.ingestion_id}")
            else:
                logger.error(f"❌ Failed to save ingestion history")
                
        except Exception as e:
            logger.error(f"❌ Error saving ingestion history: {e}")
    
    async def get_ingestion_history(self) -> List[Dict[str, Any]]:
        """Get ingestion history for this brand"""
        
        try:
            files = await self.storage.list_files(self.brand_domain, "ingestion_history")
            
            history = []
            for file in files:
                if file.endswith('.json'):
                    content = await self.storage.read_file(self.brand_domain, f"ingestion_history/{file}")
                    if content:
                        history.append(json.loads(content))
            
            # Sort by timestamp (newest first)
            history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            return history
            
        except Exception as e:
            logger.error(f"❌ Error getting ingestion history: {e}")
            return []


# CLI Functions for backward compatibility
async def display_preview_results(preview_data: Dict[str, Any]):
    """Display preview results in a user-friendly format"""
    
    sample_products = preview_data['sample_products']
    filter_labels = preview_data['filter_labels']
    
    print("\n" + "="*60)
    print("PREVIEW: Product Processing")
    print("="*60)
    
    for i, product in enumerate(sample_products):
        print(f"\n--- Product {i+1} ---")
        
        # Show processed results
        print(f"Product ID: {getattr(product, 'id', 'N/A')}")
        print(f"Name: {getattr(product, 'name', 'N/A')}")
        print(f"Categories: {getattr(product, 'categories', [])}")
        
        # Show enhanced descriptor
        descriptor = getattr(product, 'descriptor', '')
        if descriptor:
            print(f"\nRAG-Optimized Descriptor ({len(descriptor.split())} words):")
            print(f"{descriptor[:300]}...")
        
        voice_summary = getattr(product, 'voice_summary', '')
        if voice_summary:
            print(f"\nVoice Summary: {voice_summary}")
        
        search_keywords = getattr(product, 'search_keywords', []) or []
        if search_keywords:
            print(f"\nSearch Keywords: {', '.join(search_keywords[:10])}")
        
        key_points = getattr(product, 'key_selling_points', []) or []
        if key_points:
            print(f"\nKey Selling Points:")
            for point in key_points:
                print(f"  • {point}")
        
        # Show quality metadata
        metadata = getattr(product, 'descriptor_metadata', None)
        if metadata:
            print(f"\nQuality Score: {getattr(metadata, 'quality_score', 'N/A')}")
            print(f"Generator Version: {getattr(metadata, 'generator_version', 'N/A')}")
    
    # Show filter labels
    print("\n" + "="*60)
    print("EXTRACTED FILTER LABELS")
    print("="*60)
    
    for filter_name, filter_data in filter_labels.items():
        if filter_name != '_metadata':
            print(f"\n{filter_name}:")
            print(f"  Label: {filter_data.get('label', 'N/A')}")
            values = list(filter_data.get('values', []))
            print(f"  Values: {', '.join(values[:10])}")
            if len(values) > 10:
                print(f"  ... and {len(values) - 10} more")
    
    print(f"\n📄 Filter summary saved via storage provider")


async def display_filter_results(filters: Dict[str, Any]):
    """Display filter analysis results in a user-friendly format"""
    
    print("\n" + "="*60)
    print("FILTER ANALYSIS RESULTS")
    print("="*60)
    
    filter_count = len([k for k in filters.keys() if not k.startswith('_')])
    print(f"\n🎉 Extracted {filter_count} filter types")
        
    # Show key filters
    for filter_name, filter_config in filters.items():
        if filter_name.startswith('_'):
            continue
            
        filter_type = filter_config.get('type')
        if filter_type == "categorical":
            values = filter_config.get('values', [])
            print(f"   📂 {filter_name}: {len(values)} categories")
        elif filter_type == "numeric_range":
            min_val = filter_config.get('min')
            max_val = filter_config.get('max')
            print(f"   📊 {filter_name}: {min_val} to {max_val}")
        elif filter_type == "multi_select":
            values = filter_config.get('values', [])
            print(f"   ☑️  {filter_name}: {len(values)} options")
    
    print(f"\n💾 Filters saved via storage provider: catalog_filters.json")
    print(f"📄 Summary saved via storage provider: filter_summary.md")


async def main():
    parser = argparse.ArgumentParser(
        description='Ingest products into separate Pinecone indexes using ProductManager'
    )
    parser.add_argument(
        'brand_domain',
        help='Brand domain (e.g., specialized.com)'
    )
    parser.add_argument(
        '--dense-index',
        help='Dense index name (default: {brand}-dense-v2)'
    )
    parser.add_argument(
        '--sparse-index',
        help='Sparse index name (default: {brand}-sparse-v2)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force update all products'
    )
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Preview processing without ingesting'
    )
    parser.add_argument(
        '--preview-size',
        type=int,
        default=3,
        help='Number of products to preview (default: 3)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for ingestion (default: 100)'
    )
    parser.add_argument(
        '--namespace',
        default='products',
        help='Namespace for vectors (default: products)'
    )
    parser.add_argument(
        '--create-indexes',
        action='store_true',
        help='Create indexes if they don\'t exist'
    )
    parser.add_argument(
        '--filters-only',
        action='store_true',
        help='Analyze filters only (no ingestion or vector processing)'
    )
    parser.add_argument(
        '--history',
        action='store_true',
        help='Show ingestion history for this brand'
    )
    
    args = parser.parse_args()
    
    # Initialize ProductCatalogIngestor
    try:
        ingestor = ProductCatalogIngestor(
            brand_domain=args.brand_domain,
            dense_index_name=args.dense_index,
            sparse_index_name=args.sparse_index
        )
        
        print(f"🏭 ProductCatalogIngestor initialized for {args.brand_domain}")
        print(f"   Ingestion ID: {ingestor.ingestion_id}")
        
    except Exception as e:
        logger.error(f"Failed to initialize ingestor: {e}")
        return 1
    
    # Show ingestion history
    if args.history:
        try:
            history = await ingestor.get_ingestion_history()
            
            print("\n📜 Ingestion History")
            print("="*50)
            
            if history:
                for i, record in enumerate(history[:10]):  # Show last 10
                    stats = record.get('stats', {})
                    timestamp = record.get('timestamp', 'unknown')
                    ingestion_id = record.get('ingestion_id', 'unknown')
                    
                    print(f"\n{i+1}. {ingestion_id}")
                    print(f"   Timestamp: {timestamp}")
                    print(f"   Products: {stats.get('products_processed', 0)}")
                    print(f"   Added: {stats.get('added', 0)}, Updated: {stats.get('updated', 0)}")
            else:
                print("No ingestion history found.")
                
            return 0
            
        except Exception as e:
            logger.error(f"Failed to get ingestion history: {e}")
            return 1
    
    # Filters-only mode
    if args.filters_only:
        try:
            filters = await ingestor.analyze_filters_only()
            await display_filter_results(filters)
            return 0
        except Exception as e:
            logger.error(f"Filter analysis failed: {e}")
            return 1
    
    # Preview mode
    if args.preview:
        try:
            preview_data = await ingestor.preview_processing(args.preview_size)
            await display_preview_results(preview_data)
            
            print("\n" + "="*60)
            print("INDEX CONFIGURATION")
            print("="*60)
            print(f"Dense Index: {ingestor.dense_index_name}")
            print(f"Sparse Index: {ingestor.sparse_index_name}")
            print(f"Namespace: {args.namespace}")
            print(f"Batch Size: {args.batch_size}")
            print(f"Ingestion ID: {ingestor.ingestion_id}")
            
            print("\n✅ Preview complete. Use --create-indexes to create indexes and ingest.")
            return 0
            
        except Exception as e:
            logger.error(f"Preview failed: {e}")
            return 1
    
    # Create indexes if requested
    if args.create_indexes:
        try:
            print("\n🔨 Creating indexes...")
            await ingestor.create_indexes()
            print("✅ Indexes created successfully")
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
            return 1
    
    # Perform ingestion
    try:
        stats = await ingestor.ingest_catalog(
            namespace=args.namespace,
            batch_size=args.batch_size,
            force_update=args.force
        )
        
        print("\n✅ Ingestion Complete!")
        print("\n📊 Statistics:")
        print(f"  Ingestion ID: {stats['ingestion_id']}")
        print(f"  Timestamp: {stats['ingestion_timestamp']}")
        print(f"  Products Processed: {stats['products_processed']}")
        print(f"  Added: {stats['added']}")
        print(f"  Updated: {stats['updated']}")
        print(f"  Deleted: {stats['deleted']}")
        print(f"  Dense Vectors: {stats['dense_vectors']}")
        print(f"  Sparse Vectors: {stats['sparse_vectors']}")
        print(f"  Filter Labels: {stats['filter_labels']}")
        
        print(f"\n📁 Filter dictionary saved via storage provider")
        print(f"📝 Ingestion history saved: {stats['ingestion_id']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))