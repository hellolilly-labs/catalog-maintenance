#!/usr/bin/env python3
"""
Product Ingestion with Separate Indexes

This script ingests product catalogs into separate dense and sparse indexes
following Pinecone's best practices.

Usage:
    python ingest_products_separate_indexes.py specialized.com data/products.json
    python ingest_products_separate_indexes.py balenciaga.com catalog.json --force
    python ingest_products_separate_indexes.py sundayriley.com products.json --preview
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import asyncio

from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.separate_index_ingestion import SeparateIndexIngestion
from src.ingestion.universal_product_processor import UniversalProductProcessor
from src.catalog.enhanced_descriptor_generator import EnhancedDescriptorGenerator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_product_catalog(file_path: str) -> List[Dict[str, Any]]:
    """Load product catalog from JSON file."""
    
    logger.info(f"📂 Loading catalog from {file_path}")
    
    with open(file_path) as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, list):
        products = data
    elif isinstance(data, dict):
        # Try common keys
        for key in ['products', 'items', 'catalog', 'data']:
            if key in data and isinstance(data[key], list):
                products = data[key]
                break
        else:
            # Single product
            products = [data]
    else:
        raise ValueError("Invalid catalog format")
    
    logger.info(f"✅ Loaded {len(products)} products")
    return products


def preview_processing(brand_domain: str, products: List[Dict[str, Any]], sample_size: int = 3):
    """Preview how products will be processed."""
    
    print("\n" + "="*60)
    print("PREVIEW: Product Processing")
    print("="*60)
    
    processor = UniversalProductProcessor(brand_domain)
    generator = EnhancedDescriptorGenerator(brand_domain)
    
    # Process sample products
    sample_products = products[:sample_size]
    processed_samples = processor.process_catalog(sample_products)
    enhanced_samples, filter_labels = generator.process_catalog(processed_samples)
    
    for i, product in enumerate(enhanced_samples):
        print(f"\n--- Product {i+1} ---")
        
        # Show original
        original = next(p for p in products if processor._generate_product_id(p) == product['id'])
        print(f"Original: {json.dumps(original, indent=2)[:300]}...")
        
        # Show processed results
        print(f"\nProcessed ID: {product['id']}")
        print(f"Universal Fields: {json.dumps(product['universal_fields'], indent=2)}")
        
        # Show enhanced descriptor
        print(f"\nEnhanced Descriptor ({len(product.get('enhanced_description', '').split())} words):")
        print(f"{product.get('enhanced_description', '')[:300]}...")
        
        print(f"\nVoice Summary: {product.get('voice_summary', '')}")
        print(f"\nSearch Keywords: {', '.join(product.get('search_keywords', [])[:10])}")
        
        print(f"\nKey Selling Points:")
        for point in product.get('key_selling_points', []):
            print(f"  • {point}")
        
        print(f"\nFilters: {json.dumps(product.get('filter_metadata', {}), indent=2)}")
    
    # Show filter labels
    print("\n" + "="*60)
    print("EXTRACTED FILTER LABELS")
    print("="*60)
    
    for filter_name, filter_data in filter_labels.items():
        if filter_name != '_metadata':
            print(f"\n{filter_name}:")
            print(f"  Label: {filter_data.get('label', 'N/A')}")
            print(f"  Values: {', '.join(list(filter_data.get('values', []))[:10])}")
            if len(filter_data.get('values', [])) > 10:
                print(f"  ... and {len(filter_data['values']) - 10} more")


async def main():
    parser = argparse.ArgumentParser(
        description='Ingest products into separate Pinecone indexes'
    )
    parser.add_argument(
        'brand_domain',
        help='Brand domain (e.g., specialized.com)'
    )
    parser.add_argument(
        'catalog_path',
        help='Path to product catalog JSON file'
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
    
    args = parser.parse_args()
    
    # Derive index names if not provided
    brand_prefix = args.brand_domain.split('.')[0]
    dense_index = args.dense_index or f"{brand_prefix}-dense-v2"
    sparse_index = args.sparse_index or f"{brand_prefix}-sparse-v2"
    
    # Load catalog
    try:
        products = load_product_catalog(args.catalog_path)
    except Exception as e:
        logger.error(f"Failed to load catalog: {e}")
        return 1
    
    # Preview mode
    if args.preview:
        preview_processing(args.brand_domain, products, args.preview_size)
        
        print("\n" + "="*60)
        print("INDEX CONFIGURATION")
        print("="*60)
        print(f"Dense Index: {dense_index}")
        print(f"Sparse Index: {sparse_index}")
        print(f"Namespace: {args.namespace}")
        print(f"Batch Size: {args.batch_size}")
        
        print("\n✅ Preview complete. Use --create-indexes to create indexes and ingest.")
        return 0
    
    # Initialize ingestion system
    try:
        ingestion = SeparateIndexIngestion(
            brand_domain=args.brand_domain,
            dense_index_name=dense_index,
            sparse_index_name=sparse_index
        )
    except Exception as e:
        logger.error(f"Failed to initialize ingestion: {e}")
        return 1
    
    # Create indexes if requested
    if args.create_indexes:
        try:
            print("\n🔨 Creating indexes...")
            await ingestion.create_indexes()
            print("✅ Indexes created successfully")
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
            return 1
    
    # Perform ingestion
    try:
        print("\n🚀 Starting ingestion...")
        print(f"  Brand: {args.brand_domain}")
        print(f"  Products: {len(products)}")
        print(f"  Dense Index: {dense_index}")
        print(f"  Sparse Index: {sparse_index}")
        print(f"  Force Update: {args.force}")
        
        stats = await ingestion.ingest_catalog(
            catalog_path=args.catalog_path,
            namespace=args.namespace,
            batch_size=args.batch_size,
            force_update=args.force
        )
        
        print("\n✅ Ingestion Complete!")
        print("\n📊 Statistics:")
        print(f"  Products Processed: {stats['products_processed']}")
        print(f"  Added: {stats['added']}")
        print(f"  Updated: {stats['updated']}")
        print(f"  Deleted: {stats['deleted']}")
        print(f"  Dense Vectors: {stats['dense_vectors']}")
        print(f"  Sparse Vectors: {stats['sparse_vectors']}")
        print(f"  Filter Labels: {stats['filter_labels']}")
        
        # Save filter dictionary location
        filter_path = Path(f"accounts/{args.brand_domain}/filter_dictionary.json")
        if filter_path.exists():
            print(f"\n📁 Filter dictionary saved to: {filter_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))