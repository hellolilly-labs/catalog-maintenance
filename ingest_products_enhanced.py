#!/usr/bin/env python3
"""
Enhanced Product Ingestion for Pinecone RAG

This script ingests product catalogs into Pinecone with:
- Universal product processing (works for any brand/category)
- Enhanced descriptors optimized for voice AI
- Hybrid search support (dense + sparse embeddings)
- Automatic change detection and incremental updates
- Filter metadata integration

Usage:
    python ingest_products_enhanced.py specialized.com data/products.json --index specialized-llama-2048
    python ingest_products_enhanced.py balenciaga.com catalog.json --index balenciaga-llama-2048 --force
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import asyncio

from src.ingestion import PineconeIngestion, UniversalProductProcessor
from src.agents.catalog_filter_analyzer import CatalogFilterAnalyzer

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
            raise ValueError("Could not find product list in JSON structure")
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
    
    # Process sample products
    sample_products = products[:sample_size]
    
    for i, product in enumerate(sample_products):
        print(f"\n--- Product {i+1} ---")
        
        # Show original
        print(f"Original: {json.dumps(product, indent=2)[:500]}...")
        
        # Process
        processed = processor.process_product(product)
        
        # Show results
        print(f"\nProcessed ID: {processed['id']}")
        print(f"Universal Fields: {json.dumps(processed['universal_fields'], indent=2)}")
        print(f"\nEnhanced Descriptor ({len(processed['enhanced_descriptor'].split())} words):")
        print(f"{processed['enhanced_descriptor'][:200]}...")
        print(f"\nVoice Summary: {processed['voice_summary']}")
        print(f"\nSearch Keywords: {', '.join(processed['search_keywords'][:10])}")
        print(f"\nKey Selling Points:")
        for point in processed['key_selling_points']:
            print(f"  • {point}")
        print(f"\nFilters: {json.dumps(processed['filter_metadata'], indent=2)}")


def preview_filters(brand_domain: str, products: List[Dict[str, Any]]):
    """Preview filter extraction."""
    
    print("\n" + "="*60)
    print("PREVIEW: Filter Extraction")
    print("="*60)
    
    analyzer = CatalogFilterAnalyzer(brand_domain)
    filters = analyzer.analyze_product_catalog(products)
    
    # Show filter summary
    for filter_name, filter_config in filters.items():
        if filter_name.startswith('_'):
            continue
        
        filter_type = filter_config.get('type', 'unknown')
        print(f"\n{filter_name} ({filter_type}):")
        
        if filter_type == 'categorical':
            values = filter_config.get('values', [])
            print(f"  Categories: {len(values)}")
            print(f"  Sample: {', '.join(values[:5])}")
            
        elif filter_type == 'numeric_range':
            print(f"  Range: {filter_config.get('min')} to {filter_config.get('max')}")
            
        elif filter_type == 'multi_select':
            values = filter_config.get('values', [])
            print(f"  Options: {len(values)}")
            print(f"  Top 5: {', '.join(values[:5])}")


async def test_search(index_name: str, namespace: str, test_queries: List[str]):
    """Test search functionality after ingestion."""
    
    print("\n" + "="*60)
    print("TESTING: Search Functionality")
    print("="*60)
    
    from pinecone import Pinecone
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pc.Index(index_name)
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        
        try:
            # Search with hybrid approach
            results = index.search_records(
                namespace=namespace,
                query={
                    "text": query
                },
                top_k=3
            )
            
            if results and hasattr(results, 'result') and results.result.hits:
                for i, hit in enumerate(results.result.hits[:3]):
                    metadata = json.loads(hit.fields.get('metadata', '{}'))
                    print(f"\n  {i+1}. {metadata.get('name', 'Unknown')}")
                    print(f"     Score: {hit._score:.3f}")
                    print(f"     Brand: {metadata.get('brand', 'N/A')}")
                    print(f"     Price: ${metadata.get('price', 0):.2f}")
                    print(f"     Category: {metadata.get('category', 'N/A')}")
            else:
                print("  No results found")
                
        except Exception as e:
            print(f"  Error: {e}")


def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(
        description="Enhanced product catalog ingestion for Pinecone RAG"
    )
    
    # Required arguments
    parser.add_argument(
        "brand_domain",
        help="Brand domain (e.g., specialized.com, balenciaga.com)"
    )
    parser.add_argument(
        "catalog_file",
        help="Path to product catalog JSON file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--index",
        required=True,
        help="Pinecone index name"
    )
    parser.add_argument(
        "--namespace",
        default="products",
        help="Pinecone namespace (default: products)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force update all products"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview processing without ingesting"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test queries after ingestion"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=45,
        help="Batch size for ingestion (default: 45)"
    )
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.environ.get("PINECONE_API_KEY"):
        print("❌ Error: PINECONE_API_KEY environment variable not set")
        sys.exit(1)
    
    print(f"🚀 Enhanced Product Ingestion")
    print(f"Brand: {args.brand_domain}")
    print(f"Catalog: {args.catalog_file}")
    print(f"Index: {args.index}")
    print(f"Namespace: {args.namespace}")
    print("-" * 60)
    
    try:
        # Load products
        products = load_product_catalog(args.catalog_file)
        
        if args.preview:
            # Preview mode
            preview_processing(args.brand_domain, products)
            preview_filters(args.brand_domain, products)
            
            print("\n✅ Preview complete. Use without --preview to ingest.")
            return
        
        # Initialize ingestion system
        ingestion = PineconeIngestion(
            brand_domain=args.brand_domain,
            index_name=args.index,
            namespace=args.namespace,
            batch_size=args.batch_size
        )
        
        # Show current index stats
        print("\n📊 Current Index Stats:")
        stats = ingestion.get_index_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Ingest products
        print("\n🔄 Starting ingestion...")
        results = ingestion.ingest_products(
            products=products,
            force_update=args.force,
            update_prompts=True
        )
        
        # Show results
        print("\n📈 Ingestion Results:")
        print(f"  Added: {results.get('added', 0)}")
        print(f"  Updated: {results.get('updated', 0)}")
        print(f"  Deleted: {results.get('deleted', 0)}")
        print(f"  Errors: {results.get('errors', 0)}")
        print(f"  Duration: {results.get('duration', 0):.2f}s")
        
        # Test search if requested
        if args.test:
            # Generate test queries based on brand
            if "specialized" in args.brand_domain.lower():
                test_queries = [
                    "carbon road bike under 3000",
                    "mountain bike for beginners",
                    "cycling helmet with MIPS"
                ]
            elif "balenciaga" in args.brand_domain.lower():
                test_queries = [
                    "leather handbag",
                    "designer sneakers",
                    "luxury accessories"
                ]
            else:
                # Generic queries
                test_queries = [
                    "best selling products",
                    "premium items",
                    "new arrivals"
                ]
            
            # Wait a bit for indexing
            print("\n⏳ Waiting for indexing...")
            import time
            time.sleep(5)
            
            # Run tests
            asyncio.run(test_search(args.index, args.namespace, test_queries))
        
        print("\n✅ Ingestion complete!")
        
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()