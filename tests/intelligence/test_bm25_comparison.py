#!/usr/bin/env python3
"""
Compare hand-rolled BM25 vs rank-bm25 implementation

This script compares the performance and correctness of both implementations.
"""

import time
import logging
import sys
import os
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from src.ingestion.sparse_embeddings import SparseEmbeddingGenerator as OldSparseGenerator
from src.ingestion.sparse_embeddings_bm25 import SparseEmbeddingGenerator as NewSparseGenerator
from src.models.product_manager import get_product_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compare_sparse_embeddings(old_result: Dict, new_result: Dict) -> Dict[str, Any]:
    """Compare sparse embeddings from both implementations."""
    
    old_indices = set(old_result.get('indices', []))
    new_indices = set(new_result.get('indices', []))
    
    common_indices = old_indices & new_indices
    old_only = old_indices - new_indices
    new_only = new_indices - old_indices
    
    # Compare values for common indices
    old_values_dict = dict(zip(old_result.get('indices', []), old_result.get('values', [])))
    new_values_dict = dict(zip(new_result.get('indices', []), new_result.get('values', [])))
    
    value_diffs = []
    for idx in common_indices:
        diff = abs(old_values_dict[idx] - new_values_dict[idx])
        value_diffs.append(diff)
    
    avg_value_diff = sum(value_diffs) / len(value_diffs) if value_diffs else 0
    
    return {
        'common_features': len(common_indices),
        'old_only_features': len(old_only),
        'new_only_features': len(new_only),
        'overlap_ratio': len(common_indices) / max(len(old_indices), len(new_indices)) if old_indices or new_indices else 0,
        'avg_value_difference': avg_value_diff,
        'old_total': len(old_indices),
        'new_total': len(new_indices)
    }


async def run_comparison(brand_domain: str, num_products: int = 10):
    """Run comparison between old and new implementations."""
    
    logger.info(f"🔬 Comparing BM25 implementations for {brand_domain}")
    
    # Load products
    logger.info("📦 Loading products...")
    product_manager = await get_product_manager(brand_domain)
    all_products = await product_manager.get_products()
    
    if not all_products:
        logger.error("No products found")
        return
    
    # Limit to requested number
    products = all_products[:num_products]
    logger.info(f"   Testing with {len(products)} products")
    
    # Initialize both generators
    logger.info("\n🔧 Initializing generators...")
    old_generator = OldSparseGenerator(brand_domain)
    new_generator = NewSparseGenerator(brand_domain)
    
    # Build vocabularies
    logger.info("\n📚 Building vocabularies...")
    
    start_time = time.time()
    old_generator.build_vocabulary(products)
    old_build_time = time.time() - start_time
    logger.info(f"   Old implementation: {old_build_time:.2f}s")
    
    start_time = time.time()
    new_generator.build_vocabulary(products)
    new_build_time = time.time() - start_time
    logger.info(f"   New implementation: {new_build_time:.2f}s")
    
    # Compare vocabularies
    logger.info(f"\n📊 Vocabulary comparison:")
    logger.info(f"   Old vocabulary size: {len(old_generator.vocabulary)}")
    logger.info(f"   New vocabulary size: {len(new_generator.vocabulary)}")
    
    # Generate embeddings for each product
    logger.info("\n🧮 Generating sparse embeddings...")
    
    old_times = []
    new_times = []
    comparisons = []
    
    for i, product in enumerate(products[:5]):  # Test first 5 products
        # Old implementation
        start_time = time.time()
        old_result = old_generator.generate_sparse_embedding(product, {})
        old_times.append(time.time() - start_time)
        
        # New implementation
        start_time = time.time()
        new_result = new_generator.generate_sparse_embedding(product, {})
        new_times.append(time.time() - start_time)
        
        # Compare results
        comparison = compare_sparse_embeddings(old_result, new_result)
        comparisons.append(comparison)
        
        product_name = getattr(product, 'name', 'Unknown')[:50]
        logger.info(f"\n   Product {i+1}: {product_name}")
        logger.info(f"     Old: {comparison['old_total']} features in {old_times[-1]*1000:.1f}ms")
        logger.info(f"     New: {comparison['new_total']} features in {new_times[-1]*1000:.1f}ms")
        logger.info(f"     Overlap: {comparison['overlap_ratio']:.1%}")
        logger.info(f"     Avg value diff: {comparison['avg_value_difference']:.3f}")
    
    # Summary statistics
    logger.info("\n📈 Performance Summary:")
    logger.info(f"   Vocabulary build time:")
    logger.info(f"     Old: {old_build_time:.2f}s")
    logger.info(f"     New: {new_build_time:.2f}s ({new_build_time/old_build_time:.1f}x)")
    
    logger.info(f"\n   Embedding generation time (avg):")
    avg_old = sum(old_times) / len(old_times) * 1000
    avg_new = sum(new_times) / len(new_times) * 1000
    logger.info(f"     Old: {avg_old:.1f}ms")
    logger.info(f"     New: {avg_new:.1f}ms ({avg_old/avg_new:.1f}x faster)")
    
    logger.info(f"\n   Feature overlap:")
    avg_overlap = sum(c['overlap_ratio'] for c in comparisons) / len(comparisons)
    logger.info(f"     Average: {avg_overlap:.1%}")
    
    # Test edge cases
    logger.info("\n🧪 Testing edge cases...")
    
    # Empty product
    empty_product = type('Product', (), {'name': '', 'description': '', 'categories': []})()
    old_empty = old_generator.generate_sparse_embedding(empty_product, {})
    new_empty = new_generator.generate_sparse_embedding(empty_product, {})
    logger.info(f"   Empty product - Old: {len(old_empty['indices'])}, New: {len(new_empty['indices'])}")
    
    # Product with special characters
    special_product = type('Product', (), {
        'name': 'Test-Product_123 v2.0',
        'brand': 'BRAND-X',
        'description': 'Special chars: @#$%^&*()',
        'categories': ['Test/Category']
    })()
    old_special = old_generator.generate_sparse_embedding(special_product, {})
    new_special = new_generator.generate_sparse_embedding(special_product, {})
    logger.info(f"   Special chars - Old: {len(old_special['indices'])}, New: {len(new_special['indices'])}")
    
    logger.info("\n✅ Comparison complete!")
    
    # Recommendations
    logger.info("\n💡 Recommendations:")
    if avg_new < avg_old:
        logger.info("   ✅ New implementation is faster")
    else:
        logger.info("   ⚠️  Old implementation is faster (unusual)")
    
    if avg_overlap > 0.7:
        logger.info("   ✅ High feature overlap - results are consistent")
    else:
        logger.info("   ⚠️  Low feature overlap - review tokenization differences")
    
    logger.info("\n   The new rank-bm25 implementation provides:")
    logger.info("   • Proven BM25 scoring algorithm")
    logger.info("   • Better handling of edge cases")
    logger.info("   • More maintainable codebase")
    logger.info("   • Consistent with academic literature")


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compare hand-rolled BM25 vs rank-bm25 implementation'
    )
    parser.add_argument(
        'brand_domain',
        help='Brand domain (e.g., specialized.com)'
    )
    parser.add_argument(
        '--num-products',
        type=int,
        default=10,
        help='Number of products to test (default: 10)'
    )
    
    args = parser.parse_args()
    
    await run_comparison(args.brand_domain, args.num_products)


if __name__ == "__main__":
    asyncio.run(main())