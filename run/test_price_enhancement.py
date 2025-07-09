#!/usr/bin/env python3
"""
Test Price Enhancement System

This script tests the complete price enhancement system including:
1. Terminology research integration
2. Multi-modal price distribution handling
3. Category-specific pricing
4. Semantic phrase generation

Usage:
    python run/test_price_enhancement.py specialized.com
    python run/test_price_enhancement.py specialized.com --test-search
"""

import os
import sys
import asyncio
import argparse
import logging
from typing import List, Dict, Any

# Add parent directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'packages'))

from liddy.models.product import Product
from liddy.storage import get_account_storage_provider
from liddy_intelligence.catalog.price_statistics_analyzer import PriceStatisticsAnalyzer
from liddy_intelligence.catalog.price_descriptor_updater import PriceDescriptorUpdater

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a section header"""
    print("\n" + "="*60)
    print(title)
    print("="*60)


async def test_price_statistics(account: str, products: List[Product]) -> Dict[str, Any]:
    """Test the price statistics analyzer"""
    print_section("TESTING PRICE STATISTICS ANALYZER")
    
    # Load terminology research if available
    storage_provider = get_account_storage_provider()
    terminology_research = None
    
    try:
        research_path = f"research/industry_terminology/research.md"
        research_content = await storage_provider.read_file(research_path)
        print(f"✅ Found terminology research")
        # In production, parse the research properly
        terminology_research = {"found": True}
    except:
        print(f"⚠️  No terminology research found")
    
    # Analyze pricing
    stats = PriceStatisticsAnalyzer.analyze_catalog_pricing(products, terminology_research)
    
    # Print results
    overall = stats['overall']
    print(f"\nOverall Statistics:")
    print(f"  Products: {overall['count']}")
    print(f"  Multi-modal: {overall.get('is_multimodal', False)}")
    
    if overall.get('is_multimodal') and 'price_clusters' in overall:
        print(f"\nPrice Clusters Detected:")
        for i, cluster in enumerate(overall['price_clusters']):
            print(f"  Cluster {i+1}: ${cluster['min']:.2f} - ${cluster['max']:.2f} ({cluster['count']} products)")
    
    # Print category stats
    if stats['by_category']:
        print(f"\nCategory Statistics:")
        for category, cat_stats in list(stats['by_category'].items())[:3]:
            print(f"  {category}: ${cat_stats['min']:.2f} - ${cat_stats['max']:.2f} (mean: ${cat_stats['mean']:.2f})")
    
    # Print semantic phrases
    if stats.get('semantic_phrases'):
        print(f"\nGenerated Semantic Phrases:")
        for tier, phrases in stats['semantic_phrases'].items():
            if phrases:
                print(f"  {tier}: {', '.join(phrases[:3])}")
    
    return stats


async def test_price_updater(account: str, products: List[Product], stats: Dict[str, Any]):
    """Test the price descriptor updater"""
    print_section("TESTING PRICE DESCRIPTOR UPDATER")
    
    updater = PriceDescriptorUpdater(account)
    
    # Test a few products
    test_products = products[:3]
    
    for product in test_products:
        print(f"\nProduct: {product.name}")
        
        # Get current price
        price_str = product.salePrice or product.originalPrice
        if price_str:
            try:
                price = float(price_str.replace('$', '').replace(',', ''))
                print(f"  Price: {price_str}")
                
                # Test keyword generation
                keywords = updater.get_price_category_keywords(price, stats, product)
                print(f"  Generated Keywords: {', '.join(keywords[:5])}")
                
                # Check if descriptor has price
                has_price = updater.check_descriptor_has_price(product.descriptor, product)
                print(f"  Descriptor has price: {has_price}")
                
                # Test semantic context generation
                context = updater._generate_semantic_price_context(product, stats)
                if context:
                    print(f"  Semantic Context: {context[:100]}...")
                    
            except Exception as e:
                print(f"  Error: {e}")


async def test_search_queries(account: str):
    """Test price-based search queries"""
    print_section("TESTING PRICE-BASED SEARCH")
    
    from liddy.search.service import SearchService
    
    search_service = SearchService(account)
    
    test_queries = [
        "bikes under 3000",
        "premium gravel bikes",
        "budget mountain bikes",
        "bikes around $5000",
        "top of the line road bikes",
        "affordable ebikes",
        "bikes between 2000 and 4000"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        try:
            # Extract filters
            filters = search_service.extract_filters_from_query(query)
            if 'price' in filters:
                print(f"  Price filter: {filters['price']}")
            else:
                print(f"  No price filter extracted")
                
            # Perform search
            results = await search_service.search(query, limit=3)
            
            if results:
                print(f"  Found {len(results)} results:")
                for i, result in enumerate(results[:3]):
                    product = result['product']
                    price = product.salePrice or product.originalPrice
                    print(f"    {i+1}. {product.name} - {price}")
            else:
                print(f"  No results found")
                
        except Exception as e:
            print(f"  Error: {e}")


async def main():
    parser = argparse.ArgumentParser(description='Test price enhancement system')
    parser.add_argument('account', help='Account/brand domain (e.g., specialized.com)')
    parser.add_argument('--test-search', action='store_true', help='Test search queries')
    
    args = parser.parse_args()
    
    print(f"Testing price enhancement for {args.account}")
    
    # Load products
    storage_provider = get_account_storage_provider()
    
    try:
        products_data = await storage_provider.get_product_catalog(account=args.account)
        products = [Product(**p) for p in products_data]
        print(f"Loaded {len(products)} products")
        
        # Test price statistics
        stats = await test_price_statistics(args.account, products)
        
        # Test price updater
        await test_price_updater(args.account, products, stats)
        
        # Test search if requested
        if args.test_search:
            await test_search_queries(args.account)
        
        print_section("TEST COMPLETE")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()