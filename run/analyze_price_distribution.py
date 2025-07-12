#!/usr/bin/env python3
"""
Analyze Price Distribution

This script analyzes the price distribution of a catalog and provides
recommendations for handling complex pricing structures.

Usage:
    python run/analyze_price_distribution.py specialized.com
    python run/analyze_price_distribution.py specialized.com --update-descriptors
"""

import os
import sys
import asyncio
import argparse
import logging
import json

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


async def analyze_catalog_pricing(account: str, update_descriptors: bool = False):
    """Analyze catalog pricing distribution"""
    logger.info(f"Analyzing price distribution for {account}")
    
    storage_provider = get_account_storage_provider()
    
    try:
        # Load products
        products_data = await storage_provider.get_product_catalog(account=account)
        products = [Product(**p) for p in products_data]
        logger.info(f"Loaded {len(products)} products")
        
        # Check for terminology research
        terminology_research = None
        try:
            research_path = f"research/industry_terminology/research.md"
            research_content = await storage_provider.read_file(research_path)
            logger.info("Found industry terminology research")
            # Note: In production, you'd properly parse this
            terminology_research = {"found": True}
        except:
            logger.warning("No industry terminology research found")
        
        # Analyze pricing
        analysis = PriceStatisticsAnalyzer.analyze_catalog_pricing(products, terminology_research)
        
        # Print overall statistics
        overall = analysis['overall']
        print("\n" + "="*60)
        print("OVERALL PRICE STATISTICS")
        print("="*60)
        print(f"Total Products: {overall['count']}")
        print(f"Price Range: ${overall['min']:.2f} - ${overall['max']:.2f}")
        print(f"Mean: ${overall['mean']:.2f} (Std Dev: ${overall['std']:.2f})")
        print(f"\nPercentiles:")
        print(f"  5th: ${overall['p5']:.2f}")
        print(f" 25th: ${overall['p25']:.2f}")
        print(f" 50th: ${overall['p50']:.2f} (median)")
        print(f" 75th: ${overall['p75']:.2f}")
        print(f" 95th: ${overall['p95']:.2f}")
        
        # Check for multi-modal distribution
        if overall.get('is_multimodal'):
            print(f"\n⚠️  MULTI-MODAL DISTRIBUTION DETECTED")
            if 'price_clusters' in overall:
                print("\nPrice Clusters:")
                for i, cluster in enumerate(overall['price_clusters']):
                    print(f"  Cluster {i+1}: ${cluster['min']:.2f} - ${cluster['max']:.2f} "
                          f"(mean: ${cluster['mean']:.2f}, count: {cluster['count']})")
        
        # Print thresholds
        print(f"\nDynamic Price Thresholds:")
        print(f"  Budget: < ${overall.get('budget_threshold', 0):.2f}")
        print(f"  Mid-Low: ${overall.get('budget_threshold', 0):.2f} - ${overall.get('mid_low_threshold', 0):.2f}")
        print(f"  Mid-High: ${overall.get('mid_low_threshold', 0):.2f} - ${overall.get('mid_high_threshold', 0):.2f}")
        print(f"  Premium: > ${overall.get('premium_threshold', 0):.2f}")
        
        # Print category statistics
        if analysis['by_category']:
            print("\n" + "="*60)
            print("CATEGORY-SPECIFIC STATISTICS")
            print("="*60)
            for category, stats in analysis['by_category'].items():
                print(f"\n{category}:")
                print(f"  Count: {stats['count']} products")
                print(f"  Range: ${stats['min']:.2f} - ${stats['max']:.2f}")
                print(f"  Mean: ${stats['mean']:.2f} (Std Dev: ${stats['std']:.2f})")
                print(f"  Median: ${stats['p50']:.2f}")
        
        # Print semantic phrases
        if analysis.get('semantic_phrases'):
            print("\n" + "="*60)
            print("GENERATED SEMANTIC PHRASES")
            print("="*60)
            for tier, phrases in analysis['semantic_phrases'].items():
                if phrases:
                    print(f"\n{tier.upper()}:")
                    for phrase in phrases[:5]:
                        print(f"  - {phrase}")
        
        # Print recommendations
        if analysis['recommendations']:
            print("\n" + "="*60)
            print("RECOMMENDATIONS")
            print("="*60)
            
            if analysis['recommendations'].get('strategy'):
                print(f"\nStrategy: {analysis['recommendations']['strategy']}")
            
            if analysis['recommendations'].get('warnings'):
                print("\nWarnings:")
                for warning in analysis['recommendations']['warnings']:
                    print(f"  ⚠️  {warning}")
            
            if analysis['recommendations'].get('suggestions'):
                print("\nSuggestions:")
                for suggestion in analysis['recommendations']['suggestions']:
                    print(f"  • {suggestion}")
        
        # Update descriptors if requested
        if update_descriptors:
            print("\n" + "="*60)
            print("UPDATING DESCRIPTORS")
            print("="*60)
            
            updater = PriceDescriptorUpdater(account)
            update_stats = await updater.check_and_update_products(products)
            
            print(f"\nUpdate Results:")
            print(f"  Total Checked: {update_stats['total_checked']}")
            print(f"  Updated: {update_stats['updated_count']}")
            print(f"  Already Had Price: {update_stats['already_had_price']}")
            print(f"  Errors: {update_stats['errors']}")
        
        # Save analysis results
        analysis_path = f"analysis/price_distribution_{account}.json"
        # Remove the lambda function before saving
        if 'get_relevant_category' in analysis['overall']:
            del analysis['overall']['get_relevant_category']
        
        await storage_provider.write_json(analysis_path, analysis)
        print(f"\n✅ Analysis saved to: accounts/{account}/{analysis_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description='Analyze catalog price distribution')
    parser.add_argument('account', help='Account/brand domain (e.g., specialized.com)')
    parser.add_argument('--update-descriptors', action='store_true', 
                       help='Update product descriptors with price information')
    
    args = parser.parse_args()
    
    # Run the analysis
    success = asyncio.run(analyze_catalog_pricing(args.account, args.update_descriptors))
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()