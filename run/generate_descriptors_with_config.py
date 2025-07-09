#!/usr/bin/env python3
"""
Generate Descriptors with Configuration

This script generates product descriptors with configurable options including:
- Auto-run terminology research if missing
- Use existing brand research
- Extract filters
- Quality thresholds

Usage:
    python run/generate_descriptors_with_config.py specialized.com
    python run/generate_descriptors_with_config.py specialized.com --no-auto-research
    python run/generate_descriptors_with_config.py specialized.com --force-regenerate
"""

import os
import sys
import asyncio
import argparse
import logging

# Add parent directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'packages'))

from liddy_intelligence.catalog.unified_descriptor_generator import UnifiedDescriptorGenerator, DescriptorConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def generate_descriptors(
    account: str,
    auto_research: bool = True,
    force_regenerate: bool = False,
    limit: int = None
):
    """Generate descriptors with configuration"""
    logger.info(f"Generating descriptors for {account}")
    
    # Configure descriptor generation
    config = DescriptorConfig(
        use_research=True,  # Use brand research
        extract_filters=True,  # Extract filter labels
        cache_enabled=True,  # Cache results
        quality_threshold=0.8,  # Quality threshold
        descriptor_length=(100, 200),  # Min/max words
        max_search_terms=30,
        max_selling_points=5,
        auto_run_terminology_research=auto_research  # Control auto-run
    )
    
    # Print configuration
    print("\nConfiguration:")
    print(f"  Use Research: {config.use_research}")
    print(f"  Extract Filters: {config.extract_filters}")
    print(f"  Auto-run Terminology Research: {config.auto_run_terminology_research}")
    print(f"  Quality Threshold: {config.quality_threshold}")
    print(f"  Descriptor Length: {config.descriptor_length[0]}-{config.descriptor_length[1]} words")
    
    # Create generator
    generator = UnifiedDescriptorGenerator(account, config=config)
    
    try:
        # Process catalog
        products, filter_labels = await generator.process_catalog(
            force_regenerate=force_regenerate,
            limit=limit
        )
        
        print(f"\n✅ Generated descriptors for {len(products)} products")
        
        if filter_labels:
            print(f"\nExtracted Filter Labels:")
            for filter_type, labels in filter_labels.items():
                if filter_type != 'catalog_name' and isinstance(labels, dict):
                    print(f"  {filter_type}: {len(labels)} unique values")
        
        # Show sample product
        if products:
            sample = products[0]
            print(f"\nSample Product: {sample.name}")
            print(f"  Descriptor Length: {len(sample.descriptor.split()) if sample.descriptor else 0} words")
            print(f"  Search Keywords: {len(sample.search_keywords)} terms")
            if sample.descriptor:
                print(f"  Descriptor Preview: {sample.descriptor[:200]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"Descriptor generation failed: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description='Generate product descriptors with configuration')
    parser.add_argument('account', help='Account/brand domain (e.g., specialized.com)')
    parser.add_argument('--no-auto-research', action='store_true',
                       help='Do not automatically run terminology research if missing')
    parser.add_argument('--force-regenerate', action='store_true',
                       help='Force regeneration of all descriptors')
    parser.add_argument('--limit', type=int, help='Limit number of products to process')
    
    args = parser.parse_args()
    
    # Run descriptor generation
    success = asyncio.run(generate_descriptors(
        args.account,
        auto_research=not args.no_auto_research,
        force_regenerate=args.force_regenerate,
        limit=args.limit
    ))
    
    if success:
        print(f"\n✅ Descriptor generation complete for {args.account}")
        print(f"\nNext steps:")
        print(f"  1. Analyze price distribution:")
        print(f"     python run/analyze_price_distribution.py {args.account}")
        print(f"  2. Test price-based search:")
        print(f"     python run/test_price_enhancement.py {args.account} --test-search")
    else:
        print(f"\n❌ Descriptor generation failed for {args.account}")
        sys.exit(1)


if __name__ == "__main__":
    main()