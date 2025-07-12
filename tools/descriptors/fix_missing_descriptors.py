#!/usr/bin/env python3
"""
Fix missing descriptors for specialized.com products.

This script identifies products without descriptors and generates them.
"""

import asyncio
import sys
import os
import logging

# Add packages to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'packages'))

from liddy.models.product_manager import get_product_manager
from liddy_intelligence.catalog.unified_descriptor_generator import UnifiedDescriptorGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    brand = "specialized.com"
    
    # Get product manager
    logger.info(f"Loading products for {brand}...")
    pm = await get_product_manager(brand)
    products = await pm.get_products()
    
    # Count current state
    total = len(products)
    with_desc = sum(1 for p in products if p.descriptor and len(p.descriptor.strip()) > 0)
    without_desc = total - with_desc
    
    logger.info(f"Current state:")
    logger.info(f"  Total products: {total}")
    logger.info(f"  With descriptors: {with_desc} ({with_desc/total*100:.1f}%)")
    logger.info(f"  Without descriptors: {without_desc} ({without_desc/total*100:.1f}%)")
    
    if without_desc == 0:
        logger.info("All products already have descriptors!")
        return
    
    # Generate missing descriptors
    logger.info(f"\nGenerating descriptors for {without_desc} products...")
    logger.info("This will take some time as it processes in batches...")
    
    # Use the unified descriptor generator
    generator = UnifiedDescriptorGenerator(brand)
    
    # Process all products (it will skip those with descriptors)
    enhanced_products, filter_labels = await generator.process_catalog(
        force_regenerate=False  # This will only generate for missing ones
    )
    
    logger.info(f"\nCompleted processing!")
    
    # Reload to check final state
    pm = await get_product_manager(brand)
    products = await pm.get_products()
    
    final_with_desc = sum(1 for p in products if p.descriptor and len(p.descriptor.strip()) > 0)
    final_without_desc = total - final_with_desc
    
    logger.info(f"\nFinal state:")
    logger.info(f"  Total products: {total}")
    logger.info(f"  With descriptors: {final_with_desc} ({final_with_desc/total*100:.1f}%)")
    logger.info(f"  Without descriptors: {final_without_desc} ({final_without_desc/total*100:.1f}%)")
    logger.info(f"  Newly generated: {final_with_desc - with_desc}")
    
    if final_without_desc > 0:
        logger.warning(f"\nStill missing descriptors for {final_without_desc} products.")
        logger.warning("This might be due to API limits or errors. Try running again later.")


if __name__ == "__main__":
    asyncio.run(main())