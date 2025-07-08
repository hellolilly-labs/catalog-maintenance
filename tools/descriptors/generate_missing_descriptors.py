#!/usr/bin/env python3
"""
Generate missing descriptors for specialized.com products.

This script specifically targets products without descriptors and processes them
in batches to avoid API rate limits and ensure completion.
"""

import asyncio
import sys
import os
import logging
from typing import List

# Add packages to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'packages'))

from liddy.models.product import Product
from liddy.models.product_manager import get_product_manager
from liddy_intelligence.catalog.unified_descriptor_generator import UnifiedDescriptorGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def process_batch(generator: UnifiedDescriptorGenerator, products: List[Product], batch_num: int, total_batches: int):
    """Process a batch of products"""
    logger.info(f"\n=== Processing batch {batch_num}/{total_batches} ({len(products)} products) ===")
    
    for i, product in enumerate(products):
        logger.info(f"  [{i+1}/{len(products)}] Processing: {product.name}")
        try:
            await generator._generate_descriptor(product)
            logger.info(f"    ✅ Generated descriptor (quality: {product.descriptor_metadata.quality_score:.2f})")
        except Exception as e:
            logger.error(f"    ❌ Failed: {str(e)}")
    
    return products


async def main():
    brand = "specialized.com"
    batch_size = 50  # Process in smaller batches
    save_frequency = 25  # Save every 25 products
    
    # Get product manager
    logger.info(f"Loading products for {brand}...")
    pm = await get_product_manager(brand)
    all_products = await pm.get_products()
    
    # Identify products without descriptors
    products_without_descriptors = [
        p for p in all_products 
        if not p.descriptor or len(p.descriptor.strip()) == 0
    ]
    
    total = len(all_products)
    missing = len(products_without_descriptors)
    
    logger.info(f"\nCurrent state:")
    logger.info(f"  Total products: {total}")
    logger.info(f"  With descriptors: {total - missing} ({(total - missing)/total*100:.1f}%)")
    logger.info(f"  Without descriptors: {missing} ({missing/total*100:.1f}%)")
    
    if missing == 0:
        logger.info("All products already have descriptors!")
        return
    
    # Initialize generator
    logger.info(f"\nInitializing descriptor generator...")
    generator = UnifiedDescriptorGenerator(brand)
    
    # Load research if enabled
    if generator.config.use_research:
        logger.info("Loading product catalog intelligence...")
        generator.product_catalog_intelligence = await generator._load_product_catalog_intelligence()
    
    # Process in batches
    total_batches = (missing + batch_size - 1) // batch_size
    logger.info(f"\nProcessing {missing} products in {total_batches} batches of {batch_size}")
    
    processed_count = 0
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, missing)
        batch = products_without_descriptors[start_idx:end_idx]
        
        # Process batch
        try:
            await process_batch(generator, batch, batch_num + 1, total_batches)
            processed_count += len(batch)
            
            # Save progress
            if processed_count % save_frequency == 0 or batch_num == total_batches - 1:
                logger.info(f"\n💾 Saving progress ({processed_count} products processed)...")
                success = await pm.save_products(all_products)
                if success:
                    logger.info("  ✅ Saved successfully")
                else:
                    logger.error("  ❌ Save failed!")
            
            # Brief pause between batches to avoid rate limits
            if batch_num < total_batches - 1:
                logger.info("  ⏸️  Pausing briefly before next batch...")
                await asyncio.sleep(2)
                
        except Exception as e:
            logger.error(f"\n❌ Batch {batch_num + 1} failed: {str(e)}")
            logger.info("Saving progress and continuing...")
            await pm.save_products(all_products)
            continue
    
    # Final save
    logger.info(f"\n💾 Final save...")
    await pm.save_products(all_products)
    
    # Final count
    logger.info(f"\n=== Completed Processing ===")
    pm = await get_product_manager(brand)
    final_products = await pm.get_products()
    final_with_desc = sum(1 for p in final_products if p.descriptor and len(p.descriptor.strip()) > 0)
    final_missing = total - final_with_desc
    
    logger.info(f"Final state:")
    logger.info(f"  Total products: {total}")
    logger.info(f"  With descriptors: {final_with_desc} ({final_with_desc/total*100:.1f}%)")
    logger.info(f"  Without descriptors: {final_missing} ({final_missing/total*100:.1f}%)")
    logger.info(f"  Newly generated: {final_with_desc - (total - missing)}")
    
    if final_missing > 0:
        logger.warning(f"\n⚠️  Still missing descriptors for {final_missing} products.")
        logger.warning("This might be due to API limits or errors. Try running again later.")


if __name__ == "__main__":
    asyncio.run(main())