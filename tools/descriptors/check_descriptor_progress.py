#!/usr/bin/env python3
"""Check descriptor generation progress for specialized.com"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'packages'))

from liddy.models.product_manager import get_product_manager


async def main():
    brand = "specialized.com"
    pm = await get_product_manager(brand)
    products = await pm.get_products()
    
    total = len(products)
    with_desc = sum(1 for p in products if p.descriptor and len(p.descriptor.strip()) > 0)
    high_quality = sum(1 for p in products 
                      if p.descriptor and len(p.descriptor.strip()) > 0 
                      and hasattr(p, 'descriptor_metadata') 
                      and p.descriptor_metadata 
                      and hasattr(p.descriptor_metadata, 'quality_score')
                      and p.descriptor_metadata.quality_score >= 0.8)
    
    print(f"\n=== Descriptor Progress for {brand} ===")
    print(f"Total products: {total}")
    print(f"With descriptors: {with_desc} ({with_desc/total*100:.1f}%)")
    print(f"Without descriptors: {total - with_desc} ({(total - with_desc)/total*100:.1f}%)")
    print(f"High quality (≥0.8): {high_quality} ({high_quality/total*100:.1f}%)")
    print(f"Low quality (<0.8): {with_desc - high_quality} ({(with_desc - high_quality)/total*100:.1f}%)")
    
    # Sample quality scores
    quality_scores = []
    for p in products:
        if (p.descriptor and hasattr(p, 'descriptor_metadata') 
            and p.descriptor_metadata 
            and hasattr(p.descriptor_metadata, 'quality_score')):
            quality_scores.append(p.descriptor_metadata.quality_score)
    
    if quality_scores:
        avg_quality = sum(quality_scores) / len(quality_scores)
        print(f"\nAverage quality score: {avg_quality:.2f}")
        print(f"Min quality: {min(quality_scores):.2f}")
        print(f"Max quality: {max(quality_scores):.2f}")


if __name__ == "__main__":
    asyncio.run(main())