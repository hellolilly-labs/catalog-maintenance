#!/usr/bin/env python3
"""
Test script for STT vocabulary extraction
"""

import asyncio
import json
import logging
from pathlib import Path

from src.ingestion.stt_vocabulary_extractor import STTVocabularyExtractor
from src.models.product_manager import get_product_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_stt_vocabulary(brand_domain: str = "specialized.com"):
    """Test STT vocabulary extraction for a brand."""
    
    logger.info(f"Testing STT vocabulary extraction for {brand_domain}")
    
    # Initialize product manager and load products
    product_manager = await get_product_manager(brand_domain)
    products = await product_manager.get_products()
    
    if not products:
        logger.error(f"No products found for {brand_domain}")
        return
    
    logger.info(f"Loaded {len(products)} products")
    
    # Initialize STT vocabulary extractor
    extractor = STTVocabularyExtractor(brand_domain)
    
    # Extract vocabulary
    extractor.extract_from_catalog(products)
    
    # Get statistics
    stats = extractor.get_stats()
    
    # Display results
    print("\n" + "="*60)
    print(f"STT Vocabulary Extraction Results for {brand_domain}")
    print("="*60)
    print(f"Total unique terms: {stats['term_count']}")
    print(f"Character count: {stats['character_count']}/{extractor.MAX_VOCAB_SIZE}")
    print(f"Coverage: {stats['coverage_percentage']:.1f}%")
    print(f"\nTop 20 terms by frequency:")
    for i, term in enumerate(stats['top_terms'], 1):
        print(f"  {i:2d}. {term}")
    
    # Show some example extractions
    print(f"\n\nSample product vocabulary extraction:")
    sample_product = products[0]
    print(f"Product: {sample_product.name}")
    print(f"Brand: {sample_product.brand}")
    if sample_product.search_keywords:
        print(f"Search keywords: {', '.join(sample_product.search_keywords[:5])}")
    
    # Save vocabulary
    success = await extractor.save_vocabulary()
    if success:
        print(f"\n✅ Vocabulary saved successfully")
        
        # Show what was saved
        vocab_data = await extractor.load_vocabulary()
        print(f"\nVocabulary preview (first 500 chars):")
        print(vocab_data.get('vocabulary_string', '')[:500] + "...")
    else:
        print(f"\n❌ Failed to save vocabulary")
    
    # Example of how to use with AssemblyAI
    print("\n\nExample AssemblyAI configuration:")
    print("```python")
    print("config = {")
    print(f'    "word_boost": {json.dumps(stats["top_terms"][:20])},')
    print('    "boost_param": "high"')
    print("}")
    print("```")


async def compare_brands():
    """Compare vocabulary extraction across different brands."""
    brands = ["specialized.com", "balenciaga.com", "sundayriley.com"]
    
    for brand in brands:
        try:
            await test_stt_vocabulary(brand)
            print("\n" + "="*80 + "\n")
        except Exception as e:
            logger.error(f"Failed to process {brand}: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        brand_domain = sys.argv[1]
        asyncio.run(test_stt_vocabulary(brand_domain))
    else:
        # Test with default brand
        asyncio.run(test_stt_vocabulary("specialized.com"))