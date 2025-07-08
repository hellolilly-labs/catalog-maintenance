#!/usr/bin/env python3
"""
Test the separate index ingestion with the new STT Vocabulary Extractor V2
"""

import asyncio
import sys
import os
sys.path.append('/Users/collinbrown/Dropbox/Development/Lily/code/catalog-maintenance')

# Set environment variables
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/collinbrown/Dropbox/Development/Lily/code/catalog-maintenance/auth/laure-430512-218943a78475.json'

from packages.liddy_intelligence.ingestion.scripts.ingest_catalog import main as ingest_main


async def test_ingestion():
    """Run ingestion for specialized.com to test the new STT extractor"""
    # This will use the new STTVocabularyExtractorV2
    await ingest_main(
        brand_domain="specialized.com",
        preview=False,  # Actually run the ingestion
        descriptor_model="gpt-4o",
        descriptor_regenerate=False,  # Don't regenerate existing descriptors
        max_products=50  # Test with first 50 products
    )


if __name__ == "__main__":
    print("Testing ingestion with STT Vocabulary Extractor V2...")
    print("=" * 60)
    asyncio.run(test_ingestion())