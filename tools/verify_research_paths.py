#!/usr/bin/env python3
"""
Verify research file paths are correct in GCS
"""
import sys
import os
import asyncio
import logging

# Add parent directory to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'packages'))

from liddy.storage import get_account_storage_provider
from liddy_intelligence.catalog.unified_descriptor_generator import UnifiedDescriptorGenerator
from liddy_intelligence.research.product_catalog_research import get_product_catalog_researcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def verify_research_paths(brand_domain: str):
    """Verify research paths are working correctly"""
    
    storage = get_account_storage_provider()
    
    # List research phases
    research_phases = [
        "foundation",
        "market_positioning",
        "product_style",
        "brand_style",
        "customer_cultural",
        "voice_messaging",
        "interview_synthesis",
        "linearity_analysis",
        "product_catalog"
    ]
    
    print(f"\n=== Checking research files for {brand_domain} ===\n")
    
    # Check GCS paths
    for phase in research_phases:
        new_path = f"accounts/{brand_domain}/research/{phase}/research.md"
        old_path = f"accounts/{brand_domain}/research_phases/{phase}_research.md"
        
        try:
            # Check new path
            if hasattr(storage, 'bucket'):
                new_exists = storage.bucket.blob(new_path).exists()
                old_exists = storage.bucket.blob(old_path).exists()
            else:
                # Local storage
                import os
                new_exists = os.path.exists(os.path.join(storage.base_dir, new_path))
                old_exists = os.path.exists(os.path.join(storage.base_dir, old_path))
                
            print(f"{phase}:")
            print(f"  New path ({new_path}): {'✅ EXISTS' if new_exists else '❌ MISSING'}")
            print(f"  Old path ({old_path}): {'exists' if old_exists else 'missing'}")
            
        except Exception as e:
            print(f"{phase}: ❌ Error - {e}")
    
    # Test product catalog researcher
    print(f"\n=== Testing ProductCatalogResearcher ===\n")
    try:
        researcher = get_product_catalog_researcher(brand_domain)
        cached_result = await researcher._load_cached_results()
        
        if cached_result:
            print(f"✅ Successfully loaded product catalog research")
            print(f"   Quality score: {cached_result.get('quality_score', 'unknown')}")
            print(f"   Content length: {len(cached_result.get('content', ''))} chars")
        else:
            print("❌ No product catalog research found")
            
    except Exception as e:
        print(f"❌ Error loading product catalog research: {e}")
    
    # Test unified descriptor generator
    print(f"\n=== Testing UnifiedDescriptorGenerator ===\n")
    try:
        generator = UnifiedDescriptorGenerator(brand_domain)
        intelligence = await generator._load_product_catalog_intelligence()
        
        if intelligence:
            print(f"✅ Successfully loaded product catalog intelligence")
            print(f"   Content length: {len(intelligence)} chars")
        else:
            print("❌ No product catalog intelligence found")
            
    except Exception as e:
        print(f"❌ Error loading product catalog intelligence: {e}")

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify research paths')
    parser.add_argument('brand', help='Brand domain to check')
    args = parser.parse_args()
    
    await verify_research_paths(args.brand)

if __name__ == "__main__":
    asyncio.run(main())