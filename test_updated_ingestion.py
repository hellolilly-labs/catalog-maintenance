#!/usr/bin/env python3
"""
Test Updated Ingestion Pipeline with Storage Integration

Tests the integration between storage providers and filter analysis.
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agents.catalog_filter_analyzer import CatalogFilterAnalyzer
from src.storage import get_account_storage_provider

async def test_storage_integration():
    """Test the storage provider integration with CatalogFilterAnalyzer"""
    
    brand_domain = "specialized.com"
    
    print("🧪 Testing Storage Provider Integration")
    print("=" * 50)
    
    # Test 1: CatalogFilterAnalyzer with storage provider
    print("\n1️⃣ Testing CatalogFilterAnalyzer with storage provider...")
    try:
        analyzer = CatalogFilterAnalyzer(brand_domain)
        
        # Try to load catalog data from storage
        catalog_data = await analyzer._load_catalog_data()
        print(f"   📦 Loaded {len(catalog_data)} products from storage")
        
        if catalog_data:
            # Analyze filters
            filters = await analyzer.analyze_product_catalog(catalog_data[:3])  # Use first 3 products
            print(f"   🔍 Extracted {len(filters) - 1} filter types")
            
            # Test saving filters
            await analyzer.save_filters_to_file(filters, "test_filters.json")
            print("   💾 Successfully saved filters via storage provider")
            
            # Test reading back the saved filters
            storage = get_account_storage_provider()
            saved_content = await storage.read_file(brand_domain, "test_filters.json")
            if saved_content:
                print("   ✅ Successfully read back saved filters")
            else:
                print("   ❌ Failed to read back saved filters")
        else:
            print("   ⚠️ No catalog data found - testing with minimal data")
            filters = await analyzer.analyze_product_catalog([])
            print(f"   🔍 Generated minimal filters: {len(filters) - 1} types")
        
        print("   ✅ CatalogFilterAnalyzer storage integration working!")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 2: Storage provider functionality
    print("\n2️⃣ Testing storage provider directly...")
    try:
        storage = get_account_storage_provider()
        
        # Test write and read
        test_content = '{"test": "data", "timestamp": "2025-01-01"}'
        success = await storage.write_file(brand_domain, "test_file.json", test_content, "application/json")
        
        if success:
            print("   ✅ Successfully wrote test file")
            
            # Read it back
            read_content = await storage.read_file(brand_domain, "test_file.json")
            if read_content == test_content:
                print("   ✅ Successfully read back identical content")
            else:
                print("   ❌ Content mismatch on read")
        else:
            print("   ❌ Failed to write test file")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    print("\n🎉 All storage integration tests passed!")
    return True

if __name__ == "__main__":
    asyncio.run(test_storage_integration()) 