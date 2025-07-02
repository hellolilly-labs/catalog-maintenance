#!/usr/bin/env python3
"""
Test script for catalog intelligence functionality with new SearchService.

This script tests:
1. Account manager catalog intelligence
2. Search with query enhancement
3. Filter extraction from natural language
4. Performance comparison

Usage:
    python test_catalog_labels_caching.py
"""

import asyncio
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

async def test_catalog_intelligence():
    """Test catalog intelligence system."""
    print("🧪 Testing Catalog Intelligence System")
    print("=" * 50)
    
    try:
        # Import the new components
        from src.search.search_service import SearchService
        from src.account_manager import get_account_manager
        
        # Test account
        test_account = "specialized.com"
        
        # Phase 1: Test account manager setup
        print(f"\n📋 Phase 1: Testing account manager for {test_account}")
        start_time = time.time()
        
        account_manager = await get_account_manager(test_account)
        setup_time = time.time() - start_time
        
        print(f"   ✅ Account manager initialized in {setup_time:.3f}s")
        
        # Get catalog intelligence
        catalog_intel = await account_manager.get_catalog_intelligence()
        print(f"   📊 Catalog intelligence loaded:")
        print(f"      - Categories: {len(catalog_intel.get('categories', []))}")
        print(f"      - Has brand insights: {'brand_insights' in catalog_intel}")
        print(f"      - Has product labels: {'product_labels' in catalog_intel}")
        
        # Phase 2: Test query enhancement
        print(f"\n🚀 Phase 2: Testing query enhancement")
        
        test_queries = [
            ("mountain bike", "Basic query"),
            ("best bike", "Generic query needing enhancement"),
            ("comfortable", "Attribute query")
        ]
        
        for query, description in test_queries:
            enhanced_results, metrics = await SearchService.search_products(
                query=query,
                account=test_account,
                top_k=3,
                enable_enhancement=True
            )
            
            print(f"\n   Query: '{query}' ({description})")
            print(f"   Enhanced to: '{metrics.enhanced_query}'")
            print(f"   Results: {metrics.total_results} found in {metrics.search_time:.3f}s")
            if metrics.enhancements_used:
                print(f"   Enhancements: {', '.join(metrics.enhancements_used)}")
        
        # Phase 3: Test filter extraction
        print(f"\n🎯 Phase 3: Testing filter extraction")
        
        filter_queries = [
            "mountain bike under $3000",
            "road bikes between $2000 and $5000",
            "electric bikes over $4000"
        ]
        
        for query in filter_queries:
            results, metrics = await SearchService.search_products(
                query=query,
                account=test_account,
                top_k=3,
                enable_filter_extraction=True
            )
            
            print(f"\n   Query: '{query}'")
            print(f"   Filters extracted: {metrics.filters_applied}")
            print(f"   Results: {metrics.total_results} products")
        
        # Phase 4: Test search modes
        print(f"\n🔍 Phase 4: Testing different search modes")
        
        modes = ["dense", "sparse", "hybrid"]
        query = "lightweight carbon frame"
        
        for mode in modes:
            results, metrics = await SearchService.search_products(
                query=query,
                account=test_account,
                top_k=3,
                search_mode=mode,
                enable_enhancement=False
            )
            
            print(f"\n   {mode.upper()} search:")
            print(f"   Time: {metrics.search_time:.3f}s")
            print(f"   Results: {metrics.total_results}")
            if results:
                print(f"   Top result: {results[0]['metadata'].get('name', 'Unknown')}")
        
        # Phase 5: Test knowledge search
        print(f"\n📚 Phase 5: Testing knowledge search")
        
        knowledge_queries = [
            "warranty policy",
            "maintenance tips",
            "shipping information"
        ]
        
        for query in knowledge_queries:
            try:
                results, metrics = await SearchService.search_knowledge(
                    query=query,
                    account=test_account,
                    top_k=2
                )
                
                print(f"\n   Query: '{query}'")
                print(f"   Results: {metrics.total_results} articles found")
                if results:
                    print(f"   Top result: {results[0]['metadata'].get('title', 'Unknown')}")
            except Exception as e:
                print(f"   ⚠️  Knowledge search not available: {e}")
        
        print(f"\n🎉 All tests completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you're running from the catalog-maintenance directory")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_performance_comparison():
    """Compare performance with and without enhancements."""
    print("\n" + "=" * 50)
    print("⚡ Testing Performance Comparison")
    print("=" * 50)
    
    try:
        from src.search.search_service import SearchService
        
        test_account = "specialized.com"
        queries = [
            "mountain bike",
            "road bike under $3000",
            "comfortable bike for long rides",
            "beginner friendly"
        ]
        
        # Test with enhancements
        print("\n📊 With Enhancements:")
        total_time_enhanced = 0
        for query in queries:
            results, metrics = await SearchService.search_products(
                query=query,
                account=test_account,
                top_k=5,
                enable_enhancement=True,
                enable_filter_extraction=True
            )
            total_time_enhanced += metrics.search_time
            print(f"   '{query}': {metrics.search_time:.3f}s ({len(results)} results)")
        
        # Test without enhancements
        print("\n📊 Without Enhancements:")
        total_time_plain = 0
        for query in queries:
            results, metrics = await SearchService.search_products(
                query=query,
                account=test_account,
                top_k=5,
                enable_enhancement=False,
                enable_filter_extraction=False
            )
            total_time_plain += metrics.search_time
            print(f"   '{query}': {metrics.search_time:.3f}s ({len(results)} results)")
        
        print(f"\n📈 Performance Summary:")
        print(f"   Average with enhancements: {total_time_enhanced/len(queries):.3f}s")
        print(f"   Average without: {total_time_plain/len(queries):.3f}s")
        print(f"   Overhead: {((total_time_enhanced/total_time_plain) - 1) * 100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    # Run catalog intelligence tests
    success1 = await test_catalog_intelligence()
    
    # Run performance comparison
    success2 = await test_performance_comparison()
    
    if success1 and success2:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)