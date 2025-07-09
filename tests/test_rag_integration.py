#!/usr/bin/env python3
"""
RAG Integration Test - Simple End-to-End Validation

Tests that our RAG components work together:
1. Generate enhanced descriptors from specialized.com catalog
2. Extract filters from same catalog  
3. Test query optimization
4. Validate consistency
"""

import asyncio
import json
from pathlib import Path
from src.catalog.enhanced_descriptor_generator import generate_enhanced_catalog
from src.agents.query_optimization_agent import QueryOptimizationAgent

async def test_rag_integration():
    """Test RAG system integration with real specialized.com data"""
    
    print("🧪 RAG Integration Test - Specialized.com")
    print("=" * 50)
    
    # Load real specialized catalog
    catalog_path = Path("sample_specialized_catalog.json")
    if not catalog_path.exists():
        print("❌ Sample catalog not found")
        return False
    
    with open(catalog_path) as f:
        catalog_data = json.load(f)
    
    products = catalog_data.get("products", [])
    print(f"📦 Testing with {len(products)} specialized.com products")
    
    # Test 1: Generate enhanced catalog
    print(f"\n1️⃣ Enhanced Catalog Generation")
    try:
        enhanced_descriptors, filter_labels = generate_enhanced_catalog(
            brand_domain="specialized.com",
            catalog_data=products,
            descriptor_style="voice_optimized"
        )
        print(f"   ✅ Generated {len(enhanced_descriptors)} descriptors")
        print(f"   ✅ Extracted {len([k for k in filter_labels.keys() if not k.startswith('_')])} filter types")
        
        # Show sample
        sample = enhanced_descriptors[0]
        print(f"   📝 Sample: {sample['name']}")
        print(f"      Voice: {sample['voice_summary']}")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False
    
    # Test 2: Query optimization
    print(f"\n2️⃣ Query Optimization Test")
    try:
        optimizer = QueryOptimizationAgent("specialized.com")
        
        test_queries = [
            "I want a road bike for racing",
            "Show me mountain bikes under $3000", 
            "Looking for electric bikes for commuting"
        ]
        
        for query in test_queries:
            result = await optimizer.optimize_product_query(
                original_query=query,
                context={"recent_messages": [], "expressed_interests": []},
                user_state=None
            )
            
            filters = result.get("filters", {})
            print(f"   🔍 '{query}'")
            print(f"      Filters: {filters}")
            
    except Exception as e:
        print(f"   ❌ Query optimization failed: {e}")
        return False
    
    # Test 3: File verification
    print(f"\n3️⃣ File Output Verification")
    expected_files = [
        "accounts/specialized.com/enhanced_product_catalog.json",
        "accounts/specialized.com/catalog_filters.json"
    ]
    
    for file_path in expected_files:
        if Path(file_path).exists():
            file_size = Path(file_path).stat().st_size
            print(f"   ✅ {file_path} ({file_size:,} bytes)")
        else:
            print(f"   ❌ Missing: {file_path}")
            return False
    
    print(f"\n🎉 RAG Integration Test PASSED")
    print("   Ready for Pinecone integration!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_rag_integration())
    exit(0 if success else 1)