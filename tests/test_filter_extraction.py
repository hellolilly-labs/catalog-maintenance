#!/usr/bin/env python3
"""
Test Advanced Filter Extraction for Query Optimization

Demonstrates how natural language queries are converted to structured filters
for more precise RAG searches.
"""

import asyncio
import json
from src.agents.query_optimization_agent import QueryOptimizationAgent


async def test_filter_extraction():
    """Test filter extraction from various natural language queries"""
    
    print("🔍 Testing Advanced Filter Extraction for Query Optimization")
    print("=" * 70)
    
    # Initialize agent with Specialized.com catalog filters
    agent = QueryOptimizationAgent("specialized.com")
    
    # Test queries with expected filter extractions
    test_queries = [
        {
            "query": "I need a carbon road bike under 3000 for racing",
            "expected_filters": {
                "category": "road",
                "frame_material": "carbon",
                "price": [0, 3000],
                "intended_use": ["racing"]
            }
        },
        {
            "query": "Looking for a women's mountain bike with disc brakes",
            "expected_filters": {
                "category": "mountain",
                "gender": "womens", 
                "features": ["disc_brakes"]
            }
        },
        {
            "query": "Need an e-bike for commuting, budget is 2k to 4k",
            "expected_filters": {
                "category": "electric",
                "intended_use": ["commuting"],
                "price": [2000, 4000]
            }
        },
        {
            "query": "Show me lightweight gravel bikes with electronic shifting",
            "expected_filters": {
                "category": "gravel",
                "weight": [5, 8],  # "lightweight" range
                "features": ["electronic_shifting"]
            }
        },
        {
            "query": "Budget mountain bike for beginners",
            "expected_filters": {
                "category": "mountain",
                "price": [0, 1500]  # "budget" range
            }
        },
        {
            "query": "Premium Tarmac with Di2",
            "expected_filters": {
                "product_line": "Tarmac",
                "price": [6000, 15000],  # "premium" range
                "features": ["electronic_shifting"]  # Di2 is electronic
            }
        }
    ]
    
    print(f"\n📊 Testing {len(test_queries)} query examples:")
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        expected = test_case["expected_filters"]
        
        print(f"\n{i}. Query: \"{query}\"")
        print("-" * 50)
        
        # Extract filters
        result = await agent.optimize_product_query(
            original_query=query,
            context={"recent_messages": [], "expressed_interests": []},
            user_state=None
        )
        
        extracted_filters = result.get("filters", {})
        
        print(f"🔍 Extracted Filters:")
        for filter_name, filter_value in extracted_filters.items():
            print(f"   {filter_name}: {filter_value}")
        
        print(f"\n✅ Expected Filters:")
        for filter_name, filter_value in expected.items():
            print(f"   {filter_name}: {filter_value}")
        
        # Validate extraction accuracy
        print(f"\n📋 Validation:")
        all_correct = True
        for exp_filter, exp_value in expected.items():
            extracted_value = extracted_filters.get(exp_filter)
            if extracted_value == exp_value:
                print(f"   ✅ {exp_filter}: Correct")
            else:
                print(f"   ❌ {exp_filter}: Expected {exp_value}, got {extracted_value}")
                all_correct = False
        
        # Check for extra filters
        extra_filters = set(extracted_filters.keys()) - set(expected.keys())
        if extra_filters:
            print(f"   ℹ️  Extra filters found: {list(extra_filters)}")
        
        if all_correct:
            print("   🎯 Perfect extraction!")
        
        print(f"\n🎯 Optimized Query: \"{result.get('optimized_query', query)}\"")
        print(f"🔮 Confidence: {result.get('confidence', 0):.2f}")


async def test_filter_enhancement_with_user_state():
    """Test how user state enhances filter extraction"""
    
    print("\n\n🧠 Testing Filter Enhancement with User State")
    print("=" * 70)
    
    agent = QueryOptimizationAgent("specialized.com")
    
    # Simulate user state from previous conversation turns
    user_state = {
        "budget_range": [2000, 5000],
        "preferred_categories": ["road", "gravel"],
        "interested_features": ["disc_brakes", "tubeless_ready"],
        "mentioned_use_cases": ["commuting", "fitness"]
    }
    
    # Test query that would benefit from user state
    query = "Show me some good options"  # Very vague query
    
    print(f"📝 Vague Query: \"{query}\"")
    print(f"🧠 User State: {json.dumps(user_state, indent=2)}")
    
    result = await agent.optimize_product_query(
        original_query=query,
        context={"recent_messages": [], "expressed_interests": ["road bikes"]},
        user_state=user_state
    )
    
    extracted_filters = result.get("filters", {})
    
    print(f"\n🔍 Enhanced Filters (using user state):")
    for filter_name, filter_value in extracted_filters.items():
        print(f"   {filter_name}: {filter_value}")
    
    print(f"\n🎯 Enhanced Query: \"{result.get('optimized_query', query)}\"")
    print(f"💡 Follow-up Questions: {result.get('follow_up_questions', [])}")


async def test_catalog_filter_coverage():
    """Test coverage of all catalog filter types"""
    
    print("\n\n📚 Testing Catalog Filter Coverage")
    print("=" * 70)
    
    agent = QueryOptimizationAgent("specialized.com")
    
    print("🏗️ Available Filter Types in Catalog:")
    for filter_name, filter_config in agent.catalog_filters.items():
        filter_type = filter_config.get("type")
        
        if filter_type == "categorical":
            values = filter_config.get("values", [])
            print(f"   📂 {filter_name} ({filter_type}): {len(values)} options")
            print(f"      Values: {values[:5]}{'...' if len(values) > 5 else ''}")
            
        elif filter_type == "numeric_range":
            min_val = filter_config.get("min")
            max_val = filter_config.get("max")
            print(f"   📊 {filter_name} ({filter_type}): {min_val} to {max_val}")
            
        elif filter_type == "multi_select":
            values = filter_config.get("values", [])
            print(f"   ☑️  {filter_name} ({filter_type}): {len(values)} options")
    
    # Test a complex query that uses multiple filter types
    complex_query = "I want a carbon Tarmac road bike with electronic shifting and disc brakes under 5k for racing"
    
    print(f"\n🔬 Complex Query Test:")
    print(f"Query: \"{complex_query}\"")
    
    result = await agent.optimize_product_query(
        original_query=complex_query,
        context={},
        user_state=None
    )
    
    extracted_filters = result.get("filters", {})
    
    print(f"\n🎯 All Extracted Filters:")
    for filter_name, filter_value in extracted_filters.items():
        filter_config = agent.catalog_filters.get(filter_name, {})
        filter_type = filter_config.get("type", "unknown")
        print(f"   {filter_name} ({filter_type}): {filter_value}")
    
    print(f"\n📈 Filter Types Used: {len(set(agent.catalog_filters[f].get('type') for f in extracted_filters.keys()))}")
    print(f"📊 Total Filters Extracted: {len(extracted_filters)}")


async def main():
    """Run all filter extraction tests"""
    
    await test_filter_extraction()
    await test_filter_enhancement_with_user_state()
    await test_catalog_filter_coverage()
    
    print("\n\n🎉 All Filter Extraction Tests Complete!")
    print("\n📋 Key Benefits:")
    print("✅ Natural language → Structured filters")
    print("✅ Brand-specific terminology mapping")
    print("✅ Price range extraction (including 'k' notation)")
    print("✅ Multi-select feature detection")
    print("✅ User state enhancement")
    print("✅ Comprehensive catalog coverage")
    print("\n🚀 Result: Much more precise RAG queries!")


if __name__ == "__main__":
    asyncio.run(main())