#!/usr/bin/env python3
"""
Test Hybrid Search Implementation

This script tests the hybrid search functionality with various query types
to demonstrate the benefits of combining dense and sparse embeddings.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any

from src.search.hybrid_search import HybridSearchEngine, HybridQueryOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_search_scenarios(engine: HybridSearchEngine, optimizer: HybridQueryOptimizer):
    """Test various search scenarios to demonstrate hybrid search benefits."""
    
    test_scenarios = [
        {
            "name": "Exact Brand/Model Search",
            "query": "Specialized Tarmac SL7",
            "description": "Should heavily favor sparse embeddings for exact matching"
        },
        {
            "name": "Category Browse",
            "query": "comfortable road bikes for long rides",
            "description": "Should favor dense embeddings for semantic understanding"
        },
        {
            "name": "Price-Constrained Search",
            "query": "carbon bikes under $3000",
            "description": "Should use filters and balanced embeddings"
        },
        {
            "name": "Technical Specification Search",
            "query": "bikes with Shimano Di2 electronic shifting",
            "description": "Should use sparse for technical terms"
        },
        {
            "name": "Natural Language Query",
            "query": "I need a bike for my daily commute that's fast but comfortable",
            "description": "Should heavily favor dense embeddings"
        },
        {
            "name": "Mixed Intent Query",
            "query": "Specialized mountain bikes under 25 pounds",
            "description": "Should balance dense and sparse with filters"
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario['name']}")
        print(f"Query: {scenario['query']}")
        print(f"Description: {scenario['description']}")
        print(f"{'='*70}")
        
        # Optimize query
        optimization = optimizer.optimize_query(scenario['query'])
        print(f"\nQuery Optimization:")
        print(f"  Strategy: {optimization['search_strategy']}")
        print(f"  Dense Weight: {optimization.get('dense_weight', 'auto')}")
        print(f"  Sparse Weight: {optimization.get('sparse_weight', 'auto')}")
        if optimization['filters']:
            print(f"  Filters: {json.dumps(optimization['filters'], indent=4)}")
        
        # Execute search
        results = engine.search(
            query=optimization['optimized_query'],
            top_k=5,
            filters=optimization['filters'],
            dense_weight=optimization.get('dense_weight'),
            sparse_weight=optimization.get('sparse_weight')
        )
        
        # Display results
        print(f"\nSearch Results ({len(results)} found):")
        for i, result in enumerate(results, 1):
            print(f"\n  {i}. {result.metadata.get('name', 'Unknown Product')}")
            print(f"     Brand: {result.metadata.get('brand', 'N/A')}")
            print(f"     Price: ${result.metadata.get('price', 0):.2f}")
            print(f"     Score: {result.score:.4f}")
            if result.metadata.get('category'):
                print(f"     Category: {result.metadata.get('category')}")
            if result.debug_info:
                print(f"     Debug: Dense={result.debug_info.get('dense_score', 'N/A')}, "
                      f"Sparse={result.debug_info.get('sparse_score', 'N/A')}")


def compare_search_methods(engine: HybridSearchEngine):
    """Compare hybrid search with dense-only and sparse-only approaches."""
    
    print("\n" + "="*70)
    print("COMPARISON: Hybrid vs Dense-Only vs Sparse-Only")
    print("="*70)
    
    test_queries = [
        "Specialized Roubaix Expert",  # Exact product
        "comfortable endurance road bike",  # Semantic query
        "bikes with carbon frame under $4000",  # Mixed query
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 50)
        
        # Dense-only search
        print("\nDense-Only Results:")
        dense_results = engine.search(query, top_k=3, dense_weight=1.0, sparse_weight=0.0)
        for i, r in enumerate(dense_results, 1):
            print(f"  {i}. {r.metadata.get('name', 'Unknown')} (Score: {r.score:.4f})")
        
        # Sparse-only search
        print("\nSparse-Only Results:")
        sparse_results = engine.search(query, top_k=3, dense_weight=0.0, sparse_weight=1.0)
        for i, r in enumerate(sparse_results, 1):
            print(f"  {i}. {r.metadata.get('name', 'Unknown')} (Score: {r.score:.4f})")
        
        # Hybrid search
        print("\nHybrid Results (Auto-weighted):")
        hybrid_results = engine.search(query, top_k=3)
        for i, r in enumerate(hybrid_results, 1):
            print(f"  {i}. {r.metadata.get('name', 'Unknown')} (Score: {r.score:.4f})")


def test_filter_combinations(engine: HybridSearchEngine):
    """Test search with various filter combinations."""
    
    print("\n" + "="*70)
    print("FILTER COMBINATION TESTS")
    print("="*70)
    
    filter_tests = [
        {
            "query": "road bikes",
            "filters": {"category": "road"},
            "description": "Category filter only"
        },
        {
            "query": "bikes",
            "filters": {"price": {"max": 3000}},
            "description": "Price range filter"
        },
        {
            "query": "performance bikes",
            "filters": {
                "category": "road",
                "price": {"min": 2000, "max": 5000}
            },
            "description": "Multiple filters"
        },
        {
            "query": "carbon bikes",
            "filters": {
                "material": ["carbon", "carbon fiber"],
                "category": {"$in": ["road", "mountain"]}
            },
            "description": "Complex filters with OR conditions"
        }
    ]
    
    for test in filter_tests:
        print(f"\n{test['description']}:")
        print(f"Query: '{test['query']}'")
        print(f"Filters: {json.dumps(test['filters'], indent=2)}")
        
        results = engine.search(
            query=test['query'],
            filters=test['filters'],
            top_k=3
        )
        
        print(f"Results: {len(results)} products found")
        for i, r in enumerate(results[:3], 1):
            print(f"  {i}. {r.metadata.get('name', 'Unknown')} - "
                  f"${r.metadata.get('price', 0):.2f}")


def main():
    """Main execution function."""
    
    import sys
    import os
    
    if len(sys.argv) != 3:
        print("Usage: python test_hybrid_search.py <brand_domain> <index_name>")
        print("Example: python test_hybrid_search.py specialized.com specialized-llama-2048")
        sys.exit(1)
    
    brand_domain = sys.argv[1]
    index_name = sys.argv[2]
    
    # Check for API key
    if not os.environ.get("PINECONE_API_KEY"):
        print("❌ Error: PINECONE_API_KEY environment variable not set")
        sys.exit(1)
    
    print(f"🚀 Testing Hybrid Search for {brand_domain}")
    print(f"Index: {index_name}")
    print("-" * 70)
    
    # Initialize components
    engine = HybridSearchEngine(brand_domain, index_name)
    optimizer = HybridQueryOptimizer(brand_domain)
    
    try:
        # Run tests
        test_search_scenarios(engine, optimizer)
        compare_search_methods(engine)
        test_filter_combinations(engine)
        
        print("\n✅ All tests complete!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()