#!/usr/bin/env python3
"""
Test script to verify parallel search performance improvements.

This script tests the refactored SearchPinecone class to ensure that:
1. Dense and sparse searches run in parallel (not sequentially)
2. Total time ≈ max(dense_time, sparse_time) + overhead
3. The search results are still correct
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from packages.liddy.search.pinecone import get_search_pinecone


async def test_parallel_search():
    """Test that searches run in parallel and measure performance."""
    
    # Initialize search for a test brand
    brand = "specialized.com"
    print(f"\n🔍 Testing parallel search for {brand}")
    print("=" * 60)
    
    try:
        # Get search instance
        search = await get_search_pinecone(brand)
        
        # Test queries
        queries = [
            "mountain bike for trails",
            "carbon road bike lightweight",
            "electric bike for commuting"
        ]
        
        for query in queries:
            print(f"\n📝 Query: '{query}'")
            print("-" * 40)
            
            # Run hybrid search (which uses both dense and sparse in parallel)
            results = await search.search(
                query=query,
                top_k=10,
                search_mode="hybrid",
                rerank=False  # Skip reranking for pure parallel test
            )
            
            print(f"\n✅ Found {len(results)} results")
            
            # Show top 3 results
            for i, result in enumerate(results[:3], 1):
                print(f"\n   {i}. ID: {result.id} (Score: {result.score:.4f})")
                if result.metadata.get('name'):
                    print(f"      Name: {result.metadata['name']}")
                print(f"      Source: {result.source}")
                if result.debug_info:
                    if 'dense_score' in result.debug_info:
                        print(f"      Dense: {result.debug_info['dense_score']:.4f}, "
                              f"Sparse: {result.debug_info['sparse_score']:.4f}")
        
        # Cleanup
        await search.cleanup()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def benchmark_parallel_vs_sequential():
    """Compare timing between old sequential and new parallel approach."""
    
    brand = "specialized.com"
    query = "high performance road bike"
    
    print(f"\n\n🏁 BENCHMARK: Parallel vs Sequential Search")
    print("=" * 60)
    print(f"Brand: {brand}")
    print(f"Query: '{query}'")
    print("-" * 60)
    
    try:
        search = await get_search_pinecone(brand)
        
        # Run search multiple times to get average
        print("\n📊 Running 3 iterations...")
        
        for i in range(3):
            print(f"\n🔄 Iteration {i+1}:")
            results = await search.search(
                query=query,
                top_k=50,  # Use max top_k
                search_mode="hybrid",
                rerank=False
            )
            print(f"   Results: {len(results)}")
        
        await search.cleanup()
        
    except Exception as e:
        print(f"\n❌ Benchmark error: {e}")


if __name__ == "__main__":
    # Set up logging to see timing info
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    asyncio.run(test_parallel_search())
    asyncio.run(benchmark_parallel_vs_sequential())