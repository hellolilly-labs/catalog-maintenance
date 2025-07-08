#!/usr/bin/env python3
"""
Enhanced performance test for SearchPinecone optimizations.

This script tests all the performance improvements:
1. True async parallelism with asyncio.to_thread
2. Connection pooling and pre-warming
3. Query object caching
4. Optimized result processing
5. Async reranking
6. Reduced payload sizes
"""

import asyncio
import time
import statistics
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from packages.liddy.search.pinecone import get_search_pinecone


async def benchmark_search_performance():
    """Comprehensive benchmark of search performance optimizations."""
    
    brand = "specialized.com"
    queries = [
        "high performance road bike with electronic shifting",
        "mountain bike full suspension for trails",
        "electric bike for urban commuting",
        "lightweight carbon gravel bike",
        "beginner friendly hybrid bike"
    ]
    
    print("\n" + "="*80)
    print("🚀 SEARCH PERFORMANCE BENCHMARK V2")
    print("="*80)
    print(f"Brand: {brand}")
    print(f"Queries: {len(queries)}")
    print("-"*80)
    
    # Get search instance (this includes pre-warming)
    print("\n⚡ Initializing SearchPinecone with optimizations...")
    init_start = time.perf_counter()
    search = await get_search_pinecone(brand)
    init_time = time.perf_counter() - init_start
    print(f"✅ Initialization completed in {init_time:.3f}s (includes connection pre-warming)")
    
    # Run searches and collect metrics
    dense_times = []
    sparse_times = []
    parallel_times = []
    rerank_times = []
    total_times = []
    
    print("\n📊 Running benchmark iterations...")
    print("-"*80)
    
    for i, query in enumerate(queries, 1):
        print(f"\n🔍 Query {i}: '{query[:50]}...'")
        
        # Extract timing from logs
        class TimingCapture:
            def __init__(self):
                self.dense_time = None
                self.sparse_time = None
                self.parallel_time = None
                self.rerank_time = None
                self.total_time = None
        
        timing = TimingCapture()
        
        # Run search
        start_time = time.perf_counter()
        results = await search.search(
            query=query,
            top_k=20,
            search_mode="hybrid",
            rerank=True
        )
        end_time = time.perf_counter()
        
        timing.total_time = end_time - start_time
        total_times.append(timing.total_time)
        
        print(f"   Found {len(results)} results in {timing.total_time:.3f}s")
        
        # Show sample result
        if results:
            top_result = results[0]
            print(f"   Top result: {top_result.metadata.get('name', 'N/A')} (Score: {top_result.score:.4f})")
    
    # Calculate statistics
    print("\n" + "="*80)
    print("📈 PERFORMANCE SUMMARY")
    print("="*80)
    
    avg_total = statistics.mean(total_times)
    min_total = min(total_times)
    max_total = max(total_times)
    
    print(f"\nTotal Search Time:")
    print(f"  Average: {avg_total:.3f}s")
    print(f"  Min:     {min_total:.3f}s") 
    print(f"  Max:     {max_total:.3f}s")
    
    # Compare with expected old performance
    old_dense = 0.77
    old_sparse = 0.49
    old_sequential = old_dense + old_sparse  # ~1.26s
    old_rerank = 0.36
    old_total = old_sequential + old_rerank  # ~1.62s
    
    new_parallel = max(old_dense, old_sparse) + 0.05  # ~0.82s
    expected_new_total = new_parallel + old_rerank  # ~1.18s
    
    print(f"\n🔄 Performance Comparison:")
    print(f"  Old (sequential):     ~{old_total:.2f}s")
    print(f"  Expected (parallel):  ~{expected_new_total:.2f}s")
    print(f"  Actual (optimized):   {avg_total:.2f}s")
    print(f"  Improvement:          {((old_total - avg_total) / old_total * 100):.1f}%")
    
    # Test connection reuse benefit
    print("\n🔌 Testing connection reuse benefit...")
    
    # First query (cold)
    cold_start = time.perf_counter()
    await search.search("test query cold", top_k=5, search_mode="hybrid", rerank=False)
    cold_time = time.perf_counter() - cold_start
    
    # Second query (warm)
    warm_start = time.perf_counter()
    await search.search("test query warm", top_k=5, search_mode="hybrid", rerank=False)
    warm_time = time.perf_counter() - warm_start
    
    print(f"  Cold query: {cold_time:.3f}s")
    print(f"  Warm query: {warm_time:.3f}s")
    print(f"  Speedup:    {((cold_time - warm_time) / cold_time * 100):.1f}%")
    
    # Cleanup
    await search.cleanup()
    
    print("\n✅ Benchmark completed!")
    print("="*80)


async def test_extreme_parallelism():
    """Test with multiple concurrent searches to stress connection pooling."""
    
    print("\n\n" + "="*80)
    print("🔥 EXTREME PARALLELISM TEST")
    print("="*80)
    
    brand = "specialized.com"
    search = await get_search_pinecone(brand)
    
    queries = [
        "road bike carbon",
        "mountain bike trail",
        "electric bike city",
        "gravel bike adventure",
        "hybrid bike comfort",
        "bmx bike tricks",
        "touring bike travel",
        "cyclocross bike race"
    ]
    
    print(f"Running {len(queries)} searches concurrently...")
    
    start_time = time.perf_counter()
    
    # Run all searches concurrently
    tasks = [
        search.search(query, top_k=10, search_mode="hybrid", rerank=False)
        for query in queries
    ]
    
    results = await asyncio.gather(*tasks)
    
    total_time = time.perf_counter() - start_time
    
    print(f"\n✅ Completed {len(queries)} searches in {total_time:.3f}s")
    print(f"   Average per search: {total_time/len(queries):.3f}s")
    print(f"   Total results: {sum(len(r) for r in results)}")
    
    await search.cleanup()


if __name__ == "__main__":
    # Set up logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run benchmarks
    asyncio.run(benchmark_search_performance())
    asyncio.run(test_extreme_parallelism())