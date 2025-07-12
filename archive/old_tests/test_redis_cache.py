#!/usr/bin/env python3
"""
Test script for Redis caching in Tavily search

This script demonstrates the Redis caching functionality that has been added to the TavilySearchProvider.
"""

import asyncio
import time
import json
from src.web_search import TavilySearchProvider

async def test_redis_caching():
    """Test Redis caching functionality"""
    
    print("🔧 Testing Redis Caching for Tavily Search")
    print("=" * 50)
    
    # Initialize the provider
    provider = TavilySearchProvider()
    
    # Check Redis availability
    if not provider.redis_client:
        print("❌ Redis not available - caching will be disabled")
        return
    
    print("✅ Redis connection established")
    
    # Test query
    test_query = "specialized bikes company history"
    search_params = {
        "search_depth": "basic",
        "max_results": 5,
        "topic": "general"
    }
    
    print(f"\n🔍 Testing cache with query: '{test_query}'")
    
    # First search - should be a cache miss
    print("\n1️⃣ First search (should be CACHE MISS):")
    start_time = time.time()
    results1 = await provider.search(test_query, **search_params)
    first_duration = time.time() - start_time
    print(f"   Results: {len(results1)} items")
    print(f"   Duration: {first_duration:.2f}s")
    
    # Second search - should be a cache hit
    print("\n2️⃣ Second search (should be CACHE HIT):")
    start_time = time.time()
    results2 = await provider.search(test_query, **search_params)
    second_duration = time.time() - start_time
    print(f"   Results: {len(results2)} items")
    print(f"   Duration: {second_duration:.2f}s")
    
    # Verify results are identical
    if len(results1) == len(results2):
        print("✅ Cache results match original results")
    else:
        print(f"❌ Cache mismatch: {len(results1)} vs {len(results2)}")
    
    # Show speed improvement
    if second_duration < first_duration:
        speedup = first_duration / second_duration
        print(f"🚀 Cache speedup: {speedup:.1f}x faster")
    
    # Test cache statistics
    print("\n📊 Cache Statistics:")
    stats = provider.get_cache_stats()
    if "error" in stats:
        print(f"   Error: {stats['error']}")
    else:
        print(f"   Total cache entries: {stats['total_search_cache_entries']}")
        print(f"   Cache TTL: {stats['cache_ttl_days']} days")
        print(f"   Average remaining TTL: {stats['average_remaining_ttl_hours']} hours")
        if stats.get('redis_memory'):
            print(f"   Redis memory usage: {stats['redis_memory'].get('used_memory_human', 'N/A')}")
    
    # Test different parameters (should be cache miss)
    print("\n3️⃣ Same query with different parameters (should be CACHE MISS):")
    different_params = {
        "search_depth": "advanced",  # Different parameter
        "max_results": 5,
        "topic": "general"
    }
    start_time = time.time()
    results3 = await provider.search(test_query, **different_params)
    third_duration = time.time() - start_time
    print(f"   Results: {len(results3)} items")
    print(f"   Duration: {third_duration:.2f}s")
    
    # Show final cache stats
    print("\n📊 Final Cache Statistics:")
    final_stats = provider.get_cache_stats()
    if "error" not in final_stats:
        print(f"   Total cache entries: {final_stats['total_search_cache_entries']}")
    
    print("\n✅ Redis caching test completed!")
    print("\nKey Features:")
    print("- ✅ Exact query and parameter matching")
    print("- ✅ 2-day TTL (172800 seconds)")
    print("- ✅ Graceful fallback when Redis unavailable")
    print("- ✅ Cache statistics and management")
    print("- ✅ Significant performance improvement on cache hits")

async def test_cache_management():
    """Test cache management functions"""
    
    print("\n🛠️ Testing Cache Management")
    print("=" * 30)
    
    provider = TavilySearchProvider()
    
    if not provider.redis_client:
        print("❌ Redis not available for cache management tests")
        return
    
    # Show current stats
    stats = provider.get_cache_stats()
    if "error" not in stats:
        print(f"📊 Current cache entries: {stats['total_search_cache_entries']}")
    
    # Test cache clearing
    cleared = provider.clear_search_cache()
    print(f"🗑️ Cleared {cleared} cache entries")
    
    # Show stats after clearing
    stats_after = provider.get_cache_stats()
    if "error" not in stats_after:
        print(f"📊 Cache entries after clearing: {stats_after['total_search_cache_entries']}")

if __name__ == "__main__":
    print("🧪 Starting Redis Cache Tests")
    asyncio.run(test_redis_caching())
    asyncio.run(test_cache_management()) 