#!/usr/bin/env python3
"""
Benchmark Hybrid Search Performance

Tests the performance characteristics of hybrid search vs dense-only search.
"""

import time
import statistics
import json
import os
import sys
from typing import List, Dict, Any

from src.search.hybrid_search import HybridSearchEngine


def benchmark_search_methods(engine: HybridSearchEngine, test_queries: List[str], iterations: int = 5):
    """Benchmark different search methods."""
    
    results = {
        "dense_only": {"times": [], "avg_results": []},
        "sparse_only": {"times": [], "avg_results": []},
        "hybrid_auto": {"times": [], "avg_results": []},
        "hybrid_balanced": {"times": [], "avg_results": []}
    }
    
    print(f"🏃 Running benchmark with {len(test_queries)} queries, {iterations} iterations each...")
    print("-" * 70)
    
    for method_name, method_config in [
        ("dense_only", {"dense_weight": 1.0, "sparse_weight": 0.0}),
        ("sparse_only", {"dense_weight": 0.0, "sparse_weight": 1.0}),
        ("hybrid_auto", {}),  # Auto-weighted
        ("hybrid_balanced", {"dense_weight": 0.5, "sparse_weight": 0.5})
    ]:
        print(f"\nBenchmarking: {method_name}")
        
        for query in test_queries:
            query_times = []
            query_result_counts = []
            
            # Warm-up run
            engine.search(query, top_k=10, **method_config)
            
            # Timed runs
            for _ in range(iterations):
                start_time = time.time()
                results_list = engine.search(query, top_k=10, **method_config)
                end_time = time.time()
                
                query_times.append((end_time - start_time) * 1000)  # Convert to ms
                query_result_counts.append(len(results_list))
            
            avg_time = statistics.mean(query_times)
            results[method_name]["times"].append(avg_time)
            results[method_name]["avg_results"].append(statistics.mean(query_result_counts))
            
            print(f"  Query: '{query[:50]}...' - Avg: {avg_time:.2f}ms")
    
    return results


def analyze_results(benchmark_results: Dict[str, Any], test_queries: List[str]):
    """Analyze and display benchmark results."""
    
    print("\n" + "="*70)
    print("PERFORMANCE ANALYSIS")
    print("="*70)
    
    # Overall statistics
    print("\n📊 Overall Performance (ms):")
    print(f"{'Method':<20} {'Min':<10} {'Avg':<10} {'Max':<10} {'Std Dev':<10}")
    print("-" * 60)
    
    for method, data in benchmark_results.items():
        times = data["times"]
        if times:
            print(f"{method:<20} {min(times):<10.2f} {statistics.mean(times):<10.2f} "
                  f"{max(times):<10.2f} {statistics.stdev(times) if len(times) > 1 else 0:<10.2f}")
    
    # Query type analysis
    print("\n🔍 Performance by Query Type:")
    
    query_types = {
        "exact_match": [],
        "semantic": [],
        "mixed": []
    }
    
    for i, query in enumerate(test_queries):
        # Classify query
        if any(word in query.lower() for word in ["model", "sku", "exact"]) or query.count(" ") < 3:
            query_type = "exact_match"
        elif any(word in query.lower() for word in ["comfortable", "need", "looking for", "best"]):
            query_type = "semantic"
        else:
            query_type = "mixed"
        
        query_types[query_type].append(i)
    
    print(f"\n{'Query Type':<15} {'Dense':<15} {'Sparse':<15} {'Hybrid Auto':<15}")
    print("-" * 60)
    
    for query_type, indices in query_types.items():
        if indices:
            dense_avg = statistics.mean([benchmark_results["dense_only"]["times"][i] for i in indices])
            sparse_avg = statistics.mean([benchmark_results["sparse_only"]["times"][i] for i in indices])
            hybrid_avg = statistics.mean([benchmark_results["hybrid_auto"]["times"][i] for i in indices])
            
            print(f"{query_type:<15} {dense_avg:<15.2f} {sparse_avg:<15.2f} {hybrid_avg:<15.2f}")
    
    # Speedup analysis
    print("\n⚡ Hybrid vs Single-Method Speedup:")
    
    hybrid_times = benchmark_results["hybrid_auto"]["times"]
    dense_times = benchmark_results["dense_only"]["times"]
    
    if hybrid_times and dense_times:
        avg_hybrid = statistics.mean(hybrid_times)
        avg_dense = statistics.mean(dense_times)
        
        speedup = ((avg_dense - avg_hybrid) / avg_dense) * 100
        print(f"Hybrid is {abs(speedup):.1f}% {'faster' if speedup > 0 else 'slower'} than dense-only")


def generate_test_queries(brand_domain: str) -> List[str]:
    """Generate test queries based on brand."""
    
    base_queries = [
        # Exact match queries
        f"{brand_domain.split('.')[0]} flagship product",
        "model XYZ-123",
        "SKU 12345",
        
        # Semantic queries
        "comfortable product for everyday use",
        "best value for money",
        "looking for something lightweight and durable",
        
        # Mixed queries
        "high-end products under $1000",
        f"{brand_domain.split('.')[0]} products with premium features",
        "latest collection with modern design"
    ]
    
    # Add brand-specific queries
    if "specialized" in brand_domain.lower():
        base_queries.extend([
            "Specialized Tarmac SL7",
            "carbon road bikes for racing",
            "mountain bikes with SRAM Eagle"
        ])
    elif "balenciaga" in brand_domain.lower():
        base_queries.extend([
            "Balenciaga Triple S sneakers",
            "leather handbags for evening",
            "designer accessories under $500"
        ])
    
    return base_queries


def main():
    """Main benchmark execution."""
    
    if len(sys.argv) != 3:
        print("Usage: python benchmark_hybrid_search.py <brand_domain> <index_name>")
        print("Example: python benchmark_hybrid_search.py specialized.com specialized-llama-2048")
        sys.exit(1)
    
    brand_domain = sys.argv[1]
    index_name = sys.argv[2]
    
    # Check for API key
    if not os.environ.get("PINECONE_API_KEY"):
        print("❌ Error: PINECONE_API_KEY environment variable not set")
        sys.exit(1)
    
    print(f"🚀 Benchmarking Hybrid Search Performance")
    print(f"Brand: {brand_domain}")
    print(f"Index: {index_name}")
    print("="*70)
    
    # Initialize engine
    print("\n⚙️ Initializing search engine...")
    engine = HybridSearchEngine(brand_domain, index_name)
    
    # Generate test queries
    test_queries = generate_test_queries(brand_domain)
    print(f"\n📝 Generated {len(test_queries)} test queries")
    
    try:
        # Run benchmarks
        results = benchmark_search_methods(engine, test_queries)
        
        # Analyze results
        analyze_results(results, test_queries)
        
        # Save results
        output_file = f"benchmark_results_{brand_domain.replace('.', '_')}.json"
        with open(output_file, 'w') as f:
            json.dump({
                "brand_domain": brand_domain,
                "index_name": index_name,
                "test_queries": test_queries,
                "results": results
            }, f, indent=2)
        
        print(f"\n💾 Results saved to {output_file}")
        print("\n✅ Benchmark complete!")
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()