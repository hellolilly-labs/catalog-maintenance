#!/usr/bin/env python3
"""
Test harness for SearchService with enhanced RAG integration.

This test harness validates both product_search and knowledge_search functionality
with the integrated advanced RAG solution from catalog-maintenance.

Usage:
    python test_search_service.py --test all
    python test_search_service.py --test product_search
    python test_search_service.py --test knowledge_search
    python test_search_service.py --test query_enhancement
"""

import asyncio
import logging
import json
import time
import argparse
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch

# Import the search service and related components
from search_service import SearchService
from spence.model import UserState
from livekit.agents import llm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestSearchService:
    """Test harness for SearchService functionality."""
    
    def __init__(self, account: str = "specialized.com"):
        self.account = account
        self.test_results = []
        
    async def setup_test_environment(self):
        """Setup mock test environment."""
        # Create mock user state
        self.mock_user_state = UserState(
            user_id="test_user",
            account=self.account,
            name="Test User",
            email="test@example.com"
        )
        
        # Create mock chat context
        self.mock_chat_ctx = llm.ChatContext([])
        
        # Add some sample conversation messages
        self.mock_chat_ctx.add_message(
            role="user",
            content=["Hi, I'm looking for a mountain bike for trail riding"]
        )
        self.mock_chat_ctx.add_message(
            role="assistant", 
            content=["I'd be happy to help you find a mountain bike! What's your experience level and budget range?"]
        )
        self.mock_chat_ctx.add_message(
            role="user",
            content=["I'm intermediate level and have around $3000 to spend"]
        )
        
        logger.info(f"Test environment setup complete for account: {self.account}")
    
    def log_test_result(self, test_name: str, success: bool, duration: float, details: Dict[str, Any] = None):
        """Log test result."""
        result = {
            "test_name": test_name,
            "success": success,
            "duration_ms": round(duration * 1000, 2),
            "details": details or {}
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} {test_name} ({result['duration_ms']}ms)")
        
        if details:
            for key, value in details.items():
                logger.info(f"  {key}: {value}")
    
    async def test_query_enhancement_with_filters(self):
        """Test enhanced query optimization with filter extraction."""
        test_name = "Query Enhancement with Filters"
        start_time = time.time()
        
        try:
            # Test queries with different complexity levels
            test_queries = [
                "carbon road bike under 3000",
                "mountain bike for downhill trails",
                "entry level gravel bike with disc brakes",
                "women's mountain bike for cross country racing"
            ]
            
            results = []
            for query in test_queries:
                enhanced_query, filters = await SearchService.enhance_product_query_with_filters(
                    query=query,
                    user_state=self.mock_user_state,
                    chat_ctx=self.mock_chat_ctx,
                    account=self.account,
                    product_knowledge="Specialized bikes for various cycling disciplines"
                )
                
                results.append({
                    "original_query": query,
                    "enhanced_query": enhanced_query,
                    "extracted_filters": filters,
                    "filter_count": len(filters)
                })
            
            # Validate results
            success = all(r["enhanced_query"] for r in results)
            
            duration = time.time() - start_time
            self.log_test_result(
                test_name, 
                success, 
                duration,
                {
                    "queries_tested": len(test_queries),
                    "all_enhanced": success,
                    "avg_filter_count": sum(r["filter_count"] for r in results) / len(results),
                    "sample_result": results[0] if results else None
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(test_name, False, duration, {"error": str(e)})
    
    async def test_product_search_rag_with_filters(self):
        """Test enhanced product search with RAG and filters."""
        test_name = "Product Search RAG with Filters"
        start_time = time.time()
        
        try:
            # Test product search with different scenarios
            test_cases = [
                {
                    "query": "carbon road bike for racing under 4000",
                    "expected_filters": ["category", "frame_material", "price"]
                },
                {
                    "query": "beginner mountain bike",
                    "expected_filters": ["category", "skill_level"]
                }
            ]
            
            results = []
            for case in test_cases:
                # First enhance the query and extract filters
                enhanced_query, filters = await SearchService.enhance_product_query_with_filters(
                    query=case["query"],
                    user_state=self.mock_user_state,
                    chat_ctx=self.mock_chat_ctx,
                    account=self.account,
                    product_knowledge=""
                )
                
                # Then search with filters
                search_results = await SearchService.search_products_rag_with_filters(
                    query=enhanced_query,
                    filters=filters,
                    account=self.account,
                    user_state=self.mock_user_state,
                    top_k=10,
                    top_n=5,
                    min_score=0.1,
                    min_n=1
                )
                
                results.append({
                    "query": case["query"],
                    "enhanced_query": enhanced_query,
                    "filters": filters,
                    "result_count": len(search_results),
                    "has_results": len(search_results) > 0,
                    "top_score": search_results[0].get("score", 0) if search_results else 0
                })
            
            # Validate results
            success = all(r["has_results"] for r in results) if results else False
            
            duration = time.time() - start_time
            self.log_test_result(
                test_name,
                success,
                duration,
                {
                    "test_cases": len(test_cases),
                    "successful_searches": sum(1 for r in results if r["has_results"]),
                    "avg_results_per_query": sum(r["result_count"] for r in results) / len(results) if results else 0,
                    "sample_result": results[0] if results else None
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(test_name, False, duration, {"error": str(e)})
    
    async def test_knowledge_search_with_context(self):
        """Test enhanced knowledge search with conversation context."""
        test_name = "Knowledge Search with Context"
        start_time = time.time()
        
        try:
            # Test knowledge search queries
            test_queries = [
                "bike sizing guide",
                "maintenance tips for mountain bikes",
                "difference between road and gravel bikes",
                "what is bike geometry"
            ]
            
            results = []
            for query in test_queries:
                search_results = await SearchService.search_knowledge_rag_with_context(
                    query=query,
                    user_state=self.mock_user_state,
                    chat_ctx=self.mock_chat_ctx,
                    account=self.account,
                    knowledge_base="Cycling guides and educational content",
                    top_k=15,
                    top_n=5,
                    min_score=0.1
                )
                
                results.append({
                    "query": query,
                    "result_count": len(search_results),
                    "has_results": len(search_results) > 0,
                    "top_score": search_results[0].get("score", 0) if search_results else 0,
                    "result_types": list(set(r.get("type", "unknown") for r in search_results))
                })
            
            # Validate results
            success = all(r["has_results"] for r in results) if results else False
            
            duration = time.time() - start_time
            self.log_test_result(
                test_name,
                success,
                duration,
                {
                    "queries_tested": len(test_queries),
                    "successful_searches": sum(1 for r in results if r["has_results"]),
                    "avg_results_per_query": sum(r["result_count"] for r in results) / len(results) if results else 0,
                    "sample_result": results[0] if results else None
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(test_name, False, duration, {"error": str(e)})
    
    async def test_basic_search_functions(self):
        """Test basic search functions for backwards compatibility."""
        test_name = "Basic Search Functions"
        start_time = time.time()
        
        try:
            # Test basic product search
            product_results = await SearchService.search_products_rag(
                query="mountain bike",
                account=self.account,
                top_k=10,
                top_n=5,
                min_score=0.1
            )
            
            # Test basic knowledge search
            knowledge_results = await SearchService.search_knowledge(
                query="bike maintenance",
                account=self.account,
                top_k=10,
                top_n=5,
                min_score=0.1
            )
            
            success = True  # Basic functions should not error
            
            duration = time.time() - start_time
            self.log_test_result(
                test_name,
                success,
                duration,
                {
                    "product_results": len(product_results),
                    "knowledge_results": len(knowledge_results),
                    "basic_functions_working": True
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(test_name, False, duration, {"error": str(e)})
    
    async def test_performance_benchmarks(self):
        """Test performance of enhanced vs basic search."""
        test_name = "Performance Benchmarks"
        start_time = time.time()
        
        try:
            test_query = "carbon road bike for racing"
            iterations = 3
            
            # Test enhanced search performance
            enhanced_times = []
            for _ in range(iterations):
                start = time.time()
                enhanced_query, filters = await SearchService.enhance_product_query_with_filters(
                    query=test_query,
                    user_state=self.mock_user_state,
                    chat_ctx=self.mock_chat_ctx,
                    account=self.account,
                    product_knowledge=""
                )
                enhanced_times.append(time.time() - start)
            
            # Test basic search performance
            basic_times = []
            for _ in range(iterations):
                start = time.time()
                basic_results = await SearchService.search_products_rag(
                    query=test_query,
                    account=self.account,
                    top_k=10
                )
                basic_times.append(time.time() - start)
            
            avg_enhanced_time = sum(enhanced_times) / len(enhanced_times)
            avg_basic_time = sum(basic_times) / len(basic_times)
            
            duration = time.time() - start_time
            self.log_test_result(
                test_name,
                True,
                duration,
                {
                    "avg_enhanced_time_ms": round(avg_enhanced_time * 1000, 2),
                    "avg_basic_time_ms": round(avg_basic_time * 1000, 2),
                    "performance_overhead": f"{round((avg_enhanced_time / avg_basic_time - 1) * 100, 1)}%" if avg_basic_time > 0 else "N/A",
                    "iterations": iterations
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(test_name, False, duration, {"error": str(e)})
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r["success"])
        total_time = sum(r["duration_ms"] for r in self.test_results)
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {round(passed_tests/total_tests*100, 1)}%" if total_tests > 0 else "N/A")
        print(f"Total Time: {total_time}ms")
        print(f"Average Time: {round(total_time/total_tests, 2)}ms" if total_tests > 0 else "N/A")
        
        print("\nDETAILED RESULTS:")
        for result in self.test_results:
            status = "✅" if result["success"] else "❌"
            print(f"{status} {result['test_name']}: {result['duration_ms']}ms")
            
            if not result["success"] and "error" in result["details"]:
                print(f"   Error: {result['details']['error']}")
        
        print("\n" + "="*60)

async def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Test SearchService functionality")
    parser.add_argument(
        "--test", 
        choices=["all", "query_enhancement", "product_search", "knowledge_search", "basic", "performance"],
        default="all",
        help="Which tests to run"
    )
    parser.add_argument(
        "--account",
        default="specialized.com",
        help="Account to test with"
    )
    
    args = parser.parse_args()
    
    # Initialize test harness
    test_harness = TestSearchService(account=args.account)
    await test_harness.setup_test_environment()
    
    print(f"Starting SearchService tests for account: {args.account}")
    print(f"Test selection: {args.test}")
    print("-" * 60)
    
    # Run selected tests
    if args.test in ["all", "query_enhancement"]:
        await test_harness.test_query_enhancement_with_filters()
    
    if args.test in ["all", "product_search"]:
        await test_harness.test_product_search_rag_with_filters()
    
    if args.test in ["all", "knowledge_search"]:
        await test_harness.test_knowledge_search_with_context()
    
    if args.test in ["all", "basic"]:
        await test_harness.test_basic_search_functions()
    
    if args.test in ["all", "performance"]:
        await test_harness.test_performance_benchmarks()
    
    # Print summary
    test_harness.print_summary()

if __name__ == "__main__":
    asyncio.run(main())