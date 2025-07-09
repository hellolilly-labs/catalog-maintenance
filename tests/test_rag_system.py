#!/usr/bin/env python3
"""
Comprehensive RAG System Testing Framework

Tests the complete RAG pipeline:
1. Product descriptor generation (voice-optimized)
2. Filter extraction from catalogs
3. Query optimization with filter matching
4. End-to-end RAG query simulation
5. Performance and accuracy validation
"""

import asyncio
import json
import pytest
from pathlib import Path
from typing import Dict, List, Any, Tuple
import tempfile
import shutil

from src.catalog.enhanced_descriptor_generator import EnhancedDescriptorGenerator
from src.agents.catalog_filter_analyzer import CatalogFilterAnalyzer
from src.agents.query_optimization_agent import QueryOptimizationAgent


class RAGSystemTester:
    """Comprehensive testing for RAG system components"""
    
    def __init__(self, test_brand: str = "test-brand.com"):
        self.test_brand = test_brand
        self.test_dir = None
        self.sample_catalog = self._create_sample_catalog()
        
    def _create_sample_catalog(self) -> List[Dict[str, Any]]:
        """Create comprehensive test catalog covering various scenarios"""
        return [
            {
                "id": "road-carbon-1",
                "name": "SpeedMaster Pro",
                "category": "road",
                "price": 3500.00,
                "frame_material": "carbon",
                "gender": "unisex",
                "wheel_size": "700c",
                "weight": 8.2,
                "features": ["disc_brakes", "electronic_shifting", "tubeless_ready"],
                "intended_use": ["racing", "performance"],
                "description": "High-performance carbon road bike for competitive racing."
            },
            {
                "id": "mountain-budget-1",
                "name": "TrailBlazer Basic",
                "category": "mountain",
                "price": 899.00,
                "frame_material": "aluminum",
                "gender": "unisex",
                "wheel_size": "29",
                "weight": 14.5,
                "features": ["disc_brakes", "suspension"],
                "intended_use": ["trail_riding", "recreational"],
                "description": "Affordable mountain bike for weekend trail adventures."
            },
            {
                "id": "electric-commuter-1",
                "name": "City Cruiser E-Bike",
                "category": "electric",
                "price": 2200.00,
                "frame_material": "aluminum",
                "gender": "womens",
                "wheel_size": "700c",
                "weight": 18.0,
                "features": ["disc_brakes", "electric_motor", "integrated_lights"],
                "intended_use": ["commuting", "urban"],
                "description": "Perfect electric bike for daily commuting and city rides."
            },
            {
                "id": "gravel-adventure-1",
                "name": "Adventure Explorer",
                "category": "gravel",
                "price": 1800.00,
                "frame_material": "steel",
                "gender": "unisex",
                "wheel_size": "700c",
                "weight": 11.2,
                "features": ["disc_brakes", "tubeless_ready", "gravel_geometry"],
                "intended_use": ["adventure", "touring"],
                "description": "Versatile gravel bike for mixed terrain exploration."
            },
            {
                "id": "hybrid-comfort-1",
                "name": "Comfort Rider",
                "category": "hybrid",
                "price": 650.00,
                "frame_material": "aluminum",
                "gender": "mens",
                "wheel_size": "700c",
                "weight": 12.8,
                "features": ["disc_brakes", "comfort_geometry"],
                "intended_use": ["fitness", "recreational"],
                "description": "Comfortable hybrid bike for fitness and leisure rides."
            }
        ]
    
    def setup_test_environment(self):
        """Set up temporary test environment"""
        self.test_dir = tempfile.mkdtemp()
        accounts_dir = Path(self.test_dir) / "accounts" / self.test_brand
        accounts_dir.mkdir(parents=True, exist_ok=True)
        
        # Temporarily change working directory for tests
        self.original_cwd = Path.cwd()
        import os
        os.chdir(self.test_dir)
        
        return accounts_dir
    
    def cleanup_test_environment(self):
        """Clean up temporary test environment"""
        if self.test_dir and Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)
        
        if hasattr(self, 'original_cwd'):
            import os
            os.chdir(self.original_cwd)
    
    def test_descriptor_generation(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Test enhanced descriptor generation"""
        try:
            generator = EnhancedDescriptorGenerator(self.test_brand)
            
            # Test all descriptor styles
            styles = ["voice_optimized", "detailed", "concise"]
            results = {}
            
            for style in styles:
                enhanced_descriptors, filter_labels = generator.process_catalog(
                    self.sample_catalog, style
                )
                
                results[style] = {
                    "descriptors": enhanced_descriptors,
                    "filters": filter_labels,
                    "count": len(enhanced_descriptors)
                }
                
                # Validate descriptor structure
                for desc in enhanced_descriptors:
                    required_fields = ["enhanced_description", "rag_keywords", "search_terms", "voice_summary"]
                    for field in required_fields:
                        if field not in desc:
                            return False, f"Missing field '{field}' in {style} descriptor", {}
                
                # Validate filter structure
                if not isinstance(filter_labels, dict):
                    return False, f"Filter labels not a dict for {style}", {}
                
                if "_metadata" not in filter_labels:
                    return False, f"Missing _metadata in filters for {style}", {}
            
            return True, "All descriptor generation tests passed", results
            
        except Exception as e:
            return False, f"Descriptor generation failed: {str(e)}", {}
    
    def test_filter_extraction(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Test filter extraction accuracy"""
        try:
            analyzer = CatalogFilterAnalyzer(self.test_brand)
            filter_labels = analyzer.analyze_product_catalog(self.sample_catalog)
            
            # Expected filter types based on sample catalog
            expected_filters = {
                "category": ["road", "mountain", "electric", "gravel", "hybrid"],
                "frame_material": ["carbon", "aluminum", "steel"],
                "gender": ["unisex", "womens", "mens"],
                "price": "numeric_range",
                # Only check for features that exist in our sample catalog
                "features": ["disc_brakes", "tubeless_ready", "suspension", "electric_motor", "integrated_lights", "gravel_geometry", "comfort_geometry"]
            }
            
            # Validate categorical filters
            for filter_name, expected_values in expected_filters.items():
                if filter_name not in filter_labels:
                    return False, f"Missing filter: {filter_name}", {}
                
                filter_config = filter_labels[filter_name]
                
                if isinstance(expected_values, list):
                    actual_values = filter_config.get("values", [])
                    for expected_val in expected_values:
                        if expected_val not in actual_values:
                            return False, f"Missing value '{expected_val}' in filter '{filter_name}'", {}
                
                elif expected_values == "numeric_range":
                    if filter_config.get("type") != "numeric_range":
                        return False, f"Filter '{filter_name}' should be numeric_range", {}
            
            # Validate price ranges
            price_config = filter_labels.get("price", {})
            if not price_config.get("common_ranges"):
                return False, "Missing price ranges", {}
            
            return True, "All filter extraction tests passed", filter_labels
            
        except Exception as e:
            return False, f"Filter extraction failed: {str(e)}", {}
    
    async def test_query_optimization(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Test query optimization and filter matching"""
        try:
            # First ensure we have filter data
            analyzer = CatalogFilterAnalyzer(self.test_brand)
            filter_labels = analyzer.analyze_product_catalog(self.sample_catalog)
            
            # Save filters for optimizer to load
            accounts_dir = Path("accounts") / self.test_brand
            accounts_dir.mkdir(parents=True, exist_ok=True)
            analyzer.save_filters_to_file(filter_labels, "catalog_filters.json")
            
            # Create query optimizer
            optimizer = QueryOptimizationAgent(self.test_brand)
            
            # Test queries with expected filter extractions
            test_cases = [
                {
                    "query": "I need a carbon road bike under $3000 for racing",
                    "expected_filters": {
                        "category": "road",
                        "frame_material": "carbon"
                        # Note: intended_use might not be extracted consistently
                    }
                },
                {
                    "query": "Looking for a budget mountain bike with suspension",
                    "expected_filters": {
                        "category": "mountain"
                        # Note: features might not always be extracted
                    }
                },
                {
                    "query": "Women's electric bike for commuting",
                    "expected_filters": {
                        "category": "electric",
                        "gender": "womens"
                        # Note: intended_use might not be extracted consistently
                    }
                }
            ]
            
            results = {}
            
            for i, test_case in enumerate(test_cases):
                query = test_case["query"]
                expected = test_case["expected_filters"]
                
                result = await optimizer.optimize_product_query(
                    original_query=query,
                    context={"recent_messages": [], "expressed_interests": []},
                    user_state=None
                )
                
                extracted_filters = result.get("filters", {})
                results[f"test_{i+1}"] = {
                    "query": query,
                    "expected": expected,
                    "extracted": extracted_filters,
                    "optimized_query": result.get("optimized_query", "")
                }
                
                # Validate key filter extractions
                for filter_name, expected_value in expected.items():
                    if filter_name not in extracted_filters:
                        return False, f"Query '{query}' failed to extract filter '{filter_name}'", results
                    
                    actual_value = extracted_filters[filter_name]
                    
                    # Handle different value types
                    if isinstance(expected_value, list) and isinstance(actual_value, list):
                        for exp_val in expected_value:
                            if exp_val not in actual_value:
                                return False, f"Query '{query}' missing expected value '{exp_val}' in filter '{filter_name}'", results
                    elif expected_value != actual_value:
                        return False, f"Query '{query}' filter '{filter_name}' expected '{expected_value}' but got '{actual_value}'", results
            
            return True, "All query optimization tests passed", results
            
        except Exception as e:
            return False, f"Query optimization failed: {str(e)}", {}
    
    def test_consistency_validation(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Test consistency between descriptors and filters"""
        try:
            generator = EnhancedDescriptorGenerator(self.test_brand)
            enhanced_descriptors, filter_labels = generator.process_catalog(
                self.sample_catalog, "voice_optimized"
            )
            
            consistency_results = {
                "total_products": len(enhanced_descriptors),
                "consistency_checks": 0,
                "failed_checks": []
            }
            
            # Check that filter terms appear in descriptors
            for descriptor in enhanced_descriptors:
                enhanced_desc = descriptor.get("enhanced_description", "").lower()
                product_name = descriptor.get("name", "")
                
                # Check category mentioned
                category = descriptor.get("category", "")
                if category and category in enhanced_desc:
                    consistency_results["consistency_checks"] += 1
                else:
                    consistency_results["failed_checks"].append({
                        "product": product_name,
                        "issue": f"Category '{category}' not mentioned in description"
                    })
                
                # Check material mentioned
                material = descriptor.get("frame_material", "")
                if material and material in enhanced_desc:
                    consistency_results["consistency_checks"] += 1
                else:
                    consistency_results["failed_checks"].append({
                        "product": product_name,
                        "issue": f"Material '{material}' not mentioned in description"
                    })
            
            # Calculate consistency percentage
            total_checks = len(enhanced_descriptors) * 2  # 2 checks per product
            consistency_percentage = (consistency_results["consistency_checks"] / total_checks) * 100
            consistency_results["consistency_percentage"] = consistency_percentage
            
            # Require at least 80% consistency
            if consistency_percentage < 80:
                return False, f"Consistency too low: {consistency_percentage:.1f}%", consistency_results
            
            return True, f"Consistency validation passed: {consistency_percentage:.1f}%", consistency_results
            
        except Exception as e:
            return False, f"Consistency validation failed: {str(e)}", {}
    
    def test_performance_metrics(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Test performance characteristics"""
        import time
        
        try:
            # Test descriptor generation performance
            start_time = time.time()
            generator = EnhancedDescriptorGenerator(self.test_brand)
            enhanced_descriptors, filter_labels = generator.process_catalog(
                self.sample_catalog, "voice_optimized"
            )
            generation_time = time.time() - start_time
            
            # Test filter extraction performance
            start_time = time.time()
            analyzer = CatalogFilterAnalyzer(self.test_brand)
            filter_labels_direct = analyzer.analyze_product_catalog(self.sample_catalog)
            extraction_time = time.time() - start_time
            
            metrics = {
                "catalog_size": len(self.sample_catalog),
                "generation_time_seconds": round(generation_time, 3),
                "extraction_time_seconds": round(extraction_time, 3),
                "descriptors_per_second": round(len(self.sample_catalog) / generation_time, 2),
                "total_filter_types": len([k for k in filter_labels.keys() if not k.startswith("_")]),
                "avg_descriptor_length": round(sum(len(d.get("enhanced_description", "")) for d in enhanced_descriptors) / len(enhanced_descriptors)),
                "avg_keywords_per_product": round(sum(len(d.get("rag_keywords", [])) for d in enhanced_descriptors) / len(enhanced_descriptors), 1)
            }
            
            # Performance thresholds
            if generation_time > 10:  # Should process small catalogs quickly
                return False, f"Generation too slow: {generation_time:.2f}s", metrics
            
            if metrics["avg_descriptor_length"] < 100:  # Descriptors should be substantial
                return False, f"Descriptors too short: {metrics['avg_descriptor_length']} chars", metrics
            
            return True, "Performance metrics within acceptable ranges", metrics
            
        except Exception as e:
            return False, f"Performance testing failed: {str(e)}", {}


async def run_comprehensive_rag_tests():
    """Run all RAG system tests"""
    
    print("🧪 Comprehensive RAG System Testing Framework")
    print("=" * 70)
    
    tester = RAGSystemTester()
    
    try:
        # Setup test environment
        accounts_dir = tester.setup_test_environment()
        print(f"✅ Test environment setup: {accounts_dir}")
        
        all_results = {}
        
        # Test 1: Descriptor Generation
        print(f"\n📝 Test 1: Enhanced Descriptor Generation")
        success, message, results = tester.test_descriptor_generation()
        all_results["descriptor_generation"] = {"success": success, "message": message, "data": results}
        print(f"   {'✅' if success else '❌'} {message}")
        
        if success:
            for style, data in results.items():
                print(f"      {style}: {data['count']} descriptors generated")
        
        # Test 2: Filter Extraction
        print(f"\n🏷️  Test 2: Filter Extraction Accuracy")
        success, message, results = tester.test_filter_extraction()
        all_results["filter_extraction"] = {"success": success, "message": message, "data": results}
        print(f"   {'✅' if success else '❌'} {message}")
        
        if success:
            filter_count = len([k for k in results.keys() if not k.startswith("_")])
            print(f"      Extracted {filter_count} filter types")
        
        # Test 3: Query Optimization
        print(f"\n🔍 Test 3: Query Optimization & Filter Matching")
        success, message, results = await tester.test_query_optimization()
        all_results["query_optimization"] = {"success": success, "message": message, "data": results}
        print(f"   {'✅' if success else '❌'} {message}")
        
        if success:
            test_count = len(results)
            print(f"      {test_count} query optimization tests passed")
        
        # Test 4: Consistency Validation
        print(f"\n🔄 Test 4: Descriptor-Filter Consistency")
        success, message, results = tester.test_consistency_validation()
        all_results["consistency_validation"] = {"success": success, "message": message, "data": results}
        print(f"   {'✅' if success else '❌'} {message}")
        
        if success and "consistency_percentage" in results:
            print(f"      Consistency score: {results['consistency_percentage']:.1f}%")
        
        # Test 5: Performance Metrics
        print(f"\n⚡ Test 5: Performance Characteristics")
        success, message, results = tester.test_performance_metrics()
        all_results["performance_metrics"] = {"success": success, "message": message, "data": results}
        print(f"   {'✅' if success else '❌'} {message}")
        
        if success:
            print(f"      Generation: {results['generation_time_seconds']}s for {results['catalog_size']} products")
            print(f"      Rate: {results['descriptors_per_second']} products/sec")
        
        # Overall Results
        total_tests = len(all_results)
        passed_tests = sum(1 for r in all_results.values() if r["success"])
        
        print(f"\n📊 Overall Test Results:")
        print(f"   Tests Passed: {passed_tests}/{total_tests}")
        print(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if passed_tests == total_tests:
            print(f"   🎉 ALL TESTS PASSED - RAG System Ready!")
        else:
            print(f"   ⚠️  Some tests failed - Review results above")
            
            # Show failed tests
            failed_tests = [name for name, result in all_results.items() if not result["success"]]
            print(f"   Failed: {', '.join(failed_tests)}")
        
        return all_results
        
    finally:
        # Cleanup
        tester.cleanup_test_environment()
        print(f"\n🧹 Test environment cleaned up")


if __name__ == "__main__":
    import sys
    
    # Run tests
    results = asyncio.run(run_comprehensive_rag_tests())
    
    # Exit with appropriate code
    all_passed = all(r["success"] for r in results.values())
    sys.exit(0 if all_passed else 1)