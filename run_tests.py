#!/usr/bin/env python3
"""
Test Runner for Catalog Maintenance System

Runs all important tests in order:
1. RAG Integration Test (quick validation)
2. Full RAG System Test (comprehensive)
3. Filter Extraction Tests  
4. System Prompt Builder Tests

Usage:
    python run_tests.py                # Run all tests
    python run_tests.py --quick        # Run only RAG integration test
    python run_tests.py --filter       # Run only filter tests
"""

import asyncio
import subprocess
import sys
from pathlib import Path

class TestRunner:
    """Test runner for catalog maintenance system"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.test_results = {}
    
    def run_test(self, test_name: str, test_command: list) -> bool:
        """Run a single test and capture results"""
        print(f"\n{'='*60}")
        print(f"🧪 Running: {test_name}")
        print(f"{'='*60}")
        
        try:
            result = subprocess.run(
                test_command,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            
            if result.returncode == 0:
                print(f"✅ {test_name} PASSED")
                self.passed += 1
                self.test_results[test_name] = "PASSED"
                
                # Show key output lines
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if '✅' in line or '🎉' in line or 'PASSED' in line:
                        print(f"   {line}")
                
                return True
            else:
                print(f"❌ {test_name} FAILED")
                print(f"Exit code: {result.returncode}")
                print(f"STDOUT:\n{result.stdout}")
                print(f"STDERR:\n{result.stderr}")
                self.failed += 1
                self.test_results[test_name] = "FAILED"
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏱️ {test_name} TIMEOUT")
            self.failed += 1
            self.test_results[test_name] = "TIMEOUT"
            return False
        except Exception as e:
            print(f"💥 {test_name} ERROR: {e}")
            self.failed += 1
            self.test_results[test_name] = "ERROR"
            return False
    
    def run_quick_tests(self):
        """Run quick validation tests"""
        print("🚀 Running Quick Tests")
        
        tests = [
            ("RAG Integration Test", ["python", "-m", "tests.test_rag_integration"])
        ]
        
        for test_name, command in tests:
            self.run_test(test_name, command)
    
    def run_filter_tests(self):
        """Run filter-related tests"""
        print("🏷️ Running Filter Tests")
        
        tests = [
            ("Dynamic Catalog Analysis", ["python", "-m", "tests.test_dynamic_catalog_analysis"]),
            ("Filter Extraction", ["python", "-m", "tests.test_filter_extraction"]),
            ("Proper Filter Flow", ["python", "-m", "tests.test_proper_filter_flow"])
        ]
        
        for test_name, command in tests:
            self.run_test(test_name, command)
    
    def run_comprehensive_tests(self):
        """Run comprehensive test suite"""
        print("🧪 Running Comprehensive Tests")
        
        tests = [
            ("RAG System (Comprehensive)", ["python", "-m", "tests.test_rag_system"]),
            ("System Prompt Builder", ["python", "-m", "tests.test_system_prompt_builder"])
        ]
        
        for test_name, command in tests:
            self.run_test(test_name, command)
    
    def run_all_tests(self):
        """Run all tests in order"""
        print("🎯 Running All Tests")
        
        # Quick validation first
        self.run_quick_tests()
        
        # If quick tests pass, run comprehensive
        if self.test_results.get("RAG Integration Test") == "PASSED":
            self.run_filter_tests()
            self.run_comprehensive_tests()
        else:
            print("⚠️ Quick tests failed, skipping comprehensive tests")
    
    def print_summary(self):
        """Print test summary"""
        total = self.passed + self.failed
        
        print(f"\n{'='*60}")
        print(f"📊 TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"Success Rate: {(self.passed/total)*100:.1f}%" if total > 0 else "No tests run")
        
        if self.test_results:
            print(f"\n📋 Individual Results:")
            for test_name, result in self.test_results.items():
                status_emoji = "✅" if result == "PASSED" else "❌"
                print(f"   {status_emoji} {test_name}: {result}")
        
        if self.failed == 0:
            print(f"\n🎉 ALL TESTS PASSED! Ready for Pinecone integration.")
        else:
            print(f"\n⚠️ {self.failed} test(s) failed. Review results above.")


def main():
    """Main test runner"""
    runner = TestRunner()
    
    if len(sys.argv) > 1:
        if "--quick" in sys.argv:
            runner.run_quick_tests()
        elif "--filter" in sys.argv:
            runner.run_filter_tests()
        elif "--comprehensive" in sys.argv:
            runner.run_comprehensive_tests()
        else:
            print("Unknown option. Use --quick, --filter, --comprehensive, or no args for all tests.")
            sys.exit(1)
    else:
        runner.run_all_tests()
    
    runner.print_summary()
    
    # Exit with appropriate code
    sys.exit(0 if runner.failed == 0 else 1)


if __name__ == "__main__":
    main()