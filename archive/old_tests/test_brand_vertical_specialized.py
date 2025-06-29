"""
Test Enhanced Brand Vertical Detection with Specialized.com

This script tests our enhanced brand vertical detection system that addresses user feedback:
1. Strategic sampling instead of random sampling
2. Direct web search questions
3. Complete AI decision transparency

Usage:
    export OPENAI_API_KEY='your-key'
    export TAVILY_API_KEY='your-key'  # Optional but recommended
    python test_brand_vertical_specialized.py
"""

import asyncio
import json
import logging
import os
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_brand_vertical_detection():
    """Test the enhanced brand vertical detection with specialized.com"""
    
    print("🚴 TESTING ENHANCED BRAND VERTICAL DETECTION")
    print("=" * 50)
    print("Brand: specialized.com")
    print("Enhancements: Strategic sampling + Direct questions + Paper trail")
    print()
    
    try:
        # Import the enhanced detector
        from src.descriptor import BrandVerticalDetector
        
        # Create detector instance
        detector = BrandVerticalDetector()
        
        print("⏱️  Starting brand vertical detection...")
        start_time = datetime.now()
        
        # Run the enhanced detection
        result = await detector.detect_brand_vertical("specialized.com")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"✅ Detection completed in {duration:.1f} seconds")
        print()
        
        # Display results
        print("🎯 DETECTION RESULTS:")
        print("-" * 20)
        print(f"Detected Vertical: {result['detected_vertical']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Analysis Methods: {result['analysis_methods']}")
        print()
        
        # Show web search results if available
        if result.get('web_search_data'):
            web_data = result['web_search_data']
            print("🌐 WEB SEARCH ANALYSIS:")
            print("-" * 22)
            print(f"Method: {web_data.get('method', 'unknown')}")
            print(f"Confidence: {web_data.get('confidence', 0):.2f}")
            print(f"Reasoning: {web_data.get('reasoning', 'Not provided')}")
            print("Evidence:")
            for i, evidence in enumerate(web_data.get('evidence', []), 1):
                print(f"  {i}. {evidence}")
            print()
        
        # Show product sampling results if available
        if result.get('product_analysis'):
            product_data = result['product_analysis']
            print("📦 STRATEGIC PRODUCT SAMPLING:")
            print("-" * 31)
            print(f"Method: {product_data.get('method', 'unknown')}")
            print(f"Total Products: {product_data.get('total_products', 'unknown')}")
            print(f"Sample Size: {product_data.get('sample_size', 'unknown')}")
            print(f"Confidence: {product_data.get('confidence', 0):.2f}")
            print(f"Sampling Method: {product_data.get('sampling_method', 'unknown')}")
            print("Evidence:")
            for i, evidence in enumerate(product_data.get('evidence', []), 1):
                print(f"  {i}. {evidence}")
            
            # Show category distribution if available
            if product_data.get('category_distribution'):
                print("\n📊 Category Distribution:")
                categories = product_data['category_distribution']
                for category, count in list(categories.items())[:10]:  # Top 10
                    print(f"  {category}: {count} products")
            print()
        
        # Show synthesis results
        if len(result['analysis_methods']) > 1:
            print("🧠 MULTI-SOURCE SYNTHESIS:")
            print("-" * 26)
            print(f"Final Vertical: {result['detected_vertical']}")
            print(f"Final Confidence: {result['confidence']:.2f}")
            print(f"Methods Combined: {', '.join(result['analysis_methods'])}")
            
            if result.get('synthesis_reasoning'):
                print(f"Synthesis Reasoning: {result['synthesis_reasoning']}")
            
            if result.get('method_weights'):
                print("Method Weights:")
                for method, weight in result['method_weights'].items():
                    print(f"  {method}: {weight:.1f}")
            print()
        
        # Performance summary
        print("⚡ PERFORMANCE SUMMARY:")
        print("-" * 21)
        print(f"Total Duration: {duration:.1f} seconds")
        print(f"Methods Used: {len(result['analysis_methods'])}")
        print(f"Success Rate: {'100%' if result['confidence'] > 0.5 else 'Low confidence'}")
        print()
        
        # Show complete JSON for debugging (truncated)
        print("🔍 COMPLETE RESULT (first 500 chars):")
        print("-" * 38)
        result_json = json.dumps(result, indent=2, default=str)
        print(result_json[:500] + "..." if len(result_json) > 500 else result_json)
        print()
        
        # Validation
        print("✅ VALIDATION:")
        print("-" * 13)
        
        expected_vertical = "cycling"
        if result['detected_vertical'].lower() == expected_vertical:
            print(f"✅ Correct vertical detected: {result['detected_vertical']}")
        else:
            print(f"❌ Expected '{expected_vertical}', got '{result['detected_vertical']}'")
        
        if result['confidence'] >= 0.8:
            print(f"✅ High confidence: {result['confidence']:.2f}")
        elif result['confidence'] >= 0.6:
            print(f"⚠️  Medium confidence: {result['confidence']:.2f}")
        else:
            print(f"❌ Low confidence: {result['confidence']:.2f}")
        
        if len(result['analysis_methods']) >= 2:
            print(f"✅ Multi-source analysis: {len(result['analysis_methods'])} methods")
        else:
            print(f"⚠️  Single-source analysis: {len(result['analysis_methods'])} method(s)")
        
        return result
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install openai anthropic aiohttp")
        return None
        
    except Exception as e:
        print(f"❌ Error during detection: {e}")
        logger.exception("Full error details:")
        return None

async def test_web_search_only():
    """Test just the web search functionality"""
    
    print("🌐 TESTING WEB SEARCH FUNCTIONALITY")
    print("=" * 35)
    
    try:
        from src.web_search import get_web_search_engine
        
        web_search = get_web_search_engine()
        
        if not web_search.is_available():
            print("❌ No web search providers available")
            print("Available providers:", web_search.get_provider_status())
            return None
        
        print("✅ Web search providers available")
        print("Provider status:", web_search.get_provider_status())
        print()
        
        print("🔍 Searching for specialized.com brand information...")
        start_time = datetime.now()
        
        search_results = await web_search.search_brand_info("specialized.com")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"✅ Search completed in {duration:.1f} seconds")
        print()
        
        print("📊 SEARCH RESULTS:")
        print("-" * 16)
        print(f"Total Results: {search_results['total_results']}")
        print(f"Provider Used: {search_results['provider_used']}")
        print(f"Search Strategy: {search_results['search_strategy']}")
        print()
        
        # Show top 3 results
        print("🔍 TOP SEARCH RESULTS:")
        for i, result in enumerate(search_results['results'][:3], 1):
            print(f"\n{i}. {result.get('title', 'No title')}")
            print(f"   URL: {result.get('url', 'No URL')}")
            print(f"   Query: {result.get('query', 'No query')}")
            print(f"   Snippet: {result.get('snippet', 'No snippet')[:150]}...")
        
        return search_results
        
    except Exception as e:
        print(f"❌ Error in web search test: {e}")
        logger.exception("Full error details:")
        return None

def check_environment():
    """Check if required environment variables are set"""
    
    print("🔧 ENVIRONMENT CHECK")
    print("=" * 19)
    
    required_vars = ['OPENAI_API_KEY']
    optional_vars = ['TAVILY_API_KEY', 'ANTHROPIC_API_KEY', 'GEMINI_API_KEY']
    
    all_good = True
    
    print("Required API Keys:")
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var}: Set (***{value[-4:]})")
        else:
            print(f"  ❌ {var}: Not set")
            all_good = False
    
    print("\nOptional API Keys:")
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var}: Set (***{value[-4:]})")
        else:
            print(f"  ➖ {var}: Not set")
    
    print()
    return all_good

async def main():
    """Main test function"""
    
    print("🧪 ENHANCED BRAND VERTICAL DETECTION TEST")
    print("=" * 42)
    print("Testing improvements based on user feedback:")
    print("  1. Strategic sampling (no more random sampling misses)")
    print("  2. Direct web search questions")
    print("  3. AI decision transparency (paper trail)")
    print()
    
    # Check environment
    if not check_environment():
        print("❌ Please set required environment variables and try again.")
        return
    
    # Test web search first
    print("=" * 50)
    web_results = await test_web_search_only()
    
    print("\n" + "=" * 50)
    
    # Test full brand vertical detection
    detection_results = await test_brand_vertical_detection()
    
    print("\n" + "=" * 50)
    print("🎉 TEST COMPLETE!")
    
    if detection_results and detection_results.get('detected_vertical') == 'cycling':
        print("✅ SUCCESS: Specialized.com correctly identified as cycling brand!")
    else:
        print("⚠️  Review needed - check results above")

if __name__ == "__main__":
    asyncio.run(main())
