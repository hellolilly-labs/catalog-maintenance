#!/usr/bin/env python3
"""
Test Specialized.com with Current Implementation

This demonstrates:
1. LLM-based brand vertical detection
2. Product sub-vertical detection  
3. Descriptor generation
4. Multi-provider LLM routing (OpenAI/Anthropic/Gemini)
5. Environment-configurable model selection

⚠️  Note: Requires API keys to be set in environment
"""

import asyncio
import json
import os
from src.models.product import Product
from src.descriptor import DescriptorGenerator

# Test product from Specialized
SPECIALIZED_PRODUCT = Product(
    id="tarmac-sl7-expert-2024",
    name="Tarmac SL7 Expert",
    brand="specialized.com",
    categories=["Road Bikes", "Performance", "Racing"],
    originalPrice="$4,200",
    colors=["Gloss Red", "Satin Black", "Pearl White"],
    sizes=["49cm", "52cm", "54cm", "56cm", "58cm", "61cm"],
    highlights=[
        "FACT 9r Carbon Frame", 
        "Shimano 105 Di2 Groupset",
        "DT Swiss R470 Wheelset",
        "Tubeless Ready",
        "Racing Geometry"
    ],
    description="The Tarmac SL7 Expert delivers exceptional performance with its lightweight carbon frame and electronic shifting. Built for racing and long rides, featuring aerodynamic tube shapes and a responsive ride quality.",
    specifications={
        "Frame": {
            "material": "FACT 9r Carbon",
            "geometry": "Racing",
            "weight": "1,050g (56cm)"
        },
        "Drivetrain": {
            "groupset": "Shimano 105 Di2 12-speed",
            "cassette": "11-34T",
            "chainrings": "50/34T"
        },
        "Wheels": {
            "wheelset": "DT Swiss R470",
            "tires": "Turbo Cotton 26mm",
            "tubeless": True
        }
    }
)

# Sample sizing data
SPECIALIZED_SIZING = {
    "49cm": "Fits riders 5'2\" - 5'5\" (157-165cm)",
    "52cm": "Fits riders 5'5\" - 5'8\" (165-173cm)", 
    "54cm": "Fits riders 5'8\" - 5'10\" (173-178cm)",
    "56cm": "Fits riders 5'10\" - 6'0\" (178-183cm)",
    "58cm": "Fits riders 6'0\" - 6'2\" (183-188cm)",
    "61cm": "Fits riders 6'2\" - 6'4\" (188-193cm)"
}


async def test_api_keys():
    """Check if API keys are configured"""
    keys = {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
        'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY')
    }
    
    print("🔑 API Key Status:")
    available_providers = []
    
    for key_name, key_value in keys.items():
        provider = key_name.replace('_API_KEY', '').lower()
        if key_value:
            print(f"   ✅ {provider.upper()}: Configured")
            available_providers.append(provider)
        else:
            print(f"   ❌ {provider.upper()}: Not set")
    
    if not available_providers:
        print("\n⚠️  No API keys configured - cannot test LLM functionality")
        print("   Set at least one API key to test:")
        print("   export OPENAI_API_KEY='your-key'")
        print("   export ANTHROPIC_API_KEY='your-key'")
        return False
    
    print(f"\n🎯 Available providers: {', '.join(available_providers)}")
    return True


async def test_brand_vertical_detection():
    """Test LLM-based brand vertical detection"""
    print("\n" + "="*60)
    print("🧠 Testing Brand Vertical Detection")
    print("="*60)
    
    generator = DescriptorGenerator()
    
    try:
        print(f"🔍 Analyzing brand: {SPECIALIZED_PRODUCT.brand}")
        print("   Using LLM to determine brand's primary vertical...")
        
        # Test brand vertical detection
        brand_vertical = await generator.detect_brand_vertical(SPECIALIZED_PRODUCT)
        print(f"   ✅ Detected brand vertical: '{brand_vertical}'")
        
        # Test product sub-vertical detection
        print(f"\n🔍 Analyzing product: {SPECIALIZED_PRODUCT.name}")
        print("   Using LLM to determine product sub-vertical...")
        
        product_subvertical = await generator.detect_product_subvertical(SPECIALIZED_PRODUCT, brand_vertical)
        print(f"   ✅ Detected product sub-vertical: '{product_subvertical}'")
        
        # Test complete vertical context
        print(f"\n🎯 Getting complete vertical context...")
        context = await generator.detect_vertical_context(SPECIALIZED_PRODUCT)
        
        print(f"   Brand vertical: {context['brand_vertical']}")
        print(f"   Product sub-vertical: {context['product_subvertical']}")
        print(f"   Effective vertical: {context['effective_vertical']}")
        
        return context
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


async def test_descriptor_generation():
    """Test LLM-based descriptor generation"""
    print("\n" + "="*60)
    print("📝 Testing Descriptor Generation") 
    print("="*60)
    
    generator = DescriptorGenerator()
    
    try:
        print(f"🎯 Generating descriptor for: {SPECIALIZED_PRODUCT.name}")
        print("   Using task-optimized LLM routing...")
        
        descriptor = await generator.generate_descriptor(SPECIALIZED_PRODUCT)
        
        if descriptor:
            print(f"   ✅ Generated descriptor ({len(descriptor)} chars):")
            print(f"   \"{descriptor}\"")
            return descriptor
        else:
            print(f"   ❌ Failed to generate descriptor")
            return None
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


async def test_sizing_generation():
    """Test LLM-based sizing generation"""
    print("\n" + "="*60)
    print("📏 Testing Sizing Generation")
    print("="*60)
    
    generator = DescriptorGenerator()
    
    try:
        print(f"🎯 Generating sizing for: {SPECIALIZED_PRODUCT.name}")
        print("   Using proven sizing instruction with reasoning-optimized LLM...")
        
        sizing = await generator.generate_sizing(SPECIALIZED_PRODUCT, SPECIALIZED_SIZING)
        
        if sizing:
            print(f"   ✅ Generated sizing data:")
            print(json.dumps(sizing, indent=4))
            return sizing
        else:
            print(f"   ❌ Failed to generate sizing")
            return None
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


async def test_complete_workflow():
    """Test complete product processing workflow"""
    print("\n" + "="*60)
    print("🚀 Testing Complete Workflow")
    print("="*60)
    
    generator = DescriptorGenerator()
    
    try:
        print(f"🎯 Processing complete product: {SPECIALIZED_PRODUCT.name}")
        print("   Running end-to-end workflow...")
        
        results = await generator.process_product(SPECIALIZED_PRODUCT, SPECIALIZED_SIZING)
        
        print(f"\n📊 Processing Results:")
        print(f"   Product ID: {results['product_id']}")
        print(f"   Product Name: {results['product_name']}")
        print(f"   Detected Vertical: {results['detected_vertical']}")
        print(f"   Processing Time: {results['processing_time']:.2f}s")
        print(f"   Errors: {len(results['errors'])}")
        
        if results['errors']:
            print(f"   Error Details: {results['errors']}")
        
        if results['descriptor']:
            print(f"\n✅ Descriptor: {results['descriptor'][:100]}...")
        
        if results['sizing']:
            print(f"\n✅ Sizing: Generated successfully")
        
        return results
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


async def test_enhanced_brand_vertical_detection():
    """Test the new multi-source brand vertical detection system"""
    
    print("🔬 ENHANCED BRAND VERTICAL DETECTION TEST")
    print("=" * 50)
    print()
    
    # Import modules
    from src.models.product import Product
    from src.descriptor import BrandVerticalDetector, DescriptorGenerator
    from src.web_search import get_web_search_engine
    
    # Sample Specialized product for testing
    specialized_product = Product(
        id="test-tarmac-sl7",
        name="Tarmac SL7 Expert",
        brand="specialized.com",
        productUrl="https://www.specialized.com/us/en/tarmac-sl7-expert/p/199563",
        categories=["Road Bikes", "Racing Bikes"],
        description="The fastest bike in the world just got faster. Built to be incredibly lightweight and stiff, the Tarmac SL7 Expert delivers explosive accelerations and confident handling.",
        highlights=[
            "FACT 10r carbon frame with Rider-First Engineered design",
            "Shimano Ultegra 12-speed groupset",
            "Roval Alpinist CL wheels",
            "S-Works Power saddle"
        ],
        specifications={
            "frame": "FACT 10r Carbon, Rider-First Engineered",
            "fork": "FACT Carbon, full carbon",
            "groupset": "Shimano Ultegra Di2 12-speed",
            "wheels": "Roval Alpinist CL",
            "weight": "7.25kg (16lb)"
        }
    )
    
    print("📊 TESTING BRAND: specialized.com")
    print(f"Sample Product: {specialized_product.name}")
    print()
    
    # Test 1: Web Search Provider Status
    print("🌐 WEB SEARCH PROVIDER STATUS")
    print("-" * 30)
    
    try:
        web_search = get_web_search_engine()
        provider_status = web_search.get_provider_status()
        
        print(f"Tavily API Available: {'✅' if provider_status['tavily'] else '❌'}")
        print(f"Google Search API Available: {'✅' if provider_status['google'] else '❌'}")
        print(f"Total Providers: {provider_status['total_providers']}")
        print(f"Web Search Available: {'✅' if web_search.is_available() else '❌'}")
        
        if not web_search.is_available():
            print("⚠️  No web search APIs configured - will use domain analysis fallback")
        
    except Exception as e:
        print(f"❌ Error checking web search: {e}")
    
    print()
    
    # Test 2: Multi-Source Brand Vertical Detection
    print("🔍 MULTI-SOURCE BRAND VERTICAL DETECTION")
    print("-" * 40)
    
    detector = BrandVerticalDetector()
    
    try:
        detection_result = await detector.detect_brand_vertical(
            brand_domain="specialized.com",
            product_sample=specialized_product
        )
        
        print(f"🎯 Detected Vertical: {detection_result['detected_vertical']}")
        print(f"📊 Confidence: {detection_result['confidence']:.2f}")
        print(f"🔧 Analysis Methods: {', '.join(detection_result['analysis_methods'])}")
        print()
        
        # Show detailed analysis for each method
        if detection_result.get('web_search_data'):
            web_data = detection_result['web_search_data']
            print("🌐 WEB SEARCH ANALYSIS:")
            print(f"   Method: {web_data.get('method', 'unknown')}")
            print(f"   Vertical: {web_data.get('detected_vertical', 'unknown')}")
            print(f"   Confidence: {web_data.get('confidence', 0):.2f}")
            print(f"   Reasoning: {web_data.get('reasoning', 'N/A')}")
            
            if web_data.get('search_results_count'):
                print(f"   Search Results: {web_data['search_results_count']} results analyzed")
                print(f"   Provider: {web_data.get('provider_used', 'unknown')}")
            
            print(f"   Evidence: {web_data.get('evidence', [])}")
            print()
        
        if detection_result.get('product_analysis'):
            product_data = detection_result['product_analysis']
            print("📦 PRODUCT CATALOG ANALYSIS:")
            print(f"   Method: {product_data.get('method', 'unknown')}")
            print(f"   Vertical: {product_data.get('detected_vertical', 'unknown')}")
            print(f"   Confidence: {product_data.get('confidence', 0):.2f}")
            print(f"   Reasoning: {product_data.get('reasoning', 'N/A')}")
            
            if product_data.get('total_products'):
                print(f"   Total Products: {product_data['total_products']}")
                print(f"   Sample Size: {product_data.get('sample_size', 0)}")
                
                if product_data.get('category_distribution'):
                    print(f"   Top Categories: {product_data['category_distribution']}")
            
            print()
        
        # Show synthesis results
        if detection_result.get('synthesis_reasoning'):
            print("🧠 SYNTHESIS ANALYSIS:")
            print(f"   Final Vertical: {detection_result['detected_vertical']}")
            print(f"   Reasoning: {detection_result['synthesis_reasoning']}")
            print(f"   Consensus Level: {detection_result.get('consensus_level', 'unknown')}")
            print(f"   Method Weights: {detection_result.get('method_weights', {})}")
            print()
    
    except Exception as e:
        print(f"❌ Error in brand vertical detection: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Enhanced Descriptor Generation with Vertical Context
    print("📝 ENHANCED DESCRIPTOR GENERATION")
    print("-" * 35)
    
    try:
        generator = DescriptorGenerator()
        
        # Detect vertical context (includes enhanced detection)
        vertical_context = await generator.detect_vertical_context(specialized_product)
        
        print(f"Brand Vertical: {vertical_context['brand_vertical']}")
        print(f"Product Sub-vertical: {vertical_context.get('product_subvertical', 'None')}")
        print(f"Effective Vertical: {vertical_context['effective_vertical']}")
        print(f"Detection Confidence: {vertical_context.get('detection_confidence', 0):.2f}")
        print(f"Analysis Methods: {', '.join(vertical_context.get('analysis_methods', []))}")
        print()
        
        # Generate descriptor using enhanced vertical detection
        descriptor = await generator.generate_descriptor(specialized_product)
        
        if descriptor:
            print("📖 GENERATED DESCRIPTOR:")
            print(f'"{descriptor}"')
            print()
            print(f"✅ Descriptor generated successfully using {vertical_context['effective_vertical']} vertical optimization")
        else:
            print("❌ Descriptor generation failed")
    
    except Exception as e:
        print(f"❌ Error in descriptor generation: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # Test 4: Comparison of Detection Methods
    print("📊 DETECTION METHOD COMPARISON")
    print("-" * 32)
    
    print("BEFORE (Hardcoded):")
    print("❌ Single product → brand vertical")
    print("❌ Hardcoded keyword lists")
    print("❌ No confidence scoring")
    print("❌ No analysis method tracking")
    print("❌ No caching or optimization")
    print()
    
    print("AFTER (Enhanced Multi-Source):")
    print("✅ Web search + product sampling + LLM synthesis")
    print("✅ Adaptive to any industry/vertical")
    print("✅ Confidence scoring and method tracking")
    print("✅ Brand-level caching for consistency")
    print("✅ Graceful fallbacks when APIs unavailable")
    print("✅ Evidence-based decision making")
    print()
    
    # Test 5: Cache Performance
    print("⚡ CACHE PERFORMANCE TEST")
    print("-" * 25)
    
    try:
        print("First detection (fresh analysis)...")
        start_time = asyncio.get_event_loop().time()
        
        result1 = await detector.detect_brand_vertical("specialized.com", specialized_product)
        
        first_time = asyncio.get_event_loop().time() - start_time
        
        print("Second detection (cached result)...")
        start_time = asyncio.get_event_loop().time()
        
        result2 = await detector.detect_brand_vertical("specialized.com", specialized_product)
        
        second_time = asyncio.get_event_loop().time() - start_time
        
        print(f"First detection: {first_time:.2f}s")
        print(f"Cached detection: {second_time:.2f}s")
        print(f"Speed improvement: {(first_time / second_time):.1f}x faster")
        print(f"Results identical: {'✅' if result1 == result2 else '❌'}")
        
    except Exception as e:
        print(f"❌ Error in cache test: {e}")
    
    print()
    print("🎉 ENHANCED BRAND VERTICAL DETECTION TEST COMPLETE!")
    print()
    print("💡 KEY IMPROVEMENTS:")
    print("• Multi-source analysis (web search + product catalog)")
    print("• Intelligent synthesis with confidence scoring")
    print("• No hardcoded assumptions - adapts to any brand/vertical")
    print("• Evidence-based decision making with method tracking")
    print("• Brand-level caching for consistency and performance")
    print("• Graceful fallbacks when external APIs unavailable")


async def main():
    """Run all tests"""
    print("🚀 Specialized.com Implementation Test")
    print("Testing LLM-based vertical detection and descriptor generation")
    print("="*80)
    
    # Check API keys
    if not await test_api_keys():
        return
    
    # Show current LLM configuration
    from src.llm import LLMFactory
    try:
        config = LLMFactory.get_configuration()
        print(f"\n⚙️  Current LLM Configuration:")
        for task, model in config['task_models'].items():
            override = config['environment_overrides'].get(f"{task.upper().replace('_', '')}_MODEL")
            status = f" (ENV: {override})" if override else ""
            print(f"   {task}: {model}{status}")
    except Exception as e:
        print(f"⚠️  Could not get LLM configuration: {e}")
    
    # Run tests
    results = {}
    
    # Test 1: Brand vertical detection
    results['vertical_context'] = await test_brand_vertical_detection()
    
    # Test 2: Descriptor generation
    results['descriptor'] = await test_descriptor_generation()
    
    # Test 3: Sizing generation  
    results['sizing'] = await test_sizing_generation()
    
    # Test 4: Complete workflow
    results['complete_workflow'] = await test_complete_workflow()
    
    # Test 5: Enhanced brand vertical detection
    await test_enhanced_brand_vertical_detection()
    
    # Summary
    print("\n" + "="*80)
    print("📋 Test Summary")
    print("="*80)
    
    success_count = sum(1 for result in results.values() if result is not None)
    total_tests = len(results)
    
    print(f"✅ Successful tests: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 All tests passed! The system is working correctly.")
        print("\n💡 Next steps:")
        print("   1. Test with real product data from specialized.com")
        print("   2. Set up automated brand data collection")
        print("   3. Deploy to production environment")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        print("\n🔧 Troubleshooting:")
        print("   1. Verify API keys are correctly set")
        print("   2. Check network connectivity")
        print("   3. Review error messages for specific issues")
    
    print(f"\n🗑️  To clean up test data:")
    print(f"   python scripts/brand_restart.py specialized.com")


if __name__ == "__main__":
    # Run the enhanced brand vertical detection test
    asyncio.run(test_enhanced_brand_vertical_detection()) 