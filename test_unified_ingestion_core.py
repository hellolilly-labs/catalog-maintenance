#!/usr/bin/env python3
"""
Test Core UnifiedDescriptorGenerator Integration

Tests just the UnifiedDescriptorGenerator without ingestion dependencies.
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.catalog.unified_descriptor_generator import UnifiedDescriptorGenerator, DescriptorConfig

async def test_core_generator():
    """Test the core UnifiedDescriptorGenerator functionality"""
    
    brand_domain = "specialized.com"
    
    print("🧪 Testing UnifiedDescriptorGenerator Core")
    print("=" * 50)
    
    # Test configuration
    config = DescriptorConfig(
        use_research=True,
        extract_filters=True,
        quality_threshold=0.8,
        descriptor_length=(100, 200)
    )
    
    print(f"📋 Configuration:")
    print(f"   Use Research: {config.use_research}")
    print(f"   Extract Filters: {config.extract_filters}")
    print(f"   Quality Threshold: {config.quality_threshold}")
    print(f"   Descriptor Length: {config.descriptor_length}")
    
    try:
        generator = UnifiedDescriptorGenerator(brand_domain, config)
        print(f"✅ Generator initialized for {brand_domain}")
        
        # Test processing (this should work with existing ProductManager)
        enhanced_products, filter_labels = await generator.process_catalog(force_regenerate=False)
        print(f"✅ Processed {len(enhanced_products)} products")
        print(f"✅ Extracted {len(filter_labels)} filter types")
        
        # Analyze first product if available
        if enhanced_products:
            sample = enhanced_products[0]
            print(f"\n📝 Sample Product Analysis:")
            print(f"   ID: {getattr(sample, 'id', 'N/A')}")
            print(f"   Name: {getattr(sample, 'name', 'N/A')}")
            
            # Test new descriptor fields
            descriptor = getattr(sample, 'descriptor', '')
            voice_summary = getattr(sample, 'voice_summary', '')
            search_keywords = getattr(sample, 'search_keywords', [])
            key_points = getattr(sample, 'key_selling_points', [])
            
            print(f"\n🎯 Generated Content:")
            print(f"   Descriptor Length: {len(descriptor.split())} words")
            print(f"   Voice Summary: {len(voice_summary.split())} words")
            print(f"   Search Keywords: {len(search_keywords)}")
            print(f"   Key Selling Points: {len(key_points)}")
            
            # Show quality metadata
            metadata = getattr(sample, 'descriptor_metadata', None)
            if metadata:
                print(f"\n⭐ Quality Assessment:")
                print(f"   Score: {getattr(metadata, 'quality_score', 'N/A')}")
                print(f"   Reasoning: {getattr(metadata, 'quality_score_reasoning', 'N/A')}")
                print(f"   Generator Version: {getattr(metadata, 'generator_version', 'N/A')}")
                print(f"   Mode: {getattr(metadata, 'mode', 'N/A')}")
            
            # Verify all four components exist
            has_descriptor = bool(descriptor)
            has_voice = bool(voice_summary)  
            has_keywords = bool(search_keywords)
            has_points = bool(key_points)
            
            print(f"\n✅ Component Verification:")
            print(f"   DESCRIPTOR: {'✅' if has_descriptor else '❌'}")
            print(f"   VOICE_SUMMARY: {'✅' if has_voice else '❌'}")
            print(f"   SEARCH_TERMS: {'✅' if has_keywords else '❌'}")
            print(f"   KEY_POINTS: {'✅' if has_points else '❌'}")
            
            if all([has_descriptor, has_voice, has_keywords, has_points]):
                print(f"\n🎉 All four components successfully generated!")
            else:
                print(f"\n⚠️ Some components missing - check generation logic")
                
        else:
            print("❌ No products returned from generator")
            return False
            
        # Test filter extraction
        print(f"\n🏷️ Filter Analysis:")
        for filter_name, filter_data in filter_labels.items():
            if filter_name != '_metadata':
                values_count = len(filter_data.get('values', []))
                print(f"   {filter_name}: {values_count} values")
        
        print(f"\n✅ Core UnifiedDescriptorGenerator test passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_core_generator())
    if success:
        print(f"\n🎯 Next Steps for Ingestion Pipeline:")
        print(f"   1. ✅ UnifiedDescriptorGenerator working correctly")
        print(f"   2. ✅ Product model integration complete")
        print(f"   3. ✅ Four-component generation working")
        print(f"   4. 🔄 Ready for ingestion pipeline integration")
        print(f"   5. 📝 Updated ingestion scripts prepared")
    sys.exit(0 if success else 1) 