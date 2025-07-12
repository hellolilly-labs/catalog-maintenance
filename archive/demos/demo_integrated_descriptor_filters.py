#!/usr/bin/env python3
"""
Demo: Integrated Descriptor Generation + Filter Extraction

Demonstrates how the EnhancedDescriptorGenerator processes a catalog
to produce both RAG-optimized descriptors and filter labels in a single pass.

This addresses the user's suggestion:
"We will need to ingest the entire product catalog to come up with the complete 
set of labels. But perhaps we can do this when we generate the 'descriptor' 
for each product that we will use for RAG?"
"""

import json
import asyncio
from pathlib import Path
from src.catalog.enhanced_descriptor_generator import EnhancedDescriptorGenerator, generate_enhanced_catalog

async def demo_integrated_processing():
    """Demonstrate integrated descriptor + filter processing"""
    
    print("🏭 Demo: Integrated Descriptor Generation + Filter Extraction")
    print("=" * 70)
    
    # Load sample catalog
    catalog_path = Path("sample_specialized_catalog.json")
    
    if not catalog_path.exists():
        print(f"❌ Sample catalog not found: {catalog_path}")
        return
    
    with open(catalog_path) as f:
        catalog_data = json.load(f)
    
    products = catalog_data.get("products", [])
    
    print(f"📦 Processing {len(products)} products from specialized.com")
    print("   Single pass: Descriptors + Filters + Consistency")
    
    # Create generator
    generator = EnhancedDescriptorGenerator("specialized.com")
    
    print(f"\n🔄 Step 1: Process Catalog (Single Pass)")
    print("   ✅ Generate voice-optimized descriptors")
    print("   ✅ Extract filter labels from actual products")
    print("   ✅ Ensure descriptor-filter consistency")
    
    # Process catalog in single pass
    enhanced_descriptors, filter_labels = generator.process_catalog(
        products, 
        descriptor_style="voice_optimized"
    )
    
    print(f"\n✅ Results:")
    print(f"   📝 Generated {len(enhanced_descriptors)} enhanced descriptors")
    print(f"   🏷️  Extracted {len([k for k in filter_labels.keys() if not k.startswith('_')])} filter types")
    
    # Show sample enhanced descriptor
    print(f"\n📋 Sample Enhanced Product:")
    sample_product = enhanced_descriptors[0]
    print(f"   🚲 Product: {sample_product['name']}")
    print(f"   💬 Voice Summary: {sample_product['voice_summary']}")
    print(f"   🔍 RAG Keywords: {sample_product['rag_keywords'][:5]}...")
    print(f"   📝 Enhanced Description (first 100 chars):")
    print(f"      {sample_product['enhanced_description'][:100]}...")
    
    # Show extracted filters
    print(f"\n🏷️  Extracted Filter Labels (from actual products):")
    for filter_name, filter_config in filter_labels.items():
        if filter_name.startswith('_'):
            continue
            
        filter_type = filter_config.get('type')
        if filter_type == "categorical":
            values = filter_config.get('values', [])
            print(f"   📂 {filter_name}: {values}")
        elif filter_type == "multi_select":
            values = filter_config.get('values', [])
            print(f"   ☑️  {filter_name}: {values}")
        elif filter_type == "numeric_range":
            min_val = filter_config.get('min')
            max_val = filter_config.get('max')
            ranges = [r['label'] for r in filter_config.get('common_ranges', [])]
            print(f"   📊 {filter_name}: {min_val}-{max_val} (ranges: {ranges})")
    
    # Verify consistency between descriptors and filters
    print(f"\n🔍 Consistency Check:")
    print("   Verifying that descriptor text includes filter terms...")
    
    consistency_checks = 0
    for descriptor in enhanced_descriptors[:3]:  # Check first 3
        enhanced_desc = descriptor.get('enhanced_description', '').lower()
        
        # Check category mentioned
        category = descriptor.get('category', '')
        if category and category in enhanced_desc:
            consistency_checks += 1
            print(f"   ✅ '{descriptor['name']}' mentions category '{category}'")
        
        # Check material mentioned
        material = descriptor.get('frame_material', '')
        if material and material in enhanced_desc:
            consistency_checks += 1
            print(f"   ✅ '{descriptor['name']}' mentions material '{material}'")
    
    print(f"   📊 Consistency score: {consistency_checks}/6 checks passed")
    
    return enhanced_descriptors, filter_labels


async def demo_different_descriptor_styles():
    """Show different descriptor styles available"""
    
    print(f"\n\n🎨 Demo: Different Descriptor Styles")
    print("=" * 70)
    
    # Load sample product
    catalog_path = Path("sample_specialized_catalog.json")
    with open(catalog_path) as f:
        catalog_data = json.load(f)
    
    sample_product = catalog_data["products"][0]  # Tarmac SL8 Expert
    print(f"📦 Sample Product: {sample_product['name']}")
    
    generator = EnhancedDescriptorGenerator("specialized.com")
    
    # Test different styles
    styles = ["voice_optimized", "detailed", "concise"]
    
    for style in styles:
        print(f"\n🎯 Style: {style}")
        print("-" * 40)
        
        # Generate single product with this style
        enhanced_products, _ = generator.process_catalog([sample_product], style)
        enhanced_desc = enhanced_products[0]['enhanced_description']
        
        print(enhanced_desc)
    
    print(f"\n💡 Style Selection Guide:")
    print("   🗣️  voice_optimized: Natural language for AI conversations")
    print("   📋 detailed: Comprehensive with specifications")
    print("   ⚡ concise: Quick reference format")


async def demo_performance_benefits():
    """Demonstrate performance benefits of integrated approach"""
    
    print(f"\n\n⚡ Demo: Performance Benefits")
    print("=" * 70)
    
    print("🎯 Single-Pass Processing Benefits:")
    print("   ✅ One catalog iteration instead of two")
    print("   ✅ Consistent terminology across descriptors and filters")
    print("   ✅ Reduced memory usage")
    print("   ✅ Faster overall processing")
    
    print(f"\n📊 Architecture Comparison:")
    print("   ❌ OLD: Separate Processes")
    print("      1. Generate descriptors (iterate catalog)")
    print("      2. Extract filters (iterate catalog again)")
    print("      3. Hope for consistency")
    print("   ")
    print("   ✅ NEW: Integrated Process")
    print("      1. Generate descriptors + extract filters (single iteration)")
    print("      2. Ensure consistency automatically")
    print("      3. Enhanced descriptors use filter terminology")
    
    print(f"\n🔄 When to Run:")
    print("   📦 During catalog ingestion (when products change)")
    print("   ⚡ Once per catalog update, not per query")
    print("   💾 Results cached for real-time use")


async def demo_factory_function():
    """Demonstrate the convenience factory function"""
    
    print(f"\n\n🏭 Demo: Factory Function Usage")
    print("=" * 70)
    
    # Load sample catalog
    catalog_path = Path("sample_specialized_catalog.json")
    with open(catalog_path) as f:
        catalog_data = json.load(f)
    
    products = catalog_data.get("products", [])
    
    print("🚀 Using convenience factory function:")
    print("   generate_enhanced_catalog(brand_domain, catalog_data, style)")
    
    # Use factory function
    enhanced_descriptors, filter_labels = generate_enhanced_catalog(
        brand_domain="specialized.com",
        catalog_data=products,
        descriptor_style="voice_optimized"
    )
    
    print(f"✅ Factory function completed:")
    print(f"   📝 {len(enhanced_descriptors)} enhanced descriptors")
    print(f"   🏷️  {len(filter_labels)} filter types")
    print(f"   💾 Automatically saved to accounts/specialized.com/")
    
    # Show what files were created
    brand_path = Path("accounts/specialized.com")
    created_files = []
    
    if (brand_path / "enhanced_product_catalog.json").exists():
        created_files.append("enhanced_product_catalog.json")
    if (brand_path / "catalog_filters.json").exists():
        created_files.append("catalog_filters.json")
    
    print(f"   📁 Created files: {created_files}")


async def main():
    """Run all demos"""
    
    await demo_integrated_processing()
    await demo_different_descriptor_styles()
    await demo_performance_benefits()
    await demo_factory_function()
    
    print(f"\n\n🎉 Integrated Descriptor + Filter Demo Complete!")
    print(f"\n✅ Key Achievements:")
    print("   🏭 Single-pass catalog processing")
    print("   📝 RAG-optimized product descriptors")
    print("   🏷️  Brand-specific filter extraction")
    print("   🔄 Consistent terminology across components")
    print("   ⚡ Performance optimized architecture")
    print("   🛠️  Ready for voice-first AI integration")


if __name__ == "__main__":
    asyncio.run(main())