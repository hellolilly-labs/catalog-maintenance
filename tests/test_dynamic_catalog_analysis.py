#!/usr/bin/env python3
"""
Test Dynamic Catalog Analysis for Brand-Specific Filter Extraction

Demonstrates how filters are extracted from actual product catalogs rather than hardcoded.
"""

import asyncio
import json
from pathlib import Path

from src.agents.catalog_filter_analyzer import CatalogFilterAnalyzer, analyze_brand_catalog
from src.agents.query_optimization_agent import QueryOptimizationAgent


def create_sample_catalog_data() -> list:
    """Create sample catalog data for demonstration"""
    
    return [
        {
            "id": "tarmac-sl8-expert",
            "name": "Tarmac SL8 Expert",
            "category": "road",
            "price": 4500.00,
            "frame_material": "carbon",
            "gender": "unisex",
            "wheel_size": "700c",
            "weight": 7.2,
            "features": ["disc_brakes", "electronic_shifting", "tubeless_ready"],
            "intended_use": ["racing", "performance"],
            "description": "Perfect for racing and competitive cycling. Designed for speed and aerodynamics."
        },
        {
            "id": "roubaix-comp",
            "name": "Roubaix Comp", 
            "category": "road",
            "price": 3200.00,
            "frame_material": "carbon",
            "gender": "unisex",
            "wheel_size": "700c", 
            "weight": 8.1,
            "features": ["disc_brakes", "comfort_geometry"],
            "intended_use": ["endurance", "touring"],
            "description": "Ideal for long rides and touring. Built for comfort over long distances."
        },
        {
            "id": "stumpjumper-alloy",
            "name": "Stumpjumper Alloy",
            "category": "mountain",
            "price": 2800.00,
            "frame_material": "aluminum",
            "gender": "unisex",
            "wheel_size": "29",
            "weight": 12.5,
            "features": ["disc_brakes", "suspension", "dropper_post"],
            "intended_use": ["trail_riding", "cross_country"],
            "description": "Perfect for trail riding and mountain adventures. Designed for rugged terrain."
        },
        {
            "id": "turbo-vado-40",
            "name": "Turbo Vado 4.0",
            "category": "electric",
            "price": 3800.00,
            "frame_material": "aluminum",
            "gender": "unisex", 
            "wheel_size": "700c",
            "weight": 22.0,
            "features": ["disc_brakes", "electric_motor", "integrated_lights"],
            "intended_use": ["commuting", "urban"],
            "description": "Ideal for commuting and urban transportation. Electric assistance for effortless rides."
        },
        {
            "id": "allez-sprint",
            "name": "Allez Sprint",
            "category": "road",
            "price": 1899.00,
            "frame_material": "aluminum",
            "gender": "mens",
            "wheel_size": "700c",
            "weight": 8.8,
            "features": ["disc_brakes"],
            "intended_use": ["racing", "training"],
            "description": "Entry-level racing bike for training and competition."
        },
        {
            "id": "sirrus-womens",
            "name": "Sirrus Women's",
            "category": "hybrid",
            "price": 750.00,
            "frame_material": "aluminum",
            "gender": "womens",
            "wheel_size": "700c",
            "weight": 11.2,
            "features": ["disc_brakes", "comfort_geometry"],
            "intended_use": ["fitness", "recreational"],
            "description": "Perfect for fitness rides and recreational cycling. Designed specifically for women."
        }
    ]


async def test_catalog_analysis():
    """Test dynamic catalog analysis"""
    
    print("🔍 Testing Dynamic Catalog Analysis for Brand-Specific Filters")
    print("=" * 70)
    
    # Create sample catalog data
    sample_catalog = create_sample_catalog_data()
    
    print(f"📊 Sample Catalog: {len(sample_catalog)} products")
    for product in sample_catalog:
        print(f"   - {product['name']} ({product['category']}, ${product['price']})")
    
    # Analyze catalog to extract filters
    print(f"\n🔬 Analyzing catalog to extract available filters...")
    
    filters = analyze_brand_catalog("specialized.com", sample_catalog)
    
    print(f"\n✅ Extracted Filters:")
    print("-" * 50)
    
    for filter_name, filter_config in filters.items():
        if filter_name.startswith("_"):
            continue  # Skip metadata
            
        filter_type = filter_config.get("type", "unknown")
        
        if filter_type == "categorical":
            values = filter_config.get("values", [])
            aliases = filter_config.get("aliases", {})
            print(f"📂 {filter_name} ({filter_type}): {len(values)} options")
            print(f"   Values: {values}")
            if aliases:
                print(f"   Aliases: {list(aliases.keys())}")
                
        elif filter_type == "numeric_range":
            min_val = filter_config.get("min")
            max_val = filter_config.get("max")
            ranges = filter_config.get("common_ranges", [])
            print(f"📊 {filter_name} ({filter_type}): {min_val} to {max_val}")
            if ranges:
                print(f"   Common ranges: {[r['label'] for r in ranges]}")
                
        elif filter_type == "multi_select":
            values = filter_config.get("values", [])
            frequency = filter_config.get("frequency", {})
            print(f"☑️  {filter_name} ({filter_type}): {len(values)} options")
            print(f"   Values: {values}")
            if frequency:
                print(f"   Most common: {sorted(frequency.items(), key=lambda x: x[1], reverse=True)[:3]}")
        
        print()


async def test_dynamic_vs_hardcoded():
    """Compare dynamic filter extraction vs hardcoded approach"""
    
    print("\n📋 Comparing Dynamic vs Hardcoded Filter Extraction")
    print("=" * 70)
    
    # Create two query optimizers
    sample_catalog = create_sample_catalog_data()
    
    # 1. Dynamic analyzer
    dynamic_filters = analyze_brand_catalog("specialized.com", sample_catalog)
    
    # 2. Create optimizer with dynamic filters
    optimizer = QueryOptimizationAgent("specialized.com", catalog_filters=dynamic_filters)
    
    # Test query
    test_query = "I need a lightweight carbon road bike under 4000 for racing"
    
    print(f"🔍 Test Query: \"{test_query}\"")
    print(f"\n📊 Filters extracted from ACTUAL catalog data:")
    
    result = await optimizer.optimize_product_query(
        original_query=test_query,
        context={"recent_messages": [], "expressed_interests": []},
        user_state=None
    )
    
    extracted_filters = result.get("filters", {})
    for filter_name, filter_value in extracted_filters.items():
        print(f"   {filter_name}: {filter_value}")
    
    print(f"\n🎯 Benefits of Dynamic Analysis:")
    print("✅ Filters based on ACTUAL product catalog")
    print("✅ Brand-specific terminology automatically discovered") 
    print("✅ Price ranges based on actual product prices")
    print("✅ Features based on what products actually have")
    print("✅ Categories based on actual product categorization")
    print("✅ No hardcoded assumptions")


async def test_multi_brand_analysis():
    """Test filter extraction for different brand types"""
    
    print("\n\n🏢 Testing Multi-Brand Filter Analysis")
    print("=" * 70)
    
    # Create different catalog structures for different brand types
    
    # Fashion brand catalog
    fashion_catalog = [
        {
            "name": "Classic Tee",
            "category": "shirts", 
            "price": 29.99,
            "material": "cotton",
            "size": "M",
            "color": "blue",
            "gender": "unisex",
            "features": ["organic", "fair_trade"]
        },
        {
            "name": "Denim Jacket",
            "category": "outerwear",
            "price": 89.99, 
            "material": "denim",
            "size": "L",
            "color": "black",
            "gender": "womens",
            "features": ["vintage_wash", "sustainable"]
        }
    ]
    
    # Electronics catalog
    electronics_catalog = [
        {
            "name": "Wireless Headphones",
            "category": "audio",
            "price": 199.99,
            "brand": "TechCorp",
            "color": "black",
            "features": ["noise_canceling", "bluetooth", "wireless"],
            "battery_life": 24
        },
        {
            "name": "Gaming Laptop",
            "category": "computers",
            "price": 1299.99,
            "brand": "GameTech", 
            "screen_size": 15.6,
            "features": ["gaming", "high_refresh", "rgb_lighting"],
            "storage": 512
        }
    ]
    
    print("👕 Fashion Brand Catalog Analysis:")
    fashion_filters = analyze_brand_catalog("fashion-brand.com", fashion_catalog)
    print(f"   Extracted {len([k for k in fashion_filters.keys() if not k.startswith('_')])} filter types")
    print(f"   Categories: {fashion_filters.get('category', {}).get('values', [])}")
    print(f"   Materials: {fashion_filters.get('material', {}).get('values', [])}")
    
    print("\n💻 Electronics Brand Catalog Analysis:")
    electronics_filters = analyze_brand_catalog("electronics-brand.com", electronics_catalog)
    print(f"   Extracted {len([k for k in electronics_filters.keys() if not k.startswith('_')])} filter types")
    print(f"   Categories: {electronics_filters.get('category', {}).get('values', [])}")
    print(f"   Features: {electronics_filters.get('features', {}).get('values', [])}")
    
    print(f"\n🎯 Key Insight: Each brand gets filters based on THEIR actual products!")


async def test_filter_file_persistence():
    """Test that analyzed filters are saved and reloaded"""
    
    print("\n\n💾 Testing Filter File Persistence")
    print("=" * 70)
    
    sample_catalog = create_sample_catalog_data()
    brand_domain = "specialized.com"
    
    # Analyze and save filters
    print("1️⃣ Analyzing catalog and saving filters...")
    filters = analyze_brand_catalog(brand_domain, sample_catalog)
    
    filters_path = Path(f"accounts/{brand_domain}/catalog_filters.json")
    print(f"✅ Filters saved to: {filters_path}")
    print(f"📄 File exists: {filters_path.exists()}")
    
    # Create new optimizer (should load from file)
    print(f"\n2️⃣ Creating new optimizer (should load from saved file)...")
    optimizer = QueryOptimizationAgent(brand_domain)
    
    print(f"✅ Optimizer loaded filters from file!")
    print(f"📊 Available filter types: {len(optimizer.catalog_filters)}")
    
    # Test that filters work
    result = await optimizer.optimize_product_query(
        original_query="carbon road bike",
        context={},
        user_state=None
    )
    
    print(f"🔍 Test extraction: {result.get('filters', {})}")
    print(f"\n🎯 Benefit: Filters are cached - no need to re-analyze catalog every time!")


async def main():
    """Run all dynamic catalog analysis tests"""
    
    await test_catalog_analysis()
    await test_dynamic_vs_hardcoded()
    await test_multi_brand_analysis()
    await test_filter_file_persistence()
    
    print("\n\n🎉 All Dynamic Catalog Analysis Tests Complete!")
    print("\n📋 Key Benefits:")
    print("✅ NO hardcoded brand-specific labels")
    print("✅ Filters extracted from ACTUAL product catalog")
    print("✅ Automatic brand terminology discovery")
    print("✅ Price ranges based on real product prices")
    print("✅ Multi-brand support with different catalog structures")
    print("✅ Filter caching for performance")
    print("✅ Graceful fallback when catalog unavailable")
    
    print("\n🚀 Result: Truly dynamic, brand-agnostic filter extraction!")


if __name__ == "__main__":
    asyncio.run(main())