#!/usr/bin/env python3
"""
Test Universal Product Processor with Different Brand Types

This script tests the universal product processor with various product types
to ensure it works correctly across different industries.
"""

import json
from src.ingestion import UniversalProductProcessor


def test_cycling_product():
    """Test with a cycling product (Specialized)."""
    
    print("\n" + "="*60)
    print("TEST: Cycling Product (Specialized)")
    print("="*60)
    
    product = {
        "id": "BIKE-001",
        "name": "Tarmac SL7 Expert",
        "brand": "Specialized",
        "categories": ["Bikes", "Road", "Performance"],
        "originalPrice": 6500,
        "salePrice": 5850,
        "description": "The Tarmac SL7 Expert delivers incredible performance with a perfect balance of aerodynamics and lightweight design.",
        "specifications": {
            "Frame": {
                "frame_material": "Carbon Fiber",
                "geometry": "Race"
            },
            "Drivetrain": {
                "groupset": "Shimano Ultegra Di2",
                "speeds": "12-speed"
            }
        },
        "colors": ["Satin Carbon", "Gloss Tarmac Black"],
        "sizes": ["52", "54", "56", "58"],
        "imageUrls": ["https://example.com/tarmac.jpg"]
    }
    
    processor = UniversalProductProcessor("specialized.com")
    result = processor.process_product(product)
    
    print_processed_result(result)


def test_fashion_product():
    """Test with a fashion product (Balenciaga)."""
    
    print("\n" + "="*60)
    print("TEST: Fashion Product (Balenciaga)")
    print("="*60)
    
    product = {
        "productId": "BAG-789",
        "title": "Le Cagole Shoulder Bag",
        "brand_name": "Balenciaga",
        "product_type": "Handbags",
        "price": 2950,
        "product_description": "Iconic shoulder bag with punk-inspired studs and multiple zippers. Made from metallic silver leather.",
        "available_colors": ["Metallic Silver", "Black", "White"],
        "dimensions": "Medium",
        "materials": ["Leather", "Metal Hardware"],
        "style_tags": ["Edgy", "Contemporary", "Bold"],
        "occasions": ["Evening", "Casual", "Party"],
        "photos": ["https://example.com/cagole1.jpg", "https://example.com/cagole2.jpg"]
    }
    
    processor = UniversalProductProcessor("balenciaga.com")
    result = processor.process_product(product)
    
    print_processed_result(result)


def test_beauty_product():
    """Test with a beauty product (Sunday Riley)."""
    
    print("\n" + "="*60)
    print("TEST: Beauty Product (Sunday Riley)")
    print("="*60)
    
    product = {
        "sku": "SR-GOOD-GENES",
        "product_name": "Good Genes All-In-One Lactic Acid Treatment",
        "vendor": "Sunday Riley",
        "category": "Skincare/Treatments/Serums",
        "current_price": 85.00,
        "long_description": "An all-in-one AHA treatment that clarifies, brightens, and smooths the appearance of skin.",
        "key_ingredients": ["Lactic Acid", "Licorice", "Lemongrass"],
        "benefits": [
            "Instantly brightens and plumps the look of skin",
            "Reduces the appearance of fine lines and wrinkles",
            "Clarifies and minimizes pores"
        ],
        "skin_types": ["All Skin Types"],
        "usage": "Apply 1-2 pumps to clean, dry skin. Can be used daily.",
        "size": "1 oz / 30 ml",
        "is_available": True
    }
    
    processor = UniversalProductProcessor("sundayriley.com")
    result = processor.process_product(product)
    
    print_processed_result(result)


def test_jewelry_product():
    """Test with a jewelry product (Dara Kaye)."""
    
    print("\n" + "="*60)
    print("TEST: Jewelry Product (Dara Kaye)")
    print("="*60)
    
    product = {
        "item_id": "NK-PEARL-001",
        "item_name": "Tahitian Pearl Pendant Necklace",
        "manufacturer": "Dara Kaye Jewelry",
        "classification": ["Necklaces", "Fine Jewelry"],
        "price": {
            "amount": 1850,
            "currency": "USD"
        },
        "description": "Elegant Tahitian pearl pendant on 18k gold chain. Features a lustrous 12mm dark pearl.",
        "materials": {
            "primary": "18k Gold",
            "gemstone": "Tahitian Pearl",
            "chain_length": "18 inches"
        },
        "attributes": {
            "pearl_size": "12mm",
            "pearl_quality": "AAA",
            "clasp_type": "Lobster",
            "adjustable": True
        },
        "perfect_for": ["Special Occasions", "Anniversary Gifts", "Evening Wear"],
        "in_stock": True,
        "images": ["pearl_necklace_1.jpg"]
    }
    
    processor = UniversalProductProcessor("darakayejewelry.com")
    result = processor.process_product(product)
    
    print_processed_result(result)


def print_processed_result(result):
    """Pretty print the processed result."""
    
    print(f"\n📦 Processed Product ID: {result['id']}")
    
    print("\n🔍 Universal Fields:")
    for key, value in result['universal_fields'].items():
        print(f"  {key}: {value}")
    
    print(f"\n📝 Enhanced Descriptor ({len(result['enhanced_descriptor'].split())} words):")
    print(f"  {result['enhanced_descriptor']}")
    
    print(f"\n🎙️ Voice Summary:")
    print(f"  {result['voice_summary']}")
    
    print(f"\n🔑 Key Selling Points:")
    for i, point in enumerate(result['key_selling_points'], 1):
        print(f"  {i}. {point}")
    
    print(f"\n🏷️ Search Keywords ({len(result['search_keywords'])} total):")
    print(f"  {', '.join(result['search_keywords'][:15])}...")
    
    print(f"\n🎯 Filter Metadata:")
    print(json.dumps(result['filter_metadata'], indent=2))


def test_edge_cases():
    """Test edge cases and minimal products."""
    
    print("\n" + "="*60)
    print("TEST: Edge Cases")
    print("="*60)
    
    # Minimal product
    minimal_product = {
        "name": "Mystery Product",
        "price": "29.99"
    }
    
    print("\n--- Minimal Product ---")
    processor = UniversalProductProcessor("generic.com")
    result = processor.process_product(minimal_product)
    print(f"Enhanced Descriptor: {result['enhanced_descriptor']}")
    print(f"Voice Summary: {result['voice_summary']}")
    
    # Product with nested data
    complex_product = {
        "product": {
            "info": {
                "name": "Complex Product",
                "brand": "Test Brand"
            }
        },
        "pricing": {
            "retail": {
                "amount": 199.99
            }
        },
        "categories": "Electronics, Gadgets, Smart Home"
    }
    
    print("\n--- Complex Nested Product ---")
    result = processor.process_product(complex_product)
    print(f"Universal Fields: {result['universal_fields']}")


def main():
    """Run all tests."""
    
    print("🧪 Testing Universal Product Processor")
    print("Testing across different industries and product types")
    
    test_cycling_product()
    test_fashion_product()
    test_beauty_product()
    test_jewelry_product()
    test_edge_cases()
    
    print("\n✅ All tests complete!")


if __name__ == "__main__":
    main()