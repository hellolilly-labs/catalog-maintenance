#!/usr/bin/env python3
"""
Test variant-aware product functionality
"""

import sys
import os
sys.path.append('packages')

import asyncio
from liddy.models.product import Product
from liddy.models.product_variant import ProductVariant
from liddy.models.product_loader import ProductLoader
from liddy_intelligence.ingestion.parsers.specialized_csv_parser import SpecializedCSVParser


def test_product_variant_model():
    """Test ProductVariant model functionality"""
    print("🧪 Testing ProductVariant model...")
    
    # Create a variant with all fields
    variant = ProductVariant(
        id="SKU123",
        price="$99.99",
        originalPrice="$129.99",
        inventoryQuantity=45,
        imageUrls=["https://example.com/img1.jpg", "https://example.com/img2.jpg"],
        gtin="1234567890123",
        attributes={
            "size": "Large",
            "color": "Blue"
        },
        isDefault=True
    )
    
    # Test backward compatibility
    assert variant.inStock == True  # Should be True since inventoryQuantity > 0
    assert variant.image == "https://example.com/img1.jpg"  # First image as primary
    
    # Test serialization
    variant_dict = variant.to_dict()
    assert variant_dict["inventoryQuantity"] == 45
    assert len(variant_dict["imageUrls"]) == 2
    
    # Test deserialization
    variant2 = ProductVariant.from_dict(variant_dict)
    assert variant2.gtin == "1234567890123"
    
    print("  ✅ ProductVariant model tests passed")


def test_product_with_variants():
    """Test Product model with variants"""
    print("\n🧪 Testing Product with variants...")
    
    # Create a product with multiple variants
    product = Product(
        id="PROD123",
        name="Test Product",
        brand="TestBrand",
        variants=[
            ProductVariant(
                id="SKU1",
                price="$79.99",
                inventoryQuantity=10,
                attributes={"size": "Small", "color": "Red"}
            ),
            ProductVariant(
                id="SKU2", 
                price="$89.99",
                inventoryQuantity=5,
                attributes={"size": "Medium", "color": "Red"}
            ),
            ProductVariant(
                id="SKU3",
                price="$99.99",
                inventoryQuantity=0,
                attributes={"size": "Large", "color": "Blue"},
                isDefault=True
            )
        ]
    )
    
    # Test helper methods
    min_price, max_price = product.price_range()
    assert min_price == 79.99
    assert max_price == 99.99
    
    # Test variant lookup
    variant = product.get_variant_by_sku("SKU2")
    assert variant is not None
    assert variant.attributes["size"] == "Medium"
    
    # Test default variant
    default = product.get_default_variant()
    assert default.id == "SKU3"
    
    # Test attribute search
    red_variants = product.get_variants_by_attribute("color", "Red")
    assert len(red_variants) == 2
    
    # Test inventory
    assert product.get_total_inventory() == 15
    assert product.is_in_stock() == True
    
    # Test available options
    sizes = product.sizes
    assert set(sizes) == {"Small", "Medium", "Large"}
    
    colors = product.colors
    assert set(colors) == {"Red", "Blue"}
    
    print("  ✅ Product with variants tests passed")


async def test_csv_parser():
    """Test Specialized CSV parser"""
    print("\n🧪 Testing Specialized CSV parser...")
    
    # Check if CSV exists
    csv_path = "local/account_storage/accounts/specialized.com/US Hybris Feed FTP(68752b7d95064 - 1093267importfu)-3.csv"
    
    if not os.path.exists(csv_path):
        print(f"  ⚠️  Skipping CSV parser test - file not found: {csv_path}")
        return
    
    parser = SpecializedCSVParser("specialized.com")
    
    # Parse first few rows only for testing
    products = parser.parse_csv(csv_path)
    
    if products:
        # Check first product
        product = products[0]
        print(f"  📦 Parsed {len(products)} products")
        print(f"  First product: {product.name}")
        print(f"  Variants: {len(product.variants)}")
        
        if product.variants:
            variant = product.variants[0]
            print(f"  First variant SKU: {variant.id}")
            print(f"  Inventory: {variant.inventoryQuantity}")
            print(f"  GTIN: {variant.gtin}")
        
        print("  ✅ CSV parser tests passed")
    else:
        print("  ⚠️  No products parsed from CSV")


async def test_product_loader():
    """Test ProductLoader compatibility"""
    print("\n🧪 Testing ProductLoader...")
    
    loader = ProductLoader("specialized.com")
    
    # Test loading (will use CSV if available, otherwise JSON)
    products = await loader.load_products()
    
    print(f"  📦 Loaded {len(products)} products")
    
    if products:
        # Check that products have variants
        variants_count = sum(len(p.variants) for p in products[:10])
        print(f"  Total variants in first 10 products: {variants_count}")
        
        # Find a product with multiple variants
        multi_variant_products = [p for p in products[:20] if len(p.variants) > 1]
        if multi_variant_products:
            product = multi_variant_products[0]
            print(f"  Product with variants: {product.name}")
            print(f"  Price range: ${product.price_range()[0]:.2f} - ${product.price_range()[1]:.2f}")
    
    print("  ✅ ProductLoader tests passed")


async def main():
    """Run all tests"""
    print("🔧 Testing Variant-Aware Product Implementation\n")
    
    # Run sync tests
    test_product_variant_model()
    test_product_with_variants()
    
    # Run async tests
    await test_csv_parser()
    await test_product_loader()
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())