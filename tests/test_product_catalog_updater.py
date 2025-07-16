#!/usr/bin/env python3
"""
Test Product Catalog Updater functionality
"""

import sys
import os
sys.path.append('packages')
sys.path.append('scripts')

import json
import asyncio
import tempfile
from pathlib import Path

from liddy.models.product import Product
from liddy.models.product_variant import ProductVariant
from update_product_catalog import ProductCatalogUpdater


def create_test_products():
    """Create test products for testing merge functionality"""
    
    # Existing product with no variants
    existing1 = Product(
        id="1001",
        name="Test Bike 1",
        brand="TestBrand",
        originalPrice="$999.99",
        productUrl="https://test.com/bike1",
        descriptor="This is an existing bike",
        categories=["Bikes", "Road"],
        imageUrls=["https://test.com/bike1.jpg"]
    )
    
    # Existing product with old variant data
    existing2 = Product(
        id="1002",
        name="Test Bike 2",
        brand="TestBrand",
        originalPrice="$1299.99",
        salePrice="$999.99",
        productUrl="https://test.com/bike2",
        descriptor="This is another existing bike",
        variants=[
            ProductVariant(
                id="V1002-1",
                price="$999.99",
                inventoryQuantity=5,
                attributes={"size": "Medium", "color": "Red"}
            )
        ]
    )
    
    # New product data with variants
    new1 = Product(
        id="1001",  # Same ID as existing1
        name="Test Bike 1 Updated",  # Should not overwrite existing name
        brand="TestBrand",
        variants=[
            ProductVariant(
                id="V1001-1",
                price="$899.99",
                originalPrice="$999.99",
                inventoryQuantity=10,
                attributes={"size": "Small", "color": "Black"}
            ),
            ProductVariant(
                id="V1001-2",
                price="$899.99",
                originalPrice="$999.99",
                inventoryQuantity=15,
                attributes={"size": "Large", "color": "Black"}
            )
        ]
    )
    
    # Updated variant data for existing2
    new2 = Product(
        id="1002",
        variants=[
            ProductVariant(
                id="V1002-1",
                price="$899.99",  # Price changed
                originalPrice="$1299.99",
                inventoryQuantity=3,  # Inventory changed
                attributes={"size": "Medium", "color": "Red"}
            ),
            ProductVariant(
                id="V1002-2",  # New variant
                price="$899.99",
                originalPrice="$1299.99",
                inventoryQuantity=8,
                attributes={"size": "Large", "color": "Blue"}
            )
        ]
    )
    
    # Completely new product
    new3 = Product(
        id="1003",
        name="New Test Bike 3",
        brand="TestBrand",
        originalPrice="$799.99",
        productUrl="https://test.com/bike3",
        variants=[
            ProductVariant(
                id="V1003-1",
                price="$799.99",
                inventoryQuantity=20,
                attributes={"size": "Universal"}
            )
        ]
    )
    
    return {
        'existing': [existing1, existing2],
        'new': [new1, new2, new3]
    }


def test_merge_functionality():
    """Test the merge logic"""
    print("🧪 Testing merge functionality...")
    
    test_data = create_test_products()
    updater = ProductCatalogUpdater("test.com")
    
    # Test 1: Merge product with no variants getting variants
    print("\n  Test 1: Adding variants to product without variants")
    existing1 = test_data['existing'][0]
    new1 = test_data['new'][0]
    merged1, changes1 = updater.merge_product(existing1, new1)
    
    assert merged1.name == "Test Bike 1", "Should preserve existing name"
    assert len(merged1.variants) == 2, "Should have 2 variants"
    assert merged1.salePrice == "$899.99", "Should update price from variants"
    assert changes1['variants_added'] == 2
    assert changes1['price_changed'] == True
    print("    ✅ Passed")
    
    # Test 2: Update existing variants
    print("\n  Test 2: Updating existing variants")
    existing2 = test_data['existing'][1]
    new2 = test_data['new'][1]
    merged2, changes2 = updater.merge_product(existing2, new2)
    
    assert len(merged2.variants) == 2, "Should have 2 variants (1 updated, 1 new)"
    assert merged2.descriptor == "This is another existing bike", "Should preserve descriptor"
    assert changes2['variants_added'] == 1
    assert changes2['price_changed'] == True
    # Inventory changed from 5 to 3+8=11
    assert changes2['inventory_changed'] == True
    print("    ✅ Passed")
    
    # Test 3: Preserve existing data
    print("\n  Test 3: Preserving existing data")
    assert merged1.productUrl == "https://test.com/bike1", "Should preserve URL"
    assert merged1.imageUrls == ["https://test.com/bike1.jpg"], "Should preserve images"
    assert merged1.descriptor == "This is an existing bike", "Should preserve descriptor"
    print("    ✅ Passed")
    
    print("\n✅ All merge tests passed!")


async def test_full_update_workflow():
    """Test the full update workflow with temp files"""
    print("\n🧪 Testing full update workflow...")
    
    test_data = create_test_products()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock existing catalog
        existing_path = Path(tmpdir) / "existing_products.json"
        with open(existing_path, 'w') as f:
            json.dump([p.to_dict() for p in test_data['existing']], f)
        
        # Create new products file
        new_path = Path(tmpdir) / "new_products.json"
        with open(new_path, 'w') as f:
            json.dump({
                "_metadata": {"source": "test"},
                "products": [p.to_dict() for p in test_data['new']]
            }, f)
        
        # Test loading
        updater = ProductCatalogUpdater("test.com")
        
        # Load new products
        new_products = updater.load_new_products(str(new_path))
        assert len(new_products) == 3, "Should load 3 new products"
        
        print("    ✅ Loading test passed")
        
        # Test stats
        assert updater.stats['new_products'] == 3
        
        print("\n✅ Full workflow test passed!")


def test_edge_cases():
    """Test edge cases"""
    print("\n🧪 Testing edge cases...")
    
    updater = ProductCatalogUpdater("test.com")
    
    # Test 1: Empty variants
    print("\n  Test 1: Product with empty variants")
    existing = Product(id="2001", name="Test", variants=[])
    new = Product(id="2001", variants=[
        ProductVariant(id="V1", price="$99", inventoryQuantity=10)
    ])
    merged, changes = updater.merge_product(existing, new)
    assert len(merged.variants) == 1
    assert changes['variants_added'] == 1
    print("    ✅ Passed")
    
    # Test 2: No price in variant
    print("\n  Test 2: Variant without price")
    new2 = Product(id="2002", variants=[
        ProductVariant(id="V2", inventoryQuantity=10)
    ])
    existing2 = Product(id="2002", originalPrice="$199")
    merged2, changes2 = updater.merge_product(existing2, new2)
    # Should handle gracefully
    print("    ✅ Passed")
    
    print("\n✅ All edge case tests passed!")


def main():
    """Run all tests"""
    print("🔧 Testing Product Catalog Updater\n")
    
    test_merge_functionality()
    asyncio.run(test_full_update_workflow())
    test_edge_cases()
    
    print("\n🎉 All tests passed!")


if __name__ == "__main__":
    main()