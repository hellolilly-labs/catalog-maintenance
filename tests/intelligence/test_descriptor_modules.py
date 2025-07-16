"""Tests for the descriptor module system."""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock
from datetime import datetime

from liddy.models.product import Product, ProductVariant
from liddy_intelligence.catalog.descriptors import BaseDescriptorModule, DescriptorModuleManager
from liddy_intelligence.catalog.descriptors.modules import PriceModule, SaleModule, VariantModule


class TestPriceModule:
    """Test the PriceModule functionality."""
    
    @pytest.fixture
    def price_module(self):
        """Create a PriceModule instance with test config."""
        config = {
            'price_stats': {
                'overall': {
                    'min': 50,
                    'p25': 100,
                    'p50': 200,
                    'p75': 500,
                    'p95': 1000,
                    'max': 2000
                }
            }
        }
        return PriceModule(config)
    
    @pytest.fixture
    def test_product(self):
        """Create a test product."""
        return Product(
            id='test-123',
            name='Test Product',
            originalPrice='$299.99',
            salePrice=None,
            categories=['bikes'],
            variants=[]
        )
    
    def test_is_applicable(self, price_module, test_product):
        """Test that module is applicable to products with prices."""
        assert price_module.is_applicable(test_product) is True
        
        # Test product without price
        no_price_product = Product(id='no-price', name='No Price Product')
        assert price_module.is_applicable(no_price_product) is False
    
    def test_enhance_descriptor(self, price_module, test_product):
        """Test descriptor enhancement with price information."""
        initial = "This is a great product with many features."
        enhanced = price_module.enhance_descriptor(initial, test_product)
        
        assert "**Pricing:**" in enhanced
        assert "$299.99" in enhanced
        assert "Price: $299.99" in enhanced
    
    def test_enhance_descriptor_with_existing_price(self, price_module, test_product):
        """Test that existing price section is replaced."""
        initial = "Great product.\n\n**Pricing:**\nOld price info\n\nMore content."
        enhanced = price_module.enhance_descriptor(initial, test_product)
        
        assert enhanced.count("**Pricing:**") == 1
        assert "Old price info" not in enhanced
        assert "$299.99" in enhanced
    
    def test_enhance_search_keywords(self, price_module, test_product):
        """Test search keyword enhancement."""
        initial_keywords = ["bike", "cycling"]
        enhanced = price_module.enhance_search_keywords(initial_keywords, test_product)
        
        assert "bike" in enhanced
        assert "cycling" in enhanced
        assert "$299" in enhanced
        assert "mid-range" in enhanced
        assert len(enhanced) > len(initial_keywords)
    
    def test_get_metadata(self, price_module, test_product):
        """Test metadata generation."""
        metadata = price_module.get_metadata(test_product)
        
        assert 'price_updated_at' in metadata
        assert 'price_tier' in metadata
        assert metadata['price_tier'] == 'mid_high'  # $299 is in mid_high range
        assert metadata['has_sale'] is False


class TestSaleModule:
    """Test the SaleModule functionality."""
    
    @pytest.fixture
    def sale_module(self):
        """Create a SaleModule instance."""
        return SaleModule({'min_discount_threshold': 10})
    
    @pytest.fixture
    def sale_product(self):
        """Create a product on sale."""
        return Product(
            id='sale-123',
            name='Sale Product',
            originalPrice='$399.99',
            salePrice='$299.99',
            categories=['accessories']
        )
    
    def test_is_applicable(self, sale_module, sale_product):
        """Test that module only applies to sale products."""
        assert sale_module.is_applicable(sale_product) is True
        
        # Regular price product
        regular = Product(id='reg', name='Regular', originalPrice='$100')
        assert sale_module.is_applicable(regular) is False
        
        # Small discount (under threshold)
        small_sale = Product(
            id='small',
            name='Small Sale',
            originalPrice='$100',
            salePrice='$95'
        )
        assert sale_module.is_applicable(small_sale) is False
    
    def test_enhance_descriptor(self, sale_module, sale_product):
        """Test sale emphasis in descriptor."""
        initial = "This is a great product."
        enhanced = sale_module.enhance_descriptor(initial, sale_product)
        
        assert "25% Discount" in enhanced or "25% OFF" in enhanced
        assert "$100" in enhanced  # savings amount
        assert "💰" in enhanced or "⭐" in enhanced  # sale emphasis
    
    def test_enhance_search_keywords(self, sale_module, sale_product):
        """Test sale-specific keywords."""
        keywords = sale_module.enhance_search_keywords([], sale_product)
        
        assert "25% off" in keywords
        assert "save $100" in keywords
        assert "clearance" in keywords
        assert "deal" in keywords
        assert "limited time" in keywords


class TestVariantModule:
    """Test the VariantModule functionality."""
    
    @pytest.fixture
    def variant_module(self):
        """Create a VariantModule instance."""
        return VariantModule({
            'max_values_per_attribute': 5,
            'excluded_attributes': ['sku', 'barcode']
        })
    
    @pytest.fixture
    def variant_product(self):
        """Create a product with variants."""
        return Product(
            id='var-123',
            name='Multi-variant Product',
            originalPrice='$199.99',
            variants=[
                ProductVariant(
                    id='v1',
                    attributes={'color': 'Red', 'size': 'Small', 'sku': 'SKU-001'}
                ),
                ProductVariant(
                    id='v2',
                    attributes={'color': 'Blue', 'size': 'Medium', 'sku': 'SKU-002'}
                ),
                ProductVariant(
                    id='v3',
                    attributes={'color': 'Red', 'size': 'Large', 'sku': 'SKU-003'}
                )
            ]
        )
    
    def test_is_applicable(self, variant_module, variant_product):
        """Test that module applies to products with meaningful variants."""
        assert variant_module.is_applicable(variant_product) is True
        
        # No variants
        no_var = Product(id='no-var', name='No Variants')
        assert variant_module.is_applicable(no_var) is False
        
        # Only excluded attributes
        only_sku = Product(
            id='only-sku',
            name='Only SKU',
            variants=[
                ProductVariant(id='v1', attributes={'sku': 'ABC123'})
            ]
        )
        assert variant_module.is_applicable(only_sku) is False
    
    def test_enhance_descriptor(self, variant_module, variant_product):
        """Test variant information in descriptor."""
        initial = "Great product description."
        enhanced = variant_module.enhance_descriptor(initial, variant_product)
        
        assert "**Available Options:**" in enhanced
        assert "**Color**: Blue, Red" in enhanced
        assert "**Size**: Large, Medium, Small" in enhanced
    
    def test_enhance_search_keywords(self, variant_module, variant_product):
        """Test variant-based keywords."""
        keywords = variant_module.enhance_search_keywords([], variant_product)
        
        assert "color" in keywords
        assert "red" in keywords
        assert "blue" in keywords
        assert "size" in keywords
        assert "small" in keywords
        assert "medium" in keywords
        assert "large" in keywords
        assert "multiple colors" in keywords
        assert "multiple sizes" in keywords


class TestDescriptorModuleManager:
    """Test the DescriptorModuleManager functionality."""
    
    @pytest.fixture
    def manager(self):
        """Create a manager with test configuration."""
        config = {
            'price_stats': {
                'overall': {'p50': 200}
            }
        }
        return DescriptorModuleManager(global_config=config)
    
    @pytest.fixture
    def test_product(self):
        """Create a test product."""
        return Product(
            id='test-123',
            name='Test Product',
            originalPrice='$299.99',
            salePrice='$199.99',
            descriptor='Original descriptor text.',
            search_keywords=['original', 'keyword'],
            key_selling_points=[],
            voice_summary='',
            variants=[
                ProductVariant(
                    id='v1',
                    attributes={'color': 'Red', 'size': 'Large'}
                )
            ]
        )
    
    def test_module_loading(self, manager):
        """Test that default modules are loaded."""
        modules = manager.list_modules()
        module_names = [m['name'] for m in modules]
        
        assert 'price' in module_names
        assert 'sale' in module_names
        assert 'variant' in module_names
        assert len(modules) == 3
    
    def test_module_priority_ordering(self, manager):
        """Test that modules are ordered by priority."""
        modules = manager.list_modules()
        priorities = [m['priority'] for m in modules]
        
        assert priorities == sorted(priorities)
    
    def test_enhance_product(self, manager, test_product):
        """Test full product enhancement."""
        result = manager.enhance_product(
            product=test_product,
            initial_descriptor="Basic product description."
        )
        
        # Check all enhancements were applied
        assert "**Pricing:**" in result['descriptor']
        assert "$199.99" in result['descriptor']  # sale price
        assert "33% OFF" in result['descriptor'] or "Save 33%" in result['descriptor']
        assert "**Available Options:**" in result['descriptor']
        
        # Check keywords were enhanced
        assert len(result['search_keywords']) > 2
        assert "on sale" in result['search_keywords']
        assert "red" in result['search_keywords']
        
        # Check metadata
        assert 'modules_applied' in result
        assert set(result['modules_applied']) == {'price', 'sale', 'variant'}
    
    def test_module_removal(self, manager):
        """Test removing a module."""
        initial_count = len(manager.modules)
        
        assert manager.remove_module('sale') is True
        assert len(manager.modules) == initial_count - 1
        
        modules = manager.list_modules()
        module_names = [m['name'] for m in modules]
        assert 'sale' not in module_names
    
    def test_get_module(self, manager):
        """Test retrieving a specific module."""
        price_module = manager.get_module('price')
        assert price_module is not None
        assert isinstance(price_module, PriceModule)
        
        non_existent = manager.get_module('non-existent')
        assert non_existent is None


# Custom test module for testing extensibility
class CustomTestModule(BaseDescriptorModule):
    """A custom module for testing."""
    
    @property
    def name(self) -> str:
        return "custom_test"
    
    @property
    def priority(self) -> int:
        return 50
    
    def is_applicable(self, product: Product) -> bool:
        return True
    
    def enhance_descriptor(self, descriptor: str, product: Product, **kwargs) -> str:
        return descriptor + "\n\n**Custom Section:** Test content"
    
    def enhance_search_keywords(self, keywords: list, product: Product, **kwargs) -> list:
        keywords.append("custom-keyword")
        return keywords
    
    def get_metadata(self, product: Product, **kwargs) -> dict:
        return {"custom_data": "test"}


class TestModuleExtensibility:
    """Test adding custom modules."""
    
    def test_add_custom_module(self):
        """Test adding a custom module to the manager."""
        manager = DescriptorModuleManager()
        initial_count = len(manager.modules)
        
        manager.add_module(CustomTestModule)
        
        assert len(manager.modules) == initial_count + 1
        assert manager.get_module('custom_test') is not None
        
        # Test that custom module is used
        product = Product(id='test', name='Test', originalPrice='$100')
        result = manager.enhance_product(product, "Initial text")
        
        assert "**Custom Section:**" in result['descriptor']
        assert "custom-keyword" in result['search_keywords']
        assert 'custom_test' in result['modules_applied']


def test_module_error_handling():
    """Test that module errors don't break the whole enhancement."""
    
    class ErrorModule(BaseDescriptorModule):
        name = "error"
        priority = 1
        
        def is_applicable(self, product):
            return True
        
        def enhance_descriptor(self, descriptor, product, **kwargs):
            raise ValueError("Test error")
        
        def enhance_search_keywords(self, keywords, product, **kwargs):
            return keywords
        
        def get_metadata(self, product, **kwargs):
            return {}
    
    manager = DescriptorModuleManager(modules=[ErrorModule, PriceModule])
    product = Product(id='test', name='Test', originalPrice='$100')
    
    # Should not raise, but should skip the error module
    result = manager.enhance_product(product, "Initial")
    
    assert 'price' in result['modules_applied']
    assert 'error' not in result['modules_applied']
    assert "**Pricing:**" in result['descriptor']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])