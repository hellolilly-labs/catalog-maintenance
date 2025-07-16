# Descriptor Module System

The descriptor module system provides a flexible, extensible framework for enhancing product descriptors with various types of information. It replaces the monolithic descriptor generation approach with a modular architecture.

## Overview

The system consists of:
- **BaseDescriptorModule**: Abstract base class that all modules inherit from
- **DescriptorModuleManager**: Orchestrates multiple modules and applies them to products
- **Built-in Modules**: Price, Sale, and Variant modules
- **Custom Modules**: Easy to add new enhancement modules

## Architecture

```
descriptors/
├── __init__.py
├── base.py              # BaseDescriptorModule abstract class
├── manager.py           # DescriptorModuleManager
└── modules/
    ├── __init__.py
    ├── price_module.py  # Price information and semantic context
    ├── sale_module.py   # Sale/discount emphasis
    └── variant_module.py # Product variant options
```

## Usage

### Basic Usage

```python
from liddy_intelligence.catalog.descriptors import DescriptorModuleManager
from liddy.models.product import Product

# Initialize manager with optional configuration
manager = DescriptorModuleManager(global_config={
    'price_stats': price_statistics,
    'terminology_research': terminology_data
})

# Enhance a product
result = manager.enhance_product(
    product=product,
    initial_descriptor="Initial product description...",
    price_stats=price_statistics
)

# Access enhanced results
enhanced_descriptor = result['descriptor']
enhanced_keywords = result['search_keywords']
module_metadata = result['metadata']
modules_applied = result['modules_applied']
```

### Creating Custom Modules

```python
from liddy_intelligence.catalog.descriptors import BaseDescriptorModule

class CustomModule(BaseDescriptorModule):
    @property
    def name(self) -> str:
        return "custom"
    
    @property
    def priority(self) -> int:
        return 40  # Lower numbers run first
    
    def is_applicable(self, product: Product) -> bool:
        # Determine if module should process this product
        return True
    
    def enhance_descriptor(self, descriptor: str, product: Product, **kwargs) -> str:
        # Modify the descriptor text
        return descriptor + "\n\nCustom content here"
    
    def enhance_search_keywords(self, keywords: List[str], product: Product, **kwargs) -> List[str]:
        # Add module-specific keywords
        keywords.append("custom-keyword")
        return keywords
    
    def get_metadata(self, product: Product, **kwargs) -> Dict[str, Any]:
        # Return module-specific metadata
        return {"custom_field": "value"}

# Add to manager
manager.add_module(CustomModule)
```

## Built-in Modules

### PriceModule
- Adds pricing information with semantic context
- Categorizes products into price tiers (budget, mid-range, premium)
- Uses brand-specific terminology when available
- Priority: 10 (runs first)

### SaleModule
- Emphasizes discounts and savings
- Adds sale-specific search keywords
- Only applies to products with significant discounts (default: 10%+)
- Priority: 20

### VariantModule
- Extracts and formats product variant options (colors, sizes, etc.)
- Adds variant values as search keywords
- Excludes technical attributes like SKU, barcode
- Priority: 30

## Module Execution

1. Modules are sorted by priority (ascending)
2. For each module:
   - Check if applicable to product
   - Enhance descriptor
   - Enhance search keywords
   - Collect metadata
3. Results are combined and returned

## Configuration

Global configuration can be passed to all modules:

```python
config = {
    'price_stats': {...},  # Price statistics for categorization
    'terminology_research': {...},  # Brand-specific terminology
    'salemodule_config': {  # Module-specific config
        'min_discount_threshold': 15
    }
}
```

## Error Handling

- Module errors are caught and logged
- Failed modules are skipped without affecting others
- The system continues processing with remaining modules

## Migration from PriceDescriptorUpdater

The PriceModule replaces most functionality from PriceDescriptorUpdater:
- Price information injection → PriceModule.enhance_descriptor()
- Price category keywords → PriceModule.enhance_search_keywords()
- Semantic price context → Built into PriceModule

For backward compatibility, PriceDescriptorUpdater is still available but should be migrated to use the module system.