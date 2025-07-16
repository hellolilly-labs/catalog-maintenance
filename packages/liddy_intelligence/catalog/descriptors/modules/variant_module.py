"""Variant options module for descriptors."""

import logging
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict

from liddy.models.product import Product
from ..base import BaseDescriptorModule

logger = logging.getLogger(__name__)


class VariantModule(BaseDescriptorModule):
    """Module for injecting product variant information (colors, sizes, materials, etc.) into descriptors."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize variant module."""
        super().__init__(config)
        self.max_values_per_attribute = config.get('max_values_per_attribute', 12) if config else 12
        self.excluded_attributes = set(config.get('excluded_attributes', ['sku', 'barcode', 'upc'])) if config else {'sku', 'barcode', 'upc'}
    
    @property
    def name(self) -> str:
        return "variant"
    
    @property
    def priority(self) -> int:
        """Run after price and sale modules."""
        return 30
    
    def is_applicable(self, product: Product) -> bool:
        """Applicable to products with variants that have meaningful attributes."""
        if not product.variants:
            return False
        
        # Check if any variant has non-excluded attributes
        for variant in product.variants:
            if variant.attributes:
                for key in variant.attributes:
                    if key.lower() not in self.excluded_attributes:
                        return True
        
        return False
    
    def enhance_descriptor(self, descriptor: str, product: Product, **kwargs) -> str:
        """Add variant options information to descriptor."""
        variant_info = self._extract_variant_options(product)
        
        if not variant_info:
            return descriptor
        
        # Format variant information
        variant_section = self._format_variant_section(variant_info)
        
        # Insert variant section before search terms or at the end
        if '**Search Terms:**' in descriptor:
            parts = descriptor.split('**Search Terms:**')
            updated = parts[0].rstrip() + '\n\n' + variant_section + '\n\n**Search Terms:**' + parts[1]
        else:
            updated = descriptor.rstrip() + '\n\n' + variant_section
        
        return updated
    
    def enhance_search_keywords(self, keywords: List[str], product: Product, **kwargs) -> List[str]:
        """Add variant-specific search keywords."""
        updated_keywords = keywords.copy() if keywords else []
        
        variant_info = self._extract_variant_options(product)
        
        # Add all variant values as keywords
        for attribute, values in variant_info.items():
            # Add attribute name
            attr_lower = attribute.lower()
            if attr_lower not in updated_keywords:
                updated_keywords.append(attr_lower)
            
            # Add each value
            for value in values:
                value_lower = str(value).lower()
                if value_lower and value_lower not in updated_keywords:
                    updated_keywords.append(value_lower)
                
                # Add combined attribute+value for specific searches
                combined = f"{attr_lower} {value_lower}"
                if combined not in updated_keywords and len(combined) < 30:
                    updated_keywords.append(combined)
        
        # Add common variant-related keywords
        if 'color' in variant_info or 'colour' in variant_info:
            if 'multiple colors' not in updated_keywords:
                updated_keywords.append('multiple colors')
        
        if 'size' in variant_info:
            if 'multiple sizes' not in updated_keywords:
                updated_keywords.append('multiple sizes')
        
        return updated_keywords
    
    def get_metadata(self, product: Product, **kwargs) -> Dict[str, Any]:
        """Return variant-related metadata."""
        variant_info = self._extract_variant_options(product)
        
        metadata = {
            'variant_count': len(product.variants) if product.variants else 0,
            'variant_attributes': list(variant_info.keys()),
            'variant_options': {}
        }
        
        # Add count of options per attribute
        for attribute, values in variant_info.items():
            metadata['variant_options'][attribute] = len(values)
        
        return metadata
    
    def _extract_variant_options(self, product: Product) -> Dict[str, Set[str]]:
        """Extract all unique variant options from a product."""
        variant_options = defaultdict(set)
        
        if not product.variants:
            return {}
        
        for variant in product.variants:
            if not variant.attributes:
                continue
            
            for key, value in variant.attributes.items():
                # Skip excluded attributes
                if key.lower() in self.excluded_attributes:
                    continue
                
                # Normalize the attribute key
                normalized_key = self._normalize_attribute_key(key)
                
                # Add the value
                if value is not None:
                    variant_options[normalized_key].add(str(value))
        
        # Convert defaultdict to regular dict and limit values
        result = {}
        for key, values in variant_options.items():
            # Sort values for consistency
            sorted_values = sorted(values)
            # Limit to max values
            if len(sorted_values) > self.max_values_per_attribute:
                result[key] = sorted_values[:self.max_values_per_attribute]
            else:
                result[key] = sorted_values
        
        return result
    
    def _normalize_attribute_key(self, key: str) -> str:
        """Normalize attribute keys for consistency."""
        key = key.lower().strip()
        
        # Common normalizations
        normalizations = {
            'colour': 'color',
            'colours': 'color',
            'colors': 'color',
            'sizes': 'size',
            'materials': 'material',
            'styles': 'style',
            'types': 'type',
            'models': 'model',
            'lengths': 'length',
            'widths': 'width',
            'heights': 'height'
        }
        
        return normalizations.get(key, key)
    
    def _format_variant_section(self, variant_info: Dict[str, List[str]]) -> str:
        """Format variant information into a readable section."""
        lines = ["**Available Options:**"]
        
        # Sort attributes for consistent ordering
        sorted_attributes = sorted(variant_info.items(), key=lambda x: self._get_attribute_priority(x[0]))
        
        for attribute, values in sorted_attributes:
            # Format attribute name
            display_name = attribute.title()
            
            # Format values
            if len(values) <= 6:
                # Short list - show all
                value_list = ", ".join(values)
            else:
                # Long list - show first few with count
                shown_values = ", ".join(values[:5])
                remaining = len(values) - 5
                value_list = f"{shown_values}, and {remaining} more"
            
            lines.append(f"- **{display_name}**: {value_list}")
        
        # Add a note about availability
        lines.append("\n*Note: Not all combinations may be available. Please check specific variant for availability.*")
        
        return '\n'.join(lines)
    
    def _get_attribute_priority(self, attribute: str) -> int:
        """Get display priority for attributes (lower = higher priority)."""
        priority_map = {
            'color': 1,
            'size': 2,
            'material': 3,
            'style': 4,
            'type': 5,
            'model': 6,
            'finish': 7,
            'pattern': 8
        }
        
        return priority_map.get(attribute.lower(), 99)