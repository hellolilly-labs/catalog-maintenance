"""
Catalog Filter Analyzer

Analyzes product catalogs to dynamically extract available filters and labels
for query optimization. This ensures brand-specific terminology and categories
are discovered automatically rather than hardcoded.
"""

from liddy.models.product import Product
from liddy.storage import get_account_storage_provider, AccountStorageProvider
import json
import logging
import re
import asyncio
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, List, Set, Optional, Tuple

logger = logging.getLogger(__name__)


class CatalogFilterAnalyzer:
    """
    Analyzes product catalogs to extract available filters dynamically.
    
    This discovers:
    - Categories and subcategories
    - Price ranges and common price points
    - Features and attributes
    - Brand-specific terminology
    - Product lines and series
    - Size and specification options
    """
    
    def __init__(self, brand_domain: str, storage_provider: Optional[AccountStorageProvider] = None):
        self.brand_domain = brand_domain
        self.storage = storage_provider or get_account_storage_provider()
        
        # Thresholds for filter inclusion
        self.min_category_count = 2  # At least 2 products to be a category
        self.min_feature_frequency = 0.1  # Feature must appear in 10% of products
        
        logger.info(f"🔍 Initialized Catalog Filter Analyzer for {brand_domain}")
    
    async def analyze_product_catalog(self, catalog_data: Optional[List[Product]] = None) -> Dict[str, Any]:
        """
        Analyze product catalog to extract available filters.
        
        Args:
            catalog_data: Optional product data. If None, will try to load from storage.
            
        Returns:
            Dictionary of available filters with their types and values
        """
        
        if catalog_data is None:
            catalog_data = await self._load_catalog_data()
        
        if not catalog_data:
            logger.warning(f"⚠️ No catalog data found for {self.brand_domain}")
            return self._get_minimal_filters()
        
        logger.info(f"📊 Analyzing {len(catalog_data)} products for filter extraction")
        
        # Extract different types of filters
        categorical_filters = self._extract_categorical_filters(catalog_data)
        numeric_filters = self._extract_numeric_filters(catalog_data)
        multi_select_filters = self._extract_multi_select_filters(catalog_data)
        text_based_filters = self._extract_text_based_filters(catalog_data)
        
        # Combine all filters
        all_filters = {
            **categorical_filters,
            **numeric_filters,
            **multi_select_filters,
            **text_based_filters
        }
        
        # Add metadata
        all_filters["_metadata"] = {
            "total_products": len(catalog_data),
            "generated_at": "2025-06-28",
            "brand_domain": self.brand_domain,
            "analyzer_version": "1.0"
        }
        
        logger.info(f"✅ Extracted {len(all_filters) - 1} filter types from catalog")
        
        return all_filters
    
    async def _load_catalog_data(self) -> List[Dict[str, Any]]:
        """Load product catalog data using storage provider"""
        
        try:
            # Use storage provider to get product catalog
            products = await self.storage.get_product_catalog(self.brand_domain)
            
            if products:
                logger.info(f"📂 Loaded catalog from storage provider: {len(products)} products")
                return products
            else:
                logger.warning(f"⚠️ No product catalog found for {self.brand_domain}")
                return []
                
        except Exception as e:
            logger.error(f"⚠️ Failed to load catalog for {self.brand_domain}: {e}")
            return []
    
    def _extract_categorical_filters(self, catalog_data: List[Product]) -> Dict[str, Any]:
        """Extract categorical filters using explicit Product model structure"""
        
        categorical_filters = {}
        
        # Use explicit Product model fields instead of guessing
        explicit_categorical_fields = {
            "categories": "Product Category",
            "brand": "Brand", 
            "colors": "Color Options",
            "sizes": "Size Options"
        }
        
        # Extract from generated/enhanced fields
        generated_fields = {
            "search_keywords": "Search Keywords",
            "key_selling_points": "Key Features"
        }
        
        # Process explicit Product fields
        for field, label in explicit_categorical_fields.items():
            values = self._extract_product_field_values(catalog_data, field)
            if len(values) >= 2:  # At least 2 distinct values
                aliases = self._generate_aliases_for_values(list(values))
                
                categorical_filters[field] = {
                    "type": "categorical",
                    "label": label,
                    "values": sorted(list(values)),
                    "aliases": aliases,
                    "frequency": {val: self._count_product_field_value(catalog_data, field, val) for val in values}
                }
        
        # Process Product.specifications dynamically
        spec_filters = self._extract_specification_filters(catalog_data)
        categorical_filters.update(spec_filters)
        
        # Process generated search keywords and selling points for additional categorical data
        keyword_categories = self._extract_keyword_categories(catalog_data)
        categorical_filters.update(keyword_categories)
        
        # Process generated product labels for precise filtering
        label_filters = self._extract_product_label_filters(catalog_data)
        categorical_filters.update(label_filters)
        
        return categorical_filters
    
    def _extract_product_field_values(self, catalog_data: List[Product], field: str) -> set:
        """Extract values from explicit Product model fields"""
        values = set()
        
        for product in catalog_data:
            field_value = getattr(product, field, None)
            
            if field_value:
                if isinstance(field_value, list):
                    # Handle list fields (categories, colors, sizes, etc.)
                    for item in field_value:
                        if isinstance(item, str) and item.strip():
                            values.add(item.strip().lower())
                        elif isinstance(item, dict) and 'name' in item:
                            # Handle complex color objects
                            values.add(item['name'].strip().lower())
                elif isinstance(field_value, str) and field_value.strip():
                    values.add(field_value.strip().lower())
        
        return values
    
    def _count_product_field_value(self, catalog_data: List[Product], field: str, value: str) -> int:
        """Count occurrences of a value in a Product model field"""
        count = 0
        
        for product in catalog_data:
            field_value = getattr(product, field, None)
            
            if field_value:
                if isinstance(field_value, list):
                    for item in field_value:
                        item_str = item.get('name', item) if isinstance(item, dict) else str(item)
                        if item_str.strip().lower() == value:
                            count += 1
                elif isinstance(field_value, str) and field_value.strip().lower() == value:
                    count += 1
        
        return count
    
    def _extract_specification_filters(self, catalog_data: List[Product]) -> Dict[str, Any]:
        """Extract filters from Product.specifications dynamically"""
        spec_filters = {}
        
        # Collect all specification keys across products
        all_spec_keys = set()
        for product in catalog_data:
            if hasattr(product, 'specifications') and product.specifications:
                all_spec_keys.update(product.specifications.keys())
        
        # Analyze each specification key
        for spec_key in all_spec_keys:
            if not spec_key or len(spec_key) < 2:  # Skip empty or very short keys
                continue
                
            values = set()
            numeric_values = []
            
            for product in catalog_data:
                if hasattr(product, 'specifications') and product.specifications:
                    spec_value = product.specifications.get(spec_key)
                    
                    if spec_value:
                        # Try to parse as numeric for range filters
                        try:
                            if isinstance(spec_value, (int, float)):
                                numeric_values.append(float(spec_value))
                            elif isinstance(spec_value, str):
                                # Extract numeric part
                                numeric_match = re.search(r'[\d,]+\.?\d*', spec_value.replace(',', ''))
                                if numeric_match:
                                    numeric_values.append(float(numeric_match.group()))
                                else:
                                    # Treat as categorical
                                    values.add(str(spec_value).strip().lower())
                        except (ValueError, AttributeError):
                            values.add(str(spec_value).strip().lower())
            
            # Create filter based on data type
            if len(numeric_values) >= 3:
                # Numeric range filter
                spec_filters[f"spec_{spec_key}"] = {
                    "type": "numeric_range",
                    "label": f"{spec_key} (Specification)",
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                    "unit": self._detect_unit_from_key(spec_key),
                    "source": "specifications"
                }
            elif len(values) >= 2:
                # Categorical filter
                spec_filters[f"spec_{spec_key}"] = {
                    "type": "categorical",
                    "label": f"{spec_key} (Specification)",
                    "values": sorted(list(values)),
                    "aliases": self._generate_aliases_for_values(list(values)),
                    "frequency": {val: sum(1 for p in catalog_data 
                                         if hasattr(p, 'specifications') and p.specifications 
                                         and str(p.specifications.get(spec_key, '')).strip().lower() == val) 
                                 for val in values},
                    "source": "specifications"
                }
        
        return spec_filters
    
    def _extract_keyword_categories(self, catalog_data: List[Product]) -> Dict[str, Any]:
        """Extract categorical filters from generated search_keywords and key_selling_points"""
        keyword_filters = {}
        
        # Analyze search keywords for patterns
        all_keywords = []
        for product in catalog_data:
            if hasattr(product, 'search_keywords') and product.search_keywords:
                all_keywords.extend([kw.strip().lower() for kw in product.search_keywords])
        
        # Find common keyword patterns (themes)
        keyword_counter = Counter(all_keywords)
        frequent_keywords = {kw: count for kw, count in keyword_counter.items() 
                           if count >= max(2, len(catalog_data) * 0.1)}  # At least 10% of products
        
        if frequent_keywords:
            keyword_filters["search_themes"] = {
                "type": "multi_select",
                "label": "Search Themes",
                "values": sorted(list(frequent_keywords.keys())),
                "aliases": {},
                "frequency": frequent_keywords,
                "source": "generated_keywords"
            }
        
        # Analyze key selling points for common features
        all_selling_points = []
        for product in catalog_data:
            if hasattr(product, 'key_selling_points') and product.key_selling_points:
                all_selling_points.extend([sp.strip().lower() for sp in product.key_selling_points])
        
        selling_point_counter = Counter(all_selling_points)
        frequent_points = {sp: count for sp, count in selling_point_counter.items() 
                         if count >= max(2, len(catalog_data) * 0.1)}
        
        if frequent_points:
            keyword_filters["key_features"] = {
                "type": "multi_select", 
                "label": "Key Features",
                "values": sorted(list(frequent_points.keys())),
                "aliases": {},
                "frequency": frequent_points,
                "source": "generated_selling_points"
            }
        
        return keyword_filters
    
    def _extract_product_label_filters(self, catalog_data: List[Product]) -> Dict[str, Any]:
        """Extract precise filters from generated product_labels"""
        label_filters = {}
        
        # Collect all label categories across products
        all_categories = set()
        for product in catalog_data:
            if hasattr(product, 'product_labels') and product.product_labels:
                all_categories.update(product.product_labels.keys())
        
        # Process each label category
        for category in all_categories:
            all_labels = []
            
            # Collect all labels in this category
            for product in catalog_data:
                if hasattr(product, 'product_labels') and product.product_labels:
                    labels = product.product_labels.get(category, [])
                    if labels:
                        all_labels.extend(labels)
            
            if all_labels:
                # Count frequency of each label
                label_counter = Counter(all_labels)
                frequent_labels = {label: count for label, count in label_counter.items() 
                                 if count >= max(2, len(catalog_data) * 0.1)}  # At least 10% frequency
                
                if frequent_labels:
                    label_filters[f"{category}"] = {
                        "type": "multi_select",
                        "label": f"{category.replace('_', ' ').title()}",
                        "values": sorted(list(frequent_labels.keys())),
                        "aliases": {},
                        "frequency": frequent_labels,
                        "source": "generated_labels"
                    }
        
        return label_filters
    
    def _detect_unit_from_key(self, spec_key: str) -> Optional[str]:
        """Detect unit from specification key name"""
        key_lower = spec_key.lower()
        
        unit_patterns = {
            "weight": "kg",
            "mass": "kg", 
            "length": "cm",
            "width": "cm",
            "height": "cm",
            "diameter": "mm",
            "size": "cm",
            "price": "$",
            "cost": "$"
        }
        
        for pattern, unit in unit_patterns.items():
            if pattern in key_lower:
                return unit
                
        return None
    
    def _extract_numeric_filters(self, catalog_data: List[Product]) -> Dict[str, Any]:
        """Extract numeric range filters using explicit Product model fields"""
        
        numeric_filters = {}
        
        # Explicit numeric fields from Product model
        numeric_fields = {
            "salePrice": "Sale Price",
            "originalPrice": "Original Price"
        }
        
        for field, label in numeric_fields.items():
            numeric_values = self._extract_product_numeric_values(catalog_data, field)
            if len(numeric_values) >= 3:  # At least 3 products with this field
                min_val = min(numeric_values)
                max_val = max(numeric_values)
                
                # Generate price ranges for price fields
                common_ranges = []
                if "price" in field.lower():
                    common_ranges = self._generate_price_ranges(numeric_values)
                
                numeric_filters[field] = {
                    "type": "numeric_range",
                    "label": label,
                    "min": min_val,
                    "max": max_val,
                    "common_ranges": common_ranges,
                    "unit": "$" if "price" in field.lower() else None,
                    "distribution": self._analyze_distribution(numeric_values),
                    "source": "product_model"
                }
        
        return numeric_filters
    
    def _extract_product_numeric_values(self, catalog_data: List[Product], field: str) -> List[float]:
        """Extract numeric values from explicit Product model fields"""
        values = []
        
        for product in catalog_data:
            field_value = getattr(product, field, None)
            
            if field_value is not None:
                try:
                    if isinstance(field_value, (int, float)):
                        values.append(float(field_value))
                    elif isinstance(field_value, str):
                        # Handle price strings like "$1,999.99"
                        cleaned = field_value.replace('$', '').replace(',', '').strip()
                        if cleaned:
                            values.append(float(cleaned))
                except (ValueError, AttributeError):
                    continue
        
        return values
    
    def _extract_multi_select_filters(self, catalog_data: List[Product]) -> Dict[str, Any]:
        """Extract multi-select filters (features, tags, attributes)"""
        
        multi_select_filters = {}
        
        # Look for array/list fields or comma-separated values
        multi_value_fields = [
            "features", "attributes", "specs", "capabilities",
            "tags", "keywords", "categories", "uses", "applications",
            "components", "included", "accessories"
        ]
        
        for field in multi_value_fields:
            all_values = set()
            
            # Extract values from all products
            for product in catalog_data:
                values = self._extract_multi_values(product, field)
                all_values.update(values)
            
            # Only include if we have enough distinct values
            if len(all_values) >= 3:
                # Calculate frequency of each value
                value_counts = Counter()
                for product in catalog_data:
                    values = self._extract_multi_values(product, field)
                    value_counts.update(values)
                
                # Filter by minimum frequency
                min_count = max(2, int(len(catalog_data) * self.min_feature_frequency))
                frequent_values = [val for val, count in value_counts.items() if count >= min_count]
                
                if frequent_values:
                    aliases = self._generate_aliases_for_values(frequent_values)
                    
                    multi_select_filters[field] = {
                        "type": "multi_select",
                        "values": sorted(frequent_values),
                        "aliases": aliases,
                        "frequency": dict(value_counts)
                    }
        
        return multi_select_filters
    
    def _extract_text_based_filters(self, catalog_data: List[Product]) -> Dict[str, Any]:
        """Extract filters from text analysis (descriptions, names)"""
        
        text_filters = {}
        
        # Analyze product names for common patterns
        names = [getattr(product, "name", "") for product in catalog_data if getattr(product, "name", "")]
        name_patterns = self._extract_name_patterns(names)
        
        if name_patterns:
            text_filters["product_line"] = {
                "type": "categorical",
                "values": list(name_patterns.keys()),
                "aliases": {},
                "frequency": name_patterns,
                "source": "product_names"
            }
        
        # Analyze descriptions for use cases
        descriptions = [getattr(product, "description", "") for product in catalog_data if getattr(product, "description", "")]
        use_cases = self._extract_use_cases(descriptions)
        
        if use_cases:
            text_filters["intended_use"] = {
                "type": "multi_select",
                "values": list(use_cases.keys()),
                "aliases": self._generate_use_case_aliases(),
                "frequency": use_cases,
                "source": "descriptions"
            }
        
        return text_filters
    
    def _extract_field_values(self, catalog_data: List[Product], field: str) -> Set[str]:
        """Extract unique values for a specific field"""
        
        values = set()
        
        for product in catalog_data:
            value = getattr(product, field, "")
            if value and isinstance(value, str):
                # Clean and normalize the value
                cleaned_value = value.strip().lower()
                if cleaned_value and len(cleaned_value) > 1:
                    values.add(cleaned_value)
        
        return values
    
    def _extract_multi_values(self, product: Product, field: str) -> List[str]:
        """Extract multiple values from a field (arrays or comma-separated)"""
        
        value = getattr(product, field, None)
        if not value:
            return []
        
        if isinstance(value, list):
            return [str(v).strip().lower() for v in value if v]
        elif isinstance(value, str):
            # Try comma, semicolon, or pipe separation
            for separator in [',', ';', '|']:
                if separator in value:
                    return [v.strip().lower() for v in value.split(separator) if v.strip()]
            # Single value
            return [value.strip().lower()]
        
        return []
    
    def _generate_aliases_for_values(self, values: List[str]) -> Dict[str, List[str]]:
        """Generate aliases for categorical values"""
        
        aliases = {}
        
        for value in values:
            value_aliases = []
            
            # Add common variations
            if ' ' in value:
                # Add version without spaces
                value_aliases.append(value.replace(' ', ''))
                # Add version with underscores
                value_aliases.append(value.replace(' ', '_'))
            
            # Add plural/singular variations
            if value.endswith('s') and len(value) > 3:
                value_aliases.append(value[:-1])  # Remove 's'
            elif not value.endswith('s'):
                value_aliases.append(value + 's')  # Add 's'
            
            # Add common abbreviations based on domain knowledge
            if self.brand_domain and "bike" in self.brand_domain.lower():
                bike_aliases = {
                    "mountain": ["mtb", "mountain bike"],
                    "road": ["road bike", "roadie"],
                    "electric": ["e-bike", "ebike", "pedal assist"],
                    "hybrid": ["comfort bike", "city bike"]
                }
                if value in bike_aliases:
                    value_aliases.extend(bike_aliases[value])
            
            if value_aliases:
                aliases[value] = value_aliases
        
        return aliases
    
    def _generate_price_ranges(self, prices: List[float]) -> List[Dict[str, Any]]:
        """Generate common price ranges based on price distribution"""
        
        if not prices:
            return []
        
        sorted_prices = sorted(prices)
        min_price = min(prices)
        max_price = max(prices)
        
        # Generate quartile-based ranges
        q1 = sorted_prices[len(sorted_prices) // 4]
        q2 = sorted_prices[len(sorted_prices) // 2]  # median
        q3 = sorted_prices[3 * len(sorted_prices) // 4]
        
        ranges = [
            {"label": "budget", "range": [min_price, q1]},
            {"label": "mid-range", "range": [q1, q3]},
            {"label": "premium", "range": [q3, max_price]}
        ]
        
        return ranges
    
    def _detect_unit(self, field: str) -> Optional[str]:
        """Detect the unit for numeric fields"""
        
        unit_mapping = {
            "price": "$",
            "cost": "$",
            "msrp": "$",
            "weight": "kg",
            "length": "cm",
            "width": "cm", 
            "height": "cm",
            "diameter": "mm"
        }
        
        return unit_mapping.get(field)
    
    def _analyze_distribution(self, values: List[float]) -> Dict[str, float]:
        """Analyze the distribution of numeric values"""
        
        if not values:
            return {}
        
        sorted_vals = sorted(values)
        
        return {
            "mean": sum(values) / len(values),
            "median": sorted_vals[len(sorted_vals) // 2],
            "std_dev": self._calculate_std_dev(values)
        }
    
    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def _extract_name_patterns(self, names: List[str]) -> Dict[str, int]:
        """Extract product line patterns from product names"""
        
        # Look for common prefixes/series names
        pattern_counts = Counter()
        
        for name in names:
            if not name:
                continue
                
            # Extract first word (often the series name)
            first_word = name.split()[0] if name.split() else ""
            if len(first_word) > 2 and first_word.isalpha():
                pattern_counts[first_word.lower()] += 1
            
            # Look for patterns like "Series 123" or "Model ABC"
            series_match = re.search(r'(\w+)\s+\d+', name)
            if series_match:
                pattern_counts[series_match.group(1).lower()] += 1
        
        # Only return patterns that appear multiple times
        return {pattern: count for pattern, count in pattern_counts.items() if count >= 2}
    
    def _extract_use_cases(self, descriptions: List[str]) -> Dict[str, int]:
        """Extract use cases from product descriptions"""
        
        use_case_patterns = [
            r'\bfor (\w+ing)\b',  # "for racing", "for commuting"
            r'\bideal for (\w+)\b',  # "ideal for beginners"
            r'\bperfect for (\w+)\b',  # "perfect for trails"
            r'\bdesigned for (\w+)\b'  # "designed for speed"
        ]
        
        use_case_counts = Counter()
        
        for desc in descriptions:
            if not desc:
                continue
                
            desc_lower = desc.lower()
            for pattern in use_case_patterns:
                matches = re.findall(pattern, desc_lower)
                use_case_counts.update(matches)
        
        # Filter by minimum frequency
        min_count = max(2, int(len(descriptions) * 0.05))  # 5% minimum
        return {use_case: count for use_case, count in use_case_counts.items() if count >= min_count}
    
    def _generate_use_case_aliases(self) -> Dict[str, List[str]]:
        """Generate aliases for common use cases"""
        
        return {
            "racing": ["competition", "competitive", "performance"],
            "commuting": ["daily", "work", "transport", "urban"],
            "touring": ["long distance", "travel", "bikepacking"],
            "training": ["fitness", "exercise", "workout"],
            "recreational": ["leisure", "casual", "fun"]
        }
    
    def _count_field_value(self, catalog_data: List[Product], field: str, value: str) -> int:
        """Count how many products have a specific field value"""
        
        count = 0
        for product in catalog_data:
            product_value = getattr(product, field, "").strip().lower()
            if product_value == value:
                count += 1
        return count
    
    def _get_minimal_filters(self) -> Dict[str, Any]:
        """Return minimal filter set when no catalog data is available"""
        
        return {
            "category": {
                "type": "categorical",
                "values": [],
                "aliases": {},
                "note": "No catalog data available"
            },
            "price": {
                "type": "numeric_range",
                "min": 0,
                "max": 10000,
                "common_ranges": [
                    {"label": "budget", "range": [0, 1000]},
                    {"label": "mid-range", "range": [1000, 5000]},
                    {"label": "premium", "range": [5000, 10000]}
                ]
            }
        }
    
    async def save_filters_to_file(self, filters: Dict[str, Any], filename: str = "catalog_filters.json"):
        """Save extracted filters using storage provider"""
        
        try:
            # Convert filters to JSON string
            filters_json = json.dumps(filters, indent=2)
            
            # Save using storage provider
            success = await self.storage.write_file(
                account=self.brand_domain,
                file_path=filename,
                content=filters_json,
                content_type="application/json"
            )
            
            if success:
                logger.info(f"💾 Saved catalog filters to {filename} via storage provider")
            else:
                logger.error(f"❌ Failed to save catalog filters to {filename}")
                
        except Exception as e:
            logger.error(f"❌ Error saving catalog filters: {e}")


# Factory function for easy usage
async def analyze_brand_catalog(brand_domain: str, catalog_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Analyze a brand's catalog and return available filters"""
    
    analyzer = CatalogFilterAnalyzer(brand_domain)
    filters = await analyzer.analyze_product_catalog(catalog_data)
    
    # Save for future use
    await analyzer.save_filters_to_file(filters)
    
    return filters