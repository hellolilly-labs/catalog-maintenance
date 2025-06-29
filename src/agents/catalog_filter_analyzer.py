"""
Catalog Filter Analyzer

Analyzes product catalogs to dynamically extract available filters and labels
for query optimization. This ensures brand-specific terminology and categories
are discovered automatically rather than hardcoded.
"""

import json
import logging
import re
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
    
    def __init__(self, brand_domain: str):
        self.brand_domain = brand_domain
        self.brand_path = Path(f"accounts/{brand_domain}")
        
        # Thresholds for filter inclusion
        self.min_category_count = 2  # At least 2 products to be a category
        self.min_feature_frequency = 0.1  # Feature must appear in 10% of products
        
        logger.info(f"🔍 Initialized Catalog Filter Analyzer for {brand_domain}")
    
    def analyze_product_catalog(self, catalog_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Analyze product catalog to extract available filters.
        
        Args:
            catalog_data: Optional product data. If None, will try to load from filesystem.
            
        Returns:
            Dictionary of available filters with their types and values
        """
        
        if catalog_data is None:
            catalog_data = self._load_catalog_data()
        
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
    
    def _load_catalog_data(self) -> List[Dict[str, Any]]:
        """Load product catalog data from filesystem"""
        
        # Try multiple potential catalog file locations
        potential_paths = [
            self.brand_path / "catalog.json",
            self.brand_path / "products.json", 
            self.brand_path / "product_catalog.json",
            self.brand_path / "data" / "products.json"
        ]
        
        for path in potential_paths:
            if path.exists():
                logger.info(f"📂 Loading catalog from {path}")
                try:
                    with open(path) as f:
                        data = json.load(f)
                        
                    # Handle different data structures
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict):
                        # Try common keys
                        for key in ["products", "items", "catalog", "data"]:
                            if key in data and isinstance(data[key], list):
                                return data[key]
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load {path}: {e}")
        
        logger.warning(f"⚠️ No product catalog found for {self.brand_domain}")
        return []
    
    def _extract_categorical_filters(self, catalog_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract categorical filters (single-value categories)"""
        
        categorical_filters = {}
        
        # Analyze common categorical fields
        categorical_fields = [
            "category", "type", "product_type", "classification",
            "gender", "target_gender", "style",
            "brand", "manufacturer", "collection", "series",
            "material", "frame_material", "construction",
            "color", "primary_color", "finish"
        ]
        
        for field in categorical_fields:
            values = self._extract_field_values(catalog_data, field)
            if len(values) >= 2:  # At least 2 distinct values
                # Create aliases by analyzing value patterns
                aliases = self._generate_aliases_for_values(list(values))
                
                categorical_filters[field] = {
                    "type": "categorical",
                    "values": sorted(list(values)),
                    "aliases": aliases,
                    "frequency": {val: self._count_field_value(catalog_data, field, val) for val in values}
                }
        
        return categorical_filters
    
    def _extract_numeric_filters(self, catalog_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract numeric range filters (price, weight, dimensions)"""
        
        numeric_filters = {}
        
        # Analyze numeric fields
        numeric_fields = [
            "price", "cost", "msrp", "retail_price",
            "weight", "mass", "total_weight",
            "size", "frame_size", "wheel_size",
            "length", "width", "height", "diameter"
        ]
        
        for field in numeric_fields:
            numeric_values = self._extract_numeric_values(catalog_data, field)
            if len(numeric_values) >= 3:  # At least 3 products with this field
                min_val = min(numeric_values)
                max_val = max(numeric_values)
                
                # Generate common ranges for price
                common_ranges = []
                if field in ["price", "cost", "msrp", "retail_price"]:
                    common_ranges = self._generate_price_ranges(numeric_values)
                
                numeric_filters[field] = {
                    "type": "numeric_range",
                    "min": min_val,
                    "max": max_val,
                    "common_ranges": common_ranges,
                    "unit": self._detect_unit(field),
                    "distribution": self._analyze_distribution(numeric_values)
                }
        
        return numeric_filters
    
    def _extract_multi_select_filters(self, catalog_data: List[Dict[str, Any]]) -> Dict[str, Any]:
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
    
    def _extract_text_based_filters(self, catalog_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract filters from text analysis (descriptions, names)"""
        
        text_filters = {}
        
        # Analyze product names for common patterns
        names = [product.get("name", "") for product in catalog_data if product.get("name")]
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
        descriptions = [product.get("description", "") for product in catalog_data if product.get("description")]
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
    
    def _extract_field_values(self, catalog_data: List[Dict[str, Any]], field: str) -> Set[str]:
        """Extract unique values for a specific field"""
        
        values = set()
        
        for product in catalog_data:
            value = product.get(field)
            if value and isinstance(value, str):
                # Clean and normalize the value
                cleaned_value = value.strip().lower()
                if cleaned_value and len(cleaned_value) > 1:
                    values.add(cleaned_value)
        
        return values
    
    def _extract_numeric_values(self, catalog_data: List[Dict[str, Any]], field: str) -> List[float]:
        """Extract numeric values for a specific field"""
        
        values = []
        
        for product in catalog_data:
            value = product.get(field)
            if value is not None:
                # Try to convert to float
                try:
                    if isinstance(value, (int, float)):
                        values.append(float(value))
                    elif isinstance(value, str):
                        # Extract numeric part from strings like "$1,999" or "7.2kg"
                        numeric_match = re.search(r'[\d,]+\.?\d*', value.replace(',', ''))
                        if numeric_match:
                            values.append(float(numeric_match.group()))
                except (ValueError, AttributeError):
                    continue
        
        return values
    
    def _extract_multi_values(self, product: Dict[str, Any], field: str) -> List[str]:
        """Extract multiple values from a field (arrays or comma-separated)"""
        
        value = product.get(field)
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
    
    def _count_field_value(self, catalog_data: List[Dict[str, Any]], field: str, value: str) -> int:
        """Count how many products have a specific field value"""
        
        count = 0
        for product in catalog_data:
            product_value = product.get(field, "").strip().lower()
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
    
    def save_filters_to_file(self, filters: Dict[str, Any], filename: str = "catalog_filters.json"):
        """Save extracted filters to file for reuse"""
        
        output_path = self.brand_path / filename
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(filters, f, indent=2)
        
        logger.info(f"💾 Saved catalog filters to {output_path}")


# Factory function for easy usage
def analyze_brand_catalog(brand_domain: str, catalog_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Analyze a brand's catalog and return available filters"""
    
    analyzer = CatalogFilterAnalyzer(brand_domain)
    filters = analyzer.analyze_product_catalog(catalog_data)
    
    # Save for future use
    analyzer.save_filters_to_file(filters)
    
    return filters