"""
Universal Product Processor

Brand-agnostic product processing for any product type (fashion, beauty, sports, etc.)
Generates enhanced descriptors, extracts metadata, and prepares products for RAG ingestion.
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class UniversalProductProcessor:
    """
    Processes products from any brand/category into a universal format
    optimized for voice AI and semantic search.
    """
    
    def __init__(self, brand_domain: str):
        self.brand_domain = brand_domain
        self.brand_path = Path(f"accounts/{brand_domain}")
        
        # Load catalog insights if available
        self.catalog_insights = self._load_catalog_insights()
        
        # Universal field mappings (common across industries)
        self.universal_fields = {
            'identifier': ['id', 'sku', 'product_id', 'item_id', 'productId'],
            'name': ['name', 'title', 'product_name', 'productName', 'item_name'],
            'brand': ['brand', 'manufacturer', 'brand_name', 'vendor'],
            'price': ['price', 'salePrice', 'sale_price', 'originalPrice', 'original_price', 'current_price'],
            'description': ['description', 'product_description', 'long_description', 'descriptor'],
            'category': ['category', 'categories', 'product_type', 'type', 'classification'],
            'images': ['images', 'imageUrls', 'image_urls', 'photos', 'pictures'],
            'available': ['in_stock', 'inStock', 'available', 'availability', 'is_available']
        }
        
        # Voice optimization settings
        self.max_voice_length = 150  # words
        self.key_points_count = 3
        
        logger.info(f"🎯 Initialized Universal Product Processor for {brand_domain}")
    
    def process_product(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single product into universal format with enhanced content.
        
        Args:
            product: Raw product data
            
        Returns:
            Processed product with enhanced descriptor and metadata
        """
        
        # Extract universal fields
        universal_data = self._extract_universal_fields(product)
        
        # Generate enhanced descriptor for voice
        enhanced_descriptor = self._generate_enhanced_descriptor(product, universal_data)
        
        # Extract search keywords
        search_keywords = self._extract_search_keywords(product, universal_data)
        
        # Extract key selling points
        key_selling_points = self._extract_key_selling_points(product, universal_data)
        
        # Build filter metadata based on catalog insights
        filter_metadata = self._build_filter_metadata(product, universal_data)
        
        # Generate voice-optimized summary
        voice_summary = self._generate_voice_summary(universal_data, key_selling_points)
        
        return {
            'id': universal_data.get('identifier', f"product_{hash(str(product))}")[:100],
            'universal_fields': universal_data,
            'enhanced_descriptor': enhanced_descriptor,
            'voice_summary': voice_summary,
            'search_keywords': search_keywords,
            'key_selling_points': key_selling_points,
            'filter_metadata': filter_metadata,
            'original_data': product
        }
    
    def _extract_universal_fields(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Extract common fields using universal mappings."""
        
        universal_data = {}
        
        for universal_key, possible_keys in self.universal_fields.items():
            for key in possible_keys:
                if key in product and product[key]:
                    if universal_key == 'price':
                        # Handle price extraction
                        universal_data[universal_key] = self._extract_price(product[key])
                    elif universal_key == 'category':
                        # Handle category extraction
                        universal_data[universal_key] = self._extract_categories(product[key])
                    else:
                        universal_data[universal_key] = product[key]
                    break
        
        # Extract any additional important fields based on catalog insights
        if self.catalog_insights:
            important_fields = self.catalog_insights.get('important_fields', [])
            for field in important_fields:
                if field in product and field not in universal_data:
                    universal_data[field] = product[field]
        
        return universal_data
    
    def _extract_price(self, price_value: Any) -> Optional[float]:
        """Extract numeric price from various formats."""
        
        if isinstance(price_value, (int, float)):
            return float(price_value)
        
        if isinstance(price_value, str):
            # Remove currency symbols and extract number
            price_str = re.sub(r'[^\d.,]', '', price_value)
            price_str = price_str.replace(',', '')
            try:
                return float(price_str)
            except ValueError:
                return None
        
        if isinstance(price_value, dict):
            # Handle structured price objects
            for key in ['amount', 'value', 'price']:
                if key in price_value:
                    return self._extract_price(price_value[key])
        
        return None
    
    def _extract_categories(self, category_value: Any) -> List[str]:
        """Extract categories as a list."""
        
        if isinstance(category_value, list):
            return [str(cat).strip() for cat in category_value if cat]
        
        if isinstance(category_value, str):
            # Handle comma-separated or path-like categories
            if ',' in category_value:
                return [cat.strip() for cat in category_value.split(',')]
            elif '/' in category_value or '>' in category_value:
                separator = '/' if '/' in category_value else '>'
                return [cat.strip() for cat in category_value.split(separator)]
            else:
                return [category_value.strip()]
        
        return []
    
    def _generate_enhanced_descriptor(self, product: Dict[str, Any], universal_data: Dict[str, Any]) -> str:
        """
        Generate an enhanced descriptor optimized for voice AI and semantic search.
        """
        
        parts = []
        
        # Start with brand and name
        if universal_data.get('brand') and universal_data.get('name'):
            parts.append(f"{universal_data['brand']} {universal_data['name']}")
        elif universal_data.get('name'):
            parts.append(universal_data['name'])
        
        # Add main description
        if universal_data.get('description'):
            desc = str(universal_data['description'])
            # Clean up description
            desc = re.sub(r'\s+', ' ', desc).strip()
            # Limit length for voice
            if len(desc.split()) > 100:
                desc = ' '.join(desc.split()[:100]) + '...'
            parts.append(desc)
        
        # Add category context
        categories = universal_data.get('category', [])
        if categories:
            parts.append(f"This is a {', '.join(categories)}.")
        
        # Add price if available
        if universal_data.get('price'):
            parts.append(f"Priced at ${universal_data['price']:.2f}.")
        
        # Add any notable features from catalog insights
        if self.catalog_insights:
            notable_attributes = self._extract_notable_attributes(product)
            if notable_attributes:
                parts.append(f"Features include: {', '.join(notable_attributes)}.")
        
        # Add use cases or benefits
        use_cases = self._extract_use_cases(product)
        if use_cases:
            parts.append(f"Perfect for {', '.join(use_cases)}.")
        
        return ' '.join(parts)
    
    def _extract_search_keywords(self, product: Dict[str, Any], universal_data: Dict[str, Any]) -> List[str]:
        """Extract keywords for search optimization."""
        
        keywords = set()
        
        # Add brand and name variations
        if universal_data.get('brand'):
            brand = universal_data['brand']
            keywords.add(brand.lower())
            keywords.add(brand.upper())
            
        if universal_data.get('name'):
            name = universal_data['name']
            keywords.add(name.lower())
            # Add individual words from name
            for word in name.split():
                if len(word) > 2:
                    keywords.add(word.lower())
        
        # Add categories
        for category in universal_data.get('category', []):
            keywords.add(category.lower())
        
        # Extract keywords from description
        if universal_data.get('description'):
            desc_words = re.findall(r'\b\w{4,}\b', universal_data['description'].lower())
            keywords.update(desc_words[:20])  # Top 20 words
        
        # Add any SKU or product codes
        if 'identifier' in universal_data:
            keywords.add(str(universal_data['identifier']))
        
        # Add price range keywords
        if universal_data.get('price'):
            price = universal_data['price']
            if price < 50:
                keywords.add('budget')
                keywords.add('affordable')
            elif price < 200:
                keywords.add('mid-range')
            elif price < 1000:
                keywords.add('premium')
            else:
                keywords.add('luxury')
                keywords.add('high-end')
        
        return list(keywords)
    
    def _extract_key_selling_points(self, product: Dict[str, Any], universal_data: Dict[str, Any]) -> List[str]:
        """Extract key selling points for voice presentation."""
        
        selling_points = []
        
        # Look for features, benefits, highlights
        for key in ['features', 'benefits', 'highlights', 'key_features', 'selling_points']:
            if key in product:
                value = product[key]
                if isinstance(value, list):
                    selling_points.extend([str(item) for item in value[:3]])
                elif isinstance(value, str):
                    # Extract bullet points or sentences
                    points = re.split(r'[•\n]|(?<=[.!?])\s+', value)
                    selling_points.extend([p.strip() for p in points if len(p.strip()) > 10][:3])
        
        # If no explicit selling points, generate from description
        if not selling_points and universal_data.get('description'):
            sentences = re.split(r'(?<=[.!?])\s+', universal_data['description'])
            # Find sentences with positive indicators
            positive_indicators = ['perfect', 'ideal', 'great', 'excellent', 'best', 'premium', 'quality']
            for sentence in sentences[:10]:
                if any(indicator in sentence.lower() for indicator in positive_indicators):
                    selling_points.append(sentence.strip())
                    if len(selling_points) >= 3:
                        break
        
        # Ensure we have exactly 3 points (pad or trim)
        if len(selling_points) < 3:
            # Add generic points based on available data
            if universal_data.get('brand'):
                selling_points.append(f"Authentic {universal_data['brand']} product")
            if universal_data.get('category'):
                selling_points.append(f"Quality {universal_data['category'][0]}")
            if universal_data.get('price'):
                selling_points.append("Competitive pricing")
        
        return selling_points[:3]
    
    def _build_filter_metadata(self, product: Dict[str, Any], universal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build filter metadata based on catalog insights."""
        
        filters = {}
        
        # Add universal filters
        if universal_data.get('brand'):
            filters['brand'] = universal_data['brand']
        
        if universal_data.get('category'):
            filters['category'] = universal_data['category'][0] if universal_data['category'] else 'general'
            if len(universal_data['category']) > 1:
                filters['subcategories'] = universal_data['category'][1:]
        
        if universal_data.get('price'):
            price = universal_data['price']
            filters['price'] = price
            filters['price_range'] = self._get_price_range(price)
        
        # Add availability
        filters['available'] = universal_data.get('available', True)
        
        # Add dynamic filters from catalog insights
        if self.catalog_insights and 'filter_definitions' in self.catalog_insights:
            for filter_name, filter_config in self.catalog_insights['filter_definitions'].items():
                if filter_name in product:
                    filters[filter_name] = product[filter_name]
                elif 'extraction_path' in filter_config:
                    # Try to extract using path
                    value = self._extract_by_path(product, filter_config['extraction_path'])
                    if value:
                        filters[filter_name] = value
        
        return filters
    
    def _get_price_range(self, price: float) -> str:
        """Categorize price into range."""
        
        if price < 50:
            return 'budget'
        elif price < 200:
            return 'mid-range'
        elif price < 1000:
            return 'premium'
        else:
            return 'luxury'
    
    def _generate_voice_summary(self, universal_data: Dict[str, Any], key_points: List[str]) -> str:
        """Generate a concise summary optimized for voice output."""
        
        parts = []
        
        # Introduction
        name = universal_data.get('name', 'This product')
        brand = universal_data.get('brand', '')
        
        if brand:
            intro = f"{brand} {name}"
        else:
            intro = name
            
        parts.append(intro)
        
        # Add one key point
        if key_points:
            parts.append(key_points[0])
        
        # Add price if available
        if universal_data.get('price'):
            parts.append(f"Available for ${universal_data['price']:.2f}")
        
        summary = '. '.join(parts) + '.'
        
        # Ensure it's not too long for voice
        words = summary.split()
        if len(words) > self.max_voice_length:
            summary = ' '.join(words[:self.max_voice_length]) + '...'
        
        return summary
    
    def _load_catalog_insights(self) -> Optional[Dict[str, Any]]:
        """Load pre-analyzed catalog insights if available."""
        
        insights_path = self.brand_path / "catalog_filters.json"
        if insights_path.exists():
            try:
                with open(insights_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load catalog insights: {e}")
        
        return None
    
    def _extract_notable_attributes(self, product: Dict[str, Any]) -> List[str]:
        """Extract notable attributes based on catalog insights."""
        
        attributes = []
        
        if not self.catalog_insights:
            return attributes
        
        # Get important attributes for this brand
        important_attrs = self.catalog_insights.get('important_attributes', [])
        
        for attr in important_attrs[:5]:  # Limit to 5 attributes
            if attr in product and product[attr]:
                value = product[attr]
                if isinstance(value, bool):
                    if value:
                        attributes.append(attr.replace('_', ' '))
                elif isinstance(value, (str, int, float)):
                    attributes.append(f"{attr.replace('_', ' ')}: {value}")
                elif isinstance(value, list) and value:
                    attributes.append(f"{attr.replace('_', ' ')}: {value[0]}")
        
        return attributes
    
    def _extract_use_cases(self, product: Dict[str, Any]) -> List[str]:
        """Extract use cases or occasions for the product."""
        
        use_cases = []
        
        # Look for explicit use case fields
        for key in ['use_cases', 'occasions', 'perfect_for', 'ideal_for', 'suitable_for']:
            if key in product:
                value = product[key]
                if isinstance(value, list):
                    use_cases.extend([str(item) for item in value[:3]])
                elif isinstance(value, str):
                    use_cases.extend(re.split(r'[,;]', value)[:3])
        
        # If no explicit use cases, try to infer from categories
        if not use_cases:
            categories = self._extract_categories(product.get('category', []))
            for category in categories:
                category_lower = category.lower()
                if 'casual' in category_lower:
                    use_cases.append('everyday wear')
                elif 'formal' in category_lower or 'dress' in category_lower:
                    use_cases.append('special occasions')
                elif 'sport' in category_lower or 'athletic' in category_lower:
                    use_cases.append('active lifestyle')
                elif 'work' in category_lower or 'professional' in category_lower:
                    use_cases.append('professional settings')
        
        return use_cases[:3]
    
    def _extract_by_path(self, data: Dict[str, Any], path: str) -> Any:
        """Extract value from nested dict using dot notation path."""
        
        parts = path.split('.')
        current = data
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        return current
    
    def process_catalog(self, products: List[Dict[str, Any]], batch_size: int = 100) -> List[Dict[str, Any]]:
        """
        Process entire product catalog.
        
        Args:
            products: List of raw products
            batch_size: Batch size for processing
            
        Returns:
            List of processed products
        """
        
        logger.info(f"🏭 Processing {len(products)} products for {self.brand_domain}")
        
        processed = []
        for i in range(0, len(products), batch_size):
            batch = products[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(products) + batch_size - 1)//batch_size}")
            
            for product in batch:
                try:
                    processed_product = self.process_product(product)
                    processed.append(processed_product)
                except Exception as e:
                    logger.error(f"Error processing product: {e}")
                    logger.debug(f"Product data: {product}")
        
        logger.info(f"✅ Processed {len(processed)} products successfully")
        
        return processed