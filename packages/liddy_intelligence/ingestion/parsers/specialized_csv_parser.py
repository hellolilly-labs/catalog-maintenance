"""
Specialized CSV Parser for Variant-Aware Product Ingestion

Parses Specialized's Hybris CSV feed format which includes variant-level data
(inventory, pricing, images, etc.) and converts to unified Product/ProductVariant format.
"""

import csv
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from collections import defaultdict

from liddy.models.product import Product
from liddy.models.product_variant import ProductVariant

logger = logging.getLogger(__name__)


class SpecializedCSVParser:
    """
    Parser for Specialized's CSV product feed with variant data.
    
    The CSV contains one row per variant with fields like:
    - SKU_ARTICLE_NUMBER: Unique variant SKU
    - PID: Parent product ID
    - MPL_PRODUCT_ID: Model/product grouping ID
    - SIZE, COLOR-DEFAULT: Variant attributes
    - CONSUMERPRICE, CONSUMER_DISCOUNTED_PRICE: Pricing
    - AVAILABLE_QTY: Inventory
    - BARCODE_VALUES: GTIN (pipe-separated)
    - PART_IMAGE_URL_VALUES: Images (pipe-separated)
    """
    
    # CSV column mappings
    COLUMN_MAPPINGS = {
        # Variant identifiers
        'SKU_ARTICLE_NUMBER': 'variant_sku',
        'PID': 'product_id',
        'MPL_PRODUCT_ID': 'model_id',
        
        # Product info
        'ITEM_DESCRIPTION': 'name',
        'PRODUCT_TEXT': 'description',
        'PRODUCT_URL': 'product_url',
        'PRODUCT_CATEGORY': 'category',
        'MODEL_YEAR': 'year',
        
        # Variant attributes
        'SIZE': 'size',
        'COLOR-DEFAULT': 'color',
        'COLOR_ID': 'color_id',
        'COLOR_VALUE': 'color_hex',
        
        # Pricing
        'CONSUMERPRICE': 'price',
        'CONSUMER_DISCOUNTED_PRICE': 'sale_price',
        'CURRENCY': 'currency',
        
        # Inventory
        'AVAILABLE_QTY': 'inventory',
        'IN_STOCK': 'in_stock',
        
        # Media
        'PART_IMAGE_URL_VALUES': 'images',
        'VIDEO_URL': 'video_url',
        
        # Additional data
        'BARCODE_VALUES': 'gtin',
        'DEFAULT_VARIANT_ID': 'default_variant_id',
        'PRODUCT_SPECIFICATION': 'specifications',
        'BULLET_POINTS': 'highlights'
    }
    
    def __init__(self, brand_domain: str):
        self.brand_domain = brand_domain
        self.products_by_id: Dict[str, Product] = {}
        self.variants_by_product: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def parse_csv(self, csv_path: str) -> List[Product]:
        """
        Parse Specialized CSV and return list of Products with variants.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            List of Product objects with populated variants
        """
        logger.info(f"📄 Parsing Specialized CSV: {csv_path}")
        
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Read and parse CSV with UTF-8 encoding, handling BOM
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            
            row_count = 0
            for row in reader:
                self._process_row(row)
                row_count += 1
            
            logger.info(f"  ✅ Processed {row_count} variant rows")
        
        # Build Product objects from collected data
        products = self._build_products()
        
        logger.info(f"  📦 Created {len(products)} products from variants")
        return products
    
    def _process_row(self, row: Dict[str, str]):
        """Process a single CSV row representing a variant"""
        # Map CSV columns to our internal names
        mapped_data = {}
        for csv_col, internal_name in self.COLUMN_MAPPINGS.items():
            if csv_col in row:
                mapped_data[internal_name] = row[csv_col]
        
        # Extract key identifiers
        variant_sku = mapped_data.get('variant_sku', '').strip()
        product_id = mapped_data.get('product_id', '').strip()
        model_id = mapped_data.get('model_id', '').strip()
        
        if not variant_sku or not product_id:
            logger.debug(f"Skipping row - SKU: '{variant_sku}', PID: '{product_id}', Model: '{model_id}'")
            logger.debug(f"Row data: {list(row.keys())[:5]}")
            return
        
        # Use model_id as the primary product grouping if available
        parent_id = model_id or product_id
        
        # Store variant data grouped by parent product
        self.variants_by_product[parent_id].append({
            'raw_data': mapped_data,
            'variant_sku': variant_sku,
            'is_default': mapped_data.get('default_variant_id') == variant_sku
        })
    
    def _build_products(self) -> List[Product]:
        """Build Product objects from collected variant data"""
        products = []
        
        for parent_id, variant_data_list in self.variants_by_product.items():
            if not variant_data_list:
                continue
            
            # Use first variant's data for product-level fields
            first_variant = variant_data_list[0]['raw_data']
            
            # Create base product
            product = Product(
                id=parent_id,
                name=self._clean_text(first_variant.get('name', '')),
                brand='Specialized',  # Hardcoded for Specialized parser
                productUrl=first_variant.get('product_url', ''),
                categories=self._parse_categories(first_variant.get('category', '')),
                description=self._clean_text(first_variant.get('description', '')),
                year=first_variant.get('year', ''),
                variants=[]
            )
            
            # Process specifications (same for all variants)
            if 'specifications' in first_variant:
                product.specifications = self._parse_specifications(
                    first_variant['specifications']
                )
            
            # Process highlights
            if 'highlights' in first_variant:
                product.highlights = self._parse_pipe_separated(
                    first_variant['highlights']
                )
            
            # Build variants
            for variant_info in variant_data_list:
                variant = self._build_variant(variant_info, parent_id)
                product.variants.append(variant)
            
            # Set product-level price from variants
            self._set_product_pricing(product)
            
            # Collect all variant images for product
            self._collect_product_images(product)
            
            products.append(product)
        
        return products
    
    def _build_variant(self, variant_info: Dict, parent_id: str) -> ProductVariant:
        """Build a ProductVariant from CSV data"""
        data = variant_info['raw_data']
        
        # Parse pricing
        price = self._format_price(data.get('sale_price') or data.get('price', ''))
        original_price = self._format_price(data.get('price', ''))
        
        # Parse inventory
        inventory_qty = None
        try:
            qty_str = data.get('inventory', '').strip()
            if qty_str and qty_str.isdigit():
                inventory_qty = int(qty_str)
        except:
            pass
        
        # Parse images (pipe-separated URLs)
        image_urls = self._parse_pipe_separated(data.get('images', ''))
        
        # Parse GTIN/barcode (may be pipe-separated)
        gtin_values = self._parse_pipe_separated(data.get('gtin', ''))
        gtin = gtin_values[0] if gtin_values else None
        
        # Build variant
        variant = ProductVariant(
            id=variant_info['variant_sku'],
            productId=parent_id,
            price=price,
            originalPrice=original_price if original_price != price else None,
            inventoryQuantity=inventory_qty,
            imageUrls=image_urls,
            gtin=gtin,
            isDefault=variant_info['is_default'],
            attributes={
                'size': data.get('size', '').strip(),
                'color': data.get('color', '').strip(),
                'color_id': data.get('color_id', '').strip(),
                'color_hex': data.get('color_hex', '').strip()
            }
        )
        
        # Set stock status based on inventory
        if inventory_qty is not None:
            variant.inStock = inventory_qty > 0
        elif 'in_stock' in data:
            variant.inStock = data['in_stock'].lower() in ('true', 'yes', '1')
        
        return variant
    
    def _set_product_pricing(self, product: Product):
        """Set product-level pricing from variants"""
        if not product.variants:
            return
        
        # Get price range from variants
        min_price, max_price = product.price_range()
        
        # Set product prices from default variant or first variant
        default_variant = product.get_default_variant()
        if default_variant:
            product.originalPrice = default_variant.originalPrice or default_variant.price
            product.salePrice = default_variant.price if default_variant.price != default_variant.originalPrice else None
        elif product.variants:
            # Use first variant
            first = product.variants[0]
            product.originalPrice = first.originalPrice or first.price
            product.salePrice = first.price if first.price != first.originalPrice else None
    
    def _collect_product_images(self, product: Product):
        """Collect all unique images from variants"""
        all_images = []
        seen_urls = set()
        
        for variant in product.variants:
            for url in variant.imageUrls:
                if url and url not in seen_urls:
                    all_images.append(url)
                    seen_urls.add(url)
        
        product.imageUrls = all_images
    
    def _parse_categories(self, category_str: str) -> List[str]:
        """Parse category string into list"""
        if not category_str:
            return []
        
        # Handle different separators
        if '>' in category_str:
            return [c.strip() for c in category_str.split('>')]
        elif '/' in category_str:
            return [c.strip() for c in category_str.split('/')]
        else:
            return [category_str.strip()]
    
    def _parse_specifications(self, spec_str: str) -> Dict[str, Any]:
        """Parse specification string into dictionary"""
        if not spec_str:
            return {}
        
        specs = {}
        
        # Handle pipe-separated key:value pairs
        if '|' in spec_str:
            for pair in spec_str.split('|'):
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    specs[key.strip()] = value.strip()
        # Handle line-separated
        elif '\n' in spec_str:
            for line in spec_str.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    specs[key.strip()] = value.strip()
        
        return specs
    
    def _parse_pipe_separated(self, value: str) -> List[str]:
        """Parse pipe-separated values into list"""
        if not value:
            return []
        
        return [v.strip() for v in value.split('|') if v.strip()]
    
    def _format_price(self, price_str: str) -> Optional[str]:
        """Format price string to standard format"""
        if not price_str:
            return None
        
        price_str = price_str.strip()
        
        # Already formatted
        if price_str.startswith('$'):
            return price_str
        
        # Parse and format
        try:
            # Remove any currency symbols or commas
            cleaned = price_str.replace('$', '').replace(',', '').strip()
            if cleaned and cleaned.replace('.', '').isdigit():
                price_float = float(cleaned)
                return f"${price_float:,.2f}"
        except:
            pass
        
        return None
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text fields"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove special characters that might break parsing
        text = text.replace('\x00', '').replace('\r', '\n')
        
        return text.strip()


# Convenience function
def parse_specialized_csv(csv_path: str, brand_domain: str = "specialized.com") -> List[Product]:
    """
    Parse Specialized CSV file and return products with variants.
    
    Args:
        csv_path: Path to CSV file
        brand_domain: Brand domain (defaults to specialized.com)
        
    Returns:
        List of Product objects with variants
    """
    parser = SpecializedCSVParser(brand_domain)
    return parser.parse_csv(csv_path)