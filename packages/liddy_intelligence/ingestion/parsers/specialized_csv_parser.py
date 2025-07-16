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
        'TITLE-DEFAULT': 'name',
        'PRODUCT_TITLE-DEFAULT': 'product_title',
        'DESCRIPTION-DEFAULT': 'description',
        'LINK-DEFAULT': 'product_url',
        'CATEGORY_STRUCTURE-DEFAULT': 'category',
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
        'AVAILABILITY': 'in_stock',
        
        # Media
        'IMAGE_LINK': 'image_link',
        'PLP_IMAGE': 'plp_image',
        'VIDEO_URL': 'video_url',
        
        # Additional data
        'GTIN': 'gtin',
        'MPN': 'mpn',
        'DEFAULT_VARIANT_ID': 'default_variant_id',
        'PRODUCT_SPECIFICATION': 'specifications',
        'web_bullets_en_all': 'highlights',
        'FEATURES-DEFAULT': 'features',
        
        # Specifications
        'FRAMEMATERIAL-DEFAULT': 'frame_material',
        'WHEELSIZE-DEFAULT': 'wheel_size',
        'SUSPENSION-DEFAULT': 'suspension',
        'BRAKETYPE-DEFAULT': 'brake_type',
        'SUSPENSION BRAND-DEFAULT': 'suspension_brand',
        'MATERIAL-DEFAULT': 'material',
        'TECHNOLOGY-DEFAULT': 'technology',
        'REAR SUSPENSION TRAVEL-DEFAULT': 'rear_suspension_travel',
        'DRIVETRAIN-DEFAULT': 'drivetrain',
        'DRIVETRAIN BRAND-DEFAULT': 'drivetrain_brand',
        'MOTOR TYPE-DEFAULT': 'motor_type',
        'BATTERY SIZE-DEFAULT': 'battery_size',
        'TORQUE-DEFAULT': 'torque',
        'POWER-DEFAULT': 'power'
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
        
        # Check if this variant is the default
        default_variant_id = mapped_data.get('default_variant_id', '').strip()
        is_default = default_variant_id == variant_sku if default_variant_id else False
        
        # Store variant data grouped by parent product
        self.variants_by_product[parent_id].append({
            'raw_data': mapped_data,
            'variant_sku': variant_sku,
            'is_default': is_default
        })
    
    def _build_products(self) -> List[Product]:
        """Build Product objects from collected variant data"""
        products = []
        
        for parent_id, variant_data_list in self.variants_by_product.items():
            if not variant_data_list:
                continue
            
            # Use first variant's data for product-level fields
            first_variant = variant_data_list[0]['raw_data']
            
            # Get name from available fields (prefer product_title over name)
            product_name = (first_variant.get('product_title') or 
                          first_variant.get('name') or '').strip()
            
            # Create base product
            product = Product(
                id=parent_id,
                name=self._clean_text(product_name),
                brand='Specialized',  # Hardcoded for Specialized parser
                productUrl=first_variant.get('product_url', ''),
                categories=self._parse_categories(first_variant.get('category', '')),
                description=self._clean_text(first_variant.get('description', '')),
                year=first_variant.get('year', ''),
                variants=[]
            )
            
            # Build specifications from individual fields
            product.specifications = self._build_specifications(first_variant)
            
            # Process highlights
            if 'highlights' in first_variant:
                product.highlights = self._parse_pipe_separated(
                    first_variant['highlights']
                )
            
            # Build variants
            for i, variant_info in enumerate(variant_data_list):
                # Mark first variant as default if no explicit default is set
                if not variant_info['is_default'] and i == 0:
                    variant_info['is_default'] = True
                    
                variant = self._build_variant(variant_info, parent_id)
                product.variants.append(variant)
            
            # Collect all variant images for product  
            self._collect_product_images(product)
            
            products.append(product)
        
        return products
    
    def _build_variant(self, variant_info: Dict, parent_id: str) -> ProductVariant:
        """Build a ProductVariant from CSV data"""
        data = variant_info['raw_data']
        
        # Parse pricing
        raw_price = data.get('price', '').strip()
        raw_sale_price = data.get('sale_price', '').strip()
        
        # Always set original price from the regular price field
        original_price = self._format_price(raw_price) if raw_price else None
        
        # If there's a sale price, use it as the current price
        if raw_sale_price:
            price = self._format_price(raw_sale_price)
        else:
            # No sale price, use regular price as current price
            price = original_price
        
        # Parse inventory
        inventory_qty = None
        try:
            qty_str = data.get('inventory', '').strip()
            if qty_str and qty_str.isdigit():
                inventory_qty = int(qty_str)
        except:
            pass
        
        # Parse images from available fields
        image_urls = []
        if data.get('image_link'):
            image_urls.append(data['image_link'])
        if data.get('plp_image'):
            image_urls.append(data['plp_image'])
        
        # Remove duplicates while preserving order
        unique_images = []
        seen = set()
        for url in image_urls:
            if url and url not in seen:
                unique_images.append(url)
                seen.add(url)
        
        # Parse GTIN/barcode
        gtin = data.get('gtin', '').strip() or None
        
        # Build variant
        variant = ProductVariant(
            id=variant_info['variant_sku'],
            productId=parent_id,
            price=price,
            originalPrice=original_price,
            inventoryQuantity=inventory_qty,
            imageUrls=unique_images,
            gtin=gtin,
            isDefault=variant_info['is_default'],
            url=data.get('product_url', ''),  # Use product URL for variant URL
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
        
        # The categories appear to be comma-separated tags rather than hierarchical
        # Split by comma and clean up each category
        categories = []
        for cat in category_str.split(','):
            cat = cat.strip()
            if cat:
                categories.append(cat)
        
        return categories
    
    def _build_specifications(self, data: Dict[str, str]) -> Dict[str, Any]:
        """Build specifications dictionary from CSV data"""
        specs = {}
        
        # Map specification fields
        spec_mappings = {
            'frame_material': 'Frame Material',
            'wheel_size': 'Wheel Size',
            'suspension': 'Suspension',
            'brake_type': 'Brake Type',
            'suspension_brand': 'Suspension Brand',
            'material': 'Material',
            'technology': 'Technology',
            'rear_suspension_travel': 'Rear Suspension Travel',
            'drivetrain': 'Drivetrain',
            'drivetrain_brand': 'Drivetrain Brand',
            'motor_type': 'Motor Type',
            'battery_size': 'Battery Size',
            'torque': 'Torque',
            'power': 'Power'
        }
        
        for field_name, display_name in spec_mappings.items():
            value = data.get(field_name, '').strip()
            if value:
                specs[display_name] = value
        
        return specs
    
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