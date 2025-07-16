#!/usr/bin/env python3
"""
Product Catalog Updater - Merge variant data with existing catalog

This script:
1. Loads existing product catalog
2. Loads new products with variant data from JSON
3. Intelligently merges:
   - Updates existing products with variant data
   - Preserves existing descriptors/metadata
   - Adds new products not in catalog
   - Updates inventory/pricing from variants
4. Saves updated catalog back to storage
"""

import sys
import os
sys.path.append('packages')

import json
import asyncio
import logging
import copy
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from liddy.models.product import Product, DescriptorMetadata
from liddy.models.product_manager import get_product_manager
from liddy.models.product_loader import ProductLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProductCatalogUpdater:
    """Updates existing product catalog with new variant data"""
    
    def __init__(self, account: str):
        self.account = account
        self.stats = {
            'existing_products': 0,
            'new_products': 0,
            'products_updated': 0,
            'products_added': 0,
            'products_unchanged': 0,
            'variants_added': 0,
            'prices_updated': 0,
            'inventory_updated': 0,
            'descriptor_regeneration_needed': 0,
            # Granular descriptor stats
            'needs_llm_regeneration': 0,
            'needs_price_update': 0,
            'needs_variant_update': 0,
            'missing_descriptors': 0,
            'incomplete_descriptors': 0
        }
    
    async def load_existing_catalog(self) -> Dict[str, Product]:
        """Load existing product catalog indexed by ID"""
        logger.info("📚 Loading existing product catalog...")
        
        # Use ProductManager to load existing products
        product_manager = await get_product_manager(self.account)
        products = await product_manager.get_products()
        
        # Index by ID for quick lookup
        catalog = {str(p.id): p for p in products}
        
        self.stats['existing_products'] = len(catalog)
        logger.info(f"  ✅ Loaded {len(catalog)} existing products")
        
        return catalog
    
    def load_new_products(self, json_path: str) -> List[Product]:
        """Load new products from JSON file"""
        logger.info(f"📄 Loading new products from: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both formats: with/without metadata
        if isinstance(data, dict) and 'products' in data:
            products_data = data['products']
            metadata = data.get('_metadata', {})
            logger.info(f"  📊 Source: {metadata.get('source', 'unknown')}")
            logger.info(f"  📊 Converted: {metadata.get('converted_at', 'unknown')}")
        else:
            products_data = data if isinstance(data, list) else []
        
        # Convert to Product objects
        products = [Product.from_dict(p) for p in products_data]
        
        self.stats['new_products'] = len(products)
        logger.info(f"  ✅ Loaded {len(products)} new products")
        
        # Calculate variant stats
        total_variants = sum(len(p.variants) for p in products)
        logger.info(f"  📊 Total variants: {total_variants}")
        
        return products
    
    def merge_product(self, existing: Product, new: Product) -> Tuple[Product, Dict[str, int]]:
        """
        Merge new product data with existing, preserving important fields
        
        Returns:
            Tuple of (merged_product, change_stats)
        """
        changes = {
            'variants_added': 0,
            'price_changed': False,
            'inventory_changed': False,
            'fields_updated': 0,
            'descriptor_relevant_changed': False,
            'descriptor_fields_changed': [],
            # Granular descriptor component tracking
            'needs_llm_regeneration': False,  # Content changes (name, description, highlights, etc.)
            'needs_price_update': False,       # Price changes only
            'needs_variant_update': False,     # Color/size changes from variants
            'llm_fields_changed': [],
            'variant_fields_changed': []
        }
        
        # Start with existing product (make a copy to avoid modifying original)
        merged = copy.deepcopy(existing)
        
        # Capture state before variant update for intelligent comparison
        old_variant_attributes = existing.get_all_variant_attributes() if hasattr(existing, 'get_all_variant_attributes') else {}
        
        # Update variants (this is the main update)
        if new.variants:
            old_variant_count = len(existing.variants) if existing.variants else 0
            merged.variants = new.variants
            changes['variants_added'] = len(new.variants) - old_variant_count
        
        # Update pricing from variants
        if new.variants:
            # Get existing prices for comparison
            existing_sale = existing.sale_price if hasattr(existing, 'sale_price') else existing.salePrice
            existing_orig = existing.original_price if hasattr(existing, 'original_price') else existing.originalPrice
            
            # Get prices from new variants
            default_variant = new.get_default_variant()
            if default_variant:
                new_price = default_variant.price
                new_original = default_variant.originalPrice or default_variant.price
            else:
                new_price = new.variants[0].price
                new_original = new.variants[0].originalPrice or new.variants[0].price
            
            # Normalize prices for comparison (handle $$ vs $ issue)
            normalized_existing_sale = existing_sale.replace('$$', '$') if existing_sale else None
            normalized_existing_orig = existing_orig.replace('$$', '$') if existing_orig else None
            
            # Only consider it a price change if values actually differ
            # Don't flag as changed if we're just setting the same price from variants
            price_actually_changed = False
            
            # Special case: if salePrice was None but originalPrice equals new price,
            # this isn't really a change that affects the descriptor
            if not normalized_existing_sale and normalized_existing_orig == new_price:
                # Just formatting the price into salePrice field
                logger.debug(f"    Price formatting only: moving {normalized_existing_orig} to salePrice")
            elif normalized_existing_sale != new_price:
                # Check if it's a real change or just format difference
                if normalized_existing_sale and new_price:
                    # Both have values, compare them
                    if normalized_existing_sale != new_price:
                        price_actually_changed = True
                elif new_price:
                    # Sale price is being added where there was none
                    price_actually_changed = True
            
            if normalized_existing_orig != new_original:
                if normalized_existing_orig and new_original:
                    if normalized_existing_orig != new_original:
                        price_actually_changed = True
                elif normalized_existing_orig or new_original:
                    price_actually_changed = True
            
            if price_actually_changed:
                changes['price_changed'] = True
                logger.debug(f"    Price changed: sale={normalized_existing_sale}->{new_price}, orig={normalized_existing_orig}->{new_original}")
        
        # Track descriptor-relevant changes by component
        descriptor_fields = []
        llm_fields = []  # Fields that require LLM regeneration
        
        # Update basic fields if they're empty in existing
        if not merged.name and new.name:
            merged.name = new.name
            changes['fields_updated'] += 1
            descriptor_fields.append('name')
            llm_fields.append('name')
        
        if not merged.productUrl and new.productUrl:
            merged.productUrl = new.productUrl
            changes['fields_updated'] += 1
        
        if not merged.imageUrls and new.imageUrls:
            merged.imageUrls = new.imageUrls
            changes['fields_updated'] += 1
            descriptor_fields.append('images')
            llm_fields.append('images')
        
        if not merged.categories and new.categories:
            merged.categories = new.categories
            changes['fields_updated'] += 1
            descriptor_fields.append('categories')
            llm_fields.append('categories')
        
        # Update year if newer
        if new.year:
            try:
                # Convert to int for comparison
                new_year_int = int(new.year)
                merged_year_int = int(merged.year) if merged.year else 0
                
                if new_year_int > merged_year_int:
                    merged.year = new.year
                    changes['fields_updated'] += 1
                    descriptor_fields.append('year')
                    llm_fields.append('year')
            except (ValueError, TypeError):
                # If year can't be converted to int, just update if merged.year is empty
                if not merged.year:
                    merged.year = new.year
                    changes['fields_updated'] += 1
                    descriptor_fields.append('year')
                    llm_fields.append('year')
        
        # Update highlights (affects descriptor)
        if new.highlights and new.highlights != existing.highlights:
            merged.highlights = new.highlights
            changes['fields_updated'] += 1
            descriptor_fields.append('highlights')
            llm_fields.append('highlights')
        
        # Update description (affects descriptor)
        if new.description and new.description != existing.description:
            merged.description = new.description
            changes['fields_updated'] += 1
            descriptor_fields.append('description')
            llm_fields.append('description')
        
        # Update specifications (affects descriptor)
        if new.specifications and new.specifications != existing.specifications:
            merged.specifications = new.specifications
            changes['fields_updated'] += 1
            descriptor_fields.append('specifications')
            llm_fields.append('specifications')
        
        # Intelligent comparison for variant attributes
        variant_fields = []
        if new.variants:
            # Get new variant attributes after update
            new_variant_attributes = merged.get_all_variant_attributes() if hasattr(merged, 'get_all_variant_attributes') else {}
            
            # Compare all variant attributes
            all_attr_names = set(old_variant_attributes.keys()) | set(new_variant_attributes.keys())
            
            for attr_name in all_attr_names:
                old_values = sorted(old_variant_attributes.get(attr_name, []))
                new_values = sorted(new_variant_attributes.get(attr_name, []))
                
                if old_values != new_values:
                    descriptor_fields.append(f'{attr_name}_from_variants')
                    variant_fields.append(attr_name)
                    logger.debug(f"    {attr_name.capitalize()} changed: {old_values} -> {new_values}")
        
        # Set granular component flags
        if llm_fields:
            changes['needs_llm_regeneration'] = True
            changes['llm_fields_changed'] = llm_fields
        
        if changes['price_changed']:
            changes['needs_price_update'] = True
            descriptor_fields.append('price')
        
        if variant_fields:
            changes['needs_variant_update'] = True
            changes['variant_fields_changed'] = variant_fields
        
        # Overall descriptor change flag
        if descriptor_fields or changes['price_changed']:
            changes['descriptor_relevant_changed'] = True
            changes['descriptor_fields_changed'] = descriptor_fields
        
        # Check inventory change
        old_inventory = existing.get_total_inventory() if existing.variants else 0
        new_inventory = merged.get_total_inventory()
        if old_inventory != new_inventory:
            changes['inventory_changed'] = True
        
        # Update timestamp
        merged.updated = datetime.now().isoformat()
        
        # Mark for descriptor regeneration if relevant fields changed
        if changes['descriptor_relevant_changed']:
            # Handle DescriptorMetadata object - add custom attributes
            if hasattr(merged, 'descriptor_metadata') and merged.descriptor_metadata:
                # Add attributes directly to the object
                merged.descriptor_metadata.needs_regeneration = True
                merged.descriptor_metadata.fields_changed = changes['descriptor_fields_changed']
                merged.descriptor_metadata.last_catalog_update = datetime.now().isoformat()
            else:
                # Create a basic metadata object if none exists
                merged.descriptor_metadata = DescriptorMetadata()
                merged.descriptor_metadata.needs_regeneration = True
                merged.descriptor_metadata.fields_changed = changes['descriptor_fields_changed']
                merged.descriptor_metadata.last_catalog_update = datetime.now().isoformat()
        
        return merged, changes
    
    def check_descriptor_completeness(self, product: Product) -> Dict[str, Any]:
        """
        Check if a product's descriptor has all three components:
        1. LLM-generated content
        2. Price information
        3. Variant information (colors/sizes)
        
        Returns dict with component status and missing info
        """
        completeness = {
            'has_descriptor': False,
            'has_price_info': False,
            'has_variant_info': False,
            'is_complete': False,
            'missing_variant_info': []
        }
        
        if not product.descriptor:
            return completeness
        
        descriptor = product.descriptor.lower()
        completeness['has_descriptor'] = True
        
        # Check for price information
        sale_price = product.sale_price if hasattr(product, 'sale_price') else product.salePrice
        orig_price = product.original_price if hasattr(product, 'original_price') else product.originalPrice
        if sale_price or orig_price:
            price = (sale_price or orig_price).replace('$', '').replace(',', '')
            if price in descriptor:
                completeness['has_price_info'] = True
        
        # Check for variant information (any attributes)
        variant_attributes = {}  # attribute_name -> list of values
        
        # Collect all unique attribute types and their values from variants
        if product.variants:
            for variant in product.variants:
                if variant.attributes:
                    for attr_name, attr_value in variant.attributes.items():
                        if attr_name not in variant_attributes:
                            variant_attributes[attr_name] = set()
                        variant_attributes[attr_name].add(str(attr_value))
        
        # For backward compatibility, also check legacy color/size fields
        if not variant_attributes:
            colors = product.colors if hasattr(product, 'colors') else []
            sizes = product.sizes if hasattr(product, 'sizes') else []
            if colors:
                variant_attributes['color'] = set(colors)
            if sizes:
                variant_attributes['size'] = set(sizes)
        
        # Check if all variant attributes are mentioned in descriptor
        missing_attributes = []
        all_attributes_found = True
        
        for attr_name, attr_values in variant_attributes.items():
            attr_found = False
            # Check if any value of this attribute is in the descriptor
            for value in attr_values:
                if value.lower() in descriptor:
                    attr_found = True
                    break
            
            if not attr_found:
                all_attributes_found = False
                missing_attributes.append(attr_name)
        
        # Set completeness based on whether all attributes are represented
        if variant_attributes:
            completeness['has_variant_info'] = all_attributes_found
            completeness['missing_variant_info'] = missing_attributes
        else:
            # No variant data, so not required
            completeness['has_variant_info'] = True
        
        # Complete if all components present
        completeness['is_complete'] = all([
            completeness['has_descriptor'],
            completeness['has_price_info'],
            completeness['has_variant_info']
        ])
        
        return completeness
    
    async def update_catalog(self, new_products_path: str, dry_run: bool = False) -> Dict[str, int]:
        """
        Update product catalog with new variant data
        
        Args:
            new_products_path: Path to JSON file with new products
            dry_run: If True, don't save changes
            
        Returns:
            Statistics about the update
        """
        # Load existing catalog
        existing_catalog = await self.load_existing_catalog()
        
        # Load new products
        new_products = self.load_new_products(new_products_path)
        
        # Process updates
        logger.info("\n🔄 Processing updates...")
        updated_products = []
        
        for new_product in new_products:
            product_id = str(new_product.id)
            
            if product_id in existing_catalog:
                # Merge with existing
                existing = existing_catalog[product_id]
                merged, changes = self.merge_product(existing, new_product)
                
                # Check if anything changed
                if any([changes['variants_added'], changes['price_changed'], 
                       changes['inventory_changed'], changes['fields_updated']]):
                    updated_products.append(merged)
                    self.stats['products_updated'] += 1
                    
                    if changes['variants_added'] > 0:
                        self.stats['variants_added'] += changes['variants_added']
                    if changes['price_changed']:
                        self.stats['prices_updated'] += 1
                    if changes['inventory_changed']:
                        self.stats['inventory_updated'] += 1
                    if changes['descriptor_relevant_changed']:
                        self.stats['descriptor_regeneration_needed'] += 1
                    if changes['needs_llm_regeneration']:
                        self.stats['needs_llm_regeneration'] += 1
                    if changes['needs_price_update']:
                        self.stats['needs_price_update'] += 1
                    if changes['needs_variant_update']:
                        self.stats['needs_variant_update'] += 1
                    
                    logger.info(f"  Updated {product_id}: +{changes['variants_added']} variants")
                    if changes['descriptor_relevant_changed']:
                        components = []
                        if changes['needs_llm_regeneration']:
                            components.append(f"LLM ({', '.join(changes['llm_fields_changed'])})")
                        if changes['needs_price_update']:
                            components.append("Price")
                        if changes['needs_variant_update']:
                            components.append(f"Variants ({', '.join(changes['variant_fields_changed'])})")
                        logger.info(f"    Descriptor updates needed: {' | '.join(components)}")
                else:
                    self.stats['products_unchanged'] += 1
                
                # Remove from catalog so we know what's left
                del existing_catalog[product_id]
            else:
                # New product
                updated_products.append(new_product)
                self.stats['products_added'] += 1
                self.stats['variants_added'] += len(new_product.variants)
                logger.debug(f"  Added new product {product_id}")
        
        # Add remaining existing products (not in update)
        for existing in existing_catalog.values():
            updated_products.append(existing)
        
        # Check descriptor completeness for all products
        logger.info("\n🔍 Checking descriptor completeness...")
        for product in updated_products:
            completeness = self.check_descriptor_completeness(product)
            
            if not completeness['has_descriptor']:
                self.stats['missing_descriptors'] += 1
            elif not completeness['is_complete']:
                self.stats['incomplete_descriptors'] += 1
                
                # Mark for appropriate updates
                if hasattr(product, 'descriptor_metadata') and product.descriptor_metadata:
                    if not completeness['has_price_info']:
                        product.descriptor_metadata.needs_price_update = True
                    if not completeness['has_variant_info']:
                        product.descriptor_metadata.needs_variant_update = True
                        # Use the missing_variant_info from completeness check
                        if completeness['missing_variant_info']:
                            product.descriptor_metadata.missing_variant_info = completeness['missing_variant_info']
        
        # Save if not dry run
        if not dry_run:
            logger.info("\n💾 Saving updated catalog...")
            product_manager = await get_product_manager(self.account)
            success = await product_manager.save_products(updated_products)
            
            if success:
                logger.info(f"  ✅ Saved {len(updated_products)} products")
            else:
                logger.error("  ❌ Failed to save catalog")
                
        else:
            logger.info("\n🔍 Dry run - no changes saved")
        
        return self.stats


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Update product catalog with variant data')
    parser.add_argument('account', help='Account name (e.g., specialized.com)')
    parser.add_argument('--input', required=True, help='Path to new products JSON')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without saving')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed change information')
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create updater
    updater = ProductCatalogUpdater(args.account)
    
    # Run update
    stats = await updater.update_catalog(args.input, dry_run=args.dry_run)
    
    # Print summary
    logger.info("\n📊 Update Summary:")
    logger.info(f"  Existing products: {stats['existing_products']}")
    logger.info(f"  New products file: {stats['new_products']}")
    logger.info(f"  Products updated: {stats['products_updated']}")
    logger.info(f"  Products added: {stats['products_added']}")
    logger.info(f"  Products unchanged: {stats['products_unchanged']}")
    logger.info(f"  Variants added: {stats['variants_added']}")
    logger.info(f"  Prices updated: {stats['prices_updated']}")
    logger.info(f"  Inventory updated: {stats['inventory_updated']}")
    
    # Descriptor statistics
    logger.info("\n📝 Descriptor Status:")
    logger.info(f"  Missing descriptors: {stats['missing_descriptors']}")
    logger.info(f"  Incomplete descriptors: {stats['incomplete_descriptors']}")
    logger.info(f"  Total needing updates: {stats['descriptor_regeneration_needed']}")
    
    if stats['descriptor_regeneration_needed'] > 0 or stats['incomplete_descriptors'] > 0:
        logger.info("\n⚠️  Descriptor Updates Required:")
        if stats['needs_llm_regeneration'] > 0:
            logger.info(f"  • LLM regeneration needed: {stats['needs_llm_regeneration']} products")
        if stats['needs_price_update'] > 0:
            logger.info(f"  • Price update needed: {stats['needs_price_update']} products")
        if stats['needs_variant_update'] > 0:
            logger.info(f"  • Variant update needed: {stats['needs_variant_update']} products")
        
        logger.info("\nRun the following commands as needed:")
        if stats['needs_llm_regeneration'] > 0:
            logger.info("  python run/generate_descriptors.py specialized.com --regenerate-marked")
        if stats['needs_price_update'] > 0:
            logger.info("  python run/update_descriptor_prices.py specialized.com")
        if stats['needs_variant_update'] > 0:
            logger.info("  python run/update_descriptor_variants.py specialized.com")
    
    if args.dry_run:
        logger.info("\n⚠️  This was a dry run - no changes were saved")
        logger.info("Remove --dry-run to apply changes")


if __name__ == "__main__":
    asyncio.run(main())