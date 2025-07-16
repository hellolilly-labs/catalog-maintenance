#!/usr/bin/env python3
"""
Convert Specialized CSV to products.json with variant data

This script:
1. Parses the Specialized Hybris CSV feed
2. Creates Product objects with full variant data
3. Exports to a standardized products.json format
4. Preserves all variant information (inventory, GTIN, images, etc.)
"""

import sys
import os
sys.path.append('packages')

import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from liddy_intelligence.ingestion.parsers.specialized_csv_parser import parse_specialized_csv
from liddy.models.product import Product

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_latest_csv(account: str) -> Optional[str]:
    """Find the latest CSV file for an account"""
    csv_dir = Path(f"local/account_storage/accounts/{account}")
    csv_files = list(csv_dir.glob("*.csv"))
    
    if not csv_files:
        return None
    
    # Return most recent file
    csv_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(csv_files[0])


def convert_csv_to_json(csv_path: str, output_path: str, account: str = "specialized.com") -> bool:
    """
    Convert Specialized CSV to products.json
    
    Args:
        csv_path: Path to input CSV file
        output_path: Path to output JSON file
        account: Account name (defaults to specialized.com)
        
    Returns:
        bool: True if successful
    """
    try:
        logger.info(f"📄 Converting CSV to JSON...")
        logger.info(f"  Input: {csv_path}")
        logger.info(f"  Output: {output_path}")
        
        # Parse CSV
        logger.info(f"  Parsing CSV...")
        products = parse_specialized_csv(csv_path, account)
        
        if not products:
            logger.error("  ❌ No products parsed from CSV")
            return False
        
        logger.info(f"  ✅ Parsed {len(products)} products")
        
        # Calculate statistics
        total_variants = sum(len(p.variants) for p in products)
        multi_variant_products = [p for p in products if len(p.variants) > 1]
        
        logger.info(f"  📊 Statistics:")
        logger.info(f"     - Total products: {len(products)}")
        logger.info(f"     - Total variants: {total_variants}")
        logger.info(f"     - Products with multiple variants: {len(multi_variant_products)}")
        logger.info(f"     - Average variants per product: {total_variants/len(products):.1f}")
        
        # Show sample product with variants
        if multi_variant_products:
            sample = multi_variant_products[0]
            logger.info(f"\n  📦 Sample product with variants:")
            logger.info(f"     Name: {sample.name}")
            logger.info(f"     ID: {sample.id}")
            logger.info(f"     Variants: {len(sample.variants)}")
            logger.info(f"     Price range: ${sample.price_range()[0]:.2f} - ${sample.price_range()[1]:.2f}")
            logger.info(f"     Total inventory: {sample.get_total_inventory()}")
            logger.info(f"     Sizes: {sample.sizes}")
            logger.info(f"     Colors: {sample.colors}")
        
        # Convert to dict format
        logger.info(f"\n  💾 Converting to JSON format...")
        products_data = [p.to_dict() for p in products]
        
        # Add metadata
        metadata = {
            "_metadata": {
                "source": "specialized_csv",
                "csv_file": os.path.basename(csv_path),
                "converted_at": datetime.now().isoformat(),
                "product_count": len(products),
                "variant_count": total_variants,
                "converter_version": "1.0.0"
            }
        }
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Write JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write metadata as first object, then products
            json.dump({
                "_metadata": metadata["_metadata"],
                "products": products_data
            }, f, indent=2, ensure_ascii=False)
        
        # Calculate file size
        file_size = os.path.getsize(output_path)
        file_size_mb = file_size / 1024 / 1024
        
        logger.info(f"  ✅ Written {output_path} ({file_size_mb:.1f} MB)")
        
        # Verify the file can be read back
        logger.info(f"\n  🔍 Verifying JSON file...")
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            verified_products = data.get("products", [])
            verified_metadata = data.get("_metadata", {})
            
        logger.info(f"  ✅ Verified {len(verified_products)} products in JSON")
        logger.info(f"  ✅ Metadata: {verified_metadata.get('source')} - {verified_metadata.get('converted_at')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert Specialized CSV to products.json')
    parser.add_argument('--account', default='specialized.com', help='Account name')
    parser.add_argument('--csv', help='Path to CSV file (auto-detects latest if not provided)')
    parser.add_argument('--output', help='Output JSON path (defaults to account storage)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without converting')
    
    args = parser.parse_args()
    
    # Find CSV file
    csv_path = args.csv
    if not csv_path:
        csv_path = find_latest_csv(args.account)
        if not csv_path:
            logger.error(f"❌ No CSV files found for {args.account}")
            return 1
    
    # Determine output path
    output_path = args.output
    if not output_path:
        output_path = f"local/account_storage/accounts/{args.account}/products_with_variants.json"
    
    # Dry run
    if args.dry_run:
        logger.info(f"🔍 Dry run mode:")
        logger.info(f"  Would convert: {csv_path}")
        logger.info(f"  To: {output_path}")
        
        # Check if files exist
        if os.path.exists(csv_path):
            csv_size = os.path.getsize(csv_path) / 1024 / 1024
            logger.info(f"  CSV exists: {csv_size:.1f} MB")
        else:
            logger.error(f"  CSV not found!")
            
        if os.path.exists(output_path):
            logger.warning(f"  Output file already exists and would be overwritten")
            
        return 0
    
    # Perform conversion
    success = convert_csv_to_json(csv_path, output_path, args.account)
    
    if success:
        logger.info(f"\n✅ Conversion complete!")
        logger.info(f"Next steps:")
        logger.info(f"  1. Review the generated JSON: {output_path}")
        logger.info(f"  2. Run product catalog updater to ingest into main catalog")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())