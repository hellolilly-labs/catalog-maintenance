#!/usr/bin/env python3
"""
Update Product Descriptor Prices

This script checks all product descriptors and ensures they contain
current price information. It can also handle bulk price updates.

Usage:
    # Check and update all products for an account
    python run/update_descriptor_prices.py specialized.com
    
    # Update a single product's price
    python run/update_descriptor_prices.py specialized.com --product-id "12345" --sale-price "$999.99"
    
    # Bulk update from CSV
    python run/update_descriptor_prices.py specialized.com --csv-file price_updates.csv
"""

import os
import sys
import csv
import asyncio
import argparse
import logging

# Add packages directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # catalog-maintenance directory
packages_dir = os.path.join(project_root, 'packages')

# Add to Python path
sys.path.insert(0, packages_dir)

from liddy_intelligence.catalog.price_descriptor_updater import PriceDescriptorUpdater

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def check_and_update_all_products(account: str):
    """Check all products and update with price information"""
    logger.info(f"Checking and updating product descriptors for {account}")
    
    updater = PriceDescriptorUpdater(account)
    stats = await updater.check_and_update_products(force_refresh=True)
    
    logger.info("\n=== Update Statistics ===")
    logger.info(f"Total products checked: {stats['total_checked']}")
    logger.info(f"Products updated with price: {stats['updated_count']}")
    logger.info(f"Products already had price: {stats['already_had_price']}")
    logger.info(f"Errors encountered: {stats['errors']}")
    
    return stats


async def update_single_product(account: str, product_id: str, sale_price: str = None, original_price: str = None):
    """Update a single product's price"""
    logger.info(f"Updating price for product {product_id}")
    
    updater = PriceDescriptorUpdater(account)
    success = await updater.update_single_product_price(
        product_id=product_id,
        new_sale_price=sale_price,
        new_original_price=original_price
    )
    
    if success:
        logger.info(f"Successfully updated product {product_id}")
    else:
        logger.error(f"Failed to update product {product_id}")
    
    return success


async def bulk_update_from_csv(account: str, csv_file: str):
    """Bulk update prices from a CSV file"""
    logger.info(f"Bulk updating prices from {csv_file}")
    
    # Read CSV file
    price_updates = []
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'product_id' in row and 'sale_price' in row:
                    price_updates.append({
                        'product_id': row['product_id'],
                        'sale_price': row['sale_price']
                    })
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        return
    
    if not price_updates:
        logger.warning("No valid price updates found in CSV")
        return
    
    logger.info(f"Found {len(price_updates)} price updates")
    
    updater = PriceDescriptorUpdater(account)
    stats = await updater.bulk_update_sale_prices(price_updates)
    
    logger.info("\n=== Bulk Update Statistics ===")
    logger.info(f"Products updated: {stats['updated']}")
    logger.info(f"Products not found: {stats['not_found']}")
    logger.info(f"Failed updates: {stats['failed']}")
    
    return stats


async def create_sample_csv(filename: str = "sample_price_updates.csv"):
    """Create a sample CSV file for bulk updates"""
    sample_data = [
        {'product_id': '12345', 'sale_price': '$999.99'},
        {'product_id': '67890', 'sale_price': '$1,299.00'},
        {'product_id': 'ABC123', 'sale_price': '$449.99'}
    ]
    
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['product_id', 'sale_price'])
        writer.writeheader()
        writer.writerows(sample_data)
    
    logger.info(f"Created sample CSV file: {filename}")


def main():
    parser = argparse.ArgumentParser(description='Update product descriptors with price information')
    parser.add_argument('account', help='Account/brand domain (e.g., specialized.com)')
    parser.add_argument('--product-id', help='Update a specific product ID')
    parser.add_argument('--sale-price', help='New sale price (e.g., $999.99)')
    parser.add_argument('--original-price', help='New original price')
    parser.add_argument('--csv-file', help='CSV file with bulk price updates')
    parser.add_argument('--create-sample-csv', action='store_true', help='Create a sample CSV file')
    
    args = parser.parse_args()
    
    # Run the appropriate action
    if args.create_sample_csv:
        asyncio.run(create_sample_csv())
    elif args.csv_file:
        asyncio.run(bulk_update_from_csv(args.account, args.csv_file))
    elif args.product_id:
        asyncio.run(update_single_product(
            args.account, 
            args.product_id, 
            args.sale_price, 
            args.original_price
        ))
    else:
        asyncio.run(check_and_update_all_products(args.account))


if __name__ == "__main__":
    main()