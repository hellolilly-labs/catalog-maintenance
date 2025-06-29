#!/usr/bin/env python3
"""
Product Catalog Ingestion and Filter Pre-Analysis

This script should be run when:
1. A new product catalog is added
2. An existing catalog is updated
3. Products are added/removed/modified

It analyzes the catalog once and generates the finite set of labels/filters
that will be used for real-time query optimization.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.agents.catalog_filter_analyzer import CatalogFilterAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ingest_catalog_from_file(brand_domain: str, catalog_file_path: str) -> Dict[str, Any]:
    """
    Ingest product catalog from file and generate filters.
    
    Args:
        brand_domain: The brand domain (e.g., "specialized.com")
        catalog_file_path: Path to the product catalog JSON file
        
    Returns:
        Dictionary of extracted filters
    """
    
    logger.info(f"🔄 Ingesting catalog for {brand_domain} from {catalog_file_path}")
    
    # Load catalog data
    catalog_path = Path(catalog_file_path)
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog file not found: {catalog_file_path}")
    
    with open(catalog_path) as f:
        catalog_data = json.load(f)
    
    # Handle different data structures
    if isinstance(catalog_data, dict):
        for key in ["products", "items", "catalog", "data"]:
            if key in catalog_data and isinstance(catalog_data[key], list):
                catalog_data = catalog_data[key]
                break
        else:
            raise ValueError("Could not find product list in catalog file")
    
    if not isinstance(catalog_data, list):
        raise ValueError("Catalog data must be a list of products")
    
    logger.info(f"📊 Found {len(catalog_data)} products in catalog")
    
    # Analyze catalog and extract filters
    analyzer = CatalogFilterAnalyzer(brand_domain)
    filters = analyzer.analyze_product_catalog(catalog_data)
    
    # Save filters for query optimization
    analyzer.save_filters_to_file(filters, "catalog_filters.json")
    
    # Also save a human-readable summary
    save_filter_summary(brand_domain, filters)
    
    logger.info(f"✅ Catalog ingestion complete for {brand_domain}")
    
    return filters


def ingest_catalog_from_api(brand_domain: str, api_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ingest product catalog from API endpoint.
    
    Args:
        brand_domain: The brand domain
        api_config: API configuration (endpoint, auth, etc.)
        
    Returns:
        Dictionary of extracted filters
    """
    
    logger.info(f"🌐 Ingesting catalog for {brand_domain} from API")
    
    # This would implement API fetching
    # For now, just a placeholder
    raise NotImplementedError("API ingestion not yet implemented")


def save_filter_summary(brand_domain: str, filters: Dict[str, Any]) -> None:
    """Save a human-readable summary of extracted filters"""
    
    summary_path = Path(f"accounts/{brand_domain}/filter_summary.md")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        f.write(f"# Filter Summary for {brand_domain}\n\n")
        f.write(f"Generated: {filters.get('_metadata', {}).get('generated_at', 'unknown')}\n")
        f.write(f"Total Products Analyzed: {filters.get('_metadata', {}).get('total_products', 'unknown')}\n\n")
        
        for filter_name, filter_config in filters.items():
            if filter_name.startswith('_'):
                continue
                
            filter_type = filter_config.get('type', 'unknown')
            f.write(f"## {filter_name} ({filter_type})\n\n")
            
            if filter_type == "categorical":
                values = filter_config.get('values', [])
                f.write(f"**Values:** {', '.join(values)}\n\n")
                
                aliases = filter_config.get('aliases', {})
                if aliases:
                    f.write("**Aliases:**\n")
                    for value, alias_list in aliases.items():
                        f.write(f"- {value}: {', '.join(alias_list)}\n")
                    f.write("\n")
                    
            elif filter_type == "numeric_range":
                min_val = filter_config.get('min')
                max_val = filter_config.get('max')
                unit = filter_config.get('unit', '')
                f.write(f"**Range:** {min_val} to {max_val} {unit}\n\n")
                
                ranges = filter_config.get('common_ranges', [])
                if ranges:
                    f.write("**Common Ranges:**\n")
                    for range_def in ranges:
                        label = range_def['label']
                        range_vals = range_def['range']
                        f.write(f"- {label}: {range_vals[0]} to {range_vals[1]}\n")
                    f.write("\n")
                    
            elif filter_type == "multi_select":
                values = filter_config.get('values', [])
                f.write(f"**Options:** {', '.join(values)}\n\n")
                
                frequency = filter_config.get('frequency', {})
                if frequency:
                    sorted_freq = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
                    f.write("**Most Common:**\n")
                    for value, count in sorted_freq[:5]:
                        f.write(f"- {value}: {count} products\n")
                    f.write("\n")
    
    logger.info(f"📄 Filter summary saved to {summary_path}")


def update_brand_filters(brand_domain: str, catalog_source: str) -> None:
    """
    Update filters for a specific brand.
    
    Args:
        brand_domain: The brand domain to update
        catalog_source: Path to catalog file or API config
    """
    
    try:
        if catalog_source.endswith('.json'):
            # File-based ingestion
            filters = ingest_catalog_from_file(brand_domain, catalog_source)
        else:
            # Assume API config
            api_config = json.loads(catalog_source)
            filters = ingest_catalog_from_api(brand_domain, api_config)
        
        # Display summary
        print(f"\n🎉 Successfully ingested catalog for {brand_domain}")
        print(f"📊 Extracted {len([k for k in filters.keys() if not k.startswith('_')])} filter types")
        
        # Show key filters
        for filter_name, filter_config in filters.items():
            if filter_name.startswith('_'):
                continue
                
            filter_type = filter_config.get('type')
            if filter_type == "categorical":
                values = filter_config.get('values', [])
                print(f"   📂 {filter_name}: {len(values)} categories")
            elif filter_type == "numeric_range":
                min_val = filter_config.get('min')
                max_val = filter_config.get('max')
                print(f"   📊 {filter_name}: {min_val} to {max_val}")
            elif filter_type == "multi_select":
                values = filter_config.get('values', [])
                print(f"   ☑️  {filter_name}: {len(values)} options")
        
        print(f"\n💾 Filters saved to: accounts/{brand_domain}/catalog_filters.json")
        print(f"📄 Summary saved to: accounts/{brand_domain}/filter_summary.md")
        
    except Exception as e:
        logger.error(f"❌ Failed to ingest catalog for {brand_domain}: {e}")
        raise


def main():
    """Main CLI interface for catalog ingestion"""
    
    parser = argparse.ArgumentParser(
        description="Ingest product catalogs and generate filters for query optimization"
    )
    parser.add_argument(
        "brand_domain",
        help="Brand domain (e.g., specialized.com)"
    )
    parser.add_argument(
        "catalog_source", 
        help="Path to catalog JSON file or API config"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print(f"🔄 Product Catalog Ingestion")
    print(f"Brand: {args.brand_domain}")
    print(f"Source: {args.catalog_source}")
    print("-" * 50)
    
    update_brand_filters(args.brand_domain, args.catalog_source)


if __name__ == "__main__":
    main()