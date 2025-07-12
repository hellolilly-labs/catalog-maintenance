#!/usr/bin/env python3
"""
Brand Restart Utility

🚨 DANGER: This script permanently deletes ALL brand data!

Usage:
    python scripts/brand_restart.py specialized.com
    python scripts/brand_restart.py specialized.com --force  # Skip confirmations (DANGEROUS)
    python scripts/brand_restart.py specialized.com --inspect  # Just show what would be deleted

This script will permanently delete:
- Account configuration
- Product catalogs  
- All backups
- Cached vertical detection data
- Any generated descriptors/sizing data

⚠️  THIS OPERATION IS IRREVERSIBLE ⚠️
"""

import sys
import os
import asyncio
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from liddy.storage import get_brand_data_manager


async def inspect_brand_data(brand_domain: str):
    """Show what data exists for a brand without deleting anything"""
    print(f"\n🔍 Inspecting brand data for '{brand_domain}'...")
    
    manager = get_brand_data_manager()
    summary = await manager.list_brand_data(brand_domain)
    
    if "error" in summary:
        print(f"❌ Error inspecting data: {summary['error']}")
        return
    
    print(f"\n📊 Brand Data Summary for '{brand_domain}':")
    print("=" * 50)
    
    # Account config
    if summary["account_config"]:
        config = summary["account_config"]
        print(f"✅ Account Configuration:")
        print(f"   • Settings count: {config['settings_count']}")
        print(f"   • Last updated: {config['last_updated']}")
    else:
        print("❌ No account configuration found")
    
    # Product catalog
    if summary["product_catalog"]:
        catalog = summary["product_catalog"]
        print(f"\n✅ Product Catalog:")
        if catalog.get("product_count"):
            print(f"   • Products: {catalog['product_count']}")
        print(f"   • Size: {catalog.get('size', 0):,} bytes")
        print(f"   • Compressed: {catalog.get('compressed', False)}")
        print(f"   • Last updated: {catalog.get('last_updated', 'unknown')}")
    else:
        print("\n❌ No product catalog found")
    
    # Backups
    if summary["backups"]:
        print(f"\n✅ Backup Files ({len(summary['backups'])}):")
        total_backup_size = sum(b["size"] for b in summary["backups"])
        print(f"   • Total backup size: {total_backup_size:,} bytes")
        for backup in summary["backups"][:5]:  # Show first 5
            print(f"   • {backup['filename']} ({backup['size']:,} bytes)")
        if len(summary["backups"]) > 5:
            print(f"   • ... and {len(summary['backups']) - 5} more backup files")
    else:
        print("\n❌ No backup files found")
    
    # Total size
    total_size = summary.get("total_size_estimate", 0)
    if total_size > 0:
        print(f"\n📦 Total estimated size: {total_size:,} bytes ({total_size / 1024 / 1024:.2f} MB)")
    
    print(f"\n💡 To delete this data, run:")
    print(f"   python scripts/brand_restart.py {brand_domain}")


async def restart_brand_data(brand_domain: str, force: bool = False):
    """Restart brand data with safety checks"""
    manager = get_brand_data_manager()
    
    success = await manager.restart_brand(brand_domain, force=force)
    
    if success:
        print(f"\n🎉 Brand '{brand_domain}' successfully restarted!")
        print("   You can now begin fresh data collection and processing.")
        sys.exit(0)
    else:
        print(f"\n❌ Brand restart failed or was cancelled.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="🚨 DANGER: Permanently delete ALL brand data and restart fresh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/brand_restart.py specialized.com
  python scripts/brand_restart.py specialized.com --inspect
  python scripts/brand_restart.py specialized.com --force

⚠️  WARNING: This operation is IRREVERSIBLE! ⚠️
        """
    )
    
    parser.add_argument(
        "brand_domain",
        help="Brand domain to restart (e.g., specialized.com)"
    )
    
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Just show what data exists without deleting anything"
    )
    
    parser.add_argument(
        "--force",
        action="store_true", 
        help="Skip safety confirmations (EXTREMELY DANGEROUS - USE WITH CAUTION)"
    )
    
    args = parser.parse_args()
    
    # Validate brand domain
    if not args.brand_domain or "." not in args.brand_domain:
        print("❌ Error: Please provide a valid brand domain (e.g., specialized.com)")
        sys.exit(1)
    
    # Show warning for force mode
    if args.force and not args.inspect:
        print("🚨 WARNING: --force flag enabled - all safety checks will be skipped!")
        print("This will immediately delete all data without confirmation!")
    
    # Run the appropriate operation
    if args.inspect:
        asyncio.run(inspect_brand_data(args.brand_domain))
    else:
        asyncio.run(restart_brand_data(args.brand_domain, force=args.force))


if __name__ == "__main__":
    main() 