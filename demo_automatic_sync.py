#!/usr/bin/env python3
"""
Demo: Automatic Catalog Synchronization

This demo shows how automatic sync detects and handles catalog changes.
"""

import json
import time
import os
from pathlib import Path
from datetime import datetime
import shutil

from src.sync import CatalogMonitor, SyncOrchestrator


def create_demo_catalog(path: str, num_products: int = 50):
    """Create a demo product catalog."""
    products = []
    
    for i in range(num_products):
        product = {
            "id": f"DEMO-{i:04d}",
            "name": f"Demo Product {i}",
            "brand": "DemoBrand",
            "price": 100 + (i * 10),
            "category": ["electronics", "gadgets"][i % 2],
            "description": f"This is demo product number {i}. Great features and quality.",
            "available": True,
            "lastUpdated": datetime.now().isoformat()
        }
        products.append(product)
    
    with open(path, 'w') as f:
        json.dump({"products": products}, f, indent=2)
    
    return products


def modify_catalog(path: str, modifications: dict):
    """Apply modifications to catalog."""
    with open(path) as f:
        data = json.load(f)
    
    products = data["products"]
    
    # Apply modifications
    if "add" in modifications:
        for new_product in modifications["add"]:
            products.append(new_product)
    
    if "modify" in modifications:
        for product_id, changes in modifications["modify"].items():
            for product in products:
                if product["id"] == product_id:
                    product.update(changes)
                    product["lastUpdated"] = datetime.now().isoformat()
                    break
    
    if "delete" in modifications:
        products = [p for p in products if p["id"] not in modifications["delete"]]
        data["products"] = products
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def run_demo():
    """Run the automatic sync demo."""
    print("🎯 Automatic Catalog Synchronization Demo")
    print("="*60)
    
    # Setup
    brand_domain = "demo.com"
    catalog_path = "demo_catalog.json"
    
    # Clean up any existing state
    brand_path = Path(f"accounts/{brand_domain}")
    if brand_path.exists():
        shutil.rmtree(brand_path)
    
    # Create initial catalog
    print("\n1️⃣ Creating initial catalog with 50 products...")
    create_demo_catalog(catalog_path)
    
    # Initialize monitor
    monitor = CatalogMonitor(brand_domain, catalog_path)
    
    # Initial check (establishes baseline)
    print("\n2️⃣ Establishing baseline...")
    monitor.check_for_changes()
    stats = monitor.get_catalog_stats()
    print(f"✅ Baseline established: {stats['total_products']} products")
    
    # Simulate catalog changes
    print("\n3️⃣ Simulating catalog changes...")
    time.sleep(1)  # Ensure different timestamp
    
    modifications = {
        "add": [
            {
                "id": "DEMO-NEW-001",
                "name": "Brand New Product",
                "brand": "DemoBrand",
                "price": 599.99,
                "category": "premium",
                "description": "Just launched! Latest and greatest.",
                "available": True,
                "featured": True
            }
        ],
        "modify": {
            "DEMO-0010": {"price": 299.99, "sale": True},
            "DEMO-0020": {"available": False, "reason": "Out of stock"}
        },
        "delete": ["DEMO-0045", "DEMO-0046"]
    }
    
    modify_catalog(catalog_path, modifications)
    print("✅ Applied changes: 1 addition, 2 modifications, 2 deletions")
    
    # Check for changes
    print("\n4️⃣ Detecting changes...")
    changes = monitor.check_for_changes()
    
    print(f"\n📊 Change Summary:")
    change_summary = {}
    for change in changes:
        change_type = change.change_type
        change_summary[change_type] = change_summary.get(change_type, 0) + 1
        
        print(f"  • {change.change_type.upper()}: {change.product_id}")
        if change.fields_changed:
            print(f"    Fields: {', '.join(change.fields_changed)}")
    
    # Test sync triggering logic
    print(f"\n5️⃣ Testing sync trigger logic...")
    print(f"  Pending changes: {len(monitor.pending_changes)}")
    print(f"  Should trigger sync: {monitor.should_trigger_sync()}")
    
    # Demonstrate orchestrator (without actual Pinecone sync)
    print("\n6️⃣ Sync Orchestrator Demo (dry run)...")
    
    # Show what would be synced
    batch = monitor.get_sync_batch()
    print(f"\n📦 Sync Batch:")
    print(f"  Products to add: {len(batch['add'])}")
    print(f"  Products to update: {len(batch['update'])}")
    print(f"  Products to delete: {len(batch['delete'])}")
    
    # Simulate continuous monitoring
    print("\n7️⃣ Simulating continuous monitoring...")
    print("Making incremental changes over time...\n")
    
    for i in range(3):
        time.sleep(2)
        
        # Make a small change
        small_mod = {
            "modify": {
                f"DEMO-{i:04d}": {"price": 150 + (i * 20), "updated": True}
            }
        }
        modify_catalog(catalog_path, small_mod)
        
        # Check for changes
        new_changes = monitor.check_for_changes()
        if new_changes:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Detected {len(new_changes)} change(s)")
            
            # Check if sync should trigger
            if monitor.should_trigger_sync():
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 Sync triggered!")
                batch = monitor.get_sync_batch()
                print(f"                    Syncing {sum(len(v) for v in batch.values())} products")
                monitor.mark_sync_complete()
            else:
                print(f"                    Waiting for more changes or timeout...")
    
    # Final statistics
    print("\n8️⃣ Final Statistics:")
    final_stats = monitor.get_catalog_stats()
    print(f"  Total products: {final_stats['total_products']}")
    print(f"  Total checks: {final_stats.get('check_count', 'N/A')}")
    print(f"  Pending changes: {final_stats['pending_changes']}")
    
    # Show change log
    change_log_path = Path(f"accounts/{brand_domain}/catalog_changes.jsonl")
    if change_log_path.exists():
        print(f"\n📜 Change Log ({change_log_path}):")
        with open(change_log_path) as f:
            for i, line in enumerate(f):
                if i < 5:  # Show first 5 entries
                    change = json.loads(line)
                    print(f"  {change['timestamp']}: {change['change_type']} - {change['product_id']}")
        
        total_lines = sum(1 for _ in open(change_log_path))
        if total_lines > 5:
            print(f"  ... and {total_lines - 5} more entries")
    
    # Cleanup
    print("\n9️⃣ Cleanup...")
    os.remove(catalog_path)
    print("✅ Demo complete!")
    
    print("\n💡 Key Takeaways:")
    print("  • Changes are detected using content hashing")
    print("  • Field-level tracking identifies what changed")
    print("  • Sync triggers based on count or time thresholds")
    print("  • All changes are logged for audit trail")
    print("  • State is persisted between runs")


if __name__ == "__main__":
    run_demo()