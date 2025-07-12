"""
CLI for Catalog Synchronization

Provides command-line interface for monitoring and syncing catalogs.
"""

import argparse
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

from .sync_orchestrator import SyncOrchestrator
from .catalog_monitor import CatalogMonitor


def monitor_command(args):
    """Handle monitor command."""
    monitor = CatalogMonitor(args.brand_domain, args.catalog_file)
    
    print(f"📊 Monitoring catalog: {args.catalog_file}")
    print(f"Brand: {args.brand_domain}")
    print("-" * 60)
    
    # Check for changes
    changes = monitor.check_for_changes()
    
    if changes:
        print(f"\n✅ Detected {len(changes)} changes:")
        
        # Group by type
        by_type = {'added': [], 'modified': [], 'deleted': []}
        for change in changes:
            by_type[change.change_type].append(change)
        
        for change_type, type_changes in by_type.items():
            if type_changes:
                print(f"\n{change_type.upper()} ({len(type_changes)}):")
                for change in type_changes[:10]:  # Show first 10
                    print(f"  - {change.product_id}")
                    if change.fields_changed:
                        print(f"    Fields: {', '.join(change.fields_changed)}")
                
                if len(type_changes) > 10:
                    print(f"  ... and {len(type_changes) - 10} more")
    else:
        print("\n✅ No changes detected")
    
    # Show stats
    stats = monitor.get_catalog_stats()
    print(f"\n📈 Catalog Statistics:")
    print(f"  Total products: {stats['total_products']}")
    print(f"  Last check: {stats['last_check'] or 'Never'}")
    print(f"  Last sync: {stats['last_sync'] or 'Never'}")
    print(f"  Sync count: {stats['sync_count']}")


def sync_command(args):
    """Handle sync command."""
    orchestrator = SyncOrchestrator(
        brand_domain=args.brand_domain,
        catalog_path=args.catalog_file,
        index_name=args.index,
        namespace=args.namespace
    )
    
    print(f"🔄 Synchronizing catalog to Pinecone")
    print(f"Brand: {args.brand_domain}")
    print(f"Catalog: {args.catalog_file}")
    print(f"Index: {args.index}")
    print(f"Namespace: {args.namespace}")
    print("-" * 60)
    
    # Check current state
    stats = orchestrator.get_sync_stats()
    if stats['catalog_stats']['pending_changes'] > 0:
        print(f"\n📊 Pending changes: {stats['catalog_stats']['pending_changes']}")
        if stats['catalog_stats']['pending_by_type']:
            for change_type, count in stats['catalog_stats']['pending_by_type'].items():
                print(f"  - {change_type}: {count}")
    
    # Perform sync
    print("\n🚀 Starting synchronization...")
    start_time = time.time()
    
    success = orchestrator.sync_changes(force_full_sync=args.force)
    
    duration = time.time() - start_time
    
    if success:
        print(f"\n✅ Sync completed successfully in {duration:.2f}s")
        
        # Show results
        final_stats = orchestrator.get_sync_stats()
        print(f"\n📈 Sync Statistics:")
        print(f"  Total syncs: {final_stats['total_syncs']}")
        print(f"  Successful: {final_stats['successful_syncs']}")
        print(f"  Failed: {final_stats['failed_syncs']}")
        print(f"  Products synced: {final_stats['total_products_synced']}")
    else:
        print(f"\n❌ Sync failed after {duration:.2f}s")
        sys.exit(1)


def watch_command(args):
    """Handle watch command for continuous monitoring."""
    orchestrator = SyncOrchestrator(
        brand_domain=args.brand_domain,
        catalog_path=args.catalog_file,
        index_name=args.index,
        namespace=args.namespace,
        auto_start=False
    )
    
    # Set check interval
    if args.interval:
        orchestrator.check_interval = args.interval
    
    print(f"👁️ Watching catalog for changes")
    print(f"Brand: {args.brand_domain}")
    print(f"Catalog: {args.catalog_file}")
    print(f"Index: {args.index}")
    print(f"Check interval: {orchestrator.check_interval}s")
    print("-" * 60)
    print("Press Ctrl+C to stop\n")
    
    # Define callbacks
    def on_sync_complete(stats):
        print(f"\n✅ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Sync completed")
        print(f"   Products synced: {stats['total_products_synced']}")
    
    def on_sync_error(error):
        print(f"\n❌ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Sync error: {error}")
    
    orchestrator.on_sync_complete = on_sync_complete
    orchestrator.on_sync_error = on_sync_error
    
    # Start monitoring
    orchestrator.start_monitoring()
    
    try:
        while True:
            time.sleep(30)  # Status update every 30s
            
            stats = orchestrator.get_sync_stats()
            pending = stats['catalog_stats']['pending_changes']
            
            if pending > 0:
                print(f"⏳ Pending changes: {pending}", end='\r')
            else:
                print(f"✅ No pending changes", end='\r')
    
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping monitor...")
        orchestrator.stop_monitoring()
        
        # Final stats
        final_stats = orchestrator.get_sync_stats()
        print(f"\n📊 Session Summary:")
        print(f"  Total syncs: {final_stats['total_syncs']}")
        print(f"  Successful: {final_stats['successful_syncs']}")
        print(f"  Failed: {final_stats['failed_syncs']}")
        print(f"  Products synced: {final_stats['total_products_synced']}")


def status_command(args):
    """Handle status command."""
    # Check monitor state
    brand_path = Path(f"accounts/{args.brand_domain}")
    monitor_state_path = brand_path / "catalog_monitor_state.json"
    
    if not monitor_state_path.exists():
        print(f"❌ No monitoring state found for {args.brand_domain}")
        print(f"Run 'monitor' or 'sync' command first to initialize")
        sys.exit(1)
    
    # Load and display state
    with open(monitor_state_path) as f:
        state = json.load(f)
    
    print(f"📊 Sync Status for {args.brand_domain}")
    print("-" * 60)
    
    print(f"\n📅 Timeline:")
    print(f"  Last check: {state.get('last_check', 'Never')}")
    print(f"  Last sync: {state.get('last_sync', 'Never')}")
    print(f"  Sync count: {state.get('sync_count', 0)}")
    
    print(f"\n📦 Catalog:")
    print(f"  Total products: {len(state.get('product_hashes', {}))}")
    
    # Check for change log
    change_log_path = brand_path / "catalog_changes.jsonl"
    if change_log_path.exists():
        # Count recent changes
        recent_changes = []
        with open(change_log_path) as f:
            for line in f:
                try:
                    change = json.loads(line)
                    change_time = datetime.fromisoformat(change['timestamp'])
                    # Changes in last 24 hours
                    if (datetime.now() - change_time).days < 1:
                        recent_changes.append(change)
                except:
                    pass
        
        if recent_changes:
            print(f"\n📈 Recent Changes (last 24h): {len(recent_changes)}")
            by_type = {}
            for change in recent_changes:
                change_type = change['change_type']
                by_type[change_type] = by_type.get(change_type, 0) + 1
            
            for change_type, count in by_type.items():
                print(f"  - {change_type}: {count}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Catalog synchronization for RAG system"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Check catalog for changes')
    monitor_parser.add_argument('brand_domain', help='Brand domain (e.g., specialized.com)')
    monitor_parser.add_argument('catalog_file', help='Path to catalog JSON file')
    
    # Sync command
    sync_parser = subparsers.add_parser('sync', help='Synchronize changes to Pinecone')
    sync_parser.add_argument('brand_domain', help='Brand domain')
    sync_parser.add_argument('catalog_file', help='Path to catalog JSON file')
    sync_parser.add_argument('--index', required=True, help='Pinecone index name')
    sync_parser.add_argument('--namespace', default='products', help='Pinecone namespace')
    sync_parser.add_argument('--force', action='store_true', help='Force full sync')
    
    # Watch command
    watch_parser = subparsers.add_parser('watch', help='Watch catalog and auto-sync')
    watch_parser.add_argument('brand_domain', help='Brand domain')
    watch_parser.add_argument('catalog_file', help='Path to catalog JSON file')
    watch_parser.add_argument('--index', required=True, help='Pinecone index name')
    watch_parser.add_argument('--namespace', default='products', help='Pinecone namespace')
    watch_parser.add_argument('--interval', type=int, help='Check interval in seconds')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show sync status')
    status_parser.add_argument('brand_domain', help='Brand domain')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Check API key for sync/watch
    if args.command in ['sync', 'watch'] and not os.environ.get("PINECONE_API_KEY"):
        print("❌ Error: PINECONE_API_KEY environment variable not set")
        sys.exit(1)
    
    # Execute command
    if args.command == 'monitor':
        monitor_command(args)
    elif args.command == 'sync':
        sync_command(args)
    elif args.command == 'watch':
        watch_command(args)
    elif args.command == 'status':
        status_command(args)


if __name__ == "__main__":
    main()