#!/usr/bin/env python3
"""
Catalog Sync CLI

Main entry point for catalog synchronization commands.

Usage:
    # Check for changes
    python catalog_sync.py monitor specialized.com data/products.json
    
    # Sync changes to Pinecone
    python catalog_sync.py sync specialized.com data/products.json --index specialized-hybrid-v2
    
    # Watch and auto-sync
    python catalog_sync.py watch specialized.com data/products.json --index specialized-hybrid-v2
    
    # Check status
    python catalog_sync.py status specialized.com
"""

from src.sync.sync_cli import main

if __name__ == "__main__":
    main()