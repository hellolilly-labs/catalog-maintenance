#!/usr/bin/env python3
"""
Manual runner for Product Catalog Ingestion

Usage:
    python run/ingest_catalog.py specialized.com
    python run/ingest_catalog.py specialized.com --index-name custom-index
    python run/ingest_catalog.py specialized.com --force-update
"""

import sys
import os

# Add parent directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'packages'))

from liddy_intelligence.ingestion.scripts.ingest_catalog import main

if __name__ == "__main__":
    # Pass through all arguments except the script name
    sys.argv[0] = "ingest_product_catalog.py"
    import asyncio
    asyncio.run(main())