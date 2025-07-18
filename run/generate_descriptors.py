#!/usr/bin/env python3
"""
Manual runner for Product Descriptor Generation

Usage:
    python run/generate_descriptors.py specialized.com
    python run/generate_descriptors.py specialized.com --force
    python run/generate_descriptors.py specialized.com --evaluate-models
"""

import sys
import os

# Add parent directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'packages'))

from liddy_intelligence.ingestion.scripts.pre_generate_descriptors import main

if __name__ == "__main__":
    # Pass through all arguments except the script name
    sys.argv[0] = "pre_generate_descriptors.py"
    import asyncio
    asyncio.run(main())