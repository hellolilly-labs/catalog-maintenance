#!/usr/bin/env python3
"""
Manual runner for Brand Intelligence Pipeline

Usage:
    python run/brand_research.py specialized.com
    python run/brand_research.py specialized.com --phase foundation
    python run/brand_research.py specialized.com --auto-continue
    python run/brand_research.py specialized.com --status
"""

import sys
import os

# Add parent directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'packages'))

from liddy_intelligence.research.brand_intelligence_pipeline import main

if __name__ == "__main__":
    # Pass through all arguments except the script name
    sys.argv[0] = "brand_intelligence_pipeline.py"
    import asyncio
    asyncio.run(main())