#!/usr/bin/env python3
"""
Manual runner for Voice Search Testing

Usage:
    python run/voice_search.py flexfits.com
    python run/voice_search.py flexfits.com --setup-baseline
    python run/voice_search.py specialized.com
"""

import sys
import os

# Add parent directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'packages'))

from liddy_intelligence.voice_testing.voice_search_comparison import main

if __name__ == "__main__":
    # Pass through all arguments except the script name
    sys.argv[0] = "voice_search_comparison.py"
    import asyncio
    asyncio.run(main())