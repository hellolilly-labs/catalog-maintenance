#!/usr/bin/env python3
"""
Run Industry Terminology Research

This script runs research to identify industry-specific terminology including:
- Price tier indicators (e.g., "Pro", "Epic", "Sport")
- Industry slang and synonyms
- Technical jargon

Usage:
    python run/research_industry_terminology.py specialized.com
    python run/research_industry_terminology.py specialized.com --force
"""

import os
import sys
import asyncio
import argparse
import logging

# Add parent directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'packages'))

from liddy_intelligence.research.industry_terminology_researcher import IndustryTerminologyResearcher

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_terminology_research(account: str, force_refresh: bool = False):
    """Run industry terminology research"""
    logger.info(f"Starting industry terminology research for {account}")
    
    try:
        researcher = IndustryTerminologyResearcher(
            account=account,
            force_refresh=force_refresh
        )
        
        # Run the research
        research_content = await researcher.conduct_research()
        
        logger.info(f"Research completed successfully for {account}")
        
        # Display summary
        if research_content:
            lines = research_content.split('\n')
            print("\n" + "="*60)
            print("RESEARCH SUMMARY")
            print("="*60)
            # Print first 50 lines as summary
            for line in lines[:50]:
                print(line)
            if len(lines) > 50:
                print(f"\n... (truncated, full research saved to storage)")
            print("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Research failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Research industry-specific terminology')
    parser.add_argument('account', help='Account/brand domain (e.g., specialized.com)')
    parser.add_argument('--force', action='store_true', help='Force refresh even if recent research exists')
    
    args = parser.parse_args()
    
    # Run the research
    success = asyncio.run(run_terminology_research(args.account, args.force))
    
    if success:
        print(f"\n✅ Industry terminology research completed for {args.account}")
        print(f"📍 Research saved to: accounts/{args.account}/research/industry_terminology/")
    else:
        print(f"\n❌ Research failed for {args.account}")
        sys.exit(1)


if __name__ == "__main__":
    main()