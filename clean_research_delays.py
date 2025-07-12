#!/usr/bin/env python3
"""
Script to remove artificial time delays from research phase implementations.
"""

import os
import re

def clean_file(filepath):
    """Remove artificial delay patterns from a research file"""
    
    print(f"🔧 Cleaning {filepath}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Remove min_research_time and max_research_time properties
    content = re.sub(r'\s*self\.min_research_time\s*=\s*\d+.*?\n', '', content)
    content = re.sub(r'\s*self\.max_research_time\s*=\s*\d+.*?\n', '', content)
    
    # Remove artificial delay blocks
    # Pattern: "elapsed = time.time() - start_time" through "await asyncio.sleep(...)"
    pattern = r'(\s*# Ensure minimum research time.*?\n)?\s*elapsed = time\.time\(\) - start_time\s*\n\s*if elapsed < self\.min_research_time:\s*\n.*?await asyncio\.sleep\(.*?\)\s*\n'
    content = re.sub(pattern, '', content, flags=re.DOTALL)
    
    # Remove _additional_*_research method calls and definitions
    content = re.sub(r'\s*await self\._additional_.*?_research\(.*?\)\s*\n', '', content)
    
    # Remove entire _additional_*_research method definitions
    pattern = r'\s*async def _additional_.*?_research\(.*?\):.*?(?=\n\s*(?:async )?def|\n\s*class|\Z)'
    content = re.sub(pattern, '', content, flags=re.DOTALL)
    
    # Remove standalone asyncio.sleep calls that aren't meaningful
    content = re.sub(r'\s*await asyncio\.sleep\(min\(.*?\)\).*?\n', '', content)
    content = re.sub(r'\s*await asyncio\.sleep\(additional_time\).*?\n', '', content)
    
    # Clean up extra blank lines
    content = re.sub(r'\n\n\n+', '\n\n', content)
    
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"✅ Cleaned {filepath}")
        return True
    else:
        print(f"ℹ️  No changes needed for {filepath}")
        return False

def main():
    """Clean all research files"""
    
    research_files = [
        'src/research/foundation_research.py',
        'src/research/market_positioning_research.py', 
        'src/research/product_style_research.py',
        'src/research/customer_cultural_research.py',
        'src/research/voice_messaging_research.py',
        'src/research/interview_synthesis_research.py',
        'src/research/linearity_analysis_research.py',
        'src/research/research_integration.py'
    ]
    
    cleaned_count = 0
    
    for filepath in research_files:
        if os.path.exists(filepath):
            if clean_file(filepath):
                cleaned_count += 1
        else:
            print(f"⚠️  File not found: {filepath}")
    
    print(f"\n🎉 Cleaning complete! Modified {cleaned_count} files.")
    print("🚀 All artificial time delays have been removed. Research will now complete when actually finished!")

if __name__ == "__main__":
    main() 