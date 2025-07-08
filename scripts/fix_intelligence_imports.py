#!/usr/bin/env python3
"""Fix incorrect imports in intelligence package."""

import os
import re
from pathlib import Path

# Fix incorrect import mappings
IMPORT_FIXES = {
    # Fix agents __init__.py
    r'from liddy_intelligence\.ingestion\.base_agent import': 'from liddy_intelligence.agents.base_agent import',
    r'from liddy_intelligence\.ingestion\.communication_hub import': 'from liddy_intelligence.agents.communication_hub import',
    r'from liddy_intelligence\.ingestion\.context import': 'from liddy_intelligence.agents.context import',
    
    # Fix llm __init__.py
    r'from liddy_intelligence\.ingestion\.base import': 'from liddy_intelligence.llm.base import',
    r'from liddy_intelligence\.ingestion\.errors import': 'from liddy_intelligence.llm.errors import',
    r'from liddy_intelligence\.ingestion\.simple_factory import': 'from liddy.llm.simple_factory import',
    r'from liddy_intelligence\.ingestion\.openai_service import': 'from liddy_intelligence.llm.openai_service import',
    r'from liddy_intelligence\.ingestion\.anthropic_service import': 'from liddy_intelligence.llm.anthropic_service import',
    r'from liddy_intelligence\.ingestion\.gemini_service import': 'from liddy_intelligence.llm.gemini_service import',
}

def fix_imports_in_file(filepath):
    """Fix imports in a single file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Apply fixes
        for old_import, new_import in IMPORT_FIXES.items():
            content = re.sub(old_import, new_import, content)
        
        if content != original_content:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"✅ Fixed imports in {filepath}")
            return True
        return False
    except Exception as e:
        print(f"❌ Error fixing {filepath}: {e}")
        return False

def fix_all_imports(root_dir='packages/liddy_intelligence'):
    """Fix imports in all Python files."""
    updated_count = 0
    for filepath in Path(root_dir).rglob('*.py'):
        if fix_imports_in_file(filepath):
            updated_count += 1
    
    print(f"\n📊 Fixed {updated_count} files")

if __name__ == "__main__":
    import sys
    root_dir = sys.argv[1] if len(sys.argv) > 1 else 'packages/liddy_intelligence'
    fix_all_imports(root_dir)