#!/usr/bin/env python3
"""Fix all incorrect web_search imports."""

import os
import re
from pathlib import Path

# Mapping of incorrect to correct imports
IMPORT_FIXES = {
    r'from liddy_intelligence\.web_search import': 'from liddy_intelligence.research.data_sources.web_search import',
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
    fix_all_imports()