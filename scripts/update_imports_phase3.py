#!/usr/bin/env python3
"""Update imports for Phase 3 - Intelligence package modules."""

import os
import re
from pathlib import Path

# Phase 3 specific import mappings
IMPORT_MAPPINGS = {
    # Within intelligence package - relative imports to absolute
    r'from \.\.agents\.': 'from liddy_intelligence.agents.',
    r'from \.\.catalog\.': 'from liddy_intelligence.catalog.',
    r'from \.\.llm\.': 'from liddy_intelligence.llm.',
    r'from \.\.research\.': 'from liddy_intelligence.research.',
    r'from \.\.workflow\.': 'from liddy_intelligence.workflow.',
    r'from \.\.ingestion\.': 'from liddy_intelligence.ingestion.',
    
    # Single dot relative imports
    r'from \.': 'from liddy_intelligence.ingestion.',  # Will need context-aware updates
    
    # Cross-package imports
    r'from src\.progress_tracker import': 'from liddy_intelligence.progress_tracker import',
    r'from src\.research\.quality\.': 'from liddy_intelligence.research.quality.',
    
    # Remaining src imports should point to intelligence
    r'from src\.': 'from liddy_intelligence.',
}

def get_module_path(filepath):
    """Get the module path for proper import replacement."""
    parts = Path(filepath).parts
    if 'ingestion' in parts:
        return 'liddy_intelligence.ingestion'
    elif 'research' in parts:
        return 'liddy_intelligence.research'
    elif 'agents' in parts:
        return 'liddy_intelligence.agents'
    elif 'workflow' in parts:
        return 'liddy_intelligence.workflow'
    elif 'llm' in parts:
        return 'liddy_intelligence.llm'
    elif 'catalog' in parts:
        return 'liddy_intelligence.catalog'
    return 'liddy_intelligence'

def update_imports_in_file(filepath):
    """Update imports in a single file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Handle context-aware single dot imports
        module_path = get_module_path(filepath)
        content = re.sub(r'from \. import', f'from {module_path} import', content)
        
        # Apply standard mappings
        for old_import, new_import in IMPORT_MAPPINGS.items():
            content = re.sub(old_import, new_import, content)
        
        if content != original_content:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"✅ Updated imports in {filepath}")
            return True
        return False
    except Exception as e:
        print(f"❌ Error updating {filepath}: {e}")
        return False

def update_all_imports(root_dir='packages/liddy_intelligence'):
    """Update imports in all Python files."""
    updated_count = 0
    for filepath in Path(root_dir).rglob('*.py'):
        if update_imports_in_file(filepath):
            updated_count += 1
    
    print(f"\n📊 Updated {updated_count} files")

if __name__ == "__main__":
    import sys
    root_dir = sys.argv[1] if len(sys.argv) > 1 else 'packages/liddy_intelligence'
    update_all_imports(root_dir)