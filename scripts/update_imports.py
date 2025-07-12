#!/usr/bin/env python3
"""Update imports for monorepo structure."""

import os
import re
from pathlib import Path

# Import mappings
IMPORT_MAPPINGS = {
    # Core models
    r'from src\.models\.product import': 'from liddy.models.product import',
    r'from src\.models\.product_manager import': 'from liddy.models.product_manager import',
    r'from src\.models\.user import': 'from liddy.models.user import',
    r'from src\.models import': 'from liddy.models import',
    
    # Storage
    r'from src\.storage import': 'from liddy.storage import',
    
    # Search
    r'from src\.search\.base import': 'from liddy.search.base import',
    r'from src\.search\.search_pinecone import': 'from liddy.search.pinecone import',
    r'from src\.search\.search_service import': 'from liddy.search.service import',
    r'from src\.search import': 'from liddy.search import',
    
    # Account manager
    r'from src\.account_manager import': 'from liddy.account_manager import',
    
    # Prompt manager
    r'from src\.llm\.prompt_manager import': 'from liddy.prompt_manager import',
    
    # Intelligence-specific
    r'from src\.research\.': 'from liddy_intelligence.research.',
    r'from src\.ingestion\.': 'from liddy_intelligence.ingestion.',
    r'from src\.agents\.': 'from liddy_intelligence.agents.',
    r'from src\.workflow\.': 'from liddy_intelligence.workflow.',
    r'from src\.llm\.': 'from liddy_intelligence.llm.',
    r'from src\.catalog\.': 'from liddy_intelligence.catalog.',
}

def update_imports_in_file(filepath):
    """Update imports in a single file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original_content = content
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

def update_all_imports(root_dir='packages'):
    """Update imports in all Python files."""
    updated_count = 0
    for filepath in Path(root_dir).rglob('*.py'):
        if update_imports_in_file(filepath):
            updated_count += 1
    
    print(f"\n📊 Updated {updated_count} files")

if __name__ == "__main__":
    import sys
    root_dir = sys.argv[1] if len(sys.argv) > 1 else 'packages'
    update_all_imports(root_dir)