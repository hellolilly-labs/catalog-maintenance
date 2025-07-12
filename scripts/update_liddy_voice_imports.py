#!/usr/bin/env python3
"""
Update imports in liddy_voice to remove spence nesting.
"""

import os
import re


def update_imports_in_file(file_path):
    """Update imports in a single file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Update imports from liddy_voice.* to liddy_voice.*
    import_patterns = [
        (r'from liddy_voice\.spence\.', 'from liddy_voice.'),
        (r'import liddy_voice\.spence\.', 'import liddy_voice.'),
    ]
    
    for pattern, replacement in import_patterns:
        content = re.sub(pattern, replacement, content)
    
    # Write back if changed
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Updated imports in: {file_path}")
        return True
    return False


def update_all_imports():
    """Update imports in all Python files."""
    directories = [
        "packages/liddy_voice",
        "packages/liddy_intelligence", 
        "run",
        "scripts"
    ]
    
    updated_count = 0
    for directory in directories:
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        if update_imports_in_file(file_path):
                            updated_count += 1
    
    print(f"\nTotal files updated: {updated_count}")


if __name__ == "__main__":
    update_all_imports()