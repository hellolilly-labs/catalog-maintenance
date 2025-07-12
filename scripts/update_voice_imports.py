#!/usr/bin/env python3
"""
Update imports in voice assistant files to use the new package structure.
"""

import os
import re


def update_imports_in_file(file_path):
    """Update imports in a single file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Update imports from spence.* to liddy_voice.spence.*
    # But skip relative imports and external package imports
    import_patterns = [
        (r'from spence\.', 'from liddy_voice.'),
        (r'import spence\.', 'import liddy_voice.'),
        # Update the rag import to use unified version
        (r'from liddy_voice\.spence\.rag import', 'from liddy_voice.rag_unified import'),
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


def update_all_voice_imports():
    """Update imports in all voice assistant files."""
    voice_dir = "packages/liddy_voice"
    
    updated_count = 0
    for root, dirs, files in os.walk(voice_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if update_imports_in_file(file_path):
                    updated_count += 1
    
    print(f"\nTotal files updated: {updated_count}")


if __name__ == "__main__":
    update_all_voice_imports()