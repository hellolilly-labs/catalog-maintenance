#!/usr/bin/env python3
"""
Consolidate duplicate classes between liddy_voice and liddy core.
This script updates imports to use shared classes from liddy core.
"""

import os
import re


def update_imports_in_file(file_path):
    """Update imports in a single file to use liddy core classes."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Map of imports to replace
    import_replacements = [
        # Storage consolidation
        (r'from liddy_voice\.storage import AccountStorageProvider',
         'from liddy.storage import AccountStorageProvider'),
        (r'from liddy_voice\.storage import GCPAccountStorageProvider',
         'from liddy.storage import GCPAccountStorageProvider'),
        (r'from liddy_voice\.storage import LocalAccountStorageProvider',
         'from liddy.storage import LocalAccountStorageProvider'),
        (r'from liddy_voice\.storage import get_account_storage_provider',
         'from liddy.storage import get_account_storage_provider'),
        (r'import liddy_voice\.storage',
         'import liddy.storage'),
        
        # Product consolidation
        (r'from liddy_voice\.product import Product',
         'from liddy.models.product import Product'),
        (r'import liddy_voice\.product',
         'import liddy.models.product'),
    ]
    
    for pattern, replacement in import_replacements:
        content = re.sub(pattern, replacement, content)
    
    # Write back if changed
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Updated imports in: {file_path}")
        return True
    return False


def remove_duplicate_files():
    """Remove duplicate files that have been consolidated."""
    files_to_remove = [
        "packages/liddy_voice/storage.py",
        "packages/liddy_voice/product.py",
    ]
    
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed duplicate file: {file_path}")


def main():
    """Main consolidation process."""
    print("Starting consolidation of duplicate classes...")
    print("=" * 60)
    
    # Update imports in all liddy_voice files
    voice_dir = "packages/liddy_voice"
    updated_count = 0
    
    for root, dirs, files in os.walk(voice_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if update_imports_in_file(file_path):
                    updated_count += 1
    
    # Also update runner scripts that might use these classes
    for runner_file in ["run/voice_main.py", "run/test_voice_search.py"]:
        if os.path.exists(runner_file):
            if update_imports_in_file(runner_file):
                updated_count += 1
    
    print(f"\nTotal files updated: {updated_count}")
    
    # Remove duplicate files
    print("\nRemoving duplicate files...")
    remove_duplicate_files()
    
    print("\n✅ Consolidation complete!")
    print("\nNext steps:")
    print("1. Review the changes")
    print("2. Run tests to ensure everything still works")
    print("3. Commit the consolidation")


if __name__ == "__main__":
    main()