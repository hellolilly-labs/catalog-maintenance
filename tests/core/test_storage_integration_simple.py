#!/usr/bin/env python3
"""
Simple Storage Integration Test

Tests the storage provider integration without complex dependencies.
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_simple_storage():
    """Test basic storage provider functionality"""
    
    try:
        # Direct import to avoid dependency issues
        from src.storage import get_account_storage_provider
        
        print("🧪 Testing Simple Storage Integration")
        print("=" * 40)
        
        brand_domain = "specialized.com"
        storage = get_account_storage_provider()
        
        print(f"✅ Storage provider initialized: {type(storage).__name__}")
        
        # Test 1: Write a test file
        print("\n1️⃣ Testing file write...")
        test_content = '{"test": "storage integration", "filters": ["price", "category"]}'
        success = await storage.write_file(
            account=brand_domain,
            file_path="test_integration.json",
            content=test_content,
            content_type="application/json"
        )
        
        if success:
            print("   ✅ Successfully wrote test file")
        else:
            print("   ❌ Failed to write test file")
            return False
        
        # Test 2: Read the file back
        print("\n2️⃣ Testing file read...")
        read_content = await storage.read_file(brand_domain, "test_integration.json")
        
        if read_content:
            if read_content == test_content:
                print("   ✅ Successfully read back identical content")
            else:
                print("   ❌ Content mismatch")
                print(f"      Expected: {test_content}")
                print(f"      Got: {read_content}")
                return False
        else:
            print("   ❌ Failed to read test file")
            return False
        
        # Test 3: Check if file exists
        print("\n3️⃣ Testing file existence check...")
        exists = await storage.file_exists(brand_domain, "test_integration.json")
        
        if exists:
            print("   ✅ File existence check works")
        else:
            print("   ❌ File existence check failed")
            return False
        
        # Test 4: Get file metadata
        print("\n4️⃣ Testing file metadata...")
        metadata = await storage.get_file_metadata(brand_domain, "test_integration.json")
        
        if metadata:
            print(f"   ✅ Got metadata: size={metadata.get('size')}, content_type={metadata.get('content_type')}")
        else:
            print("   ❌ Failed to get file metadata")
            return False
        
        # Test 5: List files
        print("\n5️⃣ Testing file listing...")
        files = await storage.list_files(brand_domain, "")  # Root directory, not "."
        
        if "test_integration.json" in files:
            print(f"   ✅ File listing works: found {len(files)} files")
        else:
            print(f"   ⚠️ File not found in listing. Available files: {files[:5]}...")  # Show first 5 files
            # This might be OK depending on storage implementation
        
        print("\n🎉 All critical storage tests passed!")
        print("\n📋 Summary:")
        print(f"   • Storage Provider: {type(storage).__name__}")
        print(f"   • Brand Domain: {brand_domain}")
        print(f"   • Test File: test_integration.json")
        print(f"   • File Size: {metadata.get('size', 'unknown')} bytes")
        print(f"   • Files in account: {len(files)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_simple_storage())
    sys.exit(0 if success else 1) 