"""
Test Enhanced Brand Vertical Detection with Persistent Storage

This script tests the enhanced brand vertical detection system that now:
1. Saves results to GCP storage (or local storage as fallback)
2. Loads cached results from storage
3. Shows you exactly where files are stored

Usage:
    export STORAGE_PROVIDER=gcp
    export OPENAI_API_KEY='your-key'
    python test_storage_brand_vertical.py
"""

import asyncio
import json
import logging
import os
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_brand_vertical_with_storage():
    """Test enhanced brand vertical detection with persistent storage"""
    
    print("🏗️  TESTING ENHANCED BRAND VERTICAL DETECTION WITH STORAGE")
    print("=" * 58)
    print("✨ New Features:")
    print("  • Saves results to GCP/local storage")
    print("  • Loads cached results from storage")
    print("  • Metadata tracking and versioning")
    print()
    
    try:
        # Check storage configuration
        from src.storage import get_account_storage_provider
        from src.descriptor import BrandVerticalDetector
        
        storage = get_account_storage_provider()
        print(f"📦 Storage Provider: {type(storage).__name__}")
        
        if hasattr(storage, 'bucket'):
            print(f"🪣 GCP Bucket: {storage.bucket.name}")
            print(f"📁 Storage Path: accounts/{{brand_domain}}/brand_vertical.json")
        else:
            print(f"📁 Local Storage Dir: {storage.base_dir}")
            print(f"📄 Storage Path: {storage.base_dir}/accounts/{{brand_domain}}/brand_vertical.json")
        print()
        
        # Test with specialized.com
        brand_domain = "specialized.com"
        detector = BrandVerticalDetector()
        
        print(f"🚴 Testing Brand: {brand_domain}")
        print("⏱️  Starting enhanced brand vertical detection...")
        
        start_time = datetime.now()
        result = await detector.detect_brand_vertical(brand_domain)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"✅ Analysis completed in {duration:.1f} seconds")
        print()
        
        # Show results
        print("🎯 DETECTION RESULTS:")
        print("-" * 20)
        print(f"Brand: {result['brand_domain']}")
        print(f"Detected Vertical: {result['detected_vertical']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Analysis Methods: {result['analysis_methods']}")
        print(f"Duration: {result.get('_duration', 'Unknown'):.1f}s")
        print()
        
        # Show where files are stored
        print("💾 STORAGE LOCATIONS:")
        print("-" * 19)
        
        if hasattr(storage, 'bucket'):
            # GCP Storage
            main_path = f"accounts/{brand_domain}/brand_vertical.json"
            metadata_path = f"accounts/{brand_domain}/brand_vertical_metadata.json"
            
            print(f"🌐 GCP Bucket: gs://{storage.bucket.name}/")
            print(f"📄 Main Results: gs://{storage.bucket.name}/{main_path}")
            print(f"📊 Metadata: gs://{storage.bucket.name}/{metadata_path}")
            
            # Check if files exist
            main_blob = storage.bucket.blob(main_path)
            metadata_blob = storage.bucket.blob(metadata_path)
            
            print()
            print("📁 File Status:")
            if main_blob.exists():
                main_blob.reload()
                print(f"  ✅ Main results: {main_blob.size} bytes, updated {main_blob.updated}")
            else:
                print(f"  ❌ Main results: Not found")
            
            if metadata_blob.exists():
                metadata_blob.reload()
                print(f"  ✅ Metadata: {metadata_blob.size} bytes, updated {metadata_blob.updated}")
            else:
                print(f"  ❌ Metadata: Not found")
            
        else:
            # Local Storage
            import os
            main_path = os.path.join(storage.base_dir, "accounts", brand_domain, "brand_vertical.json")
            metadata_path = os.path.join(storage.base_dir, "accounts", brand_domain, "brand_vertical_metadata.json")
            
            print(f"💻 Local Storage:")
            print(f"📄 Main Results: {main_path}")
            print(f"📊 Metadata: {metadata_path}")
            
            print()
            print("📁 File Status:")
            if os.path.exists(main_path):
                stat = os.stat(main_path)
                print(f"  ✅ Main results: {stat.st_size} bytes, modified {datetime.fromtimestamp(stat.st_mtime)}")
            else:
                print(f"  ❌ Main results: Not found")
            
            if os.path.exists(metadata_path):
                stat = os.stat(metadata_path)
                print(f"  ✅ Metadata: {stat.st_size} bytes, modified {datetime.fromtimestamp(stat.st_mtime)}")
            else:
                print(f"  ❌ Metadata: Not found")
        
        print()
        
        # Test caching behavior
        print("🔄 TESTING CACHE BEHAVIOR:")
        print("-" * 25)
        
        print("🔍 Running analysis again (should use cached results)...")
        cache_start = datetime.now()
        cached_result = await detector.detect_brand_vertical(brand_domain)
        cache_end = datetime.now()
        cache_duration = (cache_end - cache_start).total_seconds()
        
        print(f"⚡ Cache retrieval completed in {cache_duration:.1f} seconds")
        
        if cache_duration < 1.0:
            print("✅ SUCCESS: Results loaded from cache (fast retrieval)")
        else:
            print("⚠️  Results may have been recomputed (slow retrieval)")
        
        # Verify results are identical
        if cached_result['detected_vertical'] == result['detected_vertical']:
            print("✅ SUCCESS: Cached results match original analysis")
        else:
            print("❌ ERROR: Cached results differ from original")
        
        print()
        
        # Show file contents preview
        print("📖 FILE CONTENTS PREVIEW:")
        print("-" * 26)
        
        try:
            if hasattr(storage, 'bucket'):
                # GCP Storage
                metadata_blob = storage.bucket.blob(f"accounts/{brand_domain}/brand_vertical_metadata.json")
                if metadata_blob.exists():
                    metadata_content = json.loads(metadata_blob.download_as_text())
                    print("📊 Metadata Summary:")
                    for key, value in metadata_content.items():
                        print(f"  {key}: {value}")
            else:
                # Local Storage
                metadata_path = os.path.join(storage.base_dir, "accounts", brand_domain, "brand_vertical_metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata_content = json.load(f)
                    print("📊 Metadata Summary:")
                    for key, value in metadata_content.items():
                        print(f"  {key}: {value}")
                        
        except Exception as e:
            print(f"⚠️  Could not load metadata: {e}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during enhanced storage test: {e}")
        logger.exception("Full error details:")
        return None

async def show_storage_instructions():
    """Show instructions for accessing stored files"""
    
    print("\n" + "=" * 60)
    print("📚 HOW TO ACCESS YOUR STORED BRAND ANALYSIS FILES")
    print("=" * 60)
    
    print("\n🔑 First, authenticate with GCP:")
    print("   gcloud auth application-default login")
    
    print("\n🌐 To view files in GCP Console:")
    print("   https://console.cloud.google.com/storage/browser/liddy-account-documents-dev")
    
    print("\n💻 To list files via command line:")
    print("   gsutil ls gs://liddy-account-documents-dev/accounts/")
    print("   gsutil ls gs://liddy-account-documents-dev/accounts/specialized.com/")
    
    print("\n📄 To download and view specific files:")
    print("   gsutil cp gs://liddy-account-documents-dev/accounts/specialized.com/brand_vertical.json .")
    print("   gsutil cp gs://liddy-account-documents-dev/accounts/specialized.com/brand_vertical_metadata.json .")
    print("   cat brand_vertical_metadata.json | jq '.'")
    
    print("\n🔍 File Structure:")
    print("   accounts/")
    print("   └── specialized.com/")
    print("       ├── brand_vertical.json           # Complete analysis results")
    print("       ├── brand_vertical_metadata.json  # Analysis metadata")
    print("       ├── account.json                  # Account configuration")
    print("       └── products.json                 # Product catalog (if available)")

def check_environment():
    """Check environment configuration"""
    
    print("🔧 ENVIRONMENT CHECK")
    print("=" * 19)
    
    # Check required variables
    openai_key = os.getenv('OPENAI_API_KEY')
    storage_provider = os.getenv('STORAGE_PROVIDER', 'local')
    
    print(f"Storage Provider: {storage_provider}")
    
    if openai_key:
        print(f"✅ OPENAI_API_KEY: Set (***{openai_key[-4:]})")
    else:
        print("❌ OPENAI_API_KEY: Not set")
        return False
    
    # Optional keys
    optional_keys = ['TAVILY_API_KEY', 'ANTHROPIC_API_KEY', 'GEMINI_API_KEY']
    for key in optional_keys:
        value = os.getenv(key)
        if value:
            print(f"✅ {key}: Set (***{value[-4:]})")
        else:
            print(f"➖ {key}: Not set")
    
    print()
    return True

async def main():
    """Main test function"""
    
    print("🧪 ENHANCED BRAND VERTICAL DETECTION WITH STORAGE")
    print("=" * 50)
    print("Testing brand analysis with persistent storage capabilities")
    print()
    
    # Check environment
    if not check_environment():
        print("❌ Please set required environment variables and try again.")
        return
    
    # Run the enhanced test
    result = await test_brand_vertical_with_storage()
    
    # Show access instructions
    await show_storage_instructions()
    
    print("\n" + "=" * 60)
    print("🎉 ENHANCED STORAGE TEST COMPLETE!")
    
    if result and result.get('detected_vertical') == 'cycling':
        print("✅ SUCCESS: Specialized.com correctly identified and stored!")
        print(f"📊 Confidence: {result['confidence']:.2f}")
        print(f"⚡ Duration: {result.get('_duration', 0):.1f}s")
    else:
        print("⚠️  Review needed - check results above")

if __name__ == "__main__":
    asyncio.run(main())
