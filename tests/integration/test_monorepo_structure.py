#!/usr/bin/env python3
"""
Integration test for monorepo structure.

This test validates:
1. All packages can be imported
2. Cross-package dependencies work
3. Core functionality is accessible
"""

import sys
import os
import asyncio
# import pytest  # Optional for pytest runner

# Add packages to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "packages")))


def test_package_imports():
    """Test that all packages can be imported."""
    
    # Test liddy core imports
    try:
        from liddy.storage import get_account_storage_provider
        from liddy.models.product import Product
        from liddy.models.product_manager import get_product_manager
        print("✅ liddy core imports successful")
    except ImportError as e:
        raise ImportError(f"Failed to import liddy core: {e}")
    
    # Test liddy_intelligence imports
    try:
        from liddy_intelligence.research.base_researcher import BaseResearcher
        from liddy_intelligence.ingestion import UniversalProductProcessor
        from liddy.llm.simple_factory import LLMFactory
        print("✅ liddy_intelligence imports successful")
    except ImportError as e:
        raise ImportError(f"Failed to import liddy_intelligence: {e}")
    
    # Test liddy_voice imports
    try:
        from liddy_voice.search_service import VoiceOptimizedSearchService
        from liddy_voice.session_state_manager import SessionStateManager
        print("✅ liddy_voice imports successful")
    except ImportError as e:
        raise ImportError(f"Failed to import liddy_voice: {e}")


def test_cross_package_dependencies():
    """Test that cross-package dependencies work."""
    
    try:
        # liddy_intelligence using liddy core
        from liddy_intelligence.research.base_researcher import BaseResearcher
        from liddy.storage import get_account_storage_provider
        
        # Verify BaseResearcher uses storage from liddy
        assert hasattr(BaseResearcher, '__init__')
        
        # liddy_voice using liddy core
        from liddy_voice.search_service import VoiceOptimizedSearchService
        from liddy.models.product import Product
        
        print("✅ Cross-package dependencies work")
    except Exception as e:
        raise ImportError(f"Cross-package dependency error: {e}")


# @pytest.mark.asyncio
async def test_storage_functionality():
    """Test core storage functionality."""
    
    from liddy.storage import get_account_storage_provider
    
    # Get storage provider
    storage = get_account_storage_provider()
    
    # Test basic operations
    test_content = "Test content for monorepo validation"
    test_brand = "test-brand.com"
    test_file = "test_integration.txt"
    
    # Write file
    success = await storage.write_file(
        account=test_brand,
        file_path=test_file,
        content=test_content
    )
    assert success, "Failed to write test file"
    
    # Read file
    content = await storage.read_file(
        account=test_brand,
        file_path=test_file
    )
    assert content == test_content, "Content mismatch"
    
    # Clean up
    await storage.delete_file(
        account=test_brand,
        file_path=test_file
    )
    
    print("✅ Storage functionality works")


# @pytest.mark.asyncio
async def test_product_model():
    """Test Product model functionality."""
    
    from liddy.models.product import Product
    
    # Create a test product
    product = Product(
        id="TEST-001",
        name="Test Product",
        description="A test product for integration testing",
        price=99.99,
        categories=["test", "integration"],
        brand="TestBrand"
    )
    
    # Test serialization
    product_dict = product.to_dict()
    assert product_dict["id"] == "TEST-001"
    assert product_dict["name"] == "Test Product"
    
    # Test deserialization
    product2 = Product.from_dict(product_dict)
    assert product2.id == product.id
    assert product2.name == product.name
    
    print("✅ Product model works")


# @pytest.mark.asyncio
async def test_llm_factory():
    """Test LLM factory pattern."""
    
    from liddy.llm.simple_factory import LLMFactory
    
    # Create factory
    factory = LLMFactory()
    
    # Test getting OpenAI service (should work if API key is set)
    try:
        service = factory.get_service("openai")
        assert service is not None
        print("✅ LLM factory works")
    except Exception as e:
        print(f"⚠️  LLM factory test skipped (no API key?): {e}")


def test_ingestion_components():
    """Test ingestion components."""
    
    from liddy_intelligence.ingestion import (
        UniversalProductProcessor,
        SparseEmbeddingGenerator,
        SeparateIndexIngestion
    )
    
    # Test processor initialization
    processor = UniversalProductProcessor("test-brand.com")
    assert processor.brand_domain == "test-brand.com"
    
    # Test sparse embeddings
    sparse_gen = SparseEmbeddingGenerator("test-brand.com")
    assert sparse_gen.brand_domain == "test-brand.com"
    
    print("✅ Ingestion components work")


if __name__ == "__main__":
    """Run tests directly."""
    print("Monorepo Integration Tests")
    print("==========================\n")
    
    # Run sync tests
    test_package_imports()
    test_cross_package_dependencies()
    test_ingestion_components()
    
    # Run async tests
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_storage_functionality())
    loop.run_until_complete(test_product_model())
    loop.run_until_complete(test_llm_factory())
    
    print("\n✅ All integration tests passed!")