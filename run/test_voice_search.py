#!/usr/bin/env python3
"""
Test voice search functionality with the new unified search service.

This script tests:
1. Legacy index compatibility (<brand>-llama-2048)
2. New unified search service
3. Backward compatibility with existing voice assistant code
"""

import asyncio
import logging
import sys
import os

# Add packages to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "packages")))

from liddy_voice.rag_unified import PineconeRAG
from liddy_voice.search_service import SearchService
from liddy_voice.model import UserState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_legacy_index_compatibility():
    """Test that we can connect to legacy indexes."""
    print("\n=== Testing Legacy Index Compatibility ===")
    
    brands = [
        ("specialized.com", "specialized-llama-2048"),
        ("gucci.com", "gucci-llama-2048"),
        ("balenciaga.com", "balenciaga-llama-2048"),
    ]
    
    for account, expected_index in brands:
        print(f"\nTesting {account}...")
        try:
            # Create RAG instance
            rag = PineconeRAG(
                account=account,
                index_name=expected_index,
                namespace=account.split('.')[0]
            )
            
            # Wait for initialization
            await asyncio.sleep(2)
            
            # Test search
            results = await rag.search(
                query="test query",
                top_k=5,
                top_n=3
            )
            
            print(f"✅ Successfully connected to {expected_index}")
            print(f"   Found {len(results)} results")
            
        except Exception as e:
            print(f"❌ Failed to connect to {expected_index}: {e}")


async def test_unified_search():
    """Test the unified search service."""
    print("\n=== Testing Unified Search Service ===")
    
    account = "specialized.com"
    user_state = UserState(account=account, user_id="test_user")
    
    # Test product search
    print(f"\nTesting product search for {account}...")
    try:
        results = await SearchService.search_products_rag(
            query="mountain bike",
            account=account,
            top_k=10,
            top_n=5
        )
        
        print(f"✅ Product search successful")
        print(f"   Found {len(results)} products")
        
        # Display first result if available
        if results:
            first = results[0]
            metadata = first.get('metadata', {})
            print(f"\n   First result:")
            print(f"   - Score: {first.get('score', 0):.3f}")
            print(f"   - Name: {metadata.get('name', 'Unknown')}")
            print(f"   - ID: {metadata.get('product_id', metadata.get('id', 'Unknown'))}")
            
    except Exception as e:
        print(f"❌ Product search failed: {e}")
        import traceback
        traceback.print_exc()


async def test_query_enhancement():
    """Test query enhancement functionality."""
    print("\n=== Testing Query Enhancement ===")
    
    from livekit.agents import llm
    
    # Create test context
    ctx = llm.ChatContext([])
    ctx.add_message(role="user", content="I need a bike for trails")
    ctx.add_message(role="assistant", content="What type of trails will you be riding?")
    ctx.add_message(role="user", content="Mostly technical downhill")
    
    user_state = UserState(account="specialized.com", user_id="test_user")
    
    try:
        enhanced = await SearchService.enhance_product_query(
            query="mountain bike",
            user_state=user_state,
            chat_ctx=ctx,
            product_knowledge="Specialized makes Stumpjumper, Demo, and Enduro for mountain biking"
        )
        
        print(f"✅ Query enhancement successful")
        print(f"   Original: 'mountain bike'")
        print(f"   Enhanced: '{enhanced}'")
        
    except Exception as e:
        print(f"❌ Query enhancement failed: {e}")


async def main():
    """Run all tests."""
    print("Voice Search Integration Tests")
    print("==============================")
    
    # Check for required environment variables
    if not os.getenv('PINECONE_API_KEY'):
        print("❌ Error: PINECONE_API_KEY environment variable not set")
        return
        
    # Run tests
    await test_legacy_index_compatibility()
    await test_unified_search()
    await test_query_enhancement()
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())