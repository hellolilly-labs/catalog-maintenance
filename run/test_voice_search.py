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

from liddy_voice.search_service import VoiceOptimizedSearchService
from liddy_voice.session_state_manager import SessionStateManager

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
        ("specialized.com", "specialized-dense", "specialized-sparse"),
    ]
    
    for account, dense_index, sparse_index in brands:
        print(f"\nTesting {account}...")
        try:
            # Create search service
            search_service = VoiceOptimizedSearchService(
                brand_domain=account,
                dense_index_name=dense_index,
                sparse_index_name=sparse_index
            )
            
            # Test search
            results = await search_service.search(
                query="test query",
                top_k=5
            )
            
            print(f"✅ Successfully connected to {dense_index}/{sparse_index}")
            print(f"   Found {len(results.products)} results")
            
        except Exception as e:
            print(f"❌ Failed to connect: {e}")


async def test_unified_search():
    """Test the unified search service."""
    print("\n=== Testing Unified Search Service ===")
    
    account = "specialized.com"
    
    # Test product search
    print(f"\nTesting product search for {account}...")
    try:
        # Create search service
        search_service = VoiceOptimizedSearchService(
            brand_domain=account,
            dense_index_name="specialized-dense",
            sparse_index_name="specialized-sparse"
        )
        
        results = await search_service.search(
            query="mountain bike",
            top_k=10
        )
        
        print(f"✅ Product search successful")
        print(f"   Found {len(results.products)} products")
        
        # Display first result if available
        if results.products:
            first = results.products[0]
            print(f"\n   First result:")
            print(f"   - Score: {first.score:.3f}")
            print(f"   - Name: {first.name}")
            print(f"   - ID: {first.id}")
            
    except Exception as e:
        print(f"❌ Product search failed: {e}")
        import traceback
        traceback.print_exc()


async def test_query_enhancement():
    """Test session management functionality."""
    print("\n=== Testing Session Management ===")
    
    try:
        # Create session manager
        session = SessionStateManager(agent_id="specialized-assistant")
        
        # Add some messages
        session.add_message("user", "I need a bike for trails")
        session.add_message("assistant", "What type of trails will you be riding?")
        session.add_message("user", "Mostly technical downhill")
        
        # Get conversation history
        history = session.get_conversation_history()
        
        print(f"✅ Session management successful")
        print(f"   Messages in history: {len(history)}")
        print(f"   Agent ID: {session.agent_id}")
        
    except Exception as e:
        print(f"❌ Session management failed: {e}")


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