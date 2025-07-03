#!/usr/bin/env python3
"""
Voice Assistant Main Entry Point

This is a simplified runner for the voice assistant that demonstrates
how to use the migrated voice components from the monorepo.
"""

import asyncio
import logging
import os
import sys

# Add packages to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "packages")))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_voice_components():
    """Test basic voice component functionality."""
    
    # Test imports
    try:
        from liddy_voice.spence.assistant import Assistant, UserState
        from liddy_voice.spence.account_manager import get_account_manager
        from liddy_voice.spence.search_service import SearchService
        from liddy_voice.spence.rag_unified import PineconeRAG
        
        print("✅ All imports successful!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
        
    # Test account manager
    print("\n=== Testing Account Manager ===")
    try:
        account = "specialized.com"
        account_manager = await get_account_manager(account)
        
        # Get RAG details
        index_name, embedding_model = account_manager.get_rag_details()
        print(f"Account: {account}")
        print(f"Index: {index_name}")
        print(f"Embedding Model: {embedding_model}")
        
        # Get TTS settings
        tts_settings = account_manager.get_tts_settings()
        primary_provider = tts_settings.primary_provider
        print(f"TTS Provider: {primary_provider.voice_provider if hasattr(primary_provider, 'voice_provider') else 'elevenlabs'}")
        print(f"Voice Name: {primary_provider.voice_name}")
        
    except Exception as e:
        print(f"❌ Account manager error: {e}")
        
    # Test search functionality
    print("\n=== Testing Search Service ===")
    try:
        user_state = UserState(account="specialized.com", user_id="test_user")
        
        # Test product search
        results = await SearchService.search_products_rag(
            query="mountain bike",
            account="specialized.com",
            top_k=5
        )
        
        print(f"Search Results: {len(results)} products found")
        if results:
            first = results[0]
            metadata = first.get('metadata', {})
            print(f"First result: {metadata.get('name', 'Unknown')} (score: {first.get('score', 0):.3f})")
            
    except Exception as e:
        print(f"❌ Search service error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main entry point."""
    print("Voice Assistant Test Runner")
    print("==========================")
    
    # Check environment
    required_vars = ['PINECONE_API_KEY', 'OPENAI_API_KEY']
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        print("Please set these in your .env file or environment")
        return
        
    # Run tests
    await test_voice_components()
    
    print("\n✅ Test completed!")
    
    # Note about full voice assistant
    print("\n" + "="*50)
    print("NOTE: This is a simplified test runner.")
    print("For the full voice assistant with LiveKit integration,")
    print("you'll need to:")
    print("1. Copy main.py from ../voice-service/")
    print("2. Update imports to use liddy_voice.spence.*")
    print("3. Set up LiveKit configuration")
    print("="*50)


if __name__ == "__main__":
    asyncio.run(main())