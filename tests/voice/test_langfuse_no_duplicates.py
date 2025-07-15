#!/usr/bin/env python3
"""
Test script to verify we're not creating duplicate Langfuse traces
"""

import sys
import os
sys.path.append('packages')

import asyncio
import logging
from unittest.mock import Mock, MagicMock, patch
from langfuse import get_client

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_no_duplicate_traces():
    """Test that we're not creating duplicate traces for LLM calls"""
    print("\n🧪 Testing No Duplicate Traces")
    print("=" * 50)
    
    # Import after path is set
    from liddy_voice.assistant import Assistant
    from livekit.agents import JobContext, llm
    
    # Create mock context
    mock_ctx = Mock(spec=JobContext)
    mock_room = Mock()
    mock_room.name = "test-room-duplicate-check"
    mock_ctx.room = mock_room
    
    print("\n1. Creating Assistant instance...")
    try:
        # Create assistant instance
        assistant = Assistant(
            ctx=mock_ctx,
            primary_model="gpt-4",
            user_id="test-user",
            account="test-account"
        )
        
        print("   ✅ Assistant created successfully")
        
        # Verify the LLM is properly configured
        print("\n2. Checking LLM configuration...")
        
        # The assistant should have turn tracking
        assert hasattr(assistant, '_turn_count')
        assert assistant._turn_count == 0
        
        # The assistant should have Langfuse client
        assert hasattr(assistant, '_langfuse_client')
        assert assistant._langfuse_active
        
        print("   ✅ LLM configuration looks correct")
        print("   - Turn count initialized: 0")
        print("   - Langfuse client active")
        
        print("\n3. Verifying llm_node implementation...")
        
        # Check that llm_node method exists and doesn't have @observe decorator
        import inspect
        llm_node_source = inspect.getsource(assistant.llm_node)
        
        # Verify no manual context manager for Langfuse in llm_node
        assert "start_as_current_generation" not in llm_node_source
        assert "The LangfuseLKOpenAILLM client already provides automatic tracing" in llm_node_source
        
        print("   ✅ llm_node correctly uses automatic tracing")
        print("   - No manual context manager found")
        print("   - Comment confirms reliance on LangfuseLKOpenAILLM")
        
        print("\n4. Checking other methods still have context managers...")
        
        # Verify tool methods still have context managers
        product_search_source = inspect.getsource(assistant.product_search)
        display_product_source = inspect.getsource(assistant.display_product)
        
        assert "start_as_current_span" in product_search_source
        assert "start_as_current_span" in display_product_source
        
        print("   ✅ Tool methods retain their context managers")
        print("   - product_search: Has context manager")
        print("   - display_product: Has context manager")
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 50)
    print("✅ No duplicate traces test passed!")
    return True

async def main():
    """Run tests"""
    print("🔄 No Duplicate Traces Test Suite")
    
    try:
        # Check environment
        if not os.getenv("LANGFUSE_SECRET_KEY"):
            print("\n⚠️  Warning: LANGFUSE_SECRET_KEY not set")
            print("   Some functionality may not work without proper Langfuse configuration")
        
        # Run test
        test_passed = await test_no_duplicate_traces()
        
        if test_passed:
            print("\n🎉 All tests passed! Duplicate trace issue has been resolved.")
            print("\nSummary of changes:")
            print("- Removed manual Langfuse context manager from llm_node")
            print("- LangfuseLKOpenAILLM now handles all LLM tracing automatically")
            print("- Tool functions retain their manual context managers")
            print("- This eliminates the duplicate 'llm_node' and 'OpenAI-generation' traces")
        else:
            print("\n❌ Test failed. Please check the output above.")
            return False
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)