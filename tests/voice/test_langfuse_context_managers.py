#!/usr/bin/env python3
"""
Test script to verify Langfuse context managers implementation
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

async def test_langfuse_context_managers():
    """Test the Langfuse context manager implementation"""
    print("\n🧪 Testing Langfuse Context Managers")
    print("=" * 50)
    
    # Test 1: Verify Langfuse client can be initialized
    print("\n1. Testing Langfuse client initialization...")
    try:
        client = get_client()
        client.auth_check()
        print("   ✅ Langfuse client initialized successfully")
    except Exception as e:
        print(f"   ❌ Failed to initialize Langfuse client: {e}")
        print("   Note: Make sure LANGFUSE_* environment variables are set")
        return False
    
    # Test 2: Test context manager for spans
    print("\n2. Testing span context manager...")
    try:
        with client.start_as_current_span(
            name="test_span",
            input={"test": "data"},
            metadata={"operation": "test"}
        ) as span:
            span.update(
                output={"result": "success"},
                metadata={"additional": "info"}
            )
        print("   ✅ Span context manager works correctly")
    except Exception as e:
        print(f"   ❌ Span context manager failed: {e}")
        return False
    
    # Test 3: Test context manager for generations
    print("\n3. Testing generation context manager...")
    try:
        with client.start_as_current_generation(
            name="test_generation",
            model="test-model",
            input=[{"role": "user", "content": "test"}],
            metadata={"is_primary": True}
        ) as generation:
            generation.update(
                output="Test response",
                usage_details={"total_tokens": 10}
            )
        print("   ✅ Generation context manager works correctly")
    except Exception as e:
        print(f"   ❌ Generation context manager failed: {e}")
        return False
    
    # Test 4: Test nested context managers
    print("\n4. Testing nested context managers...")
    try:
        with client.start_as_current_span(
            name="parent_span",
            metadata={"level": "parent"}
        ) as parent_span:
            # Nested generation
            with client.start_as_current_generation(
                name="nested_generation",
                model="test-model",
                metadata={"level": "child"}
            ) as child_gen:
                child_gen.update(output="Nested output")
            
            # Nested span (events are also spans in Langfuse v3)
            with client.start_as_current_span(
                name="nested_event",
                metadata={"level": "child", "type": "event"}
            ) as event_span:
                event_span.update(level="INFO")
            
            parent_span.update(metadata={"children_created": 2})
        
        print("   ✅ Nested context managers work correctly")
    except Exception as e:
        print(f"   ❌ Nested context managers failed: {e}")
        return False
    
    # Test 5: Test event as span (Langfuse v3 uses spans for events)
    print("\n5. Testing event as span...")
    try:
        with client.start_as_current_span(
            name="test_event",
            input={"event_type": "security"},
            metadata={"severity": "high", "type": "event"}
        ) as event_span:
            event_span.update(level="WARNING")
        print("   ✅ Event as span works correctly")
    except Exception as e:
        print(f"   ❌ Event as span failed: {e}")
        return False
    
    # Flush to ensure all data is sent
    client.flush()
    
    print("\n" + "=" * 50)
    print("✅ All Langfuse context manager tests passed!")
    return True

async def test_assistant_langfuse_integration():
    """Test the Assistant class Langfuse integration"""
    print("\n\n🧪 Testing Assistant Langfuse Integration")
    print("=" * 50)
    
    # Mock the necessary components
    from liddy_voice.assistant import Assistant
    from livekit.agents import JobContext, llm
    
    # Create mock context
    mock_ctx = Mock(spec=JobContext)
    mock_room = Mock()
    mock_room.name = "test-room-123"
    mock_ctx.room = mock_room
    
    print("\n1. Testing Assistant initialization with Langfuse...")
    try:
        # Create assistant instance
        assistant = Assistant(
            ctx=mock_ctx,
            primary_model="gpt-4",
            user_id="test-user",
            account="test-account"
        )
        
        # Check Langfuse initialization
        assert hasattr(assistant, '_langfuse_client')
        assert hasattr(assistant, '_langfuse_active')
        assert hasattr(assistant, '_turn_count')
        assert hasattr(assistant, '_conversation_metadata')
        
        print("   ✅ Assistant initialized with Langfuse attributes")
        print(f"   - Session ID: {assistant._session_id}")
        print(f"   - Conversation ID: {assistant._conversation_id}")
        print(f"   - Langfuse active: {assistant._langfuse_active}")
        
    except Exception as e:
        print(f"   ❌ Assistant initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n2. Testing conversation metadata...")
    try:
        assert assistant._conversation_metadata['account'] == 'test-account'
        assert assistant._conversation_metadata['session_id'] == 'test-room-123'
        assert assistant._conversation_metadata['primary_model'] == 'gpt-4'
        assert 'security_mode' in assistant._conversation_metadata
        print("   ✅ Conversation metadata properly configured")
    except Exception as e:
        print(f"   ❌ Metadata verification failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✅ Assistant Langfuse integration tests passed!")
    return True

async def main():
    """Run all tests"""
    print("🔄 Langfuse Context Manager Test Suite")
    
    try:
        # Check environment
        if not os.getenv("LANGFUSE_SECRET_KEY"):
            print("\n⚠️  Warning: LANGFUSE_SECRET_KEY not set")
            print("   Some tests may fail without proper Langfuse configuration")
        
        # Run tests
        test1_passed = await test_langfuse_context_managers()
        test2_passed = await test_assistant_langfuse_integration()
        
        if test1_passed and test2_passed:
            print("\n🎉 All tests passed! Langfuse context managers are working correctly.")
        else:
            print("\n❌ Some tests failed. Please check the output above.")
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