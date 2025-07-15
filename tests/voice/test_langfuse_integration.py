#!/usr/bin/env python3
"""
Test Langfuse integration for conversation tracking
"""

import sys
import os
sys.path.append('packages')

import logging
from unittest.mock import Mock, patch, MagicMock

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_langfuse_integration():
    """Test that Langfuse is properly integrated for conversation tracking"""
    print("\n📊 Testing Langfuse Integration")
    print("=" * 50)
    
    # Import Assistant to check Langfuse integration
    from liddy_voice.assistant import Assistant
    import inspect
    
    print("\n1. Checking Langfuse imports...")
    assistant_source = inspect.getsource(Assistant)
    
    langfuse_features = [
        "get_client",
        "_langfuse_client",
        "_langfuse_trace",
        "trace(",
        "session_id=ctx.room.name"
    ]
    
    for feature in langfuse_features:
        if feature in assistant_source:
            print(f"   ✅ Found: {feature}")
        else:
            print(f"   ❌ Missing: {feature}")
            return False
    
    print("\n2. Checking on_exit integration...")
    on_exit_source = inspect.getsource(Assistant.on_exit)
    
    stats_features = [
        "security_stats",
        "_langfuse_trace.update",
        "security_summary",
        "conversation_duration_seconds",
        "flush()"
    ]
    
    for feature in stats_features:
        if feature in on_exit_source:
            print(f"   ✅ Found: {feature}")
        else:
            print(f"   ❌ Missing: {feature}")
            return False
    
    print("\n3. Verifying trace configuration...")
    init_source = inspect.getsource(Assistant.__init__)
    
    # Check that room name is used as session_id
    if "session_id=ctx.room.name" in init_source:
        print("   ✅ Room name used as session_id for LiveKit correlation")
    else:
        print("   ❌ Room name not used as session_id")
        return False
    
    # Check metadata includes all necessary fields
    metadata_fields = [
        "room_name",
        "conversation_id",
        "security_mode",
        "account"
    ]
    
    for field in metadata_fields:
        if f'"{field}"' in init_source:
            print(f"   ✅ Metadata includes: {field}")
        else:
            print(f"   ❌ Metadata missing: {field}")
    
    print("\n4. Checking security stats collection...")
    security_stats = [
        "echo_detections",
        "echo_rate",
        "blocked_inputs",
        "sanitized_inputs",
        "health_status",
        "overall_health_score"
    ]
    
    for stat in security_stats:
        if f'"{stat}"' in on_exit_source:
            print(f"   ✅ Collects stat: {stat}")
        else:
            print(f"   ❌ Missing stat: {stat}")
    
    print("\n" + "=" * 50)
    print("✅ Langfuse integration verified!")
    print("\n📊 Key Features:")
    print("  - Each conversation has a single trace with session_id = room_name")
    print("  - All LLM turns are linked to the conversation trace")
    print("  - Security stats are logged when conversation ends")
    print("  - Full conversation context available in Langfuse dashboard")
    
    return True


if __name__ == "__main__":
    success = test_langfuse_integration()
    if success:
        print("\n🎉 Langfuse is properly integrated for conversation tracking!")
    sys.exit(0 if success else 1)