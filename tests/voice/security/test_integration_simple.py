#!/usr/bin/env python3
"""
Simple verification that security components are integrated into Assistant
"""

import sys
import os
sys.path.append('packages')

import logging
import asyncio
from unittest.mock import Mock, patch, AsyncMock

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_security_integration():
    """Verify security components are properly integrated"""
    print("\n🛡️ Verifying Security Integration in Assistant")
    print("=" * 50)
    
    # Import Assistant to check integration
    from liddy_voice.assistant import Assistant
    from liddy_voice.security import RuntimeSecurityManager
    
    print("\n1. Checking imports...")
    # Check that security imports are present
    import inspect
    assistant_source = inspect.getsource(Assistant)
    
    required_imports = [
        "RuntimeSecurityManager",
        "get_voice_security_config", 
        "create_security_manager"
    ]
    
    for import_name in required_imports:
        if import_name in assistant_source:
            print(f"   ✅ Found import: {import_name}")
        else:
            print(f"   ❌ Missing import: {import_name}")
            return False
    
    print("\n2. Checking initialization...")
    # Check that security manager is initialized in __init__
    init_source = inspect.getsource(Assistant.__init__)
    
    required_init = [
        "_security_manager",
        "_security_config",
        "_conversation_id",
        "create_security_manager"
    ]
    
    for init_item in required_init:
        if init_item in init_source:
            print(f"   ✅ Found initialization: {init_item}")
        else:
            print(f"   ❌ Missing initialization: {init_item}")
            return False
    
    print("\n3. Checking llm_node integration...")
    # Check that llm_node has security checks
    llm_node_source = inspect.getsource(Assistant.llm_node)
    
    required_security = [
        "process_conversation_turn",
        "process_response",
        "security_result",
        "echo_result"
    ]
    
    for security_item in required_security:
        if security_item in llm_node_source:
            print(f"   ✅ Found security check: {security_item}")
        else:
            print(f"   ❌ Missing security check: {security_item}")
            return False
    
    print("\n4. Checking lifecycle management...")
    # Check on_exit and on_room_disconnected
    on_exit_source = inspect.getsource(Assistant.on_exit)
    on_disconnect_source = inspect.getsource(Assistant.on_room_disconnected)
    
    if "end_conversation" in on_exit_source:
        print("   ✅ Found end_conversation in on_exit")
    else:
        print("   ❌ Missing end_conversation in on_exit")
        return False
    
    if "_security_manager" in on_disconnect_source:
        print("   ✅ Found security cleanup in on_room_disconnected")
    else:
        print("   ❌ Missing security cleanup in on_room_disconnected")
        return False
    
    print("\n5. Checking security configuration...")
    # Verify environment variable handling
    os.environ['VOICE_ONLY_MODE'] = 'true'
    from liddy_voice.security import get_voice_security_config
    config = get_voice_security_config()
    
    if config.is_voice_only_mode():
        print("   ✅ Voice-only mode configuration working")
    else:
        print("   ❌ Voice-only mode configuration not working")
        return False
    
    # Reset for other tests
    os.environ['VOICE_ONLY_MODE'] = 'false'
    
    print("\n" + "=" * 50)
    print("✅ All security integrations verified!")
    print("\n🎉 Security components are successfully integrated!")
    
    return True


if __name__ == "__main__":
    success = verify_security_integration()
    sys.exit(0 if success else 1)