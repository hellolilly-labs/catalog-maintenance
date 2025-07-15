#!/usr/bin/env python3
"""
Test frequency penalty implementation for echo monitor
"""

import sys
import os
sys.path.append('packages')

import asyncio
import logging

from liddy_voice.security.echo_monitor import EchoMonitor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_frequency_penalty_implementation():
    """Test that frequency penalty is properly implemented"""
    print("\n🧪 Testing Frequency Penalty Implementation")
    print("=" * 50)
    
    monitor = EchoMonitor()
    conversation_id = "test_freq_penalty"
    
    # Test 1: Baseline - no echo, no penalty
    print("\n1. Testing baseline (no echo)...")
    result = await monitor.check_response(
        "What products do you have?",
        "We have mountain bikes, road bikes, and accessories.",
        conversation_id
    )
    
    assert result['is_echo'] is False
    assert 'temp_penalty_adjustment' not in result or result['temp_penalty_adjustment'] == {}
    assert monitor.pending_reset is False
    print("   ✅ No penalty applied for non-echo response")
    
    # Test 2: Trigger intervention - should add frequency penalty
    print("\n2. Testing echo intervention trigger...")
    
    # Feed 3 consecutive echoes to trigger intervention
    echo_inputs = [
        ("Tell me about bikes", "Tell me about bikes"),
        ("Show me products", "Show me products"),
        ("What's available?", "What's available?")
    ]
    
    for i, (user_input, ai_response) in enumerate(echo_inputs):
        result = await monitor.check_response(user_input, ai_response, conversation_id)
        print(f"   Echo {i+1}: score={result['echo_score']:.3f}, consecutive={result['consecutive_count']}")
    
    # Last one should trigger intervention
    assert result['requires_intervention'] is True
    assert result['temp_penalty_adjustment'] == {'frequency_penalty': 0.2}
    assert monitor.pending_reset is True
    print("   ✅ Frequency penalty=0.2 applied after 3 consecutive echoes")
    print("   ✅ pending_reset flag set to True")
    
    # Test 3: Auto-reset on next turn
    print("\n3. Testing auto-reset...")
    
    # Check that should_reset returns True
    should_reset = monitor.should_reset_frequency_penalty()
    assert should_reset is True
    assert monitor.pending_reset is False  # Should be cleared after check
    print("   ✅ should_reset_frequency_penalty() returned True")
    print("   ✅ pending_reset flag cleared")
    
    # Verify next non-echo turn doesn't have penalty
    result = await monitor.check_response(
        "How much does it cost?",
        "Our mountain bikes range from $500 to $5000 depending on the model.",
        conversation_id
    )
    
    assert result['is_echo'] is False
    assert 'temp_penalty_adjustment' not in result or result['temp_penalty_adjustment'] == {}
    assert monitor.pending_reset is False
    print("   ✅ No penalty on subsequent non-echo turn")
    
    # Test 4: Verify reset doesn't happen twice
    print("\n4. Testing reset only happens once...")
    should_reset = monitor.should_reset_frequency_penalty()
    assert should_reset is False
    print("   ✅ Second reset check returns False")
    
    print("\n" + "=" * 50)
    print("✅ All frequency penalty tests passed!")
    
    return True


async def test_integration_scenario():
    """Test a realistic conversation flow with echo intervention"""
    print("\n🧪 Testing Integration Scenario")
    print("=" * 50)
    
    monitor = EchoMonitor()
    conversation_id = "test_integration"
    
    # Simulate a conversation that develops echo behavior
    conversation = [
        # Normal start
        ("Hi, I'm looking for a bike", "Hello! I'd be happy to help you find the perfect bike.", False),
        ("What types do you have?", "We offer mountain bikes, road bikes, and hybrid bikes.", False),
        # Echo behavior starts (using exact echoes to ensure they exceed 0.75 threshold)
        ("Show me mountain bikes", "Show me mountain bikes", True),
        ("What colors are available", "What colors are available", True),
        ("Do you have red ones", "Do you have red ones", True),  # Triggers intervention
        # Recovery after intervention
        ("How much do they cost?", "Our red mountain bikes range from $800 to $3500.", False),
        ("Tell me about the features", "The key features include hydraulic disc brakes and carbon frames.", False)
    ]
    
    reset_checked = False
    
    for i, (user_input, ai_response, expected_echo) in enumerate(conversation):
        print(f"\nTurn {i+1}:")
        
        # Check for reset before processing
        if monitor.pending_reset and not reset_checked:
            should_reset = monitor.should_reset_frequency_penalty()
            print(f"   🔄 Reset check: {should_reset}")
            reset_checked = True
        
        result = await monitor.check_response(user_input, ai_response, conversation_id)
        
        print(f"   User: {user_input}")
        print(f"   AI: {ai_response}")
        print(f"   Echo: {result['is_echo']} (score: {result['echo_score']:.3f})")
        
        if result['requires_intervention']:
            print(f"   🚨 Intervention triggered!")
            print(f"   📊 Penalty adjustment: {result['temp_penalty_adjustment']}")
        
        assert result['is_echo'] == expected_echo, f"Turn {i+1}: Expected echo={expected_echo}, got {result['is_echo']}"
    
    print("\n" + "=" * 50)
    print("✅ Integration scenario completed successfully!")
    
    return True


async def main():
    """Run all tests"""
    print("🔄 Echo Monitor Frequency Penalty Test Suite")
    
    try:
        # Run tests
        await test_frequency_penalty_implementation()
        await test_integration_scenario()
        
        print("\n🎉 All tests passed! Frequency penalty implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)