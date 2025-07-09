#!/usr/bin/env python3
"""
Test Multi-Agent System Infrastructure

Tests the core multi-agent infrastructure including:
- Agent Communication Hub
- Customer Psychology Agent
- Real-time processing and insight fusion
- LiveKit integration considerations
"""

import asyncio
import logging
from datetime import datetime

from src.agents.communication_hub import AgentCommunicationHub
from src.agents.psychology_agent import create_psychology_agent
from src.agents.context import ConversationContext

# Configure logging for testing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_multi_agent_infrastructure():
    """Test the multi-agent infrastructure with psychology agent"""
    
    print("🤖 Testing Multi-Agent Intelligent Discovery System")
    print("=" * 60)
    
    # Initialize communication hub
    hub = AgentCommunicationHub(max_total_processing_time_ms=200)
    
    # Create and register psychology agent
    psychology_agent = create_psychology_agent()
    hub.register_agent(psychology_agent)
    
    print(f"✅ Registered {len(hub.active_agents)} agents")
    
    # Test conversation scenarios
    test_scenarios = [
        {
            "conversation_id": "test_001",
            "brand_domain": "specialized.com",
            "messages": [
                "Hi, I'm looking for a bike for my daily commute to work",
                "I need something reliable and fast. What do you recommend?",
                "That sounds expensive. Do you have anything more affordable?",
                "I'm really excited about this! When can I get it delivered?"
            ],
            "livekit_context": {
                "room_id": "room_specialized_001",
                "audio_enabled": True,
                "video_enabled": False
            }
        },
        {
            "conversation_id": "test_002", 
            "brand_domain": "specialized.com",
            "messages": [
                "I'm confused about all these bike options. Can you help me understand the differences?",
                "I want to compare the technical specifications of your top 3 models",
                "This is frustrating. I just want a simple recommendation.",
                "Ok, I think I'm ready to make a decision. What's the best value?"
            ],
            "livekit_context": {
                "room_id": "room_specialized_002",
                "audio_enabled": True,
                "video_enabled": True
            }
        }
    ]
    
    # Process each test scenario
    for scenario in test_scenarios:
        print(f"\n🎯 Testing Scenario: {scenario['conversation_id']}")
        print("-" * 40)
        
        for i, message in enumerate(scenario["messages"]):
            print(f"\n📝 Customer Message {i+1}: '{message}'")
            
            # Process message through multi-agent system
            start_time = datetime.now()
            
            enhanced_context = await hub.process_customer_message(
                message=message,
                conversation_id=scenario["conversation_id"],
                brand_domain=scenario["brand_domain"],
                livekit_room_id=scenario["livekit_context"]["room_id"],
                audio_enabled=scenario["livekit_context"]["audio_enabled"],
                video_enabled=scenario["livekit_context"]["video_enabled"]
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Display results
            print(f"⏱️  Processing time: {processing_time:.1f}ms")
            print(f"🎯 Confidence score: {enhanced_context.confidence_score:.2f}")
            print(f"🔗 Insights used: {enhanced_context.insights_used}")
            
            # Show psychology insights
            if enhanced_context.psychology_insights:
                psych = enhanced_context.psychology_insights
                print(f"🧠 Psychology Analysis:")
                print(f"   Emotional state: {psych.insights.get('emotional_state', 'unknown')}")
                print(f"   Communication style: {psych.insights.get('communication_style', 'unknown')}")
                print(f"   Decision style: {psych.insights.get('decision_making_style', 'unknown')}")
                print(f"   Urgency level: {psych.insights.get('urgency_level', 'unknown')}")
                print(f"   Price sensitivity: {psych.insights.get('price_sensitivity', 'unknown')}")
                
                if psych.recommendations:
                    print(f"💡 Recommendations:")
                    for rec in psych.recommendations[:3]:
                        print(f"   • {rec}")
            
            # Show real-time recommendations
            if enhanced_context.immediate_opportunities:
                print(f"⚡ Immediate Opportunities: {enhanced_context.immediate_opportunities}")
            
            print(f"📊 Conversation Context:")
            context_summary = enhanced_context.conversation_context.get_context_summary()
            print(f"   Stage: {context_summary['conversation_stage']}")
            print(f"   Intent: {context_summary['customer_intent']}")
            print(f"   Messages: {context_summary['message_count']}")
            print(f"   LiveKit Room: {context_summary['livekit_context']['room_id']}")
    
    # Test system health
    print(f"\n🏥 System Health Check:")
    print("-" * 30)
    health = hub.get_system_health()
    
    hub_health = health["hub_health"]
    print(f"✅ Hub Success Rate: {hub_health['success_rate']:.1%}")
    print(f"⏱️  Average Processing: {hub_health['avg_processing_time_ms']:.1f}ms")
    print(f"🤖 Active Agents: {hub_health['active_agents']}")
    print(f"💚 Healthy Agents: {hub_health['healthy_agents']}")
    
    for agent_name, agent_health in health["agent_health"].items():
        status = "✅" if agent_health["is_healthy"] else "❌"
        print(f"{status} {agent_name}: {agent_health['success_rate']:.1%} success, "
              f"{agent_health['avg_processing_time_ms']:.1f}ms avg")
    
    print(f"\n🎉 Multi-Agent System Test Complete!")
    print(f"📈 Processed {hub_health['total_requests']} requests across {len(test_scenarios)} conversations")

async def test_psychology_agent_standalone():
    """Test psychology agent in isolation"""
    
    print("\n🧠 Testing Psychology Agent Standalone")
    print("=" * 40)
    
    # Create psychology agent
    agent = create_psychology_agent()
    
    # Create test context
    context = ConversationContext(
        conversation_id="psych_test",
        brand_domain="specialized.com"
    )
    
    # Add some conversation history
    context.add_message("customer", "Hi, I'm looking for a high-performance bike")
    context.add_message("agent", "Great! What type of riding will you be doing?")
    context.add_message("customer", "I need something for racing and training")
    
    # Test messages with different psychology patterns
    test_messages = [
        "I'm really excited about getting a new bike! What's the fastest one you have?",
        "I'm confused by all these options. Can you just recommend something simple?",
        "This is too expensive. Do you have anything cheaper that's still good quality?",
        "I need detailed specifications and performance comparisons before I decide",
        "My friends all have Specialized bikes and love them. What's most popular?"
    ]
    
    for i, message in enumerate(test_messages):
        print(f"\n📝 Test Message {i+1}: '{message}'")
        
        # Add message to context
        context.add_message("customer", message)
        
        # Analyze with psychology agent
        start_time = datetime.now()
        insight = await agent.analyze_with_timeout(message, context)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        if insight:
            print(f"⏱️  Processing: {processing_time:.1f}ms")
            print(f"🎯 Confidence: {insight.confidence_score:.2f}")
            print(f"📊 Insights:")
            for key, value in insight.insights.items():
                if not key.endswith("_confidence") and key != "analysis_method":
                    print(f"   {key}: {value}")
            
            if insight.recommendations:
                print(f"💡 Top Recommendations:")
                for rec in insight.recommendations[:2]:
                    print(f"   • {rec}")
        else:
            print("❌ No insight generated")
    
    # Check agent health
    metrics = agent.get_performance_metrics()
    print(f"\n📈 Agent Performance:")
    print(f"   Success Rate: {metrics.success_rate:.1%}")
    print(f"   Avg Processing: {metrics.avg_processing_time_ms:.1f}ms")
    print(f"   Error Count: {metrics.error_count}")
    print(f"   Healthy: {'✅' if agent.is_healthy() else '❌'}")

async def main():
    """Run all multi-agent system tests"""
    
    try:
        # Test psychology agent standalone
        await test_psychology_agent_standalone()
        
        # Test full multi-agent infrastructure
        await test_multi_agent_infrastructure()
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())