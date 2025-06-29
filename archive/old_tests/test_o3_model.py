#!/usr/bin/env python3
"""
Test script to verify o3 model usage for research tasks
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llm.simple_factory import LLMFactory


async def test_o3_configuration():
    """Test that o3 model is properly configured for research tasks"""
    
    print("🔍 Testing LLM Factory Configuration for O3 Model")
    print("=" * 60)
    
    # Test task to model mapping
    research_tasks = [
        'brand_research',
        'foundation_research', 
        'market_research',
        'product_research',
        'customer_research',
        'voice_research',
        'interview_research',
        'synthesis_research',
        'quality_evaluation',
        'summarization'
    ]
    
    print("📋 Task → Model Mapping:")
    for task in research_tasks:
        model = LLMFactory.get_model_for_task(task)
        status = "✅" if "o3" in model else "❌"
        print(f"  {status} {task}: {model}")
    
    print("\n🎯 Testing O3 Model Service Creation:")
    try:
        service = LLMFactory.get_service("openai/o3")
        print(f"  ✅ Service created: {service.__class__.__name__}")
        
        # Check if o3 is in supported models
        supported_models = service.list_supported_models()
        if "o3" in supported_models:
            print(f"  ✅ O3 model supported: Yes")
            model_info = service.get_model_info("o3")
            print(f"  📊 Context Window: {model_info['context_window']:,} tokens")
            print(f"  📝 Description: {model_info['description']}")
        else:
            print(f"  ❌ O3 model supported: No")
            print(f"  📋 Supported models: {supported_models}")
            
    except Exception as e:
        print(f"  ❌ Error creating service: {e}")
    
    print("\n🧪 Testing Simple O3 Research Task:")
    try:
        response = await LLMFactory.chat_completion(
            task="brand_research",
            system="You are a helpful research assistant.",
            messages=[{
                "role": "user", 
                "content": "What are the key elements of effective brand research? Provide a brief overview."
            }],
            max_tokens=300,
            temperature=0.1
        )
        
        if response and response.get("content"):
            print(f"  ✅ O3 Research Response Generated")
            print(f"  📊 Model Used: {response.get('model', 'unknown')}")
            print(f"  📝 Token Usage: {response.get('usage', {}).get('total_tokens', 0)} tokens")
            print(f"  🎯 Sample Response: {response['content'][:150]}...")
        else:
            print(f"  ❌ No response generated")
            
    except Exception as e:
        print(f"  ❌ Error in research task: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 O3 Configuration Test Complete")


if __name__ == "__main__":
    asyncio.run(test_o3_configuration()) 