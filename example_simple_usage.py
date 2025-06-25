#!/usr/bin/env python3
"""
Example: Simple LLM Factory vs Complex Router

Demonstrates the simplified approach you preferred over the complex router system.
"""

import asyncio
import os
from src.llm import LLMFactory

async def main():
    """Show the simple approach vs complex router"""
    
    print("🎯 YOUR PREFERRED APPROACH: Simple & Direct")
    print("=" * 50)
    
    # 1. Direct service creation (your fetch_llm_model_service pattern)
    print("✅ Direct service creation:")
    print("   service = LLMFactory.get_service('openai/gpt-4-turbo')")
    
    # 2. One-line LLM calls
    print("\n✅ One-line LLM calls:")
    print("   response = await LLMFactory.chat_completion(")
    print("       task='descriptor_generation',")
    print("       messages=[{'role': 'user', 'content': 'Hello'}]")
    print("   )")
    
    # 3. Environment-configurable
    print("\n✅ Environment-configurable:")
    print("   export DESCRIPTOR_MODEL=anthropic/claude-3-5-sonnet")
    print("   # Now all descriptor tasks use Claude")
    
    print("\n\n❌ COMPLEX ROUTER (What We're Avoiding):")
    print("=" * 50)
    print("❌ Complex registration:")
    print("   router = LLMRouter()")
    print("   router.register_provider('openai', service, models, priority=100)")
    print("   router.set_task_routing('task', 'provider/model')")
    
    print("\n❌ Complex usage:")
    print("   provider, service, model = router.get_optimal_provider(task)")
    print("   response = await router.chat_completion(...)")
    
    print("\n\n🎯 CONCLUSION:")
    print("   Your fetch_llm_model_service approach is much cleaner!")
    print("   Simple factory pattern > Complex router pattern")

if __name__ == "__main__":
    asyncio.run(main())
