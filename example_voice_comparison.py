#!/usr/bin/env python3
"""
Example: Running Voice-Realistic Search Comparison

This script demonstrates how to run a voice-realistic comparison
between enhanced and baseline search approaches.
"""

import asyncio
import logging
from voice_search_comparison import VoiceSearchTestRunner
from test_search_comparison import BaselineSearchIndex
from src.models.product_manager import ProductManager
from src.storage import get_account_storage_provider

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def setup_baseline_if_needed(account: str, max_products: int = 500):
    """Set up baseline index if it doesn't exist."""
    baseline = BaselineSearchIndex(account)
    
    # Check if index has data
    try:
        results = await baseline.search("bike", top_k=1)
        if results:
            print(f"✅ Baseline index already has data")
            return
    except:
        pass
    
    print(f"📦 Setting up baseline index for {account}...")
    
    # Load and ingest products
    storage_manager = get_account_storage_provider()
    product_manager = ProductManager(storage_manager)
    products = await product_manager.fetch_products(account, num_products=max_products)
    
    print(f"📥 Ingesting {len(products)} products...")
    await baseline.create_index()
    await baseline.ingest_products(products)
    
    print(f"✅ Baseline index ready!")

async def run_single_scenario_demo(account: str):
    """Run a single voice scenario for demonstration."""
    runner = VoiceSearchTestRunner(account)
    
    print(f"\n🎯 Initializing voice search comparison for {account}...")
    await runner.setup()
    
    # Create a custom scenario
    print("\n🎭 Running voice scenario: Parent Shopping")
    comparator = runner.comparator
    
    # Simulate the conversation
    conversation_result = await comparator.simulate_voice_conversation(
        scenario_description="Parent needs a bike for their 10-year-old daughter who likes pink/purple colors",
        max_turns=3
    )
    
    print("\n💬 Conversation Flow:")
    for i, msg in enumerate(conversation_result['conversation']):
        role = msg['role'].upper()
        content = msg['content']
        print(f"{i+1}. {role}: {content[:100]}{'...' if len(content) > 100 else ''}")
    
    print(f"\n🔧 Tool Calls Made: {len(conversation_result['tool_calls'])}")
    for tc in conversation_result['tool_calls']:
        print(f"  - {tc.tool_name}: {tc.parameters}")
    
    # Compare search results for each query
    if conversation_result['search_queries']:
        print(f"\n🔍 Comparing {len(conversation_result['search_queries'])} search queries...")
        
        for sq in conversation_result['search_queries']:
            print(f"\n📊 Query: '{sq['query']}'")
            
            comparison = await comparator.compare_search_approaches(sq)
            
            print(f"  Enhanced: {len(comparison.enhanced_results)} results in {comparison.enhanced_time:.3f}s")
            print(f"  Baseline: {len(comparison.baseline_results)} results in {comparison.baseline_time:.3f}s")
            print(f"  Overlap: {comparison.overlap_ratio:.2%}")
            
            # Show top result from each
            if comparison.enhanced_results:
                top_enhanced = comparison.enhanced_results[0]
                print(f"\n  Top Enhanced Result: {top_enhanced.name} (${top_enhanced.price})")
                if top_enhanced.relevance_explanation:
                    print(f"    Relevance: {top_enhanced.relevance_explanation}")
            
            if comparison.baseline_results:
                top_baseline = comparison.baseline_results[0]
                print(f"  Top Baseline Result: {top_baseline.name} (${top_baseline.price})")

async def main():
    """Main function."""
    account = "specialized.com"  # Change this to test different brands
    
    print(f"🚀 Voice-Realistic Search Comparison Demo")
    print(f"🏢 Account: {account}\n")
    
    # Ensure baseline is set up
    await setup_baseline_if_needed(account)
    
    # Run single scenario demo
    await run_single_scenario_demo(account)
    
    # Optional: Run full test suite
    run_full = input("\n\nRun full test suite? (y/n): ")
    if run_full.lower() == 'y':
        print("\n🏃 Running full voice test suite...")
        runner = VoiceSearchTestRunner(account)
        await runner.setup()
        results = await runner.run_voice_tests()
        await runner.save_results()
        print(f"\n✅ Complete! Results saved to voice_search_results/")
    
    print("\n🎉 Demo complete!")

if __name__ == "__main__":
    asyncio.run(main())