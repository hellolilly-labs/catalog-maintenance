"""
Test Just Web Search for Specialized.com

This tests only the web search functionality to isolate the issue.
"""

import asyncio
import json
import os
from datetime import datetime

async def test_web_search_specialized():
    """Test just the web search functionality for specialized.com"""
    
    print("🌐 TESTING WEB SEARCH FOR SPECIALIZED.COM")
    print("=" * 43)
    
    try:
        from src.web_search import get_web_search_engine
        from src.llm.simple_factory import LLMFactory
        
        # Test web search engine
        web_search = get_web_search_engine()
        
        if not web_search.is_available():
            print("❌ No web search providers available")
            print("Available providers:", web_search.get_provider_status())
            return None
        
        print("✅ Web search providers available")
        print("Provider status:", web_search.get_provider_status())
        print()
        
        # Do the search
        print("🔍 Searching for specialized.com brand information...")
        start_time = datetime.now()
        
        search_results = await web_search.search_brand_info("specialized.com")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"✅ Search completed in {duration:.1f} seconds")
        print()
        
        print("📊 SEARCH RESULTS:")
        print("-" * 16)
        print(f"Total Results: {search_results['total_results']}")
        print(f"Provider Used: {search_results['provider_used']}")
        print(f"Search Strategy: {search_results['search_strategy']}")
        print()
        
        # Show top 5 results
        print("🔍 TOP SEARCH RESULTS:")
        for i, result in enumerate(search_results['results'][:5], 1):
            print(f"\n{i}. {result.get('title', 'No title')}")
            print(f"   URL: {result.get('url', 'No URL')}")
            print(f"   Query: {result.get('query', 'No query')}")
            print(f"   Snippet: {result.get('snippet', 'No snippet')[:150]}...")
        
        print(f"\n🧠 TESTING LLM ANALYSIS OF SEARCH RESULTS")
        print("-" * 42)
        
        # Test simple LLM call
        search_context = ""
        for result in search_results["results"][:5]:
            search_context += f"Title: {result.get('title', '')}\n"
            search_context += f"URL: {result.get('url', '')}\n"
            search_context += f"Snippet: {result.get('snippet', '')}\n\n"
        
        llm_prompt = f"""Based on these search results about specialized.com, what is their primary business vertical?

SEARCH RESULTS:
{search_context}

Choose from: cycling, fashion, electronics, sports, automotive, home, beauty, food, etc.

Respond with just the vertical name and a brief explanation."""

        print("⚡ Calling LLM for analysis...")
        llm_start = datetime.now()
        
        response = await LLMFactory.chat_completion(
            task="brand_research",
            system="You are a business analyst. Analyze search results to determine company verticals.",
            messages=[{
                "role": "user", 
                "content": llm_prompt
            }],
            max_tokens=200,
            temperature=0.1
        )
        
        llm_end = datetime.now()
        llm_duration = (llm_end - llm_start).total_seconds()
        
        print(f"✅ LLM analysis completed in {llm_duration:.1f} seconds")
        print()
        
        if response and response.get("content"):
            print("🎯 LLM ANALYSIS RESULT:")
            print("-" * 22)
            print(response["content"])
            print()
            
            # Check if it mentions cycling
            content_lower = response["content"].lower()
            if "cycling" in content_lower or "bicycle" in content_lower or "bike" in content_lower:
                print("✅ SUCCESS: LLM correctly identified cycling/bicycle vertical!")
            else:
                print("⚠️  LLM didn't identify cycling - may need better search results")
        else:
            print("❌ No LLM response received")
        
        return {
            "search_duration": duration,
            "llm_duration": llm_duration,
            "total_results": search_results['total_results'],
            "llm_response": response.get("content") if response else None
        }
        
    except Exception as e:
        print(f"❌ Error in web search test: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """Main test function"""
    print("🧪 ISOLATED WEB SEARCH TEST")
    print("=" * 28)
    print("Testing just web search + LLM analysis")
    print("(Bypassing product catalog and complex imports)")
    print()
    
    # Check environment
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not set")
        return
    
    print(f"✅ OPENAI_API_KEY: Set (***{os.getenv('OPENAI_API_KEY')[-4:]})")
    print()
    
    result = await test_web_search_specialized()
    
    print("\n" + "=" * 50)
    print("🎉 TEST COMPLETE!")
    
    if result and result.get("llm_response"):
        print("✅ Web search + LLM analysis working!")
        print(f"⚡ Performance: {result['search_duration']:.1f}s search + {result['llm_duration']:.1f}s LLM")
    else:
        print("⚠️  Issues found - see above")

if __name__ == "__main__":
    asyncio.run(main())
