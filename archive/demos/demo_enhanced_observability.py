#!/usr/bin/env python3
"""
Demo: Enhanced Observability + Tavily Crawl/Map Integration
Shows real-time progress tracking and comprehensive brand research capabilities.
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, Any

from src.progress_tracker import (
    get_progress_tracker, 
    StepType, 
    create_console_listener
)
from src.web_search import get_web_search_engine
from src.research.foundation_research import FoundationResearcher


async def demo_tavily_enhanced_capabilities():
    """Demo the enhanced Tavily crawl and map capabilities"""
    print("🚀 **DEMO: Enhanced Tavily Capabilities**")
    print("=" * 60)
    
    web_search = get_web_search_engine()
    if not web_search.is_available():
        print("❌ No web search providers available")
        return
    
    # Find Tavily provider with enhanced capabilities
    tavily_provider = None
    for provider in web_search.providers:
        if hasattr(provider, 'comprehensive_brand_research'):
            tavily_provider = provider
            break
    
    if not tavily_provider:
        print("❌ Enhanced Tavily provider not available")
        return
    
    brand_domain = "specialized.com"
    
    print(f"\n🎯 **Target Brand**: {brand_domain}")
    print("\n📋 **Enhanced Capabilities Available**:")
    print("   ✅ Tavily Map - Sitemap discovery")
    print("   ✅ Tavily Crawl - Intelligent content extraction")
    print("   ✅ Comprehensive Research - Multi-method analysis")
    
    try:
        # Demo 1: Site Mapping
        print(f"\n🗺️ **DEMO 1: Site Structure Mapping**")
        print(f"   Discovering {brand_domain} site structure...")
        
        start_time = time.time()
        sitemap_result = await tavily_provider.map_site(f"https://{brand_domain}")
        map_duration = time.time() - start_time
        
        if sitemap_result:
            print(f"   ✅ Found {sitemap_result.total_pages} pages in {map_duration:.1f}s")
            print(f"   📊 Response time: {sitemap_result.response_time:.2f}s")
            
            # Show sample URLs by category
            categories = tavily_provider._categorize_urls(sitemap_result.urls)
            for category, urls in categories.items():
                print(f"   📁 {category.title()}: {len(urls)} pages")
                if urls:
                    print(f"      Example: {urls[0]}")
        else:
            print("   ❌ Site mapping failed")
        
        # Demo 2: Targeted Crawling
        print(f"\n🕷️ **DEMO 2: Targeted Content Crawling**")
        print(f"   Crawling {brand_domain} for company information...")
        
        start_time = time.time()
        crawl_result = await tavily_provider.crawl_site(
            f"https://{brand_domain}",
            instructions="Find all pages about company history, mission, values, and brand story"
        )
        crawl_duration = time.time() - start_time
        
        if crawl_result:
            print(f"   ✅ Crawled {crawl_result.total_pages} relevant pages in {crawl_duration:.1f}s")
            print(f"   📊 Response time: {crawl_result.response_time:.2f}s")
            
            # Show content analysis
            total_content = sum(len(content) for content in crawl_result.content_by_url.values())
            print(f"   📝 Total content extracted: {total_content:,} characters")
            
            # Show sample URLs
            print(f"   🔗 Sample URLs found:")
            for i, url in enumerate(crawl_result.urls[:3]):
                content_size = len(crawl_result.content_by_url.get(url, ""))
                print(f"      {i+1}. {url} ({content_size:,} chars)")
        else:
            print("   ❌ Targeted crawling failed")
        
        # Demo 3: Comprehensive Research
        print(f"\n🎯 **DEMO 3: Comprehensive Brand Research**")
        print(f"   Running full multi-method research for {brand_domain}...")
        
        start_time = time.time()
        comprehensive_result = await tavily_provider.comprehensive_brand_research(
            brand_domain=brand_domain,
            research_focus=[
                "company history and founding story",
                "mission, vision, values",
                "business model and products",
                "brand positioning"
            ]
        )
        comprehensive_duration = time.time() - start_time
        
        if comprehensive_result and "error" not in comprehensive_result:
            synthesis = comprehensive_result.get("synthesis", {})
            print(f"   ✅ Comprehensive research completed in {comprehensive_duration:.1f}s")
            print(f"   📊 Pages crawled: {synthesis.get('total_pages_crawled', 0)}")
            print(f"   📊 External sources: {synthesis.get('total_external_sources', 0)}")
            print(f"   📊 Data quality: {synthesis.get('data_quality', 'unknown')}")
            print(f"   📊 Research coverage: {', '.join(synthesis.get('research_coverage', []))}")
        else:
            error_msg = comprehensive_result.get("error", "Unknown error") if comprehensive_result else "No results"
            print(f"   ❌ Comprehensive research failed: {error_msg}")
        
        print(f"\n✅ **Tavily Enhanced Demo Complete**")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")


async def demo_progress_tracking():
    """Demo the progress tracking system"""
    print("\n🔄 **DEMO: Progress Tracking System**")
    print("=" * 60)
    
    # Initialize progress tracker
    progress_tracker = get_progress_tracker()
    
    # Add console listener for real-time updates
    console_listener = create_console_listener()
    progress_tracker.add_progress_listener(console_listener)
    
    brand = "specialized.com"
    
    # Demo different types of steps
    print(f"\n📋 **Simulating Multi-Step Research Process for {brand}**")
    
    try:
        # Step 1: Foundation Research
        foundation_step = progress_tracker.create_step(
            step_type=StepType.FOUNDATION_RESEARCH,
            brand=brand,
            phase_name="Foundation Research",
            total_operations=5
        )
        
        progress_tracker.start_step(foundation_step, "Initializing foundation research...")
        await asyncio.sleep(1)
        
        progress_tracker.update_progress(foundation_step, 1, "🔍 Gathering brand data...")
        await asyncio.sleep(2)
        
        progress_tracker.update_progress(foundation_step, 2, "🤖 Analyzing with LLM...")
        await asyncio.sleep(3)
        
        progress_tracker.update_progress(foundation_step, 3, "📝 Generating markdown...")
        await asyncio.sleep(1)
        
        progress_tracker.update_progress(foundation_step, 4, "✅ Quality validation...")
        await asyncio.sleep(1)
        
        progress_tracker.complete_step(
            foundation_step,
            output_files=["foundation_research.md", "foundation_metadata.json"],
            quality_score=0.85
        )
        
        # Step 2: Market Research
        market_step = progress_tracker.create_step(
            step_type=StepType.MARKET_POSITIONING,
            brand=brand,
            phase_name="Market Research",
            total_operations=4
        )
        
        progress_tracker.start_step(market_step, "Starting market analysis...")
        await asyncio.sleep(1)
        
        progress_tracker.update_progress(market_step, 1, "📊 Competitor analysis...")
        await asyncio.sleep(2)
        
        progress_tracker.update_progress(market_step, 2, "🎯 Market positioning...")
        await asyncio.sleep(2)
        
        progress_tracker.update_progress(market_step, 3, "📈 Market size analysis...")
        await asyncio.sleep(1)
        
        progress_tracker.complete_step(
            market_step,
            output_files=["market_research.md"],
            quality_score=0.78
        )
        
        # Step 3: Product Intelligence (with warning)
        product_step = progress_tracker.create_step(
            step_type=StepType.PRODUCT_INTELLIGENCE,
            brand=brand,
            phase_name="Product Intelligence",
            total_operations=3
        )
        
        progress_tracker.start_step(product_step, "Analyzing product catalog...")
        await asyncio.sleep(1)
        
        progress_tracker.update_progress(product_step, 1, "🛍️ Product categorization...")
        await asyncio.sleep(1)
        
        # Add a warning
        progress_tracker.add_warning(product_step, "Limited product data available for accessories category")
        
        progress_tracker.update_progress(product_step, 2, "💰 Pricing analysis...")
        await asyncio.sleep(2)
        
        progress_tracker.complete_step(
            product_step,
            output_files=["product_intelligence.md"],
            quality_score=0.72
        )
        
        # Show live status
        print(f"\n📊 **LIVE STATUS REPORT**")
        progress_tracker.print_live_status(brand)
        
        # Show summary report
        print(f"\n📋 **SUMMARY REPORT**")
        summary = progress_tracker.get_summary_report(brand)
        print(f"Brand: {summary['brand']}")
        print(f"Total Steps: {summary['total_steps']}")
        print(f"Completed: {summary['completed_steps']}")
        print(f"Failed: {summary['failed_steps']}")
        print(f"Average Quality: {summary['average_quality_score']:.2f}" if summary['average_quality_score'] else "N/A")
        print(f"Total Duration: {summary['total_duration_seconds']:.1f}s")
        
        print(f"\n✅ **Progress Tracking Demo Complete**")
        
    except Exception as e:
        print(f"❌ Progress tracking demo error: {e}")


async def demo_integrated_foundation_research():
    """Demo the integrated foundation research with progress tracking"""
    print("\n🏗️ **DEMO: Integrated Foundation Research**")
    print("=" * 60)
    
    try:
        researcher = FoundationResearcher()
        brand = "specialized.com"
        
        print(f"\n🎯 **Target**: {brand}")
        print("📋 **Features**:")
        print("   ✅ Real-time progress tracking")
        print("   ✅ Enhanced Tavily crawl + search")
        print("   ✅ Quality evaluation")
        print("   ✅ Persistent storage")
        print("   ✅ Comprehensive error handling")
        
        # Run foundation research with observability
        start_time = time.time()
        result = await researcher.research(brand, force_refresh=True)
        duration = time.time() - start_time
        
        print(f"\n📊 **RESULTS SUMMARY**")
        print(f"   Duration: {duration:.1f}s")
        print(f"   Quality Score: {result.get('confidence_score', 'N/A'):.2f}" if result.get('confidence_score') else "   Quality Score: N/A")
        print(f"   Research Method: Enhanced Foundation Research")
        
        print(f"\n✅ **Integrated Demo Complete**")
        
    except Exception as e:
        print(f"❌ Integrated demo error: {e}")


async def main():
    """Run all demos"""
    print("🎯 **ENHANCED OBSERVABILITY + TAVILY INTEGRATION DEMO**")
    print("=" * 80)
    print(f"⏰ Start Time: {datetime.now().strftime('%H:%M:%S')}")
    
    # Demo 1: Enhanced Tavily Capabilities
    await demo_tavily_enhanced_capabilities()
    
    # Demo 2: Progress Tracking System  
    await demo_progress_tracking()
    
    # Demo 3: Integrated Foundation Research
    await demo_integrated_foundation_research()
    
    print(f"\n🎉 **ALL DEMOS COMPLETE**")
    print(f"⏰ End Time: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main()) 