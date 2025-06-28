#!/usr/bin/env python3
"""
Demo: Complete Enhanced Quality Evaluation System

This demo showcases the complete enhanced quality evaluation framework:
1. Standard quality evaluation
2. Gap analysis and search recommendations  
3. Automatic search execution
4. Search result integration and enhanced feedback
5. Quality score improvement through additional data

Features demonstrated:
- EnhancedQualityEvaluator with web search capabilities
- Automatic gap analysis when quality is below threshold
- Targeted web search execution for missing information
- Search result integration into improvement feedback
- Quality score enhancement based on additional data
"""

import asyncio
import logging
import json
from datetime import datetime
from src.research.market_positioning_research import MarketPositioningResearcher
from src.research.quality.enhanced_evaluator import EnhancedQualityEvaluator
from src.web_search import get_web_search_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def demo_enhanced_quality_evaluation():
    """
    Demonstrate the complete enhanced quality evaluation system
    """
    
    print("🚀 Enhanced Quality Evaluation with Web Search Demo")
    print("=" * 60)
    
    brand_domain = "specialized.com"
    
    try:
        # Initialize web search engine
        web_search_engine = get_web_search_engine()
        if not web_search_engine or not web_search_engine.is_available():
            print("❌ Web search engine not available - cannot run enhanced demo")
            return
        
        print(f"✅ Web search engine available: {type(web_search_engine).__name__}")
        
        # Initialize enhanced evaluator
        enhanced_evaluator = EnhancedQualityEvaluator(web_search_engine=web_search_engine)
        print("✅ Enhanced quality evaluator initialized")
        
        # Initialize researcher with quality evaluation enabled
        researcher = MarketPositioningResearcher(
            brand_domain=brand_domain,
            enable_quality_evaluation=True
        )
        
        print(f"\n📊 Starting market positioning research for {brand_domain}")
        print("   This will demonstrate the complete enhanced quality workflow...")
        
        # Execute research with quality evaluation
        start_time = datetime.now()
        research_result = await researcher.research(force_refresh=True)
        duration = (datetime.now() - start_time).total_seconds()
        
        print(f"\n✅ Research completed in {duration:.1f}s")
        
        # Display results
        print("\n" + "=" * 60)
        print("📋 RESEARCH RESULTS SUMMARY")
        print("=" * 60)
        
        print(f"Brand: {research_result.get('brand', 'Unknown')}")
        print(f"Quality Score: {research_result.get('quality_score', 0.0):.1f}/10.0")
        print(f"Data Sources: {research_result.get('data_sources', 0)}")
        print(f"Research Method: {research_result.get('research_method', 'Unknown')}")
        
        # Display quality evaluation details
        quality_eval = research_result.get('quality_evaluation', {})
        if quality_eval:
            print(f"\n🎯 QUALITY EVALUATION:")
            print(f"   Score: {quality_eval.get('quality_score', 0.0):.1f}/10.0")
            print(f"   Passes Threshold: {quality_eval.get('passes_threshold', False)}")
            print(f"   Evaluation Method: {quality_eval.get('evaluation_method', 'Unknown')}")
            print(f"   Confidence Level: {quality_eval.get('confidence_level', 'Unknown')}")
            
            # Display search enhancement details if available
            search_enhancement = quality_eval.get('search_enhancement')
            if search_enhancement:
                print(f"\n🔍 SEARCH ENHANCEMENT:")
                print(f"   Original Score: {search_enhancement.get('original_score', 0.0):.1f}")
                print(f"   Enhanced Score: {search_enhancement.get('enhanced_score', 0.0):.1f}")
                print(f"   Score Improvement: +{search_enhancement.get('score_improvement', 0.0):.1f}")
                print(f"   Search Count: {search_enhancement.get('search_count', 0)}")
                print(f"   Total Results Found: {search_enhancement.get('total_results', 0)}")
            
            # Display search results if available
            search_results = quality_eval.get('search_results', [])
            if search_results:
                print(f"\n🔍 SEARCH RESULTS ({len(search_results)} searches executed):")
                for i, search in enumerate(search_results, 1):
                    print(f"   {i}. Query: {search.get('query', 'Unknown')}")
                    print(f"      Purpose: {search.get('purpose', 'Unknown')}")
                    print(f"      Priority: {search.get('priority', 'Unknown')}")
                    print(f"      Results Found: {search.get('result_count', 0)}")
                    print(f"      Duration: {search.get('search_duration', 0.0):.1f}s")
            
            # Display improvement feedback
            improvement_feedback = quality_eval.get('improvement_feedback', [])
            if improvement_feedback:
                print(f"\n💡 IMPROVEMENT FEEDBACK ({len(improvement_feedback)} suggestions):")
                for i, feedback in enumerate(improvement_feedback, 1):
                    print(f"   {i}. {feedback}")
        
        # Display quality attempts if multiple were made
        quality_attempts = research_result.get('quality_attempts', 1)
        max_attempts = research_result.get('max_quality_attempts', 3)
        if quality_attempts > 1:
            print(f"\n🔄 QUALITY ITERATIONS:")
            print(f"   Attempts Made: {quality_attempts}/{max_attempts}")
            
            if research_result.get('quality_warning'):
                print(f"   ⚠️ Quality Warning: Threshold not met after {quality_attempts} attempts")
                print(f"   Final Score: {research_result.get('final_quality_score', 0.0):.1f}")
        
        # Display content preview
        content = research_result.get('content', '')
        if content:
            print(f"\n📄 CONTENT PREVIEW (first 500 chars):")
            print("-" * 40)
            print(content[:500] + "..." if len(content) > 500 else content)
            print("-" * 40)
        
        print(f"\n✅ Enhanced quality evaluation demo completed successfully!")
        
        # Summary statistics
        print(f"\n📊 DEMO SUMMARY:")
        print(f"   Total Duration: {duration:.1f}s")
        print(f"   Quality Threshold Met: {quality_eval.get('passes_threshold', False)}")
        print(f"   Search Enhancement Used: {'Yes' if search_enhancement else 'No'}")
        print(f"   Additional Searches: {len(search_results) if search_results else 0}")
        print(f"   Research Quality: {'High' if quality_eval.get('quality_score', 0) >= 8.0 else 'Medium' if quality_eval.get('quality_score', 0) >= 6.0 else 'Low'}")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise

async def demo_manual_enhanced_evaluation():
    """
    Demonstrate manual enhanced evaluation on existing research
    """
    
    print("\n🧪 Manual Enhanced Evaluation Demo")
    print("=" * 40)
    
    # Initialize web search engine
    web_search_engine = get_web_search_engine()
    if not web_search_engine or not web_search_engine.is_available():
        print("❌ Web search engine not available")
        return
    
    # Initialize enhanced evaluator
    enhanced_evaluator = EnhancedQualityEvaluator(web_search_engine=web_search_engine)
    
    # Mock research result with intentionally low quality
    mock_research = {
        "content": """
        # Market Positioning for Specialized.com
        
        Specialized is a bike company. They make good bikes for people who like cycling.
        Their bikes are popular and well-made. They have different types of bikes.
        
        ## Competition
        They compete with other bike companies.
        
        ## Customers  
        Their customers are cyclists.
        """,
        "confidence_score": 0.6,
        "data_sources": 5
    }
    
    print("📋 Evaluating intentionally low-quality research...")
    print("   This should trigger search recommendations and enhancement...")
    
    # Run enhanced evaluation
    evaluation_result = await enhanced_evaluator.evaluate_with_search_recommendations(
        research_result=mock_research,
        phase_name="market_positioning",
        brand_domain="specialized.com",
        quality_threshold=8.0
    )
    
    # Display evaluation results
    print(f"\n🎯 EVALUATION RESULTS:")
    print(f"   Quality Score: {evaluation_result.get('quality_score', 0.0):.1f}/10.0")
    print(f"   Passes Threshold: {evaluation_result.get('passes_threshold', False)}")
    print(f"   Evaluation Method: {evaluation_result.get('evaluation_method', 'Unknown')}")
    
    # Show search recommendations
    recommended_searches = evaluation_result.get('recommended_searches', [])
    if recommended_searches:
        print(f"\n🔍 RECOMMENDED SEARCHES ({len(recommended_searches)}):")
        for i, search in enumerate(recommended_searches, 1):
            print(f"   {i}. {search.get('query', 'Unknown')}")
            print(f"      Purpose: {search.get('purpose', 'Unknown')}")
            print(f"      Priority: {search.get('priority', 'Unknown')}")
    
    # Show search results if executed
    search_results = evaluation_result.get('search_results', [])
    if search_results:
        print(f"\n✅ SEARCH EXECUTION RESULTS:")
        total_results = sum(sr.get('result_count', 0) for sr in search_results)
        successful_searches = len([sr for sr in search_results if sr.get('success', False)])
        print(f"   Successful Searches: {successful_searches}/{len(search_results)}")
        print(f"   Total Results Found: {total_results}")
        
        # Show search enhancement
        search_enhancement = evaluation_result.get('search_enhancement')
        if search_enhancement:
            original_score = search_enhancement.get('original_score', 0.0)
            enhanced_score = search_enhancement.get('enhanced_score', 0.0)
            improvement = search_enhancement.get('score_improvement', 0.0)
            
            print(f"\n📈 QUALITY IMPROVEMENT:")
            print(f"   Original Score: {original_score:.1f}")
            print(f"   Enhanced Score: {enhanced_score:.1f}")
            print(f"   Improvement: +{improvement:.1f} points")
    
    # Show enhanced feedback
    enhanced_feedback = evaluation_result.get('improvement_feedback', [])
    if enhanced_feedback:
        print(f"\n💡 ENHANCED FEEDBACK ({len(enhanced_feedback)} suggestions):")
        for i, feedback in enumerate(enhanced_feedback, 1):
            print(f"   {i}. {feedback}")

async def main():
    """Run all demos"""
    
    print("🎯 Enhanced Quality Evaluation System Demo")
    print("=" * 50)
    print("This demo showcases:")
    print("✓ Automatic quality evaluation with web search enhancement")
    print("✓ Gap analysis and targeted search recommendations")
    print("✓ Search execution and result integration")
    print("✓ Quality score improvement through additional data")
    print("✓ Enhanced feedback generation with search context")
    print()
    
    # Run the full system demo
    await demo_enhanced_quality_evaluation()
    
    # Run manual evaluation demo
    await demo_manual_enhanced_evaluation()
    
    print("\n🎉 All enhanced quality evaluation demos completed!")

if __name__ == "__main__":
    asyncio.run(main()) 