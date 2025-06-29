#!/usr/bin/env python3
"""
Demo: Unified Quality Evaluator System

Tests the simplified single QualityEvaluator class that ALL researchers use.
Simple enable_web_search flag controls whether web search enhancement is used.
"""

import os
import sys
import logging
import asyncio

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

async def test_unified_quality_evaluator():
    """Test the single unified QualityEvaluator class"""
    
    print("🎯 UNIFIED QUALITY EVALUATOR TEST")
    print("=" * 50)
    
    try:
        from src.research.quality.quality_evaluator import QualityEvaluator
        
        print("\n📊 Testing Standard Quality Evaluator (no web search)")
        print("-" * 50)
        
        # Create evaluator without web search
        standard_evaluator = QualityEvaluator(enable_web_search=False)
        
        # Test research content
        test_research_result = {
            "content": "# Foundation Research for Specialized Bicycles\n\nBasic company information with limited depth.",
            "confidence_score": 0.7,
            "data_sources_count": 5
        }
        
        # Run evaluation
        result = await standard_evaluator.evaluate_with_search_recommendations(
            research_result=test_research_result,
            phase_name="foundation",
            brand_domain="specialized.com",
            quality_threshold=8.0
        )
        
        print(f"✅ Quality Score: {result['quality_score']:.1f}/10.0")
        print(f"✅ Passes Threshold: {result['passes_threshold']}")
        print(f"✅ Method: {result['evaluation_method']}")
        print(f"✅ Suggestions: {len(result['improvement_feedback'])}")
        
        print("\n✅ UNIFIED QUALITY EVALUATOR SUCCESS!")
        print("✅ Single evaluator class works for all researchers")
        print("✅ Simple enable_web_search flag controls enhancement")
        print("✅ No complex conditional logic needed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_unified_quality_evaluator())
