#!/usr/bin/env python3
"""
Demo: Quality Evaluation Framework Integration
Shows how the new quality evaluation system works with research phases
"""

import asyncio
import json
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def demo_quality_evaluation():
    """Demonstrate quality evaluation integration"""
    
    print("🎯 Quality Evaluation Framework Demo")
    print("=" * 50)
    
    # Mock research result for demonstration
    mock_research_result = {
        "brand_domain": "specialized.com",
        "content": """# Foundation Intelligence: Specialized

## 1. Company Founding & History
- **Founding Year & Location:** Founded in 1974 in Morgan Hill, California [1]
- **Founder Background:** Mike Sinyard started with importing Italian bike components [1]
- **Key Historical Milestones:** Introduced first production mountain bike in 1981 [2]
- **Evolution & Changes:** Grew from component importer to full bicycle manufacturer [1][2]

## 2. Mission, Vision & Values
- **Mission Statement:** "Innovate and inspire to improve riders' lives" [3]
- **Vision & Goals:** Leading innovation in cycling technology and performance [3]
- **Core Values:** Quality, innovation, rider-focused design [3]
- **Corporate Culture:** Engineering excellence and rider community focus [4]

## Summary
Specialized is a pioneering bicycle company founded in 1974, known for innovation in cycling technology and strong rider community focus.

## Sources
[1] - Company History Page (specialized.com/about)
[2] - Industry Analysis Report (bicyclingnews.com)
[3] - Mission Statement (specialized.com/mission)
[4] - Corporate Culture Analysis (glassdoor.com)
""",
        "confidence_score": 0.85,
        "data_sources_count": 15,
        "search_success_rate": 0.92,
        "detailed_sources": []
    }
    
    # Simulate quality evaluation
    print("\n🔍 Running Quality Evaluation...")
    
    # This would normally be done by the LLM evaluator
    quality_evaluation = {
        "quality_score": 8.5,
        "passes_threshold": True,
        "improvement_feedback": [
            "Consider adding more specific financial information if available",
            "Include more recent strategic initiatives"
        ],
        "criteria_met": {
            "accuracy": True,
            "completeness": True,
            "consistency": True,
            "authenticity": True,
            "actionability": True
        },
        "evaluator_model": "claude-3-5-sonnet",
        "evaluation_timestamp": "2024-12-20T20:16:30.123456",
        "confidence_level": "high",
        "raw_evaluation": "Excellent foundation research with comprehensive coverage..."
    }
    
    # Simulate enhanced metadata structure
    enhanced_metadata = {
        "phase": "foundation",
        "confidence_score": 0.85,
        "data_sources_count": 15,
        "research_metadata": {
            "phase": "foundation",
            "research_duration_seconds": 45.2,
            "timestamp": "2024-12-20T20:16:30.123456Z",
            "cache_expires": "2025-06-20T20:16:30.123456Z",
            "quality_threshold": 8.0,
            "version": "1.0",
            "quality_attempts": 1,
            "max_quality_attempts": 3
        },
        "quality_evaluation": quality_evaluation,
        "quality_warning": False
    }
    
    print(f"✅ Quality Evaluation Complete!")
    print(f"   📊 Quality Score: {quality_evaluation['quality_score']:.1f}/10.0")
    print(f"   🎯 Passes Threshold: {'✅ YES' if quality_evaluation['passes_threshold'] else '❌ NO'}")
    print(f"   🤖 Evaluator Model: {quality_evaluation['evaluator_model']}")
    print(f"   📈 Confidence Level: {quality_evaluation['confidence_level']}")
    
    print(f"\n📋 Criteria Assessment:")
    for criterion, met in quality_evaluation['criteria_met'].items():
        status = "✅" if met else "❌"
        print(f"   {status} {criterion.title()}")
    
    if quality_evaluation['improvement_feedback']:
        print(f"\n💡 Improvement Suggestions:")
        for i, suggestion in enumerate(quality_evaluation['improvement_feedback'], 1):
            print(f"   {i}. {suggestion}")
    
    print(f"\n📁 Enhanced Metadata Structure:")
    print(json.dumps({
        "research_metadata.json": {
            "phase": enhanced_metadata["phase"],
            "confidence_score": enhanced_metadata["confidence_score"],
            "quality_evaluation": enhanced_metadata["quality_evaluation"],
            "quality_warning": enhanced_metadata["quality_warning"]
        }
    }, indent=2))
    
    print(f"\n🔄 Quality Control Workflow:")
    print(f"   1. Research execution (attempt 1/3)")
    print(f"   2. LLM quality evaluation")
    print(f"   3. Threshold check (8.5 >= 8.0) ✅")
    print(f"   4. Save results with quality metadata")
    print(f"   5. Return enhanced research results")
    
    print(f"\n🎯 Benefits of Integrated Quality Evaluation:")
    print(f"   ✅ No separate storage system needed")
    print(f"   ✅ Quality data stored in existing metadata files")
    print(f"   ✅ Automatic feedback loops with improvement suggestions")
    print(f"   ✅ Configurable quality thresholds per phase")
    print(f"   ✅ Multiple LLM evaluator models (Claude 3.5, o3)")
    print(f"   ✅ Quality history tracking in metadata")
    print(f"   ✅ Seamless integration with existing research workflow")


async def demo_feedback_loop():
    """Demonstrate feedback loop with quality improvement"""
    
    print(f"\n🔄 Quality Feedback Loop Demo")
    print("=" * 50)
    
    # Simulate multiple attempts with improvement
    attempts = [
        {
            "attempt": 1,
            "quality_score": 6.2,
            "passes_threshold": False,
            "feedback": [
                "Add more specific founding details",
                "Include leadership background information",
                "Provide clearer mission statement analysis"
            ]
        },
        {
            "attempt": 2,
            "quality_score": 7.8,
            "passes_threshold": False,
            "feedback": [
                "Include more recent company milestones",
                "Add financial scale indicators"
            ]
        },
        {
            "attempt": 3,
            "quality_score": 8.3,
            "passes_threshold": True,
            "feedback": []
        }
    ]
    
    print(f"Quality Threshold: 8.0")
    print(f"Max Attempts: 3\n")
    
    for attempt_data in attempts:
        attempt = attempt_data["attempt"]
        score = attempt_data["quality_score"]
        passes = attempt_data["passes_threshold"]
        feedback = attempt_data["feedback"]
        
        status = "✅ PASSED" if passes else "⚠️ RETRY"
        print(f"🎯 Attempt {attempt}: {score:.1f}/10.0 - {status}")
        
        if feedback:
            print(f"   💡 Improvement suggestions:")
            for suggestion in feedback:
                print(f"      • {suggestion}")
        
        if passes:
            print(f"   🎉 Quality threshold met! Saving results...")
            break
        elif attempt < 3:
            print(f"   🔄 Retrying with improvement context...")
        else:
            print(f"   ⚠️ Max attempts reached. Saving best result with quality warning.")
        
        print()


async def demo_quality_analytics():
    """Demonstrate quality analytics and reporting"""
    
    print(f"\n📊 Quality Analytics Demo")
    print("=" * 50)
    
    # Mock quality history data
    quality_history = [
        {"phase": "foundation", "score": 8.5, "passes": True, "date": "2024-12-20"},
        {"phase": "market_positioning", "score": 7.9, "passes": False, "date": "2024-12-20"},
        {"phase": "product_style", "score": 8.8, "passes": True, "date": "2024-12-20"},
        {"phase": "customer_cultural", "score": 8.2, "passes": True, "date": "2024-12-20"},
        {"phase": "voice_messaging", "score": 7.6, "passes": False, "date": "2024-12-20"}
    ]
    
    total_phases = len(quality_history)
    passed_phases = len([p for p in quality_history if p["passes"]])
    avg_score = sum(p["score"] for p in quality_history) / total_phases
    
    print(f"Brand: specialized.com")
    print(f"Total Phases Evaluated: {total_phases}")
    print(f"Phases Passed: {passed_phases}/{total_phases} ({passed_phases/total_phases*100:.1f}%)")
    print(f"Average Quality Score: {avg_score:.1f}/10.0")
    
    print(f"\nPhase-by-Phase Quality Results:")
    for phase_data in quality_history:
        phase = phase_data["phase"]
        score = phase_data["score"]
        passes = phase_data["passes"]
        date = phase_data["date"]
        
        status = "✅" if passes else "⚠️"
        print(f"   {status} {phase}: {score:.1f}/10.0 ({date})")
    
    print(f"\nQuality Trends:")
    print(f"   📈 Highest: product_style (8.8)")
    print(f"   📉 Lowest: voice_messaging (7.6)")
    print(f"   🎯 Above Threshold: {passed_phases}/{total_phases} phases")


async def main():
    """Run all quality evaluation demos"""
    
    try:
        await demo_quality_evaluation()
        await demo_feedback_loop()
        await demo_quality_analytics()
        
        print(f"\n🎉 Quality Evaluation Demo Complete!")
        print(f"\nNext Steps:")
        print(f"   1. Test with real research phases")
        print(f"   2. Configure Langfuse quality evaluator prompts")
        print(f"   3. Adjust quality thresholds per phase")
        print(f"   4. Monitor quality improvements over time")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 