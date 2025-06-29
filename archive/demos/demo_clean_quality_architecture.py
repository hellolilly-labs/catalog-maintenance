#!/usr/bin/env python3
"""
Demo: Clean Quality Evaluation Architecture
Shows how the new wrapper pattern works with existing researcher classes
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockBaseResearcher:
    """Simplified BaseResearcher to demonstrate the architecture"""
    
    def __init__(self, researcher_name: str, quality_threshold: float = 8.0):
        self.researcher_name = researcher_name
        self.quality_threshold = quality_threshold
        self.enable_quality_evaluation = True
        self.max_quality_attempts = 3
    
    async def research(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Main research method with automatic quality evaluation wrapping"""
        if self.enable_quality_evaluation:
            return await self._research_with_quality_wrapper(force_refresh)
        else:
            return await self._execute_core_research(force_refresh)
    
    async def _research_with_quality_wrapper(self, force_refresh: bool) -> Dict[str, Any]:
        """Quality evaluation wrapper around core research logic"""
        logger.info(f"🎯 Starting quality-controlled research for {self.researcher_name}")
        
        best_result = None
        best_quality_score = 0.0
        improvement_context = None
        
        for attempt in range(1, self.max_quality_attempts + 1):
            logger.info(f"🔄 Quality attempt {attempt}/{self.max_quality_attempts}")
            
            try:
                # Execute core research (always force refresh on retries)
                should_force = force_refresh or (attempt > 1)
                core_result = await self._execute_core_research(
                    force_refresh=should_force,
                    improvement_context=improvement_context
                )
                
                # Run quality evaluation
                quality_evaluation = await self._mock_quality_evaluation(core_result, attempt)
                
                # Add quality evaluation to result
                enhanced_result = core_result.copy()
                enhanced_result["quality_evaluation"] = quality_evaluation
                enhanced_result["quality_attempts"] = attempt
                
                # Check if quality threshold is met
                quality_score = quality_evaluation["quality_score"]
                passes_threshold = quality_score >= self.quality_threshold
                
                if passes_threshold:
                    logger.info(f"✅ Quality threshold met on attempt {attempt}: {quality_score:.1f}/{self.quality_threshold:.1f}")
                    enhanced_result["research_method"] = "quality_controlled_research"
                    return enhanced_result
                
                # Track best result
                if quality_score > best_quality_score:
                    best_result = enhanced_result
                    best_quality_score = quality_score
                
                # Prepare for next attempt
                if attempt < self.max_quality_attempts:
                    improvement_context = quality_evaluation["improvement_feedback"]
                    logger.warning(f"⚠️ Quality below threshold ({quality_score:.1f}/{self.quality_threshold:.1f}). Retrying...")
                    await asyncio.sleep(0.5)  # Brief pause
                
            except Exception as e:
                logger.error(f"❌ Quality attempt {attempt} failed: {e}")
                if attempt == self.max_quality_attempts:
                    raise
        
        # Return best result with warning
        if best_result:
            best_result["quality_warning"] = True
            best_result["research_method"] = "quality_controlled_with_warning"
            return best_result
        else:
            raise RuntimeError("Quality control failed")
    
    async def _execute_core_research(self, force_refresh: bool = False, improvement_context: Optional[List[str]] = None) -> Dict[str, Any]:
        """Core research execution - subclasses can override this"""
        logger.info(f"🔬 Executing core research for {self.researcher_name}")
        
        if improvement_context:
            logger.info(f"💡 Using improvement context: {improvement_context}")
        
        # Simulate research work
        await asyncio.sleep(0.2)
        
        return {
            "content": f"# {self.researcher_name.title()} Research Results\n\nComprehensive analysis...",
            "confidence_score": 0.85,
            "data_sources": 15,
            "research_method": "mock_research"
        }
    
    async def _mock_quality_evaluation(self, result: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        """Mock quality evaluation that improves with attempts"""
        base_scores = {
            1: 6.5,  # First attempt usually needs improvement
            2: 7.8,  # Second attempt better but may not meet threshold
            3: 8.4   # Third attempt usually passes
        }
        
        quality_score = base_scores.get(attempt, 8.0)
        
        feedback = []
        if attempt == 1:
            feedback = [
                "Add more specific details and examples",
                "Include stronger source citations",
                "Improve structural organization"
            ]
        elif attempt == 2:
            feedback = [
                "Add more quantitative data",
                "Strengthen conclusion section"
            ]
        
        return {
            "quality_score": quality_score,
            "passes_threshold": quality_score >= self.quality_threshold,
            "improvement_feedback": feedback,
            "criteria_met": {
                "accuracy": True,
                "completeness": attempt >= 2,
                "consistency": True,
                "authenticity": attempt >= 2,
                "actionability": attempt >= 3
            },
            "evaluator_model": "mock-evaluator",
            "confidence_level": "high" if attempt >= 3 else "medium"
        }


class ExistingFoundationResearcher(MockBaseResearcher):
    """
    Example of existing researcher class that overrides core research
    Shows how quality evaluation automatically applies
    """
    
    def __init__(self):
        super().__init__("foundation", quality_threshold=8.0)
    
    async def _execute_core_research(self, force_refresh: bool = False, improvement_context: Optional[List[str]] = None) -> Dict[str, Any]:
        """Override core research with foundation-specific logic"""
        logger.info(f"🏗️ Executing Foundation-specific research")
        
        # Foundation research specific work
        await asyncio.sleep(0.3)
        
        enhanced_content = "# Foundation Research\n\n## Company History\n- Founded in 1974\n\n## Mission & Values\n- Innovation focus"
        
        if improvement_context:
            enhanced_content += "\n\n## Enhanced Analysis\n"
            for suggestion in improvement_context:
                enhanced_content += f"- {suggestion}\n"
        
        return {
            "content": enhanced_content,
            "confidence_score": 0.88,
            "data_sources": 20,
            "research_method": "foundation_specific_research"
        }


class LegacyResearcher(MockBaseResearcher):
    """
    Example of legacy researcher that overrides the main research method
    This will BYPASS quality evaluation (demonstrates the problem with old approach)
    """
    
    def __init__(self):
        super().__init__("legacy", quality_threshold=8.0)
    
    async def research(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Legacy override that bypasses quality evaluation"""
        logger.warning("⚠️ Legacy researcher bypassing quality evaluation!")
        
        return {
            "content": "# Legacy Research\n\nBasic research without quality control",
            "confidence_score": 0.70,
            "data_sources": 5,
            "research_method": "legacy_bypass"
        }


async def demo_clean_architecture():
    """Demonstrate the clean quality evaluation architecture"""
    
    print("🏗️ Clean Quality Evaluation Architecture Demo")
    print("=" * 60)
    
    # Test 1: Modern researcher using core research override
    print("\n1. 🎯 Modern Researcher (Overrides _execute_core_research)")
    print("-" * 50)
    
    foundation_researcher = ExistingFoundationResearcher()
    result = await foundation_researcher.research(force_refresh=True)
    
    print(f"✅ Research completed!")
    print(f"   📊 Quality Score: {result['quality_evaluation']['quality_score']:.1f}")
    print(f"   🎯 Quality Attempts: {result['quality_attempts']}")
    print(f"   🔧 Research Method: {result['research_method']}")
    print(f"   📝 Content Preview: {result['content'][:80]}...")
    
    # Test 2: Legacy researcher that bypasses quality evaluation
    print("\n2. ⚠️ Legacy Researcher (Overrides main research method)")
    print("-" * 50)
    
    legacy_researcher = LegacyResearcher()
    legacy_result = await legacy_researcher.research(force_refresh=True)
    
    print(f"⚠️ Research completed (bypassed quality evaluation)")
    print(f"   📊 Quality Evaluation: {'Present' if 'quality_evaluation' in legacy_result else 'MISSING'}")
    print(f"   🔧 Research Method: {legacy_result['research_method']}")
    print(f"   📝 Content Preview: {legacy_result['content'][:80]}...")
    
    # Test 3: Quality evaluation disabled
    print("\n3. 🔧 Quality Evaluation Disabled")
    print("-" * 50)
    
    no_quality_researcher = ExistingFoundationResearcher()
    no_quality_researcher.enable_quality_evaluation = False
    no_quality_result = await no_quality_researcher.research(force_refresh=True)
    
    print(f"✅ Research completed (quality evaluation disabled)")
    print(f"   📊 Quality Evaluation: {'Present' if 'quality_evaluation' in no_quality_result else 'Not Applied'}")
    print(f"   🔧 Research Method: {no_quality_result['research_method']}")


async def demo_benefits():
    """Show benefits of the clean architecture"""
    
    print(f"\n🎉 Benefits of Clean Quality Architecture")
    print("=" * 60)
    
    print(f"✅ Clean Separation of Concerns:")
    print(f"   • Quality evaluation logic isolated in wrapper")
    print(f"   • Core research logic remains clean and focused")
    print(f"   • No code duplication between quality/non-quality paths")
    
    print(f"\n✅ Automatic Quality Integration:")
    print(f"   • Existing researchers get quality evaluation automatically")
    print(f"   • No need to modify existing research implementations")
    print(f"   • Quality can be enabled/disabled per researcher")
    
    print(f"\n✅ Flexible Override Pattern:")
    print(f"   • Subclasses override _execute_core_research (not research)")
    print(f"   • Quality wrapper applies to all implementations")
    print(f"   • Legacy code clearly identified if it bypasses quality")
    
    print(f"\n✅ Maintainable Architecture:")
    print(f"   • Single point of quality control logic")
    print(f"   • Easy to add new quality features")
    print(f"   • Clear contract for researcher implementations")
    
    print(f"\n⚠️ Migration Path for Legacy Code:")
    print(f"   • Legacy researchers that override research() are identified")
    print(f"   • Can be migrated to override _execute_core_research instead")
    print(f"   • Automatic quality evaluation then applies")


async def main():
    """Run the clean architecture demo"""
    
    try:
        await demo_clean_architecture()
        await demo_benefits()
        
        print(f"\n🎯 Summary:")
        print(f"   • Quality evaluation wrapper cleanly separates concerns")
        print(f"   • Existing researchers automatically get quality control")
        print(f"   • Legacy code is identified and can be migrated")
        print(f"   • Architecture is maintainable and extensible")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 