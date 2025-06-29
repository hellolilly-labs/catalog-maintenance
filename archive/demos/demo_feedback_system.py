#!/usr/bin/env python3
"""
Quality Evaluation Feedback System Demonstration

This demo shows how the feedback system works in the quality evaluation framework
and provides migration guidance for researcher classes that override the main research method.

Key Features:
1. How improvement feedback flows through quality evaluation iterations
2. Migration pattern for existing researcher classes that override research()
3. Best practices for implementing custom research methods with quality control
"""

import asyncio
import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our research system
from src.research.base_researcher import BaseResearcher
from src.progress_tracker import ProgressTracker
from src.storage import LocalAccountStorageProvider

class LegacyResearcherProblem(BaseResearcher):
    """
    Example of a PROBLEMATIC researcher class that overrides research() method completely.
    This bypasses the quality evaluation wrapper and feedback system entirely.
    
    ❌ THIS IS THE WRONG WAY - researchers that override research() miss quality control
    """
    
    async def research(self, force_refresh: bool = False, improvement_feedback: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Legacy override of main research method
        ❌ PROBLEM: This completely bypasses quality evaluation!
        """
        logger.warning("🚨 Legacy researcher bypassing quality evaluation - THIS IS BAD!")
        
        # This would be some custom research logic that bypasses quality control
        return {
            "brand": self.brand_domain,
            "content": "Legacy research content with no quality control",
            "quality_score": 0.5,
            "files": [],
            "data_sources": 0,
            "research_method": "legacy_bypass_quality"
        }

class FixedLegacyResearcher(BaseResearcher):
    """
    Example of FIXED researcher class that properly handles improvement feedback.
    
    ✅ CORRECT APPROACH: Override research() but properly handle feedback parameter
    """
    
    async def research(self, force_refresh: bool = False, improvement_feedback: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Properly fixed research method that participates in quality evaluation
        
        ✅ SOLUTION: Accept improvement_feedback parameter and use it in custom logic
        """
        logger.info(f"🔧 Fixed researcher with proper feedback handling")
        
        if improvement_feedback:
            logger.info(f"💡 Incorporating {len(improvement_feedback)} improvement suggestions:")
            for i, suggestion in enumerate(improvement_feedback, 1):
                logger.info(f"   {i}. {suggestion}")
        
        # Enable quality evaluation if configured
        if self.enable_quality_evaluation:
            return await self._research_with_quality_wrapper(force_refresh, improvement_feedback)
        
        # Custom research logic that can use improvement_feedback
        custom_result = await self._custom_research_logic(force_refresh, improvement_feedback)
        
        return custom_result
    
    async def _custom_research_logic(self, force_refresh: bool, improvement_feedback: Optional[List[str]]) -> Dict[str, Any]:
        """Custom research implementation that uses improvement feedback"""
        
        # Simulate some research work
        research_content = "Basic research content"
        
        # Apply improvement feedback if available
        if improvement_feedback:
            research_content += "\n\nIMPROVEMENTS APPLIED:\n"
            for suggestion in improvement_feedback:
                research_content += f"- Applied: {suggestion}\n"
        
        return {
            "brand": self.brand_domain,
            "content": research_content,
            "quality_score": 0.8,
            "files": [],
            "data_sources": 5,
            "research_method": "fixed_custom_with_feedback",
            "feedback_applied": len(improvement_feedback) if improvement_feedback else 0
        }

class ModernResearcher(BaseResearcher):
    """
    Example of MODERN researcher class using the clean architecture.
    
    ✅ BEST PRACTICE: Override _execute_core_research() instead of research()
    This automatically gets quality evaluation and feedback loops.
    """
    
    async def _execute_core_research(self, force_refresh: bool = False, improvement_feedback: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Modern approach: Override _execute_core_research() for automatic quality control
        
        ✅ BEST PRACTICE: Gets quality evaluation and feedback loops automatically
        """
        logger.info(f"🏗️ Modern researcher with automatic quality control")
        
        # The improvement_feedback is automatically injected into data by base class
        data = await self._gather_data()
        
        # Check if improvement feedback is available (automatically added by base class)
        improvement_context = data.get("improvement_context", [])
        if improvement_context:
            logger.info(f"💡 Using {len(improvement_context)} improvement suggestions automatically")
        
        # Simulate research analysis
        research_content = "Modern research with automatic quality control"
        
        return {
            "brand": self.brand_domain,
            "content": research_content,
            "quality_score": 0.85,
            "files": [],
            "data_sources": 15,
            "research_method": "modern_with_auto_quality",
            "improvement_context_available": bool(improvement_context)
        }
    
    async def _gather_data(self) -> Dict[str, Any]:
        """Mock data gathering"""
        return {
            "search_results": [
                {"title": "Sample Result", "snippet": "Sample content", "url": "https://example.com"}
            ],
            "total_sources": 15,
            "search_stats": {"success_rate": 0.8}
        }

async def demonstrate_feedback_system():
    """
    Comprehensive demonstration of the feedback system and migration patterns
    """
    print("\n" + "="*80)
    print("🎯 QUALITY EVALUATION FEEDBACK SYSTEM DEMONSTRATION")
    print("="*80)
    
    # Setup
    progress_tracker = ProgressTracker()
    storage_manager = LocalAccountStorageProvider()
    brand_domain = "example.com"
    
    print("\n📋 TESTING DIFFERENT RESEARCHER PATTERNS:")
    print("-" * 50)
    
    # Test 1: Legacy Problematic Researcher
    print("\n1️⃣ TESTING LEGACY PROBLEMATIC RESEARCHER (bypasses quality)")
    print("   ❌ This pattern breaks quality evaluation entirely")
    
    legacy_problem = LegacyResearcherProblem(
        brand_domain=brand_domain,
        researcher_name="legacy_problem",
        step_type="research",
        quality_threshold=8.0
    )
    legacy_problem.enable_quality_evaluation = True  # This won't work due to override
    legacy_problem.progress_tracker = progress_tracker
    legacy_problem.storage_manager = storage_manager
    
    try:
        result1 = await legacy_problem.research(force_refresh=True)
        print(f"   Result: {result1.get('research_method', 'unknown')}")
        print(f"   Quality Score: {result1.get('quality_score', 'N/A')}")
        print(f"   ❌ No quality evaluation - BYPASSED!")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Fixed Legacy Researcher  
    print("\n2️⃣ TESTING FIXED LEGACY RESEARCHER (proper feedback handling)")
    print("   ✅ This pattern works with quality evaluation and feedback")
    
    legacy_fixed = FixedLegacyResearcher(
        brand_domain=brand_domain,
        researcher_name="legacy_fixed",
        step_type="research",
        quality_threshold=8.0
    )
    legacy_fixed.enable_quality_evaluation = True
    legacy_fixed.progress_tracker = progress_tracker
    legacy_fixed.storage_manager = storage_manager
    
    # Test with external feedback (simulating human/external feedback)
    external_feedback = [
        "Add more specific brand personality details",
        "Include competitor comparison data",
        "Improve actionability of recommendations"
    ]
    
    try:
        result2 = await legacy_fixed.research(
            force_refresh=True, 
            improvement_feedback=external_feedback
        )
        print(f"   Result: {result2.get('research_method', 'unknown')}")
        print(f"   Quality Score: {result2.get('quality_score', 'N/A')}")
        print(f"   Feedback Applied: {result2.get('feedback_applied', 0)} suggestions")
        print(f"   ✅ Quality evaluation and feedback working!")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Modern Researcher (Best Practice)
    print("\n3️⃣ TESTING MODERN RESEARCHER (automatic quality control)")
    print("   ✅ This is the recommended pattern - override _execute_core_research()")
    
    modern = ModernResearcher(
        brand_domain=brand_domain,
        researcher_name="modern",
        step_type="research", 
        quality_threshold=8.0
    )
    modern.enable_quality_evaluation = True
    modern.progress_tracker = progress_tracker
    modern.storage_manager = storage_manager
    
    try:
        result3 = await modern.research(force_refresh=True)
        print(f"   Result: {result3.get('research_method', 'unknown')}")
        print(f"   Quality Score: {result3.get('quality_score', 'N/A')}")
        print(f"   ✅ Automatic quality evaluation and feedback loops!")
        
        # Check if quality evaluation data was added
        if "quality_evaluation" in result3:
            quality_eval = result3["quality_evaluation"]
            print(f"   Quality Evaluation Score: {quality_eval.get('quality_score', 'N/A')}")
            print(f"   Passes Threshold: {quality_eval.get('passes_threshold', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

def print_migration_guide():
    """Print comprehensive migration guide for existing researcher classes"""
    
    print("\n" + "="*80)
    print("📖 MIGRATION GUIDE FOR EXISTING RESEARCHER CLASSES")
    print("="*80)
    
    print("""
🎯 PROBLEM: Researcher classes that override research() bypass quality evaluation

If your researcher class looks like this:

❌ PROBLEMATIC PATTERN:
```python
class MyResearcher(BaseResearcher):
    async def research(self, force_refresh: bool = False) -> Dict[str, Any]:
        # Custom logic here
        return {"content": "research result"}
```

This completely bypasses quality evaluation and feedback loops!

✅ SOLUTION 1: ADD FEEDBACK PARAMETER (for existing overrides)
```python
class MyResearcher(BaseResearcher):
    async def research(self, force_refresh: bool = False, improvement_feedback: Optional[List[str]] = None) -> Dict[str, Any]:
        # Enable quality evaluation if configured
        if self.enable_quality_evaluation:
            return await self._research_with_quality_wrapper(force_refresh, improvement_feedback)
        
        # Your custom logic with feedback support
        return await self._custom_research_logic(force_refresh, improvement_feedback)
    
    async def _custom_research_logic(self, force_refresh: bool, improvement_feedback: Optional[List[str]]) -> Dict[str, Any]:
        # Use improvement_feedback in your custom logic
        if improvement_feedback:
            # Apply suggestions to improve research quality
            pass
        
        return {"content": "improved research result"}
```

✅ SOLUTION 2: USE MODERN PATTERN (recommended for new code)
```python
class MyResearcher(BaseResearcher):
    async def _execute_core_research(self, force_refresh: bool = False, improvement_feedback: Optional[List[str]] = None) -> Dict[str, Any]:
        # Your custom logic here
        # improvement_feedback is automatically available in data["improvement_context"]
        data = await self._gather_data()
        
        # Base class automatically injects improvement_feedback into data
        improvement_context = data.get("improvement_context", [])
        
        # Use improvement context in your analysis
        analysis = await self._analyze_data(data)
        
        return analysis
```

🔄 FEEDBACK LOOP FLOW:
1. Research runs with quality evaluation enabled
2. If quality score < threshold, LLM generates improvement suggestions  
3. improvement_feedback parameter contains these suggestions
4. Research reruns (up to 3 attempts) with improvement context
5. Your custom logic can use this feedback to improve results

💡 KEY BENEFITS:
- Automatic quality control for all researchers
- Iterative improvement through LLM feedback
- Clean separation of research logic and quality evaluation
- Backward compatibility with existing code

🚨 IMPORTANT NOTES:
- Always accept improvement_feedback parameter in research() overrides
- Use the feedback to actually improve your research logic
- Modern pattern (_execute_core_research override) is preferred
- Quality evaluation can be disabled per researcher if needed
""")

async def main():
    """Main demonstration"""
    try:
        await demonstrate_feedback_system()
        print_migration_guide()
        
        print("\n" + "="*80)
        print("✅ FEEDBACK SYSTEM DEMONSTRATION COMPLETE")
        print("="*80)
        print("\n🎯 KEY TAKEAWAYS:")
        print("1. ✅ Modern researchers should override _execute_core_research() for automatic quality control")
        print("2. 🔧 Legacy researchers that override research() must accept improvement_feedback parameter")
        print("3. 💡 Feedback loops enable iterative improvement of research quality")
        print("4. 🎯 Quality evaluation can be enabled/disabled per researcher")
        print("5. 📈 Up to 3 automatic improvement attempts with LLM feedback")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 