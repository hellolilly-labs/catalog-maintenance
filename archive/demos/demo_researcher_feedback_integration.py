#!/usr/bin/env python3
"""
Researcher Feedback Integration Enhancement
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedResearcherExample:
    def __init__(self, brand_domain: str, enable_quality_evaluation: bool = True):
        self.brand_domain = brand_domain
        self.enable_quality_evaluation = enable_quality_evaluation
        self.researcher_name = "Enhanced Researcher"
    
    async def research(self, force_refresh: bool = False, improvement_feedback: Optional[List[str]] = None) -> Dict[str, Any]:
        if self.enable_quality_evaluation:
            return await self._research_with_quality_wrapper(force_refresh, improvement_feedback)
        else:
            return await self._execute_core_research(force_refresh, improvement_feedback)
    
    async def _research_with_quality_wrapper(self, force_refresh: bool, improvement_feedback: Optional[List[str]]) -> Dict[str, Any]:
        logger.info(f"🎯 Quality evaluation enabled for {self.researcher_name}")
        
        result = await self._execute_core_research(force_refresh, improvement_feedback)
        quality_score = 7.0 + (1.5 if improvement_feedback else 0)
        
        result['quality_evaluation'] = {
            "quality_score": quality_score,
            "passes_threshold": quality_score >= 8.0
        }
        return result
    
    async def _execute_core_research(self, force_refresh: bool, improvement_feedback: Optional[List[str]]) -> Dict[str, Any]:
        logger.info(f"🔍 Executing research for {self.brand_domain}")
        
        feedback_context = ""
        if improvement_feedback:
            logger.info(f"📋 Incorporating {len(improvement_feedback)} improvement suggestions")
            feedback_context = " (with improvement feedback)"
        
        return {
            "content": f"Research analysis for {self.brand_domain}{feedback_context}",
            "confidence": 0.75,
            "feedback_incorporated": len(improvement_feedback) if improvement_feedback else 0,
            "timestamp": datetime.now().isoformat()
        }

async def main():
    print("\n" + "="*60)
    print("RESEARCHER FEEDBACK INTEGRATION DEMO")
    print("="*60)
    
    researcher = EnhancedResearcherExample("example.com")
    result = await researcher.research(force_refresh=True)
    
    quality_eval = result.get('quality_evaluation', {})
    print(f"\n📊 Result Quality: {quality_eval.get('quality_score', 0):.1f}/10.0")
    print(f"✅ Threshold Met: {quality_eval.get('passes_threshold', False)}")
    print(f"📝 Feedback Incorporated: {result.get('feedback_incorporated', 0)}")
    
    print("\n🔄 Testing with feedback...")
    result2 = await researcher.research(force_refresh=True, improvement_feedback=["Add more detail", "Include citations"])
    
    quality_eval2 = result2.get('quality_evaluation', {})
    print(f"\n📊 Result Quality (with feedback): {quality_eval2.get('quality_score', 0):.1f}/10.0")
    print(f"✅ Threshold Met: {quality_eval2.get('passes_threshold', False)}")
    print(f"📝 Feedback Incorporated: {result2.get('feedback_incorporated', 0)}")
    
if __name__ == "__main__":
    asyncio.run(main())
