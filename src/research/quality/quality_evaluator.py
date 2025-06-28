#!/usr/bin/env python3
"""
Unified Quality Evaluator

Single quality evaluator class that ALL researchers use.
Simple flag to enable/disable web search enhancement.
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.llm.simple_factory import LLMFactory
from src.web_search import get_web_search_engine

logger = logging.getLogger(__name__)

class QualityEvaluator:
    """
    Single quality evaluator class for ALL researchers
    
    Features:
    - Standard LLM-based quality evaluation (always)
    - Optional web search enhancement (flag-controlled)
    - Unified interface for all researcher classes
    """
    
    def __init__(self, enable_web_search: bool = True):
        """
        Initialize quality evaluator
        
        Args:
            enable_web_search: Whether to enable web search enhancement
        """
        self.enable_web_search = enable_web_search
        
        # Initialize web search if enabled
        if self.enable_web_search:
            self.web_search_engine = get_web_search_engine()
            if self.web_search_engine:
                logger.info("🚀 Quality evaluator with web search enhancement enabled")
            else:
                logger.warning("⚠️ Web search requested but not available - using standard evaluation only")
                self.enable_web_search = False
        else:
            self.web_search_engine = None
            logger.info("📊 Quality evaluator with standard evaluation only")
    
    async def evaluate_with_search_recommendations(
        self,
        research_result: Dict[str, Any],
        phase_name: str,
        brand_domain: str,
        quality_threshold: float = 8.0
    ) -> Dict[str, Any]:
        """
        Evaluate research quality with optional web search enhancement
        
        This is the main entry point - used by all researchers
        """
        
        # Step 1: Always do standard LLM evaluation first
        evaluation_result = await self._standard_llm_evaluation(
            research_result, phase_name, brand_domain, quality_threshold
        )
        
        # Step 2: If web search enabled and quality below threshold, enhance
        if (self.enable_web_search and 
            self.web_search_engine and 
            not evaluation_result.get('passes_threshold', False)):
            
            logger.info(f"🔍 Quality below threshold ({evaluation_result['quality_score']:.1f}/{quality_threshold:.1f}) - enhancing with web search")
            
            enhanced_result = await self._enhance_with_web_search(
                research_result, evaluation_result, phase_name, brand_domain, quality_threshold
            )
            
            return enhanced_result
        
        # Return standard evaluation
        return evaluation_result
    
    async def evaluate(self, research_result: Dict[str, Any], quality_threshold: float = 8.0) -> Dict[str, Any]:
        """
        Simple evaluation interface (for compatibility)
        """
        return await self._standard_llm_evaluation(research_result, "unknown", "unknown", quality_threshold)
    
    async def _standard_llm_evaluation(
        self,
        research_result: Dict[str, Any],
        phase_name: str,
        brand_domain: str,
        quality_threshold: float
    ) -> Dict[str, Any]:
        """
        Standard LLM-based quality evaluation
        """
        
        try:
            # Prepare evaluation prompt
            content = research_result.get("content", "")
            confidence = research_result.get("confidence_score", 0.0)
            
            evaluation_prompt = self._get_evaluation_prompt()
            
            template_vars = {
                "phase_name": phase_name,
                "brand_domain": brand_domain,
                "content": content,
                "confidence": confidence,
                "quality_threshold": quality_threshold
            }
            
            for var, value in template_vars.items():
                evaluation_prompt["system"] = evaluation_prompt["system"].replace(f"{{{{{var}}}}}", str(value))
                evaluation_prompt["user"] = evaluation_prompt["user"].replace(f"{{{{{var}}}}}", str(value))
            
            # Run LLM evaluation
            response = await LLMFactory.chat_completion(
                task="quality_evaluation_advanced",
                system=evaluation_prompt["system"],
                messages=[{
                    "role": "user",
                    "content": evaluation_prompt["user"]
                }],
                temperature=0.1
            )
            
            # Parse evaluation results
            evaluation_content = response.get("content", "")
            parsed_evaluation = self._parse_evaluation_response(evaluation_content)
            
            quality_score = parsed_evaluation.get("quality_score", 5.0)
            
            return {
                "quality_score": quality_score,
                "passes_threshold": quality_score >= quality_threshold,
                "improvement_feedback": parsed_evaluation.get("improvement_feedback", []),
                "criteria_met": parsed_evaluation.get("criteria_met", {}),
                "evaluator_model": "claude-3-5-sonnet",
                "evaluation_timestamp": datetime.now().isoformat(),
                "confidence_level": parsed_evaluation.get("confidence_level", "medium"),
                "evaluation_method": "standard_llm_evaluation",
                "raw_evaluation": evaluation_content[:1000]
            }
            
        except Exception as e:
            logger.error(f"❌ Standard evaluation failed: {e}")
            return self._create_fallback_evaluation(quality_threshold)
    
    async def _enhance_with_web_search(
        self,
        research_result: Dict[str, Any],
        evaluation_result: Dict[str, Any],
        phase_name: str,
        brand_domain: str,
        quality_threshold: float
    ) -> Dict[str, Any]:
        """
        Enhance evaluation with web search recommendations
        """
        
        try:
            # Analyze gaps and recommend searches
            search_recommendations = await self._analyze_gaps_and_recommend_searches(
                research_result, evaluation_result, phase_name, brand_domain
            )
            
            if not search_recommendations:
                logger.warning("No search recommendations generated")
                return evaluation_result
            
            # Execute recommended searches
            search_results = await self._execute_searches(search_recommendations)
            
            if not search_results:
                logger.warning("No search results obtained")
                return evaluation_result
            
            # Enhance evaluation with search results
            enhanced_evaluation = await self._integrate_search_results(
                evaluation_result, search_results, quality_threshold
            )
            
            return enhanced_evaluation
            
        except Exception as e:
            logger.error(f"❌ Web search enhancement failed: {e}")
            return evaluation_result
    
    async def _analyze_gaps_and_recommend_searches(
        self,
        research_result: Dict[str, Any],
        evaluation_result: Dict[str, Any],
        phase_name: str,
        brand_domain: str
    ) -> List[Dict[str, Any]]:
        """
        Analyze research gaps and recommend specific searches
        """
        
        brand_name = brand_domain.replace('.com', '').replace('.', ' ').title()
        content = research_result.get('content', '')[:3000]
        quality_score = evaluation_result['quality_score']
        improvement_feedback = evaluation_result['improvement_feedback']
        
        gap_analysis_prompt = f"""
Analyze this {phase_name} research for {brand_name} ({brand_domain}) and recommend specific web searches to improve quality.

CURRENT RESEARCH CONTENT:
{content}

QUALITY EVALUATION:
Score: {quality_score:.1f}/10.0
Issues: {improvement_feedback}

Based on the research gaps, recommend 3-5 specific web searches that would fill missing information.

SEARCH QUERY GUIDELINES:
- Use domain name + brand name for best targeting: {brand_domain} "{brand_name}" [search terms]
- Be specific about information gaps to fill
- Focus on factual, verifiable information

Respond with JSON only:
{{
  "searches": [
    {{
      "query": "specific search query using domain + brand name format",
      "purpose": "what information gap this fills",
      "priority": "high|medium|low",
      "max_results": 5
    }}
  ]
}}
"""
        
        try:
            response = await LLMFactory.chat_completion(
                task="gap_analysis",
                system="You are an expert research analyst. Recommend targeted web searches to fill specific information gaps. Always respond with valid JSON only.",
                messages=[{
                    "role": "user",
                    "content": gap_analysis_prompt
                }],
                temperature=0.2
            )
            
            content = response.get("content", "").strip()
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            
            search_data = json.loads(content)
            searches = search_data.get("searches", [])
            
            logger.info(f"📋 Recommended {len(searches)} searches for quality improvement")
            return searches
            
        except Exception as e:
            logger.warning(f"Gap analysis failed: {e}")
            return []
    
    async def _execute_searches(self, search_recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute recommended web searches
        """
        
        search_results = []
        
        for search_rec in search_recommendations:
            query = search_rec.get("query", "")
            max_results = search_rec.get("max_results", 5)
            
            if not query:
                continue
            
            try:
                logger.info(f"🔍 Searching: {query}")
                
                results = await asyncio.wait_for(
                    self.web_search_engine.search(query=query, max_results=max_results),
                    timeout=30.0
                )
                
                if results.get("results"):
                    search_results.append({
                        "query": query,
                        "purpose": search_rec.get("purpose", ""),
                        "priority": search_rec.get("priority", "medium"),
                        "results": results["results"],
                        "result_count": len(results["results"]),
                        "success": True
                    })
                    
                    logger.info(f"✅ Found {len(results['results'])} results for: {query}")
                
                await asyncio.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Search failed for '{query}': {e}")
        
        logger.info(f"🔍 Completed {len(search_results)} successful searches")
        return search_results
    
    async def _integrate_search_results(
        self,
        original_evaluation: Dict[str, Any],
        search_results: List[Dict[str, Any]],
        quality_threshold: float
    ) -> Dict[str, Any]:
        """
        Integrate search results and enhance quality evaluation
        """
        
        # Calculate enhanced quality score
        original_score = original_evaluation['quality_score']
        
        # Base enhancement for having additional data
        base_enhancement = 0.5
        search_count_bonus = min(1.0, len(search_results) * 0.2)  # Up to 1.0 for 5+ searches
        result_count_bonus = min(0.5, sum(sr.get('result_count', 0) for sr in search_results) * 0.02)  # Up to 0.5
        
        total_enhancement = base_enhancement + search_count_bonus + result_count_bonus
        enhanced_score = min(10.0, original_score + total_enhancement)
        
        logger.info(f"📊 Quality enhanced: {original_score:.1f} → {enhanced_score:.1f} (+{enhanced_score - original_score:.1f})")
        
        # Generate enhanced feedback
        enhanced_feedback = await self._generate_enhanced_feedback(
            original_evaluation['improvement_feedback'], search_results
        )
        
        return {
            **original_evaluation,
            "quality_score": enhanced_score,
            "passes_threshold": enhanced_score >= quality_threshold,
            "improvement_feedback": enhanced_feedback,
            "search_results": search_results,
            "search_enhancement": {
                "original_score": original_score,
                "enhanced_score": enhanced_score,
                "score_improvement": enhanced_score - original_score,
                "search_count": len(search_results),
                "total_results": sum(sr.get("result_count", 0) for sr in search_results)
            },
            "evaluation_method": "web_search_enhanced"
        }
    
    async def _generate_enhanced_feedback(
        self,
        original_feedback: List[str],
        search_results: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate enhanced feedback incorporating search results
        """
        
        if not search_results:
            return original_feedback
        
        # Create summary of additional data found
        search_summary = []
        for search in search_results:
            search_summary.append(f"- {search['query']}: {search['result_count']} results found")
        
        enhanced_feedback = [
            "Research enhanced with additional web search data:",
            *search_summary,
            "Incorporate the additional search findings to address these improvements:",
            *original_feedback[:3]  # Include top 3 original suggestions
        ]
        
        return enhanced_feedback
    
    def _get_evaluation_prompt(
        self, 
    ) -> Dict[str, str]:
        """
        Get quality evaluation prompt
        """
        
        system_prompt = """You are an expert quality evaluator for {{phase_name}} brand research.

Evaluate research quality on a scale of 0.0 to 10.0 based on these criteria:

EVALUATION CRITERIA:
- Accuracy: Information is factual and well-sourced (0-2 points)
- Completeness: All required elements present (0-2 points)  
- Consistency: No contradictions or conflicts (0-2 points)
- Authenticity: Captures genuine brand voice (0-2 points)
- Actionability: Provides implementable insights (0-2 points)

QUALITY STANDARDS:
- 9.0-10.0: Exceptional quality, production ready
- 8.0-8.9: High quality, minor improvements possible
- 7.0-7.9: Good quality, some improvements needed
- 6.0-6.9: Acceptable quality, significant improvements needed
- Below 6.0: Poor quality, major rework required

Respond in JSON format with your evaluation."""

        user_prompt = """Evaluate this {{phase_name}} research for {{brand_domain}}:

RESEARCH CONTENT:
{{content}}

CONTEXT:
- Current confidence score: {{confidence}}
- Quality threshold: {{quality_threshold}}/10.0

Provide evaluation in this JSON format:
{
    "quality_score": 8.2,
    "criteria_met": {
        "accuracy": true,
        "completeness": true, 
        "consistency": true,
        "authenticity": true,
        "actionability": true
    },
    "improvement_feedback": [
        "Specific suggestion 1",
        "Specific suggestion 2"
    ],
    "confidence_level": "high"
}"""

        return {
            "system": system_prompt,
            "user": user_prompt
        }
    
    def _parse_evaluation_response(self, evaluation_text: str) -> Dict[str, Any]:
        """
        Parse LLM evaluation response
        """
        try:
            # Clean up response
            content = evaluation_text.strip()
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            
            evaluation = json.loads(content)
            return {
                "quality_score": evaluation.get("quality_score", 5.0),
                "criteria_met": evaluation.get("criteria_met", {}),
                "improvement_feedback": evaluation.get("improvement_feedback", []),
                "confidence_level": evaluation.get("confidence_level", "medium")
            }
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse evaluation JSON: {e}")
            return {
                "quality_score": 5.0,
                "criteria_met": {},
                "improvement_feedback": ["Evaluation parsing failed - manual review recommended"],
                "confidence_level": "low"
            }
    
    def _create_fallback_evaluation(self, quality_threshold: float) -> Dict[str, Any]:
        """
        Create fallback evaluation when everything else fails
        """
        return {
            "quality_score": 6.0,  # Neutral but below most thresholds
            "passes_threshold": 6.0 >= quality_threshold,
            "improvement_feedback": ["Quality evaluation system encountered errors - manual review recommended"],
            "criteria_met": {},
            "evaluator_model": "fallback_system",
            "evaluation_timestamp": datetime.now().isoformat(),
            "confidence_level": "low",
            "evaluation_method": "fallback_evaluation"
        } 