"""
LLM-based Quality Evaluator for Research Phases
Integrates with existing research_metadata.json structure
"""

import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
from src.llm.simple_factory import LLMFactory
from src.llm.prompt_manager import PromptManager

logger = logging.getLogger(__name__)

@dataclass
class QualityEvaluation:
    """Quality evaluation result that integrates with research metadata"""
    
    quality_score: float  # 0.0 - 10.0 scale
    passes_threshold: bool
    improvement_feedback: List[str]
    criteria_met: Dict[str, bool]
    evaluator_model: str
    evaluation_timestamp: datetime
    confidence_level: str  # "high", "medium", "low"
    
    def to_metadata_dict(self) -> Dict[str, Any]:
        """Convert to dict for inclusion in research_metadata.json"""
        return {
            "quality_evaluation": {
                "quality_score": self.quality_score,
                "passes_threshold": self.passes_threshold,
                "improvement_feedback": self.improvement_feedback,
                "criteria_met": self.criteria_met,
                "evaluator_model": self.evaluator_model,
                "evaluation_timestamp": self.evaluation_timestamp.isoformat(),
                "confidence_level": self.confidence_level
            }
        }


class PhaseQualityEvaluator:
    """LLM-based quality evaluator that integrates with existing research phases"""
    
    def __init__(self):
        self.prompt_manager = PromptManager()
        
        # Quality thresholds by phase (0.0 - 10.0 scale)
        self.quality_thresholds = {
            "foundation": 8.0,
            "market_positioning": 7.5, 
            "product_style": 8.0,
            "customer_cultural": 8.5,
            "voice_messaging": 8.0,
            "interview_synthesis": 9.0,  # High threshold for human insights
            "linearity_analysis": 7.5,
            "research_integration": 8.5
        }
        
        # Evaluation models by phase complexity
        self.evaluator_models = {
            "foundation": "claude-3-5-sonnet",
            "market_positioning": "claude-3-5-sonnet",
            "product_style": "claude-3-5-sonnet",
            "customer_cultural": "o3",  # More complex customer psychology
            "voice_messaging": "claude-3-5-sonnet",
            "interview_synthesis": "o3",  # Complex human insight synthesis
            "linearity_analysis": "o3",  # Complex cross-phase analysis
            "research_integration": "o3"  # Most complex synthesis task
        }
    
    async def evaluate_phase_quality(
        self, 
        phase_name: str, 
        research_content: str,
        brand_domain: str,
        current_confidence_score: float,
        context: Optional[Dict[str, Any]] = None
    ) -> QualityEvaluation:
        """
        Evaluate research phase quality using LLM judge
        Returns evaluation that can be integrated into research_metadata.json
        """
        
        logger.info(f"🎯 Evaluating {phase_name} quality for {brand_domain}")
        
        try:
            # Get quality threshold for phase
            quality_threshold = self.quality_thresholds.get(phase_name, 8.0)
            
            # Select evaluator model
            evaluator_model = self.evaluator_models.get(phase_name, "claude-3-5-sonnet")
            
            # Get evaluator prompt
            evaluator_prompt = await self._get_evaluator_prompt(phase_name)
            
            # Prepare evaluation context
            evaluation_context = {
                "phase_name": phase_name,
                "brand_domain": brand_domain,
                "research_content": research_content[:8000],  # Limit for token management
                "current_confidence_score": current_confidence_score,
                "quality_threshold": quality_threshold
            }
            
            if context:
                evaluation_context.update(context)
            
            # Run LLM evaluation
            evaluation_result = await self._run_llm_evaluation(
                evaluator_prompt,
                evaluation_context,
                evaluator_model
            )
            
            # Parse evaluation results
            quality_score = self._extract_quality_score(evaluation_result)
            improvement_feedback = self._extract_improvement_suggestions(evaluation_result)
            criteria_analysis = self._analyze_criteria_compliance(evaluation_result)
            
            # Determine confidence level
            confidence_level = self._determine_confidence_level(quality_score, criteria_analysis)
            
            # Create evaluation object
            evaluation = QualityEvaluation(
                quality_score=quality_score,
                passes_threshold=quality_score >= quality_threshold,
                improvement_feedback=improvement_feedback,
                criteria_met=criteria_analysis,
                evaluator_model=evaluator_model,
                evaluation_timestamp=datetime.now(),
                confidence_level=confidence_level
            )
            
            # Log evaluation results
            status_icon = "✅" if evaluation.passes_threshold else "⚠️"
            logger.info(
                f"{status_icon} {phase_name} quality: {quality_score:.1f}/{quality_threshold:.1f} "
                f"({confidence_level} confidence)"
            )
            
            if not evaluation.passes_threshold and improvement_feedback:
                logger.warning(f"💡 Improvement suggestions: {'; '.join(improvement_feedback[:2])}")
            
            return evaluation
            
        except Exception as e:
            logger.error(f"❌ Quality evaluation failed for {phase_name}: {e}")
            
            # Return minimal evaluation with failure indication
            return QualityEvaluation(
                quality_score=0.0,
                passes_threshold=False,
                improvement_feedback=[f"Evaluation failed: {str(e)}"],
                criteria_met={},
                evaluator_model="evaluation_failed",
                evaluation_timestamp=datetime.now(),
                confidence_level="low"
            )
    
    async def _get_evaluator_prompt(self, phase_name: str) -> Dict[str, str]:
        """Get phase-specific evaluator prompt (Langfuse or fallback)"""
        
        try:
            # Try to get phase-specific prompt from Langfuse
            prompt_key = f"liddy/catalog/quality/{phase_name}_evaluator"
            prompt_client = await self.prompt_manager.get_prompt(prompt_key)
            
            if prompt_client and prompt_client.prompt:
                return {
                    "system": prompt_client.prompt[0]["content"],
                    "user_template": prompt_client.prompt[1]["content"] if len(prompt_client.prompt) > 1 else ""
                }
            
        except Exception as e:
            logger.warning(f"Using fallback evaluator prompt for {phase_name}: {e}")
        
        return self._get_fallback_evaluator_prompt(phase_name)
    
    def _get_fallback_evaluator_prompt(self, phase_name: str) -> Dict[str, str]:
        """Fallback evaluator prompts when Langfuse is unavailable"""
        
        system_prompt = f"""You are an expert quality evaluator for {phase_name} brand research.

Evaluate the research quality on a scale of 0.0 to 10.0 based on these criteria:

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

        user_template = """Evaluate this {phase_name} research for {brand_domain}:

RESEARCH CONTENT:
{research_content}

CONTEXT:
- Current confidence score: {current_confidence_score}
- Quality threshold: {quality_threshold}/10.0

Provide evaluation in this JSON format:
{{
    "quality_score": 8.2,
    "criteria_met": {{
        "accuracy": true,
        "completeness": true, 
        "consistency": true,
        "authenticity": true,
        "actionability": true
    }},
    "improvement_feedback": [
        "Specific suggestion 1",
        "Specific suggestion 2"
    ],
    "confidence_level": "high"
}}"""

        return {
            "system": system_prompt,
            "user_template": user_template
        }
    
    async def _run_llm_evaluation(
        self, 
        evaluator_prompt: Dict[str, str],
        evaluation_context: Dict[str, Any],
        evaluator_model: str
    ) -> Dict[str, Any]:
        """Run LLM evaluation with the specified model"""
        
        # Format user prompt with context
        user_prompt = evaluator_prompt["user_template"].format(**evaluation_context)
        
        # Map our model names to LLMFactory task names
        task_mapping = {
            "claude-3-5-sonnet": "quality_evaluation",
            "o3": "quality_evaluation_advanced"
        }
        
        task_name = task_mapping.get(evaluator_model, "quality_evaluation")
        
        try:
            response = await LLMFactory.chat_completion(
                task=task_name,
                system=evaluator_prompt["system"],
                messages=[{
                    "role": "user",
                    "content": user_prompt
                }],
                temperature=0.1,  # Low temperature for consistent evaluation
            )
            
            return response
            
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            raise
    
    def _extract_quality_score(self, evaluation_result: Dict[str, Any]) -> float:
        """Extract quality score from LLM evaluation result"""
        
        content = evaluation_result.get("content", "")
        
        try:
            # Try to parse as JSON first
            if content.strip().startswith("{"):
                evaluation_json = json.loads(content)
                return float(evaluation_json.get("quality_score", 5.0))
            
            # Fallback: look for score patterns in text
            import re
            score_patterns = [
                r"quality[_\s]*score[:\s]*(\d+\.?\d*)",
                r"score[:\s]*(\d+\.?\d*)",
                r"(\d+\.?\d*)\s*/\s*10"
            ]
            
            for pattern in score_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    score = float(match.group(1))
                    return min(10.0, max(0.0, score))  # Clamp to 0-10 range
            
            logger.warning("Could not extract quality score, using default 5.0")
            return 5.0
            
        except Exception as e:
            logger.error(f"Error extracting quality score: {e}")
            return 5.0
    
    def _extract_improvement_suggestions(self, evaluation_result: Dict[str, Any]) -> List[str]:
        """Extract improvement suggestions from evaluation result"""
        
        content = evaluation_result.get("content", "")
        
        try:
            # Try JSON parsing first
            if content.strip().startswith("{"):
                evaluation_json = json.loads(content)
                return evaluation_json.get("improvement_feedback", [])
            
            # Fallback: extract suggestions from text
            suggestions = []
            lines = content.split('\n')
            
            for line in lines:
                line = line.strip()
                if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                    suggestion = line[1:].strip()
                    if suggestion and len(suggestion) > 10:  # Filter out short/empty suggestions
                        suggestions.append(suggestion)
                        if len(suggestions) >= 5:  # Limit suggestions
                            break
            
            return suggestions[:5]  # Return top 5 suggestions
            
        except Exception as e:
            logger.error(f"Error extracting improvement suggestions: {e}")
            return ["Unable to extract improvement suggestions"]
    
    def _analyze_criteria_compliance(self, evaluation_result: Dict[str, Any]) -> Dict[str, bool]:
        """Analyze which quality criteria were met"""
        
        content = evaluation_result.get("content", "")
        
        try:
            # Try JSON parsing first
            if content.strip().startswith("{"):
                evaluation_json = json.loads(content)
                return evaluation_json.get("criteria_met", {})
            
            # Fallback: simple criteria analysis
            criteria = {
                "accuracy": True,      # Default to true for basic implementation
                "completeness": True,
                "consistency": True,
                "authenticity": True,
                "actionability": True
            }
            
            # Basic keyword analysis for negative indicators
            content_lower = content.lower()
            negative_indicators = ['poor', 'weak', 'lacking', 'incomplete', 'inaccurate', 'insufficient']
            
            for criterion in criteria.keys():
                if criterion in content_lower:
                    # Check for negative indicators near the criterion
                    criterion_context = self._get_context_around_word(content_lower, criterion, 100)
                    has_negative = any(neg in criterion_context for neg in negative_indicators)
                    criteria[criterion] = not has_negative
            
            return criteria
            
        except Exception as e:
            logger.error(f"Error analyzing criteria compliance: {e}")
            return {}
    
    def _determine_confidence_level(
        self, 
        quality_score: float, 
        criteria_analysis: Dict[str, bool]
    ) -> str:
        """Determine confidence level for the evaluation"""
        
        criteria_met_count = sum(1 for met in criteria_analysis.values() if met)
        total_criteria = max(1, len(criteria_analysis))
        criteria_ratio = criteria_met_count / total_criteria
        
        if quality_score >= 8.0 and criteria_ratio >= 0.8:
            return "high"
        elif quality_score >= 6.0 and criteria_ratio >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _get_context_around_word(self, text: str, word: str, context_length: int = 50) -> str:
        """Get context around a specific word in text"""
        
        index = text.find(word)
        if index == -1:
            return ""
        
        start = max(0, index - context_length)
        end = min(len(text), index + len(word) + context_length)
        
        return text[start:end]


# Factory function to get the evaluator
def get_phase_quality_evaluator() -> PhaseQualityEvaluator:
    """Get a configured phase quality evaluator instance"""
    return PhaseQualityEvaluator() 