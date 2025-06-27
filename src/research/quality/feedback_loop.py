"""
Feedback Loop Manager for Research Phase Quality Control
Implements re-runs with improvement suggestions per ROADMAP Section 4.4
"""

import logging
from typing import Dict, Any, Optional, List
from .phase_evaluator import get_phase_quality_evaluator, QualityEvaluation

logger = logging.getLogger(__name__)

class FeedbackLoopManager:
    """Manages feedback loops and re-runs for research phase quality control"""
    
    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts
        self.quality_evaluator = get_phase_quality_evaluator()
    
    async def research_phase_with_quality_control(
        self,
        researcher_instance,
        phase_name: str,
        brand_domain: str,
        force_refresh: bool = False,
        improvement_context: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute research phase with quality control and feedback loops
        
        Args:
            researcher_instance: The specific researcher instance (e.g., FoundationResearcher)
            phase_name: Name of the research phase
            brand_domain: Brand domain being researched
            force_refresh: Whether to force refresh cached results
            improvement_context: Previous improvement suggestions to incorporate
            
        Returns:
            Research results with quality evaluation metadata
        """
        
        logger.info(f"🔄 Starting quality-controlled research for {phase_name} ({brand_domain})")
        
        quality_threshold = self.quality_evaluator.quality_thresholds.get(phase_name, 8.0)
        best_result = None
        best_quality_score = 0.0
        
        for attempt in range(1, self.max_attempts + 1):
            logger.info(f"🎯 Attempt {attempt}/{self.max_attempts} for {phase_name}")
            
            try:
                # Add improvement context if this is a retry
                context = {}
                if improvement_context and attempt > 1:
                    context["improvement_suggestions"] = improvement_context
                    context["attempt_number"] = attempt
                    logger.info(f"💡 Incorporating {len(improvement_context)} improvement suggestions")
                
                # Execute research phase
                phase_result = await researcher_instance.research(
                    force_refresh=True  # Always refresh on retries
                )
                
                # Extract research content for evaluation
                research_content = phase_result.get("content", "")
                current_confidence = phase_result.get("quality_score", 0.0)
                
                # Run quality evaluation
                quality_evaluation = await self.quality_evaluator.evaluate_phase_quality(
                    phase_name=phase_name,
                    research_content=research_content,
                    brand_domain=brand_domain,
                    current_confidence_score=current_confidence,
                    context=context
                )
                
                # Update phase result with quality evaluation
                phase_result.update(quality_evaluation.to_metadata_dict())
                
                # Check if quality threshold is met
                if quality_evaluation.passes_threshold:
                    logger.info(
                        f"✅ {phase_name} passed quality check on attempt {attempt}: "
                        f"{quality_evaluation.quality_score:.1f}/{quality_threshold:.1f}"
                    )
                    return phase_result
                
                # Track best result in case all attempts fail
                if quality_evaluation.quality_score > best_quality_score:
                    best_result = phase_result
                    best_quality_score = quality_evaluation.quality_score
                
                # Prepare for next attempt if quality threshold not met
                if attempt < self.max_attempts:
                    logger.warning(
                        f"⚠️ {phase_name} quality below threshold "
                        f"({quality_evaluation.quality_score:.1f}/{quality_threshold:.1f}). "
                        f"Preparing retry {attempt + 1}/{self.max_attempts}"
                    )
                    
                    # Use improvement feedback for next attempt
                    improvement_context = quality_evaluation.improvement_feedback
                    
                    # Brief pause before retry
                    import asyncio
                    await asyncio.sleep(2)
                
                else:
                    # Final attempt failed
                    logger.error(
                        f"❌ {phase_name} failed quality check after {self.max_attempts} attempts. "
                        f"Best score: {quality_evaluation.quality_score:.1f}/{quality_threshold:.1f}"
                    )
                    
                    # Mark as quality warning but proceed with best result
                    if best_result:
                        best_result["quality_warning"] = True
                        best_result["final_attempts"] = self.max_attempts
                        best_result["best_quality_score"] = best_quality_score
                        
                        # Add final quality evaluation to best result
                        final_quality_eval = quality_evaluation.to_metadata_dict()
                        final_quality_eval["quality_evaluation"]["quality_warning"] = True
                        final_quality_eval["quality_evaluation"]["total_attempts"] = self.max_attempts
                        best_result.update(final_quality_eval)
                        
                        return best_result
                    else:
                        return phase_result  # Return last attempt if no best result
                
            except Exception as e:
                logger.error(f"❌ Research attempt {attempt} failed for {phase_name}: {e}")
                
                if attempt == self.max_attempts:
                    # Final attempt failed with exception
                    return {
                        "error": f"Research failed after {self.max_attempts} attempts: {str(e)}",
                        "quality_warning": True,
                        "final_attempts": self.max_attempts
                    }
                
                # Continue to next attempt
                continue
        
        # Should not reach here, but return error if it does
        return {
            "error": "Unexpected end of feedback loop",
            "quality_warning": True
        }
    
    def format_improvement_context(self, improvement_suggestions: List[str]) -> str:
        """Format improvement suggestions for inclusion in research prompts"""
        
        if not improvement_suggestions:
            return ""
        
        formatted_context = "\n\n**IMPROVEMENT CONTEXT (from previous quality evaluation):**\n"
        for i, suggestion in enumerate(improvement_suggestions[:5], 1):
            formatted_context += f"{i}. {suggestion}\n"
        
        formatted_context += "\nPlease address these specific improvement areas in your research.\n"
        
        return formatted_context
    
    async def evaluate_cross_phase_consistency(
        self,
        brand_domain: str,
        completed_phases: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate consistency across multiple completed research phases
        
        Args:
            brand_domain: Brand being evaluated
            completed_phases: Dict of {phase_name: research_results}
            
        Returns:
            Cross-phase consistency evaluation
        """
        
        logger.info(f"🔍 Evaluating cross-phase consistency for {brand_domain}")
        
        try:
            # Extract key information from each phase
            phase_summaries = {}
            for phase_name, phase_data in completed_phases.items():
                content = phase_data.get("content", "")
                quality_score = phase_data.get("quality_score", 0.0)
                
                # Extract first 500 chars as summary
                summary = content[:500] + "..." if len(content) > 500 else content
                
                phase_summaries[phase_name] = {
                    "summary": summary,
                    "quality_score": quality_score
                }
            
            # Use LLM to evaluate cross-phase consistency
            consistency_prompt = self._get_consistency_evaluation_prompt()
            
            evaluation_context = {
                "brand_domain": brand_domain,
                "phase_summaries": phase_summaries,
                "total_phases": len(completed_phases)
            }
            
            # Format context for LLM
            context_text = f"Brand: {brand_domain}\n\n"
            for phase_name, phase_info in phase_summaries.items():
                context_text += f"**{phase_name.upper()}:**\n{phase_info['summary']}\n\n"
            
            # Run consistency evaluation (simplified for now)
            consistency_score = self._calculate_basic_consistency_score(completed_phases)
            
            consistency_evaluation = {
                "consistency_score": consistency_score,
                "phases_evaluated": list(completed_phases.keys()),
                "evaluation_timestamp": "datetime.now().isoformat()",
                "consistency_issues": self._identify_consistency_issues(completed_phases),
                "overall_assessment": "high" if consistency_score >= 8.0 else "medium" if consistency_score >= 6.0 else "low"
            }
            
            logger.info(f"✅ Cross-phase consistency: {consistency_score:.1f}/10.0")
            
            return consistency_evaluation
            
        except Exception as e:
            logger.error(f"❌ Cross-phase consistency evaluation failed: {e}")
            return {
                "error": str(e),
                "consistency_score": 0.0,
                "overall_assessment": "failed"
            }
    
    def _get_consistency_evaluation_prompt(self) -> Dict[str, str]:
        """Get prompt for cross-phase consistency evaluation"""
        
        return {
            "system": """You are an expert brand research consistency evaluator. 
            
Evaluate the consistency and alignment across multiple research phases for the same brand.
Look for contradictions, alignment issues, and overall coherence in the brand narrative.

Rate consistency on a scale of 0.0 to 10.0 where:
- 9.0-10.0: Excellent consistency, fully aligned narrative
- 7.0-8.9: Good consistency, minor alignment issues
- 5.0-6.9: Moderate consistency, some contradictions
- Below 5.0: Poor consistency, major contradictions""",
            
            "user_template": """Evaluate cross-phase research consistency for {brand_domain}:

{phase_summaries}

Identify any contradictions, misalignments, or consistency issues across these research phases.
Provide a consistency score (0.0-10.0) and list specific issues if found."""
        }
    
    def _calculate_basic_consistency_score(self, completed_phases: Dict[str, Dict[str, Any]]) -> float:
        """Calculate basic consistency score based on quality scores and content overlap"""
        
        if len(completed_phases) < 2:
            return 10.0  # Single phase is perfectly consistent with itself
        
        # Average quality scores as a proxy for consistency
        quality_scores = [
            phase_data.get("quality_score", 0.0) 
            for phase_data in completed_phases.values()
        ]
        
        avg_quality = sum(quality_scores) / len(quality_scores)
        quality_variance = sum((score - avg_quality) ** 2 for score in quality_scores) / len(quality_scores)
        
        # Lower variance indicates more consistent quality across phases
        consistency_score = avg_quality * (1 - min(0.3, quality_variance / 10))
        
        return min(10.0, max(0.0, consistency_score))
    
    def _identify_consistency_issues(self, completed_phases: Dict[str, Dict[str, Any]]) -> List[str]:
        """Identify potential consistency issues (basic implementation)"""
        
        issues = []
        
        # Check for significant quality score variations
        quality_scores = [
            phase_data.get("quality_score", 0.0) 
            for phase_data in completed_phases.values()
        ]
        
        if quality_scores:
            min_quality = min(quality_scores)
            max_quality = max(quality_scores)
            
            if max_quality - min_quality > 2.0:
                issues.append(f"Significant quality variation across phases ({min_quality:.1f} to {max_quality:.1f})")
        
        # Check for missing quality evaluations
        missing_quality = [
            phase_name for phase_name, phase_data in completed_phases.items()
            if "quality_evaluation" not in phase_data
        ]
        
        if missing_quality:
            issues.append(f"Missing quality evaluations for phases: {', '.join(missing_quality)}")
        
        return issues


# Factory function to get the feedback loop manager
def get_feedback_loop_manager(max_attempts: int = 3) -> FeedbackLoopManager:
    """Get a configured feedback loop manager instance"""
    return FeedbackLoopManager(max_attempts=max_attempts) 