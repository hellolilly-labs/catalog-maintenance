"""
AI Decision Tracking & Transparency Framework

Provides standardized patterns for tracking AI decisions with:
- Confidence scoring
- Reasoning documentation  
- Evidence collection
- Method tracking
- Decision audit trails

Apply this pattern across the entire stack for complete AI transparency.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Standardized confidence levels for AI decisions"""
    VERY_LOW = "very_low"      # 0.0 - 0.3
    LOW = "low"                # 0.3 - 0.5  
    MEDIUM = "medium"          # 0.5 - 0.7
    HIGH = "high"              # 0.7 - 0.9
    VERY_HIGH = "very_high"    # 0.9 - 1.0


@dataclass
class AIDecision:
    """
    Standardized AI decision structure with complete transparency
    
    This provides the "paper trail" pattern for all AI decisions
    throughout the system.
    """
    
    # Core decision data
    decision_type: str                    # e.g., "brand_vertical_detection"
    result: Any                          # The actual decision/result
    confidence: float                    # 0.0 - 1.0 confidence score
    
    # Transparency & reasoning
    reasoning: str                       # Why this decision was made
    evidence: List[str]                  # Supporting evidence
    method: str                          # Analysis method used
    
    # Metadata & tracking
    timestamp: datetime                  # When decision was made
    duration_seconds: float              # How long analysis took
    model_used: Optional[str] = None     # LLM model if applicable
    temperature: Optional[float] = None  # LLM temperature if applicable
    
    # Multi-source analysis
    analysis_methods: List[str] = None   # Multiple methods used
    method_weights: Dict[str, float] = None  # Weighting of methods
    
    # Alternative options
    alternatives: List[Dict[str, Any]] = None  # Other options considered
    consensus_level: Optional[str] = None      # Agreement level across methods
    
    # Error handling & fallbacks
    errors: List[str] = None             # Any errors encountered
    fallback_used: bool = False          # Whether fallback method was used
    
    # Additional context
    context: Dict[str, Any] = None       # Extra context data
    
    def __post_init__(self):
        """Initialize default values and validate"""
        if self.analysis_methods is None:
            self.analysis_methods = []
        if self.method_weights is None:
            self.method_weights = {}
        if self.alternatives is None:
            self.alternatives = []
        if self.errors is None:
            self.errors = []
        if self.context is None:
            self.context = {}
            
        # Validate confidence score
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get human-readable confidence level"""
        if self.confidence < 0.3:
            return ConfidenceLevel.VERY_LOW
        elif self.confidence < 0.5:
            return ConfidenceLevel.LOW
        elif self.confidence < 0.7:
            return ConfidenceLevel.MEDIUM
        elif self.confidence < 0.9:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result_dict = asdict(self)
        result_dict["confidence_level"] = self.confidence_level.value
        result_dict["timestamp"] = self.timestamp.isoformat()
        return result_dict
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    def log_decision(self, log_level: int = logging.INFO):
        """Log this decision for audit trails"""
        log_msg = (f"AI Decision: {self.decision_type} → {self.result} "
                  f"(confidence: {self.confidence:.2f}, method: {self.method})")
        logger.log(log_level, log_msg)


class AIDecisionTracker:
    """
    Base class for components that make AI decisions
    
    Provides standardized methods for tracking and documenting
    AI decision-making processes.
    """
    
    def __init__(self):
        self.decisions: List[AIDecision] = []
        
    def start_decision(self, decision_type: str) -> datetime:
        """Start timing a decision process"""
        return datetime.now()
    
    def record_decision(
        self,
        decision_type: str,
        result: Any,
        confidence: float,
        reasoning: str,
        evidence: List[str],
        method: str,
        start_time: datetime,
        model_used: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AIDecision:
        """
        Record an AI decision with complete transparency
        
        Args:
            decision_type: Type of decision (e.g., "brand_vertical_detection")
            result: The actual decision result
            confidence: Confidence score 0.0-1.0
            reasoning: Why this decision was made
            evidence: Supporting evidence list
            method: Analysis method used
            start_time: When decision process started
            model_used: LLM model if applicable
            temperature: LLM temperature if applicable
            **kwargs: Additional fields for AIDecision
            
        Returns:
            AIDecision object for further use
        """
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        decision = AIDecision(
            decision_type=decision_type,
            result=result,
            confidence=confidence,
            reasoning=reasoning,
            evidence=evidence,
            method=method,
            timestamp=end_time,
            duration_seconds=duration,
            model_used=model_used,
            temperature=temperature,
            **kwargs
        )
        
        # Store decision for audit trail
        self.decisions.append(decision)
        
        # Log decision
        decision.log_decision()
        
        return decision
    
    def get_decision_history(self, decision_type: Optional[str] = None) -> List[AIDecision]:
        """Get history of decisions, optionally filtered by type"""
        if decision_type:
            return [d for d in self.decisions if d.decision_type == decision_type]
        return self.decisions.copy()
    
    def get_decision_summary(self) -> Dict[str, Any]:
        """Get summary statistics of decisions made"""
        if not self.decisions:
            return {"total_decisions": 0}
        
        total = len(self.decisions)
        avg_confidence = sum(d.confidence for d in self.decisions) / total
        avg_duration = sum(d.duration_seconds for d in self.decisions) / total
        
        decision_types = {}
        confidence_levels = {}
        methods = {}
        
        for decision in self.decisions:
            # Count by decision type
            decision_types[decision.decision_type] = decision_types.get(decision.decision_type, 0) + 1
            
            # Count by confidence level
            level = decision.confidence_level.value
            confidence_levels[level] = confidence_levels.get(level, 0) + 1
            
            # Count by method
            methods[decision.method] = methods.get(decision.method, 0) + 1
        
        return {
            "total_decisions": total,
            "average_confidence": round(avg_confidence, 3),
            "average_duration_seconds": round(avg_duration, 2),
            "decision_types": decision_types,
            "confidence_levels": confidence_levels,
            "methods_used": methods,
            "latest_decision": self.decisions[-1].timestamp.isoformat()
        }


class EnhancedBrandVerticalDetector(AIDecisionTracker):
    """
    Enhanced brand vertical detector with AI decision tracking
    
    Example of how to apply the paper trail pattern to existing components.
    """
    
    def __init__(self):
        super().__init__()
        from configs.settings import get_settings
        self.settings = get_settings()
        self._cache = {}
        self._web_search_engine = None
    
    def _get_web_search_engine(self):
        """Lazy initialization of web search engine"""
        if self._web_search_engine is None:
            try:
                from src.web_search import get_web_search_engine
                self._web_search_engine = get_web_search_engine()
            except ImportError:
                logger.warning("Web search module not available")
                self._web_search_engine = None
        return self._web_search_engine
    
    async def detect_brand_vertical_with_tracking(
        self, 
        brand_domain: str, 
        product_sample: Optional[Any] = None
    ) -> AIDecision:
        """
        Detect brand vertical with complete AI decision tracking
        
        This replaces the original detect_brand_vertical method but adds
        the paper trail pattern for full transparency.
        """
        
        start_time = self.start_decision("brand_vertical_detection")
        
        # Check cache first
        if brand_domain in self._cache:
            cached_result = self._cache[brand_domain]
            
            return self.record_decision(
                decision_type="brand_vertical_detection",
                result=cached_result["detected_vertical"],
                confidence=cached_result["confidence"],
                reasoning=f"Retrieved from cache: {cached_result.get('reasoning', 'Previously analyzed')}",
                evidence=[f"Cached result from previous analysis"],
                method="cache_lookup",
                start_time=start_time,
                context={"cached_data": cached_result, "brand_domain": brand_domain}
            )
        
        # Perform multi-source analysis
        analysis_methods = []
        all_evidence = []
        method_results = {}
        
        try:
            # Method 1: Web search analysis
            web_search_result = await self._analyze_brand_via_web_search_tracked(brand_domain, start_time)
            if web_search_result:
                analysis_methods.append("web_search")
                method_results["web_search"] = web_search_result
                all_evidence.extend(web_search_result.get("evidence", []))
            
            # Method 2: Product catalog sampling
            product_analysis_result = await self._analyze_brand_via_product_sampling_tracked(brand_domain, start_time)
            if product_analysis_result:
                analysis_methods.append("product_sampling")
                method_results["product_sampling"] = product_analysis_result
                all_evidence.extend(product_analysis_result.get("evidence", []))
            
            # Method 3: Single product fallback
            if not analysis_methods and product_sample:
                single_product_result = await self._analyze_single_product_tracked(product_sample, start_time)
                if single_product_result:
                    analysis_methods.append("single_product_fallback")
                    method_results["single_product_fallback"] = single_product_result
                    all_evidence.extend(single_product_result.get("evidence", []))
            
            # Synthesize results if we have multiple methods
            if len(analysis_methods) > 1:
                synthesis_result = await self._synthesize_vertical_analysis_tracked(method_results, start_time)
                final_vertical = synthesis_result["detected_vertical"]
                final_confidence = synthesis_result["confidence"]
                final_reasoning = synthesis_result["synthesis_reasoning"]
                method_weights = synthesis_result.get("method_weights", {})
                consensus_level = synthesis_result.get("consensus_level", "unknown")
            
            elif len(analysis_methods) == 1:
                # Single method result
                method_name = analysis_methods[0]
                result_data = method_results[method_name]
                final_vertical = result_data["detected_vertical"]
                final_confidence = result_data["confidence"]
                final_reasoning = f"Single method analysis: {result_data.get('reasoning', method_name)}"
                method_weights = {method_name: 1.0}
                consensus_level = "single_method"
            
            else:
                # No methods available - fallback
                final_vertical = "general"
                final_confidence = 0.1
                final_reasoning = "No analysis methods available, using general fallback"
                method_weights = {}
                consensus_level = "fallback"
                analysis_methods = ["fallback"]
            
            # Cache the result
            cache_data = {
                "detected_vertical": final_vertical,
                "confidence": final_confidence,
                "analysis_methods": analysis_methods,
                "reasoning": final_reasoning
            }
            self._cache[brand_domain] = cache_data
            
            # Record the decision with complete transparency
            return self.record_decision(
                decision_type="brand_vertical_detection",
                result=final_vertical,
                confidence=final_confidence,
                reasoning=final_reasoning,
                evidence=all_evidence,
                method="multi_source_synthesis" if len(analysis_methods) > 1 else analysis_methods[0],
                start_time=start_time,
                analysis_methods=analysis_methods,
                method_weights=method_weights,
                consensus_level=consensus_level,
                alternatives=[{
                    "method": method,
                    "result": data["detected_vertical"],
                    "confidence": data["confidence"]
                } for method, data in method_results.items()],
                context={
                    "brand_domain": brand_domain,
                    "method_results": method_results,
                    "cached_for_future": True
                }
            )
            
        except Exception as e:
            logger.error(f"Error in brand vertical detection: {e}")
            
            return self.record_decision(
                decision_type="brand_vertical_detection",
                result="general",
                confidence=0.0,
                reasoning=f"Analysis failed due to error: {str(e)}",
                evidence=[f"Error encountered: {str(e)}"],
                method="error_fallback",
                start_time=start_time,
                errors=[str(e)],
                fallback_used=True,
                context={"brand_domain": brand_domain, "error": str(e)}
            )
    
    # Placeholder methods for the tracked analysis methods
    async def _analyze_brand_via_web_search_tracked(self, brand_domain: str, start_time: datetime) -> Optional[Dict[str, Any]]:
        """Web search analysis with decision tracking"""
        # This would call the existing web search method but add tracking
        # Implementation would be similar to the existing method
        return None
    
    async def _analyze_brand_via_product_sampling_tracked(self, brand_domain: str, start_time: datetime) -> Optional[Dict[str, Any]]:
        """Product sampling analysis with decision tracking"""  
        # This would call the existing product sampling method but add tracking
        return None
    
    async def _analyze_single_product_tracked(self, product: Any, start_time: datetime) -> Optional[Dict[str, Any]]:
        """Single product analysis with decision tracking"""
        # This would call the existing single product method but add tracking
        return None
    
    async def _synthesize_vertical_analysis_tracked(self, method_results: Dict[str, Any], start_time: datetime) -> Dict[str, Any]:
        """Multi-source synthesis with decision tracking"""
        # This would call the existing synthesis method but add tracking
        return {
            "detected_vertical": "cycling",
            "confidence": 0.9,
            "synthesis_reasoning": "Example synthesis",
            "method_weights": {"web_search": 0.6, "product_sampling": 0.4},
            "consensus_level": "high"
        }


# Factory function for easy integration
def create_ai_decision_tracker() -> AIDecisionTracker:
    """Create a new AI decision tracker instance"""
    return AIDecisionTracker()


# Example usage patterns
def example_usage():
    """Example of how to use the AI decision tracking pattern"""
    
    # Create a tracker
    tracker = AIDecisionTracker()
    
    # Start a decision process
    start_time = tracker.start_decision("example_analysis")
    
    # Do some AI analysis...
    result = "cycling"
    confidence = 0.85
    
    # Record the decision
    decision = tracker.record_decision(
        decision_type="example_analysis",
        result=result,
        confidence=confidence,
        reasoning="Example reasoning based on analysis",
        evidence=["Evidence point 1", "Evidence point 2"],
        method="example_method",
        start_time=start_time,
        model_used="openai/o3",
        temperature=0.1
    )
    
    # Get decision summary
    summary = tracker.get_decision_summary()
    print("Decision Summary:", summary)
    
    # Convert decision to JSON for storage/logging
    decision_json = decision.to_json()
    print("Decision JSON:", decision_json) 