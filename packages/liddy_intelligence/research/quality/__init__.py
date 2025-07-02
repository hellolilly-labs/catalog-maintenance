"""
Quality Evaluation Framework for Brand Research
Implements LLM-based quality judges with feedback loops per ROADMAP Section 4.4
"""

from liddy_intelligence.ingestion.quality_evaluator import QualityEvaluator
from liddy_intelligence.ingestion.quality_storage import QualityStorageManager

__all__ = [
    "QualityEvaluator",
    "QualityStorageManager"
] 