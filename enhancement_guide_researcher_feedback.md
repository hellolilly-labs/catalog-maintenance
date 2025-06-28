# Researcher Feedback Integration Enhancement Guide

## Overview

This guide provides the specific enhancements needed for researcher classes that override the main `research()` method to properly participate in the quality evaluation feedback loop.

## Current Status

✅ **Base Class Enhanced**: `BaseResearcher` has full feedback support  
✅ **Method Signatures Updated**: All three classes already have `improvement_feedback` parameter  
❌ **Feedback Integration Missing**: Classes don't actually use the feedback parameter  

## Classes Requiring Enhancement

1. **MarketPositioningResearcher** (`src/research/market_positioning_research.py`)
2. **ProductStyleResearcher** (`src/research/product_style_research.py`)  
3. **ResearchIntegrationProcessor** (`src/research/research_integration.py`)

## Enhancement Pattern

Each class needs these specific modifications:

### 1. Research Method Enhancement

The research method should call the quality wrapper when quality evaluation is enabled:

```python
async def research(self, force_refresh: bool = False, improvement_feedback: Optional[List[str]] = None) -> Dict[str, Any]:
    """Enhanced research method that properly handles improvement feedback"""
    # CRITICAL: Use quality wrapper when quality evaluation is enabled
    if self.enable_quality_evaluation:
        return await self._research_with_quality_wrapper(force_refresh, improvement_feedback)
    else:
        return await self._execute_core_research(force_refresh, improvement_feedback)
```

### 2. Feedback Integration

Add feedback handling to the core research logic:

```python
# Handle improvement feedback
feedback_context = ""
if improvement_feedback:
    logger.info(f"📋 Incorporating {len(improvement_feedback)} improvement suggestions")
    feedback_context = self._format_improvement_feedback(improvement_feedback)

# Enhanced prompt with feedback integration
enhanced_prompt = self._get_enhanced_prompt_with_feedback(research_data, feedback_context)
```

## Key Benefits

1. **Automatic Quality Control**: All research phases get quality evaluation
2. **Iterative Improvement**: Poor results automatically get improvement suggestions  
3. **Consistent Quality**: All researchers follow same quality standards
4. **Feedback Loop**: Research improves automatically with each iteration

## Implementation Status

The three researcher classes already have the `improvement_feedback` parameter but need to:
- Call the quality wrapper when quality evaluation is enabled
- Integrate feedback into their prompts and analysis logic
- Test the complete feedback loop

This ensures all researcher classes participate fully in the quality evaluation framework.
