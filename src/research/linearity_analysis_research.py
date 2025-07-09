"""
Linearity Analysis Research Phase

Implements Phase 7 of the Brand Research Pipeline per ROADMAP Section 4.7.

Focus: Linearity and consistency analysis across research phases
Cache Duration: 1-2 months (high stability)
Research Time: 2-4 minutes
Quality Threshold: 7.5
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from src.storage import get_account_storage_provider
from src.progress_tracker import ProgressTracker, StepType, create_console_listener
from src.llm.simple_factory import LLMFactory
from src.llm.prompt_manager import PromptManager
from src.research.base_researcher import BaseResearcher

logger = logging.getLogger(__name__)


class LinearityAnalysisResearcher(BaseResearcher):
    """Linearity Analysis Research Phase Implementation"""
    
    def __init__(self, brand_domain: str, storage_manager=None):
        super().__init__(
            brand_domain=brand_domain,
            researcher_name="linearity_analysis",
            step_type=StepType.SYNTHESIS,
            quality_threshold=7.5,
            cache_duration_days=45,
            storage_manager=storage_manager
        )
        
    # async def research_linearity_analysis(self, brand_domain: str, force_refresh: bool = False) -> Dict[str, Any]:
    #     """Research linearity analysis phase for a brand"""
        
    #     logger.info(f"📊 Starting Linearity Analysis Research for {brand_domain}")
        
    #     # Use the base class research method
    #     result = await self.research(force_refresh=force_refresh)
        
    #     return {
    #         "brand": brand_domain,
    #         "linearity_analysis_content": result.get("content", ""),
    #         "quality_score": result.get("quality_score", 0.8),
    #         "files": result.get("files", []),
    #         "data_sources": result.get("data_sources", 0),
    #         "research_method": result.get("research_method", "enhanced_linearity_analysis")
    #     }

    async def _gather_data(self) -> Dict[str, Any]:
        """Load foundation context from all previous research phases - implements BaseResearcher abstract method"""
        
        context = {}
        
        try:
            # Load all previous research phases
            foundation_files = [
                "foundation",
                "market_positioning",
                "product_style",
                "customer_cultural",
                "voice_messaging",
                "interview_synthesis"
            ]
            
            for phase_key in foundation_files:
                try:
                    blob_path = f"research/{phase_key}/research.md"
                    context[phase_key] = await self.storage_manager.read_file(account=self.brand_domain, file_path=blob_path)
                                
                except Exception as e:
                    logger.debug(f"Could not load {phase_key} research: {e}")
            
            logger.info(f"📋 Loaded {len(context)} foundation research phases for linearity analysis")
            
            # Return in format expected by base class
            return {
                "brand_domain": self.brand_domain,
                "brand_name": self.brand_domain.replace('.com', '').replace('.', ' ').title(),
                "search_results": [],  # Linearity analysis doesn't use web search
                "detailed_sources": [],
                "context": context,
                "total_sources": len(context),
                "search_stats": {
                    "successful_searches": len(context),
                    "failed_searches": 0,
                    "success_rate": 1.0 if context else 0.0,
                    "ssl_errors": 0
                }
            }
            
        except Exception as e:
            logger.warning(f"Error loading foundation context: {e}")
            raise RuntimeError(f"Failed to load foundation context: {e}")

    def _get_default_user_prompt(self) -> str:
        """Get the default user prompt for linearity analysis - implements BaseResearcher abstract method"""
        
        default_prompt = """Analyze foundation research data to create comprehensive linearity analysis across all research phases.

**Brand:** {{brand_name}}
**Domain:** {{brand_domain}}

## Foundation Research Context:
{{search_context}}

## Linearity Analysis Requirements:

Create a comprehensive linearity analysis in **markdown format** that examines consistency across all research phases.

Structure your analysis as follows:

# Linearity Analysis: {{brand_name}}

## 1. Cross-Phase Consistency Analysis
- **Brand Value Alignment:** [Consistency across foundation, positioning, and voice research] [cite sources]
- **Product-Customer Alignment:** [Product style vs. customer cultural intelligence consistency] [cite sources]
- **Message Consistency:** [Voice messaging vs. interview synthesis alignment] [cite sources]
- **Quality Themes:** [Reliability and quality themes throughout all phases] [cite sources]

## 2. Brand Positioning Linearity Assessment
- **Market vs. Product Alignment:** [Market positioning vs. actual product offering] [cite sources]
- **Customer Segmentation Consistency:** [Cultural and voice research alignment] [cite sources]
- **Competitive Positioning:** [Positioning vs. brand personality consistency] [cite sources]
- **Innovation Claims:** [Innovation leadership vs. product style evidence] [cite sources]

## 3. Voice & Messaging Alignment Evaluation
- **Foundation Values vs. Voice:** [Brand values vs. voice messaging patterns] [cite sources]
- **Interview Authenticity:** [Interview synthesis vs. customer cultural insights] [cite sources]
- **Technical Balance:** [Expertise demonstration vs. accessibility balance] [cite sources]
- **Professional Consistency:** [Tone consistency across customer touchpoints] [cite sources]

## 4. Product-Brand Integration Analysis
- **Style-Foundation Alignment:** [Product style vs. brand foundation consistency] [cite sources]
- **Customer-Product Alignment:** [Customer insights vs. product positioning] [cite sources]
- **Innovation Evidence:** [Innovation claims vs. product feature evidence] [cite sources]
- **Quality Messaging:** [Quality messaging vs. product characteristics] [cite sources]

## 5. Market-Customer Alignment Review
- **Market-Customer Consistency:** [Market positioning vs. customer intelligence] [cite sources]
- **Competitive Landscape:** [Competition vs. customer preference alignment] [cite sources]
- **Price Positioning:** [Price positioning vs. customer segmentation] [cite sources]
- **Heritage Alignment:** [Brand heritage vs. customer expectations] [cite sources]

## 6. Overall Linearity & Recommendations
- **Consistency Score:** [Overall brand consistency assessment] [cite sources]
- **Linear Strengths:** [Key consistent areas to leverage] [cite sources]
- **Critical Inconsistencies:** [Areas needing attention] [cite sources]
- **Integration Recommendations:** [Actions for maintaining brand linearity] [cite sources]

## Analysis Quality & Confidence

**Foundation Sources:** {{total_sources}} research phases analyzed
**Information Quality:** {{information_quality}}
**Confidence Level:** {{confidence_level}} confidence in analysis

## Summary

[Provide a 2-3 sentence executive summary of the brand's linearity across research phases]

---

**Important Instructions:**
- Base analysis on cross-phase patterns and consistency themes
- Provide specific examples from each research phase with citations
- Quantify consistency levels where possible (e.g., "85% alignment")
- Focus on actionable insights for brand development and integration
- Use markdown formatting for structure and readability"""

        return default_prompt

    def _get_default_instruction_prompt(self) -> str:
        """Get the default instruction prompt for linearity analysis - implements BaseResearcher abstract method"""
        
        return "You are an expert brand consistency analyst specializing in cross-phase research linearity analysis. Generate comprehensive consistency assessments and actionable insights based on complete foundation research across multiple phases. Focus on identifying patterns, alignments, and areas for improvement in brand consistency."

    async def _analyze_data(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze brand linearity and consistency across research phases"""
        
        step_id = await self.progress_tracker.create_step(
            step_type=StepType.LINEARITY_ANALYSIS,
            brand=self.brand_domain,
            phase_name="Linearity Analysis",
            total_operations=2
        )
        
        await self.progress_tracker.update_progress(step_id, 1, "🔍 Analyzing consistency patterns...")
        
        context_data = context.get('context')
        # Prepare analysis data
        analysis_data = {
            "brand_domain": self.brand_domain,
            "foundation": context_data.get("foundation", ""),
            "market_positioning": context_data.get("market_positioning", ""),
            "product_style": context_data.get("product_style", ""),
            "customer_cultural": context_data.get("customer_cultural", ""),
            "voice_messaging": context_data.get("voice_messaging", ""),
            "interview_synthesis": context_data.get("interview_synthesis", "")
        }
        
        # Generate linearity analysis using LLM
        analysis_content = await self._generate_linearity_analysis(
            analysis_data
        )
        
        await self.progress_tracker.update_progress(step_id, 2, "📈 Calculating consistency score...")
        
        # Calculate quality score based on foundation context richness
        quality_score = self._calculate_linearity_analysis_quality_score(context_data, analysis_content)
        
        return {
            "content": analysis_content,
            "confidence": quality_score,
            "source_count": len(context_data),
            "analysis_type": "foundation_enhanced_linearity_analysis",
            "foundation_phases": list(context_data.keys())
        }

    def _get_default_instruction_prompt(self) -> str:
        """Get linearity analysis prompt from Langfuse"""
        
        default_prompt = """You are an expert brand consistency analyst specializing in cross-phase research linearity analysis. Generate comprehensive consistency assessments and actionable insights based on complete foundation research across multiple phases."""

        return default_prompt
    
    def _get_default_user_prompt(self) -> str:
        """Get the default user prompt for linearity analysis - implements BaseResearcher abstract method"""
        
        default_prompt = """You are conducting comprehensive Linearity Analysis for {{brand_domain}} across all completed research phases.

You have access to complete foundation research from 6 research phases to analyze brand consistency, alignment, and linearity.

Your task is to identify patterns, consistencies, inconsistencies, and overall brand linearity across all research dimensions.

**LINEARITY ANALYSIS REQUIREMENTS:**

Generate comprehensive linearity analysis covering these 6 sections:

## 1. Cross-Phase Consistency Analysis
Evaluate consistency across all research phases:
- Brand value alignment across foundation, positioning, and voice research
- Product-customer alignment between product style and customer cultural intelligence
- Message consistency between voice messaging and interview synthesis
- Quality and reliability themes throughout all phases
*Identify specific consistent themes and highlight any discrepancies*

## 2. Brand Positioning Linearity Assessment
Analyze positioning consistency:
- Market positioning vs. actual product offering alignment
- Customer segmentation consistency across cultural and voice research
- Competitive positioning vs. brand personality consistency
- Innovation leadership claims vs. product style evidence
*Reference specific examples from multiple research phases*

## 3. Voice & Messaging Alignment Evaluation
Examine voice consistency patterns:
- Foundation brand values vs. voice messaging patterns
- Interview synthesis authenticity vs. customer cultural insights
- Technical expertise demonstration vs. accessibility balance
- Professional tone consistency across all customer touchpoints
*Cross-reference voice elements with customer cultural patterns*

## 4. Product-Brand Integration Analysis
Assess product-brand linearity:
- Product style research vs. brand foundation alignment
- Customer cultural insights vs. actual product positioning
- Innovation claims vs. product feature evidence
- Quality messaging vs. product style characteristics
*Evaluate how well products represent brand values*

## 5. Market-Customer Alignment Review
Analyze customer-market consistency:
- Market positioning vs. customer cultural intelligence alignment
- Competitive landscape vs. customer preference consistency
- Price positioning vs. customer segmentation accuracy
- Brand heritage vs. customer expectation alignment
*Identify gaps or strong alignments between market and customer insights*

## 6. Overall Linearity & Recommendations
Provide comprehensive linearity assessment:
- Overall brand consistency score and rationale
- Key linear strengths to leverage in final integration
- Critical inconsistencies that need addressing
- Recommendations for maintaining brand linearity
*Include actionable insights for research integration phase*

**ANALYSIS REQUIREMENTS:**
- Use numbered citations [1], [2], [3] for all cross-phase references
- Provide specific examples from each research phase
- Quantify consistency levels where possible (e.g., "85% alignment")
- Highlight both strengths and improvement areas
- Focus on actionable insights for brand development

**OUTPUT FORMAT:**
Write professional linearity analysis suitable for strategic brand development.
Focus on cross-phase insights and patterns that support comprehensive brand understanding.
Include confidence levels based on foundation data quality and consistency patterns.

BRAND: {{brand_domain}}

COMPREHENSIVE RESEARCH FOUNDATION FOR LINEARITY ANALYSIS:
Foundation Research: {{foundation}}

Market Positioning: {{market_positioning}}

Product Style: {{product_style}}

Customer Cultural: {{customer_cultural}}

Voice & Messaging: {{voice_messaging}}

Interview Synthesis: {{interview_synthesis}}        
"""
        return default_prompt

    async def _generate_linearity_analysis(
        self, 
        analysis_data: Dict[str, Any]
    ) -> str:
        """Generate linearity analysis using foundation research"""
        
        # Prepare template variables
        template_vars = {
            "brand_domain": analysis_data["brand_domain"],
            "foundation": analysis_data["foundation"][:5000] if analysis_data["foundation"] else "No foundation research available",
            "market_positioning": analysis_data["market_positioning"][:2000] if analysis_data["market_positioning"] else "No market positioning available",
            "product_style": analysis_data["product_style"][:2000] if analysis_data["product_style"] else "No product style research available",
            "customer_cultural": analysis_data["customer_cultural"][:2000] if analysis_data["customer_cultural"] else "No customer cultural research available",
            "voice_messaging": analysis_data["voice_messaging"][:2000] if analysis_data["voice_messaging"] else "No voice messaging research available",
            "interview_synthesis": analysis_data["interview_synthesis"][:2000] if analysis_data["interview_synthesis"] else "No interview synthesis available"
        }
        
        # Get prompt from Langfuse
        prompt_template = await self.prompt_manager.get_prompt(
            prompt_name="internal/researcher/linearity_analysis",
            prompt_type="chat",
            prompt=[
                {"role": "system", "content": self._get_default_instruction_prompt()},
                {"role": "user", "content": self._get_default_user_prompt()}
            ]
        )
        prompts = prompt_template.prompt
        
        system_prompt = next((msg["content"] for msg in prompts if msg["role"] == "system"), None)
        user_prompt = next((msg["content"] for msg in prompts if msg["role"] == "user"), None)
        
        # Replace template variables
        for var, value in template_vars.items():
            system_prompt = system_prompt.replace(f"{{{{{var}}}}}", str(value))
            user_prompt = user_prompt.replace(f"{{{{{var}}}}}", str(value))
                
        response = await LLMFactory.chat_completion(
            task="linearity_analysis_research",
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.6,
            # max_tokens=4000
        )
        
        return response.get("content", "Linearity analysis generation failed")

    def _calculate_linearity_analysis_quality_score(
        self, 
        foundation_context: Dict[str, Any], 
        analysis_content: str
    ) -> float:
        """Calculate quality score for linearity analysis"""
        
        base_score = 0.65  # Base score for linearity analysis
        
        # Foundation context quality
        context_count = len(foundation_context)
        context_bonus = min(0.18, context_count * 0.03)  # Up to 0.18 for 6+ phases
        
        # Analysis content quality
        content_length = len(analysis_content)
        content_bonus = min(0.10, content_length / 12000)  # Up to 0.10 for comprehensive analysis
        
        # Check for citations in content
        citation_count = analysis_content.count("[") + analysis_content.count("]")
        citation_bonus = min(0.05, citation_count * 0.003)  # Up to 0.05 for good citations
        
        # Check for consistency analysis patterns
        consistency_indicators = analysis_content.count("consistency") + analysis_content.count("alignment") + analysis_content.count("linear")
        consistency_bonus = min(0.04, consistency_indicators * 0.002)  # Bonus for consistency focus
        
        # Check for cross-phase references
        cross_phase_indicators = analysis_content.count("vs.") + analysis_content.count("across") + analysis_content.count("between")
        cross_phase_bonus = min(0.03, cross_phase_indicators * 0.002)  # Bonus for cross-phase analysis
        
        final_score = base_score + context_bonus + content_bonus + citation_bonus + consistency_bonus + cross_phase_bonus
        return min(0.85, final_score)  # Cap at 0.85 for linearity analysis

    async def _save_linearity_analysis_research(self, brand_domain: str, analysis_result: Dict[str, Any]) -> List[str]:
        """Save linearity analysis research in three-file format"""
        
        saved_files = []
        
        try:
            if hasattr(self.storage_manager, 'base_dir'):
                # Local storage
                research_dir = os.path.join(self.storage_manager.base_dir, "accounts", brand_domain, "research_phases")
                os.makedirs(research_dir, exist_ok=True)
                
                # Save content
                content_path = os.path.join(research_dir, "linearity_analysis_research.md")
                with open(content_path, "w") as f:
                    f.write(analysis_result["content"])
                saved_files.append(content_path)
                
                # Save metadata
                metadata = {
                    "phase": "linearity_analysis",
                    "confidence_score": analysis_result.get("confidence", 0.8),
                    "analysis_type": analysis_result.get("analysis_type", "enhanced_linearity_analysis"),
                    "foundation_phases": analysis_result.get("foundation_phases", []),
                    "source_count": analysis_result.get("source_count", 0),
                    "research_metadata": {
                        "phase": "linearity_analysis", 
                        "research_duration_seconds": time.time(),
                        "timestamp": datetime.now().isoformat() + "Z",
                        "quality_threshold": self.quality_threshold,
                        "cache_expires": (datetime.now() + timedelta(days=self.cache_duration_days)).isoformat() + "Z",
                        "version": "2.0_enhanced"
                    }
                }
                
                metadata_path = os.path.join(research_dir, "linearity_analysis_research_metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                saved_files.append(metadata_path)
                
                # Save sources/data
                sources_data = {
                    "analysis_sources": analysis_result.get("foundation_phases", []),
                    "foundation_research_phases": len(analysis_result.get("foundation_phases", [])),
                    "collection_timestamp": datetime.now().isoformat() + "Z"
                }
                
                sources_path = os.path.join(research_dir, "linearity_analysis_research_sources.json")
                with open(sources_path, "w") as f:
                    json.dump(sources_data, f, indent=2)
                saved_files.append(sources_path)
                
            else:
                # GCP storage
                research_dir = f"accounts/{brand_domain}/research_phases"
                
                # Save content
                content_blob = f"{research_dir}/linearity_analysis_research.md"
                blob = self.storage_manager.bucket.blob(content_blob)
                blob.upload_from_string(analysis_result["content"])
                saved_files.append(content_blob)
                
                # Save metadata  
                metadata = {
                    "phase": "linearity_analysis",
                    "confidence_score": analysis_result.get("confidence", 0.8),
                    "analysis_type": analysis_result.get("analysis_type", "enhanced_linearity_analysis"),
                    "foundation_phases": analysis_result.get("foundation_phases", []),
                    "source_count": analysis_result.get("source_count", 0),
                    "research_metadata": {
                        "phase": "linearity_analysis",
                        "research_duration_seconds": time.time(),
                        "timestamp": datetime.now().isoformat() + "Z",
                        "quality_threshold": self.quality_threshold,
                        "cache_expires": (datetime.now() + timedelta(days=self.cache_duration_days)).isoformat() + "Z",
                        "version": "2.0_enhanced"
                    }
                }
                
                metadata_blob = f"{research_dir}/linearity_analysis_research_metadata.json"
                blob = self.storage_manager.bucket.blob(metadata_blob)
                blob.upload_from_string(json.dumps(metadata, indent=2))
                saved_files.append(metadata_blob)
                
                # Save sources
                sources_data = {
                    "analysis_sources": analysis_result.get("foundation_phases", []),
                    "foundation_research_phases": len(analysis_result.get("foundation_phases", [])),
                    "collection_timestamp": datetime.now().isoformat() + "Z"
                }
                
                sources_blob = f"{research_dir}/linearity_analysis_research_sources.json"
                blob = self.storage_manager.bucket.blob(sources_blob)
                blob.upload_from_string(json.dumps(sources_data, indent=2))
                saved_files.append(sources_blob)
            
            logger.info(f"✅ Saved linearity analysis research for {brand_domain}")
            
        except Exception as e:
            logger.error(f"❌ Error saving linearity analysis research: {e}")
            raise
        
        return saved_files

    async def _load_cached_linearity_analysis(self, brand_domain: str) -> Optional[Dict[str, Any]]:
        """Load cached linearity analysis research"""
        try:
            if hasattr(self.storage_manager, 'base_dir'):
                # Local storage
                research_dir = os.path.join(self.storage_manager.base_dir, "accounts", brand_domain, "research_phases")
                
                content_path = os.path.join(research_dir, "linearity_analysis_research.md")
                metadata_path = os.path.join(research_dir, "linearity_analysis_research_metadata.json")
                
                if all(os.path.exists(p) for p in [content_path, metadata_path]):
                    # Check cache expiry
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    
                    research_metadata = metadata.get("research_metadata", {})
                    cache_expires = research_metadata.get("cache_expires")
                    
                    if cache_expires and datetime.now() < datetime.fromisoformat(cache_expires.replace("Z", "")):
                        # Load cached data
                        with open(content_path, "r") as f:
                            content = f.read()
                        
                        return {
                            "brand": brand_domain,
                            "linearity_analysis_content": content,
                            "quality_score": metadata.get("confidence_score", 0.8),
                            "files": [content_path, metadata_path],
                            "data_sources": metadata.get("source_count", 0),
                            "research_method": metadata.get("analysis_type", "cached_analysis")
                        }
            else:
                # GCP storage 
                research_dir = f"accounts/{brand_domain}/research_phases"
                
                content_blob = f"{research_dir}/linearity_analysis_research.md"
                metadata_blob = f"{research_dir}/linearity_analysis_research_metadata.json"
                
                content_blob_obj = self.storage_manager.bucket.blob(content_blob)
                metadata_blob_obj = self.storage_manager.bucket.blob(metadata_blob)
                
                if content_blob_obj.exists() and metadata_blob_obj.exists():
                    # Check cache expiry
                    metadata_content = metadata_blob_obj.download_as_text()
                    metadata = json.loads(metadata_content)
                    
                    research_metadata = metadata.get("research_metadata", {})
                    cache_expires = research_metadata.get("cache_expires")
                    
                    if cache_expires and datetime.now() < datetime.fromisoformat(cache_expires.replace("Z", "")):
                        # Load cached data
                        content = content_blob_obj.download_as_text()
                        
                        return {
                            "brand": brand_domain,
                            "linearity_analysis_content": content,
                            "quality_score": metadata.get("confidence_score", 0.8),
                            "files": [content_blob, metadata_blob],
                            "data_sources": metadata.get("source_count", 0),
                            "research_method": metadata.get("analysis_type", "cached_analysis")
                        }
        except Exception as e:
            logger.debug(f"Cache check failed: {e}")
        
        return None


def get_linearity_analysis_researcher(brand_domain: str) -> LinearityAnalysisResearcher:
    """Get linearity analysis researcher instance"""
    return LinearityAnalysisResearcher(brand_domain)
