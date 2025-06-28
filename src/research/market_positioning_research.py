"""
Market Positioning Research Phase

Implements Phase 2 of the Brand Research Pipeline per ROADMAP Section 4.2.

Focus: Competitive landscape and market position
Cache Duration: 3-6 months (moderate stability)
Research Time: 2-4 minutes
Quality Threshold: 7.5

Research Sources:
- Direct and indirect competitor analysis
- Market share and industry position
- Pricing strategy and value proposition
- Strategic partnerships and alliances
- Industry awards and recognition
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from src.llm.simple_factory import LLMFactory
from src.llm.prompt_manager import PromptManager
from src.storage import get_account_storage_provider
from configs.settings import get_settings
from src.progress_tracker import (
    get_progress_tracker, 
    StepType, 
    create_console_listener,
    ProgressTracker
)
from src.research.base_researcher import BaseResearcher
from src.research.data_sources import WebSearchDataSource, DataGatheringContext

logger = logging.getLogger(__name__)


class MarketPositioningResearcher(BaseResearcher):
    """
    Market Positioning Research Phase Implementation
    
    Researches competitive landscape and market position:
    - Direct and indirect competitor analysis
    - Market share and industry position
    - Pricing strategy and value proposition
    - Strategic partnerships and alliances
    - Industry awards and recognition
    """
    
    def __init__(self, brand_domain: str, storage_manager=None):
        super().__init__(brand_domain=brand_domain, researcher_name="market_positioning", step_type=StepType.MARKET_POSITIONING_RESEARCH, quality_threshold=7.5, cache_duration_days=120, storage_manager=storage_manager)
        """Initialize market positioning researcher with storage integration"""
        # self.quality_threshold = 7.5
        # self.cache_duration_days = 120  # 4 months default
        # self.brand_domain = brand_domain.lower()
        
        # # Create progress tracker with storage integration and checkpoint logging
        # self.progress_tracker = ProgressTracker(
        #     storage_manager=self.storage_manager,
        #     enable_checkpoints=True  # Enable persistent checkpoint logging
        # )
        
        # # Add console listener for real-time updates
        # console_listener = create_console_listener()
        # self.progress_tracker.add_progress_listener(console_listener)
        
        # # Initialize prompt manager
        # self.prompt_manager = PromptManager()
        
    async def research(self, force_refresh: bool = False, improvement_feedback: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Enhanced research method that properly handles improvement feedback
        
        Args:
            force_refresh: Force new research even if cached
            improvement_feedback: Optional feedback from previous quality evaluation for iterative improvement
            
        Returns:
            Market positioning research results with metadata
        """
        # CRITICAL: Use quality wrapper when quality evaluation is enabled
        if self.enable_quality_evaluation:
            return await self._research_with_quality_wrapper(force_refresh, improvement_feedback)
        else:
            return await self._execute_core_research(force_refresh, improvement_feedback)
    
    async def _execute_core_research(self, force_refresh: bool = False, improvement_feedback: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Core research execution method that handles improvement feedback
        """
        start_time = time.time()
        
        logger.info(f"🏁 Starting Market Positioning Research for {self.brand_domain}")
        
        # Handle improvement feedback
        feedback_context = ""
        if improvement_feedback:
            logger.info(f"📋 Incorporating {len(improvement_feedback)} improvement suggestions")
            feedback_context = self._format_improvement_feedback(improvement_feedback)
        
        # Create main progress step
        step_id = await self.progress_tracker.create_step(
            step_type=StepType.MARKET_POSITIONING_RESEARCH,
            brand=self.brand_domain,
            phase_name="Market Positioning Research",
            total_operations=6  # data gathering, analysis, synthesis, validation, storage, completion
        )
        
        try:
            await self.progress_tracker.start_step(step_id, "Checking cache and initializing...")
            
            # Check for cached results first
            if not force_refresh:
                cached_result = await self._load_cached_results()
                if cached_result:
                    await self.progress_tracker.complete_step(
                        step_id, 
                        output_files=cached_result.get("files", []),
                        quality_score=cached_result.get("quality_score"),
                        cache_hit=True
                    )
                    logger.info(f"✅ Using cached market positioning research for {self.brand_domain}")
                    return cached_result
            
            # Step 1: Comprehensive competitive data gathering (60-90 seconds)
            await self.progress_tracker.update_progress(step_id, 1, "🏁 Step 1: Gathering competitive landscape data...")
            positioning_data = await self._gather_data()
            
            # Step 2: Multi-round LLM analysis with feedback integration (60-90 seconds)
            await self.progress_tracker.update_progress(step_id, 2, "🧠 Step 2: Analyzing market positioning with LLM...")
            positioning_analysis = await self._analyze_data_with_feedback(positioning_data, feedback_context)
            
            # Step 3: Quality evaluation and synthesis (30-60 seconds)
            await self.progress_tracker.update_progress(step_id, 3, "🎯 Step 3: Quality evaluation and synthesis...")
            final_positioning = await self._synthesize_results(positioning_analysis)
            
            # Add metadata
            final_positioning.update({
                "research_metadata": {
                    "phase": "market_positioning",
                    "research_duration_seconds": time.time() - start_time,
                    "timestamp": datetime.now().isoformat() + "Z",
                    "cache_expires": (datetime.now() + timedelta(days=self.cache_duration_days)).isoformat() + "Z",
                    "quality_threshold": self.quality_threshold,
                    "version": "1.0",
                    "feedback_incorporated": len(improvement_feedback) if improvement_feedback else 0
                }
            })
            
            # Save to storage
            saved_files = await self._save_results(final_positioning)
            
            duration = time.time() - start_time
            logger.info(f"✅ Market Positioning Research completed for {self.brand_domain} in {duration:.1f}s")
            
            await self.progress_tracker.complete_step(
                step_id,
                output_files=saved_files,
                quality_score=final_positioning.get("confidence_score", 0.75),
                cache_hit=False
            )
            
            return {
                "brand": self.brand_domain,
                "content": final_positioning.get("content", ""),
                "quality_score": final_positioning.get("confidence_score", 0.75),
                "files": saved_files,
                "data_sources": len(positioning_data.get("search_results", [])),
                "research_method": "enhanced_competitive_analysis",
                "feedback_incorporated": len(improvement_feedback) if improvement_feedback else 0
            }
            
        except Exception as e:
            await self.progress_tracker.fail_step(step_id, str(e))
            logger.error(f"❌ Error in market positioning research for {self.brand_domain}: {e}")
            raise
    
    def _format_improvement_feedback(self, feedback: List[str]) -> str:
        """
        Format improvement feedback for inclusion in prompts
        """
        if not feedback:
            return ""
        
        formatted = "\n\n## IMPROVEMENT FEEDBACK FROM PREVIOUS QUALITY EVALUATION:\n\n"
        formatted += "Please address the following areas for improvement:\n\n"
        
        for i, suggestion in enumerate(feedback, 1):
            formatted += f"{i}. {suggestion}\n"
        
        formatted += "\nPlease specifically address these points in your analysis to improve quality.\n"
        return formatted
    
    async def _analyze_data_with_feedback(self, positioning_data: Dict[str, Any], feedback_context: str) -> Dict[str, Any]:
        """
        Analyze market positioning data using LLM with improvement feedback integration
        """
        # Check if we have sufficient data for analysis
        total_sources = positioning_data.get("total_sources", 0)
        search_stats = positioning_data.get("search_stats", {})
        success_rate = search_stats.get("success_rate", 0)
        
        if total_sources == 0:
            error_msg = "ANALYSIS ABORTED: No search results available for analysis. Cannot generate quality research without external data."
            logger.error(f"🚨 {error_msg}")
            raise RuntimeError(error_msg)
        
        # Compile search context with source IDs for citation
        search_context = ""
        source_citations = {}  # Map source_id to citation format
        
        for result in positioning_data["search_results"][:50]:  # Use top 50 results
            source_id = result.get("source_id", f"source_{len(source_citations)}")
            citation = f"[{len(source_citations) + 1}]"
            source_citations[source_id] = citation
            
            search_context += f"**Source {citation}:**\n"
            search_context += f"**Title:** {result.get('title', '')}\n"
            search_context += f"**URL:** {result.get('url', '')}\n"
            search_context += f"**Content:** {result.get('snippet', '')}\n"
            search_context += f"**Query:** {result.get('source_query', '')}\n\n---\n\n"
        
        # Create source reference guide for LLM
        source_reference_guide = "\n".join([
            f"{citation} - {result.get('title', 'Untitled')} ({result.get('url', 'No URL')})"
            for result, citation in zip(positioning_data["search_results"][:50], source_citations.values())
        ])
        
        # Get enhanced prompt with feedback integration
        enhanced_prompt = self._get_enhanced_prompt_with_feedback(positioning_data, search_context, source_reference_guide, feedback_context)
        
        try:
            logger.info(f"🧠 Analyzing {total_sources} sources with LLM (success rate: {success_rate:.1%})")
            if feedback_context:
                logger.info("📋 Including improvement feedback in analysis prompt")
            
            response = await LLMFactory.chat_completion(
                task="brand_research",
                system=self._get_default_instruction_prompt(),
                messages=[{
                    "role": "user",
                    "content": enhanced_prompt
                }],
                temperature=0.1
            )
            
            if response and response.get("content"):
                # Adjust confidence based on data quality and feedback integration
                base_confidence = 0.75
                if success_rate < 0.5:
                    base_confidence = 0.5
                elif success_rate < 0.7:
                    base_confidence = 0.6
                
                if total_sources < 10:
                    base_confidence *= 0.9
                elif total_sources < 5:
                    base_confidence *= 0.8
                
                # Boost confidence if feedback was incorporated
                if feedback_context:
                    base_confidence = min(0.95, base_confidence * 1.1)
                
                return {
                    "content": response["content"],
                    "analysis_method": "markdown_report_with_citations_and_feedback",
                    "confidence": base_confidence,
                    "data_sources": total_sources,
                    "search_success_rate": success_rate,
                    "detailed_sources": positioning_data.get('detailed_sources', []),
                    "source_citations": source_citations,
                    "search_stats": search_stats,
                    "feedback_incorporated": bool(feedback_context)
                }
            else:
                error_msg = "ANALYSIS FAILED: No response from LLM analysis despite having valid data"
                logger.error(f"🚨 {error_msg}")
                raise RuntimeError(error_msg)
                
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise e
            error_msg = f"ANALYSIS FAILED: Error in LLM analysis: {str(e)}"
            logger.error(f"🚨 {error_msg}")
            raise RuntimeError(error_msg)
    
    def _get_enhanced_prompt_with_feedback(self, positioning_data: Dict[str, Any], search_context: str, source_reference_guide: str, feedback_context: str) -> str:
        """
        Get enhanced prompt template that includes improvement feedback
        """
        total_sources = positioning_data.get("total_sources", 0)
        search_stats = positioning_data.get("search_stats", {})
        success_rate = search_stats.get("success_rate", 0)
        
        # Prepare template variables
        template_vars = {
            "brand_name": positioning_data.get('brand_name', self.brand_domain),
            "brand_domain": self.brand_domain,
            "total_sources": str(total_sources),
            "success_rate": f"{success_rate:.1%}",
            "data_quality": "High" if success_rate > 0.7 else "Medium" if success_rate > 0.5 else "Limited",
            "search_context": search_context,
            "source_reference_guide": source_reference_guide,
            "information_quality": "High quality with comprehensive data" if success_rate > 0.7 else "Medium quality with adequate data" if success_rate > 0.5 else "Limited quality with minimal data",
            "confidence_level": "High" if success_rate > 0.7 and total_sources >= 20 else "Medium" if success_rate > 0.5 and total_sources >= 10 else "Low",
            "data_quality_text": "high" if success_rate > 0.7 else "medium" if success_rate > 0.5 else "limited"
        }
        
        # Get base prompt and add feedback context
        base_prompt = self._get_default_user_prompt()
        
        # Replace template variables
        enhanced_prompt = base_prompt
        for var, value in template_vars.items():
            enhanced_prompt = enhanced_prompt.replace(f"{{{{{var}}}}}", str(value))
        
        # Add feedback context if available
        if feedback_context:
            enhanced_prompt += feedback_context
            enhanced_prompt += "\n\nEnsure your analysis specifically addresses the improvement feedback above to enhance quality and completeness."
        
        return enhanced_prompt
    
    async def _gather_data(self) -> Dict[str, Any]:
        """Gather comprehensive market positioning data using WebSearchDataSource"""
        
        # Competitive analysis queries (with enhanced parameters)
        brand_name = self.brand_domain.replace('.com', '').replace('.', ' ').title()
        research_queries = [
            {"query": f"{brand_name} ({self.brand_domain}) competitive landscape", "max_results": 10, "include_domains": ["statista.com", "ibisworld.com", "crunchbase.com", "cbinsights.com", "pitchbook.com", "sec.gov", "reuters.com", "businesswire.com", "prnewswire.com", "g2.com", "capterra.com"]},
            {"query": f"{brand_name} ({self.brand_domain}) competitors direct competition analysis"},
            {"query": f"{brand_name} ({self.brand_domain}) vs competitors comparison market position"},
            {"query": f"{brand_name} ({self.brand_domain}) market share industry position ranking"},
            {"query": f"{brand_name} ({self.brand_domain}) pricing strategy value proposition"},
            {"query": f"{brand_name} ({self.brand_domain}) competitive advantages differentiation"},
            {"query": f"{brand_name} ({self.brand_domain}) partnerships alliances strategic relationships"},
            {"query": f"{brand_name} ({self.brand_domain}) industry awards recognition achievements"},
            {"query": f"{brand_name} ({self.brand_domain}) market leadership position analysis"},
            {"query": f"{brand_name} ({self.brand_domain}) competitive landscape industry overview"},
            {"query": f"{brand_name} ({self.brand_domain}) brand positioning strategy market approach"},
            {"query": f"{brand_name} ({self.brand_domain}) target market segment positioning"},
            {"query": f"{brand_name} ({self.brand_domain}) industry trends market dynamics"},
        ]
        
        # Use WebSearchDataSource for data gathering
        try:
            web_search_source = WebSearchDataSource()
            
            if not web_search_source.is_available():
                error_msg = "ABORTING: Web search service not available. Cannot proceed with research without external data sources."
                logger.error(f"🚨 {error_msg}")
                raise RuntimeError(error_msg)
            
            # Create data gathering context
            context = DataGatheringContext(
                brand_domain=self.brand_domain,
                researcher_name=self.researcher_name,
                phase_name="market_positioning"
            )
            
            # Gather data using the data source strategy
            search_result = await web_search_source.gather(research_queries, context)
            
            # 🚨 ABORT CONDITIONS - Don't continue with poor quality data
            total_searches = len(research_queries)
            success_rate = search_result.successful_searches / total_searches if total_searches > 0 else 0
            
            # Abort if we have SSL errors affecting most searches
            if search_result.ssl_errors >= 3:  # 3+ SSL errors indicates systemic SSL issues
                error_msg = f"ABORTING: SSL certificate verification failed for {search_result.ssl_errors} searches. Cannot proceed with research without reliable web access."
                logger.error(f"🚨 {error_msg}")
                raise RuntimeError(error_msg)
            
            # Abort if overall success rate is too low for quality research
            if success_rate < 0.3:  # Less than 30% success rate
                error_msg = f"ABORTING: Only {search_result.successful_searches}/{total_searches} searches succeeded ({success_rate:.1%}). Insufficient data for quality research."
                logger.error(f"🚨 {error_msg}")
                raise RuntimeError(error_msg)
            
            # Warn about reduced quality but continue if we have some data
            if success_rate < 0.7:  # Less than 70% success rate
                logger.warning(f"⚠️ Reduced data quality: Only {search_result.successful_searches}/{total_searches} searches succeeded ({success_rate:.1%})")
            
            logger.info(f"✅ Web search completed: {search_result.successful_searches}/{total_searches} successful searches, {len(search_result.results)} total sources")
            
            # Return data in the expected format (maintaining backward compatibility)
            return {
                "brand_domain": self.brand_domain,
                "brand_name": brand_name,
                "search_results": search_result.results,
                "detailed_sources": search_result.sources,
                "research_queries": research_queries,
                "total_sources": len(search_result.results),
                "search_stats": {
                    "successful_searches": search_result.successful_searches,
                    "failed_searches": search_result.failed_searches,
                    "success_rate": success_rate,
                    "ssl_errors": search_result.ssl_errors
                },
                "sources_by_type": {
                    "web_search": len(search_result.sources)
                }
            }
                
        except RuntimeError:
            # Re-raise abort conditions
            raise
        except Exception as e:
            error_msg = f"ABORTING: Critical error in data gathering: {str(e)}"
            logger.error(f"🚨 {error_msg}")
            raise RuntimeError(error_msg)
    
    def _get_default_user_prompt(self) -> str:
        """Get market positioning analysis prompt from Langfuse"""
        
        # total_sources = positioning_data.get("total_sources", 0)
        # search_stats = positioning_data.get("search_stats", {})
        # success_rate = search_stats.get("success_rate", 0)
        
        default_prompt = """Analyze this market positioning research data to extract comprehensive competitive intelligence.

**Brand:** {{brand_name}}
**Domain:** {{brand_domain}}

## Research Data Quality Notice:
- **Total Sources**: {{total_sources}} sources analyzed
- **Search Success Rate**: {{success_rate}} 
- **Data Quality**: {{data_quality}}

## Research Data Sources:

{{search_context}}

## Source Reference Guide:
{{source_reference_guide}}

## Market Positioning Analysis Requirements:

Please create a comprehensive market positioning intelligence report in **markdown format**. When referencing information, cite your sources using the numbers provided (e.g., [1], [2], [3]).

Structure your analysis as follows:

# Market Positioning Intelligence: {{brand_name}}

## 1. Competitive Landscape
- **Direct Competitors:** [Main competitors in same category] [cite sources]
- **Indirect Competitors:** [Competitors from adjacent categories] [cite sources]
- **Market Position:** [Where they rank in competitive hierarchy] [cite sources]
- **Competitive Advantages:** [Key differentiators vs competitors] [cite sources]

## 2. Market Share & Industry Position
- **Market Share:** [Market share information if available] [cite sources]
- **Industry Ranking:** [Position within industry] [cite sources]
- **Market Size:** [Target market size and scope] [cite sources]
- **Growth Trajectory:** [Market position trends] [cite sources]

## 3. Pricing Strategy & Value Proposition
- **Pricing Position:** [Premium, mid-market, or value positioning] [cite sources]
- **Value Proposition:** [Core value delivered to customers] [cite sources]
- **Pricing Strategy:** [How they compete on price] [cite sources]
- **Price Comparison:** [Relative to competitors] [cite sources]

## 4. Strategic Partnerships & Alliances
- **Key Partnerships:** [Important strategic relationships] [cite sources]
- **Distribution Partners:** [Retail and channel partnerships] [cite sources]
- **Technology Alliances:** [Technical and innovation partnerships] [cite sources]
- **Strategic Impact:** [How partnerships enhance market position] [cite sources]

## 5. Industry Recognition & Awards
- **Industry Awards:** [Recognition and accolades received] [cite sources]
- **Media Recognition:** [Press coverage and industry mentions] [cite sources]
- **Thought Leadership:** [Industry leadership and influence] [cite sources]
- **Brand Reputation:** [Overall market reputation] [cite sources]

## 6. Market Dynamics & Trends
- **Industry Trends:** [Key trends affecting the market] [cite sources]
- **Market Challenges:** [Industry-wide challenges] [cite sources]
- **Opportunity Areas:** [Growth opportunities in market] [cite sources]
- **Future Outlook:** [Market direction and brand positioning] [cite sources]

## Analysis Quality & Confidence

**Data Sources:** {{total_sources}} search results analyzed
**Search Success Rate:** {{success_rate}}
**Information Quality:** {{information_quality}}
**Confidence Level:** {{confidence_level}} confidence in findings
**Key Gaps:** [Note any information that was missing or unclear due to limited data availability]

## Summary

[Provide a 2-3 sentence executive summary of the brand's market positioning]

## Sources

{{source_reference_guide}}

---

**Important Instructions:**
- **ALWAYS cite sources** using the provided reference numbers [1], [2], [3], etc.
- Focus on factual, verifiable information only
- Clearly distinguish between official statements and external analysis  
- Note confidence levels for different claims based on data availability
- If information is missing, clearly state "Not available in research data"
- Given the {{data_quality_text}} data quality, be appropriately cautious in claims
- Use markdown formatting for structure and readability
- Include the complete sources list at the end"""

        return default_prompt


    def _get_default_instruction_prompt(self) -> str:
        default_prompt = "You are an expert business researcher specializing in competitive market analysis. Generate comprehensive, well-structured markdown reports with proper source citations based on research data. Always cite your sources using the provided reference numbers. Adjust confidence levels based on data quality."
        return default_prompt

    # async def _analyze_data(self, brand_domain: str, positioning_data: Dict[str, Any]) -> Dict[str, Any]:
    #     """Analyze market positioning data using LLM and generate markdown content with source references"""
        
    #     # Check if we have sufficient data for analysis
    #     total_sources = positioning_data.get("total_sources", 0)
    #     search_stats = positioning_data.get("search_stats", {})
    #     success_rate = search_stats.get("success_rate", 0)
        
    #     if total_sources == 0:
    #         error_msg = "ANALYSIS ABORTED: No search results available for analysis. Cannot generate quality research without external data."
    #         logger.error(f"🚨 {error_msg}")
    #         raise RuntimeError(error_msg)
        
    #     # Warn about data quality but proceed if we have some sources
    #     if total_sources < 10:
    #         logger.warning(f"⚠️ Limited data available: Only {total_sources} sources for analysis. Research quality may be reduced.")
        
    #     if success_rate < 0.5:
    #         logger.warning(f"⚠️ Low search success rate: {success_rate:.1%}. Research confidence may be reduced.")
        
    #     # Compile search context with source IDs for citation
    #     search_context = ""
    #     source_citations = {}  # Map source_id to citation format
        
    #     for result in positioning_data["search_results"][:50]:  # Use top 50 results
    #         source_id = result.get("source_id", f"source_{len(source_citations)}")
    #         citation = f"[{len(source_citations) + 1}]"
    #         source_citations[source_id] = citation
            
    #         search_context += f"**Source {citation}:**\n"
    #         search_context += f"**Title:** {result.get('title', '')}\n"
    #         search_context += f"**URL:** {result.get('url', '')}\n"
    #         search_context += f"**Content:** {result.get('snippet', '')}\n"
    #         search_context += f"**Query:** {result.get('source_query', '')}\n\n---\n\n"
        
    #     # Create source reference guide for LLM
    #     source_reference_guide = "\n".join([
    #         f"{citation} - {result.get('title', 'Untitled')} ({result.get('url', 'No URL')})"
    #         for result, citation in zip(positioning_data["search_results"][:50], source_citations.values())
    #     ])
        
    #     # Get market positioning prompt from PromptManager
    #     prompt_template = await self._get_market_positioning_prompt(brand_domain, positioning_data)
        
    #     # Prepare template variables
    #     template_vars = {
    #         "brand_name": positioning_data.get('brand_name', brand_domain),
    #         "brand_domain": brand_domain,
    #         "total_sources": str(total_sources),
    #         "success_rate": f"{success_rate:.1%}",
    #         "data_quality": "High" if success_rate > 0.7 else "Medium" if success_rate > 0.5 else "Limited",
    #         "search_context": search_context,
    #         "source_reference_guide": source_reference_guide,
    #         "information_quality": "High quality with comprehensive data" if success_rate > 0.7 else "Medium quality with adequate data" if success_rate > 0.5 else "Limited quality with minimal data",
    #         "confidence_level": "High" if success_rate > 0.7 and total_sources >= 20 else "Medium" if success_rate > 0.5 and total_sources >= 10 else "Low",
    #         "data_quality_text": "high" if success_rate > 0.7 else "medium" if success_rate > 0.5 else "limited"
    #     }
        
    #     # Replace template variables
    #     final_prompt = prompt_template
    #     for var, value in template_vars.items():
    #         final_prompt = final_prompt.replace(f"{{{{{var}}}}}", str(value))

    #     try:
    #         logger.info(f"🧠 Analyzing {total_sources} sources with O3 model (success rate: {success_rate:.1%})")
            
    #         response = await LLMFactory.chat_completion(
    #             task="brand_research",
    #             system="You are an expert business researcher specializing in competitive market analysis. Generate comprehensive, well-structured markdown reports with proper source citations based on research data. Always cite your sources using the provided reference numbers. Adjust confidence levels based on data quality.",
    #             messages=[{
    #                 "role": "user",
    #                 "content": final_prompt
    #             }],
    #             # max_tokens=2500,
    #             temperature=0.1
    #         )
            
    #         if response and response.get("content"):
    #             # Adjust confidence based on data quality
    #             base_confidence = 0.75
    #             if success_rate < 0.5:
    #                 base_confidence = 0.5
    #             elif success_rate < 0.7:
    #                 base_confidence = 0.6
                
    #             if total_sources < 10:
    #                 base_confidence *= 0.9
    #             elif total_sources < 5:
    #                 base_confidence *= 0.8
                
    #             return {
    #                 "market_positioning_markdown": response["content"],
    #                 "analysis_method": "markdown_report_with_citations",
    #                 "confidence": base_confidence,
    #                 "data_sources": total_sources,
    #                 "search_success_rate": success_rate,
    #                 "detailed_sources": positioning_data.get('detailed_sources', []),
    #                 "source_citations": source_citations,
    #                 "search_stats": search_stats
    #             }
    #         else:
    #             error_msg = "ANALYSIS FAILED: No response from LLM analysis despite having valid data"
    #             logger.error(f"🚨 {error_msg}")
    #             raise RuntimeError(error_msg)
                
    #     except Exception as e:
    #         if isinstance(e, RuntimeError):
    #             raise e
    #         error_msg = f"ANALYSIS FAILED: Error in LLM analysis: {str(e)}"
    #         logger.error(f"🚨 {error_msg}")
    #         raise RuntimeError(error_msg)
    
    # async def _synthesize_results(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
    #     """Synthesize final market positioning intelligence"""
        
    #     return {
    #         "content": analysis["markdown"],
    #         "confidence_score": analysis["confidence"],
    #         "data_quality": "high" if analysis["search_success_rate"] > 0.7 else "medium" if analysis["search_success_rate"] > 0.5 else "limited",
    #         "data_sources_count": analysis["data_sources"],
    #         "analysis_method": analysis["analysis_method"],
    #         "detailed_sources": analysis["detailed_sources"],
    #         "source_citations": analysis["source_citations"]
    #     }
    
    # async def _save_results(self, positioning: Dict[str, Any]) -> List[str]:
    #     """Save market positioning research to storage with three-file architecture"""
        
    #     saved_files = []
        
    #     try:
    #         # File 1: Clean markdown content
    #         markdown_content = positioning.get("content", "")
            
    #         # File 2: Pure metadata
    #         metadata = {
    #             "phase": "market_positioning",
    #             "confidence_score": positioning.get("confidence_score", 0.75),
    #             "data_quality": positioning.get("data_quality", "medium"),
    #             "data_sources_count": positioning.get("data_sources_count", 0),
    #             "analysis_method": positioning.get("analysis_method", "unknown"),
    #             "research_metadata": positioning.get("research_metadata", {})
    #         }
            
    #         # File 3: Source tracking
    #         sources_data = {
    #             "detailed_sources": positioning.get("detailed_sources", []),
    #             "source_citations": positioning.get("source_citations", {}),
    #             "total_sources": len(positioning.get("detailed_sources", [])),
    #             "unique_domains": len(set(source.get("url", "").split("/")[2] for source in positioning.get("detailed_sources", []) if source.get("url"))),
    #             "collection_timestamp": datetime.now().isoformat() + "Z"
    #         }
            
    #         if hasattr(self.storage_manager, 'bucket'):
    #             # GCP storage
    #             if markdown_content:
    #                 content_blob = self.storage_manager.bucket.blob(f"accounts/{brand_domain}/research_phases/market_positioning.md")
    #                 content_blob.upload_from_string(
    #                     markdown_content,
    #                     content_type="text/markdown"
    #                 )
    #                 saved_files.append(f"accounts/{brand_domain}/research_phases/market_positioning.md")
    #                 logger.info(f"💾 Saved market positioning content to GCP")
                
    #             metadata_blob = self.storage_manager.bucket.blob(f"accounts/{brand_domain}/research_phases/market_positioning_metadata.json")
    #             metadata_blob.upload_from_string(
    #                 json.dumps(metadata, indent=2),
    #                 content_type="application/json"
    #             )
    #             saved_files.append(f"accounts/{brand_domain}/research_phases/market_positioning_metadata.json")
    #             logger.info(f"💾 Saved market positioning metadata to GCP")
                
    #             sources_blob = self.storage_manager.bucket.blob(f"accounts/{brand_domain}/research_phases/market_positioning_sources.json")
    #             sources_blob.upload_from_string(
    #                 json.dumps(sources_data, indent=2),
    #                 content_type="application/json"
    #             )
    #             saved_files.append(f"accounts/{brand_domain}/research_phases/market_positioning_sources.json")
    #             logger.info(f"💾 Saved market positioning sources to GCP")
                
    #         else:
    #             # Local storage
    #             import os
    #             research_dir = os.path.join(
    #                 self.storage_manager.base_dir,
    #                 "accounts", 
    #                 brand_domain, 
    #                 "research_phases"
    #             )
    #             os.makedirs(research_dir, exist_ok=True)
                
    #             # Save markdown content
    #             if markdown_content:
    #                 content_path = os.path.join(research_dir, "market_positioning.md")
    #                 with open(content_path, "w", encoding="utf-8") as f:
    #                     f.write(markdown_content)
    #                 saved_files.append(content_path)
    #                 logger.info(f"💾 Saved market positioning content: {content_path}")
                
    #             # Save metadata
    #             metadata_path = os.path.join(research_dir, "market_positioning_metadata.json")
    #             with open(metadata_path, "w", encoding="utf-8") as f:
    #                 json.dump(metadata, f, indent=2)
    #             saved_files.append(metadata_path)
    #             logger.info(f"💾 Saved market positioning metadata: {metadata_path}")
                
    #             # Save sources
    #             sources_path = os.path.join(research_dir, "market_positioning_sources.json")
    #             with open(sources_path, "w", encoding="utf-8") as f:
    #                 json.dump(sources_data, f, indent=2)
    #             saved_files.append(sources_path)
    #             logger.info(f"💾 Saved market positioning sources: {sources_path}")
            
    #     except Exception as e:
    #         logger.error(f"❌ Error saving market positioning research: {e}")
    #         raise
        
    #     return saved_files
    
    # async def _load_cached_results(self, brand_domain: str) -> Optional[Dict[str, Any]]:
    #     """Load cached market positioning research if available and not expired"""
        
    #     try:
    #         if hasattr(self.storage_manager, 'bucket'):
    #             # GCP storage
    #             metadata_blob = self.storage_manager.bucket.blob(f"accounts/{brand_domain}/research_phases/market_positioning_metadata.json")
    #             content_blob = self.storage_manager.bucket.blob(f"accounts/{brand_domain}/research_phases/market_positioning.md")
    #             sources_blob = self.storage_manager.bucket.blob(f"accounts/{brand_domain}/research_phases/market_positioning_sources.json")
                
    #             if not metadata_blob.exists():
    #                 return None
                
    #             # Load metadata
    #             metadata_content = metadata_blob.download_as_text()
    #             cached_metadata = json.loads(metadata_content)
    #             research_metadata = cached_metadata.get("research_metadata", {})
                
    #             # Check if cache is expired
    #             if research_metadata.get("cache_expires"):
    #                 expire_date = datetime.fromisoformat(research_metadata["cache_expires"].replace("Z", ""))
    #                 if datetime.now() > expire_date:
    #                     logger.info(f"🔄 Market positioning cache expired for {brand_domain}")
    #                     return None
                
    #             # Load content and sources
    #             content = ""
    #             if content_blob.exists():
    #                 content = content_blob.download_as_text()
                
    #             sources_data = {}
    #             if sources_blob.exists():
    #                 sources_content = sources_blob.download_as_text()
    #                 sources_data = json.loads(sources_content)
                
    #             if content:
    #                 logger.info(f"📋 Loaded cached market positioning research for {brand_domain}")
    #                 return {
    #                     "brand": brand_domain,
    #                     "market_positioning_content": content,
    #                     "quality_score": cached_metadata.get("confidence_score", 0.75),
    #                     "files": [
    #                         f"accounts/{brand_domain}/research_phases/market_positioning.md",
    #                         f"accounts/{brand_domain}/research_phases/market_positioning_metadata.json", 
    #                         f"accounts/{brand_domain}/research_phases/market_positioning_sources.json"
    #                     ],
    #                     "data_sources": sources_data.get("total_sources", 0),
    #                     "research_method": "cached_market_positioning"
    #                 }
                
    #         else:
    #             # Local storage
    #             import os
    #             research_dir = os.path.join(
    #                 self.storage_manager.base_dir, 
    #                 "accounts", 
    #                 brand_domain, 
    #                 "research_phases"
    #             )
                
    #             metadata_path = os.path.join(research_dir, "market_positioning_metadata.json")
    #             content_path = os.path.join(research_dir, "market_positioning.md")
    #             sources_path = os.path.join(research_dir, "market_positioning_sources.json")
                
    #             if not os.path.exists(metadata_path):
    #                 return None
                
    #             # Load metadata
    #             with open(metadata_path, "r", encoding="utf-8") as f:
    #                 cached_metadata = json.load(f)
                
    #             research_metadata = cached_metadata.get("research_metadata", {})
                
    #             # Check if cache is expired
    #             if research_metadata.get("cache_expires"):
    #                 expire_date = datetime.fromisoformat(research_metadata["cache_expires"].replace("Z", ""))
    #                 if datetime.now() > expire_date:
    #                     logger.info(f"🔄 Market positioning cache expired for {brand_domain}")
    #                     return None
                
    #             # Load content and sources
    #             content = ""
    #             if os.path.exists(content_path):
    #                 with open(content_path, "r", encoding="utf-8") as f:
    #                     content = f.read()
                
    #             sources_data = {}
    #             if os.path.exists(sources_path):
    #                 with open(sources_path, "r", encoding="utf-8") as f:
    #                     sources_data = json.load(f)
                
    #             if content:
    #                 logger.info(f"📋 Loaded cached market positioning research for {brand_domain}")
    #                 return {
    #                     "brand": brand_domain,
    #                     "market_positioning_content": content,
    #                     "quality_score": cached_metadata.get("confidence_score", 0.75),
    #                     "files": [
    #                         content_path,
    #                         metadata_path, 
    #                         sources_path
    #                     ],
    #                     "data_sources": sources_data.get("total_sources", 0),
    #                     "research_method": "cached_market_positioning"
    #                 }
            
    #     except Exception as e:
    #         logger.warning(f"⚠️ Error loading cached market positioning research for {brand_domain}: {e}")
        
    #     return None


def get_market_positioning_researcher(brand_domain: str) -> MarketPositioningResearcher:
    """Get market positioning researcher instance"""
    return MarketPositioningResearcher(brand_domain=brand_domain) 