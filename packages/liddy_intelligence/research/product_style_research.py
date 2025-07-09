"""
Enhanced Product & Style Intelligence Research Phase

Implements Phase 3 of the Brand Research Pipeline with catalog-enhanced analysis.
Analyzes product aesthetics, design philosophy, and technical specifications.

Provides both catalog-enhanced analysis (when product data available) and 
web-only analysis (fallback when no catalog).

Focus: Product design intelligence and aesthetic analysis
Cache Duration: 3 months (moderate stability)  
Research Time: 2-3 minutes
Quality Threshold: 8.5 (enhanced)
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

from liddy.llm.openai_service import OpenAIService
from liddy.prompt_manager import PromptManager
from liddy.llm.simple_factory import LLMFactory
from liddy_intelligence.progress_tracker import ProgressTracker, StepType, create_console_listener
from liddy_intelligence.web_search import TavilySearchProvider
from liddy.storage import get_account_storage_provider, AccountStorageProvider
from liddy.config import get_settings
from liddy_intelligence.research.base_researcher import BaseResearcher
from liddy_intelligence.research.data_sources import WebSearchDataSource, DataGatheringContext

logger = logging.getLogger(__name__)


class ProductStyleResearcher(BaseResearcher):
    """Product & Style Intelligence Research Phase Implementation"""
    
    def __init__(self, brand_domain: str, storage_manager=None):
        super().__init__(
            brand_domain=brand_domain,
            researcher_name="product_style",
            step_type=StepType.PRODUCT_STYLE,
            quality_threshold=8.5,
            cache_duration_days=90,
            storage_manager=storage_manager
        )
                
    # async def research_product_style(self, force_refresh: bool = False) -> Dict[str, Any]:
    #     """Research product style phase for a brand - ENHANCED VERSION with product catalog integration"""
        
    #     logger.info(f"🎨 Starting Enhanced Product & Style Intelligence Research for {brand_domain}")
        
    #     # Use the base class research method
    #     result = await self.research(force_refresh=force_refresh)
        
    #     # Transform result format to match expected CLI interface
    #     return {
    #         "brand": self.brand_domain,
    #         "content": result.get("content", ""),
    #         "quality_score": result.get("quality_score", 0.75),
    #         "files": result.get("files", []),
    #         "data_sources": result.get("data_sources", 0),
    #         "research_method": result.get("research_method", "enhanced_catalog_analysis")
    #     }

    async def _get_brand_name(self) -> str:
        """Get the brand name from account config or fallback to domain-derived name"""
        try:
            config = await self.storage_manager.get_account_config(self.brand_domain)
            if config and config.get("brand_name"):
                return config["brand_name"]
            else:
                # Fallback to domain-derived name
                return self.brand_domain.replace('.com', '').replace('.', ' ').title()
        except Exception as e:
            logger.warning(f"⚠️ Error getting brand name from config: {e}")
            return self.brand_domain.replace('.com', '').replace('.', ' ').title()

    async def _gather_data(self) -> Dict[str, Any]:
        """Gather comprehensive product style data using WebSearchDataSource"""
        
        # Get discovered brand name from account config
        brand_name = await self._get_brand_name()
        
        # Product style research queries using domain + brand name for optimal targeting
        research_queries = [
            f'{self.brand_domain} "{brand_name}" product line design philosophy aesthetic',
            f'{self.brand_domain} "{brand_name}" design language visual identity style guide',
            f'{self.brand_domain} "{brand_name}" product photography styling presentation',
            f'{self.brand_domain} "{brand_name}" collections seasonal themes design approach',
            f'{self.brand_domain} "{brand_name}" color palette typography visual elements',
            f'{self.brand_domain} "{brand_name}" product design innovation aesthetic principles',
            f'{self.brand_domain} "{brand_name}" aesthetic visual style design DNA',
            f'{self.brand_domain} "{brand_name}" packaging design visual presentation',
            f'{self.brand_domain} "{brand_name}" product categories lineup design consistency',
            f'{self.brand_domain} "{brand_name}" design awards recognition aesthetic innovation',
            f'{self.brand_domain} "{brand_name}" visual brand elements logo iconography',
            f'{self.brand_domain} "{brand_name}" design trends influence aesthetic direction'
        ]
        
        # Use WebSearchDataSource for data gathering
        try:
            web_search_source = WebSearchDataSource()
            
            if not web_search_source.is_available():
                error_msg = "ABORTING: Web search service not available."
                logger.error(f"🚨 {error_msg}")
                raise RuntimeError(error_msg)
            
            # Create data gathering context
            context = DataGatheringContext(
                brand_domain=self.brand_domain,
                researcher_name=self.researcher_name,
                phase_name="product_style"
            )
            
            # Gather data using the data source strategy
            search_result = await web_search_source.gather(research_queries, context)
            
            # 🚨 ABORT CONDITIONS - Don't continue with poor quality data
            total_searches = len(research_queries)
            success_rate = search_result.successful_searches / total_searches if total_searches > 0 else 0
            
            # Abort if we have SSL errors affecting most searches
            if search_result.ssl_errors >= 3:
                error_msg = f"ABORTING: SSL certificate verification failed for {search_result.ssl_errors} searches."
                logger.error(f"🚨 {error_msg}")
                raise RuntimeError(error_msg)
            
            # Abort if overall success rate is too low for quality research
            if success_rate < 0.3:
                error_msg = f"ABORTING: Only {search_result.successful_searches}/{total_searches} searches succeeded ({success_rate:.1%})."
                logger.error(f"🚨 {error_msg}")
                raise RuntimeError(error_msg)
            
            # Warn about reduced quality but continue if we have some data
            if success_rate < 0.7:
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
                }
            }
                
        except RuntimeError:
            raise
        except Exception as e:
            error_msg = f"ABORTING: Critical error in data gathering: {str(e)}"
            logger.error(f"🚨 {error_msg}")
            raise RuntimeError(error_msg)

    def _get_default_user_prompt(self) -> str:
        """Get the default user prompt for product style analysis - implements BaseResearcher abstract method"""
        
        default_prompt = """Analyze this product & style research data to extract comprehensive design intelligence.

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

## Product & Style Intelligence Analysis Requirements:

Please create a comprehensive product & style intelligence report in **markdown format**. When referencing information, cite your sources using the numbers provided (e.g., [1], [2], [3]).

Structure your analysis as follows:

# Product & Style Intelligence: {{brand_name}}

## 1. Design Philosophy & Aesthetic Principles
- **Core Design Philosophy:** [Fundamental design approach and principles] [cite sources]
- **Aesthetic DNA:** [Visual identity and style characteristics] [cite sources]
- **Design Evolution:** [How design language has evolved] [cite sources]
- **Innovation Approach:** [Design innovation methodology] [cite sources]

## 2. Product Line Analysis
- **Product Categories:** [Main product lines and categories] [cite sources]
- **Design Consistency:** [How design language spans products] [cite sources]
- **Product Hierarchy:** [Premium to entry-level design differentiation] [cite sources]
- **Seasonal Collections:** [How collections and themes are structured] [cite sources]

## 3. Visual Brand Elements
- **Color Palette:** [Primary and secondary color usage] [cite sources]
- **Typography:** [Font choices and text styling] [cite sources]
- **Logo & Iconography:** [Brand symbols and visual marks] [cite sources]
- **Visual Style:** [Photography, illustration, and visual treatment] [cite sources]

## 4. Product Presentation & Styling
- **Product Photography:** [Visual presentation approach] [cite sources]
- **Packaging Design:** [Package aesthetics and approach] [cite sources]
- **Display & Merchandising:** [How products are showcased] [cite sources]
- **Digital Presentation:** [Online visual strategy] [cite sources]

## 5. Design Recognition & Innovation
- **Design Awards:** [Recognition for design excellence] [cite sources]
- **Industry Influence:** [Design leadership and trendsetting] [cite sources]
- **Innovation Highlights:** [Notable design innovations] [cite sources]
- **Design Partnerships:** [Collaborations and design alliances] [cite sources]

## 6. Style Trends & Direction
- **Current Trends:** [Current design trends being followed] [cite sources]
- **Future Direction:** [Emerging design directions] [cite sources]
- **Cultural Influences:** [Cultural and social influences on design] [cite sources]
- **Competitive Differentiation:** [How design sets brand apart] [cite sources]

## Analysis Quality & Confidence

**Data Sources:** {{total_sources}} search results analyzed
**Search Success Rate:** {{success_rate}}
**Information Quality:** {{information_quality}}
**Confidence Level:** {{confidence_level}} confidence in findings
**Key Gaps:** [Note any information that was missing or unclear due to limited data availability]

## Summary

[Provide a 2-3 sentence executive summary of the brand's design philosophy and style approach]

## Sources

{{source_reference_guide}}

---

**Important Instructions:**
- **ALWAYS cite sources** using the provided reference numbers [1], [2], [3], etc.
- Focus on factual, verifiable information only
- Clearly distinguish between official design statements and external analysis  
- Note confidence levels for different claims based on data availability
- If information is missing, clearly state "Not available in research data"
- Given the {{data_quality_text}} data quality, be appropriately cautious in claims
- Use markdown formatting for structure and readability
- Include the complete sources list at the end"""

        return default_prompt

    def _get_default_instruction_prompt(self) -> str:
        """Get the default instruction prompt for product style analysis - implements BaseResearcher abstract method"""
        
        return "You are an expert design researcher specializing in product aesthetics and style analysis. Generate comprehensive, well-structured markdown reports with proper source citations based on research data. Always cite your sources using the provided reference numbers. Adjust confidence levels based on data quality."

    # async def _analyze_data(self, brand_domain: str, style_data: Dict[str, Any]) -> Dict[str, Any]:
    #     """Analyze product style data using LLM with Langfuse prompt management"""
        
    #     total_sources = style_data.get("total_sources", 0)
    #     search_stats = style_data.get("search_stats", {})
    #     success_rate = search_stats.get("success_rate", 0)
        
    #     if total_sources == 0:
    #         error_msg = "ANALYSIS ABORTED: No search results available for analysis."
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
        
    #     for result in style_data["search_results"][:20]:  # Use top 20 results
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
    #         for result, citation in zip(style_data["search_results"][:20], source_citations.values())
    #     ])
        
    #     # Get or create Langfuse prompt for product style analysis
    #     prompt_manager = PromptManager.get_prompt_manager()
        
    #     # Default comprehensive product style analysis prompt
    #     default_prompt = self._get_default_user_prompt()

    #     try:
    #         # Get the Langfuse prompt (create if doesn't exist)
    #         langfuse_prompt = await prompt_manager.get_prompt(
    #             prompt_name=f"internal/research/{self.researcher_name}/intelligence_analysis",
    #             default_prompt=default_prompt
    #         )
            
    #         if not langfuse_prompt:
    #             error_msg = "ANALYSIS FAILED: Could not load or create Langfuse prompt"
    #             logger.error(f"🚨 {error_msg}")
    #             raise RuntimeError(error_msg)
            
    #         # Prepare template variables
    #         brand_name = style_data.get('brand_name', brand_domain)
    #         data_quality = "High" if success_rate > 0.7 else "Medium" if success_rate > 0.5 else "Limited"
    #         information_quality = f"High quality with comprehensive data" if success_rate > 0.7 else f"Medium quality with adequate data" if success_rate > 0.5 else f"Limited quality with minimal data"
    #         confidence_level = "High" if success_rate > 0.7 and total_sources >= 20 else "Medium" if success_rate > 0.5 and total_sources >= 10 else "Low"
    #         data_quality_level = "high" if success_rate > 0.7 else "medium" if success_rate > 0.5 else "limited"
            
    #         # Compile the final prompt with template substitution
    #         prompt_content = langfuse_prompt.prompt
            
    #         # Replace template variables
    #         prompt_variables = {
    #             "brand_name": brand_name,
    #             "brand_domain": brand_domain,
    #             "total_sources": total_sources,
    #             "success_rate": f"{success_rate:.1%}",
    #             "data_quality": data_quality,
    #             "search_context": search_context,
    #             "source_reference_guide": source_reference_guide,
    #             "information_quality": information_quality,
    #             "confidence_level": confidence_level,
    #             "data_quality_level": data_quality_level
    #         }
            
    #         # Simple template substitution
    #         for key, value in prompt_variables.items():
    #             prompt_content = prompt_content.replace(f"{{{{{key}}}}}", str(value))
            
    #         logger.info(f"🧠 Analyzing {total_sources} sources with O3 model using Langfuse prompt (success rate: {success_rate:.1%})")
            
    #         # Execute LLM analysis with the Langfuse-managed prompt
    #         response = await LLMFactory.chat_completion(
    #             task=f"brand_research_{self.researcher_name}",
    #             system=self._get_default_instruction_prompt(),
    #             messages=[{
    #                 "role": "user",
    #                 "content": prompt_content
    #             }],
    #             temperature=0.1
    #         )
            
    #         if response and response.get("content"):
    #             # Adjust confidence based on data quality
    #             base_confidence = 0.85  # Higher base for full analysis
    #             if success_rate < 0.5:
    #                 base_confidence = 0.6
    #             elif success_rate < 0.7:
    #                 base_confidence = 0.75
                
    #             if total_sources < 10:
    #                 base_confidence *= 0.9
    #             elif total_sources < 5:
    #                 base_confidence *= 0.8
                
    #             logger.info(f"✅ Generated comprehensive product style analysis ({len(response['content'])} chars, confidence: {base_confidence:.2f})")
                
    #             return {
    #                 "markdown": response["content"],
    #                 "analysis_method": "langfuse_prompt_comprehensive_analysis",
    #                 "confidence": base_confidence,
    #                 "data_sources": total_sources,
    #                 "search_success_rate": success_rate,
    #                 "detailed_sources": style_data.get('detailed_sources', []),
    #                 "source_citations": source_citations,
    #                 "search_stats": search_stats,
    #                 "langfuse_prompt_used": langfuse_prompt.name if hasattr(langfuse_prompt, 'name') else "intelligence_analysis"
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
    
    # async def _synthesize_product_style_intelligence(self, brand_domain: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
    #     """Synthesize final product style intelligence with enhanced analysis tracking"""
        
    #     return {
    #         "content": analysis["markdown"],
    #         "confidence_score": analysis["confidence"],
    #         "data_quality": "high" if analysis["search_success_rate"] > 0.7 else "medium" if analysis["search_success_rate"] > 0.5 else "limited",
    #         "data_sources_count": analysis["data_sources"],
    #         "analysis_method": analysis["analysis_method"],
    #         "detailed_sources": analysis["detailed_sources"],
    #         "source_citations": analysis["source_citations"],
    #         "langfuse_prompt_used": analysis.get("langfuse_prompt_used", "unknown"),
    #         "content_length": len(analysis["markdown"]),
    #         "has_source_citations": len(analysis["source_citations"]) > 0
    #     }
    
    # async def _save_product_style_research(self, brand_domain: str, style: Dict[str, Any]) -> List[str]:
    #     """Save product style research using foundation research pattern"""
        
    #     saved_files = []
        
    #     try:
    #         markdown_content = style.get("content", "")
            
    #         metadata = {
    #             "phase": "product_style",
    #             "confidence_score": style.get("confidence_score", 0.75),
    #             "data_quality": style.get("data_quality", "medium"),
    #             "data_sources_count": style.get("data_sources_count", 0),
    #             "analysis_method": style.get("analysis_method", "unknown"),
    #             "research_metadata": style.get("research_metadata", {})
    #         }
            
    #         sources_data = {
    #             "detailed_sources": style.get("detailed_sources", []),
    #             "source_citations": style.get("source_citations", {}),
    #             "total_sources": len(style.get("detailed_sources", [])),
    #             "collection_timestamp": datetime.now().isoformat() + "Z"
    #         }
            
    #         if hasattr(self.storage_manager, 'bucket'):
    #             # GCP storage
    #             if markdown_content:
    #                 content_blob = self.storage_manager.bucket.blob(f"accounts/{brand_domain}/research_phases/product_style.md")
    #                 content_blob.upload_from_string(markdown_content, content_type="text/markdown")
    #                 saved_files.append(f"accounts/{brand_domain}/research_phases/product_style.md")
                
    #             metadata_blob = self.storage_manager.bucket.blob(f"accounts/{brand_domain}/research_phases/product_style_metadata.json")
    #             metadata_blob.upload_from_string(json.dumps(metadata, indent=2), content_type="application/json")
    #             saved_files.append(f"accounts/{brand_domain}/research_phases/product_style_metadata.json")
                
    #             sources_blob = self.storage_manager.bucket.blob(f"accounts/{brand_domain}/research_phases/product_style_sources.json")
    #             sources_blob.upload_from_string(json.dumps(sources_data, indent=2), content_type="application/json")
    #             saved_files.append(f"accounts/{brand_domain}/research_phases/product_style_sources.json")
                
    #         else:
    #             # Local storage
    #             research_dir = os.path.join(self.storage_manager.base_dir, "accounts", brand_domain, "research_phases")
    #             os.makedirs(research_dir, exist_ok=True)
                
    #             if markdown_content:
    #                 content_path = os.path.join(research_dir, "product_style.md")
    #                 with open(content_path, "w", encoding="utf-8") as f:
    #                     f.write(markdown_content)
    #                 saved_files.append(content_path)
    #                 logger.info(f"💾 Saved product style content: {content_path}")
                
    #             metadata_path = os.path.join(research_dir, "product_style_metadata.json")
    #             with open(metadata_path, "w", encoding="utf-8") as f:
    #                 json.dump(metadata, f, indent=2)
    #             saved_files.append(metadata_path)
                
    #             sources_path = os.path.join(research_dir, "product_style_sources.json")
    #             with open(sources_path, "w", encoding="utf-8") as f:
    #                 json.dump(sources_data, f, indent=2)
    #             saved_files.append(sources_path)
            
    #     except Exception as e:
    #         logger.error(f"❌ Error saving product style research: {e}")
    #         raise
        
    #     return saved_files
    
    # async def _load_cached_product_style(self, brand_domain: str) -> Optional[Dict[str, Any]]:
    #     """Load cached product style research"""
        
    #     try:
    #         if hasattr(self.storage_manager, 'bucket'):
    #             # GCP storage
    #             metadata_blob = self.storage_manager.bucket.blob(f"accounts/{brand_domain}/research_phases/product_style_metadata.json")
    #             if not metadata_blob.exists():
    #                 return None
                    
    #             metadata_content = metadata_blob.download_as_text()
    #             metadata = json.loads(metadata_content)
                
    #             # Check cache expiry
    #             research_metadata = metadata.get("research_metadata", {})
    #             if research_metadata.get("cache_expires"):
    #                 expire_date = datetime.fromisoformat(research_metadata["cache_expires"].replace("Z", ""))
    #                 if datetime.now() > expire_date:
    #                     return None
                
    #             content_blob = self.storage_manager.bucket.blob(f"accounts/{brand_domain}/research_phases/product_style.md")
    #             if content_blob.exists():
    #                 content = content_blob.download_as_text()
    #                 return {
    #                     "brand": brand_domain,
    #                     "content": content,
    #                     "quality_score": metadata.get("confidence_score", 0.75),
    #                     "files": [],
    #                     "data_sources": 0,
    #                     "research_method": "cached_product_style"
    #                 }
                
    #         else:
    #             # Local storage
    #             research_dir = os.path.join(self.storage_manager.base_dir, "accounts", brand_domain, "research_phases")
    #             metadata_path = os.path.join(research_dir, "product_style_metadata.json")
                
    #             if not os.path.exists(metadata_path):
    #                 return None
                
    #             with open(metadata_path, "r", encoding="utf-8") as f:
    #                 metadata = json.load(f)
                
    #             research_metadata = metadata.get("research_metadata", {})
    #             if research_metadata.get("cache_expires"):
    #                 expire_date = datetime.fromisoformat(research_metadata["cache_expires"].replace("Z", ""))
    #                 if datetime.now() > expire_date:
    #                     return None
                
    #             content_path = os.path.join(research_dir, "product_style.md")
    #             if os.path.exists(content_path):
    #                 with open(content_path, "r", encoding="utf-8") as f:
    #                     content = f.read()
    #                 return {
    #                     "brand": brand_domain,
    #                     "content": content,
    #                     "quality_score": metadata.get("confidence_score", 0.75),
    #                     "files": [content_path],
    #                     "data_sources": 0,
    #                     "research_method": "cached_product_style"
    #                 }
            
    #     except Exception as e:
    #         logger.warning(f"⚠️ Error loading cached product style research for {brand_domain}: {e}")
        
    #     return None

    async def research(
        self, 
        force_refresh: bool = False,
        improvement_feedback: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Enhanced research comprehensive product and style intelligence with feedback integration.
        
        Args:
            force_refresh: Force refresh of cached results
            improvement_feedback: Optional feedback from previous quality evaluation for iterative improvement
            
        Returns:
            Dictionary with research results and metadata
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
        session_id = f"product_style_{int(time.time())}"
        
        # Handle improvement feedback
        feedback_context = ""
        if improvement_feedback:
            logger.info(f"📋 Incorporating {len(improvement_feedback)} improvement suggestions")
            feedback_context = self._format_improvement_feedback(improvement_feedback)
        
        # Initialize progress tracker with correct parameters
        tracker = ProgressTracker(
            storage_manager=self.storage_manager,
            enable_checkpoints=True
        )
        
        # Create and start tracking step
        step_id = await tracker.create_step(
            step_type=StepType.PRODUCT_STYLE,
            brand=self.brand_domain,
            phase_name="product_style",
            total_operations=8
        )
        await tracker.start_step(step_id, "Initializing product style research")
        
        try:
            # Check cache
            await tracker.update_progress(step_id, 1, "Checking for existing research")
            
            cached_result = await self._check_cache(self.brand_domain, force_refresh)
            if cached_result:
                logger.info(f"Using cached product style research for {self.brand_domain}")
                
                await tracker.complete_step(
                    step_id,
                    output_files=[],
                    quality_score=cached_result.get("metadata", {}).get("confidence", 0.75),
                    cache_hit=True
                )
                return cached_result
            
            # Load foundation research for context
            await tracker.update_progress(step_id, 2, "Loading foundation research context")
            foundation_context = await self._load_foundation_context(self.brand_domain)
            
            # ENHANCED: Load product catalog for analysis
            await tracker.update_progress(step_id, 3, "Loading product catalog")
            product_catalog = await self.storage_manager.get_product_catalog(self.brand_domain)
            
            # Phase-specific web research
            await tracker.update_progress(step_id, 4, "Conducting product-focused web research")
            web_research_data = await self._conduct_product_research(self.brand_domain, tracker)
            
            # ENHANCED: Comprehensive product analysis with feedback integration
            await tracker.update_progress(step_id, 5, "Analyzing product catalog and design patterns")
            
            if product_catalog:
                logger.info(f"Analyzing {len(product_catalog)} products from catalog")
                analysis_result = await self._analyze_with_product_catalog_and_feedback(
                    foundation_context=foundation_context,
                    web_research_data=web_research_data,
                    product_catalog=product_catalog,
                    feedback_context=feedback_context,
                    tracker=tracker
                )
            else:
                logger.info("No product catalog available, using web research only")
                analysis_result = await self._analyze_without_product_catalog_and_feedback(
                    foundation_context=foundation_context,
                    web_research_data=web_research_data,
                    feedback_context=feedback_context,
                    tracker=tracker
                )
            
            # Save results
            await tracker.update_progress(step_id, 7, "Saving research results")
            await self._save_results(analysis_result)
            
            # Complete session
            await tracker.update_progress(step_id, 8, "Finalizing research")
            await tracker.complete_step(
                step_id,
                output_files=[],
                quality_score=analysis_result['metadata']['confidence'],
                cache_hit=False
            )
            
            # Add feedback metadata
            analysis_result['metadata']['feedback_incorporated'] = len(improvement_feedback) if improvement_feedback else 0
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in product style research: {e}")
            
            tracker.fail_step(step_id, str(e))
            
            return {
                "error": str(e),
                "metadata": {
                    "success": False,
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                    "feedback_incorporated": len(improvement_feedback) if improvement_feedback else 0
                }
            }
    
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
    
    async def _analyze_with_product_catalog_and_feedback(
        self,
        foundation_context: Dict[str, Any],
        web_research_data: Dict[str, Any],
        product_catalog: List[Dict[str, Any]],
        feedback_context: str,
        tracker: ProgressTracker
    ) -> Dict[str, Any]:
        """
        Enhanced analysis using actual product catalog data with feedback integration.
        """
        # Analyze product catalog structure
        catalog_analysis = await self._analyze_product_catalog(product_catalog)
        
        # Get enhanced prompt for catalog-based analysis with feedback
        prompt_template = await self._get_catalog_analysis_prompt_with_feedback(feedback_context)
        
        # Prepare comprehensive analysis data
        analysis_data = {
            "brand_domain": self.brand_domain,
            "foundation_context": foundation_context,
            "web_research": web_research_data,
            "product_catalog": {
                "total_products": len(product_catalog),
                "structure_analysis": catalog_analysis,
                "sample_products": product_catalog[:10],  # First 10 products for analysis
                "category_breakdown": self._analyze_categories(product_catalog),
                "price_analysis": self._analyze_pricing(product_catalog),
                "feature_patterns": self._analyze_features(product_catalog)
            }
        }
        
        # Generate enhanced analysis using LLM with feedback
        analysis_content = await self._generate_enhanced_analysis_with_feedback(
            prompt_template, 
            analysis_data, 
            feedback_context,
            tracker
        )
        
        # Calculate quality score based on available data richness and feedback integration
        quality_score = self._calculate_enhanced_quality_score(
            web_research_data, 
            product_catalog, 
            analysis_content
        )
        
        # Boost quality score if feedback was incorporated
        if feedback_context:
            quality_score = min(0.95, quality_score * 1.1)
        
        return {
            "content": analysis_content,
            "sources": web_research_data.get("sources", []),
            "metadata": {
                "confidence": quality_score,
                "source_count": len(web_research_data.get("sources", [])),
                "product_count": len(product_catalog),
                "analysis_type": "catalog_enhanced_with_feedback" if feedback_context else "catalog_enhanced",
                "timestamp": datetime.now().isoformat(),
                "feedback_incorporated": bool(feedback_context)
            }
        }
    
    async def _analyze_without_product_catalog_and_feedback(
        self,
        foundation_context: Dict[str, Any],
        web_research_data: Dict[str, Any],
        feedback_context: str,
        tracker: ProgressTracker
    ) -> Dict[str, Any]:
        """
        Fallback analysis using web research only with feedback integration.
        """
        # Get standard prompt for web-only analysis with feedback
        prompt_template = await self._get_web_analysis_prompt_with_feedback(feedback_context)
        
        # Prepare analysis data
        analysis_data = {
            "brand_domain": self.brand_domain,
            "foundation_context": foundation_context,
            "web_research": web_research_data,
            "note": "Limited to web research - product catalog not available"
        }
        
        # Generate analysis using LLM with feedback
        analysis_content = await self._generate_standard_analysis_with_feedback(
            prompt_template, 
            analysis_data,
            feedback_context,
            tracker
        )
        
        # Calculate standard quality score
        quality_score = self._calculate_standard_quality_score(
            web_research_data, 
            analysis_content
        )
        
        # Boost quality score if feedback was incorporated
        if feedback_context:
            quality_score = min(0.95, quality_score * 1.1)
        
        return {
            "content": analysis_content,
            "sources": web_research_data.get("sources", []),
            "metadata": {
                "confidence": quality_score,
                "source_count": len(web_research_data.get("sources", [])),
                "product_count": 0,
                "analysis_type": "web_only_with_feedback" if feedback_context else "web_only",
                "timestamp": datetime.now().isoformat(),
                "feedback_incorporated": bool(feedback_context)
            }
        }
    
    async def _get_catalog_analysis_prompt_with_feedback(self, feedback_context: str) -> str:
        """Get enhanced prompt template for catalog-based analysis with feedback integration."""
        
        base_prompt = await self._get_catalog_analysis_prompt()
        
        # Add feedback context if available
        if feedback_context:
            enhanced_prompt = base_prompt + feedback_context
            enhanced_prompt += "\n\nEnsure your analysis specifically addresses the improvement feedback above to enhance quality and completeness."
            return enhanced_prompt
        
        return base_prompt
    
    async def _get_web_analysis_prompt_with_feedback(self, feedback_context: str) -> str:
        """Get standard prompt template for web-only analysis with feedback integration."""
        
        base_prompt = await self._get_web_analysis_prompt()
        
        # Add feedback context if available
        if feedback_context:
            enhanced_prompt = base_prompt + feedback_context
            enhanced_prompt += "\n\nEnsure your analysis specifically addresses the improvement feedback above to enhance quality and completeness."
            return enhanced_prompt
        
        return base_prompt
    
    async def _generate_enhanced_analysis_with_feedback(
        self, 
        prompt_template: str, 
        analysis_data: Dict[str, Any], 
        feedback_context: str,
        tracker: ProgressTracker
    ) -> str:
        """Generate comprehensive analysis using all research phases with feedback integration"""
        
        # Log feedback integration
        if feedback_context:
            logger.info("📋 Including improvement feedback in enhanced analysis prompt")
        
        # Prepare template variables (same as original method)
        template_vars = {
            "brand_domain": analysis_data["brand_domain"],
            "foundation_summary": str(analysis_data.get("foundation_context", {}))[:1000],
            "web_source_count": len(analysis_data.get("web_research", {}).get("sources", [])),
            "product_count": analysis_data.get("product_catalog", {}).get("total_products", 0),
            "available_fields": ", ".join(analysis_data.get("product_catalog", {}).get("structure_analysis", {}).get("available_fields", [])),
            "categories": str(analysis_data.get("product_catalog", {}).get("category_breakdown", {}))[:500],
            "price_range": str(analysis_data.get("product_catalog", {}).get("price_analysis", {}))[:300],
            "feature_patterns": str(analysis_data.get("product_catalog", {}).get("feature_patterns", {}))[:300],
            "sample_products": str(analysis_data.get("product_catalog", {}).get("sample_products", []))[:1000]
        }
        
        # Replace template variables
        final_prompt = prompt_template
        for var, value in template_vars.items():
            final_prompt = final_prompt.replace(f"{{{{{var}}}}}", str(value))
        
        response = await LLMFactory.chat_completion(
            task="product_style_research",
            system="You are an expert product and style analyst specializing in comprehensive brand intelligence research. Generate detailed, well-structured analysis with specific examples and actionable insights.",
            messages=[
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.3
        )
        
        return response.get("content", "Product style analysis generation failed")
    
    async def _generate_standard_analysis_with_feedback(
        self, 
        prompt_template: str, 
        analysis_data: Dict[str, Any],
        feedback_context: str,
        tracker: ProgressTracker
    ) -> str:
        """Generate standard analysis using web research only with feedback integration"""
        
        # Log feedback integration
        if feedback_context:
            logger.info("📋 Including improvement feedback in standard analysis prompt")
        
        # Prepare template variables (same as original method)
        template_vars = {
            "brand_domain": analysis_data["brand_domain"],
            "foundation_summary": str(analysis_data.get("foundation_context", {}))[:1000],
            "web_source_count": len(analysis_data.get("web_research", {}).get("sources", [])),
            "note": analysis_data.get("note", "")
        }
        
        # Replace template variables
        final_prompt = prompt_template
        for var, value in template_vars.items():
            final_prompt = final_prompt.replace(f"{{{{{var}}}}}", str(value))
        
        response = await LLMFactory.chat_completion(
            task="product_style_research",
            system="You are an expert product and style analyst specializing in web-based brand research. Generate comprehensive analysis based on available web sources with clear limitations noted.",
            messages=[
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.3
        )
        
        return response.get("content", "Product style analysis generation failed")

    async def _analyze_product_catalog(self, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the structure and patterns in the product catalog."""
        
        if not products:
            return {"error": "Empty product catalog"}
        
        # Analyze data structure
        sample_product = products[0]
        available_fields = list(sample_product.keys())
        
        # Common analysis patterns
        analysis = {
            "total_products": len(products),
            "available_fields": available_fields,
            "data_completeness": {},
            "value_patterns": {}
        }
        
        # Analyze field completeness
        for field in available_fields:
            non_empty_count = sum(1 for p in products if p.get(field) and str(p.get(field)).strip())
            analysis["data_completeness"][field] = {
                "filled_count": non_empty_count,
                "completion_rate": non_empty_count / len(products)
            }
        
        # Analyze common value patterns for key fields
        key_fields = ["category", "type", "collection", "color", "material", "price", "name", "title"]
        for field in key_fields:
            if field in available_fields:
                values = [str(p.get(field, "")).lower().strip() for p in products if p.get(field)]
                unique_values = list(set(values))
                analysis["value_patterns"][field] = {
                    "unique_count": len(unique_values),
                    "sample_values": unique_values[:10]  # First 10 unique values
                }
        
        return analysis
    
    async def _get_catalog_analysis_prompt(self) -> str:
        """Get enhanced prompt template for catalog-based analysis."""
        
        default_prompt = """
You are conducting comprehensive Product & Style Intelligence research for {{brand_domain}}.

You have access to:
1. Foundation research context
2. Web research data  
3. **ACTUAL PRODUCT CATALOG** with {{product_count}} products

Your task is to analyze the brand's product design philosophy, aesthetic direction, and technical specifications using this comprehensive data.

**CONTEXT:**
Foundation Research: {{foundation_summary}}
Web Research Sources: {{web_source_count}} sources analyzed
Product Catalog: {{product_count}} products with {{available_fields}} data fields

**PRODUCT CATALOG ANALYSIS:**
Categories: {{categories}}
Price Range: {{price_range}}
Key Features: {{feature_patterns}}
Sample Products: {{sample_products}}

**ANALYSIS REQUIREMENTS:**

Generate a comprehensive analysis covering these 6 sections:

## 1. Design Philosophy & Aesthetic Direction
Analyze the brand's core design principles based on actual product data:
- Visual design patterns across product lines
- Material choices and quality indicators  
- Color palettes and style consistency
- Innovation in design approach
*Cite specific examples from the product catalog [1], [2], etc.*

## 2. Product Line Architecture
Map the brand's product strategy:
- Category organization and hierarchy
- Product line relationships and positioning
- Target market segmentation through products
- Product lifecycle and seasonal patterns
*Reference actual product data and categorization [3], [4], etc.*

## 3. Technical Specifications & Quality Standards
Document technical approach:
- Manufacturing quality indicators
- Technical innovation areas
- Performance specifications
- Quality control standards
*Use specific examples from product specifications [5], [6], etc.*

## 4. Pricing Strategy & Value Positioning
Analyze pricing intelligence:
- Price point distribution across categories
- Value proposition communication
- Competitive positioning
- Premium vs. accessible product tiers
*Reference actual pricing data [7], [8], etc.*

## 5. Style Evolution & Trends
Track design evolution:
- Historical style progression
- Current trend adoption
- Innovation leadership areas
- Style differentiation strategies
*Combine web research with product patterns [9], [10], etc.*

## 6. Brand Consistency & Guidelines
Assess brand coherence:
- Visual consistency across products
- Brand expression through design
- Style guide adherence
- Brand differentiation elements
*Use both web research and catalog analysis [11], [12], etc.*

**SOURCING REQUIREMENTS:**
- Use numbered citations [1], [2], [3] for all facts
- Prioritize product catalog insights over web speculation
- Include specific product examples and data points
- Acknowledge data limitations honestly

**OUTPUT FORMAT:**
Write professional analysis suitable for client delivery.
Focus on actionable insights and specific examples.
Be honest about data quality and availability.
"""

        prompt = await self.prompt_manager.get_prompt(
            "internal/product_style_catalog_analysis",
            default_prompt
        )
        
        return prompt.prompt if prompt else default_prompt
    
    async def _get_web_analysis_prompt(self) -> str:
        """Get standard prompt template for web-only analysis."""
        
        default_prompt = """
You are conducting Product & Style Intelligence research for {{brand_domain}}.

**NOTE: This analysis is limited to web research only - product catalog data is not available.**

Your task is to analyze the brand's product design philosophy and style direction using available web sources.

**CONTEXT:**
Foundation Research: {{foundation_summary}}
Web Research Sources: {{web_source_count}} sources analyzed

**ANALYSIS REQUIREMENTS:**

Generate analysis covering these 6 sections:

## 1. Design Philosophy & Aesthetic Direction
## 2. Product Line Architecture  
## 3. Technical Specifications & Quality Standards
## 4. Pricing Strategy & Value Positioning
## 5. Style Evolution & Trends
## 6. Brand Consistency & Guidelines

**LIMITATIONS:**
- Analysis based on web research only
- May lack specific product details
- Cannot provide comprehensive catalog insights
- Recommendations may be more general

**SOURCING REQUIREMENTS:**
- Use numbered citations [1], [2], [3] for all facts
- Be honest about data limitations
- Focus on what can be determined from web sources
"""

        prompt = await self.prompt_manager.get_prompt(
            "internal/product_style_web_analysis", 
            default_prompt
        )
        
        return prompt.prompt if prompt else default_prompt

    def _analyze_categories(self, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze product categories and hierarchy."""
        category_fields = ["category", "type", "collection", "department", "section"]
        
        category_analysis = {}
        
        for field in category_fields:
            if any(p.get(field) for p in products):
                values = [str(p.get(field, "")).strip() for p in products if p.get(field)]
                category_analysis[field] = {
                    "unique_count": len(set(values)),
                    "top_categories": list(set(values))[:10],
                    "distribution": {}
                }
                
                # Count distribution
                for value in values:
                    category_analysis[field]["distribution"][value] = category_analysis[field]["distribution"].get(value, 0) + 1
        
        return category_analysis

    def _analyze_pricing(self, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze pricing patterns in the product catalog."""
        price_fields = ["price", "cost", "retail_price", "sale_price", "msrp"]
        
        pricing_analysis = {
            "available_price_fields": [],
            "price_ranges": {},
            "pricing_tiers": {}
        }
        
        for field in price_fields:
            prices = []
            for product in products:
                price_value = product.get(field)
                if price_value:
                    # Try to extract numeric value
                    try:
                        # Handle different price formats
                        price_str = str(price_value).replace("$", "").replace(",", "").strip()
                        if price_str and price_str.replace(".", "").isdigit():
                            prices.append(float(price_str))
                    except (ValueError, AttributeError):
                        continue
            
            if prices:
                pricing_analysis["available_price_fields"].append(field)
                pricing_analysis["price_ranges"][field] = {
                    "min": min(prices),
                    "max": max(prices),
                    "average": sum(prices) / len(prices),
                    "count": len(prices)
                }
                
                # Create pricing tiers
                sorted_prices = sorted(prices)
                if len(sorted_prices) >= 3:
                    pricing_analysis["pricing_tiers"][field] = {
                        "entry_level": sorted_prices[len(sorted_prices) // 3],
                        "mid_range": sorted_prices[2 * len(sorted_prices) // 3],
                        "premium": sorted_prices[-1]
                    }
        
        return pricing_analysis

    def _analyze_features(self, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze product features and specifications."""
        feature_fields = [
            "features", "specifications", "description", "details", 
            "material", "color", "size", "dimensions", "weight"
        ]
        
        feature_analysis = {
            "available_feature_fields": [],
            "common_materials": {},
            "color_patterns": {},
            "feature_keywords": {}
        }
        
        for field in feature_fields:
            if any(p.get(field) for p in products):
                feature_analysis["available_feature_fields"].append(field)
                
                # Analyze material patterns
                if field == "material":
                    materials = [str(p.get(field, "")).lower().strip() for p in products if p.get(field)]
                    material_counts = {}
                    for material in materials:
                        material_counts[material] = material_counts.get(material, 0) + 1
                    feature_analysis["common_materials"] = dict(sorted(material_counts.items(), key=lambda x: x[1], reverse=True)[:10])
                
                # Analyze color patterns
                elif field == "color":
                    colors = [str(p.get(field, "")).lower().strip() for p in products if p.get(field)]
                    color_counts = {}
                    for color in colors:
                        color_counts[color] = color_counts.get(color, 0) + 1
                    feature_analysis["color_patterns"] = dict(sorted(color_counts.items(), key=lambda x: x[1], reverse=True)[:15])
        
        return feature_analysis

    async def _load_foundation_context(self, brand_domain: str) -> Dict[str, Any]:
        """Load foundation research context for enhanced analysis."""
        try:
            # Check if we're using local storage (has base_dir) or GCP storage (has bucket)
            if hasattr(self.storage_manager, 'base_dir'):
                # Local storage
                foundation_path = os.path.join(
                    self.storage_manager.base_dir, 
                    "accounts", 
                    brand_domain, 
                    "research", 
                    "foundation",
                    "research.md"
                )
                
                if os.path.exists(foundation_path):
                    with open(foundation_path, "r") as f:
                        content = f.read()
                    
                    # Extract key insights for context
                    return {
                        "available": True,
                        "summary": content[:1000] + "..." if len(content) > 1000 else content,
                        "length": len(content)
                    }
                else:
                    return {"available": False, "note": "Foundation research not found"}
                    
            elif hasattr(self.storage_manager, 'bucket'):
                # GCP storage
                blob = self.storage_manager.bucket.blob(f"accounts/{brand_domain}/research_phases/foundation_research.md")
                if blob.exists():
                    content = blob.download_as_text()
                    
                    # Extract key insights for context
                    return {
                        "available": True,
                        "summary": content[:1000] + "..." if len(content) > 1000 else content,
                        "length": len(content)
                    }
                else:
                    return {"available": False, "note": "Foundation research not found"}
            else:
                return {"available": False, "error": "Unknown storage provider type"}
                
        except Exception as e:
            logger.warning(f"Could not load foundation context: {e}")
            return {"available": False, "error": str(e)}

    async def _conduct_product_research(self, brand_domain: str, tracker: ProgressTracker) -> Dict[str, Any]:
        """Conduct product-focused web research."""
        search_queries = [
            f"{brand_domain} product design philosophy",
            f"{brand_domain} product line architecture",
            f"{brand_domain} design aesthetic principles", 
            f"{brand_domain} product quality standards",
            f"{brand_domain} pricing strategy analysis",
            f"{brand_domain} brand style guidelines"
        ]
        
        all_sources = []
        search_results = {}
        
        for i, query in enumerate(search_queries):
            try:
                # await tracker.log_checkpoint(f"search_{i+1}", f"Searching: {query}")
                
                if not self.web_search:
                    logger.warning(f"Web search not available for '{query}'")
                    search_results[query] = {"error": "Web search service not available"}
                    continue
                
                results = await self.web_search.search(
                    query=query,
                    max_results=5,
                    search_depth="basic"
                )
                
                search_results[query] = results
                # Fix: Convert SearchResult objects to dictionaries before storing
                if isinstance(results, dict) and results.get("sources"):
                    # Convert sources to dict format
                    for source in results["sources"]:
                        if hasattr(source, '__dict__'):
                            # SearchResult object - convert to dict
                            source_dict = {}
                            for attr in ['title', 'url', 'snippet', 'score', 'source_query', 'source_id']:
                                if hasattr(source, attr):
                                    source_dict[attr] = getattr(source, attr)
                            all_sources.append(source_dict)
                        elif isinstance(source, dict):
                            # Already a dict
                            all_sources.append(source)
                elif isinstance(results, list):
                    # List of SearchResult objects - convert each to dict
                    for source in results:
                        if hasattr(source, '__dict__'):
                            # SearchResult object - convert to dict
                            source_dict = {}
                            for attr in ['title', 'url', 'snippet', 'score', 'source_query', 'source_id']:
                                if hasattr(source, attr):
                                    source_dict[attr] = getattr(source, attr)
                            all_sources.append(source_dict)
                        elif isinstance(source, dict):
                            # Already a dict
                            all_sources.append(source)
                    
            except Exception as e:
                logger.warning(f"Search failed for '{query}': {e}")
                search_results[query] = {"error": str(e)}
        
        return {
            "sources": all_sources,
            "search_results": search_results,
            "total_sources": len(all_sources)
        }

    def _calculate_enhanced_quality_score(
        self, 
        web_research_data: Dict[str, Any], 
        product_catalog: List[Dict[str, Any]], 
        analysis_content: str
    ) -> float:
        """Calculate quality score for catalog-enhanced analysis."""
        
        base_score = 0.6  # Higher base for catalog data
        
        # Web research quality
        source_count = len(web_research_data.get("sources", []))
        web_bonus = min(0.1, source_count * 0.01)  # Up to 0.1 for web sources
        
        # Product catalog quality
        product_count = len(product_catalog)
        catalog_bonus = min(0.15, product_count * 0.001)  # Up to 0.15 for product richness
        
        # Analysis content quality
        content_length = len(analysis_content)
        content_bonus = min(0.1, content_length / 10000)  # Up to 0.1 for comprehensive analysis
        
        # Check for citations in content
        citation_count = analysis_content.count("[") + analysis_content.count("]")
        citation_bonus = min(0.05, citation_count * 0.002)  # Up to 0.05 for good citations
        
        final_score = base_score + web_bonus + catalog_bonus + content_bonus + citation_bonus
        return min(0.95, final_score)  # Cap at 0.95

    def _calculate_standard_quality_score(
        self, 
        web_research_data: Dict[str, Any], 
        analysis_content: str
    ) -> float:
        """Calculate quality score for web-only analysis."""
        
        base_score = 0.4  # Lower base for web-only
        
        # Web research quality
        source_count = len(web_research_data.get("sources", []))
        web_bonus = min(0.2, source_count * 0.02)  # Up to 0.2 for web sources
        
        # Analysis content quality
        content_length = len(analysis_content)
        content_bonus = min(0.15, content_length / 8000)  # Up to 0.15 for content
        
        # Check for citations
        citation_count = analysis_content.count("[") + analysis_content.count("]")
        citation_bonus = min(0.05, citation_count * 0.003)  # Up to 0.05 for citations
        
        final_score = base_score + web_bonus + content_bonus + citation_bonus
        return min(0.8, final_score)  # Cap at 0.8 for web-only

    # async def _save_results(self, brand_domain: str, analysis_result: Dict[str, Any]) -> None:
    #     """Save analysis results in three-file format."""
    #     try:
    #         # Check if we're using local storage (has base_dir) or GCP storage (has bucket)
    #         if hasattr(self.storage_manager, 'base_dir'):
    #             # Local storage
    #             research_dir = os.path.join(
    #                 self.storage_manager.base_dir, 
    #                 "accounts", 
    #                 brand_domain, 
    #                 "research_phases"
    #             )
    #             os.makedirs(research_dir, exist_ok=True)
                
    #             # Save content
    #             content_path = os.path.join(research_dir, "product_style_research.md")
    #             with open(content_path, "w") as f:
    #                 f.write(analysis_result["content"])
                
    #             # Save metadata
    #             metadata_path = os.path.join(research_dir, "product_style_research_metadata.json")
    #             with open(metadata_path, "w") as f:
    #                 json.dump(analysis_result["metadata"], f, indent=2)
                
    #             # Save sources - convert SearchResult objects to dictionaries first
    #             sources_path = os.path.join(research_dir, "product_style_research_sources.json")
    #             with open(sources_path, "w") as f:
    #                 serializable_sources = self._convert_sources_to_dict(analysis_result["sources"])
    #                 json.dump(serializable_sources, f, indent=2)
                    
    #         elif hasattr(self.storage_manager, 'bucket'):
    #             # GCP storage
    #             # Save content
    #             content_blob = self.storage_manager.bucket.blob(f"accounts/{brand_domain}/research_phases/product_style_research.md")
    #             content_blob.upload_from_string(analysis_result["content"], content_type="text/markdown")
                
    #             # Save metadata
    #             metadata_blob = self.storage_manager.bucket.blob(f"accounts/{brand_domain}/research_phases/product_style_research_metadata.json")
    #             metadata_blob.upload_from_string(json.dumps(analysis_result["metadata"], indent=2), content_type="application/json")
                
    #             # Save sources - convert SearchResult objects to dictionaries first
    #             sources_blob = self.storage_manager.bucket.blob(f"accounts/{brand_domain}/research_phases/product_style_research_sources.json")
    #             serializable_sources = self._convert_sources_to_dict(analysis_result["sources"])
    #             sources_blob.upload_from_string(json.dumps(serializable_sources, indent=2), content_type="application/json")
    #         else:
    #             raise Exception("Unknown storage provider type")
            
    #         logger.info(f"Saved product style research for {brand_domain}")
            
    #     except Exception as e:
    #         logger.error(f"Error saving results: {e}")
    #         raise

    def _convert_sources_to_dict(self, sources: List[Any]) -> List[Dict[str, Any]]:
        """Convert sources (including SearchResult objects) to JSON-serializable dictionaries."""
        serializable_sources = []
        for source in sources:
            if hasattr(source, '__dict__'):
                # Handle SearchResult objects by converting to dict
                source_dict = {}
                for attr in ['title', 'url', 'snippet', 'score', 'source_query', 'source_id']:
                    if hasattr(source, attr):
                        source_dict[attr] = getattr(source, attr)
                serializable_sources.append(source_dict)
            elif isinstance(source, dict):
                # Handle dict objects directly
                serializable_sources.append(source)
            else:
                # Handle other objects by converting to string
                serializable_sources.append({"content": str(source), "type": type(source).__name__})
        return serializable_sources

    async def _check_cache(self, brand_domain: str, force_regenerate: bool) -> Optional[Dict[str, Any]]:
        """Check for existing cached research."""
        if force_regenerate:
            return None
        
        try:
            # Check if we're using local storage (has base_dir) or GCP storage (has bucket)
            if hasattr(self.storage_manager, 'base_dir'):
                # Local storage
                research_dir = os.path.join(
                    self.storage_manager.base_dir, 
                    "accounts", 
                    brand_domain, 
                    "research_phases"
                )
                
                content_path = os.path.join(research_dir, "product_style_research.md")
                metadata_path = os.path.join(research_dir, "product_style_research_metadata.json")
                sources_path = os.path.join(research_dir, "product_style_research_sources.json")
                
                if all(os.path.exists(p) for p in [content_path, metadata_path, sources_path]):
                    # Load cached data
                    with open(content_path, "r") as f:
                        content = f.read()
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    with open(sources_path, "r") as f:
                        sources = json.load(f)
                    
                    return {
                        "content": content,
                        "metadata": metadata,
                        "sources": sources
                    }
                    
            elif hasattr(self.storage_manager, 'bucket'):
                # GCP storage
                content_blob = self.storage_manager.bucket.blob(f"accounts/{brand_domain}/research_phases/product_style_research.md")
                metadata_blob = self.storage_manager.bucket.blob(f"accounts/{brand_domain}/research_phases/product_style_research_metadata.json")
                sources_blob = self.storage_manager.bucket.blob(f"accounts/{brand_domain}/research_phases/product_style_research_sources.json")
                
                if all(blob.exists() for blob in [content_blob, metadata_blob, sources_blob]):
                    # Load cached data
                    content = content_blob.download_as_text()
                    metadata = json.loads(metadata_blob.download_as_text())
                    sources = json.loads(sources_blob.download_as_text())
                    
                    return {
                        "content": content,
                        "metadata": metadata,
                        "sources": sources
                    }
        
        except Exception as e:
            logger.debug(f"Cache check failed: {e}")
        
        return None

    def _serialize_sources(self, sources: List[Any]) -> str:
        """Serialize sources to a readable string format, handling both dict and SearchResult objects."""
        try:
            serialized_sources = []
            for source in sources[:10]:  # Limit to first 10 for readability
                if hasattr(source, '__dict__'):
                    # Handle SearchResult objects
                    title = getattr(source, 'title', 'Unknown Title')
                    url = getattr(source, 'url', 'Unknown URL')
                    serialized_sources.append(f"- {title} ({url})")
                elif isinstance(source, dict):
                    # Handle dict objects
                    title = source.get('title', 'Unknown Title')
                    url = source.get('url', 'Unknown URL')
                    serialized_sources.append(f"- {title} ({url})")
                else:
                    serialized_sources.append(f"- {str(source)}")
            return "\n".join(serialized_sources) if serialized_sources else "No sources available"
        except Exception as e:
            logger.warning(f"Error serializing sources: {e}")
            return f"Error serializing sources: {str(e)}"


def get_product_style_researcher(brand_domain: str) -> ProductStyleResearcher:
    """Get product style researcher instance"""
    return ProductStyleResearcher(brand_domain)
