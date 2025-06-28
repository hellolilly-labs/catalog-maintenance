"""
Foundation Research Phase

Implements Phase 1 of the Brand Research Pipeline per ROADMAP Section 4.2.

Focus: Core brand identity that rarely changes
Cache Duration: 6-12 months (most stable)
Research Time: 3-5 minutes
Quality Threshold: 8.0

Research Sources:
- Company founding story, history, timeline
- Mission, vision, core values statements
- Leadership team and organizational culture
- Legal structure, ownership, headquarters
- Patents, foundational innovations
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from src.llm.simple_factory import LLMFactory
from src.storage import get_account_storage_provider
from configs.settings import get_settings
from src.progress_tracker import (
    get_progress_tracker, 
    StepType, 
    create_console_listener,
    ProgressTracker
)
from langfuse.types import PromptClient
from src.research.base_researcher import BaseResearcher
from src.research.data_sources import WebSearchDataSource, DataGatheringContext

logger = logging.getLogger(__name__)

class FoundationResearcher(BaseResearcher):
    """
    Foundation Research Phase Implementation
    
    Researches core brand identity that rarely changes:
    - Company founding story and history
    - Mission, vision, core values
    - Leadership and organizational culture
    - Legal structure and headquarters
    - Patents and foundational innovations
    """
    
    def __init__(self, brand_domain: str, storage_manager=None):
        """_summary_

        Args:
            brand_domain (str): _description_
            storage_manager (_type_, optional): _description_. Defaults to None.
        """
        super().__init__(brand_domain=brand_domain, researcher_name="foundation", step_type=StepType.FOUNDATION_RESEARCH, quality_threshold=8.0, cache_duration_days=180, storage_manager=storage_manager, enable_quality_evaluation=True)
        # self.settings = get_settings()
        # self.quality_threshold = 8.0
        # self.cache_duration_days = 180  # 6 months default
        
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
        
    # async def research(self, force_refresh: bool = False) -> Dict[str, Any]:
    #     """
    #     Research for a brand
        
    #     Args:
    #         force_refresh: Force new research even if cached
            
    #     Returns:
    #         Research results with metadata
    #     """
    #     start_time = time.time()
        
    #     logger.info(f"🏗️ Starting {self.researcher_name} Research for {self.brand_domain}")
        
    #     # Create main progress step
    #     step_id = self.progress_tracker.create_step(
    #         step_type=StepType.FOUNDATION_RESEARCH,
    #         brand=self.brand_domain,
    #         phase_name=f"{self.researcher_name.capitalize()} Research",
    #         total_operations=6  # data gathering, analysis, synthesis, validation, storage, completion
    #     )
        
    #     try:
    #         self.progress_tracker.start_step(step_id, "Checking cache and initializing...")
            
    #         # Check for cached results first
    #         if not force_refresh:
    #             cached_result = await self._load_cached_results()
    #             if cached_result:
    #                 self.progress_tracker.complete_step(
    #                     step_id, 
    #                     output_files=cached_result.get("files", []),
    #                     quality_score=cached_result.get("quality_score"),
    #                     cache_hit=True
    #                 )
    #                 logger.info(f"✅ Using cached {self.researcher_name} research for {self.brand_domain}")
    #                 return cached_result
            
    #         # Step 1: Comprehensive data gathering (60-90 seconds)
    #         self.progress_tracker.update_progress(step_id, 1, f"📊 Step 1: Gathering comprehensive {self.researcher_name} data...")
    #         data = await self._gather_data()
            
    #         # Step 2: Multi-round LLM analysis (90-120 seconds)
    #         self.progress_tracker.update_progress(step_id, 2, f"🧠 Step 2: Analyzing {self.researcher_name} data with LLM...")
    #         foundation_analysis = await self._analyze_data(data)
            
    #         # Step 3: Quality evaluation and synthesis (30-60 seconds)
    #         self.progress_tracker.update_progress(step_id, 3, f"🎯 Step 3: Quality evaluation and synthesis...")
    #         final_results = await self._synthesize_results(
    #             foundation_analysis
    #         )
            
    #         # Add metadata
    #         final_results.update({
    #             "research_metadata": {
    #                 "phase": self.researcher_name,
    #                 "research_duration_seconds": time.time() - start_time,
    #                 "timestamp": datetime.now().isoformat() + "Z",
    #                 "cache_expires": (datetime.now() + timedelta(days=self.cache_duration_days)).isoformat() + "Z",
    #                 "quality_threshold": self.quality_threshold,
    #                 "version": "1.0"
    #             }
    #         })
            
    #         # Save to storage
    #         saved_files = await self._save_results(final_results)
            
    #         duration = time.time() - start_time
    #         logger.info(f"✅ {self.researcher_name.capitalize()} Research completed for {self.brand_domain} in {duration:.1f}s")
            
    #         self.progress_tracker.complete_step(
    #             step_id,
    #             output_files=saved_files,
    #             quality_score=final_results.get("confidence_score", 0.8),
    #             cache_hit=False
    #         )
            
    #         return {
    #             "brand": self.brand_domain,
    #             "content": final_results.get("content", ""),
    #             "quality_score": final_results.get("confidence_score", 0.8),
    #             "files": saved_files,
    #             "data_sources": len(data.get("search_results", [])),
    #             "research_method": "enhanced_tavily_crawl_search"
    #         }
            
    #     except Exception as e:
    #         self.progress_tracker.fail_step(step_id, str(e))
    #         logger.error(f"❌ Error in {self.researcher_name.capitalize()} research for {self.brand_domain}: {e}")
    #         raise
    
    async def _gather_data(self) -> Dict[str, Any]:
        """Gather comprehensive data from multiple sources using WebSearchDataSource"""
        
        # Official company sources (preserved for backward compatibility)
        official_sources = [
            f"https://{self.brand_domain}/about",
            f"https://{self.brand_domain}/company", 
            f"https://{self.brand_domain}/mission",
            f"https://{self.brand_domain}/values",
            f"https://{self.brand_domain}/history",
            f"https://{self.brand_domain}/story",
            f"https://{self.brand_domain}/leadership",
            f"https://{self.brand_domain}/team",
            f"https://{self.brand_domain}/investors",
            f"https://{self.brand_domain}/careers"
        ]
        
        # Company research queries
        brand_name = self.brand_domain.replace('.com', '').replace('.', ' ').title()
        research_queries = [
            f"{brand_name} company founding story history",
            f"{brand_name} founder CEO leadership team",
            f"{brand_name} mission vision values statement",
            f"{brand_name} company headquarters location",
            f"{brand_name} company timeline milestones",
            f"{brand_name} organizational culture values",
            f"{brand_name} patents innovations technology",
            f"{brand_name} legal structure ownership private public",
            f"{brand_name} company size employees revenue",
            f"{brand_name} business model strategy approach",
            f"{brand_name} awards recognition achievements",
            f"{brand_name} sustainability initiatives corporate responsibility"
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
                phase_name="foundation_research"
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
                "official_sources": official_sources,
                "research_queries": research_queries,
                "total_sources": len(search_result.results),
                "search_stats": {
                    "successful_searches": search_result.successful_searches,
                    "failed_searches": search_result.failed_searches,
                    "success_rate": success_rate,
                    "ssl_errors": search_result.ssl_errors
                },
                "sources_by_type": {
                    "web_search": len(search_result.sources),
                    "official_urls": len(official_sources)
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
        default_prompt = """Analyze this brand research data to extract comprehensive foundation intelligence.

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

## Foundation Analysis Requirements:

Please create a comprehensive foundation intelligence report in **markdown format**. When referencing information, cite your sources using the numbers provided (e.g., [1], [2], [3]).

Structure your analysis as follows:

# Foundation Intelligence: {{brand_name}}

## 1. Company Founding & History
- **Founding Year & Location:** [When and where founded] [cite sources]
- **Founder Background:** [Founder(s) background and motivation] [cite sources]
- **Key Historical Milestones:** [Major timeline events] [cite sources]
- **Evolution & Changes:** [How company evolved over time] [cite sources]

## 2. Mission, Vision & Values
- **Mission Statement:** [Official mission if available] [cite sources]
- **Vision & Goals:** [Long-term vision and strategic goals] [cite sources]
- **Core Values:** [Company values and principles] [cite sources]
- **Corporate Culture:** [Cultural philosophy and approach] [cite sources]

## 3. Leadership & Organization
- **Current Leadership:** [CEO and key executives] [cite sources]
- **Leadership Philosophy:** [Management approach and style] [cite sources]
- **Organizational Culture:** [Internal culture and structure] [cite sources]
- **Geographic Presence:** [Locations and team size if known] [cite sources]

## 4. Business Fundamentals
- **Legal Structure:** [Private, public, ownership details] [cite sources]
- **Headquarters:** [Main office location and facilities] [cite sources]
- **Revenue Scale:** [Size indicators and financial scale] [cite sources]
- **Business Model:** [Core business approach and strategy] [cite sources]

## 5. Innovation & Differentiation
- **Key Technologies:** [Patents, proprietary tech, innovations] [cite sources]
- **Foundational Breakthroughs:** [Major innovations or firsts] [cite sources]
- **Unique Approaches:** [Distinctive methodologies or strategies] [cite sources]
- **Industry Leadership:** [Areas where they lead the industry] [cite sources]

## 6. Corporate Responsibility
- **Sustainability:** [Environmental initiatives and commitments] [cite sources]
- **Social Programs:** [Community and social responsibility] [cite sources]
- **Environmental Impact:** [Green initiatives and policies] [cite sources]
- **Community Engagement:** [Local and global community involvement] [cite sources]

## Analysis Quality & Confidence

**Data Sources:** {{total_sources}} search results analyzed
**Search Success Rate:** {{success_rate}}
**Information Quality:** {{information_quality}}
**Confidence Level:** {{confidence_level}} confidence in findings
**Key Gaps:** [Note any information that was missing or unclear due to limited data availability]

## Summary

[Provide a 2-3 sentence executive summary of the company's foundation]

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
        default_prompt = "You are an expert business researcher specializing in company foundation analysis. Generate comprehensive, well-structured markdown reports with proper source citations based on research data. Always cite your sources using the provided reference numbers. Adjust confidence levels based on data quality."
        return default_prompt

    # async def _analyze_data(self, data: Dict[str, Any], temperature: float = 0.1) -> Dict[str, Any]:
    #     """Analyze data using LLM and generate markdown content with source references"""
        
    #     # Check if we have sufficient data for analysis
    #     total_sources = data.get("total_sources", 0)
    #     search_stats = data.get("search_stats", {})
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
        
    #     for result in data["search_results"][:20]:  # Use top 20 results
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
    #         for result, citation in zip(data["search_results"][:20], source_citations.values())
    #     ])


    #     prompt_client = await self._get_prompt()
    #     prompts = prompt_client.prompt
    #     system_prompt = next((msg["content"] for msg in prompts if msg["role"] == "system"), None)
    #     user_prompt = next((msg["content"] for msg in prompts if msg["role"] == "user"), None)

    #     # Prepare template variables
    #     template_vars = {
    #         "brand_name": data.get('brand_name', self.brand_domain),
    #         "brand_domain": self.brand_domain,
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
    #     final_prompt = user_prompt
    #     for var, value in template_vars.items():
    #         final_prompt = final_prompt.replace(f"{{{{{var}}}}}", str(value))

    #     try:
    #         logger.info(f"🧠 Analyzing {total_sources} sources with o3 model (success rate: {success_rate:.1%})")
            
    #         response = await LLMFactory.chat_completion(
    #             task=f"brand_research_{self.researcher_name}",
    #             system=system_prompt,
    #             messages=[{
    #                 "role": "user",
    #                 "content": final_prompt
    #             }],
    #             # max_tokens=2500,
    #             temperature=temperature
    #         )
            
    #         if response and response.get("content"):
    #             # Adjust confidence based on data quality
    #             base_confidence = 0.8
    #             if success_rate < 0.5:
    #                 base_confidence = 0.5
    #             elif success_rate < 0.7:
    #                 base_confidence = 0.65
                
    #             if total_sources < 10:
    #                 base_confidence *= 0.9
    #             elif total_sources < 5:
    #                 base_confidence *= 0.8
                
    #             return {
    #                 "markdown": response["content"],
    #                 "analysis_method": "markdown_report_with_citations",
    #                 "confidence": base_confidence,
    #                 "data_sources": total_sources,
    #                 "search_success_rate": success_rate,
    #                 "detailed_sources": data.get('detailed_sources', []),
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
    #     """Synthesize and structure intelligence with source tracking"""
        
    #     # The analysis should no longer return error dictionaries - RuntimeErrors are raised instead
    #     # This method assumes analysis was successful if it gets called
        
    #     # Structure the final intelligence
    #     intelligence = {
    #         "brand_domain": self.brand_domain,
    #         "content": analysis.get("markdown", ""),
    #         "confidence_score": analysis.get("confidence", 0.8),
    #         "data_quality": "high" if analysis.get("analysis_method") == "markdown_report_with_citations" else "medium",
    #         "data_sources_count": analysis.get("data_sources", 0),
    #         "search_success_rate": analysis.get("search_success_rate", 0.0),
    #         "detailed_sources": analysis.get("detailed_sources", []),
    #         "source_citations": analysis.get("source_citations", {}),
    #         "search_stats": analysis.get("search_stats", {})
    #     }
        
    #     return intelligence
    
    # async def _save_results(self, results: Dict[str, Any]) -> List[str]:
    #     """Save research as separate markdown content, JSON metadata, and sources file"""
        
    #     try:
    #         # Extract content, metadata, and sources
    #         content = results.get("content", "")
    #         detailed_sources = results.get("detailed_sources", [])
            
    #         metadata = {
    #             "brand_domain": self.brand_domain,
    #             "confidence_score": results.get("confidence_score", 0.0),
    #             "data_quality": results.get("data_quality", "unknown"),
    #             "data_sources_count": results.get("data_sources_count", 0),
    #             "source_citations": results.get("source_citations", {}),
    #             "research_metadata": results.get("research_metadata", {})
    #         }
            
    #         # Create sources document
    #         sources_doc = {
    #             "brand_domain": self.brand_domain,
    #             "phase": "foundation",
    #             "research_timestamp": results.get("research_metadata", {}).get("timestamp"),
    #             "total_sources": len(detailed_sources),
    #             "sources": detailed_sources,
    #             "source_summary": {
    #                 "web_search_results": len([s for s in detailed_sources if s.get("source_type") == "web_search"]),
    #                 "unique_domains": len(set(s.get("url", "").split("/")[2] for s in detailed_sources if s.get("url"))),
    #                 "search_queries_used": len(set(s.get("search_query") for s in detailed_sources if s.get("search_query"))),
    #             }
    #         }
            
    #         if hasattr(self.storage_manager, 'bucket'):
    #             # GCP storage
    #             # Save markdown content
    #             content_blob = self.storage_manager.bucket.blob(f"accounts/{self.brand_domain}/research/{self.researcher_name}/research.md")
    #             content_blob.upload_from_string(
    #                 content,
    #                 content_type="text/markdown"
    #             )
                
    #             # Save metadata
    #             metadata_blob = self.storage_manager.bucket.blob(f"accounts/{self.brand_domain}/research/{self.researcher_name}/research_metadata.json")
    #             metadata_blob.upload_from_string(
    #                 json.dumps(metadata, indent=2),
    #                 content_type="application/json"
    #             )
                
    #             # Save sources
    #             sources_blob = self.storage_manager.bucket.blob(f"accounts/{self.brand_domain}/research/{self.researcher_name}/research_sources.json")
    #             sources_blob.upload_from_string(
    #                 json.dumps(sources_doc, indent=2),
    #                 content_type="application/json"
    #             )
                
    #             logger.info(f"💾 Saved {self.researcher_name} research to GCP: {self.brand_domain}")
                
    #         else:
    #             # Local storage
    #             import os
    #             research_dir = os.path.join(
    #                 self.storage_manager.base_dir,
    #                 "accounts", 
    #                 self.brand_domain, 
    #                 "research"
    #             )
    #             os.makedirs(research_dir, exist_ok=True)
                
    #             # Save markdown content
    #             content_path = os.path.join(research_dir, f"{self.researcher_name}", "research.md")
    #             with open(content_path, "w", encoding="utf-8") as f:
    #                 f.write(content)
                
    #             # Save metadata  
    #             metadata_path = os.path.join(research_dir, f"{self.researcher_name}", "research_metadata.json")
    #             with open(metadata_path, "w", encoding="utf-8") as f:
    #                 json.dump(metadata, f, indent=2)
                
    #             # Save sources
    #             sources_path = os.path.join(research_dir, f"{self.researcher_name}", "research_sources.json")
    #             with open(sources_path, "w", encoding="utf-8") as f:
    #                 json.dump(sources_doc, f, indent=2)
                
    #             logger.info(f"💾 Saved {self.researcher_name} research to local storage:")
    #             logger.info(f"   Content: {content_path}")
    #             logger.info(f"   Metadata: {metadata_path}")
    #             logger.info(f"   Sources: {sources_path}")
            
    #         return [content_path, metadata_path, sources_path]
            
    #     except Exception as e:
    #         logger.error(f"Error saving {self.researcher_name} research: {e}")
    #         return []
    
    # async def _load_cached_results(self) -> Optional[Dict[str, Any]]:
    #     """Load cached research from separate markdown, metadata, and sources files"""
        
    #     try:
    #         if hasattr(self.storage_manager, 'bucket'):
    #             # GCP storage
    #             metadata_blob = self.storage_manager.bucket.blob(f"accounts/{self.brand_domain}/research/{self.researcher_name}/research_metadata.json")
    #             content_blob = self.storage_manager.bucket.blob(f"accounts/{self.brand_domain}/research/{self.researcher_name}/research.md")
    #             sources_blob = self.storage_manager.bucket.blob(f"accounts/{self.brand_domain}/research/{self.researcher_name}/research_sources.json")
                
    #             if metadata_blob.exists() and content_blob.exists():
    #                 metadata_content = metadata_blob.download_as_text()
    #                 cached_metadata = json.loads(metadata_content)
                    
    #                 markdown_content = content_blob.download_as_text()
                    
    #                 # Load sources if available (optional - older research may not have sources)
    #                 detailed_sources = []
    #                 if sources_blob.exists():
    #                     sources_content = sources_blob.download_as_text()
    #                     sources_data = json.loads(sources_content)
    #                     detailed_sources = sources_data.get("sources", [])
                    
    #                 # Check if cache is expired
    #                 cache_expires = cached_metadata.get("research_metadata", {}).get("cache_expires")
    #                 if cache_expires:
    #                     expiry_date = datetime.fromisoformat(cache_expires.replace('Z', '+00:00'))
    #                     if datetime.now().replace(tzinfo=expiry_date.tzinfo) < expiry_date:
    #                         # Combine metadata, content, and sources
    #                         cached_metadata["content"] = markdown_content
    #                         cached_metadata["detailed_sources"] = detailed_sources
    #                         return cached_metadata
                    
    #         else:
    #             # Local storage
    #             import os
    #             research_dir = os.path.join(
    #                 self.storage_manager.base_dir, 
    #                 "accounts", 
    #                 self.brand_domain, 
    #                 "research"
    #             )
                
    #             metadata_path = os.path.join(research_dir, f"{self.researcher_name}", "research_metadata.json")
    #             content_path = os.path.join(research_dir, f"{self.researcher_name}", "research.md")
    #             sources_path = os.path.join(research_dir, f"{self.researcher_name}", "research_sources.json")
                
    #             if os.path.exists(metadata_path) and os.path.exists(content_path):
    #                 # Load metadata
    #                 with open(metadata_path, "r", encoding="utf-8") as f:
    #                     cached_metadata = json.load(f)
                    
    #                 # Load content
    #                 with open(content_path, "r", encoding="utf-8") as f:
    #                     markdown_content = f.read()
                    
    #                 # Load sources if available (optional - older research may not have sources)
    #                 detailed_sources = []
    #                 if os.path.exists(sources_path):
    #                     with open(sources_path, "r", encoding="utf-8") as f:
    #                         sources_data = json.load(f)
    #                         detailed_sources = sources_data.get("sources", [])
                    
    #                 # Check if cache is expired
    #                 cache_expires = cached_metadata.get("research_metadata", {}).get("cache_expires")
    #                 if cache_expires:
    #                     expiry_date = datetime.fromisoformat(cache_expires.replace('Z', ''))
    #                     if datetime.now() < expiry_date:
    #                         # Combine metadata, content, and sources
    #                         cached_metadata["content"] = markdown_content
    #                         cached_metadata["detailed_sources"] = detailed_sources
    #                         return cached_metadata
            
    #         return None
            
    #     except Exception as e:
    #         logger.warning(f"Error loading cached {self.researcher_name} research: {e}")
    #         return None


# Factory function
def get_foundation_researcher(brand_domain: str) -> FoundationResearcher:
    """Get configured foundation researcher"""
    return FoundationResearcher(brand_domain=brand_domain) 