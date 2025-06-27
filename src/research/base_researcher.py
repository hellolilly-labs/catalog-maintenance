"""
Base class for all researchers
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from src.llm.simple_factory import LLMFactory
from src.llm.prompt_manager import PromptManager, ChatMessageDict
from src.progress_tracker import (
    get_progress_tracker, 
    StepType, 
    create_console_listener,
    ProgressTracker
)
from src.storage import get_account_storage_provider
from configs.settings import get_settings
import time
import json
from langfuse.types import PromptClient
from src.web_search import TavilySearchProvider

logger = logging.getLogger(__name__)

class BaseResearcher:
    """
    Base class for all researchers
    """
    
    def __init__(self, brand_domain: str, researcher_name: str, step_type: StepType, quality_threshold: float = 8.0, cache_duration_days: int = 180, storage_manager=None):
        self.storage_manager = storage_manager or get_account_storage_provider()
        self.brand_domain = brand_domain.lower()
        self.researcher_name = researcher_name.lower()
        self.step_type = step_type
        self.settings = get_settings()
        self.quality_threshold = quality_threshold
        self.cache_duration_days = cache_duration_days  # 6 months default
        
        # Create progress tracker with storage integration and checkpoint logging
        self.progress_tracker = ProgressTracker(
            storage_manager=self.storage_manager,
            enable_checkpoints=True  # Enable persistent checkpoint logging
        )
        
        # Add console listener for real-time updates
        console_listener = create_console_listener()
        self.progress_tracker.add_progress_listener(console_listener)
        
        # Initialize prompt manager
        self.prompt_manager = PromptManager()

        try:
            api_key = getattr(self.settings, 'TAVILY_API_KEY', None)
            self.web_search = TavilySearchProvider(api_key) if api_key else None
        except Exception:
            self.web_search = None
        

    async def research(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Research for a brand
        
        Args:
            force_refresh: Force new research even if cached
            
        Returns:
            Research results with metadata
        """
        start_time = time.time()
        
        logger.info(f"🏗️ Starting {self.researcher_name} Research for {self.brand_domain}")
        
        # Create main progress step
        step_id = await self.progress_tracker.create_step(
            step_type=self.step_type,
            brand=self.brand_domain,
            phase_name=f"{self.researcher_name.capitalize()} Research",
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
                    logger.info(f"✅ Using cached {self.researcher_name} research for {self.brand_domain}")
                    return cached_result
            
            # Step 1: Comprehensive data gathering (60-90 seconds)
            await self.progress_tracker.update_progress(step_id, 1, f"📊 Step 1: Gathering comprehensive {self.researcher_name} data...")
            data = await self._gather_data()
            
            # Step 2: Multi-round LLM analysis (90-120 seconds)
            await self.progress_tracker.update_progress(step_id, 2, f"🧠 Step 2: Analyzing {self.researcher_name} data with LLM...")
            analysis = await self._analyze_data(data)
            
            # Step 3: Quality evaluation and synthesis (30-60 seconds)
            await self.progress_tracker.update_progress(step_id, 3, f"🎯 Step 3: Quality evaluation and synthesis...")
            final_results = await self._synthesize_results(
                analysis
            )
            
            # Add metadata
            final_results.update({
                "research_metadata": {
                    "phase": self.researcher_name,
                    "research_duration_seconds": time.time() - start_time,
                    "timestamp": datetime.now().isoformat() + "Z",
                    "cache_expires": (datetime.now() + timedelta(days=self.cache_duration_days)).isoformat() + "Z",
                    "quality_threshold": self.quality_threshold,
                    "version": "1.0"
                }
            })
            
            # Save to storage
            saved_files = await self._save_results(final_results)
            
            duration = time.time() - start_time
            logger.info(f"✅ {self.researcher_name.capitalize()} Research completed for {self.brand_domain} in {duration:.1f}s")
            
            await self.progress_tracker.complete_step(
                step_id,
                output_files=saved_files,
                quality_score=final_results.get("confidence_score", 0.8),
                cache_hit=False
            )
            
            return {
                "brand": self.brand_domain,
                "content": final_results.get("content", ""),
                "quality_score": final_results.get("confidence_score", 0.8),
                "files": saved_files,
                "data_sources": len(data.get("search_results", [])),
                "research_method": "enhanced_tavily_crawl_search"
            }
            
        except Exception as e:
            await self.progress_tracker.fail_step(step_id, str(e))
            logger.error(f"❌ Error in {self.researcher_name.capitalize()} research for {self.brand_domain}: {e}")
            raise
    
    async def _gather_data(self) -> Dict[str, Any]:
        """
        Gather data for the research
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    async def _analyze_data(self, data: Dict[str, Any], temperature: float = 0.1) -> Dict[str, Any]:
        """Analyze data using LLM and generate markdown content with source references"""
        
        # Check if we have sufficient data for analysis
        total_sources = data.get("total_sources", 0)
        search_stats = data.get("search_stats", {})
        success_rate = search_stats.get("success_rate", 0)
        
        if total_sources == 0:
            error_msg = "ANALYSIS ABORTED: No search results available for analysis. Cannot generate quality research without external data."
            logger.error(f"🚨 {error_msg}")
            raise RuntimeError(error_msg)
        
        # Warn about data quality but proceed if we have some sources
        if total_sources < 10:
            logger.warning(f"⚠️ Limited data available: Only {total_sources} sources for analysis. Research quality may be reduced.")
        
        if success_rate < 0.5:
            logger.warning(f"⚠️ Low search success rate: {success_rate:.1%}. Research confidence may be reduced.")
        
        # Compile search context with source IDs for citation
        search_context = ""
        source_citations = {}  # Map source_id to citation format
        
        for result in data["search_results"][:50]:  # Use top 50 results
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
            for result, citation in zip(data["search_results"][:50], source_citations.values())
        ])


        prompt_client = await self._get_prompt()
        prompts = prompt_client.prompt
        system_prompt = next((msg["content"] for msg in prompts if msg["role"] == "system"), None)
        user_prompt = next((msg["content"] for msg in prompts if msg["role"] == "user"), None)

        # Prepare template variables
        template_vars = {
            "brand_name": data.get('brand_name', self.brand_domain),
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
        
        # Replace template variables
        final_prompt = user_prompt
        for var, value in template_vars.items():
            final_prompt = final_prompt.replace(f"{{{{{var}}}}}", str(value))

        try:
            logger.info(f"🧠 Analyzing {total_sources} sources with o3 model (success rate: {success_rate:.1%})")
            
            response = await LLMFactory.chat_completion(
                task=f"brand_research_{self.researcher_name}",
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": final_prompt
                }],
                # max_tokens=2500,
                temperature=temperature
            )
            
            if response and response.get("content"):
                # Adjust confidence based on data quality
                base_confidence = 0.8
                if success_rate < 0.5:
                    base_confidence = 0.5
                elif success_rate < 0.7:
                    base_confidence = 0.65
                
                if total_sources < 10:
                    base_confidence *= 0.9
                elif total_sources < 5:
                    base_confidence *= 0.8
                
                return {
                    "markdown": response["content"],
                    "analysis_method": "markdown_report_with_citations",
                    "confidence": base_confidence,
                    "data_sources": total_sources,
                    "search_success_rate": success_rate,
                    "detailed_sources": data.get('detailed_sources', []),
                    "source_citations": source_citations,
                    "search_stats": search_stats
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
    
    async def _synthesize_results(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize and structure intelligence with source tracking"""
        
        # The analysis should no longer return error dictionaries - RuntimeErrors are raised instead
        # This method assumes analysis was successful if it gets called
        
        # Structure the final intelligence
        intelligence = {
            "brand_domain": self.brand_domain,
            "content": analysis.get("markdown", "") if analysis.get("markdown") else analysis.get("content", ""),
            "confidence_score": analysis.get("confidence", 0.8),
            "data_quality": "high" if analysis.get("analysis_method") == "markdown_report_with_citations" else "medium",
            "data_sources_count": analysis.get("data_sources", 0),
            "search_success_rate": analysis.get("search_success_rate", 0.0),
            "detailed_sources": analysis.get("detailed_sources", []),
            "source_citations": analysis.get("source_citations", {}),
            "search_stats": analysis.get("search_stats", {})
        }
        
        return intelligence

    async def _save_results(self, results: Dict[str, Any]) -> List[str]:
        """Save research as separate markdown content, JSON metadata, and sources file"""
        
        try:
            # Extract content, metadata, and sources
            content = results.get("content", "")
            detailed_sources = results.get("detailed_sources", [])
            
            metadata = {
                "brand_domain": self.brand_domain,
                "phase": self.researcher_name,
                "confidence_score": results.get("confidence_score", 0.0),
                "data_quality": results.get("data_quality", "unknown"),
                "data_sources_count": results.get("data_sources_count", 0),
                "analysis_method": results.get("analysis_method", "unknown"),
                "source_citations": results.get("source_citations", {}),
                "research_metadata": results.get("research_metadata", {})
            }
            
            # Create sources document
            sources_doc = {
                "brand_domain": self.brand_domain,
                "phase": self.researcher_name,
                "research_timestamp": results.get("research_metadata", {}).get("timestamp"),
                "total_sources": len(detailed_sources),
                "detailed_sources": detailed_sources,
                "source_summary": {
                    "web_search_results": len([s for s in detailed_sources if s.get("source_type") == "web_search"]),
                    "unique_domains": len(set(s.get("url", "").split("/")[2] for s in detailed_sources if s.get("url"))),
                    "search_queries_used": len(set(s.get("search_query") for s in detailed_sources if s.get("search_query"))),
                },
                "collection_timestamp": datetime.now().isoformat() + "Z"
            }
            
            # Save markdown content
            content_path = f"research/{self.researcher_name}/research.md"
            metadata_path = f"research/{self.researcher_name}/research_metadata.json"
            sources_path = f"research/{self.researcher_name}/research_sources.json"
            
            await self.storage_manager.write_file(
                account=self.brand_domain,
                file_path=content_path,
                content=content,
                content_type="text/markdown"
            )
            # content_blob = self.storage_manager.bucket.blob(f"accounts/{self.brand_domain}/research/{self.researcher_name}/research.md")
            # content_blob.upload_from_string(
            #     content,
            #     content_type="text/markdown"
            # )
            
            # Save metadata
            await self.storage_manager.write_file(
                account=self.brand_domain,
                file_path=metadata_path,
                content=json.dumps(metadata, indent=2),
                content_type="application/json"
            )
            # metadata_blob = self.storage_manager.bucket.blob(f"accounts/{self.brand_domain}/research/{self.researcher_name}/research_metadata.json")
            # metadata_blob.upload_from_string(
            #     json.dumps(metadata, indent=2),
            #     content_type="application/json"
            # )
            
            # Save sources
            await self.storage_manager.write_file(
                account=self.brand_domain,
                file_path=sources_path,
                content=json.dumps(sources_doc, indent=2),
                content_type="application/json"
            )
            # sources_blob = self.storage_manager.bucket.blob(f"accounts/{self.brand_domain}/research/{self.researcher_name}/research_sources.json")
            # sources_blob.upload_from_string(
            #     json.dumps(sources_doc, indent=2),
            #     content_type="application/json"
            # )
            
            logger.info(f"💾 Saved {self.researcher_name} research to GCP: {self.brand_domain}")
            
            return [content_path, metadata_path, sources_path]
            
        except Exception as e:
            logger.error(f"Error saving {self.researcher_name} research: {e}")
            return []
    
    async def _load_cached_results(self) -> Optional[Dict[str, Any]]:
        """Load cached research from separate markdown, metadata, and sources files"""
        
        try:
            if hasattr(self.storage_manager, 'bucket'):
                # GCP storage
                metadata_blob = self.storage_manager.bucket.blob(f"accounts/{self.brand_domain}/research/{self.researcher_name}/research_metadata.json")
                content_blob = self.storage_manager.bucket.blob(f"accounts/{self.brand_domain}/research/{self.researcher_name}/research.md")
                sources_blob = self.storage_manager.bucket.blob(f"accounts/{self.brand_domain}/research/{self.researcher_name}/research_sources.json")
                
                if metadata_blob.exists() and content_blob.exists():
                    metadata_content = metadata_blob.download_as_text()
                    cached_metadata = json.loads(metadata_content)
                    research_metadata = cached_metadata.get("research_metadata", {})
                    
                    # Check if cache is expired
                    cache_expires = research_metadata.get("cache_expires")
                    if cache_expires:
                        expiry_date = datetime.fromisoformat(cache_expires.replace('Z', '+00:00'))
                        if datetime.now().replace(tzinfo=expiry_date.tzinfo) < expiry_date:
                            logger.info(f"🔍 Cache expired for {self.researcher_name} research for {self.brand_domain}")
                            return None
                    
                    markdown_content = content_blob.download_as_text()
                    
                    # Load sources if available (optional - older research may not have sources)
                    detailed_sources = []
                    if sources_blob.exists():
                        sources_content = sources_blob.download_as_text()
                        sources_data = json.loads(sources_content)
                        detailed_sources = sources_data.get("sources", [])
                    
                    # Check if cache is expired
                    cache_expires = cached_metadata.get("research_metadata", {}).get("cache_expires")
                    if cache_expires:
                        expiry_date = datetime.fromisoformat(cache_expires.replace('Z', '+00:00'))
                        if datetime.now().replace(tzinfo=expiry_date.tzinfo) < expiry_date:
                            # Combine metadata, content, and sources
                            cached_metadata["content"] = markdown_content
                            cached_metadata["detailed_sources"] = detailed_sources
                            return cached_metadata
                    
            else:
                # Local storage
                import os
                research_dir = os.path.join(
                    self.storage_manager.base_dir, 
                    "accounts", 
                    self.brand_domain, 
                    "research"
                )
                
                metadata_path = os.path.join(research_dir, f"{self.researcher_name}", "research_metadata.json")
                content_path = os.path.join(research_dir, f"{self.researcher_name}", "research.md")
                sources_path = os.path.join(research_dir, f"{self.researcher_name}", "research_sources.json")
                
                if os.path.exists(metadata_path) and os.path.exists(content_path):
                    # Load metadata
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        cached_metadata = json.load(f)
                    
                    # Load content
                    with open(content_path, "r", encoding="utf-8") as f:
                        markdown_content = f.read()
                    
                    # Load sources if available (optional - older research may not have sources)
                    detailed_sources = []
                    if os.path.exists(sources_path):
                        with open(sources_path, "r", encoding="utf-8") as f:
                            sources_data = json.load(f)
                            detailed_sources = sources_data.get("sources", [])
                    
                    # Check if cache is expired
                    cache_expires = cached_metadata.get("research_metadata", {}).get("cache_expires")
                    if cache_expires:
                        expiry_date = datetime.fromisoformat(cache_expires.replace('Z', ''))
                        if datetime.now() < expiry_date:
                            # Combine metadata, content, and sources
                            cached_metadata["content"] = markdown_content
                            cached_metadata["detailed_sources"] = detailed_sources
                            return cached_metadata
            
            return None
            
        except Exception as e:
            logger.warning(f"Error loading cached {self.researcher_name} research: {e}")
            return None
    
    async def _get_prompt(self) -> PromptClient:
        prompts = []
        prompts.append(ChatMessageDict(role="system", content=self._get_default_instruction_prompt()))
        prompts.append(ChatMessageDict(role="user", content=self._get_default_user_prompt()))
        prompt = await self.prompt_manager.get_prompt(
            f"internal/researcher/{self.researcher_name}",
            prompt_type="chat",
            prompt=prompts
        )
        return prompt
    
    def _get_default_user_prompt(self) -> str:
        """
        Get the default user prompt for the research
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _get_default_instruction_prompt(self) -> str:
        """
        Get the default instruction prompt for the research
        """
        raise NotImplementedError("Subclasses must implement this method")
        
    
