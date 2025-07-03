"""
Base class for all researchers with integrated quality evaluation
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from liddy_intelligence.llm.simple_factory import LLMFactory
from liddy.prompt_manager import PromptManager, ChatMessageDict
from liddy_intelligence.progress_tracker import (
    get_progress_tracker, 
    StepType, 
    create_console_listener,
    ProgressTracker
)
from liddy.storage import get_account_storage_provider
from liddy.config import get_settings
import time
import json
from langfuse.types import PromptClient
from liddy_intelligence.web_search import TavilySearchProvider

logger = logging.getLogger(__name__)

class BaseResearcher:
    """
    Base class for all researchers with integrated quality evaluation
    """
    
    def __init__(self, brand_domain: str, researcher_name: str, step_type: StepType, quality_threshold: float = 8.0, cache_duration_days: int = 7, storage_manager=None, enable_quality_evaluation: bool = True):
        """
        Initialize base researcher
        
        Args:
            brand_domain: Domain to research
            storage_manager: Storage manager instance
            enable_quality_evaluation: Whether to enable quality evaluation with feedback loops
        """
        self.brand_domain = brand_domain
        self.researcher_name = researcher_name
        self.step_type = step_type
        self.quality_threshold = quality_threshold
        self.cache_duration_days = cache_duration_days
        self.storage_manager = storage_manager or get_account_storage_provider()
        self.enable_quality_evaluation = enable_quality_evaluation
        
        # Quality evaluation settings
        self.quality_threshold = 8.0
        self.max_quality_attempts = 3
        self.cache_duration_days = 7
        
        # Initialize evaluators
        self._initialize_quality_evaluators()
        
        # Initialize progress tracker
        self.progress_tracker = ProgressTracker(
            storage_manager=self.storage_manager,
            enable_checkpoints=True
        )
        
        # Initialize prompt manager for Langfuse integration
        self.prompt_manager = PromptManager()
        
        # Researcher identification
        # self.researcher_name = self.__class__.__name__
        # self.step_type = getattr(StepType, self.researcher_name.upper().replace('RESEARCHER', ''), StepType.RESEARCH)
        
        try:
            api_key = getattr(get_settings(), 'TAVILY_API_KEY', None)
            self.web_search = TavilySearchProvider(api_key) if api_key else None
        except Exception:
            self.web_search = None
        
        # Quality evaluator models by phase
        self.quality_evaluator_models = {
            "foundation": "o3",
            "market_positioning": "o3",
            "product_style": "o3",
            "customer_cultural": "o3",
            "voice_messaging": "o3",
            "interview_synthesis": "o3",
            "linearity_analysis": "o3",
            "research_integration": "o3"
        }

    def _initialize_quality_evaluators(self):
        """
        Initialize single quality evaluation component
        All researchers use the same QualityEvaluator class with enable_web_search flag
        """
        
        try:
            from liddy_intelligence.research.quality.quality_evaluator import QualityEvaluator
            
            # Single evaluator class for ALL researchers
            # Web search enhancement enabled by default (assume web search is available)
            self.quality_evaluator = QualityEvaluator(enable_web_search=True)
            
            logger.info("✅ Quality evaluator initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Quality evaluator initialization failed: {e}")
            # This should never happen - if it does, we have bigger problems
            raise RuntimeError(f"Cannot initialize quality evaluator: {e}")

    async def research(self, force_refresh: bool = False, improvement_feedback: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Main research method with automatic quality evaluation wrapping
        
        Args:
            force_refresh: Force refresh of cached results
            improvement_feedback: Optional feedback from previous quality evaluation for iterative improvement
        """
        if self.enable_quality_evaluation:
            return await self._research_with_quality_wrapper(force_refresh, improvement_feedback)
        else:
            return await self._execute_core_research(force_refresh, improvement_feedback)
    
    async def _research_with_quality_wrapper(self, force_refresh: bool, initial_improvement_feedback: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Quality evaluation wrapper around core research logic
        Handles feedback loops and quality control automatically
        """
        logger.info(f"🎯 Starting quality-controlled research for {self.researcher_name}")
        
        best_result = None
        best_quality_score = 0.0
        # Start with any externally provided feedback, then use evaluation feedback for subsequent iterations
        improvement_context = initial_improvement_feedback
        
        for attempt in range(1, self.max_quality_attempts + 1):
            logger.info(f"🔄 Quality attempt {attempt}/{self.max_quality_attempts}")
            
            try:
                # Execute core research (always force refresh on retries)
                should_force = force_refresh or (attempt > 1)
                core_result = await self._execute_core_research(
                    force_refresh=should_force,
                    improvement_feedback=improvement_context
                )
                
                # Run quality evaluation
                quality_evaluation = await self._evaluate_research_quality(core_result)
                
                # Add quality evaluation to result
                enhanced_result = core_result.copy()
                enhanced_result["quality_evaluation"] = quality_evaluation
                enhanced_result["quality_attempts"] = attempt
                enhanced_result["max_quality_attempts"] = self.max_quality_attempts
                
                # Check if quality threshold is met
                quality_score = quality_evaluation.get("quality_score", 0.0)
                passes_threshold = quality_score >= self.quality_threshold
                
                if passes_threshold:
                    logger.info(f"✅ Quality threshold met on attempt {attempt}: {quality_score:.1f}/{self.quality_threshold:.1f}")
                    enhanced_result["research_method"] = "quality_controlled_research"
                    return enhanced_result
                
                # Track best result for fallback
                if quality_score > best_quality_score:
                    best_result = enhanced_result
                    best_quality_score = quality_score
                
                # Prepare for next attempt using evaluation feedback
                if attempt < self.max_quality_attempts:
                    improvement_context = quality_evaluation.get("improvement_feedback", [])
                    logger.warning(f"⚠️ Quality below threshold ({quality_score:.1f}/{self.quality_threshold:.1f}). Retrying...")
                    
                    # Brief pause before retry
                    import asyncio
                    await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Quality attempt {attempt} failed: {e}")
                if attempt == self.max_quality_attempts:
                    raise
                continue
        
        # All attempts completed but quality threshold not met
        logger.warning(f"⚠️ Quality threshold not met after {self.max_quality_attempts} attempts. Best: {best_quality_score:.1f}/{self.quality_threshold:.1f}")
        
        if best_result:
            # Mark as quality warning and return best result
            best_result["quality_warning"] = True
            best_result["final_quality_score"] = best_quality_score
            best_result["research_method"] = "quality_controlled_with_warning"
            return best_result
        else:
            raise RuntimeError(f"Quality control failed after {self.max_quality_attempts} attempts")
    
    async def _execute_core_research(self, force_refresh: bool = False, improvement_feedback: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Core research execution method - this is what subclasses should override
        Clean separation from quality evaluation logic
        
        Args:
            force_refresh: Force refresh of cached results
            improvement_feedback: Optional feedback from quality evaluation for iterative improvement
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
            
            # Check for cached results first (only on first attempt without improvement feedback)
            if not force_refresh and not improvement_feedback:
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
            
            # Step 1: Comprehensive data gathering
            await self.progress_tracker.update_progress(step_id, 1, f"📊 Step 1: Gathering comprehensive {self.researcher_name} data...")
            data = await self._gather_data()
            
            # Add improvement feedback if available (for quality retries)
            if improvement_feedback:
                data["improvement_context"] = improvement_feedback
                logger.info(f"💡 Incorporating {len(improvement_feedback)} improvement suggestions")
            
            # Step 2: Multi-round LLM analysis
            await self.progress_tracker.update_progress(step_id, 2, f"🧠 Step 2: Analyzing {self.researcher_name} data with LLM...")
            analysis = await self._analyze_data(data)
            
            # Step 3: Quality evaluation and synthesis
            await self.progress_tracker.update_progress(step_id, 3, f"🎯 Step 3: Quality evaluation and synthesis...")
            final_results = await self._synthesize_results(analysis)
            
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
    
    async def _evaluate_research_quality(self, research_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate research quality using the unified QualityEvaluator
        Returns quality evaluation data to be stored in research_metadata.json
        """
        
        try:
            # All researchers use the same QualityEvaluator class
            # Just call the main evaluation method
            return await self.quality_evaluator.evaluate_with_search_recommendations(
                research_result=research_results,
                phase_name=self.researcher_name,
                brand_domain=self.brand_domain,
                quality_threshold=self.quality_threshold
            )
            
        except Exception as e:
            logger.error(f"❌ Quality evaluation failed: {e}")
            return {
                "quality_score": 0.0,
                "passes_threshold": False,
                "improvement_feedback": [f"Evaluation failed: {str(e)}"],
                "criteria_met": {},
                "evaluator_model": "evaluation_failed",
                "evaluation_timestamp": datetime.now().isoformat(),
                "confidence_level": "low",
                "evaluation_method": "error_fallback",
                "error": str(e)
            }
    
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
        
        # Add improvement context if available (for quality retries)
        improvement_context = data.get("improvement_context", [])
        if improvement_context:
            improvement_text = "\n\n**IMPROVEMENT CONTEXT (from previous quality evaluation):**\n"
            for i, suggestion in enumerate(improvement_context[:5], 1):
                improvement_text += f"{i}. {suggestion}\n"
            improvement_text += "\nPlease address these specific improvement areas in your research.\n"
            
            template_vars["improvement_context"] = improvement_text
            user_prompt = user_prompt + "\n\n{improvement_context}"
        else:
            template_vars["improvement_context"] = ""
        
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

    async def _save_results(self, research_results: Dict[str, Any]) -> List[str]:
        """Save research results with quality evaluation in metadata"""
        saved_files = []
        
        try:
            brand_domain = self.brand_domain
            phase_name = self.researcher_name
            
            if hasattr(self.storage_manager, 'base_dir'):
                import os
                # Local storage
                research_dir = os.path.join(self.storage_manager.base_dir, "accounts", brand_domain, "research", phase_name)
                os.makedirs(research_dir, exist_ok=True)
                
                # Save content
                content_path = os.path.join(research_dir, "research.md")
                with open(content_path, "w") as f:
                    f.write(research_results.get("content", ""))
                saved_files.append(content_path)
                
                # Save enhanced metadata with quality evaluation
                metadata = {
                    "phase": phase_name,
                    "confidence_score": research_results.get("confidence_score", 0.8),
                    "data_sources_count": research_results.get("data_sources_count", 0),
                    "research_metadata": research_results.get("research_metadata", {}),
                    "quality_evaluation": research_results.get("quality_evaluation", {}),
                    "quality_warning": research_results.get("quality_warning", False)
                }
                
                metadata_path = os.path.join(research_dir, "research_metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                saved_files.append(metadata_path)
                
                # Save sources data
                sources_data = {
                    "search_stats": research_results.get("search_stats", {}),
                    "detailed_sources": research_results.get("detailed_sources", []),
                    "source_citations": research_results.get("source_citations", {}),
                    "collection_timestamp": datetime.now().isoformat() + "Z"
                }
                
                sources_path = os.path.join(research_dir, "research_sources.json")
                with open(sources_path, "w") as f:
                    json.dump(sources_data, f, indent=2)
                saved_files.append(sources_path)
                
            else:
                # GCP storage
                research_dir = f"accounts/{brand_domain}/research/{phase_name}"
                
                # Save content
                content_blob = f"{research_dir}/research.md"
                blob = self.storage_manager.bucket.blob(content_blob)
                blob.upload_from_string(research_results.get("content", ""))
                saved_files.append(content_blob)
                
                # Save enhanced metadata with quality evaluation
                metadata = {
                    "phase": phase_name,
                    "confidence_score": research_results.get("confidence_score", 0.8),
                    "data_sources_count": research_results.get("data_sources_count", 0),
                    "research_metadata": research_results.get("research_metadata", {}),
                    "quality_evaluation": research_results.get("quality_evaluation", {}),
                    "quality_warning": research_results.get("quality_warning", False)
                }
                
                metadata_blob = f"{research_dir}/research_metadata.json"
                blob = self.storage_manager.bucket.blob(metadata_blob)
                blob.upload_from_string(json.dumps(metadata, indent=2))
                saved_files.append(metadata_blob)
                
                # Save sources
                sources_data = {
                    "search_stats": research_results.get("search_stats", {}),
                    "detailed_sources": research_results.get("detailed_sources", []),
                    "source_citations": research_results.get("source_citations", {}),
                    "collection_timestamp": datetime.now().isoformat() + "Z"
                }
                
                sources_blob = f"{research_dir}/research_sources.json"
                blob = self.storage_manager.bucket.blob(sources_blob)
                blob.upload_from_string(json.dumps(sources_data, indent=2))
                saved_files.append(sources_blob)
            
            logger.info(f"✅ Saved {phase_name} research with quality evaluation for {brand_domain}")
            
        except Exception as e:
            logger.error(f"❌ Error saving research results: {e}")
            raise
        
        return saved_files

    async def _load_cached_results(self) -> Optional[Dict[str, Any]]:
        """Load cached research results with quality evaluation data"""
        try:
            brand_domain = self.brand_domain
            phase_name = self.researcher_name
            
            if hasattr(self.storage_manager, 'base_dir'):
                # Local storage
                import os
                research_dir = os.path.join(self.storage_manager.base_dir, "accounts", brand_domain, "research", phase_name)
                
                content_path = os.path.join(research_dir, "research.md")
                metadata_path = os.path.join(research_dir, "research_metadata.json")
                
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
                            "content": content,
                            "quality_score": metadata.get("confidence_score", 0.8),
                            "files": [content_path, metadata_path],
                            "data_sources": metadata.get("data_sources_count", 0),
                            "research_method": "cached_results",
                            "quality_evaluation": metadata.get("quality_evaluation", {}),
                            "quality_warning": metadata.get("quality_warning", False)
                        }
            else:
                # GCP storage 
                research_dir = f"accounts/{brand_domain}/research/{phase_name}"
                
                content_blob = f"{research_dir}/research.md"
                metadata_blob = f"{research_dir}/research_metadata.json"
                
                content_blob_obj = self.storage_manager.bucket.blob(content_blob)
                metadata_blob_obj = self.storage_manager.bucket.blob(metadata_blob)
                
                if content_blob_obj.exists() and metadata_blob_obj.exists():
                    # Check cache expiry
                    metadata_content = metadata_blob_obj.download_as_text()
                    metadata = json.loads(metadata_content)
                    
                    research_metadata = metadata.get("research_metadata", {})
                    cache_expires = research_metadata.get("cache_expires")
                    
                    if not cache_expires or datetime.now() < datetime.fromisoformat(cache_expires.replace("Z", "")):
                        # Load cached data
                        content = content_blob_obj.download_as_text()
                        
                        return {
                            "brand": brand_domain,
                            "content": content,
                            "quality_score": metadata.get("confidence_score", 0.8),
                            "files": [content_blob, metadata_blob],
                            "data_sources": metadata.get("data_sources_count", 0),
                            "research_method": "cached_results",
                            "quality_evaluation": metadata.get("quality_evaluation", {}),
                            "quality_warning": metadata.get("quality_warning", False)
                        }
        except Exception as e:
            logger.debug(f"Cache check failed: {e}")
        
        return None

    async def _get_prompt(self) -> PromptClient:
        """Get research prompt for this phase"""
        try:
            # Try to get specific prompt from Langfuse
            prompt_key = f"internal/researcher/{self.researcher_name}"
            prompt_client = await self.prompt_manager.get_prompt(prompt_key)
            
            if prompt_client and prompt_client.prompt:
                return prompt_client
            
            # Fallback to creating default prompts
            logger.warning(f"Using fallback prompt for {self.researcher_name}")
            
            # Create default prompts for the research phase
            default_prompts = [
                {"role": "system", "content": self._get_default_system_prompt()},
                {"role": "user", "content": self._get_default_user_prompt()}
            ]
            
            # Create a mock prompt client for fallback
            class MockPromptClient:
                def __init__(self, prompts):
                    self.prompt = prompts
                    self.version = "fallback_1.0"
            
            return MockPromptClient(default_prompts)
            
        except Exception as e:
            logger.error(f"Error getting prompt for {self.researcher_name}: {e}")
            raise
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for this research phase"""
        return f"""You are an expert {self.researcher_name} researcher specializing in comprehensive brand analysis.

Your task is to conduct thorough {self.researcher_name} research by analyzing the provided web research data and generating comprehensive insights.

Focus on accuracy, completeness, and actionable insights. Use proper citations and maintain high quality standards.

Generate comprehensive markdown content that provides valuable insights for brand intelligence and AI sales agent training."""
    
    def _get_default_user_prompt(self) -> str:
        """Get default user prompt for this research phase"""
        return """Conduct comprehensive {phase_name} research for {{brand_domain}} based on the provided web research data.

**RESEARCH DATA:**
{{search_context}}

**REQUIREMENTS:**
- Generate comprehensive analysis with proper structure
- Use numbered citations [1], [2], etc. from the source reference guide
- Provide actionable insights and recommendations
- Maintain high quality and accuracy standards
- Focus on information relevant to {phase_name} research

**SOURCE REFERENCE GUIDE:**
{{source_reference_guide}}

Generate comprehensive {phase_name} intelligence that provides valuable insights for brand understanding and AI agent training."""
        
    
