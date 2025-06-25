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

logger = logging.getLogger(__name__)


class LinearityAnalysisResearcher:
    """Linearity Analysis Research Phase Implementation"""
    
    def __init__(self, storage_manager=None):
        self.storage_manager = storage_manager or get_account_storage_provider()
        self.quality_threshold = 7.5
        self.cache_duration_days = 45  # 1.5 months default
        
        self.progress_tracker = ProgressTracker(storage_manager=self.storage_manager, enable_checkpoints=True)
        console_listener = create_console_listener()
        self.progress_tracker.add_progress_listener(console_listener)
        
        # Initialize prompt manager
        self.prompt_manager = PromptManager()
        
    async def research_linearity_analysis(self, brand_domain: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Research linearity analysis phase for a brand"""
        start_time = time.time()
        
        logger.info(f"📊 Starting Linearity Analysis Research for {brand_domain}")
        
        step_id = self.progress_tracker.create_step(
            step_type=StepType.SYNTHESIS,
            brand=brand_domain,
            phase_name="Linearity Analysis Research",
            total_operations=8
        )
        
        try:
            self.progress_tracker.start_step(step_id, "Checking cache and initializing...")
            
            if not force_refresh:
                cached_result = await self._load_cached_linearity_analysis(brand_domain)
                if cached_result:
                    self.progress_tracker.complete_step(step_id, cache_hit=True)
                    return cached_result
            
            self.progress_tracker.update_progress(step_id, 1, "📋 Loading foundation research context...")
            
            # Load foundation context from all previous research phases
            foundation_context = await self._load_foundation_context(brand_domain)
            
            self.progress_tracker.update_progress(step_id, 2, "📊 Analyzing brand linearity and consistency...")
            
            # Conduct linearity analysis using foundation context
            analysis_result = await self._analyze_brand_linearity(
                brand_domain,
                foundation_context,
                step_id
            )
            
            self.progress_tracker.update_progress(step_id, 6, "💾 Saving linearity analysis research...")
            
            # Save research results in 3-file format
            saved_files = await self._save_linearity_analysis_research(brand_domain, analysis_result)
            
            self.progress_tracker.update_progress(step_id, 7, "✅ Finalizing linearity analysis...")
            
            duration = time.time() - start_time
            logger.info(f"✅ Linearity Analysis Research completed in {duration:.1f}s")
            
            self.progress_tracker.complete_step(
                step_id,
                output_files=saved_files,
                quality_score=analysis_result.get("confidence", 0.8),
                cache_hit=False
            )
            
            return {
                "brand": brand_domain,
                "linearity_analysis_content": analysis_result.get("content", ""),
                "quality_score": analysis_result.get("confidence", 0.8),
                "files": saved_files,
                "data_sources": analysis_result.get("source_count", 0),
                "research_method": analysis_result.get("analysis_type", "enhanced_linearity_analysis")
            }
            
        except Exception as e:
            self.progress_tracker.fail_step(step_id, str(e))
            logger.error(f"❌ Error in linearity analysis research: {e}")
            raise

    async def _load_foundation_context(self, brand_domain: str) -> Dict[str, Any]:
        """Load foundation context from all previous research phases"""
        
        foundation_context = {}
        
        try:
            # Determine storage base path
            if hasattr(self.storage_manager, 'base_dir'):
                # Local storage
                research_dir = os.path.join(self.storage_manager.base_dir, "accounts", brand_domain, "research_phases")
            else:
                # GCP storage - we'll adapt this for file reading
                research_dir = f"accounts/{brand_domain}/research_phases"
            
            # Load all previous research phases
            foundation_files = [
                ("foundation_research.md", "foundation"),
                ("market_positioning.md", "market_positioning"),
                ("product_style_research.md", "product_style"),
                ("customer_cultural_research.md", "customer_cultural"),
                ("voice_messaging_research.md", "voice_messaging"),
                ("interview_synthesis_research.md", "interview_synthesis")
            ]
            
            for filename, phase_key in foundation_files:
                try:
                    if hasattr(self.storage_manager, 'base_dir'):
                        # Local storage
                        file_path = os.path.join(research_dir, filename)
                        if os.path.exists(file_path):
                            with open(file_path, 'r') as f:
                                foundation_context[phase_key] = f.read()
                    else:
                        # GCP storage
                        blob_path = f"{research_dir}/{filename}"
                        if hasattr(self.storage_manager, 'bucket'):
                            blob = self.storage_manager.bucket.blob(blob_path)
                            if blob.exists():
                                foundation_context[phase_key] = blob.download_as_text()
                                
                except Exception as e:
                    logger.debug(f"Could not load {filename}: {e}")
            
            logger.info(f"📋 Loaded {len(foundation_context)} foundation research phases for linearity analysis")
            
        except Exception as e:
            logger.warning(f"Error loading foundation context: {e}")
        
        return foundation_context

    async def _analyze_brand_linearity(
        self,
        brand_domain: str,
        foundation_context: Dict[str, Any],
        step_id: str
    ) -> Dict[str, Any]:
        """Analyze brand linearity and consistency across research phases"""
        
        self.progress_tracker.update_progress(step_id, 3, "🔍 Analyzing consistency patterns...")
        
        # Get linearity analysis prompt
        prompt_template = await self._get_linearity_analysis_prompt()
        
        self.progress_tracker.update_progress(step_id, 4, "📊 Evaluating brand alignment...")
        
        # Prepare analysis data
        analysis_data = {
            "brand_domain": brand_domain,
            "foundation_research": foundation_context.get("foundation", ""),
            "market_positioning": foundation_context.get("market_positioning", ""),
            "product_style": foundation_context.get("product_style", ""),
            "customer_cultural": foundation_context.get("customer_cultural", ""),
            "voice_messaging": foundation_context.get("voice_messaging", ""),
            "interview_synthesis": foundation_context.get("interview_synthesis", "")
        }
        
        # Generate linearity analysis using LLM
        analysis_content = await self._generate_linearity_analysis(
            prompt_template, 
            analysis_data
        )
        
        self.progress_tracker.update_progress(step_id, 5, "📈 Calculating consistency score...")
        
        # Calculate quality score based on foundation context richness
        quality_score = self._calculate_linearity_analysis_quality_score(foundation_context, analysis_content)
        
        return {
            "content": analysis_content,
            "confidence": quality_score,
            "source_count": len(foundation_context),
            "analysis_type": "foundation_enhanced_linearity_analysis",
            "foundation_phases": list(foundation_context.keys())
        }

    async def _get_linearity_analysis_prompt(self) -> str:
        """Get linearity analysis prompt from Langfuse"""
        
        default_prompt = """
You are conducting comprehensive Linearity Analysis for {{brand_domain}} across all completed research phases.

You have access to complete foundation research from 6 research phases to analyze brand consistency, alignment, and linearity.

Your task is to identify patterns, consistencies, inconsistencies, and overall brand linearity across all research dimensions.

**FOUNDATION RESEARCH CONTEXT:**
Foundation Research: {{foundation_research}}
Market Positioning: {{market_positioning}}
Product Style: {{product_style}}
Customer Cultural Intelligence: {{customer_cultural}}
Voice & Messaging: {{voice_messaging}}
Interview Synthesis: {{interview_synthesis}}

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
"""

        prompt = await self.prompt_manager.get_prompt(
            "linearity_analysis",
            default_prompt
        )
        
        return prompt.prompt if prompt else default_prompt

    async def _generate_linearity_analysis(
        self, 
        prompt_template: str, 
        analysis_data: Dict[str, Any]
    ) -> str:
        """Generate linearity analysis using foundation research"""
        
        # Prepare template variables
        template_vars = {
            "brand_domain": analysis_data["brand_domain"],
            "foundation_research": analysis_data["foundation_research"][:1500] if analysis_data["foundation_research"] else "No foundation research available",
            "market_positioning": analysis_data["market_positioning"][:1200] if analysis_data["market_positioning"] else "No market positioning available",
            "product_style": analysis_data["product_style"][:1200] if analysis_data["product_style"] else "No product style research available",
            "customer_cultural": analysis_data["customer_cultural"][:1200] if analysis_data["customer_cultural"] else "No customer cultural research available",
            "voice_messaging": analysis_data["voice_messaging"][:1200] if analysis_data["voice_messaging"] else "No voice messaging research available",
            "interview_synthesis": analysis_data["interview_synthesis"][:1200] if analysis_data["interview_synthesis"] else "No interview synthesis available"
        }
        
        # Replace template variables
        final_prompt = prompt_template
        for var, value in template_vars.items():
            final_prompt = final_prompt.replace(f"{{{{{var}}}}}", str(value))
        
        # Prepare context for LLM
        context = f"""
BRAND: {analysis_data['brand_domain']}

COMPREHENSIVE RESEARCH FOUNDATION FOR LINEARITY ANALYSIS:
Foundation Research: {analysis_data['foundation_research'][:600] if analysis_data['foundation_research'] else 'Not available'}

Market Positioning: {analysis_data['market_positioning'][:500] if analysis_data['market_positioning'] else 'Not available'}

Product Style: {analysis_data['product_style'][:500] if analysis_data['product_style'] else 'Not available'}

Customer Cultural: {analysis_data['customer_cultural'][:500] if analysis_data['customer_cultural'] else 'Not available'}

Voice & Messaging: {analysis_data['voice_messaging'][:500] if analysis_data['voice_messaging'] else 'Not available'}

Interview Synthesis: {analysis_data['interview_synthesis'][:500] if analysis_data['interview_synthesis'] else 'Not available'}
"""
        
        response = await LLMFactory.chat_completion(
            task="linearity_analysis_research",
            system="You are an expert brand consistency analyst specializing in cross-phase research linearity analysis. Generate comprehensive consistency assessments and actionable insights based on complete foundation research across multiple phases.",
            messages=[
                {"role": "system", "content": final_prompt},
                {"role": "user", "content": context}
            ],
            temperature=0.6,
            max_tokens=4000
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


def get_linearity_analysis_researcher() -> LinearityAnalysisResearcher:
    """Get linearity analysis researcher instance"""
    return LinearityAnalysisResearcher()
