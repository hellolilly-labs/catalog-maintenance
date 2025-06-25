"""
Research Integration Research Phase

Implements Phase 8 of the Brand Research Pipeline per ROADMAP Section 4.8.

Focus: Cross-validation and integration of all research phases
Cache Duration: 1 month (needs regular refresh)
Research Time: 1-2 minutes
Quality Threshold: 8.0
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


class ResearchIntegrationProcessor:
    """Research Integration Phase Implementation"""
    
    def __init__(self, storage_manager=None):
        self.storage_manager = storage_manager or get_account_storage_provider()
        self.quality_threshold = 8.0
        self.cache_duration_days = 30  # 1 month default
        
        self.progress_tracker = ProgressTracker(storage_manager=self.storage_manager, enable_checkpoints=True)
        console_listener = create_console_listener()
        self.progress_tracker.add_progress_listener(console_listener)
        
        # Initialize prompt manager
        self.prompt_manager = PromptManager()
        
    async def integrate_research(self, brand_domain: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Integrate all research phases for a brand"""
        start_time = time.time()
        
        logger.info(f"🔗 Starting Research Integration for {brand_domain}")
        
        step_id = self.progress_tracker.create_step(
            step_type=StepType.SYNTHESIS,
            brand=brand_domain,
            phase_name="Research Integration",
            total_operations=8
        )
        
        try:
            self.progress_tracker.start_step(step_id, "Checking cache and initializing...")
            
            if not force_refresh:
                cached_result = await self._load_cached_research_integration(brand_domain)
                if cached_result:
                    self.progress_tracker.complete_step(step_id, cache_hit=True)
                    return cached_result
            
            self.progress_tracker.update_progress(step_id, 1, "📋 Loading all research phases...")
            
            # Load complete research foundation from all phases
            research_foundation = await self._load_complete_research_foundation(brand_domain)
            
            self.progress_tracker.update_progress(step_id, 2, "🔗 Integrating research insights...")
            
            # Conduct comprehensive research integration 
            integration_result = await self._integrate_all_research_phases(
                brand_domain,
                research_foundation,
                step_id
            )
            
            self.progress_tracker.update_progress(step_id, 6, "💾 Saving integrated research...")
            
            # Save integration results in 3-file format
            saved_files = await self._save_research_integration(brand_domain, integration_result)
            
            self.progress_tracker.update_progress(step_id, 7, "✅ Finalizing research integration...")
            
            duration = time.time() - start_time
            logger.info(f"✅ Research Integration completed in {duration:.1f}s")
            
            self.progress_tracker.complete_step(
                step_id,
                output_files=saved_files,
                quality_score=integration_result.get("confidence", 0.9),
                cache_hit=False
            )
            
            return {
                "brand": brand_domain,
                "research_integration_content": integration_result.get("content", ""),
                "quality_score": integration_result.get("confidence", 0.9),
                "files": saved_files,
                "data_sources": integration_result.get("source_count", 0),
                "research_method": integration_result.get("analysis_type", "comprehensive_research_integration")
            }
            
        except Exception as e:
            self.progress_tracker.fail_step(step_id, str(e))
            logger.error(f"❌ Error in research integration: {e}")
            raise

    async def _load_complete_research_foundation(self, brand_domain: str) -> Dict[str, Any]:
        """Load complete research foundation from all 7 previous phases"""
        
        research_foundation = {}
        
        try:
            # Determine storage base path
            if hasattr(self.storage_manager, 'base_dir'):
                # Local storage
                research_dir = os.path.join(self.storage_manager.base_dir, "accounts", brand_domain, "research_phases")
            else:
                # GCP storage - we'll adapt this for file reading
                research_dir = f"accounts/{brand_domain}/research_phases"
            
            # Load all research phases
            research_files = [
                ("foundation_research.md", "foundation"),
                ("market_positioning.md", "market_positioning"),
                ("product_style_research.md", "product_style"),
                ("customer_cultural_research.md", "customer_cultural"),
                ("voice_messaging_research.md", "voice_messaging"),
                ("interview_synthesis_research.md", "interview_synthesis"),
                ("linearity_analysis_research.md", "linearity_analysis")
            ]
            
            for filename, phase_key in research_files:
                try:
                    if hasattr(self.storage_manager, 'base_dir'):
                        # Local storage
                        file_path = os.path.join(research_dir, filename)
                        if os.path.exists(file_path):
                            with open(file_path, 'r') as f:
                                research_foundation[phase_key] = f.read()
                    else:
                        # GCP storage
                        blob_path = f"{research_dir}/{filename}"
                        if hasattr(self.storage_manager, 'bucket'):
                            blob = self.storage_manager.bucket.blob(blob_path)
                            if blob.exists():
                                research_foundation[phase_key] = blob.download_as_text()
                                
                except Exception as e:
                    logger.debug(f"Could not load {filename}: {e}")
            
            logger.info(f"📋 Loaded {len(research_foundation)} research phases for integration")
            
        except Exception as e:
            logger.warning(f"Error loading research foundation: {e}")
        
        return research_foundation

    async def _integrate_all_research_phases(
        self,
        brand_domain: str,
        research_foundation: Dict[str, Any],
        step_id: str
    ) -> Dict[str, Any]:
        """Integrate all research phases into comprehensive brand intelligence"""
        
        self.progress_tracker.update_progress(step_id, 3, "🎯 Synthesizing brand intelligence...")
        
        # Get research integration prompt
        prompt_template = await self._get_research_integration_prompt()
        
        self.progress_tracker.update_progress(step_id, 4, "🔗 Cross-validating insights...")
        
        # Prepare integration data
        integration_data = {
            "brand_domain": brand_domain,
            "foundation_research": research_foundation.get("foundation", ""),
            "market_positioning": research_foundation.get("market_positioning", ""),
            "product_style": research_foundation.get("product_style", ""),
            "customer_cultural": research_foundation.get("customer_cultural", ""),
            "voice_messaging": research_foundation.get("voice_messaging", ""),
            "interview_synthesis": research_foundation.get("interview_synthesis", ""),
            "linearity_analysis": research_foundation.get("linearity_analysis", "")
        }
        
        # Generate comprehensive research integration using LLM
        integration_content = await self._generate_research_integration(
            prompt_template, 
            integration_data
        )
        
        self.progress_tracker.update_progress(step_id, 5, "📊 Calculating integration quality...")
        
        # Calculate quality score based on research foundation completeness
        quality_score = self._calculate_research_integration_quality_score(research_foundation, integration_content)
        
        return {
            "content": integration_content,
            "confidence": quality_score,
            "source_count": len(research_foundation),
            "analysis_type": "comprehensive_research_integration",
            "research_phases": list(research_foundation.keys())
        }

    async def _get_research_integration_prompt(self) -> str:
        """Get research integration prompt from Langfuse"""
        
        default_prompt = """
You are conducting comprehensive Research Integration for {{brand_domain}} across all 7 completed research phases.

You have access to the complete brand research foundation to create the definitive, integrated brand intelligence document.

Your task is to synthesize all research phases into a cohesive, actionable brand intelligence framework suitable for strategic decision-making.

**COMPLETE RESEARCH FOUNDATION:**
Foundation Research: {{foundation_research}}
Market Positioning: {{market_positioning}}
Product Style: {{product_style}}
Customer Cultural Intelligence: {{customer_cultural}}
Voice & Messaging: {{voice_messaging}}
Interview Synthesis: {{interview_synthesis}}
Linearity Analysis: {{linearity_analysis}}

**RESEARCH INTEGRATION REQUIREMENTS:**

Generate comprehensive integrated brand intelligence covering these 7 sections:

## 1. Executive Brand Intelligence Summary
Synthesize key insights across all research phases:
- Core brand identity and positioning (from foundation & market research)
- Customer segmentation and cultural insights (from customer intelligence)
- Product strategy and style direction (from product style research)
- Voice, messaging, and communication strategy (from voice & interview synthesis)
- Brand consistency and linearity assessment (from linearity analysis)
*Provide strategic overview suitable for executive decision-making*

## 2. Integrated Market Position & Competitive Strategy
Cross-validate market insights with customer and product research:
- Market positioning validated by customer cultural insights
- Competitive advantage confirmed by product style differentiation
- Voice positioning supported by interview synthesis authenticity
- Strategic recommendations based on multi-phase validation
*Reference specific cross-phase validations and strategic recommendations*

## 3. Comprehensive Customer Intelligence
Synthesize customer insights across research phases:
- Customer segments validated across cultural, voice, and market research
- Purchase behavior patterns supported by product style analysis
- Communication preferences confirmed by voice messaging research
- Interview engagement strategies based on synthesis research
*Provide actionable customer engagement framework*

## 4. Product-Brand Alignment Framework
Integrate product intelligence with brand positioning:
- Product style consistency with brand foundation values
- Customer-product fit validated by cultural intelligence
- Market positioning supported by product differentiation
- Voice messaging alignment with product characteristics
*Document comprehensive product-brand strategic alignment*

## 5. Unified Voice & Communication Strategy
Synthesize communication insights across all phases:
- Brand voice validated by foundation, positioning, and customer research
- Interview synthesis confirmed by linearity analysis
- Messaging strategy supported by market positioning insights
- Communication framework suitable for all customer segments
*Provide implementation-ready voice and messaging guidelines*

## 6. Strategic Implementation Roadmap
Cross-validated recommendations from all research phases:
- Immediate tactical recommendations (0-3 months)
- Strategic development priorities (3-12 months)
- Long-term brand evolution direction (1-3 years)
- Risk mitigation based on linearity analysis insights
*Include specific actions, timelines, and success metrics*

## 7. Quality Assurance & Validation Framework
Document research quality and confidence levels:
- Cross-phase validation points and consistency scores
- High-confidence insights suitable for immediate implementation
- Areas requiring additional research or validation
- Overall brand intelligence confidence assessment
*Provide quality framework for ongoing brand development*

**INTEGRATION REQUIREMENTS:**
- Use numbered citations [1], [2], [3] for all cross-phase references
- Provide specific validation examples between research phases
- Quantify confidence levels and quality scores
- Include actionable strategic recommendations
- Focus on implementation-ready insights

**OUTPUT FORMAT:**
Write professional research integration suitable for strategic brand development and executive decision-making.
Focus on synthesized insights that leverage the full spectrum of completed research.
Include confidence levels and validation evidence for all major recommendations.
"""

        prompt = await self.prompt_manager.get_prompt(
            "research_integration",
            default_prompt
        )
        
        return prompt.prompt if prompt else default_prompt

    async def _generate_research_integration(
        self, 
        prompt_template: str, 
        integration_data: Dict[str, Any]
    ) -> str:
        """Generate comprehensive research integration using all research phases"""
        
        # Prepare template variables
        template_vars = {
            "brand_domain": integration_data["brand_domain"],
            "foundation_research": integration_data["foundation_research"][:1400] if integration_data["foundation_research"] else "No foundation research available",
            "market_positioning": integration_data["market_positioning"][:1000] if integration_data["market_positioning"] else "No market positioning available",
            "product_style": integration_data["product_style"][:1000] if integration_data["product_style"] else "No product style research available",
            "customer_cultural": integration_data["customer_cultural"][:1000] if integration_data["customer_cultural"] else "No customer cultural research available",
            "voice_messaging": integration_data["voice_messaging"][:1000] if integration_data["voice_messaging"] else "No voice messaging research available",
            "interview_synthesis": integration_data["interview_synthesis"][:1000] if integration_data["interview_synthesis"] else "No interview synthesis available",
            "linearity_analysis": integration_data["linearity_analysis"][:1000] if integration_data["linearity_analysis"] else "No linearity analysis available"
        }
        
        # Replace template variables
        final_prompt = prompt_template
        for var, value in template_vars.items():
            final_prompt = final_prompt.replace(f"{{{{{var}}}}}", str(value))
        
        # Prepare context for LLM
        context = f"""
BRAND: {integration_data['brand_domain']}

COMPLETE RESEARCH FOUNDATION FOR INTEGRATION:
Foundation Research: {integration_data['foundation_research'][:500] if integration_data['foundation_research'] else 'Not available'}

Market Positioning: {integration_data['market_positioning'][:400] if integration_data['market_positioning'] else 'Not available'}

Product Style: {integration_data['product_style'][:400] if integration_data['product_style'] else 'Not available'}

Customer Cultural: {integration_data['customer_cultural'][:400] if integration_data['customer_cultural'] else 'Not available'}

Voice & Messaging: {integration_data['voice_messaging'][:400] if integration_data['voice_messaging'] else 'Not available'}

Interview Synthesis: {integration_data['interview_synthesis'][:400] if integration_data['interview_synthesis'] else 'Not available'}

Linearity Analysis: {integration_data['linearity_analysis'][:400] if integration_data['linearity_analysis'] else 'Not available'}
"""
        
        response = await LLMFactory.chat_completion(
            task="research_integration",
            system="You are an expert brand strategist specializing in comprehensive research integration and synthesis. Generate definitive brand intelligence frameworks suitable for strategic decision-making based on complete multi-phase research foundations.",
            messages=[
                {"role": "system", "content": final_prompt},
                {"role": "user", "content": context}
            ],
            temperature=0.5,
            max_tokens=5000
        )
        
        return response.get("content", "Research integration generation failed")

    def _calculate_research_integration_quality_score(
        self, 
        research_foundation: Dict[str, Any], 
        integration_content: str
    ) -> float:
        """Calculate quality score for research integration"""
        
        base_score = 0.75  # Base score for research integration
        
        # Research foundation completeness
        foundation_count = len(research_foundation)
        foundation_bonus = min(0.15, foundation_count * 0.02)  # Up to 0.15 for 7+ phases
        
        # Integration content quality
        content_length = len(integration_content)
        content_bonus = min(0.08, content_length / 15000)  # Up to 0.08 for comprehensive integration
        
        # Check for citations in content
        citation_count = integration_content.count("[") + integration_content.count("]")
        citation_bonus = min(0.05, citation_count * 0.002)  # Up to 0.05 for good citations
        
        # Check for strategic integration patterns
        strategy_indicators = integration_content.count("strategic") + integration_content.count("recommend") + integration_content.count("implement")
        strategy_bonus = min(0.04, strategy_indicators * 0.002)  # Bonus for strategic focus
        
        # Check for cross-phase validation
        validation_indicators = integration_content.count("validated") + integration_content.count("confirmed") + integration_content.count("cross-phase")
        validation_bonus = min(0.03, validation_indicators * 0.002)  # Bonus for validation focus
        
        final_score = base_score + foundation_bonus + content_bonus + citation_bonus + strategy_bonus + validation_bonus
        return min(0.92, final_score)  # Cap at 0.92 for research integration

    async def _save_research_integration(self, brand_domain: str, integration_result: Dict[str, Any]) -> List[str]:
        """Save research integration in three-file format"""
        
        saved_files = []
        
        try:
            if hasattr(self.storage_manager, 'base_dir'):
                # Local storage
                research_dir = os.path.join(self.storage_manager.base_dir, "accounts", brand_domain, "research_phases")
                os.makedirs(research_dir, exist_ok=True)
                
                # Save content
                content_path = os.path.join(research_dir, "research_integration.md")
                with open(content_path, "w") as f:
                    f.write(integration_result["content"])
                saved_files.append(content_path)
                
                # Save metadata
                metadata = {
                    "phase": "research_integration",
                    "confidence_score": integration_result.get("confidence", 0.9),
                    "analysis_type": integration_result.get("analysis_type", "comprehensive_research_integration"),
                    "research_phases": integration_result.get("research_phases", []),
                    "source_count": integration_result.get("source_count", 0),
                    "research_metadata": {
                        "phase": "research_integration", 
                        "research_duration_seconds": time.time(),
                        "timestamp": datetime.now().isoformat() + "Z",
                        "quality_threshold": self.quality_threshold,
                        "cache_expires": (datetime.now() + timedelta(days=self.cache_duration_days)).isoformat() + "Z",
                        "version": "2.0_enhanced"
                    }
                }
                
                metadata_path = os.path.join(research_dir, "research_integration_metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                saved_files.append(metadata_path)
                
                # Save sources/data
                sources_data = {
                    "integration_sources": integration_result.get("research_phases", []),
                    "total_research_phases": len(integration_result.get("research_phases", [])),
                    "collection_timestamp": datetime.now().isoformat() + "Z"
                }
                
                sources_path = os.path.join(research_dir, "research_integration_sources.json")
                with open(sources_path, "w") as f:
                    json.dump(sources_data, f, indent=2)
                saved_files.append(sources_path)
                
            else:
                # GCP storage
                research_dir = f"accounts/{brand_domain}/research_phases"
                
                # Save content
                content_blob = f"{research_dir}/research_integration.md"
                blob = self.storage_manager.bucket.blob(content_blob)
                blob.upload_from_string(integration_result["content"])
                saved_files.append(content_blob)
                
                # Save metadata  
                metadata = {
                    "phase": "research_integration",
                    "confidence_score": integration_result.get("confidence", 0.9),
                    "analysis_type": integration_result.get("analysis_type", "comprehensive_research_integration"),
                    "research_phases": integration_result.get("research_phases", []),
                    "source_count": integration_result.get("source_count", 0),
                    "research_metadata": {
                        "phase": "research_integration",
                        "research_duration_seconds": time.time(),
                        "timestamp": datetime.now().isoformat() + "Z",
                        "quality_threshold": self.quality_threshold,
                        "cache_expires": (datetime.now() + timedelta(days=self.cache_duration_days)).isoformat() + "Z",
                        "version": "2.0_enhanced"
                    }
                }
                
                metadata_blob = f"{research_dir}/research_integration_metadata.json"
                blob = self.storage_manager.bucket.blob(metadata_blob)
                blob.upload_from_string(json.dumps(metadata, indent=2))
                saved_files.append(metadata_blob)
                
                # Save sources
                sources_data = {
                    "integration_sources": integration_result.get("research_phases", []),
                    "total_research_phases": len(integration_result.get("research_phases", [])),
                    "collection_timestamp": datetime.now().isoformat() + "Z"
                }
                
                sources_blob = f"{research_dir}/research_integration_sources.json"
                blob = self.storage_manager.bucket.blob(sources_blob)
                blob.upload_from_string(json.dumps(sources_data, indent=2))
                saved_files.append(sources_blob)
            
            logger.info(f"✅ Saved research integration for {brand_domain}")
            
        except Exception as e:
            logger.error(f"❌ Error saving research integration: {e}")
            raise
        
        return saved_files

    async def _load_cached_research_integration(self, brand_domain: str) -> Optional[Dict[str, Any]]:
        """Load cached research integration"""
        try:
            if hasattr(self.storage_manager, 'base_dir'):
                # Local storage
                research_dir = os.path.join(self.storage_manager.base_dir, "accounts", brand_domain, "research_phases")
                
                content_path = os.path.join(research_dir, "research_integration.md")
                metadata_path = os.path.join(research_dir, "research_integration_metadata.json")
                
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
                            "research_integration_content": content,
                            "quality_score": metadata.get("confidence_score", 0.9),
                            "files": [content_path, metadata_path],
                            "data_sources": metadata.get("source_count", 0),
                            "research_method": metadata.get("analysis_type", "cached_integration")
                        }
            else:
                # GCP storage 
                research_dir = f"accounts/{brand_domain}/research_phases"
                
                content_blob = f"{research_dir}/research_integration.md"
                metadata_blob = f"{research_dir}/research_integration_metadata.json"
                
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
                            "research_integration_content": content,
                            "quality_score": metadata.get("confidence_score", 0.9),
                            "files": [content_blob, metadata_blob],
                            "data_sources": metadata.get("source_count", 0),
                            "research_method": metadata.get("analysis_type", "cached_integration")
                        }
        except Exception as e:
            logger.debug(f"Cache check failed: {e}")
        
        return None


def get_research_integration_processor() -> ResearchIntegrationProcessor:
    """Get research integration processor instance"""
    return ResearchIntegrationProcessor()
