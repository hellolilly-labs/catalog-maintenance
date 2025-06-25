"""
Voice & Messaging Analysis Research Phase

Implements Phase 5 of the Brand Research Pipeline per ROADMAP Section 4.5.

Focus: Brand voice and messaging analysis
Cache Duration: 2-3 months (moderate stability)  
Research Time: 1-2 minutes
Quality Threshold: 7.0
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


class VoiceMessagingResearcher:
    """Voice & Messaging Analysis Research Phase Implementation"""
    
    def __init__(self, storage_manager=None):
        self.storage_manager = storage_manager or get_account_storage_provider()
        self.quality_threshold = 7.0
        self.cache_duration_days = 75  # 2.5 months default
        
        self.progress_tracker = ProgressTracker(storage_manager=self.storage_manager, enable_checkpoints=True)
        console_listener = create_console_listener()
        self.progress_tracker.add_progress_listener(console_listener)
        
        # Initialize prompt manager
        self.prompt_manager = PromptManager()
        
    async def research_voice_messaging(self, brand_domain: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Research voice messaging phase for a brand"""
        start_time = time.time()
        
        logger.info(f"🎙️ Starting Voice & Messaging Analysis Research for {brand_domain}")
        
        step_id = self.progress_tracker.create_step(
            step_type=StepType.VOICE_ANALYSIS,
            brand=brand_domain,
            phase_name="Voice & Messaging Analysis Research",
            total_operations=8
        )
        
        try:
            self.progress_tracker.start_step(step_id, "Checking cache and initializing...")
            
            if not force_refresh:
                cached_result = await self._load_cached_voice_messaging(brand_domain)
                if cached_result:
                    self.progress_tracker.complete_step(step_id, cache_hit=True)
                    return cached_result
            
            self.progress_tracker.update_progress(step_id, 1, "📋 Loading foundation context...")
            
            # Load foundation context from previous research phases
            foundation_context = await self._load_foundation_context(brand_domain)
            
            self.progress_tracker.update_progress(step_id, 2, "🎙️ Analyzing brand voice patterns...")
            
            # Conduct voice analysis using foundation context
            analysis_result = await self._analyze_brand_voice_messaging(
                brand_domain,
                foundation_context,
                step_id
            )
            
            self.progress_tracker.update_progress(step_id, 6, "💾 Saving voice messaging research...")
            
            # Save research results in 3-file format
            saved_files = await self._save_voice_messaging_research(brand_domain, analysis_result)
            
            self.progress_tracker.update_progress(step_id, 7, "✅ Finalizing voice messaging analysis...")
            
            duration = time.time() - start_time
            logger.info(f"✅ Voice & Messaging Analysis Research completed in {duration:.1f}s")
            
            self.progress_tracker.complete_step(
                step_id,
                output_files=saved_files,
                quality_score=analysis_result.get("confidence", 0.7),
                cache_hit=False
            )
            
            return {
                "brand": brand_domain,
                "voice_messaging_content": analysis_result.get("content", ""),
                "quality_score": analysis_result.get("confidence", 0.7),
                "files": saved_files,
                "data_sources": analysis_result.get("source_count", 0),
                "research_method": analysis_result.get("analysis_type", "enhanced_voice_analysis")
            }
            
        except Exception as e:
            self.progress_tracker.fail_step(step_id, str(e))
            logger.error(f"❌ Error in voice messaging research: {e}")
            raise

    async def _load_foundation_context(self, brand_domain: str) -> Dict[str, Any]:
        """Load foundation context from previous research phases"""
        
        foundation_context = {}
        
        try:
            # Determine storage base path
            if hasattr(self.storage_manager, 'base_dir'):
                # Local storage
                research_dir = os.path.join(self.storage_manager.base_dir, "accounts", brand_domain, "research_phases")
            else:
                # GCP storage - we'll adapt this for file reading
                research_dir = f"accounts/{brand_domain}/research_phases"
            
            # Load foundation research
            foundation_files = [
                ("foundation_research.md", "foundation"),
                ("market_positioning.md", "market_positioning"),
                ("product_style_research.md", "product_style"),
                ("customer_cultural_research.md", "customer_cultural")
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
            
            logger.info(f"📋 Loaded {len(foundation_context)} foundation research phases for context")
            
        except Exception as e:
            logger.warning(f"Error loading foundation context: {e}")
        
        return foundation_context

    async def _analyze_brand_voice_messaging(
        self,
        brand_domain: str,
        foundation_context: Dict[str, Any],
        step_id: str
    ) -> Dict[str, Any]:
        """Analyze brand voice and messaging patterns"""
        
        self.progress_tracker.update_progress(step_id, 3, "🎯 Analyzing messaging strategy...")
        
        # Get voice messaging analysis prompt
        prompt_template = await self._get_voice_messaging_analysis_prompt()
        
        self.progress_tracker.update_progress(step_id, 4, "🗣️ Generating voice pattern analysis...")
        
        # Prepare analysis data
        analysis_data = {
            "brand_domain": brand_domain,
            "foundation_research": foundation_context.get("foundation", ""),
            "market_positioning": foundation_context.get("market_positioning", ""),
            "product_style": foundation_context.get("product_style", ""),
            "customer_cultural": foundation_context.get("customer_cultural", "")
        }
        
        # Generate voice messaging analysis using LLM
        analysis_content = await self._generate_voice_messaging_analysis(
            prompt_template, 
            analysis_data
        )
        
        self.progress_tracker.update_progress(step_id, 5, "📊 Calculating voice analysis quality...")
        
        # Calculate quality score based on foundation context richness
        quality_score = self._calculate_voice_analysis_quality_score(foundation_context, analysis_content)
        
        return {
            "content": analysis_content,
            "confidence": quality_score,
            "source_count": len(foundation_context),
            "analysis_type": "foundation_enhanced_voice_analysis",
            "foundation_phases": list(foundation_context.keys())
        }

    async def _get_voice_messaging_analysis_prompt(self) -> str:
        """Get voice messaging analysis prompt from Langfuse"""
        
        default_prompt = """
You are conducting comprehensive Voice & Messaging Analysis for {{brand_domain}}.

You have access to foundation research from multiple completed research phases to inform your voice analysis.

Your task is to analyze brand voice, messaging strategy, and communication patterns.

**FOUNDATION RESEARCH CONTEXT:**
Foundation Research: {{foundation_research}}
Market Positioning: {{market_positioning}}
Product Style: {{product_style}}
Customer Cultural Intelligence: {{customer_cultural}}

**ANALYSIS REQUIREMENTS:**

Generate comprehensive voice & messaging intelligence covering these 6 sections:

## 1. Brand Voice Characteristics
Define the core voice attributes:
- Tone and personality (professional, friendly, technical, approachable)
- Communication style (direct, consultative, educational, inspirational)
- Vocabulary and language patterns
- Emotional resonance and connection style
*Use specific examples from foundation research*

## 2. Messaging Architecture & Strategy
Document messaging framework:
- Core value propositions and key messages
- Brand positioning statements and taglines
- Product messaging hierarchy and prioritization
- Competitive differentiation messaging
*Reference market positioning and product style insights*

## 3. Target Audience Communication
Analyze audience-specific messaging:
- Customer segment communication preferences
- Technical vs. accessible language balance
- Performance vs. lifestyle messaging approaches
- Cultural sensitivity and inclusivity in messaging
*Use customer cultural intelligence insights*

## 4. Channel & Context Adaptation
Map voice across touchpoints:
- Website and digital platform voice consistency
- Product description and technical content style
- Marketing campaign and advertising tone
- Customer service and support communication
*Reference brand foundation and style patterns*

## 5. Messaging Themes & Content Pillars
Identify consistent content themes:
- Innovation and technology messaging
- Performance and quality emphasis
- Customer experience and service focus
- Brand heritage and expertise positioning
*Integrate insights from all foundation phases*

## 6. Voice Guidelines & Implementation
Document actionable voice standards:
- Do's and don'ts for brand communication
- Tone variations for different contexts
- Messaging approval criteria and quality standards
- Brand voice evolution and adaptation strategy
*Provide implementation guidance based on research*

**SOURCING REQUIREMENTS:**
- Use numbered citations [1], [2], [3] for all insights
- Reference specific foundation research insights
- Include examples from brand positioning and style research
- Cross-reference customer cultural patterns

**OUTPUT FORMAT:**
Write professional voice & messaging analysis suitable for brand teams.
Focus on actionable guidelines backed by research insights.
Include confidence levels based on foundation data quality.
"""

        prompt = await self.prompt_manager.get_prompt(
            "voice_messaging_analysis",
            default_prompt
        )
        
        return prompt.prompt if prompt else default_prompt

    async def _generate_voice_messaging_analysis(
        self, 
        prompt_template: str, 
        analysis_data: Dict[str, Any]
    ) -> str:
        """Generate voice messaging analysis using foundation research"""
        
        # Prepare template variables
        template_vars = {
            "brand_domain": analysis_data["brand_domain"],
            "foundation_research": analysis_data["foundation_research"][:2000] if analysis_data["foundation_research"] else "No foundation research available",
            "market_positioning": analysis_data["market_positioning"][:1500] if analysis_data["market_positioning"] else "No market positioning available",
            "product_style": analysis_data["product_style"][:1500] if analysis_data["product_style"] else "No product style research available",
            "customer_cultural": analysis_data["customer_cultural"][:1500] if analysis_data["customer_cultural"] else "No customer cultural research available"
        }
        
        # Replace template variables
        final_prompt = prompt_template
        for var, value in template_vars.items():
            final_prompt = final_prompt.replace(f"{{{{{var}}}}}", str(value))
        
        # Prepare context for LLM
        context = f"""
BRAND: {analysis_data['brand_domain']}

FOUNDATION RESEARCH SUMMARY:
Foundation Research: {analysis_data['foundation_research'][:800] if analysis_data['foundation_research'] else 'Not available'}

Market Positioning: {analysis_data['market_positioning'][:600] if analysis_data['market_positioning'] else 'Not available'}

Product Style: {analysis_data['product_style'][:600] if analysis_data['product_style'] else 'Not available'}

Customer Cultural: {analysis_data['customer_cultural'][:600] if analysis_data['customer_cultural'] else 'Not available'}
"""
        
        response = await LLMFactory.chat_completion(
            task="voice_messaging_research",
            system="You are an expert brand voice analyst specializing in messaging strategy and communication patterns. Generate comprehensive, actionable voice & messaging analysis based on foundation research.",
            messages=[
                {"role": "system", "content": final_prompt},
                {"role": "user", "content": context}
            ],
            temperature=0.7,
            max_tokens=4000
        )
        
        return response.get("content", "Voice messaging analysis generation failed")

    def _calculate_voice_analysis_quality_score(
        self, 
        foundation_context: Dict[str, Any], 
        analysis_content: str
    ) -> float:
        """Calculate quality score for voice messaging analysis"""
        
        base_score = 0.6  # Base score for voice analysis
        
        # Foundation context quality
        context_count = len(foundation_context)
        context_bonus = min(0.15, context_count * 0.04)  # Up to 0.15 for 4+ phases
        
        # Analysis content quality
        content_length = len(analysis_content)
        content_bonus = min(0.08, content_length / 10000)  # Up to 0.08 for comprehensive analysis
        
        # Check for citations in content
        citation_count = analysis_content.count("[") + analysis_content.count("]")
        citation_bonus = min(0.05, citation_count * 0.003)  # Up to 0.05 for good citations
        
        # Check for actionable guidelines
        guideline_indicators = analysis_content.count("Do:") + analysis_content.count("Don't:") + analysis_content.count("Guidelines")
        guideline_bonus = min(0.05, guideline_indicators * 0.01)  # Bonus for actionable content
        
        # Check for messaging themes
        theme_indicators = analysis_content.count("messaging") + analysis_content.count("voice") + analysis_content.count("tone")
        theme_bonus = min(0.02, theme_indicators * 0.001)  # Bonus for voice-focused content
        
        final_score = base_score + context_bonus + content_bonus + citation_bonus + guideline_bonus + theme_bonus
        return min(0.85, final_score)  # Cap at 0.85 for voice analysis

    async def _save_voice_messaging_research(self, brand_domain: str, analysis_result: Dict[str, Any]) -> List[str]:
        """Save voice messaging research in three-file format"""
        
        saved_files = []
        
        try:
            if hasattr(self.storage_manager, 'base_dir'):
                # Local storage
                research_dir = os.path.join(self.storage_manager.base_dir, "accounts", brand_domain, "research_phases")
                os.makedirs(research_dir, exist_ok=True)
                
                # Save content
                content_path = os.path.join(research_dir, "voice_messaging_research.md")
                with open(content_path, "w") as f:
                    f.write(analysis_result["content"])
                saved_files.append(content_path)
                
                # Save metadata
                metadata = {
                    "phase": "voice_messaging",
                    "confidence_score": analysis_result.get("confidence", 0.7),
                    "analysis_type": analysis_result.get("analysis_type", "enhanced_voice_analysis"),
                    "foundation_phases": analysis_result.get("foundation_phases", []),
                    "source_count": analysis_result.get("source_count", 0),
                    "research_metadata": {
                        "phase": "voice_messaging", 
                        "research_duration_seconds": time.time(),
                        "timestamp": datetime.now().isoformat() + "Z",
                        "quality_threshold": self.quality_threshold,
                        "cache_expires": (datetime.now() + timedelta(days=self.cache_duration_days)).isoformat() + "Z",
                        "version": "2.0_enhanced"
                    }
                }
                
                metadata_path = os.path.join(research_dir, "voice_messaging_research_metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                saved_files.append(metadata_path)
                
                # Save sources/data
                sources_data = {
                    "analysis_sources": analysis_result.get("foundation_phases", []),
                    "foundation_research_phases": len(analysis_result.get("foundation_phases", [])),
                    "collection_timestamp": datetime.now().isoformat() + "Z"
                }
                
                sources_path = os.path.join(research_dir, "voice_messaging_research_sources.json")
                with open(sources_path, "w") as f:
                    json.dump(sources_data, f, indent=2)
                saved_files.append(sources_path)
                
            else:
                # GCP storage
                research_dir = f"accounts/{brand_domain}/research_phases"
                
                # Save content
                content_blob = f"{research_dir}/voice_messaging_research.md"
                blob = self.storage_manager.bucket.blob(content_blob)
                blob.upload_from_string(analysis_result["content"])
                saved_files.append(content_blob)
                
                # Save metadata  
                metadata = {
                    "phase": "voice_messaging",
                    "confidence_score": analysis_result.get("confidence", 0.7),
                    "analysis_type": analysis_result.get("analysis_type", "enhanced_voice_analysis"),
                    "foundation_phases": analysis_result.get("foundation_phases", []),
                    "source_count": analysis_result.get("source_count", 0),
                    "research_metadata": {
                        "phase": "voice_messaging",
                        "research_duration_seconds": time.time(),
                        "timestamp": datetime.now().isoformat() + "Z",
                        "quality_threshold": self.quality_threshold,
                        "cache_expires": (datetime.now() + timedelta(days=self.cache_duration_days)).isoformat() + "Z",
                        "version": "2.0_enhanced"
                    }
                }
                
                metadata_blob = f"{research_dir}/voice_messaging_research_metadata.json"
                blob = self.storage_manager.bucket.blob(metadata_blob)
                blob.upload_from_string(json.dumps(metadata, indent=2))
                saved_files.append(metadata_blob)
                
                # Save sources
                sources_data = {
                    "analysis_sources": analysis_result.get("foundation_phases", []),
                    "foundation_research_phases": len(analysis_result.get("foundation_phases", [])),
                    "collection_timestamp": datetime.now().isoformat() + "Z"
                }
                
                sources_blob = f"{research_dir}/voice_messaging_research_sources.json"
                blob = self.storage_manager.bucket.blob(sources_blob)
                blob.upload_from_string(json.dumps(sources_data, indent=2))
                saved_files.append(sources_blob)
            
            logger.info(f"✅ Saved voice messaging research for {brand_domain}")
            
        except Exception as e:
            logger.error(f"❌ Error saving voice messaging research: {e}")
            raise
        
        return saved_files

    async def _load_cached_voice_messaging(self, brand_domain: str) -> Optional[Dict[str, Any]]:
        """Load cached voice messaging research"""
        try:
            if hasattr(self.storage_manager, 'base_dir'):
                # Local storage
                research_dir = os.path.join(self.storage_manager.base_dir, "accounts", brand_domain, "research_phases")
                
                content_path = os.path.join(research_dir, "voice_messaging_research.md")
                metadata_path = os.path.join(research_dir, "voice_messaging_research_metadata.json")
                
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
                            "voice_messaging_content": content,
                            "quality_score": metadata.get("confidence_score", 0.7),
                            "files": [content_path, metadata_path],
                            "data_sources": metadata.get("source_count", 0),
                            "research_method": metadata.get("analysis_type", "cached_analysis")
                        }
            else:
                # GCP storage 
                research_dir = f"accounts/{brand_domain}/research_phases"
                
                content_blob = f"{research_dir}/voice_messaging_research.md"
                metadata_blob = f"{research_dir}/voice_messaging_research_metadata.json"
                
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
                            "voice_messaging_content": content,
                            "quality_score": metadata.get("confidence_score", 0.7),
                            "files": [content_blob, metadata_blob],
                            "data_sources": metadata.get("source_count", 0),
                            "research_method": metadata.get("analysis_type", "cached_analysis")
                        }
                        
        except Exception as e:
            logger.debug(f"Cache check failed: {e}")
        
        return None


def get_voice_messaging_researcher() -> VoiceMessagingResearcher:
    """Get voice messaging researcher instance"""
    return VoiceMessagingResearcher() 