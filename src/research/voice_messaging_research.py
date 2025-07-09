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
from src.research.base_researcher import BaseResearcher

logger = logging.getLogger(__name__)


class VoiceMessagingResearcher(BaseResearcher):
    """Voice & Messaging Analysis Research Phase Implementation"""
    
    def __init__(self, brand_domain: str, storage_manager=None):
        super().__init__(
            brand_domain=brand_domain,
            researcher_name="voice_messaging",
            step_type=StepType.VOICE_MESSAGING,
            quality_threshold=7.0,
            cache_duration_days=75,
            storage_manager=storage_manager
        )
        
    # async def research_voice_messaging(self, brand_domain: str, force_refresh: bool = False) -> Dict[str, Any]:
    #     """Research voice messaging phase for a brand"""
        
    #     logger.info(f"🎙️ Starting Voice & Messaging Analysis Research for {brand_domain}")
        
    #     # Use the base class research method
    #     result = await self.research(force_refresh=force_refresh)
        
    #     return {
    #         "brand": brand_domain,
    #         "voice_messaging_content": result.get("content", ""),
    #         "quality_score": result.get("quality_score", 0.7),
    #         "files": result.get("files", []),
    #         "data_sources": result.get("data_sources", 0),
    #         "research_method": result.get("research_method", "enhanced_voice_messaging")
    #     }

    async def _gather_data(self) -> Dict[str, Any]:
        """Load foundation context from previous research phases - implements BaseResearcher abstract method"""
        
        context = {}
        
        try:
            # Load foundation research
            foundation_files = [
                "foundation",
                "market_positioning",
                "product_style",
                "customer_cultural"
            ]
            
            for phase_key in foundation_files:
                try:
                    blob_path = f"research/{phase_key}/research.md"
                    context[phase_key] = await self.storage_manager.read_file(account=self.brand_domain, file_path=blob_path)
                                
                except Exception as e:
                    logger.debug(f"Could not load {phase_key} research: {e}")
            
            logger.info(f"📋 Loaded {len(context)} foundation research phases for context")
            
            # Return in format expected by base class
            return {
                "brand_domain": self.brand_domain,
                "brand_name": self.brand_domain.replace('.com', '').replace('.', ' ').title(),
                "search_results": [],  # Voice analysis doesn't use web search
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
        """Get the default user prompt for voice messaging analysis - implements BaseResearcher abstract method"""
        
        default_prompt = """Conduct a comprehensive Voice & Messaging Analysis for {{brand_domain}}.

You have access to foundation research from multiple completed research phases to inform your voice analysis.

Your task is to analyze brand voice, messaging strategy, and communication patterns.

**ANALYSIS REQUIREMENTS:**

Generate comprehensive voice & messaging intelligence covering these 6 sections:

## 1. Brand Voice Characteristics
Define the core voice attributes:
- Tone and personality (professional, friendly, technical, approachable)
- Communication style (direct, consultative, educational, inspirational)
- Vocabulary and language patterns (language choices and terminology preferences)
- Emotional resonance and connection style (connection style and emotional approach)

*Use specific examples from foundation research*

## 2. Messaging Architecture & Strategy
Document messaging framework:
- Core value propositions and key messages (key brand messages and value statements)
- Brand positioning statements and taglines (market positioning and taglines)
- Product messaging hierarchy and prioritization (message prioritization and structure)
- Competitive differentiation messaging (unique messaging advantages)

*Reference market positioning and product style insights*

## 3. Target Audience Communication
Analyze audience-specific messaging:
- Customer segment communication preferences (communication style by audience)
- Technical vs. accessible language balance (complexity level management)
- Performance vs. lifestyle messaging approaches (approach variations)
- Cultural sensitivity and inclusivity in messaging (inclusivity and cultural awareness)

*Use customer cultural intelligence insights*

## 4. Channel & Context Adaptation
Map voice across touchpoints:
- Website and digital platform voice consistency (website and online consistency)
- Product description and technical content style (description and technical content tone)
- Marketing campaign and advertising tone (advertising and campaign voice)
- Customer service and support communication (support and service style)

*Reference brand foundation and style patterns*

## 5. Messaging Themes & Content Pillars
Identify consistent content themes:
- Innovation and technology messaging (tech-focused messaging themes)
- Performance and quality emphasis (excellence and achievement emphasis)
- Customer experience and service focus (service and experience messaging)
- Brand heritage and expertise positioning (expertise and legacy positioning)

*Integrate insights from all foundation phases*

## 6. Voice Guidelines & Implementation
Document actionable voice standards:
- Do's and don'ts for brand communication (recommended voice practices)
- Tone variations for different contexts (context-specific adaptations)
- Messaging approval criteria and quality standards (messaging approval criteria and quality standards)
- Brand voice evolution and adaptation strategy (voice development and adaptation)

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

**FOUNDATION RESEARCH CONTEXT:**
# Foundation Research:

{{foundation}}

# Market Positioning:

{{market_positioning}}

# Product Style: 

{{product_style}}

# Customer Cultural Intelligence:

{{customer_cultural}}
"""

#         default_prompt = """Analyze foundation research data to extract comprehensive voice & messaging intelligence.

# **Brand:** {{brand_name}}
# **Domain:** {{brand_domain}}

# ## Foundation Research Context:
# {{search_context}}

# ## Voice & Messaging Analysis Requirements:

# Create a comprehensive voice & messaging intelligence report in **markdown format**. 

# Structure your analysis as follows:

# # Voice & Messaging Intelligence: {{brand_name}}

# ## 1. Brand Voice Characteristics
# - **Tone and Personality:** [Professional, friendly, technical, approachable characteristics] [cite sources]
# - **Communication Style:** [Direct, consultative, educational, inspirational approaches] [cite sources]
# - **Vocabulary Patterns:** [Language choices and terminology preferences] [cite sources]
# - **Emotional Resonance:** [Connection style and emotional approach] [cite sources]

# ## 2. Messaging Architecture & Strategy
# - **Core Value Propositions:** [Key brand messages and value statements] [cite sources]
# - **Brand Positioning Statements:** [Market positioning and taglines] [cite sources]
# - **Product Messaging Hierarchy:** [Message prioritization and structure] [cite sources]
# - **Competitive Differentiation:** [Unique messaging advantages] [cite sources]

# ## 3. Target Audience Communication
# - **Customer Segment Preferences:** [Communication style by audience] [cite sources]
# - **Technical vs. Accessible Balance:** [Complexity level management] [cite sources]
# - **Performance vs. Lifestyle Messaging:** [Approach variations] [cite sources]
# - **Cultural Sensitivity:** [Inclusivity and cultural awareness] [cite sources]

# ## 4. Channel & Context Adaptation
# - **Digital Platform Voice:** [Website and online consistency] [cite sources]
# - **Product Content Style:** [Description and technical content tone] [cite sources]
# - **Marketing Campaign Tone:** [Advertising and campaign voice] [cite sources]
# - **Customer Service Communication:** [Support and service style] [cite sources]

# ## 5. Messaging Themes & Content Pillars
# - **Innovation and Technology:** [Tech-focused messaging themes] [cite sources]
# - **Performance and Quality:** [Excellence and achievement emphasis] [cite sources]
# - **Customer Experience Focus:** [Service and experience messaging] [cite sources]
# - **Brand Heritage:** [Expertise and legacy positioning] [cite sources]

# ## 6. Voice Guidelines & Implementation
# - **Communication Do's:** [Recommended voice practices] [cite sources]
# - **Communication Don'ts:** [Voice practices to avoid] [cite sources]
# - **Tone Variations:** [Context-specific adaptations] [cite sources]
# - **Evolution Strategy:** [Voice development and adaptation] [cite sources]

# ## Analysis Quality & Confidence

# **Foundation Sources:** {{total_sources}} research phases analyzed
# **Information Quality:** {{information_quality}}
# **Confidence Level:** {{confidence_level}} confidence in findings

# ## Summary

# [Provide a 2-3 sentence executive summary of the brand's voice and messaging approach]

# ---

# **Important Instructions:**
# - Base analysis on foundation research insights from previous phases
# - Focus on voice and messaging patterns consistent across research
# - Provide actionable guidelines for brand communication teams
# - Note confidence levels based on foundation data availability
# - Use markdown formatting for structure and readability"""

        return default_prompt

    # def _get_default_instruction_prompt(self) -> str:
    #     """Get the default instruction prompt for voice messaging analysis - implements BaseResearcher abstract method"""
        
    #     return "You are an expert brand voice analyst specializing in messaging strategy and communication patterns. Generate comprehensive, actionable voice & messaging analysis based on foundation research from previous brand research phases. Focus on consistency, guidelines, and implementation recommendations."

    async def _analyze_data(
        self,
        context: Dict[str, Any],
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """Analyze brand voice and messaging patterns"""
        
        step_id = await self.progress_tracker.create_step(
            step_type=self.step_type,
            brand=self.brand_domain,
            phase_name=f"{self.researcher_name.capitalize()} Research",
            total_operations=6  # data gathering, analysis, synthesis, validation, storage, completion
        )
        
        await self.progress_tracker.update_progress(step_id, 3, "🎯 Analyzing messaging strategy...")
        
        await self.progress_tracker.update_progress(step_id, 4, "🗣️ Generating voice pattern analysis...")
        
        context_data = context.get('context')
        
        # Prepare analysis data
        analysis_data = {
            "brand_domain": self.brand_domain,
            "foundation": context_data.get("foundation", ""),
            "market_positioning": context_data.get("market_positioning", ""),
            "product_style": context_data.get("product_style", ""),
            "customer_cultural": context_data.get("customer_cultural", "")
        }
        
        # Generate voice messaging analysis using LLM
        analysis_content = await self._generate_voice_messaging_analysis(
            analysis_data
        )
        
        await self.progress_tracker.update_progress(step_id, 5, "📊 Calculating voice analysis quality...")
        
        # Calculate quality score based on foundation context richness
        quality_score = self._calculate_voice_messaging_quality_score(context_data, analysis_content)
        
        return {
            "markdown": analysis_content,
            "confidence": quality_score,
            "source_count": len(context_data),
            "analysis_type": "foundation_enhanced_voice_messaging",
            "foundation_phases": list(context_data.keys())
        }

    def _get_default_instruction_prompt(self) -> str:
        """Get voice messaging analysis prompt from Langfuse"""
        
        default_prompt = """
You are an expert brand voice analyst specializing in messaging strategy and communication patterns. Generate comprehensive, actionable voice & messaging analysis based on foundation research from previous brand research phases. Focus on consistency, guidelines, and implementation recommendations.
"""

        return default_prompt

    async def _generate_voice_messaging_analysis(
        self, 
        analysis_data: Dict[str, Any]
    ) -> str:
        """Generate voice messaging analysis using foundation research"""
        
        # # Prepare template variables
        # system_prompt_vars = {
        #     "brand_domain": analysis_data["brand_domain"],
        #     "foundation": analysis_data["foundation"][:2000] if analysis_data["foundation"] else "No foundation research available",
        #     "market_positioning": analysis_data["market_positioning"][:1500] if analysis_data["market_positioning"] else "No market positioning available",
        #     "product_style": analysis_data["product_style"][:1500] if analysis_data["product_style"] else "No product style research available",
        #     "customer_cultural": analysis_data["customer_cultural"][:1500] if analysis_data["customer_cultural"] else "No customer cultural research available"
        # }

        # Get user prompt from Langfuse
        user_prompt_vars = {
            "brand_domain": analysis_data["brand_domain"],
            "foundation": analysis_data["foundation"][:5000] if analysis_data["foundation"] else "No foundat{ion research available",
            "market_positioning": analysis_data["market_positioning"][:5000] if analysis_data["market_positioning"] else "No market positioning available",
            "product_style": analysis_data["product_style"][:5000] if analysis_data["product_style"] else "No product style research available",
            "customer_cultural": analysis_data["customer_cultural"][:5000] if analysis_data["customer_cultural"] else "No customer cultural research available"
        }

        # Get prompt from Langfuse
        prompt_template = await self.prompt_manager.get_prompt(
            prompt_name="internal/researcher/voice_messaging_analysis",
            prompt_type="chat",
            prompt=[
                {"role": "system", "content": self._get_default_instruction_prompt()},
                {"role": "user", "content": self._get_default_user_prompt()}
            ]
        )
        prompts = prompt_template.prompt
        
        system_prompt = next((msg["content"] for msg in prompts if msg["role"] == "system"), None)
        user_prompt = next((msg["content"] for msg in prompts if msg["role"] == "user"), None)
        
        # for var, value in system_prompt_vars.items():
        #     system_prompt = system_prompt.replace(f"{{{{{var}}}}}", str(value))
        
        for var, value in user_prompt_vars.items():
            user_prompt = user_prompt.replace(f"{{{{{var}}}}}", str(value))
        
        response = await LLMFactory.chat_completion(
            task="voice_messaging_research",
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            # max_tokens=4000
        )
        
        return response.get("content", "Voice messaging analysis generation failed")

    def _calculate_voice_messaging_quality_score(
        self, 
        context_data: Dict[str, Any], 
        analysis_content: str
    ) -> float:
        """Calculate quality score for voice messaging analysis"""
        
        base_score = 0.6  # Base score for voice analysis
        
        # Foundation context quality
        context_count = len(context_data)
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
        return min(1.0, final_score)  # Cap at 1.0 for voice analysis

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
                    "analysis_type": analysis_result.get("analysis_type", "enhanced_voice_messaging"),
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
                    "analysis_type": analysis_result.get("analysis_type", "enhanced_voice_messaging"),
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

    # async def _load_cached_voice_messaging(self, brand_domain: str) -> Optional[Dict[str, Any]]:
    #     """Load cached voice messaging research"""
    #     try:
    #         if hasattr(self.storage_manager, 'base_dir'):
    #             # Local storage
    #             research_dir = os.path.join(self.storage_manager.base_dir, "accounts", brand_domain, "research_phases")
                
    #             content_path = os.path.join(research_dir, "voice_messaging_research.md")
    #             metadata_path = os.path.join(research_dir, "voice_messaging_research_metadata.json")
                
    #             if all(os.path.exists(p) for p in [content_path, metadata_path]):
    #                 # Check cache expiry
    #                 with open(metadata_path, "r") as f:
    #                     metadata = json.load(f)
                    
    #                 research_metadata = metadata.get("research_metadata", {})
    #                 cache_expires = research_metadata.get("cache_expires")
                    
    #                 if cache_expires and datetime.now() < datetime.fromisoformat(cache_expires.replace("Z", "")):
    #                     # Load cached data
    #                     with open(content_path, "r") as f:
    #                         content = f.read()
                        
    #                     return {
    #                         "brand": brand_domain,
    #                         "voice_messaging_content": content,
    #                         "quality_score": metadata.get("confidence_score", 0.7),
    #                         "files": [content_path, metadata_path],
    #                         "data_sources": metadata.get("source_count", 0),
    #                         "research_method": metadata.get("analysis_type", "cached_analysis")
    #                     }
    #         else:
    #             # GCP storage 
    #             research_dir = f"accounts/{brand_domain}/research_phases"
                
    #             content_blob = f"{research_dir}/voice_messaging_research.md"
    #             metadata_blob = f"{research_dir}/voice_messaging_research_metadata.json"
                
    #             content_blob_obj = self.storage_manager.bucket.blob(content_blob)
    #             metadata_blob_obj = self.storage_manager.bucket.blob(metadata_blob)
                
    #             if content_blob_obj.exists() and metadata_blob_obj.exists():
    #                 # Check cache expiry
    #                 metadata_content = metadata_blob_obj.download_as_text()
    #                 metadata = json.loads(metadata_content)
                    
    #                 research_metadata = metadata.get("research_metadata", {})
    #                 cache_expires = research_metadata.get("cache_expires")
                    
    #                 if cache_expires and datetime.now() < datetime.fromisoformat(cache_expires.replace("Z", "")):
    #                     # Load cached data
    #                     content = content_blob_obj.download_as_text()
                        
    #                     return {
    #                         "brand": brand_domain,
    #                         "voice_messaging_content": content,
    #                         "quality_score": metadata.get("confidence_score", 0.7),
    #                         "files": [content_blob, metadata_blob],
    #                         "data_sources": metadata.get("source_count", 0),
    #                         "research_method": metadata.get("analysis_type", "cached_analysis")
    #                     }
                        
    #     except Exception as e:
    #         logger.debug(f"Cache check failed: {e}")
        
    #     return None


def get_voice_messaging_researcher(brand_domain: str) -> VoiceMessagingResearcher:
    """Get voice messaging researcher instance"""
    return VoiceMessagingResearcher(brand_domain) 