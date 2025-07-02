"""
AI Brand Ethos Voice Interview Synthesis Research Phase

Implements Phase 6 of the Brand Research Pipeline per ROADMAP Section 4.6.

Focus: AI brand ethos voice interview synthesis
Cache Duration: 4-6 months (high stability)
Research Time: 3-5 minutes
Quality Threshold: 8.0
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from liddy.storage import get_account_storage_provider
from liddy_intelligence.progress_tracker import ProgressTracker, StepType, create_console_listener
from liddy_intelligence.llm.simple_factory import LLMFactory
from liddy.prompt_manager import PromptManager
from liddy_intelligence.research.base_researcher import BaseResearcher

logger = logging.getLogger(__name__)


class InterviewSynthesisResearcher(BaseResearcher):
    """AI Brand Ethos Voice Interview Synthesis Research Phase Implementation"""
    
    def __init__(self, brand_domain: str, storage_manager=None):
        super().__init__(
            brand_domain=brand_domain,
            researcher_name="interview_synthesis",
            step_type=StepType.INTERVIEW_INTEGRATION,
            quality_threshold=8.0,
            cache_duration_days=150,
            storage_manager=storage_manager
        )
        
    # async def research_interview_synthesis(self, brand_domain: str, force_refresh: bool = False) -> Dict[str, Any]:
    #     """Research interview synthesis phase for a brand"""
        
    #     logger.info(f"🤖 Starting AI Brand Ethos Voice Interview Synthesis Research for {brand_domain}")
        
    #     # Use the base class research method
    #     result = await self.research(force_refresh=force_refresh)
        
    #     return {
    #         "brand": brand_domain,
    #         "interview_synthesis_content": result.get("content", ""),
    #         "quality_score": result.get("quality_score", 0.8),
    #         "files": result.get("files", []),
    #         "data_sources": result.get("data_sources", 0),
    #         "research_method": result.get("research_method", "enhanced_interview_synthesis")
    #     }

    async def _gather_data(self) -> Dict[str, Any]:
        """Load foundation context from previous research phases - implements BaseResearcher abstract method"""
        
        foundation_context = {}
        
        try:
            # Determine storage base path
            if hasattr(self.storage_manager, 'base_dir'):
                # Local storage
                research_dir = os.path.join(self.storage_manager.base_dir, "accounts", self.brand_domain, "research_phases")
            else:
                # GCP storage - we'll adapt this for file reading
                research_dir = f"accounts/{self.brand_domain}/research_phases"
            
            # Load all previous research phases
            foundation_files = [
                ("foundation_research.md", "foundation"),
                ("market_positioning.md", "market_positioning"),
                ("product_style_research.md", "product_style"),
                ("customer_cultural_research.md", "customer_cultural"),
                ("voice_messaging_research.md", "voice_messaging")
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
            
            logger.info(f"📋 Loaded {len(foundation_context)} foundation research phases for interview synthesis")
            
            # Return in format expected by base class
            return {
                "brand_domain": self.brand_domain,
                "brand_name": self.brand_domain.replace('.com', '').replace('.', ' ').title(),
                "search_results": [],  # Interview synthesis doesn't use web search
                "detailed_sources": [],
                "foundation_context": foundation_context,
                "total_sources": len(foundation_context),
                "search_stats": {
                    "successful_searches": len(foundation_context),
                    "failed_searches": 0,
                    "success_rate": 1.0 if foundation_context else 0.0,
                    "ssl_errors": 0
                }
            }
            
        except Exception as e:
            logger.warning(f"Error loading foundation context: {e}")
            raise RuntimeError(f"Failed to load foundation context: {e}")

    def _get_default_user_prompt(self) -> str:
        """Get the default user prompt for interview synthesis - implements BaseResearcher abstract method"""
        
        default_prompt = """Synthesize foundation research data to create comprehensive AI brand ethos voice interview synthesis.

**Brand:** {{brand_name}}
**Domain:** {{brand_domain}}

## Foundation Research Context:
{{search_context}}

## AI Brand Ethos Voice Interview Synthesis Requirements:

Create a comprehensive AI brand ethos voice interview synthesis in **markdown format**.

Structure your synthesis as follows:

# AI Brand Ethos Voice Interview Synthesis: {{brand_name}}

## 1. AI Brand Personality Profile
- **Personality Traits:** [Authoritative, approachable, innovative, reliable characteristics] [cite sources]
- **Communication Style:** [Speaking patterns and conversational approach] [cite sources]
- **Emotional Intelligence:** [Responsiveness style and emotional approach] [cite sources]
- **Expertise Areas:** [Knowledge depth and specialization areas] [cite sources]

## 2. Interview Voice & Tone Framework
- **Response Patterns:** [Question response depth and style] [cite sources]
- **Language Balance:** [Technical vs. accessible communication] [cite sources]
- **Storytelling Style:** [Narrative approach and structure] [cite sources]
- **Tone Adaptation:** [Professional vs. conversational flexibility] [cite sources]

## 3. Brand Ethos Expression Patterns
- **Innovation Leadership:** [How to express innovation in conversations] [cite sources]
- **Customer Philosophy:** [Customer-first approach in responses] [cite sources]
- **Quality Emphasis:** [Performance and quality discussion style] [cite sources]
- **Heritage Showcase:** [Expertise and legacy presentation] [cite sources]

## 4. Conversation Context Adaptation
- **Technical Discussions:** [Product and technical conversation approach] [cite sources]
- **Strategy Conversations:** [Business vision and strategy style] [cite sources]
- **Customer Stories:** [Experience sharing and storytelling] [cite sources]
- **Thought Leadership:** [Industry positioning and expertise] [cite sources]

## 5. AI Response Authenticity Guidelines
- **Messaging Consistency:** [Consistent themes across conversations] [cite sources]
- **Brand Terminology:** [Specific language and vocabulary patterns] [cite sources]
- **Expertise Demonstration:** [Knowledge showcase without overselling] [cite sources]
- **Passion Areas:** [Genuine energy and enthusiasm points] [cite sources]

## 6. Interview Implementation Framework
- **Voice Do's:** [Recommended AI interview practices] [cite sources]
- **Voice Don'ts:** [Practices to avoid in conversations] [cite sources]
- **Response Guidelines:** [Depth and detail standards] [cite sources]
- **Quality Standards:** [Authenticity and brand representation criteria] [cite sources]

## Analysis Quality & Confidence

**Foundation Sources:** {{total_sources}} research phases analyzed
**Information Quality:** {{information_quality}}
**Confidence Level:** {{confidence_level}} confidence in synthesis

## Summary

[Provide a 2-3 sentence executive summary of the AI brand ethos voice approach]

---

**Important Instructions:**
- Base synthesis on comprehensive foundation research from all phases
- Focus on AI implementation guidelines and authentic voice patterns
- Provide actionable voice guidelines for AI brand representation
- Include confidence levels based on foundation data quality
- Use markdown formatting for structure and readability"""

        return default_prompt

    def _get_default_instruction_prompt(self) -> str:
        """Get the default instruction prompt for interview synthesis - implements BaseResearcher abstract method"""
        
        return "You are an expert AI brand voice specialist creating authentic brand ethos interview synthesis. Generate comprehensive, implementation-ready AI voice guidelines based on complete foundation research from all previous brand research phases. Focus on creating authentic, conversational AI voice patterns."

    async def _synthesize_brand_ethos_interview(
        self,
        brand_domain: str,
        foundation_context: Dict[str, Any],
        step_id: str
    ) -> Dict[str, Any]:
        """Synthesize AI brand ethos voice interview from research"""
        
        await self.progress_tracker.update_progress(step_id, 3, "🎯 Analyzing brand personality patterns...")
        
        # Get interview synthesis prompt
        prompt_template = await self._get_interview_synthesis_prompt()
        
        await self.progress_tracker.update_progress(step_id, 4, "🗣️ Generating AI brand interview voice...")
        
        # Prepare synthesis data
        synthesis_data = {
            "brand_domain": brand_domain,
            "foundation": foundation_context.get("foundation", ""),
            "market_positioning": foundation_context.get("market_positioning", ""),
            "product_style": foundation_context.get("product_style", ""),
            "customer_cultural": foundation_context.get("customer_cultural", ""),
            "voice_messaging": foundation_context.get("voice_messaging", "")
        }
        
        # Generate interview synthesis using LLM
        synthesis_content = await self._generate_interview_synthesis(
            prompt_template, 
            synthesis_data
        )
        
        await self.progress_tracker.update_progress(step_id, 5, "📊 Calculating synthesis quality...")
        
        # Calculate quality score based on foundation context richness
        quality_score = self._calculate_interview_synthesis_quality_score(foundation_context, synthesis_content)
        
        return {
            "content": synthesis_content,
            "confidence": quality_score,
            "source_count": len(foundation_context),
            "analysis_type": "foundation_enhanced_interview_synthesis",
            "foundation_phases": list(foundation_context.keys())
        }

    async def _get_interview_synthesis_prompt(self) -> str:
        """Get interview synthesis prompt from Langfuse"""
        
        default_prompt = """
You are conducting comprehensive AI Brand Ethos Voice Interview Synthesis for {{brand_domain}}.

You have access to complete foundation research from 5 research phases to create an authentic AI brand voice interview.

Your task is to synthesize an AI-driven brand ethos voice that represents the authentic brand personality for interviews and conversations.

**FOUNDATION RESEARCH CONTEXT:**
Foundation Research: {{foundation_research}}
Market Positioning: {{market_positioning}}
Product Style: {{product_style}}
Customer Cultural Intelligence: {{customer_cultural}}
Voice & Messaging: {{voice_messaging}}

**SYNTHESIS REQUIREMENTS:**

Generate comprehensive AI brand ethos voice interview synthesis covering these 6 sections:

## 1. AI Brand Personality Profile
Define the core AI brand persona:
- Personality traits and characteristics (authoritative, approachable, innovative, reliable)
- Communication style and speaking patterns
- Emotional intelligence and responsiveness style
- Brand expertise areas and knowledge depth
*Use specific insights from foundation and voice research*

## 2. Interview Voice & Tone Framework
Document AI interview approach:
- Question response patterns and depth of answers
- Technical vs. accessible language balance in conversations
- Storytelling style and narrative approach
- Professional vs. conversational tone adaptation
*Reference voice messaging and market positioning insights*

## 3. Brand Ethos Expression Patterns
Map core brand values into conversational voice:
- Innovation leadership expression in interviews
- Customer-first philosophy in responses
- Quality and performance emphasis in discussions
- Heritage and expertise showcase approach
*Integrate insights from all foundation phases*

## 4. Conversation Context Adaptation
Define voice variations across contexts:
- Technical product discussions
- Business strategy and vision conversations
- Customer story and experience sharing
- Industry thought leadership positioning
*Reference customer cultural and product style patterns*

## 5. AI Response Authenticity Guidelines
Document authentic brand voice patterns:
- Consistent messaging themes across conversations
- Brand-specific terminology and language patterns
- Expertise demonstration without overselling
- Genuine passion points and energy areas
*Use foundation research and voice messaging analysis*

## 6. Interview Implementation Framework
Provide actionable synthesis guidelines:
- Do's and don'ts for AI brand interview voice
- Response depth and detail guidelines
- Brand story integration in conversations
- Quality standards for authentic brand representation
*Provide implementation guidance based on research synthesis*

**SOURCING REQUIREMENTS:**
- Use numbered citations [1], [2], [3] for all insights
- Reference specific foundation research insights
- Include examples from voice and positioning research
- Cross-reference customer cultural and product style patterns

**OUTPUT FORMAT:**
Write professional AI brand ethos interview synthesis suitable for AI implementation.
Focus on actionable voice guidelines backed by comprehensive research.
Include confidence levels based on foundation data quality.
"""

        prompt = await self.prompt_manager.get_prompt(
            "interview_synthesis_analysis",
            default_prompt
        )
        
        return prompt.prompt if prompt else default_prompt

    async def _generate_interview_synthesis(
        self, 
        prompt_template: str, 
        synthesis_data: Dict[str, Any]
    ) -> str:
        """Generate interview synthesis using foundation research"""
        
        # Prepare template variables
        template_vars = {
            "brand_domain": synthesis_data["brand_domain"],
            "foundation": synthesis_data["foundation"][:2000] if synthesis_data["foundation"] else "No foundation research available",
            "market_positioning": synthesis_data["market_positioning"][:1500] if synthesis_data["market_positioning"] else "No market positioning available",
            "product_style": synthesis_data["product_style"][:1500] if synthesis_data["product_style"] else "No product style research available",
            "customer_cultural": synthesis_data["customer_cultural"][:1500] if synthesis_data["customer_cultural"] else "No customer cultural research available",
            "voice_messaging": synthesis_data["voice_messaging"][:1500] if synthesis_data["voice_messaging"] else "No voice messaging research available"
        }
        
        # Replace template variables
        final_prompt = prompt_template
        for var, value in template_vars.items():
            final_prompt = final_prompt.replace(f"{{{{{var}}}}}", str(value))
        
        # Prepare context for LLM
        context = f"""
BRAND: {synthesis_data['brand_domain']}

FOUNDATION RESEARCH SYNTHESIS:
Foundation Research: {synthesis_data['foundation'][:800] if synthesis_data['foundation'] else 'Not available'}

Market Positioning: {synthesis_data['market_positioning'][:600] if synthesis_data['market_positioning'] else 'Not available'}

Product Style: {synthesis_data['product_style'][:600] if synthesis_data['product_style'] else 'Not available'}

Customer Cultural: {synthesis_data['customer_cultural'][:600] if synthesis_data['customer_cultural'] else 'Not available'}

Voice & Messaging: {synthesis_data['voice_messaging'][:600] if synthesis_data['voice_messaging'] else 'Not available'}
"""
        
        response = await LLMFactory.chat_completion(
            task="interview_synthesis",
            system="You are an expert AI brand voice specialist creating authentic brand ethos interview synthesis. Generate comprehensive, implementation-ready AI voice guidelines based on complete foundation research.",
            messages=[
                {"role": "system", "content": final_prompt},
                {"role": "user", "content": context}
            ],
            temperature=0.7,
            # max_tokens=4000
        )
        
        return response.get("content", "Interview synthesis generation failed")

    def _calculate_interview_synthesis_quality_score(
        self, 
        foundation_context: Dict[str, Any], 
        synthesis_content: str
    ) -> float:
        """Calculate quality score for interview synthesis"""
        
        base_score = 0.7  # Base score for interview synthesis
        
        # Foundation context quality
        context_count = len(foundation_context)
        context_bonus = min(0.15, context_count * 0.03)  # Up to 0.15 for 5+ phases
        
        # Synthesis content quality
        content_length = len(synthesis_content)
        content_bonus = min(0.08, content_length / 12000)  # Up to 0.08 for comprehensive synthesis
        
        # Check for citations in content
        citation_count = synthesis_content.count("[") + synthesis_content.count("]")
        citation_bonus = min(0.05, citation_count * 0.003)  # Up to 0.05 for good citations
        
        # Check for AI implementation guidance
        implementation_indicators = synthesis_content.count("Do:") + synthesis_content.count("Don't:") + synthesis_content.count("Framework")
        implementation_bonus = min(0.05, implementation_indicators * 0.01)  # Bonus for actionable content
        
        # Check for voice synthesis patterns
        voice_indicators = synthesis_content.count("voice") + synthesis_content.count("interview") + synthesis_content.count("AI")
        voice_bonus = min(0.02, voice_indicators * 0.001)  # Bonus for voice-focused content
        
        final_score = base_score + context_bonus + content_bonus + citation_bonus + implementation_bonus + voice_bonus
        return min(0.87, final_score)  # Cap at 0.87 for interview synthesis

    async def _save_interview_synthesis_research(self, brand_domain: str, analysis_result: Dict[str, Any]) -> List[str]:
        """Save interview synthesis research in three-file format"""
        
        saved_files = []
        
        try:
            if hasattr(self.storage_manager, 'base_dir'):
                # Local storage
                research_dir = os.path.join(self.storage_manager.base_dir, "accounts", brand_domain, "research_phases")
                os.makedirs(research_dir, exist_ok=True)
                
                # Save content
                content_path = os.path.join(research_dir, "interview_synthesis_research.md")
                with open(content_path, "w") as f:
                    f.write(analysis_result["content"])
                saved_files.append(content_path)
                
                # Save metadata
                metadata = {
                    "phase": "interview_synthesis",
                    "confidence_score": analysis_result.get("confidence", 0.8),
                    "analysis_type": analysis_result.get("analysis_type", "enhanced_interview_synthesis"),
                    "foundation_phases": analysis_result.get("foundation_phases", []),
                    "source_count": analysis_result.get("source_count", 0),
                    "research_metadata": {
                        "phase": "interview_synthesis", 
                        "research_duration_seconds": time.time(),
                        "timestamp": datetime.now().isoformat() + "Z",
                        "quality_threshold": self.quality_threshold,
                        "cache_expires": (datetime.now() + timedelta(days=self.cache_duration_days)).isoformat() + "Z",
                        "version": "2.0_enhanced"
                    }
                }
                
                metadata_path = os.path.join(research_dir, "interview_synthesis_research_metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                saved_files.append(metadata_path)
                
                # Save sources/data
                sources_data = {
                    "analysis_sources": analysis_result.get("foundation_phases", []),
                    "foundation_research_phases": len(analysis_result.get("foundation_phases", [])),
                    "collection_timestamp": datetime.now().isoformat() + "Z"
                }
                
                sources_path = os.path.join(research_dir, "interview_synthesis_research_sources.json")
                with open(sources_path, "w") as f:
                    json.dump(sources_data, f, indent=2)
                saved_files.append(sources_path)
                
            else:
                # GCP storage
                research_dir = f"accounts/{brand_domain}/research_phases"
                
                # Save content
                content_blob = f"{research_dir}/interview_synthesis_research.md"
                blob = self.storage_manager.bucket.blob(content_blob)
                blob.upload_from_string(analysis_result["content"])
                saved_files.append(content_blob)
                
                # Save metadata  
                metadata = {
                    "phase": "interview_synthesis",
                    "confidence_score": analysis_result.get("confidence", 0.8),
                    "analysis_type": analysis_result.get("analysis_type", "enhanced_interview_synthesis"),
                    "foundation_phases": analysis_result.get("foundation_phases", []),
                    "source_count": analysis_result.get("source_count", 0),
                    "research_metadata": {
                        "phase": "interview_synthesis",
                        "research_duration_seconds": time.time(),
                        "timestamp": datetime.now().isoformat() + "Z",
                        "quality_threshold": self.quality_threshold,
                        "cache_expires": (datetime.now() + timedelta(days=self.cache_duration_days)).isoformat() + "Z",
                        "version": "2.0_enhanced"
                    }
                }
                
                metadata_blob = f"{research_dir}/interview_synthesis_research_metadata.json"
                blob = self.storage_manager.bucket.blob(metadata_blob)
                blob.upload_from_string(json.dumps(metadata, indent=2))
                saved_files.append(metadata_blob)
                
                # Save sources
                sources_data = {
                    "analysis_sources": analysis_result.get("foundation_phases", []),
                    "foundation_research_phases": len(analysis_result.get("foundation_phases", [])),
                    "collection_timestamp": datetime.now().isoformat() + "Z"
                }
                
                sources_blob = f"{research_dir}/interview_synthesis_research_sources.json"
                blob = self.storage_manager.bucket.blob(sources_blob)
                blob.upload_from_string(json.dumps(sources_data, indent=2))
                saved_files.append(sources_blob)
            
            logger.info(f"✅ Saved interview synthesis research for {brand_domain}")
            
        except Exception as e:
            logger.error(f"❌ Error saving interview synthesis research: {e}")
            raise
        
        return saved_files

    async def _load_cached_interview_synthesis(self, brand_domain: str) -> Optional[Dict[str, Any]]:
        """Load cached interview synthesis research"""
        try:
            if hasattr(self.storage_manager, 'base_dir'):
                # Local storage
                research_dir = os.path.join(self.storage_manager.base_dir, "accounts", brand_domain, "research_phases")
                
                content_path = os.path.join(research_dir, "interview_synthesis_research.md")
                metadata_path = os.path.join(research_dir, "interview_synthesis_research_metadata.json")
                
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
                            "interview_synthesis_content": content,
                            "quality_score": metadata.get("confidence_score", 0.8),
                            "files": [content_path, metadata_path],
                            "data_sources": metadata.get("source_count", 0),
                            "research_method": metadata.get("analysis_type", "cached_analysis")
                        }
            else:
                # GCP storage 
                research_dir = f"accounts/{brand_domain}/research_phases"
                
                content_blob = f"{research_dir}/interview_synthesis_research.md"
                metadata_blob = f"{research_dir}/interview_synthesis_research_metadata.json"
                
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
                            "interview_synthesis_content": content,
                            "quality_score": metadata.get("confidence_score", 0.8),
                            "files": [content_blob, metadata_blob],
                            "data_sources": metadata.get("source_count", 0),
                            "research_method": metadata.get("analysis_type", "cached_analysis")
                        }
        except Exception as e:
            logger.debug(f"Cache check failed: {e}")
        
        return None


def get_interview_synthesis_researcher(brand_domain: str) -> InterviewSynthesisResearcher:
    """Get interview synthesis researcher instance"""
    return InterviewSynthesisResearcher(brand_domain) 