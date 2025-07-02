"""
Product Catalog Research Phase

Implements Phase 8 of the Brand Research Pipeline per ROADMAP Section 4.2.

Focus: Synthesize all research phases into product catalog intelligence
Cache Duration: 1 month (needs regular refresh when research updates)
Research Time: 1-2 minutes
Quality Threshold: 8.5

This researcher synthesizes all previous research phases (foundation through linearity_analysis)
to create two specific outputs:
1. Product descriptor generation context - for use in UnifiedDescriptorGenerator
2. Product knowledge search context - for use in product search prompts

Position: Runs after all other research phases but before ResearchIntegrationProcessor
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
from liddy_intelligence.research.data_sources import WebSearchDataSource, DataGatheringContext

logger = logging.getLogger(__name__)


class ProductCatalogResearcher(BaseResearcher):
    """
    Product Catalog Research Phase Implementation
    
    Synthesizes all research phases to create targeted content for:
    1. Product descriptor generation (for UnifiedDescriptorGenerator)
    2. Product knowledge search (for search prompt enhancement)
    """
    
    def __init__(self, brand_domain: str, storage_manager=None):
        super().__init__(
            brand_domain=brand_domain, 
            researcher_name="product_catalog", 
            step_type=StepType.PRODUCT_CATALOG, 
            storage_manager=storage_manager,
            quality_threshold=8.5,
            cache_duration_days=30
        )
        
    async def research(self, force_refresh: bool = False, improvement_feedback: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate product catalog intelligence from all research phases
        
        Args:
            force_refresh: Force refresh of cached results
            improvement_feedback: Optional feedback from previous quality evaluation
        """
        # CRITICAL: Use quality wrapper when quality evaluation is enabled
        if self.enable_quality_evaluation:
            return await self._research_with_quality_wrapper(force_refresh, improvement_feedback)
        else:
            return await self._execute_core_research(force_refresh, improvement_feedback)
    
    async def _execute_core_research(self, force_refresh: bool = False, improvement_feedback: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Core product catalog research execution method
        """
        start_time = time.time()
        
        logger.info(f"📦 Starting Product Catalog Research for {self.brand_domain}")
        
        # Handle improvement feedback
        feedback_context = ""
        if improvement_feedback:
            logger.info(f"📋 Incorporating {len(improvement_feedback)} improvement suggestions")
            feedback_context = self._format_improvement_feedback(improvement_feedback)
        
        step_id = await self.progress_tracker.create_step(
            step_type=StepType.PRODUCT_CATALOG,
            brand=self.brand_domain,
            phase_name="Product Catalog Research",
            total_operations=8
        )
        
        try:
            await self.progress_tracker.start_step(step_id, "Checking cache and initializing...")
            
            if not force_refresh:
                cached_result = await self._load_cached_results()
                if cached_result:
                    await self.progress_tracker.complete_step(step_id, cache_hit=True)
                    return cached_result
            
            await self.progress_tracker.update_progress(step_id, 1, "📋 Loading all research phases...")
            
            # Load complete research foundation from all phases
            context = await self._gather_data()
            
            await self.progress_tracker.update_progress(step_id, 2, "📦 Synthesizing product catalog intelligence...")
            
            # Generate product catalog intelligence with feedback
            catalog_result = await self._generate_product_catalog_intelligence_with_feedback(
                context,
                feedback_context,
                step_id
            )
            
            await self.progress_tracker.update_progress(step_id, 6, "💾 Saving product catalog intelligence...")
            
            # Save product catalog results in 3-file format
            saved_files = await self._save_results(catalog_result)
            
            await self.progress_tracker.update_progress(step_id, 7, "✅ Finalizing product catalog research...")
            
            duration = time.time() - start_time
            logger.info(f"✅ Product Catalog Research completed in {duration:.1f}s")
            
            await self.progress_tracker.complete_step(
                step_id,
                output_files=saved_files,
                quality_score=catalog_result.get("confidence", 0.9),
                cache_hit=False
            )
            
            return {
                "brand_domain": self.brand_domain,
                "product_catalog_content": catalog_result.get("content", ""),
                "quality_score": catalog_result.get("confidence", 0.9),
                "files": saved_files,
                "data_sources": catalog_result.get("source_count", 0),
                "research_method": catalog_result.get("analysis_type", "product_catalog_synthesis"),
                "feedback_incorporated": len(improvement_feedback) if improvement_feedback else 0
            }
            
        except Exception as e:
            await self.progress_tracker.fail_step(step_id, str(e))
            logger.error(f"❌ Error in product catalog research: {e}")
            raise
    
    def _format_improvement_feedback(self, feedback: List[str]) -> str:
        """Format improvement feedback for inclusion in prompts"""
        if not feedback:
            return ""
        
        formatted = "\n\n## IMPROVEMENT FEEDBACK FROM PREVIOUS QUALITY EVALUATION:\n\n"
        formatted += "Please address the following areas for improvement in your product catalog synthesis:\n\n"
        
        for i, suggestion in enumerate(feedback, 1):
            formatted += f"{i}. {suggestion}\n"
        
        formatted += "\nPlease specifically address these points to improve the quality and effectiveness of product catalog intelligence.\n"
        return formatted
    
    async def _generate_product_catalog_intelligence_with_feedback(
        self,
        context: Dict[str, Any],
        feedback_context: str,
        step_id: str
    ) -> Dict[str, Any]:
        """
        Generate product catalog intelligence by synthesizing all research phases
        """
        
        await self.progress_tracker.update_progress(step_id, 3, "🎯 Synthesizing descriptor generation context...")
        
        await self.progress_tracker.update_progress(step_id, 4, "🔍 Creating product knowledge search context...")
        
        context_data = context.get('context')
        
        # Prepare synthesis data
        synthesis_data = {
            "brand_domain": self.brand_domain,
            "foundation": context_data.get("foundation", ""),
            "market_positioning": context_data.get("market_positioning", ""),
            "product_style": context_data.get("product_style", ""),
            "brand_style": context_data.get("brand_style", ""),
            "customer_cultural": context_data.get("customer_cultural", ""),
            "voice_messaging": context_data.get("voice_messaging", ""),
            "interview_synthesis": context_data.get("interview_synthesis", ""),
            "linearity_analysis": context_data.get("linearity_analysis", "")
        }
        
        # Compile search context with source IDs for citation
        search_context = ""
        source_citations = {}  # Map source_id to citation format
        
        for result in context["search_results"][:50]:  # Use top 50 results
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
            for result, citation in zip(context["search_results"][:50], source_citations.values())
        ])
        
        synthesis_data["search_context"] = search_context
        synthesis_data["source_reference_guide"] = source_reference_guide


        # Generate comprehensive product catalog intelligence using LLM with feedback
        catalog_content = await self._generate_product_catalog_synthesis_with_feedback(
            synthesis_data,
            feedback_context
        )
        
        await self.progress_tracker.update_progress(step_id, 5, "📊 Calculating synthesis quality...")
        
        # Calculate quality score based on research foundation completeness and feedback integration
        quality_score = self._calculate_product_catalog_quality_score(context_data, catalog_content)
        
        # Boost quality score if feedback was incorporated
        if feedback_context:
            quality_score = min(0.95, quality_score * 1.05)
        
        return {
            "content": catalog_content,
            "confidence": quality_score,
            "source_count": len(context_data),
            "analysis_type": "product_catalog_synthesis_with_feedback" if feedback_context else "product_catalog_synthesis",
            "research_phases": list(context_data.keys()),
            "feedback_incorporated": bool(feedback_context)
        }
    
    async def _generate_product_catalog_synthesis_with_feedback(
        self, 
        synthesis_data: Dict[str, Any],
        feedback_context: str
    ) -> str:
        """
        Generate product catalog intelligence synthesis with feedback
        """
        
        # Log feedback integration
        if feedback_context:
            logger.info("📋 Including improvement feedback in product catalog synthesis prompt")
        
        # Prepare template variables (limit content length for prompt efficiency)
        template_vars = {
            "brand_domain": synthesis_data["brand_domain"],
            "foundation": synthesis_data["foundation"][:4000] if synthesis_data["foundation"] else "No foundation research available",
            "market_positioning": synthesis_data["market_positioning"][:2000] if synthesis_data["market_positioning"] else "No market positioning available",
            "product_style": synthesis_data["product_style"][:2500] if synthesis_data["product_style"] else "No product style research available",
            "brand_style": synthesis_data["brand_style"][:2000] if synthesis_data["brand_style"] else "No brand style research available",
            "customer_cultural": synthesis_data["customer_cultural"][:2000] if synthesis_data["customer_cultural"] else "No customer cultural research available",
            "voice_messaging": synthesis_data["voice_messaging"][:2000] if synthesis_data["voice_messaging"] else "No voice messaging research available",
            "interview_synthesis": synthesis_data["interview_synthesis"][:1500] if synthesis_data["interview_synthesis"] else "No interview synthesis available",
            "linearity_analysis": synthesis_data["linearity_analysis"][:1500] if synthesis_data["linearity_analysis"] else "No linearity analysis available",
            "search_context": synthesis_data["search_context"],
            "source_reference_guide": synthesis_data["source_reference_guide"]
        }
        
        # Get prompt from Langfuse (or use fallback)
        try:
            prompt_template = await self.prompt_manager.get_prompt(
                prompt_name="internal/researcher/product_catalog_synthesis",
                prompt_type="chat",
                prompt=[
                    {"role": "system", "content": self._get_default_instruction_prompt()},
                    {"role": "user", "content": self._get_enhanced_user_prompt_with_feedback(feedback_context)}
                ]
            )
            prompts = prompt_template.prompt
            
            system_prompt = next((msg["content"] for msg in prompts if msg["role"] == "system"), None)
            user_prompt = next((msg["content"] for msg in prompts if msg["role"] == "user"), None)
            
        except Exception as e:
            logger.warning(f"Could not load prompts from Langfuse, using defaults: {e}")
            system_prompt = self._get_default_instruction_prompt()
            user_prompt = self._get_enhanced_user_prompt_with_feedback(feedback_context)
        
        # Replace template variables
        for var, value in template_vars.items():
            system_prompt = system_prompt.replace(f"{{{{{var}}}}}", str(value))
            user_prompt = user_prompt.replace(f"{{{{{var}}}}}", str(value))
        
        response = await LLMFactory.chat_completion(
            task="product_catalog_synthesis",
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.4,  # Slightly lower for consistency
        )
        
        return response.get("content", "Product catalog synthesis generation failed")
    
    def _get_enhanced_user_prompt_with_feedback(self, feedback_context: str) -> str:
        """Get enhanced user prompt that includes improvement feedback"""
        base_prompt = self._get_default_user_prompt()
        
        # Add feedback context if available
        if feedback_context:
            enhanced_prompt = base_prompt + feedback_context
            enhanced_prompt += "\n\nEnsure your synthesis specifically addresses the improvement feedback above to enhance the quality and effectiveness of product catalog intelligence."
            return enhanced_prompt
        
        return base_prompt

    async def _gather_data(self) -> Dict[str, Any]:
        """Load complete research foundation from all previous phases (excluding research_integration)"""
        
        context = {}
        
        try:
            # Load all research phases except research_integration (which comes after this phase)
            research_phases = [
                "foundation",
                "market_positioning", 
                "product_style",
                "brand_style",
                "customer_cultural",
                "voice_messaging",
                "interview_synthesis",
                "linearity_analysis"
            ]
            
            for phase_key in research_phases:
                try:
                    blob_path = f"research/{phase_key}/research.md"
                    context[phase_key] = await self.storage_manager.read_file(
                        account=self.brand_domain, 
                        file_path=blob_path
                    )
                                
                except Exception as e:
                    logger.debug(f"Could not load {phase_key} research: {e}")
            
            logger.info(f"📋 Loaded {len(context)} research phases for product catalog synthesis")
            
            # Perform brand discovery search
            web_search_source = WebSearchDataSource()
            if not web_search_source.is_available():
                # Fallback to domain-based name
                fallback_name = self.brand_domain.replace('.com', '').replace('.', ' ').title()
                logger.warning(f"⚠️ Web search unavailable, using fallback brand name: {fallback_name}")
                return fallback_name
            
            # Search for the brand name
            brand_queries = [
                f"{self.brand_domain} brand values",
                f"{self.brand_domain} mission statement",
                f"how {self.brand_domain} describes products",
                f"{self.brand_domain} voice guidelines",
                f"{self.brand_domain} customer service scripts",
                f"{self.brand_domain} FAQ phrasing",
                f"{self.brand_domain} seasonal collection {datetime.now().year}",
                # f"{datetime.now().strftime('%B')} gift guide <Product Category>",
                # f"{self.brand_domain} discount code",
                f"returns policy {self.brand_domain}",
                f"{self.brand_domain} ESG report",
                # f"compare {self.brand_domain} and <Competitor> quality"
            ]
            
            discovery_context = DataGatheringContext(
                brand_domain=self.brand_domain,
                researcher_name="product_catalog",
                phase_name="product_catalog"
            )
            
            logger.info(f"🔍 Researching product catalog for {self.brand_domain}...")
            discovery_result = await web_search_source.gather(brand_queries, discovery_context)
            
            if not discovery_result.results:
                # Fallback to domain-based name
                fallback_name = self.brand_domain.replace('.com', '').replace('.', ' ').title()
                logger.warning(f"⚠️ No discovery results, using fallback brand name: {fallback_name}")
                return fallback_name
            
            
            # Return in format expected by base class
            return {
                "brand_domain": self.brand_domain,
                "brand_name": self.brand_domain.replace('.com', '').replace('.', ' ').title(),
                "search_results": discovery_result.results if discovery_result and discovery_result.results else [],
                "detailed_sources": [],
                "context": context,
                "total_sources": len(discovery_result.sources) if discovery_result and discovery_result.sources else 0,
                "search_stats": {
                    "successful_searches": len(discovery_result.results) if discovery_result and discovery_result.results else 0,
                    "failed_searches": 0,
                    "success_rate": 1.0 if discovery_result.results else 0.0,
                    "ssl_errors": 0
                }
            }
            
        except Exception as e:
            logger.warning(f"Error loading research context: {e}")
            raise RuntimeError(f"Failed to load research context for product catalog synthesis: {e}")

    def _get_default_user_prompt(self) -> str:
        """Get product catalog synthesis prompt"""
        
        default_prompt = """
You are conducting Product Catalog Intelligence Synthesis for {{brand_domain}} to create targeted content for product descriptor generation and product knowledge search.

You have access to comprehensive brand research across 8 phases. Your task is to synthesize this research into two specific, actionable frameworks for product catalog optimization.

**SYNTHESIS REQUIREMENTS:**

Generate comprehensive product catalog intelligence covering these two critical outputs:

## Part A: Product Descriptor Generation Context

Create a comprehensive framework to guide AI product descriptor generation. This context will be used by the UnifiedDescriptorGenerator to create compelling, brand-aligned product descriptions.

### A1. Brand Voice & Tone for Products
- **Writing Style**: [How product descriptions should sound] 
- **Tone Characteristics**: [Specific tonal qualities from voice research]
- **Language Patterns**: [Preferred phrases, terminology, communication style]
- **Emotional Resonance**: [How products should make customers feel]
- **Brand Personality in Products**: [How brand personality translates to product descriptions]

### A2. Customer-Centric Messaging Framework
- **Primary Customer Segments**: [Key target audiences from customer research]
- **Customer Language & Terminology**: [How customers talk about products]
- **Pain Points & Solutions**: [Problems products solve for customers]
- **Value Propositions**: [Core value messages for product descriptions]
- **Purchase Decision Factors**: [What influences buying decisions]

### A3. Product Differentiation Strategy
- **Unique Selling Points**: [What makes products special]
- **Competitive Advantages**: [How to position vs competitors]
- **Quality & Craftsmanship Messaging**: [How to communicate quality]
- **Innovation & Technology**: [Technical differentiation language]
- **Style & Aesthetic Language**: [How to describe design and style]

### A4. Technical Communication Guidelines
- **Specification Language**: [How to present technical details]
- **Feature Benefits Translation**: [Converting features to customer benefits]
- **Use Case Scenarios**: [When and how products are used]
- **Performance Characteristics**: [How to communicate performance]
- **Material & Construction**: [How to describe quality and materials]

### A5. Call-to-Action & Conversion Framework
- **Purchase Motivation**: [What drives customers to buy]
- **Urgency & Scarcity**: [How to create appropriate urgency]
- **Trust & Credibility**: [How to build confidence in products]
- **Social Proof Integration**: [How to incorporate validation]
- **Next Steps Guidance**: [How to guide customer action]

## Part B: Product Knowledge Search Context

Create a comprehensive framework to enhance product search intelligence. This context will be used in product_knowledge_search prompts to improve search relevance and customer experience.

### B1. Search Intent Understanding
- **Customer Search Patterns**: [How customers look for products]
- **Intent Categories**: [Different types of product searches]
- **Query Interpretation**: [Understanding what customers really want]
- **Context Clues**: [Reading between the lines in searches]
- **Seasonal & Temporal Patterns**: [Time-based search considerations]

### B2. Product Categorization & Taxonomy
- **Logical Product Groups**: [How products naturally group together]
- **Feature-Based Categories**: [Organizing by capabilities/features]
- **Use Case Categories**: [Organizing by customer needs/applications]
- **Skill Level Segmentation**: [Beginner, intermediate, advanced product grouping]
- **Price Point Categories**: [Budget, mid-range, premium positioning]

### B3. Semantic Search Enhancement
- **Synonyms & Alternative Terms**: [Different ways customers describe same things]
- **Technical vs Common Language**: [Bridging expert and consumer terminology]
- **Related Concepts**: [What customers might also be interested in]
- **Contextual Associations**: [Products that go together or complement each other]
- **Problem-Solution Mapping**: [Connecting customer problems to product solutions]

### B4. Recommendation Intelligence
- **Cross-Sell Opportunities**: [What products work well together]
- **Upsell Pathways**: [How to suggest premium alternatives]
- **Customer Journey Mapping**: [Products for different stages of engagement]
- **Personalization Factors**: [What makes recommendations relevant]
- **Alternative & Substitute Products**: [When main choice isn't available]

### B5. Search Result Optimization
- **Relevance Ranking Factors**: [What makes one product more relevant than another]
- **Quality Signals**: [How to identify best-match products]
- **Diversity & Balance**: [Providing good variety in results]
- **Filtering & Refinement**: [How customers can narrow down choices]
- **Search Result Presentation**: [How to display products effectively]

**SYNTHESIS REQUIREMENTS:**
- Base all recommendations on the provided research foundation
- Use numbered citations [1], [2], [3] for all research-based insights
- Provide specific, actionable guidance suitable for immediate implementation
- Focus on practical application rather than theoretical frameworks
- Include confidence levels for different recommendations
- Address both technical accuracy and customer appeal

**OUTPUT FORMAT:**
Write professional product catalog intelligence suitable for AI system integration and strategic product marketing implementation.
Focus on synthesized insights that leverage the complete research foundation.
Include clear implementation guidance and practical examples.

BRAND: {{brand_domain}}

COMPLETE RESEARCH FOUNDATION FOR SYNTHESIS:

Foundation Research: {{foundation}}

Market Positioning: {{market_positioning}}

Product Style: {{product_style}}

Brand Style: {{brand_style}}

Customer Cultural: {{customer_cultural}}

Voice & Messaging: {{voice_messaging}}

Interview Synthesis: {{interview_synthesis}}

Linearity Analysis: {{linearity_analysis}}

Search Context: {{search_context}}

Source Reference Guide: {{source_reference_guide}}

"""

        return default_prompt

    def _get_default_instruction_prompt(self) -> str:
        """Get instruction prompt for product catalog synthesis"""
        
        default_prompt = """
You are an expert product marketing strategist specializing in AI-driven product catalog optimization. You synthesize comprehensive brand research to create actionable frameworks for product descriptor generation and intelligent product search systems.

Your expertise includes:
- Product messaging and positioning strategy
- Customer-centric product communication
- AI system integration for product catalogs
- Search optimization and product discovery
- Brand consistency in product presentations

Generate practical, implementation-ready product catalog intelligence that bridges brand strategy with technical product systems.
"""
        
        return default_prompt

    def _calculate_product_catalog_quality_score(
        self, 
        context_data: Dict[str, Any], 
        catalog_content: str
    ) -> float:
        """Calculate quality score for product catalog synthesis"""
        
        base_score = 0.78  # Base score for product catalog synthesis
        
        # Research foundation completeness (8 phases expected)
        foundation_count = len(context_data)
        foundation_bonus = min(0.12, foundation_count * 0.015)  # Up to 0.12 for 8 phases
        
        # Content comprehensiveness
        content_length = len(catalog_content)
        content_bonus = min(0.06, content_length / 12000)  # Up to 0.06 for comprehensive content
        
        # Check for structured sections (Part A and Part B)
        section_indicators = catalog_content.count("Part A") + catalog_content.count("Part B")
        section_bonus = min(0.04, section_indicators * 0.02)  # Bonus for proper structure
        
        # Check for actionable recommendations
        action_indicators = (catalog_content.count("Framework") + 
                           catalog_content.count("Guidelines") + 
                           catalog_content.count("Strategy"))
        action_bonus = min(0.03, action_indicators * 0.002)  # Bonus for actionable content
        
        # Check for citations and research integration
        citation_count = catalog_content.count("[") + catalog_content.count("]")
        citation_bonus = min(0.04, citation_count * 0.002)  # Up to 0.04 for good citations
        
        # Check for product-specific focus
        product_indicators = (catalog_content.count("product") + 
                            catalog_content.count("catalog") + 
                            catalog_content.count("descriptor"))
        product_bonus = min(0.03, product_indicators * 0.0005)  # Bonus for product focus
        
        final_score = (base_score + foundation_bonus + content_bonus + 
                      section_bonus + action_bonus + citation_bonus + product_bonus)
        return min(0.94, final_score)  # Cap at 0.94 for product catalog synthesis


def get_product_catalog_researcher(brand_domain: str) -> ProductCatalogResearcher:
    """
    Factory function to create ProductCatalogResearcher instance
    
    Args:
        brand_domain: The brand domain to research
        
    Returns:
        ProductCatalogResearcher instance
    """
    storage_manager = get_account_storage_provider()
    return ProductCatalogResearcher(brand_domain=brand_domain, storage_manager=storage_manager) 