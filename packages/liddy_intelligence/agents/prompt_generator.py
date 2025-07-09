"""
Enhanced Prompt Generator for Multi-Agent System

Takes multi-agent insights and generates optimized prompts for the Conversation Engine.
Stores prompts in Langfuse using PromptManager for consumption by external systems.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from liddy.prompt_manager import PromptManager
from liddy_intelligence.agents.context import ConversationContext, EnhancedContext, ConversationStage, CustomerIntent

logger = logging.getLogger(__name__)


class EnhancedPromptGenerator:
    """
    Generates intelligent, context-aware prompts based on multi-agent insights.
    
    This is the output layer of our multi-agent system - it takes all the agent
    insights and creates optimized prompts for the Conversation Engine to use.
    """
    
    def __init__(self):
        self.prompt_manager = PromptManager()
        
        # Base prompt templates for different conversation stages
        self.base_templates = {
            ConversationStage.AWARENESS: "awareness_base_prompt",
            ConversationStage.INTEREST: "interest_base_prompt", 
            ConversationStage.CONSIDERATION: "consideration_base_prompt",
            ConversationStage.DECISION: "decision_base_prompt",
            ConversationStage.POST_PURCHASE: "post_purchase_base_prompt"
        }
        
        logger.info("🎯 Initialized Enhanced Prompt Generator with Langfuse integration")
    
    async def generate_enhanced_prompt(
        self, 
        enhanced_context: EnhancedContext,
        message: str,
        brand_intelligence: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate an enhanced prompt based on multi-agent insights.
        
        Returns:
            Dict containing the enhanced prompt and metadata for Langfuse storage
        """
        
        context = enhanced_context.conversation_context
        
        # Determine prompt key based on conversation state
        prompt_key = self._generate_prompt_key(context)
        
        # Build enhanced prompt variables
        prompt_variables = await self._build_prompt_variables(
            enhanced_context, message, brand_intelligence
        )
        
        # Get base prompt template from Langfuse
        base_prompt = await self._get_base_prompt_template(prompt_key, context.brand_domain)
        
        # Generate the enhanced prompt
        enhanced_prompt = await self._create_enhanced_prompt(
            base_prompt, prompt_variables, enhanced_context
        )
        
        # Store enhanced prompt in Langfuse
        stored_prompt_key = await self._store_enhanced_prompt(
            enhanced_prompt, prompt_variables, context
        )
        
        return {
            "prompt_key": stored_prompt_key,
            "enhanced_prompt": enhanced_prompt,
            "prompt_variables": prompt_variables,
            "confidence_score": enhanced_context.confidence_score,
            "agent_insights_used": enhanced_context.insights_used,
            "conversation_context": {
                "conversation_id": context.conversation_id,
                "brand_domain": context.brand_domain,
                "conversation_stage": context.conversation_stage.value,
                "customer_intent": context.current_intent.value if context.current_intent else None,
                "livekit_room_id": context.livekit_room_id
            }
        }
    
    def _generate_prompt_key(self, context: ConversationContext) -> str:
        """Generate unique prompt key based on conversation state"""
        
        stage = context.conversation_stage.value
        intent = context.current_intent.value if context.current_intent else "general"
        brand = context.brand_domain.replace(".", "_")
        
        return f"liddy/conversation/{brand}/enhanced_prompt_{stage}_{intent}"
    
    async def _build_prompt_variables(
        self, 
        enhanced_context: EnhancedContext,
        message: str,
        brand_intelligence: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build comprehensive prompt variables from all agent insights"""
        
        context = enhanced_context.conversation_context
        
        # Base conversation variables
        variables = {
            "customer_message": message,
            "conversation_history": self._format_conversation_history(context),
            "brand_context": self._extract_brand_context(brand_intelligence, context),
            "livekit_context": {
                "room_id": context.livekit_room_id,
                "audio_enabled": context.audio_enabled,
                "video_enabled": context.video_enabled
            }
        }
        
        # Add psychology insights
        if enhanced_context.psychology_insights:
            variables["customer_psychology"] = self._extract_psychology_variables(
                enhanced_context.psychology_insights
            )
        
        # Add product intelligence
        if enhanced_context.product_insights:
            variables["product_intelligence"] = self._extract_product_variables(
                enhanced_context.product_insights
            )
        
        # Add sales strategy
        if enhanced_context.sales_insights:
            variables["sales_strategy"] = self._extract_sales_variables(
                enhanced_context.sales_insights
            )
        
        # Add brand authenticity
        if enhanced_context.brand_insights:
            variables["brand_authenticity"] = self._extract_brand_variables(
                enhanced_context.brand_insights
            )
        
        # Add conversation optimization
        if enhanced_context.conversation_insights:
            variables["conversation_optimization"] = self._extract_conversation_variables(
                enhanced_context.conversation_insights
            )
        
        # Add real-time recommendations
        variables["real_time_optimization"] = {
            "recommended_response_length": enhanced_context.recommended_response_length,
            "suggested_tone": enhanced_context.suggested_tone,
            "priority_topics": enhanced_context.priority_topics,
            "immediate_opportunities": enhanced_context.immediate_opportunities
        }
        
        return variables
    
    def _format_conversation_history(self, context: ConversationContext) -> List[Dict[str, str]]:
        """Format recent conversation history for prompt context"""
        
        recent_messages = context.get_recent_messages(10)
        formatted_history = []
        
        for msg in recent_messages:
            formatted_history.append({
                "sender": msg["sender"],
                "content": msg["content"],
                "timestamp": msg["timestamp"]
            })
        
        return formatted_history
    
    def _extract_brand_context(
        self, 
        brand_intelligence: Optional[Dict[str, Any]], 
        context: ConversationContext
    ) -> Dict[str, Any]:
        """Extract brand context for prompt variables"""
        
        if not brand_intelligence:
            return {
                "brand_name": context.brand_domain.replace(".com", "").title(),
                "brand_domain": context.brand_domain
            }
        
        return {
            "brand_name": brand_intelligence.get("brand_name", ""),
            "brand_domain": context.brand_domain,
            "brand_voice": brand_intelligence.get("voice_messaging", {}).get("tone", "professional"),
            "core_values": brand_intelligence.get("foundation", {}).get("core_values", []),
            "brand_personality": brand_intelligence.get("voice_messaging", {}).get("personality", []),
            "target_audience": brand_intelligence.get("customer_cultural", {}).get("target_segments", [])
        }
    
    def _extract_psychology_variables(self, psychology_insight) -> Dict[str, Any]:
        """Extract psychology insights for prompt variables"""
        
        insights = psychology_insight.insights
        
        return {
            "emotional_state": insights.get("emotional_state", "neutral"),
            "confidence_level": psychology_insight.confidence_score,
            "communication_style": insights.get("communication_style", "conversational"),
            "decision_making_style": insights.get("decision_making_style", "balanced"),
            "price_sensitivity": insights.get("price_sensitivity", "medium"),
            "urgency_level": insights.get("urgency_level", "medium"),
            "adaptation_recommendations": psychology_insight.recommendations[:3],
            "personality_indicators": insights.get("personality_traits", [])
        }
    
    def _extract_product_variables(self, product_insight) -> Dict[str, Any]:
        """Extract product intelligence for prompt variables"""
        
        insights = product_insight.insights
        
        return {
            "priority_products": insights.get("priority_products", []),
            "product_recommendations": insights.get("immediate_product_suggestions", []),
            "upsell_opportunities": insights.get("upsell_opportunities", []),
            "cross_sell_products": insights.get("complementary_products", []),
            "competitive_advantages": insights.get("competitive_advantages", []),
            "technical_highlights": insights.get("technical_details_to_highlight", []),
            "confidence_level": product_insight.confidence_score
        }
    
    def _extract_sales_variables(self, sales_insight) -> Dict[str, Any]:
        """Extract sales strategy for prompt variables"""
        
        insights = sales_insight.insights
        
        return {
            "sales_approach": insights.get("sales_approach", "consultative"),
            "buying_signals": insights.get("buying_signals", []),
            "objection_signals": insights.get("objection_signals", []),
            "closing_opportunities": insights.get("closing_opportunities", []),
            "urgency_tactics": insights.get("urgency_tactics", []),
            "social_proof_opportunities": insights.get("social_proof_opportunities", []),
            "value_propositions": insights.get("value_props_to_emphasize", []),
            "objection_handling": insights.get("preemptive_objection_handling", []),
            "confidence_level": sales_insight.confidence_score
        }
    
    def _extract_brand_variables(self, brand_insight) -> Dict[str, Any]:
        """Extract brand authenticity for prompt variables"""
        
        insights = brand_insight.insights
        
        return {
            "voice_consistency_score": insights.get("voice_authenticity_score", 1.0),
            "tone_adjustments": insights.get("tone_adjustments", []),
            "brand_story_opportunities": insights.get("brand_story_opportunities", []),
            "value_alignment_moments": insights.get("value_alignment_moments", []),
            "emotional_connection_opportunities": insights.get("emotional_connection_opportunities", []),
            "authenticity_requirements": insights.get("immediate_brand_enhancements", []),
            "confidence_level": brand_insight.confidence_score
        }
    
    def _extract_conversation_variables(self, conversation_insight) -> Dict[str, Any]:
        """Extract conversation optimization for prompt variables"""
        
        insights = conversation_insight.insights
        
        return {
            "conversation_stage": insights.get("conversation_stage", "unknown"),
            "optimal_response_length": insights.get("optimal_response_length", "medium"),
            "technical_depth": insights.get("technical_depth_preference", "intermediate"),
            "engagement_level": insights.get("engagement_level", "medium"),
            "pacing_recommendations": insights.get("pacing_optimization", []),
            "question_opportunities": insights.get("question_asking_opportunities", []),
            "conversation_pivots": insights.get("conversation_pivots", []),
            "confidence_level": conversation_insight.confidence_score
        }
    
    async def _get_base_prompt_template(self, prompt_key: str, brand_domain: str) -> str:
        """Get base prompt template from Langfuse"""
        
        try:
            # Try to get specific prompt for this brand and conversation state
            prompt_data = await self.prompt_manager.get_prompt(prompt_key)
            
            if prompt_data and prompt_data.get("content"):
                return prompt_data["content"]
            
            # Fallback to generic conversation prompt
            fallback_key = f"liddy/conversation/generic/enhanced_prompt_base"
            fallback_prompt = await self.prompt_manager.get_prompt(fallback_key)
            
            if fallback_prompt and fallback_prompt.get("content"):
                return fallback_prompt["content"]
            
            # Ultimate fallback
            return self._get_default_prompt_template()
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to get prompt template: {e}")
            return self._get_default_prompt_template()
    
    def _get_default_prompt_template(self) -> str:
        """Default prompt template when Langfuse is unavailable"""
        
        return """You are an expert sales agent for {brand_name}, helping customers discover and purchase the perfect products for their needs.

BRAND CONTEXT:
- Brand: {brand_name}
- Your role: Expert sales consultant and product specialist
- Brand voice: {brand_voice}
- Core values: {core_values}

CUSTOMER ANALYSIS:
- Emotional state: {emotional_state}
- Communication preference: {communication_style}
- Decision-making style: {decision_making_style}
- Price sensitivity: {price_sensitivity}
- Urgency level: {urgency_level}

PRODUCT FOCUS:
- Priority products for this customer: {priority_products}
- Upsell opportunities: {upsell_opportunities}
- Competitive advantages to highlight: {competitive_advantages}

SALES STRATEGY:
- Approach: {sales_approach}
- Buying signals detected: {buying_signals}
- Potential objections: {objection_signals}
- Immediate opportunities: {immediate_opportunities}

RESPONSE GUIDELINES:
- Length: {recommended_response_length}
- Tone: {suggested_tone}
- Technical depth: {technical_depth}
- Priority topics: {priority_topics}

CUSTOMER MESSAGE: "{customer_message}"

Respond as the expert {brand_name} sales agent, incorporating all the above context to provide the most helpful, authentic, and sales-effective response."""
    
    async def _create_enhanced_prompt(
        self, 
        base_prompt: str, 
        variables: Dict[str, Any],
        enhanced_context: EnhancedContext
    ) -> str:
        """Create the final enhanced prompt with all variables injected"""
        
        try:
            # Format the prompt with all variables
            enhanced_prompt = base_prompt.format(**self._flatten_variables(variables))
            
            # Add multi-agent metadata
            metadata_section = f"""

MULTI-AGENT ANALYSIS METADATA:
- Total processing time: {enhanced_context.total_processing_time_ms:.1f}ms
- Confidence score: {enhanced_context.confidence_score:.2f}
- Agents used: {', '.join(enhanced_context.insights_used)}
- Fallback insights: {enhanced_context.fallback_insights}
- Generated: {datetime.now().isoformat()}
"""
            
            return enhanced_prompt + metadata_section
            
        except KeyError as e:
            logger.warning(f"⚠️ Prompt template variable missing: {e}")
            # Return prompt with available variables only
            return self._safe_format_prompt(base_prompt, variables)
    
    def _flatten_variables(self, variables: Dict[str, Any]) -> Dict[str, str]:
        """Flatten nested variables for string formatting"""
        
        flattened = {}
        
        for key, value in variables.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flattened[f"{key}_{subkey}"] = self._format_value(subvalue)
                    flattened[subkey] = self._format_value(subvalue)  # Also allow direct access
            else:
                flattened[key] = self._format_value(value)
        
        return flattened
    
    def _format_value(self, value: Any) -> str:
        """Format a value for prompt inclusion"""
        
        if isinstance(value, list):
            if not value:
                return "None specified"
            return ", ".join(str(item) for item in value)
        elif isinstance(value, dict):
            return json.dumps(value, indent=2)
        elif value is None:
            return "Not available"
        else:
            return str(value)
    
    def _safe_format_prompt(self, template: str, variables: Dict[str, Any]) -> str:
        """Safely format prompt, replacing missing variables with placeholders"""
        
        flattened = self._flatten_variables(variables)
        
        # Replace any remaining unfilled variables with safe defaults
        import re
        def replace_unfilled(match):
            var_name = match.group(1)
            return flattened.get(var_name, f"[{var_name}: not available]")
        
        safe_prompt = re.sub(r'\{([^}]+)\}', replace_unfilled, template)
        return safe_prompt
    
    async def _store_enhanced_prompt(
        self, 
        enhanced_prompt: str,
        variables: Dict[str, Any],
        context: ConversationContext
    ) -> str:
        """Store the enhanced prompt in Langfuse for Conversation Engine consumption"""
        
        # Generate unique prompt key for this enhanced prompt
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_key = f"liddy/conversation/{context.brand_domain}/enhanced_{context.conversation_id}_{timestamp}"
        
        try:
            # Store in Langfuse using PromptManager
            await self.prompt_manager.store_prompt(
                prompt_key=prompt_key,
                content=enhanced_prompt,
                variables=variables,
                metadata={
                    "conversation_id": context.conversation_id,
                    "brand_domain": context.brand_domain,
                    "conversation_stage": context.conversation_stage.value,
                    "customer_intent": context.current_intent.value if context.current_intent else None,
                    "generated_at": datetime.now().isoformat(),
                    "agent_type": "multi_agent_enhanced",
                    "livekit_room_id": context.livekit_room_id
                }
            )
            
            logger.info(f"✅ Stored enhanced prompt in Langfuse: {prompt_key}")
            return prompt_key
            
        except Exception as e:
            logger.error(f"❌ Failed to store enhanced prompt: {e}")
            return f"fallback_{context.conversation_id}_{timestamp}"


# Factory function for easy creation
def create_prompt_generator() -> EnhancedPromptGenerator:
    """Create and return an Enhanced Prompt Generator"""
    return EnhancedPromptGenerator()