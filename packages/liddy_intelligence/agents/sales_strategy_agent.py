"""
Sales Strategy Agent

Specialized agent that analyzes buying signals, identifies objections, and generates
real-time sales strategy recommendations for maximizing conversion and customer satisfaction.
"""

import logging
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from liddy_intelligence.agents.base_agent import BaseAgent, AgentInsight
from liddy_intelligence.agents.context import ConversationContext, CustomerIntent, ConversationStage
from liddy_intelligence.llm.simple_factory import LLMFactory

logger = logging.getLogger(__name__)


class SalesStrategyAgent(BaseAgent):
    """
    Real-time sales strategy and objection handling agent.
    
    Analyzes:
    - Buying signals and purchase readiness indicators
    - Objection signals and customer concerns
    - Optimal sales approach and tactics
    - Closing opportunities and timing
    - Value proposition emphasis
    - Social proof opportunities
    """
    
    def __init__(self):
        super().__init__("sales_strategy_agent", max_processing_time_ms=500)
        self.llm_factory = LLMFactory()
        
        # Sales signal detection patterns
        self.buying_signals = {
            "strong": [
                "ready to buy", "want to order", "need this", "let's do it", 
                "sounds perfect", "exactly what I need", "when can I get",
                "how do I purchase", "take my money"
            ],
            "medium": [
                "interested", "looks good", "seems right", "might work", 
                "considering", "thinking about", "leaning towards",
                "probably", "likely"
            ],
            "weak": [
                "maybe", "perhaps", "possibly", "might be", "could be",
                "not sure yet", "need to think", "let me consider"
            ]
        }
        
        self.objection_signals = {
            "price": [
                "expensive", "costly", "pricey", "budget", "cheaper", "cost",
                "afford", "money", "price", "too much", "overpriced"
            ],
            "timing": [
                "not now", "later", "next month", "next year", "waiting",
                "not ready", "too soon", "maybe later", "future"
            ],
            "fit": [
                "not sure it fits", "wrong size", "doesn't match", "not right",
                "might not work", "not suitable", "doesn't fit"
            ],
            "competition": [
                "comparing", "other options", "competitors", "alternatives", 
                "shopping around", "looking at others", "versus"
            ],
            "features": [
                "missing", "doesn't have", "lacks", "wish it had", "need more",
                "not enough", "insufficient", "limited"
            ],
            "trust": [
                "not sure about", "don't know", "uncertain", "skeptical",
                "doubtful", "worried", "concerned", "hesitant"
            ]
        }
        
        self.urgency_indicators = {
            "immediate": [
                "asap", "right now", "immediately", "urgent", "today",
                "this week", "quickly", "soon as possible"
            ],
            "seasonal": [
                "for summer", "for winter", "before spring", "holiday",
                "birthday", "anniversary", "vacation", "trip"
            ],
            "event_driven": [
                "race", "competition", "event", "deadline", "before",
                "training", "preparation", "upcoming"
            ]
        }
        
        self.value_indicators = {
            "quality_focused": [
                "quality", "best", "premium", "top", "excellence", 
                "superior", "high-end", "professional"
            ],
            "performance_focused": [
                "performance", "speed", "efficiency", "power", "capability",
                "results", "output", "metrics"
            ],
            "durability_focused": [
                "lasting", "durable", "reliable", "long-term", "investment",
                "build quality", "warranty", "lifespan"
            ],
            "convenience_focused": [
                "easy", "simple", "convenient", "hassle-free", "automatic",
                "user-friendly", "straightforward"
            ]
        }
        
        logger.info("💼 Initialized Sales Strategy Agent")
    
    async def analyze_real_time(self, message: str, context: ConversationContext) -> AgentInsight:
        """Analyze customer message for sales strategy and objection handling"""
        
        try:
            # Quick pattern-based sales analysis
            quick_analysis = self._quick_sales_analysis(message, context)
            
            # Deep LLM-based sales strategy
            deep_analysis = await self._deep_sales_analysis(message, context)
            
            # Generate sales recommendations
            sales_strategy = self._generate_sales_strategy(quick_analysis, deep_analysis, context)
            
            # Combine all insights
            combined_insights = {
                **quick_analysis,
                **deep_analysis,
                **sales_strategy
            }
            
            # Generate specific sales recommendations
            recommendations = self._generate_sales_recommendations(combined_insights, context)
            
            return AgentInsight(
                agent_name=self.agent_name,
                confidence_score=self._calculate_confidence(combined_insights, context),
                timestamp=datetime.now(),
                insights=combined_insights,
                recommendations=recommendations,
                metadata={
                    "analysis_type": "sales_strategy",
                    "buying_signals_detected": len(combined_insights.get("buying_signals", [])),
                    "objections_detected": len(combined_insights.get("objection_signals", [])),
                    "sales_opportunities": len(combined_insights.get("closing_opportunities", []))
                },
                processing_time_ms=0.0  # Will be set by base class
            )
            
        except Exception as e:
            logger.error(f"❌ Sales strategy analysis failed: {e}")
            return self._create_fallback_sales_insight(message)
    
    def _quick_sales_analysis(self, message: str, context: ConversationContext) -> Dict[str, Any]:
        """Fast pattern-based sales signal analysis"""
        
        message_lower = message.lower()
        
        # Buying signal detection
        buying_strength = "none"
        detected_signals = []
        
        for strength, signals in self.buying_signals.items():
            matches = [signal for signal in signals if signal in message_lower]
            if matches:
                buying_strength = strength
                detected_signals.extend(matches)
                break  # Take strongest signal level
        
        # Objection detection
        detected_objections = {}
        for objection_type, indicators in self.objection_signals.items():
            matches = [indicator for indicator in indicators if indicator in message_lower]
            if matches:
                detected_objections[objection_type] = matches
        
        # Urgency detection
        urgency_type = "none"
        urgency_indicators = []
        for urgency, indicators in self.urgency_indicators.items():
            matches = [indicator for indicator in indicators if indicator in message_lower]
            if matches:
                urgency_type = urgency
                urgency_indicators.extend(matches)
                break
        
        # Value focus detection
        value_focus = []
        for focus_type, indicators in self.value_indicators.items():
            matches = [indicator for indicator in indicators if indicator in message_lower]
            if matches:
                value_focus.append(focus_type)
        
        # Conversation stage analysis
        conversation_progress = self._analyze_conversation_progress(context)
        
        return {
            "buying_signal_strength": buying_strength,
            "buying_signals": detected_signals,
            "objection_signals": detected_objections,
            "urgency_type": urgency_type,
            "urgency_indicators": urgency_indicators,
            "value_focus": value_focus,
            "conversation_progress": conversation_progress,
            "analysis_method": "pattern_based"
        }
    
    def _analyze_conversation_progress(self, context: ConversationContext) -> Dict[str, Any]:
        """Analyze how far along the customer is in the buying journey"""
        
        message_count = len(context.message_history)
        recent_topics = context.expressed_interests[-5:] if context.expressed_interests else []
        
        # Determine conversation maturity
        if message_count <= 2:
            maturity = "early"
        elif message_count <= 5:
            maturity = "developing"
        else:
            maturity = "advanced"
        
        # Check for progression indicators
        progression_indicators = {
            "asked_about_price": any("price" in str(topic).lower() for topic in recent_topics),
            "requested_details": any("detail" in str(topic).lower() for topic in recent_topics),
            "compared_options": any("compar" in str(topic).lower() for topic in recent_topics),
            "showed_urgency": context.conversation_stage == ConversationStage.DECISION
        }
        
        return {
            "conversation_maturity": maturity,
            "message_count": message_count,
            "progression_indicators": progression_indicators,
            "recent_interests": recent_topics
        }
    
    async def _deep_sales_analysis(self, message: str, context: ConversationContext) -> Dict[str, Any]:
        """Deep LLM-based sales strategy analysis"""
        
        try:
            # Get recent conversation context
            recent_messages = context.get_recent_messages(7)
            conversation_history = "\n".join([
                f"{msg['sender']}: {msg['content']}" 
                for msg in recent_messages
            ])
            
            sales_analysis_prompt = f"""Analyze this customer interaction for sales strategy optimization.

CUSTOMER MESSAGE: "{message}"

CONVERSATION HISTORY:
{conversation_history}

CONVERSATION CONTEXT:
- Stage: {context.conversation_stage.value}
- Intent: {context.current_intent.value if context.current_intent else 'unknown'}
- Interests: {context.expressed_interests}

Analyze and provide insights on:

1. BUYING READINESS:
   - How ready is this customer to make a purchase? (1-10 scale)
   - What specific buying signals are present?
   - What's holding them back from purchasing?

2. OBJECTION ANALYSIS:
   - What concerns or objections does the customer have?
   - Which objections are explicit vs. implicit?
   - How serious/blocking are these objections?

3. OPTIMAL SALES APPROACH:
   - What sales approach would work best? (consultative, direct, educational, etc.)
   - Should we push for a close or continue nurturing?
   - What's the ideal next step in the sales process?

4. VALUE PROPOSITION FOCUS:
   - What value propositions matter most to this customer?
   - Which benefits should we emphasize?
   - What proof points would be most convincing?

5. COMPETITIVE POSITIONING:
   - Are there competitive threats to address?
   - What unique advantages should we highlight?
   - How do we differentiate from alternatives?

6. CLOSING STRATEGY:
   - Are there immediate closing opportunities?
   - What urgency tactics might be appropriate?
   - What would motivate them to buy today vs. later?

Respond in JSON format with specific recommendations and confidence scores."""

            llm = self.llm_factory.get_service("openai/o3")
            response = await llm.chat_completion(
                messages=[{"role": "user", "content": sales_analysis_prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            
            if response and response.get("content"):
                # Try to parse JSON response
                import json
                try:
                    analysis = json.loads(response["content"])
                    analysis["analysis_method"] = "llm_deep"
                    return analysis
                except json.JSONDecodeError:
                    # Fallback to text parsing
                    return self._parse_sales_text_response(response["content"])
            
            return {"analysis_method": "llm_failed"}
            
        except Exception as e:
            logger.warning(f"⚠️ Deep sales analysis failed: {e}")
            return {"analysis_method": "llm_error", "error": str(e)}
    
    def _parse_sales_text_response(self, text_response: str) -> Dict[str, Any]:
        """Parse text-based sales analysis response as fallback"""
        
        analysis = {"analysis_method": "llm_text_parsed"}
        text_lower = text_response.lower()
        
        # Extract key insights
        if "ready to buy" in text_lower or "high readiness" in text_lower:
            analysis["buying_readiness"] = "high"
        elif "considering" in text_lower or "medium readiness" in text_lower:
            analysis["buying_readiness"] = "medium"
        else:
            analysis["buying_readiness"] = "low"
        
        # Extract sales approach
        if "consultative" in text_lower:
            analysis["recommended_approach"] = "consultative"
        elif "direct" in text_lower:
            analysis["recommended_approach"] = "direct"
        elif "educational" in text_lower:
            analysis["recommended_approach"] = "educational"
        else:
            analysis["recommended_approach"] = "adaptive"
        
        # Extract closing opportunities
        if "close" in text_lower and "opportunity" in text_lower:
            analysis["closing_opportunity"] = True
        else:
            analysis["closing_opportunity"] = False
        
        return analysis
    
    def _generate_sales_strategy(
        self, 
        quick_analysis: Dict[str, Any], 
        deep_analysis: Dict[str, Any], 
        context: ConversationContext
    ) -> Dict[str, Any]:
        """Generate comprehensive sales strategy based on all analysis"""
        
        # Determine primary sales approach
        buying_strength = quick_analysis.get("buying_signal_strength", "none")
        objections = quick_analysis.get("objection_signals", {})
        urgency = quick_analysis.get("urgency_type", "none")
        
        # Calculate sales approach
        if buying_strength == "strong" and len(objections) == 0:
            approach = "direct_close"
        elif buying_strength in ["strong", "medium"] and len(objections) <= 1:
            approach = "consultative_close"
        elif len(objections) > 2:
            approach = "objection_handling"
        elif urgency != "none":
            approach = "urgency_based"
        else:
            approach = "educational_nurture"
        
        # Generate closing opportunities
        closing_opportunities = []
        if buying_strength == "strong":
            closing_opportunities.append("immediate_close_opportunity")
        if urgency != "none":
            closing_opportunities.append("urgency_close")
        if len(objections) == 0 and context.conversation_stage == ConversationStage.DECISION:
            closing_opportunities.append("decision_stage_close")
        
        # Generate urgency tactics
        urgency_tactics = []
        if urgency == "immediate":
            urgency_tactics.extend(["same_day_delivery", "immediate_availability"])
        elif urgency == "seasonal":
            urgency_tactics.extend(["seasonal_preparation", "advance_planning"])
        elif urgency == "event_driven":
            urgency_tactics.extend(["event_preparation", "deadline_support"])
        
        # Social proof opportunities
        social_proof = []
        value_focus = quick_analysis.get("value_focus", [])
        if "quality_focused" in value_focus:
            social_proof.extend(["expert_endorsements", "awards", "testimonials"])
        if "performance_focused" in value_focus:
            social_proof.extend(["performance_metrics", "competition_wins", "athlete_endorsements"])
        
        return {
            "sales_approach": approach,
            "closing_opportunities": closing_opportunities,
            "urgency_tactics": urgency_tactics,
            "social_proof_opportunities": social_proof,
            "objection_priority": self._prioritize_objections(objections),
            "value_props_to_emphasize": self._select_value_props(value_focus, context),
            "next_sales_action": self._determine_next_action(approach, context)
        }
    
    def _prioritize_objections(self, objections: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Prioritize objections by impact and urgency"""
        
        # Objection severity scoring
        severity_scores = {
            "price": 0.9,  # High impact, common
            "timing": 0.7,  # Medium impact, often surmountable
            "fit": 0.8,    # High impact if true concern
            "competition": 0.6,  # Medium impact, opportunity to differentiate
            "features": 0.5,  # Lower impact, often addressable
            "trust": 0.8   # High impact, but solvable with proof
        }
        
        prioritized = []
        for objection_type, mentions in objections.items():
            severity = severity_scores.get(objection_type, 0.5)
            mention_count = len(mentions)
            
            prioritized.append({
                "objection_type": objection_type,
                "severity": severity,
                "mention_count": mention_count,
                "priority_score": severity * (1 + mention_count * 0.2),
                "mentions": mentions
            })
        
        # Sort by priority score
        prioritized.sort(key=lambda x: x["priority_score"], reverse=True)
        
        return prioritized
    
    def _select_value_props(self, value_focus: List[str], context: ConversationContext) -> List[str]:
        """Select most relevant value propositions to emphasize"""
        
        value_props = []
        
        # Map value focus to specific propositions
        prop_mapping = {
            "quality_focused": ["premium_materials", "craftsmanship", "durability"],
            "performance_focused": ["speed", "efficiency", "competitive_advantage"],
            "durability_focused": ["longevity", "warranty", "reliability"],
            "convenience_focused": ["ease_of_use", "support", "service"]
        }
        
        # Add value props based on focus
        for focus in value_focus:
            if focus in prop_mapping:
                value_props.extend(prop_mapping[focus])
        
        # Add context-based value props
        if context.conversation_stage == ConversationStage.DECISION:
            value_props.extend(["proven_results", "satisfaction_guarantee"])
        
        # Remove duplicates and limit to top 5
        return list(set(value_props))[:5]
    
    def _determine_next_action(self, approach: str, context: ConversationContext) -> str:
        """Determine the optimal next sales action"""
        
        action_mapping = {
            "direct_close": "present_offer_and_close",
            "consultative_close": "summarize_benefits_and_ask_for_decision",
            "objection_handling": "address_primary_objection",
            "urgency_based": "highlight_urgency_and_limited_availability",
            "educational_nurture": "provide_additional_information_and_build_value"
        }
        
        return action_mapping.get(approach, "continue_needs_discovery")
    
    def _generate_sales_recommendations(self, insights: Dict[str, Any], context: ConversationContext) -> List[str]:
        """Generate actionable sales recommendations"""
        
        recommendations = []
        
        # Approach-specific recommendations
        approach = insights.get("sales_approach", "consultative")
        if approach == "direct_close":
            recommendations.append("Present clear call-to-action and ask for the sale")
        elif approach == "consultative_close":
            recommendations.append("Summarize value match and guide toward decision")
        elif approach == "objection_handling":
            recommendations.append("Address primary objections before advancing sale")
        
        # Objection handling recommendations
        objection_priority = insights.get("objection_priority", [])
        if objection_priority:
            top_objection = objection_priority[0]
            recommendations.append(f"Priority: Address {top_objection['objection_type']} objection")
        
        # Urgency recommendations
        urgency_tactics = insights.get("urgency_tactics", [])
        if urgency_tactics:
            recommendations.append(f"Leverage urgency: {', '.join(urgency_tactics[:2])}")
        
        # Social proof recommendations
        social_proof = insights.get("social_proof_opportunities", [])
        if social_proof:
            recommendations.append(f"Use social proof: {', '.join(social_proof[:2])}")
        
        # Value proposition recommendations
        value_props = insights.get("value_props_to_emphasize", [])
        if value_props:
            recommendations.append(f"Emphasize: {', '.join(value_props[:3])}")
        
        # Closing recommendations
        closing_opportunities = insights.get("closing_opportunities", [])
        if closing_opportunities:
            recommendations.append("Closing opportunity detected - prepare to ask for commitment")
        
        # Next action recommendation
        next_action = insights.get("next_sales_action", "")
        if next_action:
            recommendations.append(f"Next: {next_action.replace('_', ' ').title()}")
        
        return recommendations
    
    def _calculate_confidence(self, insights: Dict[str, Any], context: ConversationContext) -> float:
        """Calculate confidence score for sales analysis"""
        
        base_confidence = 0.7
        
        # Boost confidence with clear buying signals
        buying_strength = insights.get("buying_signal_strength", "none")
        if buying_strength == "strong":
            base_confidence += 0.2
        elif buying_strength == "medium":
            base_confidence += 0.1
        
        # Boost confidence with conversation history
        history_boost = min(len(context.message_history) * 0.02, 0.15)
        
        # Boost confidence with successful LLM analysis
        llm_boost = 0.0
        if insights.get("analysis_method") == "llm_deep":
            llm_boost = 0.1
        
        # Boost confidence with clear objections (easier to handle)
        objections = insights.get("objection_signals", {})
        if len(objections) > 0:
            base_confidence += 0.05  # Clear objections are easier than vague hesitation
        
        total_confidence = min(base_confidence + history_boost + llm_boost, 1.0)
        
        return total_confidence
    
    def _create_fallback_sales_insight(self, message: str) -> AgentInsight:
        """Create fallback insight when sales analysis fails"""
        
        return AgentInsight(
            agent_name=self.agent_name,
            confidence_score=0.4,
            timestamp=datetime.now(),
            insights={
                "sales_approach": "consultative",
                "buying_signal_strength": "unknown",
                "objection_signals": {},
                "closing_opportunities": [],
                "urgency_tactics": [],
                "analysis_method": "fallback"
            },
            recommendations=[
                "Use consultative sales approach",
                "Ask discovery questions to understand needs",
                "Listen for buying signals and objections",
                "Build value before attempting to close"
            ],
            metadata={
                "is_fallback": True,
                "message_length": len(message)
            },
            processing_time_ms=self.max_processing_time_ms
        )
    
    def get_agent_description(self) -> str:
        """Return description of this agent's capabilities"""
        return ("Analyzes buying signals, identifies objections, and generates real-time sales "
                "strategy recommendations including closing opportunities, urgency tactics, and "
                "objection handling priorities for maximizing conversion")


# Factory function for easy agent creation
def create_sales_strategy_agent() -> SalesStrategyAgent:
    """Create and return a Sales Strategy Agent"""
    return SalesStrategyAgent()