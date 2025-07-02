"""
Customer Psychology Analyst Agent

Specialized agent that analyzes customer psychology, emotional state, decision-making
style, and communication preferences in real-time to optimize conversation strategy.
"""

import logging
import re
from datetime import datetime
from typing import Dict, Any, List, Optional

from liddy_intelligence.agents.base_agent import BaseAgent, AgentInsight
from liddy_intelligence.agents.context import ConversationContext, CustomerIntent, ConversationStage
from liddy_intelligence.llm.simple_factory import LLMFactory

logger = logging.getLogger(__name__)


class CustomerPsychologyAgent(BaseAgent):
    """
    Real-time customer psychology analysis agent.
    
    Analyzes:
    - Emotional state and mood indicators
    - Decision-making style (analytical, intuitive, social, etc.)
    - Communication preferences (technical, casual, formal)
    - Purchase readiness and urgency indicators
    - Price sensitivity signals
    - Personality traits and behavioral patterns
    """
    
    def __init__(self):
        super().__init__("psychology_agent", max_processing_time_ms=500)
        self.llm_factory = LLMFactory()
        
        # Psychology analysis patterns
        self.emotion_indicators = {
            "excited": ["excited", "amazing", "love", "perfect", "fantastic", "awesome"],
            "frustrated": ["frustrated", "annoying", "hate", "terrible", "awful", "bad"],
            "confused": ["confused", "unclear", "don't understand", "not sure", "help"],
            "confident": ["definitely", "absolutely", "certain", "sure", "confident"],
            "hesitant": ["maybe", "perhaps", "not sure", "think about", "consider"]
        }
        
        self.urgency_indicators = {
            "high": ["asap", "urgent", "immediately", "right now", "today", "quickly"],
            "medium": ["soon", "this week", "next week", "relatively quick"],
            "low": ["eventually", "sometime", "no rush", "when I get around to it"]
        }
        
        self.price_sensitivity_indicators = {
            "high": ["cheap", "affordable", "budget", "cost", "price", "expensive", "save money"],
            "low": ["premium", "quality", "best", "top-of-the-line", "investment", "worth it"]
        }
        
        logger.info("🧠 Initialized Customer Psychology Analyst Agent")
    
    async def analyze_real_time(self, message: str, context: ConversationContext) -> AgentInsight:
        """Analyze customer psychology from their latest message and conversation context"""
        
        try:
            # Quick pattern-based analysis for immediate insights
            quick_analysis = self._quick_pattern_analysis(message, context)
            
            # Deep LLM analysis for complex psychology insights
            deep_analysis = await self._deep_psychology_analysis(message, context)
            
            # Combine insights
            combined_insights = {**quick_analysis, **deep_analysis}
            
            # Generate recommendations
            recommendations = self._generate_psychology_recommendations(combined_insights, context)
            
            return AgentInsight(
                agent_name=self.agent_name,
                confidence_score=self._calculate_confidence(combined_insights, context),
                timestamp=datetime.now(),
                insights=combined_insights,
                recommendations=recommendations,
                metadata={
                    "analysis_type": "psychology",
                    "message_length": len(message),
                    "conversation_turn": len(context.message_history)
                },
                processing_time_ms=0.0  # Will be set by base class
            )
            
        except Exception as e:
            logger.error(f"❌ Psychology analysis failed: {e}")
            return self._create_fallback_psychology_insight(message)
    
    def _quick_pattern_analysis(self, message: str, context: ConversationContext) -> Dict[str, Any]:
        """Fast pattern-based psychology analysis for real-time insights"""
        
        message_lower = message.lower()
        
        # Emotional state detection
        emotional_state = "neutral"
        emotion_confidence = 0.0
        for emotion, indicators in self.emotion_indicators.items():
            matches = sum(1 for indicator in indicators if indicator in message_lower)
            if matches > 0:
                emotional_state = emotion
                emotion_confidence = min(matches * 0.3, 1.0)
                break
        
        # Urgency level detection
        urgency_level = "medium"
        urgency_confidence = 0.0
        for urgency, indicators in self.urgency_indicators.items():
            matches = sum(1 for indicator in indicators if indicator in message_lower)
            if matches > 0:
                urgency_level = urgency
                urgency_confidence = min(matches * 0.4, 1.0)
                break
        
        # Price sensitivity detection
        price_sensitivity = "medium"
        price_confidence = 0.0
        for sensitivity, indicators in self.price_sensitivity_indicators.items():
            matches = sum(1 for indicator in indicators if indicator in message_lower)
            if matches > 0:
                price_sensitivity = sensitivity
                price_confidence = min(matches * 0.3, 1.0)
                break
        
        # Communication style inference
        communication_style = self._infer_communication_style(message)
        
        # Decision-making style indicators
        decision_style = self._analyze_decision_making_style(message, context)
        
        return {
            "emotional_state": emotional_state,
            "emotion_confidence": emotion_confidence,
            "urgency_level": urgency_level,
            "urgency_confidence": urgency_confidence,
            "price_sensitivity": price_sensitivity,
            "price_confidence": price_confidence,
            "communication_style": communication_style,
            "decision_making_style": decision_style,
            "analysis_method": "pattern_based"
        }
    
    def _infer_communication_style(self, message: str) -> str:
        """Infer preferred communication style from message patterns"""
        
        # Technical indicators
        technical_words = ["specs", "specifications", "technical", "performance", "features", "dimensions"]
        technical_score = sum(1 for word in technical_words if word in message.lower())
        
        # Casual indicators
        casual_words = ["cool", "awesome", "nice", "hey", "yeah", "kinda"]
        casual_score = sum(1 for word in casual_words if word in message.lower())
        
        # Formal indicators  
        formal_words = ["please", "would", "could", "appreciate", "thank you", "regards"]
        formal_score = sum(1 for word in formal_words if word in message.lower())
        
        # Length and complexity
        avg_word_length = sum(len(word) for word in message.split()) / max(len(message.split()), 1)
        sentence_count = len([s for s in message.split('.') if s.strip()])
        
        if technical_score > 1 or avg_word_length > 6:
            return "technical"
        elif casual_score > 1 or len(message.split()) < 10:
            return "casual"
        elif formal_score > 1 or sentence_count > 2:
            return "formal"
        else:
            return "conversational"
    
    def _analyze_decision_making_style(self, message: str, context: ConversationContext) -> str:
        """Analyze customer's decision-making style from conversation patterns"""
        
        message_lower = message.lower()
        
        # Analytical indicators
        analytical_words = ["compare", "analysis", "research", "data", "facts", "numbers", "specifications"]
        analytical_score = sum(1 for word in analytical_words if word in message_lower)
        
        # Intuitive indicators
        intuitive_words = ["feel", "seems", "looks", "appears", "sense", "gut", "instinct"]
        intuitive_score = sum(1 for word in intuitive_words if word in message_lower)
        
        # Social indicators
        social_words = ["reviews", "others", "people", "recommend", "friends", "family", "popular"]
        social_score = sum(1 for word in social_words if word in message_lower)
        
        # Question patterns
        question_count = message.count('?')
        detail_requests = len(re.findall(r'\b(how|what|why|when|where|which)\b', message_lower))
        
        if analytical_score > 1 or detail_requests > 2:
            return "analytical"
        elif social_score > 1:
            return "social"
        elif intuitive_score > 1:
            return "intuitive"
        elif question_count > 2:
            return "research_oriented"
        else:
            return "balanced"
    
    async def _deep_psychology_analysis(self, message: str, context: ConversationContext) -> Dict[str, Any]:
        """Deep LLM-based psychology analysis for complex insights"""
        
        try:
            # Get recent conversation context for analysis
            recent_messages = context.get_recent_messages(5)
            conversation_history = "\n".join([
                f"{msg['sender']}: {msg['content']}" 
                for msg in recent_messages
            ])
            
            psychology_prompt = f"""Analyze the customer's psychology and communication patterns based on their latest message and conversation context.

LATEST CUSTOMER MESSAGE: "{message}"

RECENT CONVERSATION:
{conversation_history}

CURRENT CONTEXT:
- Conversation stage: {context.conversation_stage.value}
- Customer intent: {context.current_intent.value if context.current_intent else 'unknown'}
- Message count: {len(context.message_history)}

Analyze and provide insights on:

1. EMOTIONAL STATE & MOOD:
   - Current emotional state (excited, frustrated, confident, hesitant, neutral, etc.)
   - Stress indicators or pressure points
   - Enthusiasm level for the conversation/brand

2. DECISION-MAKING PSYCHOLOGY:
   - Decision-making style (analytical, intuitive, social, impulsive)
   - Information processing preference (detailed vs summary)
   - Risk tolerance indicators

3. COMMUNICATION PREFERENCES:
   - Preferred communication style (technical, casual, formal, friendly)
   - Optimal response length (short, medium, detailed)
   - Technical depth preference (basic, intermediate, expert)

4. PURCHASE PSYCHOLOGY:
   - Purchase readiness indicators (high, medium, low)
   - Price sensitivity signals (high, medium, low)
   - Urgency level (high, medium, low)
   - Social proof responsiveness (high, medium, low)

5. PERSONALITY TRAITS:
   - Key personality indicators observed
   - Lifestyle alignment signals
   - Brand affinity indicators

Respond in JSON format with confidence scores (0.0-1.0) for each assessment."""

            llm = self.llm_factory.get_service("openai/o3")
            response = await llm.chat_completion(
                messages=[{"role": "user", "content": psychology_prompt}],
                temperature=0.1,
                max_tokens=800
            )
            
            if response and response.get("content"):
                # Try to parse JSON response
                import json
                try:
                    analysis = json.loads(response["content"])
                    analysis["analysis_method"] = "llm_deep"
                    return analysis
                except json.JSONDecodeError:
                    # Fallback to text parsing if JSON fails
                    return self._parse_psychology_text_response(response["content"])
            
            return {"analysis_method": "llm_failed"}
            
        except Exception as e:
            logger.warning(f"⚠️ Deep psychology analysis failed: {e}")
            return {"analysis_method": "llm_error", "error": str(e)}
    
    def _parse_psychology_text_response(self, text_response: str) -> Dict[str, Any]:
        """Parse text-based psychology analysis response"""
        # Simple text parsing as fallback
        analysis = {"analysis_method": "llm_text_parsed"}
        
        text_lower = text_response.lower()
        
        # Extract key insights with basic text parsing
        if "excited" in text_lower or "enthusiastic" in text_lower:
            analysis["emotional_state"] = "excited"
        elif "frustrated" in text_lower or "annoyed" in text_lower:
            analysis["emotional_state"] = "frustrated"
        elif "confident" in text_lower:
            analysis["emotional_state"] = "confident"
        elif "hesitant" in text_lower or "uncertain" in text_lower:
            analysis["emotional_state"] = "hesitant"
        
        if "analytical" in text_lower:
            analysis["decision_making_style"] = "analytical"
        elif "intuitive" in text_lower:
            analysis["decision_making_style"] = "intuitive"
        elif "social" in text_lower:
            analysis["decision_making_style"] = "social"
        
        return analysis
    
    def _generate_psychology_recommendations(self, insights: Dict[str, Any], context: ConversationContext) -> List[str]:
        """Generate actionable recommendations based on psychology insights"""
        
        recommendations = []
        
        # Emotional state recommendations
        emotional_state = insights.get("emotional_state", "neutral")
        if emotional_state == "excited":
            recommendations.append("Match customer's enthusiasm and energy level")
            recommendations.append("Introduce premium options or upgrades")
        elif emotional_state == "frustrated":
            recommendations.append("Use empathetic tone and focus on solutions")
            recommendations.append("Simplify options and provide clear guidance")
        elif emotional_state == "hesitant":
            recommendations.append("Provide reassurance and social proof")
            recommendations.append("Offer risk-free trials or guarantees")
        
        # Communication style recommendations
        communication_style = insights.get("communication_style", "conversational")
        if communication_style == "technical":
            recommendations.append("Use detailed specifications and technical details")
            recommendations.append("Focus on performance metrics and comparisons")
        elif communication_style == "casual":
            recommendations.append("Use friendly, conversational tone")
            recommendations.append("Keep explanations simple and relatable")
        elif communication_style == "formal":
            recommendations.append("Maintain professional tone and structure")
            recommendations.append("Provide comprehensive information")
        
        # Decision-making style recommendations
        decision_style = insights.get("decision_making_style", "balanced")
        if decision_style == "analytical":
            recommendations.append("Provide detailed comparisons and data")
            recommendations.append("Focus on logical benefits and ROI")
        elif decision_style == "intuitive":
            recommendations.append("Emphasize emotional benefits and feelings")
            recommendations.append("Use storytelling and experiential language")
        elif decision_style == "social":
            recommendations.append("Include customer reviews and testimonials")
            recommendations.append("Mention popularity and social proof")
        
        # Urgency and price sensitivity recommendations
        urgency_level = insights.get("urgency_level", "medium")
        price_sensitivity = insights.get("price_sensitivity", "medium")
        
        if urgency_level == "high":
            recommendations.append("Highlight immediate availability and fast delivery")
            recommendations.append("Create appropriate sense of urgency")
        elif urgency_level == "low":
            recommendations.append("Focus on long-term value and quality")
            recommendations.append("Allow time for consideration")
        
        if price_sensitivity == "high":
            recommendations.append("Emphasize value proposition and cost savings")
            recommendations.append("Mention financing options or promotions")
        elif price_sensitivity == "low":
            recommendations.append("Focus on premium features and quality")
            recommendations.append("Highlight exclusivity and craftsmanship")
        
        return recommendations
    
    def _calculate_confidence(self, insights: Dict[str, Any], context: ConversationContext) -> float:
        """Calculate confidence score for psychology analysis"""
        
        base_confidence = 0.6
        
        # Boost confidence with more conversation history
        history_boost = min(len(context.message_history) * 0.05, 0.2)
        
        # Boost confidence with clear indicators
        pattern_confidence = 0.0
        for key in ["emotion_confidence", "urgency_confidence", "price_confidence"]:
            if key in insights:
                pattern_confidence += insights[key] * 0.1
        
        # Boost confidence if LLM analysis succeeded
        llm_boost = 0.0
        if insights.get("analysis_method") == "llm_deep":
            llm_boost = 0.15
        elif insights.get("analysis_method") == "llm_text_parsed":
            llm_boost = 0.1
        
        total_confidence = min(base_confidence + history_boost + pattern_confidence + llm_boost, 1.0)
        
        return total_confidence
    
    def _create_fallback_psychology_insight(self, message: str) -> AgentInsight:
        """Create fallback insight when psychology analysis fails"""
        
        return AgentInsight(
            agent_name=self.agent_name,
            confidence_score=0.3,
            timestamp=datetime.now(),
            insights={
                "emotional_state": "neutral",
                "communication_style": "conversational",
                "decision_making_style": "balanced",
                "urgency_level": "medium",
                "price_sensitivity": "medium",
                "analysis_method": "fallback"
            },
            recommendations=[
                "Use balanced, professional communication style",
                "Provide moderate detail level in responses",
                "Focus on core value propositions"
            ],
            metadata={
                "is_fallback": True,
                "message_length": len(message)
            },
            processing_time_ms=self.max_processing_time_ms
        )
    
    def get_agent_description(self) -> str:
        """Return description of this agent's capabilities"""
        return ("Analyzes customer psychology, emotional state, communication preferences, "
                "decision-making style, and purchase readiness to optimize conversation strategy")


# Factory function for easy agent creation
def create_psychology_agent() -> CustomerPsychologyAgent:
    """Create and return a Customer Psychology Analyst Agent"""
    return CustomerPsychologyAgent()