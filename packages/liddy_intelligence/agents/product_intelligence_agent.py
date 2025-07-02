"""
Product Intelligence Agent

Specialized agent that provides real-time product discovery, ranking, and 
recommendation intelligence based on customer psychology, conversation context,
and brand positioning. Optimized for intelligent product matching and upselling.
"""

import logging
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from liddy_intelligence.agents.base_agent import BaseAgent, AgentInsight
from liddy_intelligence.agents.context import ConversationContext, CustomerIntent, ConversationStage
from liddy_intelligence.llm.simple_factory import LLMFactory

logger = logging.getLogger(__name__)


class ProductIntelligenceAgent(BaseAgent):
    """
    Real-time product intelligence and recommendation agent.
    
    Analyzes:
    - Product interest signals from customer messages
    - Use case inference and product matching
    - Competitive positioning opportunities
    - Upselling and cross-selling recommendations
    - Real-time product ranking based on customer psychology
    - Technical specification interests
    """
    
    def __init__(self, brand_products: Optional[List[Dict[str, Any]]] = None):
        super().__init__("product_intelligence_agent", max_processing_time_ms=500)
        self.llm_factory = LLMFactory()
        
        # Product catalog (would be loaded from actual product data)
        self.brand_products = brand_products or self._get_sample_products()
        
        # Product discovery patterns
        self.product_signals = {
            "bikes": ["bike", "bicycle", "cycling", "ride", "pedal", "frame"],
            "road_bikes": ["road", "racing", "speed", "lightweight", "aerodynamic", "fast"],
            "mountain_bikes": ["mountain", "trail", "mtb", "off-road", "suspension", "rugged"],
            "electric_bikes": ["electric", "e-bike", "ebike", "motor", "battery", "assist"],
            "commuter_bikes": ["commute", "city", "urban", "daily", "work", "transport"],
            "accessories": ["helmet", "lights", "lock", "bag", "clothing", "gear"]
        }
        
        self.use_case_indicators = {
            "commuting": ["work", "commute", "daily", "city", "urban", "transport"],
            "racing": ["race", "competition", "fast", "speed", "performance", "training"],
            "recreation": ["fun", "leisure", "weekend", "casual", "hobby", "enjoyment"],
            "fitness": ["exercise", "fitness", "health", "workout", "training", "cardio"],
            "adventure": ["adventure", "explore", "trail", "mountain", "outdoor", "nature"]
        }
        
        self.technical_interests = {
            "performance": ["specs", "performance", "speed", "weight", "aerodynamic"],
            "technology": ["technology", "electronic", "smart", "digital", "app"],
            "materials": ["carbon", "aluminum", "titanium", "steel", "frame", "material"],
            "components": ["shimano", "sram", "wheels", "brakes", "gears", "drivetrain"]
        }
        
        logger.info("🛍️ Initialized Product Intelligence Agent")
    
    def _get_sample_products(self) -> List[Dict[str, Any]]:
        """Sample product catalog for demonstration (would be replaced with real data)"""
        return [
            {
                "id": "tarmac-sl8-expert",
                "name": "Tarmac SL8 Expert",
                "category": "road_bikes",
                "price": 4500,
                "key_features": ["FACT 12r Carbon", "Shimano Ultegra Di2", "Aerodynamic"],
                "use_cases": ["racing", "performance", "training"],
                "technical_specs": {
                    "weight": "7.25kg",
                    "frame_material": "FACT 12r Carbon",
                    "groupset": "Shimano Ultegra Di2"
                },
                "target_segments": ["serious_cyclists", "racing_enthusiasts"],
                "competitive_advantages": ["World Tour proven", "Fastest aero design", "Rider-First Engineering"]
            },
            {
                "id": "roubaix-comp",
                "name": "Roubaix Comp",
                "category": "road_bikes", 
                "price": 3200,
                "key_features": ["Future Shock", "Endurance Geometry", "Comfort"],
                "use_cases": ["endurance", "comfort", "long_rides"],
                "technical_specs": {
                    "weight": "8.1kg",
                    "frame_material": "FACT 10r Carbon",
                    "groupset": "Shimano 105"
                },
                "target_segments": ["endurance_riders", "comfort_seekers"],
                "competitive_advantages": ["Future Shock suspension", "All-day comfort", "Stable handling"]
            },
            {
                "id": "turbo-vado-40",
                "name": "Turbo Vado 4.0",
                "category": "electric_bikes",
                "price": 3800,
                "key_features": ["Turbo Motor", "90mi Range", "Smart Control"],
                "use_cases": ["commuting", "urban", "assistance"],
                "technical_specs": {
                    "weight": "22kg",
                    "motor": "Turbo 2.2",
                    "battery": "710Wh",
                    "range": "90 miles"
                },
                "target_segments": ["commuters", "urban_riders"],
                "competitive_advantages": ["Longest range", "Smart connectivity", "Natural feel"]
            },
            {
                "id": "stumpjumper-evo-comp",
                "name": "Stumpjumper EVO Comp",
                "category": "mountain_bikes",
                "price": 4200,
                "key_features": ["29er", "150mm Travel", "Aggressive Geometry"],
                "use_cases": ["trail", "mountain", "aggressive_riding"],
                "technical_specs": {
                    "wheel_size": "29\"",
                    "travel": "150mm",
                    "frame_material": "FACT 11m Carbon"
                },
                "target_segments": ["mountain_bikers", "trail_riders"],
                "competitive_advantages": ["Aggressive geometry", "Playful handling", "Versatile performance"]
            }
        ]
    
    async def analyze_real_time(self, message: str, context: ConversationContext) -> AgentInsight:
        """Analyze customer message for product intelligence and recommendations"""
        
        try:
            # Quick pattern-based product analysis
            quick_analysis = self._quick_product_analysis(message, context)
            
            # Deep LLM-based product intelligence
            deep_analysis = await self._deep_product_analysis(message, context)
            
            # Product ranking based on customer psychology
            ranked_products = self._rank_products_for_customer(quick_analysis, context)
            
            # Generate upselling opportunities
            upsell_opportunities = self._identify_upsell_opportunities(ranked_products, context)
            
            # Combine all insights
            combined_insights = {
                **quick_analysis,
                **deep_analysis,
                "ranked_products": ranked_products,
                "upsell_opportunities": upsell_opportunities
            }
            
            # Generate recommendations
            recommendations = self._generate_product_recommendations(combined_insights, context)
            
            return AgentInsight(
                agent_name=self.agent_name,
                confidence_score=self._calculate_confidence(combined_insights, context),
                timestamp=datetime.now(),
                insights=combined_insights,
                recommendations=recommendations,
                metadata={
                    "analysis_type": "product_intelligence",
                    "products_analyzed": len(self.brand_products),
                    "top_matches": len(ranked_products[:3])
                },
                processing_time_ms=0.0  # Will be set by base class
            )
            
        except Exception as e:
            logger.error(f"❌ Product intelligence analysis failed: {e}")
            return self._create_fallback_product_insight(message)
    
    def _quick_product_analysis(self, message: str, context: ConversationContext) -> Dict[str, Any]:
        """Fast pattern-based product analysis for real-time insights"""
        
        message_lower = message.lower()
        
        # Product category detection
        detected_categories = []
        for category, keywords in self.product_signals.items():
            matches = sum(1 for keyword in keywords if keyword in message_lower)
            if matches > 0:
                detected_categories.append((category, matches))
        
        # Sort by match count
        detected_categories.sort(key=lambda x: x[1], reverse=True)
        primary_category = detected_categories[0][0] if detected_categories else "general"
        
        # Use case detection
        detected_use_cases = []
        for use_case, keywords in self.use_case_indicators.items():
            matches = sum(1 for keyword in keywords if keyword in message_lower)
            if matches > 0:
                detected_use_cases.append((use_case, matches))
        
        detected_use_cases.sort(key=lambda x: x[1], reverse=True)
        primary_use_case = detected_use_cases[0][0] if detected_use_cases else "general"
        
        # Technical interest detection
        technical_focus = []
        for tech_area, keywords in self.technical_interests.items():
            matches = sum(1 for keyword in keywords if keyword in message_lower)
            if matches > 0:
                technical_focus.append(tech_area)
        
        # Price sensitivity signals
        price_signals = self._detect_price_signals(message_lower)
        
        # Urgency indicators
        urgency_signals = self._detect_urgency_signals(message_lower)
        
        return {
            "product_category_interest": primary_category,
            "category_confidence": detected_categories[0][1] / 10.0 if detected_categories else 0.0,
            "use_case_intent": primary_use_case,
            "technical_interests": technical_focus,
            "price_sensitivity_signals": price_signals,
            "urgency_indicators": urgency_signals,
            "mentioned_products": self._extract_mentioned_products(message_lower),
            "competitor_mentions": self._detect_competitor_mentions(message_lower),
            "analysis_method": "pattern_based"
        }
    
    def _detect_price_signals(self, message: str) -> Dict[str, bool]:
        """Detect price-related signals in customer message"""
        return {
            "budget_conscious": any(word in message for word in ["cheap", "affordable", "budget", "cost", "price"]),
            "premium_interested": any(word in message for word in ["premium", "best", "top", "high-end", "quality"]),
            "value_focused": any(word in message for word in ["value", "worth", "deal", "bang for buck"]),
            "financing_interested": any(word in message for word in ["payment", "financing", "installment", "monthly"])
        }
    
    def _detect_urgency_signals(self, message: str) -> Dict[str, bool]:
        """Detect urgency-related signals in customer message"""
        return {
            "immediate_need": any(word in message for word in ["asap", "urgent", "immediately", "right now", "today"]),
            "time_sensitive": any(word in message for word in ["soon", "quickly", "this week", "deadline"]),
            "seasonal_urgency": any(word in message for word in ["spring", "summer", "winter", "season", "weather"]),
            "event_driven": any(word in message for word in ["race", "event", "trip", "vacation", "competition"])
        }
    
    def _extract_mentioned_products(self, message: str) -> List[str]:
        """Extract specific product mentions from message"""
        mentioned = []
        for product in self.brand_products:
            product_name_lower = product["name"].lower()
            # Check for partial matches of product names
            name_parts = product_name_lower.split()
            if any(part in message for part in name_parts if len(part) > 3):
                mentioned.append(product["name"])
        return mentioned
    
    def _detect_competitor_mentions(self, message: str) -> List[str]:
        """Detect competitor brand mentions"""
        competitors = ["trek", "giant", "cannondale", "bianchi", "cervelo", "pinarello", "scott"]
        mentioned_competitors = []
        for competitor in competitors:
            if competitor in message:
                mentioned_competitors.append(competitor)
        return mentioned_competitors
    
    async def _deep_product_analysis(self, message: str, context: ConversationContext) -> Dict[str, Any]:
        """Deep LLM-based product analysis for complex recommendations"""
        
        try:
            # Get recent conversation context
            recent_messages = context.get_recent_messages(5)
            conversation_history = "\n".join([
                f"{msg['sender']}: {msg['content']}" 
                for msg in recent_messages
            ])
            
            # Build product context for LLM
            product_context = self._build_product_context_for_llm()
            
            product_analysis_prompt = f"""Analyze the customer's product needs and interests based on their message and conversation context.

CUSTOMER MESSAGE: "{message}"

RECENT CONVERSATION:
{conversation_history}

AVAILABLE PRODUCTS:
{product_context}

CONVERSATION CONTEXT:
- Stage: {context.conversation_stage.value}
- Intent: {context.current_intent.value if context.current_intent else 'unknown'}
- Previous interests: {context.expressed_interests}

Analyze and provide insights on:

1. PRODUCT MATCHING:
   - Which specific products best match the customer's stated needs?
   - What's the confidence level for each product recommendation?
   - Are there any products that are clearly NOT suitable?

2. USE CASE ANALYSIS:
   - What is the customer's primary intended use case?
   - Are there secondary use cases we should consider?
   - How does this impact product recommendations?

3. FEATURE PRIORITIES:
   - What product features seem most important to this customer?
   - Are they focused on performance, comfort, technology, or value?
   - What technical specifications should we highlight?

4. UPSELLING OPPORTUNITIES:
   - Are there premium upgrades that would benefit this customer?
   - What complementary products might they need?
   - Should we suggest professional services (fitting, assembly)?

5. COMPETITIVE POSITIONING:
   - How do our recommendations compare to likely alternatives?
   - What unique advantages should we emphasize?
   - Are there potential objections we should preemptively address?

Respond in JSON format with specific product recommendations and confidence scores."""

            llm = self.llm_factory.get_service("openai/o3")
            response = await llm.chat_completion(
                messages=[{"role": "user", "content": product_analysis_prompt}],
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
                    return self._parse_product_text_response(response["content"])
            
            return {"analysis_method": "llm_failed"}
            
        except Exception as e:
            logger.warning(f"⚠️ Deep product analysis failed: {e}")
            return {"analysis_method": "llm_error", "error": str(e)}
    
    def _build_product_context_for_llm(self) -> str:
        """Build concise product context for LLM analysis"""
        product_summaries = []
        for product in self.brand_products:
            summary = f"- {product['name']}: {product['category']}, ${product['price']}, {', '.join(product['key_features'][:3])}"
            product_summaries.append(summary)
        return "\n".join(product_summaries)
    
    def _parse_product_text_response(self, text_response: str) -> Dict[str, Any]:
        """Parse text-based product analysis response as fallback"""
        analysis = {"analysis_method": "llm_text_parsed"}
        
        # Simple text parsing to extract key insights
        text_lower = text_response.lower()
        
        # Extract recommended products
        recommended_products = []
        for product in self.brand_products:
            if product["name"].lower() in text_lower:
                recommended_products.append(product["id"])
        
        analysis["recommended_products"] = recommended_products
        
        # Extract focus areas
        if "performance" in text_lower:
            analysis["feature_focus"] = "performance"
        elif "comfort" in text_lower:
            analysis["feature_focus"] = "comfort"
        elif "value" in text_lower:
            analysis["feature_focus"] = "value"
        
        return analysis
    
    def _rank_products_for_customer(self, quick_analysis: Dict[str, Any], context: ConversationContext) -> List[Dict[str, Any]]:
        """Rank products based on customer analysis and psychology"""
        
        ranked_products = []
        
        target_category = quick_analysis.get("product_category_interest", "general")
        target_use_case = quick_analysis.get("use_case_intent", "general")
        technical_interests = quick_analysis.get("technical_interests", [])
        price_signals = quick_analysis.get("price_sensitivity_signals", {})
        
        for product in self.brand_products:
            score = 0.0
            reasons = []
            
            # Category match scoring
            if target_category in product.get("category", ""):
                score += 40
                reasons.append(f"Matches {target_category} interest")
            
            # Use case match scoring
            product_use_cases = product.get("use_cases", [])
            if target_use_case in product_use_cases:
                score += 30
                reasons.append(f"Perfect for {target_use_case}")
            
            # Technical interest match
            for tech_interest in technical_interests:
                if any(tech_interest in feature.lower() for feature in product.get("key_features", [])):
                    score += 15
                    reasons.append(f"Strong {tech_interest} features")
            
            # Price alignment scoring
            if price_signals.get("budget_conscious") and product["price"] < 3000:
                score += 20
                reasons.append("Budget-friendly option")
            elif price_signals.get("premium_interested") and product["price"] > 4000:
                score += 20
                reasons.append("Premium quality")
            elif price_signals.get("value_focused"):
                # Mid-range products score well for value
                if 3000 <= product["price"] <= 4500:
                    score += 25
                    reasons.append("Excellent value proposition")
            
            # Customer profile alignment (if available)
            if hasattr(context, 'customer_profile') and context.customer_profile:
                profile = context.customer_profile
                if profile.brand_affinity_score > 0.7:
                    score += 10
                    reasons.append("Strong brand alignment")
            
            ranked_products.append({
                "product": product,
                "score": score,
                "confidence": min(score / 100.0, 1.0),
                "match_reasons": reasons
            })
        
        # Sort by score
        ranked_products.sort(key=lambda x: x["score"], reverse=True)
        
        return ranked_products[:5]  # Return top 5 matches
    
    def _identify_upsell_opportunities(self, ranked_products: List[Dict[str, Any]], context: ConversationContext) -> List[Dict[str, Any]]:
        """Identify upselling opportunities based on top product matches"""
        
        if not ranked_products:
            return []
        
        top_product = ranked_products[0]["product"]
        upsell_opportunities = []
        
        # Find higher-tier products in same category
        for product in self.brand_products:
            if (product["category"] == top_product["category"] and 
                product["price"] > top_product["price"] and
                product["id"] != top_product["id"]):
                
                price_premium = product["price"] - top_product["price"]
                premium_percentage = (price_premium / top_product["price"]) * 100
                
                # Only suggest reasonable upsells (20-50% premium)
                if 20 <= premium_percentage <= 50:
                    upsell_opportunities.append({
                        "product": product,
                        "price_premium": price_premium,
                        "premium_percentage": premium_percentage,
                        "upsell_reasons": self._generate_upsell_reasons(top_product, product)
                    })
        
        # Add accessory upsells
        accessory_upsells = self._suggest_accessory_upsells(top_product)
        upsell_opportunities.extend(accessory_upsells)
        
        return upsell_opportunities[:3]  # Top 3 upsell opportunities
    
    def _generate_upsell_reasons(self, base_product: Dict[str, Any], premium_product: Dict[str, Any]) -> List[str]:
        """Generate reasons for upselling to premium product"""
        reasons = []
        
        base_features = set(base_product.get("key_features", []))
        premium_features = set(premium_product.get("key_features", []))
        
        # Find unique features in premium product
        unique_features = premium_features - base_features
        for feature in list(unique_features)[:2]:  # Top 2 unique features
            reasons.append(f"Upgraded {feature}")
        
        # Price-based reasoning
        price_diff = premium_product["price"] - base_product["price"]
        if price_diff < 1000:
            reasons.append("Small upgrade investment for significant performance gain")
        
        # Category-specific reasons
        if base_product.get("category") == "road_bikes":
            reasons.append("Professional racing capabilities")
        elif base_product.get("category") == "mountain_bikes":
            reasons.append("Advanced trail performance")
        
        return reasons
    
    def _suggest_accessory_upsells(self, product: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest relevant accessories for the main product"""
        
        # Sample accessory suggestions (would be based on real accessory catalog)
        accessory_suggestions = []
        
        if "bikes" in product.get("category", ""):
            accessory_suggestions.extend([
                {
                    "product": {
                        "id": "specialized-helmet-pro",
                        "name": "Specialized S-Works Helmet",
                        "category": "accessories",
                        "price": 350
                    },
                    "price_premium": 350,
                    "upsell_reasons": ["Safety essential", "Aerodynamic advantage", "Matches bike design"]
                },
                {
                    "product": {
                        "id": "bike-computer-pro",
                        "name": "Specialized Bike Computer",
                        "category": "accessories", 
                        "price": 450
                    },
                    "price_premium": 450,
                    "upsell_reasons": ["Track performance", "Navigation", "Training optimization"]
                }
            ])
        
        return accessory_suggestions
    
    def _generate_product_recommendations(self, insights: Dict[str, Any], context: ConversationContext) -> List[str]:
        """Generate actionable product recommendations"""
        
        recommendations = []
        
        # Product-specific recommendations
        ranked_products = insights.get("ranked_products", [])
        if ranked_products:
            top_product = ranked_products[0]
            recommendations.append(f"Recommend {top_product['product']['name']} as primary option")
            
            if len(ranked_products) > 1:
                alt_product = ranked_products[1]
                recommendations.append(f"Present {alt_product['product']['name']} as alternative")
        
        # Technical focus recommendations
        technical_interests = insights.get("technical_interests", [])
        if "performance" in technical_interests:
            recommendations.append("Emphasize performance specifications and racing heritage")
        if "technology" in technical_interests:
            recommendations.append("Highlight electronic components and smart features")
        if "materials" in technical_interests:
            recommendations.append("Discuss frame materials and manufacturing processes")
        
        # Use case recommendations
        use_case = insights.get("use_case_intent", "")
        if use_case == "commuting":
            recommendations.append("Focus on durability, comfort, and practical features")
        elif use_case == "racing":
            recommendations.append("Emphasize aerodynamics, weight, and competitive advantages")
        elif use_case == "recreation":
            recommendations.append("Highlight versatility, comfort, and enjoyment factors")
        
        # Upselling recommendations
        upsell_opportunities = insights.get("upsell_opportunities", [])
        if upsell_opportunities:
            top_upsell = upsell_opportunities[0]
            recommendations.append(f"Present {top_upsell['product']['name']} upgrade option")
        
        # Competitive positioning
        competitor_mentions = insights.get("competitor_mentions", [])
        if competitor_mentions:
            recommendations.append("Address competitive comparison and highlight unique advantages")
        
        # Price sensitivity handling
        price_signals = insights.get("price_sensitivity_signals", {})
        if price_signals.get("budget_conscious"):
            recommendations.append("Emphasize value proposition and financing options")
        elif price_signals.get("premium_interested"):
            recommendations.append("Focus on premium features and exclusivity")
        
        return recommendations
    
    def _calculate_confidence(self, insights: Dict[str, Any], context: ConversationContext) -> float:
        """Calculate confidence score for product analysis"""
        
        base_confidence = 0.7
        
        # Boost confidence with clear product signals
        category_confidence = insights.get("category_confidence", 0.0)
        confidence_boost = category_confidence * 0.2
        
        # Boost with conversation history
        history_boost = min(len(context.message_history) * 0.02, 0.15)
        
        # Boost with successful LLM analysis
        llm_boost = 0.0
        if insights.get("analysis_method") == "llm_deep":
            llm_boost = 0.1
        
        # Boost with ranked products
        ranked_products = insights.get("ranked_products", [])
        if ranked_products and ranked_products[0]["score"] > 50:
            confidence_boost += 0.1
        
        total_confidence = min(base_confidence + confidence_boost + history_boost + llm_boost, 1.0)
        
        return total_confidence
    
    def _create_fallback_product_insight(self, message: str) -> AgentInsight:
        """Create fallback insight when product analysis fails"""
        
        return AgentInsight(
            agent_name=self.agent_name,
            confidence_score=0.4,
            timestamp=datetime.now(),
            insights={
                "product_category_interest": "general",
                "use_case_intent": "general",
                "technical_interests": [],
                "ranked_products": [],
                "upsell_opportunities": [],
                "analysis_method": "fallback"
            },
            recommendations=[
                "Ask customer about their specific product needs",
                "Inquire about intended use case and requirements",
                "Explore budget range and feature priorities"
            ],
            metadata={
                "is_fallback": True,
                "message_length": len(message)
            },
            processing_time_ms=self.max_processing_time_ms
        )
    
    def get_agent_description(self) -> str:
        """Return description of this agent's capabilities"""
        return ("Analyzes customer product interests, provides real-time product ranking, "
                "identifies upselling opportunities, and generates intelligent product recommendations "
                "based on customer psychology and conversation context")


# Factory function for easy agent creation
def create_product_intelligence_agent(brand_products: Optional[List[Dict[str, Any]]] = None) -> ProductIntelligenceAgent:
    """Create and return a Product Intelligence Agent"""
    return ProductIntelligenceAgent(brand_products=brand_products)