"""
Query Optimization Agent for RAG Enhancement

Specialized agent that optimizes search queries for product and knowledge searches
using sophisticated understanding of user intent, context, and brand-specific knowledge.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from .base_agent import BaseAgent, AgentInsight
from .context import ConversationContext
from src.llm.simple_factory import LLMFactory
from .catalog_filter_analyzer import CatalogFilterAnalyzer

logger = logging.getLogger(__name__)


class QueryOptimizationAgent(BaseAgent):
    """
    Optimizes RAG queries based on conversation context and user intent.
    
    Key capabilities:
    - Intent extraction and clarification
    - Query expansion with synonyms and related terms
    - Context-aware filtering
    - Brand-specific terminology mapping
    - Multi-query strategy for comprehensive results
    """
    
    def __init__(self, brand_domain: str, catalog_filters: Optional[Dict[str, Any]] = None):
        super().__init__("query_optimization_agent", max_processing_time_ms=1000)
        self.brand_domain = brand_domain
        self.llm_factory = LLMFactory()
        
        # Brand-specific terminology mappings (would be loaded from brand data)
        self.brand_terms = self._load_brand_terminology()
        
        # Load catalog filters - these define what filters are available in the product catalog
        self.catalog_filters = catalog_filters or self._load_catalog_filters()
        
        # Query patterns for different intents
        self.query_patterns = {
            "specific_product": {
                "indicators": ["looking for", "need", "want", "searching"],
                "expansion": ["features", "specifications", "details"]
            },
            "comparison": {
                "indicators": ["compare", "versus", "difference", "better"],
                "expansion": ["comparison", "advantages", "benefits"]
            },
            "technical": {
                "indicators": ["specs", "technical", "performance", "data"],
                "expansion": ["specifications", "metrics", "measurements"]
            },
            "use_case": {
                "indicators": ["for", "using", "need to", "want to"],
                "expansion": ["suitable for", "designed for", "best for"]
            },
            "problem_solving": {
                "indicators": ["issue", "problem", "help", "fix"],
                "expansion": ["solution", "troubleshooting", "guide"]
            }
        }
        
        logger.info(f"🔍 Initialized Query Optimization Agent for {brand_domain}")
    
    def _load_brand_terminology(self) -> Dict[str, List[str]]:
        """Load brand-specific terminology mappings"""
        
        # This would load from brand research data
        # For now, using example mappings
        if "specialized" in self.brand_domain.lower():
            return {
                "bike": ["bicycle", "cycle", "bike"],
                "road": ["road bike", "road cycling", "pavement"],
                "mountain": ["mountain bike", "mtb", "trail", "off-road"],
                "electric": ["e-bike", "ebike", "electric bike", "pedal assist"],
                "speed": ["fast", "quick", "rapid", "velocity", "performance"],
                "lightweight": ["light", "weight", "carbon", "aerodynamic"],
                "comfort": ["comfortable", "ergonomic", "endurance", "smooth"]
            }
        
        return {}
    
    def _load_catalog_filters(self) -> Dict[str, Any]:
        """Load available filters from dynamic catalog analysis"""
        
        try:
            # First, try to load pre-analyzed filters
            filters_path = Path(f"accounts/{self.brand_domain}/catalog_filters.json")
            if filters_path.exists():
                with open(filters_path) as f:
                    filters = json.load(f)
                    logger.info(f"📂 Loaded pre-analyzed filters for {self.brand_domain}")
                    return filters
            
            # If no pre-analyzed filters, warn but use fallback
            logger.warning(f"⚠️ No pre-analyzed filters found for {self.brand_domain}")
            logger.info("💡 Run catalog analysis first: CatalogFilterAnalyzer.analyze_product_catalog()")
            logger.info("💡 This should be done when the product catalog changes, not during queries")
            
            return self._get_fallback_filters()
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load/analyze catalog filters: {e}")
            return self._get_fallback_filters()
    
    def _get_fallback_filters(self) -> Dict[str, Any]:
        """Fallback filters when catalog analysis fails"""
        
        return {
            "category": {
                "type": "categorical", 
                "values": [],
                "aliases": {},
                "note": "Catalog analysis failed - using fallback"
            },
            "price": {
                "type": "numeric_range",
                "min": 0,
                "max": 10000,
                "common_ranges": [
                    {"label": "budget", "range": [0, 1000]},
                    {"label": "mid-range", "range": [1000, 5000]},
                    {"label": "premium", "range": [5000, 10000]}
                ]
            }
        }
    
    async def analyze_real_time(self, message: str, context: ConversationContext) -> AgentInsight:
        """Analyze query for optimization - used when called as an agent"""
        
        # This method allows QueryOptimizationAgent to be used as a standalone agent
        # In practice, it's usually called directly via optimize_product_query
        
        result = await self.optimize_product_query(
            original_query=message,
            context={"recent_messages": context.get_recent_messages(5)},
            user_state=None
        )
        
        return AgentInsight(
            agent_name=self.agent_name,
            confidence_score=result.get("confidence", 0.7),
            timestamp=datetime.now(),
            insights={
                "optimized_query": result.get("optimized_query"),
                "extracted_filters": result.get("filters", {}),
                "query_intent": result.get("intent"),
                "alternative_queries": result.get("alternative_queries", [])
            },
            recommendations=[
                f"Use optimized query: {result.get('optimized_query')}",
                f"Apply filters: {list(result.get('filters', {}).keys())}"
            ],
            metadata={
                "analysis_type": "query_optimization",
                "original_query": message
            },
            processing_time_ms=0.0
        )
    
    def get_agent_description(self) -> str:
        """Return description of this agent's capabilities"""
        return ("Optimizes search queries by extracting structured filters, expanding terms "
                "with brand-specific vocabulary, and generating multiple query strategies "
                "for comprehensive product and knowledge discovery")
    
    async def optimize_product_query(
        self,
        original_query: str,
        context: Dict[str, Any],
        user_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Optimize a product search query"""
        
        try:
            # Quick intent analysis
            intent_analysis = self._analyze_query_intent(original_query, context)
            
            # Extract key terms and expand
            expanded_terms = self._expand_query_terms(original_query, intent_analysis)
            
            # Build filters from context
            filters = self._extract_filters_from_context(original_query, context, user_state)
            
            # Generate multiple query strategies
            query_strategies = await self._generate_query_strategies(
                original_query, intent_analysis, expanded_terms, filters
            )
            
            # Select best strategy
            best_strategy = self._select_best_strategy(query_strategies, context)
            
            return {
                "original_query": original_query,
                "optimized_query": best_strategy["query"],
                "intent": intent_analysis["primary_intent"],
                "confidence": best_strategy["confidence"],
                "filters": filters,
                "alternative_queries": [s["query"] for s in query_strategies[1:3]],
                "follow_up_questions": self._generate_follow_up_questions(intent_analysis, filters),
                "expansion_terms": expanded_terms,
                "strategy_reasoning": best_strategy.get("reasoning", "")
            }
            
        except Exception as e:
            logger.error(f"❌ Query optimization failed: {e}")
            return {
                "original_query": original_query,
                "optimized_query": original_query,
                "confidence": 0.5,
                "error": str(e)
            }
    
    def _analyze_query_intent(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the intent behind the query"""
        
        query_lower = query.lower()
        detected_intents = []
        
        # Check for intent patterns
        for intent, patterns in self.query_patterns.items():
            for indicator in patterns["indicators"]:
                if indicator in query_lower:
                    detected_intents.append(intent)
                    break
        
        # Analyze query structure
        is_question = "?" in query or any(q in query_lower for q in ["what", "which", "how", "when", "where"])
        has_criteria = any(word in query_lower for word in ["with", "under", "over", "between", "for"])
        
        # Determine primary intent
        if not detected_intents:
            if is_question:
                primary_intent = "information_seeking"
            elif has_criteria:
                primary_intent = "specific_product"
            else:
                primary_intent = "general_browsing"
        else:
            primary_intent = detected_intents[0]
        
        return {
            "primary_intent": primary_intent,
            "secondary_intents": detected_intents[1:],
            "is_question": is_question,
            "has_criteria": has_criteria,
            "query_type": "specific" if len(query.split()) > 3 else "broad"
        }
    
    def _expand_query_terms(self, query: str, intent_analysis: Dict[str, Any]) -> List[str]:
        """Expand query with synonyms and related terms"""
        
        expanded_terms = []
        query_words = query.lower().split()
        
        # Expand based on brand terminology
        for word in query_words:
            if word in self.brand_terms:
                expanded_terms.extend(self.brand_terms[word])
        
        # Expand based on intent
        primary_intent = intent_analysis["primary_intent"]
        if primary_intent in self.query_patterns:
            expanded_terms.extend(self.query_patterns[primary_intent]["expansion"])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_expansions = []
        query_lower = query.lower()
        for term in expanded_terms:
            if term not in seen and term not in query_lower:
                seen.add(term)
                unique_expansions.append(term)
        
        return unique_expansions[:5]  # Limit to top 5 expansions
    
    def _extract_filters_from_context(
        self,
        query: str,
        context: Dict[str, Any],
        user_state: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract structured filters from query using catalog filter definitions"""
        
        filters = {}
        query_lower = query.lower()
        
        # Extract each filter type based on catalog definitions
        for filter_name, filter_config in self.catalog_filters.items():
            filter_type = filter_config.get("type")
            
            if filter_type == "categorical":
                extracted_value = self._extract_categorical_filter(query_lower, filter_config)
                if extracted_value:
                    filters[filter_name] = extracted_value
                    
            elif filter_type == "numeric_range":
                extracted_range = self._extract_numeric_filter(query_lower, filter_config, filter_name)
                if extracted_range:
                    filters[filter_name] = extracted_range
                    
            elif filter_type == "multi_select":
                extracted_values = self._extract_multi_select_filter(query_lower, filter_config)
                if extracted_values:
                    filters[filter_name] = extracted_values
        
        # Enhance with user state
        if user_state:
            filters = self._enhance_filters_with_user_state(filters, user_state)
        
        return filters
    
    def _extract_categorical_filter(self, query: str, filter_config: Dict[str, Any]) -> Optional[str]:
        """Extract categorical filter value from query"""
        
        values = filter_config.get("values", [])
        aliases = filter_config.get("aliases", {})
        
        # Check direct matches first
        for value in values:
            if value.lower() in query:
                return value
        
        # Check aliases
        for value, alias_list in aliases.items():
            for alias in alias_list:
                if alias.lower() in query:
                    return value
        
        return None
    
    def _extract_numeric_filter(self, query: str, filter_config: Dict[str, Any], filter_name: str) -> Optional[List[float]]:
        """Extract numeric range filter from query"""
        
        if filter_name == "price":
            return self._extract_price_range(query, filter_config)
        elif filter_name == "weight":
            return self._extract_weight_range(query, filter_config)
        
        return None
    
    def _extract_price_range(self, query: str, filter_config: Dict[str, Any]) -> Optional[List[float]]:
        """Extract price range from query text"""
        
        # Look for explicit price ranges: "$2000 to $4000", "between 2k and 5k"
        range_patterns = [
            r'\$?(\d+)k?\s*(?:to|and|-)\s*\$?(\d+)k?',
            r'between\s+\$?(\d+)k?\s+(?:and|to)\s+\$?(\d+)k?',
            r'under\s+\$?(\d+)k?',
            r'below\s+\$?(\d+)k?',
            r'over\s+\$?(\d+)k?',
            r'above\s+\$?(\d+)k?'
        ]
        
        for pattern in range_patterns:
            match = re.search(pattern, query)
            if match:
                if "under" in pattern or "below" in pattern:
                    max_price = float(match.group(1)) * (1000 if 'k' in match.group(0) else 1)
                    return [0, max_price]
                elif "over" in pattern or "above" in pattern:
                    min_price = float(match.group(1)) * (1000 if 'k' in match.group(0) else 1)
                    return [min_price, filter_config.get("max", 15000)]
                else:
                    min_price = float(match.group(1)) * (1000 if 'k' in match.group(0) else 1)
                    max_price = float(match.group(2)) * (1000 if 'k' in match.group(0) else 1)
                    return [min_price, max_price]
        
        # Look for common price categories
        common_ranges = filter_config.get("common_ranges", [])
        for range_def in common_ranges:
            if range_def["label"] in query:
                return range_def["range"]
        
        return None
    
    def _extract_weight_range(self, query: str, filter_config: Dict[str, Any]) -> Optional[List[float]]:
        """Extract weight range from query"""
        
        # Look for weight mentions: "under 8kg", "lightweight", "heavy"
        if "lightweight" in query or "light" in query:
            return [filter_config.get("min", 5), 8]  # Under 8kg is "light"
        elif "heavy" in query:
            return [12, filter_config.get("max", 25)]  # Over 12kg is "heavy"
        
        # Look for specific weights
        weight_match = re.search(r'under\s+(\d+(?:\.\d+)?)kg?', query)
        if weight_match:
            max_weight = float(weight_match.group(1))
            return [filter_config.get("min", 5), max_weight]
        
        return None
    
    def _extract_multi_select_filter(self, query: str, filter_config: Dict[str, Any]) -> Optional[List[str]]:
        """Extract multi-select filter values from query"""
        
        values = filter_config.get("values", [])
        aliases = filter_config.get("aliases", {})
        matched_values = []
        
        # Check direct matches
        for value in values:
            if value.replace("_", " ") in query or value.replace("_", "") in query:
                matched_values.append(value)
        
        # Check aliases
        for value, alias_list in aliases.items():
            for alias in alias_list:
                if alias.lower() in query:
                    matched_values.append(value)
                    break
        
        # Remove duplicates while preserving order
        unique_values = []
        seen = set()
        for value in matched_values:
            if value not in seen:
                seen.add(value)
                unique_values.append(value)
        
        return unique_values if unique_values else None
    
    def _enhance_filters_with_user_state(
        self, 
        filters: Dict[str, Any], 
        user_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance extracted filters with user state information"""
        
        # Add budget constraints from user state
        if "price" not in filters and "budget_range" in user_state:
            filters["price"] = user_state["budget_range"]
        
        # Add preferences from conversation history
        if "preferred_categories" in user_state and "category" not in filters:
            preferred = user_state["preferred_categories"]
            if preferred:
                filters["category"] = preferred[0]  # Use most preferred
        
        # Add features based on past interests
        if "interested_features" in user_state:
            existing_features = filters.get("features", [])
            state_features = user_state["interested_features"]
            combined_features = list(set(existing_features + state_features))
            if combined_features:
                filters["features"] = combined_features
        
        return filters
    
    async def _generate_query_strategies(
        self,
        original_query: str,
        intent_analysis: Dict[str, Any],
        expanded_terms: List[str],
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate multiple query strategies using LLM"""
        
        strategies_prompt = f"""Generate 3 different search query strategies for finding products.

ORIGINAL QUERY: "{original_query}"
INTENT: {intent_analysis['primary_intent']}
EXPANDED TERMS: {expanded_terms}
FILTERS: {json.dumps(filters)}

Create 3 strategies:
1. SPECIFIC: Highly targeted query for exact matches
2. BALANCED: Good coverage while maintaining relevance  
3. BROAD: Wider search to ensure nothing is missed

For each strategy, provide:
- query: The search query string
- confidence: How confident you are this will find good results (0.0-1.0)
- reasoning: Brief explanation of the strategy

Output as JSON array of strategies."""
        
        try:
            llm = self.llm_factory.get_service("openai/gpt-4-turbo")
            response = await llm.chat_completion(
                messages=[{"role": "user", "content": strategies_prompt}],
                temperature=0.3,
                max_tokens=400
            )
            
            if response and response.get("content"):
                strategies = json.loads(response["content"])
                return strategies
                
        except Exception as e:
            logger.warning(f"⚠️ Strategy generation failed: {e}")
        
        # Fallback strategies
        return [
            {
                "query": original_query,
                "confidence": 0.7,
                "reasoning": "Original query as-is"
            },
            {
                "query": f"{original_query} {' '.join(expanded_terms[:2])}",
                "confidence": 0.6,
                "reasoning": "Original with expansions"
            },
            {
                "query": " ".join(original_query.split()[:2]),
                "confidence": 0.5,
                "reasoning": "Simplified query"
            }
        ]
    
    def _select_best_strategy(
        self,
        strategies: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Select the best query strategy based on context"""
        
        # For now, select highest confidence
        # Could be enhanced with more sophisticated selection logic
        return max(strategies, key=lambda s: s.get("confidence", 0))
    
    def _generate_follow_up_questions(
        self,
        intent_analysis: Dict[str, Any],
        filters: Dict[str, Any]
    ) -> List[str]:
        """Generate follow-up questions to refine search"""
        
        questions = []
        
        # Based on missing filters
        if "price_range" not in filters:
            questions.append("What's your budget range?")
        
        if "category" not in filters and intent_analysis["query_type"] == "broad":
            questions.append("What type of product are you looking for specifically?")
        
        # Based on intent
        if intent_analysis["primary_intent"] == "use_case":
            questions.append("Can you tell me more about how you'll use it?")
        elif intent_analysis["primary_intent"] == "comparison":
            questions.append("What features are most important to you?")
        
        return questions[:2]  # Max 2 questions
    
    async def optimize_knowledge_query(
        self,
        original_query: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize a knowledge base search query"""
        
        try:
            # Determine knowledge type
            knowledge_type = self._determine_knowledge_type(original_query)
            
            # Expand with related topics
            related_topics = self._expand_knowledge_topics(original_query, knowledge_type)
            
            # Build optimized query
            if knowledge_type == "policy":
                optimized = f"{original_query} policy procedure guide"
            elif knowledge_type == "how-to":
                optimized = f"{original_query} instructions steps guide how to"
            elif knowledge_type == "troubleshooting":
                optimized = f"{original_query} problem solution fix troubleshooting"
            else:
                optimized = f"{original_query} {' '.join(related_topics)}"
            
            return {
                "original_query": original_query,
                "optimized_query": optimized,
                "knowledge_type": knowledge_type,
                "related_topics": related_topics,
                "confidence": 0.8
            }
            
        except Exception as e:
            logger.error(f"❌ Knowledge query optimization failed: {e}")
            return {
                "original_query": original_query,
                "optimized_query": original_query,
                "confidence": 0.5
            }
    
    def _determine_knowledge_type(self, query: str) -> str:
        """Determine what type of knowledge is being sought"""
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["policy", "return", "warranty", "shipping"]):
            return "policy"
        elif any(word in query_lower for word in ["how to", "how do", "steps", "install"]):
            return "how-to"
        elif any(word in query_lower for word in ["problem", "issue", "broken", "fix"]):
            return "troubleshooting"
        elif any(word in query_lower for word in ["review", "opinion", "experience"]):
            return "reviews"
        else:
            return "general"
    
    def _expand_knowledge_topics(self, query: str, knowledge_type: str) -> List[str]:
        """Expand query with related knowledge topics"""
        
        expansions = {
            "policy": ["terms", "conditions", "procedures"],
            "how-to": ["guide", "tutorial", "instructions"],
            "troubleshooting": ["common issues", "solutions", "FAQ"],
            "reviews": ["testimonials", "ratings", "feedback"],
            "general": ["information", "details", "overview"]
        }
        
        return expansions.get(knowledge_type, [])


# Factory function for easy agent creation
def create_query_optimization_agent(brand_domain: str) -> QueryOptimizationAgent:
    """Create and return a Query Optimization Agent"""
    return QueryOptimizationAgent(brand_domain)