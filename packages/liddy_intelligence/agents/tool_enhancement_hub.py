"""
Tool Enhancement Hub for Voice-First AI Sales Agent

Coordinates multi-agent intelligence to optimize tool calls (product_search, knowledge_search)
during the "Let me find that for you..." moments, providing 3-5 seconds for sophisticated
query optimization and result quality assessment.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from liddy_intelligence.llm.simple_factory import LLMFactory

logger = logging.getLogger(__name__)


class ToolEnhancementHub:
    """
    Orchestrates intelligent tool call enhancement for voice-first AI.
    
    When the AI calls product_search or knowledge_search, this hub:
    1. Optimizes the query using reasoning models
    2. Assesses result quality
    3. Generates follow-up questions if needed
    4. Provides enhanced context back to the AI
    """
    
    def __init__(self, brand_domain: str):
        self.brand_domain = brand_domain
        self.llm_factory = LLMFactory()
        
        # Performance tracking
        self.enhancement_stats = {
            "total_calls": 0,
            "successful_enhancements": 0,
            "query_refinements": 0,
            "avg_enhancement_time_ms": 0.0
        }
        
        logger.info(f"🔧 Initialized ToolEnhancementHub for {brand_domain}")
    
    async def enhance_product_search(
        self, 
        original_query: str,
        conversation_context: Dict[str, Any],
        user_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Enhance product search query and results.
        
        Returns:
            Enhanced query, confidence, filters, and follow-up questions
        """
        
        start_time = time.time()
        self.enhancement_stats["total_calls"] += 1
        
        try:
            # Step 1: Optimize query using reasoning model
            optimized_query_data = await self._optimize_product_query(
                original_query, conversation_context, user_state
            )
            
            # Step 2: Execute enhanced search (placeholder for actual RAG)
            search_results = await self._execute_product_search(
                optimized_query_data["query"],
                optimized_query_data.get("filters", {})
            )
            
            # Step 3: Assess result quality
            quality_assessment = await self._assess_product_results(
                search_results, original_query, conversation_context
            )
            
            # Step 4: Determine if refinement needed
            if quality_assessment["needs_refinement"]:
                # Refine query and search again
                refined_query_data = await self._refine_product_query(
                    original_query, search_results, quality_assessment
                )
                
                refined_results = await self._execute_product_search(
                    refined_query_data["query"],
                    refined_query_data.get("filters", {})
                )
                
                search_results = refined_results
                self.enhancement_stats["query_refinements"] += 1
            
            # Step 5: Generate enhanced response
            enhanced_response = {
                "original_query": original_query,
                "optimized_query": optimized_query_data["query"],
                "confidence": quality_assessment["confidence"],
                "results": search_results,
                "follow_up_questions": optimized_query_data.get("follow_up_questions", []),
                "result_summary": quality_assessment.get("summary", ""),
                "filters_applied": optimized_query_data.get("filters", {}),
                "enhancement_time_ms": (time.time() - start_time) * 1000
            }
            
            self._update_stats(enhanced_response["enhancement_time_ms"], success=True)
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"❌ Product search enhancement failed: {e}")
            self._update_stats((time.time() - start_time) * 1000, success=False)
            
            # Return original query as fallback
            return {
                "original_query": original_query,
                "optimized_query": original_query,
                "confidence": 0.5,
                "results": [],
                "error": str(e)
            }
    
    async def enhance_knowledge_search(
        self,
        original_query: str,
        conversation_context: Dict[str, Any],
        user_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Enhance knowledge base search query and results.
        
        Returns:
            Enhanced query, search type, confidence, and clarifications
        """
        
        start_time = time.time()
        self.enhancement_stats["total_calls"] += 1
        
        try:
            # Step 1: Optimize query using reasoning model
            optimized_query_data = await self._optimize_knowledge_query(
                original_query, conversation_context, user_state
            )
            
            # Step 2: Execute enhanced search (placeholder for actual RAG)
            search_results = await self._execute_knowledge_search(
                optimized_query_data["query"],
                optimized_query_data.get("search_type", "general")
            )
            
            # Step 3: Assess result quality
            quality_assessment = await self._assess_knowledge_results(
                search_results, original_query, conversation_context
            )
            
            # Step 4: Generate enhanced response
            enhanced_response = {
                "original_query": original_query,
                "optimized_query": optimized_query_data["query"],
                "search_type": optimized_query_data.get("search_type", "general"),
                "confidence": quality_assessment["confidence"],
                "results": search_results,
                "clarification_needed": optimized_query_data.get("follow_up_clarification"),
                "result_summary": quality_assessment.get("summary", ""),
                "enhancement_time_ms": (time.time() - start_time) * 1000
            }
            
            self._update_stats(enhanced_response["enhancement_time_ms"], success=True)
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"❌ Knowledge search enhancement failed: {e}")
            self._update_stats((time.time() - start_time) * 1000, success=False)
            
            return {
                "original_query": original_query,
                "optimized_query": original_query,
                "confidence": 0.5,
                "results": [],
                "error": str(e)
            }
    
    async def _optimize_product_query(
        self, 
        query: str, 
        context: Dict[str, Any],
        user_state: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Use reasoning model to optimize product search query"""
        
        # Build context for optimization
        recent_messages = context.get("recent_messages", [])
        expressed_interests = context.get("expressed_interests", [])
        
        optimization_prompt = f"""You are optimizing a product search query for {self.brand_domain} products.

ORIGINAL QUERY: "{query}"

CONVERSATION CONTEXT:
Recent messages: {recent_messages[-3:] if recent_messages else 'None'}
Expressed interests: {expressed_interests}
User emotional state: {user_state.get('emotional_journey', ['unknown'])[-1] if user_state else 'unknown'}
Products discussed: {user_state.get('products_discussed', []) if user_state else []}

TASK: Create an optimized search query that will find the most relevant products.

Consider:
1. The customer's specific use case and requirements
2. Any constraints (budget, features, etc.) mentioned in conversation
3. Their experience level and technical knowledge
4. Products they've already seen or discussed

Output a JSON object with:
{{
    "query": "The optimized search query - be specific but not overly restrictive",
    "confidence": 0.0-1.0,
    "filters": {{
        "price_range": [min, max] or null,
        "category": "specific category" or null,
        "features": ["required features"] or null
    }},
    "follow_up_questions": ["1-2 questions that could help refine the search"],
    "reasoning": "Brief explanation of optimization approach"
}}"""
        
        try:
            # Use reasoning model for sophisticated query optimization
            llm = self.llm_factory.get_service("openai/o1")
            response = await llm.chat_completion(
                messages=[{"role": "user", "content": optimization_prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            if response and response.get("content"):
                result = json.loads(response["content"])
                logger.info(f"✅ Optimized product query: '{query}' -> '{result['query']}'")
                return result
            
        except Exception as e:
            logger.warning(f"⚠️ Query optimization failed: {e}")
        
        # Fallback to simple optimization
        return {
            "query": query,
            "confidence": 0.7,
            "filters": {},
            "follow_up_questions": ["What's most important to you in this product?"]
        }
    
    async def _optimize_knowledge_query(
        self,
        query: str,
        context: Dict[str, Any],
        user_state: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Use reasoning model to optimize knowledge search query"""
        
        optimization_prompt = f"""You are optimizing a knowledge base search for {self.brand_domain}.

ORIGINAL QUERY: "{query}"

TASK: Create an optimized search query for the knowledge base.

Determine:
1. What type of information they're seeking
2. The appropriate level of detail
3. Related topics that might be helpful

Output a JSON object with:
{{
    "query": "The optimized search query",
    "confidence": 0.0-1.0,
    "search_type": "policy" | "how-to" | "specs" | "reviews" | "general",
    "follow_up_clarification": "A clarifying question if the request is ambiguous" or null,
    "reasoning": "Brief explanation"
}}"""
        
        try:
            llm = self.llm_factory.get_service("openai/o1")
            response = await llm.chat_completion(
                messages=[{"role": "user", "content": optimization_prompt}],
                temperature=0.1,
                max_tokens=400
            )
            
            if response and response.get("content"):
                result = json.loads(response["content"])
                logger.info(f"✅ Optimized knowledge query: '{query}' -> '{result['query']}'")
                return result
                
        except Exception as e:
            logger.warning(f"⚠️ Knowledge query optimization failed: {e}")
        
        return {
            "query": query,
            "confidence": 0.7,
            "search_type": "general"
        }
    
    async def _execute_product_search(self, query: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute product search (placeholder - would connect to actual RAG)"""
        
        # This is a placeholder - in production, this would call your actual RAG system
        logger.info(f"🔍 Executing product search: '{query}' with filters: {filters}")
        
        # Simulate search results
        mock_results = [
            {
                "id": "product_1",
                "name": "Sample Product 1",
                "description": "A great product that matches your needs",
                "price": 999.99,
                "score": 0.95
            },
            {
                "id": "product_2", 
                "name": "Sample Product 2",
                "description": "Another option to consider",
                "price": 799.99,
                "score": 0.82
            }
        ]
        
        return mock_results
    
    async def _execute_knowledge_search(self, query: str, search_type: str) -> List[Dict[str, Any]]:
        """Execute knowledge search (placeholder - would connect to actual RAG)"""
        
        logger.info(f"🔍 Executing knowledge search: '{query}' (type: {search_type})")
        
        # Simulate search results
        mock_results = [
            {
                "id": "kb_1",
                "title": "Relevant Knowledge Article",
                "content": "Information about the topic...",
                "type": search_type,
                "score": 0.90
            }
        ]
        
        return mock_results
    
    async def _assess_product_results(
        self,
        results: List[Dict[str, Any]],
        original_query: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess quality of product search results"""
        
        if not results:
            return {
                "confidence": 0.0,
                "needs_refinement": True,
                "summary": "No products found matching the criteria"
            }
        
        # Calculate quality metrics
        avg_score = sum(r.get("score", 0) for r in results) / len(results)
        high_relevance_count = sum(1 for r in results if r.get("score", 0) > 0.8)
        
        # Determine if refinement needed
        needs_refinement = avg_score < 0.7 or high_relevance_count == 0
        
        # Generate result summary
        if avg_score > 0.85:
            summary = f"Found {len(results)} excellent matches"
        elif avg_score > 0.7:
            summary = f"Found {len(results)} good options"
        else:
            summary = "Results may need refinement"
        
        return {
            "confidence": avg_score,
            "needs_refinement": needs_refinement,
            "summary": summary,
            "avg_relevance_score": avg_score,
            "high_relevance_count": high_relevance_count
        }
    
    async def _assess_knowledge_results(
        self,
        results: List[Dict[str, Any]],
        original_query: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess quality of knowledge search results"""
        
        if not results:
            return {
                "confidence": 0.0,
                "summary": "No relevant information found"
            }
        
        avg_score = sum(r.get("score", 0) for r in results) / len(results)
        
        return {
            "confidence": avg_score,
            "summary": f"Found {len(results)} relevant articles"
        }
    
    async def _refine_product_query(
        self,
        original_query: str,
        initial_results: List[Dict[str, Any]],
        quality_assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Refine query based on initial results"""
        
        refinement_prompt = f"""The initial product search didn't return optimal results.

ORIGINAL QUERY: "{original_query}"
INITIAL RESULTS QUALITY: {quality_assessment['summary']}
AVG RELEVANCE: {quality_assessment.get('avg_relevance_score', 0):.2f}

Create a refined search query that will find better matches.

Output JSON with:
{{
    "query": "Refined search query",
    "filters": {{}} // Updated filters if needed
}}"""
        
        try:
            llm = self.llm_factory.get_service("openai/gpt-4-turbo")
            response = await llm.chat_completion(
                messages=[{"role": "user", "content": refinement_prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            if response and response.get("content"):
                return json.loads(response["content"])
                
        except Exception as e:
            logger.warning(f"⚠️ Query refinement failed: {e}")
        
        # Fallback - broaden the search
        return {
            "query": original_query.split()[0],  # Use first word only
            "filters": {}
        }
    
    def _update_stats(self, processing_time_ms: float, success: bool):
        """Update enhancement statistics"""
        
        if success:
            self.enhancement_stats["successful_enhancements"] += 1
        
        # Update running average
        total_calls = self.enhancement_stats["total_calls"]
        current_avg = self.enhancement_stats["avg_enhancement_time_ms"]
        new_avg = ((current_avg * (total_calls - 1)) + processing_time_ms) / total_calls
        self.enhancement_stats["avg_enhancement_time_ms"] = new_avg
    
    def get_enhancement_stats(self) -> Dict[str, Any]:
        """Get tool enhancement statistics"""
        
        total = self.enhancement_stats["total_calls"]
        successful = self.enhancement_stats["successful_enhancements"]
        
        return {
            **self.enhancement_stats,
            "success_rate": successful / max(1, total),
            "refinement_rate": self.enhancement_stats["query_refinements"] / max(1, total)
        }


# Example usage functions
async def enhance_product_search_example(brand_domain: str):
    """Example of enhancing a product search"""
    
    hub = ToolEnhancementHub(brand_domain)
    
    # Simulate a product search from voice AI
    result = await hub.enhance_product_search(
        original_query="I need a bike for racing",
        conversation_context={
            "recent_messages": [
                "I want to get into competitive cycling",
                "My budget is around $5000"
            ],
            "expressed_interests": ["racing", "performance", "speed"]
        },
        user_state={
            "emotional_journey": ["excited", "researching"],
            "products_discussed": []
        }
    )
    
    return result


async def enhance_knowledge_search_example(brand_domain: str):
    """Example of enhancing a knowledge search"""
    
    hub = ToolEnhancementHub(brand_domain)
    
    # Simulate a knowledge search from voice AI
    result = await hub.enhance_knowledge_search(
        original_query="what's your return policy",
        conversation_context={
            "recent_messages": ["I'm worried about sizing"],
            "expressed_interests": ["fit", "returns"]
        }
    )
    
    return result